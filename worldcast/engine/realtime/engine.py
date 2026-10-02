"""The resident serving engine of one WorldCast client.

:class:`Engine` loads the models once and keeps them on the GPU. :meth:`Engine.start` opens a session (one player of
one round); each :meth:`Engine.step` call generates the next block from the controls it is given and yields that
block's frames as they are decoded. With the default :class:`~worldcast.engine.realtime.config.RealtimeConfig` a session
computes exactly what :class:`worldcast.engine.inference.client.Client` computes (same latents, same pool records), for the recorded
player states and for the closed loop (``own_state``); the latency options change only where and when the work
runs, except the ones the config marks as not exact (other kernels, sub-block modes).

Per block the work splits at the controls::

    prepare   scene state (own write, step record, lock-step wait, admit, follow, retrieve), window, conditions
    prefill   the window's context written to the KV cache            <- needs no controls of the new block
    ladder    the target block on the 4-step ladder                   <- needs the block's 16 rows of controls
    commit    store, publish (scene block, step record, player state)
    decode    latent frame by latent frame, encode, deliver           (a second thread and CUDA stream with
                                                                       ``decode_overlap``)

With ``early_prefill`` the next block's prepare and prefill run right after this block's ladder, while the player
is still choosing the next controls, so a step costs only the ladder before its first frame can be decoded.

Threads: one generation thread (all model and scene-state work, in block order) and, with ``decode_overlap``, one
decode thread. The caller's thread only hands over controls and receives frames. One engine per process (the
ladder draws from the device's global RNG, as the paper client did).
"""

import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from worldcast.config.inference import InferenceConfig
from worldcast.data.actions import OPENCS2_WEAPONS, read_control_ticks
from worldcast.data.index import RoundIndexRow, read_round_index
from worldcast.data.item import ClientWindow, DataPaths, load_client_window
from worldcast.data.latents import window_key
from worldcast.data.media import MediaIndex
from worldcast.data.memory_slot import MemorySlotFrames, published_block_candidate
from worldcast.data.player_frames import WindowSpec, covered_frames
from worldcast.data.ticks import read_player_ticks
from worldcast.engine.inference.client import FixedPromptEncoder, client_latents, field_config
from worldcast.engine.inference.pool import LocalDirPool, PeerBlocks, PoolBackend
from worldcast.modeling.build import (
    generator_config_from_inference,
    generator_config_from_snapshot,
    load_generator,
)
from worldcast.modeling.depth_head import PictureDepth
from worldcast.modeling.state_model import StateTables, load_state_model
from worldcast.modeling.wan22.model import (
    CausalGeneratorAdapter,
    GeneratorConfig,
    KVCache,
    WorldCastGenerator,
)
from worldcast.player_state.closed_loop import ClosedLoop, StateExchange, StateReader
from worldcast.player_state.extrapolate import PhysicsPrior, prior_channels
from worldcast.player_state.field import build_field
from worldcast.player_state.predicted_visibility import block_depth_rows
from worldcast.player_state.projection import (
    c2w_from_state_rows,
    half_angle_tangents,
    scoped_query_tans,
)
from worldcast.player_state.tables import (
    CONTINUOUS_ROWS_KEY,
    PlayerStates,
    continuous_row_state,
    peer_conditions,
)
from worldcast.player_state.visibility import GTLabelVisibility
from worldcast.sampling import window as W
from worldcast.sampling.rollouts import Sampler
from worldcast.sampling.schedulers import entry_noise, ladder_denoise, paired_cache_noise
from worldcast.scene_state.state import SceneState
from worldcast.utils.precision import enable_tf32
from worldcast.utils.seed import set_seed

from .config import RealtimeConfig
from .controls import BlockControls
from .fast import FastGenerator
from .frames import H264Encoder, JpegEncoder, encode_frames, make_decoder
from .transport import MessagePool, MessageStateExchange, PlayerStateMessage

__all__ = [
    "RoundSpec",
    "Frame",
    "BlockRecord",
    "EngineModels",
    "OwnStateSource",
    "GroundTruthTracks",
    "Engine",
]

BLOCK = W.BLOCK
ROWS = W.ROWS_PER_LATENT
CONTROL_KEYS = ("button_condition", "camera_condition", "weapon_condition")


# ----------------------------------------------------------------------------------------------- records
@dataclass(frozen=True)
class RoundSpec:
    """A round to serve: ``(match_id, map_name, round)``, the window start (source frame, 32 fps) and the player
    slots that run clients in this session (the lock-step group; empty: this client alone)."""

    match_id: int
    map_name: str
    round: int
    start_frame: int = 0
    clients: tuple[int, ...] = ()

    @classmethod
    def from_index_row(cls, path: str, row: int) -> "RoundSpec":
        """The round and client group of row ``row`` of a round index (the paper's Table-3 index)."""
        r = read_round_index(path)[int(row)]
        return cls(
            match_id=r.match_id,
            map_name=r.map_name,
            round=r.round,
            start_frame=r.start_frame,
            clients=tuple(r.group_slots),
        )

    def index_row(self, slot: int, media_index: MediaIndex) -> RoundIndexRow:
        """The client's round-index row for player ``slot``."""
        slots = media_index.slots((int(self.match_id), str(self.map_name), int(self.round)))
        if int(slot) not in slots:
            raise KeyError(f"player slot {slot} has no recording in round {self}")
        group = (
            tuple(sorted({int(s) for s in self.clients} | {int(slot)}))
            if self.clients
            else (int(slot),)
        )
        missing = [s for s in group if s not in slots]
        if missing:
            raise KeyError(f"client slots {missing} have no recording in round {self}")
        return RoundIndexRow(
            media_id=slots[int(slot)].media_id,
            start_frame=int(self.start_frame),
            match_id=int(self.match_id),
            round=int(self.round),
            map_name=str(self.map_name),
            latent_key=window_key(int(self.start_frame)),
            player_slot=int(slot),
            group_media=tuple(slots[s].media_id for s in group),
            group_slots=group,
        )


@dataclass
class Frame:
    """One video frame of the stream: ``index`` (0 = latent 0), ``block`` (first latent of its block, 0 for the
    first frame), ``image`` (uint8 ``[H, W, 3]`` or encoded bytes) and ``t_ready`` (``time.monotonic()`` when it
    was ready for the network)."""

    index: int
    block: int
    image: Any
    t_ready: float


@dataclass
class BlockRecord:
    """What one block did and when (``time.monotonic()`` seconds; ``*_ms`` durations, GPU ones from CUDA events)."""

    s: int
    kind: str
    t_controls: float = 0.0  # step() was called
    t_sample: float = 0.0  # the block's controls were taken (the input lookahead ends here)
    t_ladder: float = 0.0  # the ladder started
    t_x0: float = 0.0  # the block's latents were on the host
    t_first_frame: float = 0.0  # its first frame was ready for the network
    t_last_frame: float = 0.0
    host_ms: dict[str, float] = field(default_factory=dict)
    gpu_ms: dict[str, float] = field(default_factory=dict)
    read: dict[str, Any] | None = None


# ----------------------------------------------------------------------------------------------- models
@dataclass
class EngineModels:
    """The weights an engine keeps resident (load once, share between sessions)."""

    generator: WorldCastGenerator
    prompt_embeds: torch.Tensor
    depth: PictureDepth
    wan_vae: Any = None
    tiny: Any = None

    @classmethod
    def load(
        cls,
        cfg: InferenceConfig,
        realtime: RealtimeConfig,
        *,
        backbone: GeneratorConfig | None = None,
        attention=None,
        vae_path: str | None = None,
    ) -> "EngineModels":
        """Generator (bf16), prompt embedding, depth head and the configured decoder, on ``cfg.run.device``."""
        p, device = cfg.paths, torch.device(cfg.run.device)
        if attention is None and device.type != "cuda":  # flash-attention is CUDA-only
            from worldcast.modeling.wan22.attention import sdpa_attention

            attention = sdpa_attention
        if backbone is None:
            snapshot = Path(p.wan22_root or "") / "config.json"
            backbone = (
                generator_config_from_snapshot(snapshot)
                if p.wan22_root and snapshot.is_file()
                else GeneratorConfig()
            )
        generator = load_generator(
            p.checkpoint,
            generator_config_from_inference(cfg, backbone),
            device=device,
            attention=attention,
        )
        dtype = getattr(torch, cfg.sampler.model_input_dtype)
        if dtype != torch.bfloat16:
            generator = generator.to(dtype)
        if p.prompt_embedding:
            from worldcast.modeling.wan22.text_encoder import PromptEmbedding

            embeds = PromptEmbedding.load(
                p.prompt_embedding, prompt=cfg.model.fixed_prompt, device=device
            ).embeds
        else:
            from worldcast.modeling.wan22.text_encoder import TextEncoder

            encoder = TextEncoder.from_pretrained(p.wan22_root, device=device, dtype=torch.bfloat16)
            embeds = encoder.encode_prompt(cfg.model.fixed_prompt).embeds
            del encoder
        depth = PictureDepth.load(p.depth_head, p.depth_readout, device=device)
        models = cls(generator=generator, prompt_embeds=embeds, depth=depth)
        models.load_decoder(realtime, cfg, vae_path=vae_path)
        return models

    def load_decoder(
        self, realtime: RealtimeConfig, cfg: InferenceConfig, *, vae_path: str | None = None
    ) -> None:
        device = realtime.decode_device or cfg.run.device
        if realtime.decoder == "wan" and self.wan_vae is None:
            from worldcast.modeling.wan22.vae import load_wan22_vae

            path = vae_path or str(Path(cfg.paths.wan22_root or "") / "Wan2.2_VAE.pth")
            dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
            self.wan_vae = load_wan22_vae(path, device=device, dtype=dtype)
        if realtime.decoder == "taehv" and self.tiny is None:
            from .taehv import load_taew2_2

            dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
            self.tiny = load_taew2_2(realtime.taehv_path, device=device, dtype=dtype)


# ----------------------------------------------------------------------------------------------- own state
class OwnStateSource(Protocol):
    """Where a client's own player state comes from.

    ``query_c2w(s)``: cameras ``[4, 4, 4]`` of block ``s`` for retrieval (before its controls are known);
    ``window_c2w()``: cameras of every latent ``[N, 4, 4]`` the window geometry reads; ``observe(s, x0)``: the
    block was generated (a state model corrects the own position here); ``player_state(s)``: the
    ``[4, 6]`` state rows published for block ``s``.
    """

    def query_c2w(self, s: int) -> torch.Tensor: ...

    def window_c2w(self) -> torch.Tensor: ...

    def observe(self, s: int, x0: torch.Tensor) -> None: ...

    def player_state(self, s: int) -> np.ndarray: ...


class GroundTruthTracks:
    """The recorded track of the client's own player (Table 3, ``pose_source: oracle``)."""

    def __init__(self, states: np.ndarray, n: int, *, eye_height: float) -> None:
        self.rows = np.asarray(states)[BLOCK * np.arange(n)]
        self.c2w = c2w_from_state_rows(self.rows, eye_height=eye_height)

    def query_c2w(self, s: int) -> torch.Tensor:
        return self.c2w[s : s + BLOCK]

    def window_c2w(self) -> torch.Tensor:
        return self.c2w

    def observe(self, s: int, x0: torch.Tensor) -> None:
        return None

    def player_state(self, s: int) -> np.ndarray:
        return np.asarray(self.rows[s : s + BLOCK], dtype=np.float32)


#: Own-state sources: ``gt`` = the recorded track (``player_state.source: recorded``); ``state_model`` = the closed
#: loop of ``worldcast.player_state.closed_loop`` (``player_state.source: predicted``: the state model reads the client's
#: own frames, positions are exchanged with the peers once per block, visibility is predicted).
OWN_STATES = ("gt", "state_model")


def own_state_source(name, window: ClientWindow, n: int, cfg: InferenceConfig) -> OwnStateSource:
    """The recorded track for ``gt`` and ``state_model`` (the closed loop starts from it and replaces the cameras
    block by block), or a caller's :class:`OwnStateSource`."""
    if not isinstance(name, str):
        return name
    if name in OWN_STATES:
        return GroundTruthTracks(
            window.player_frames[window.observer.player_slot].states,
            n,
            eye_height=cfg.data.eye_height,
        )
    raise ValueError(f"own_state must be one of {OWN_STATES} or an OwnStateSource, got {name!r}")


# ----------------------------------------------------------------------------------------------- the engine
class _Timer:
    """CUDA-event timer of named GPU spans on the current stream (no sync until :meth:`collect`)."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.spans: list[tuple[str, Any, Any]] = []

    def span(self, name: str):
        timer = self

        class _Span:
            def __enter__(self):
                if timer.enabled:
                    self.start = torch.cuda.Event(enable_timing=True)
                    self.start.record()
                return self

            def __exit__(self, *exc):
                if timer.enabled:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    timer.spans.append((name, self.start, end))
                return False

        return _Span()

    def collect(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, start, end in self.spans:
            end.synchronize()
            out[name] = out.get(name, 0.0) + float(start.elapsed_time(end))
        self.spans = []
        return out


class _TimedCalls:
    """Wraps the sampler's generator: one CUDA-event span per call, named ``<label><k>`` (k = 1, 2, ... since the
    label was set: ``prefill1..5``, ``step1..4``)."""

    def __init__(self, inner, timer: _Timer) -> None:
        self.inner, self.timer = inner, timer
        self.input_dtype = inner.input_dtype
        self.set_label("call")

    def set_label(self, label: str) -> None:
        self.label, self.count = label, 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        with self.timer.span(f"{self.label}{self.count}"):
            return self.inner(*args, **kwargs)


class Engine:
    """A resident WorldCast client (see the module docstring).

    Args:
        config: the release inference config (paths, run); ``run.index_row`` is not used.
        realtime: serving options (default: the paper client).
        models: loaded weights to share (default: loaded here from ``config``).
        own_state: ``"gt"`` (recorded track), ``"state_model"`` (closed loop, needs ``paths.state_model`` and its
            tables) or an :class:`OwnStateSource`; default from ``player_state.source`` (recorded -> gt).
        on_message: called with every message this engine emits (``worldcast.engine.realtime.transport``).
        on_frame: called with every :class:`Frame` as soon as it is ready (from the decode thread).
        pool_dir: use the paper's shared-directory pool instead of messages (multi-process lock-step).
        profile: record CUDA-event timings per block (:attr:`records`).
    """

    def __init__(
        self,
        config: InferenceConfig,
        realtime: RealtimeConfig = RealtimeConfig(),
        *,
        models: EngineModels | None = None,
        own_state: Any = None,
        on_message: Callable[[Any], None] | None = None,
        on_frame: Callable[[Frame], None] | None = None,
        pool_dir: str | None = None,
        profile: bool = False,
        backbone: GeneratorConfig | None = None,
        attention=None,
    ) -> None:
        self.cfg, self.rt = config, realtime
        self.device = torch.device(config.run.device)
        self.dtype = getattr(torch, config.sampler.model_input_dtype)
        if config.sampler.tf32:
            enable_tf32()
        self.models = models or EngineModels.load(
            config, realtime, backbone=backbone, attention=attention
        )
        if models is not None:
            self.models.load_decoder(realtime, config)
        if own_state is None:
            own_state = "state_model" if config.player_state.source == "predicted" else "gt"
        self.own_state_name = own_state
        self._state_model = None
        self.on_message, self.on_frame = on_message, on_frame
        self.pool_dir = pool_dir
        self.profile = bool(profile) and self.device.type == "cuda"
        self.timer = _Timer(self.profile)
        self.records: list[BlockRecord] = []
        self._generator = self._make_generator()
        self._session = None

    # -- set-up -------------------------------------------------------------------------------------------------
    def _make_generator(self):
        fcfg = field_config(self.cfg)
        timer = self.timer

        def field_builder(c, weapon_weight, frame_offset, num_frames):
            with timer.span("field"):
                return build_field(
                    c["peer_states"],
                    c["peer_actions"],
                    c["peer_observer_slot"],
                    c["peer_team_ids"],
                    c["peer_alive"],
                    c["peer_visible"],
                    c["peer_weapons"],
                    weapon_weight,
                    frame_offset=frame_offset,
                    video_frames=num_frames,
                    config=fcfg,
                )

        g = self.models.generator
        if self.rt.generator == "fast":
            return FastGenerator(
                g,
                field_builder,
                input_dtype=self.dtype,
                attention=None if self.device.type != "cuda" else self.rt.attention,
                cuda_graphs=self.rt.cuda_graphs,
                compile=self.rt.compile,
            )
        return CausalGeneratorAdapter(g, field_builder, input_dtype=self.dtype)

    def _sampler(self) -> Sampler:
        gc = self.models.generator.config
        cache = KVCache.allocate(
            num_blocks=len(self.models.generator.blocks),
            num_heads=gc.num_heads,
            head_dim=gc.dim // gc.num_heads,
            capacity_latents=self.cfg.window.kv_cache_latents,
            frame_seq_length=self.cfg.model.frame_seq_length,
            batch_size=1,
            dtype=self.dtype,
            device=self.device,
        )
        generator = _TimedCalls(self._generator, self.timer) if self.profile else self._generator
        return Sampler.from_config(self.cfg, generator, cache)

    def _spec(self, latents: int) -> WindowSpec:
        d, pf = self.cfg.data, self.cfg.model.player_field
        return WindowSpec(
            latent_frames=int(latents),
            skip_frame=d.skip_frame,
            max_tick_gap_seconds=d.max_tick_gap_seconds,
            button_names=tuple(d.action_buttons),
            num_substeps=pf.action_substeps,
            camera_delta_scale=pf.camera_delta_scale,
            camera_encoding=d.camera_encoding,
        )

    def load_window(self, row: RoundIndexRow) -> ClientWindow:
        """The client window of ``row`` at the length the recording supports (as :meth:`worldcast.engine.inference.client.Client.load_window`)."""
        p = self.cfg.paths
        media_index = MediaIndex.load(p.media_index)
        observer = media_index.media(row.media_id)
        covered = covered_frames(
            read_player_ticks(observer, p.dataset_root),
            observer,
            row.start_frame,
            self._spec(self.cfg.run.latents),
        )
        paths = DataPaths(
            dataset_root=Path(p.dataset_root),
            media_index=Path(p.media_index),
            latent_cache_root=Path(p.latent_cache_root),
            visibility_label_root=Path(p.visibility_label_root),
            obs_signal_label_root=Path(p.obs_signal_label_root),
        )
        return load_client_window(
            row,
            media_index,
            paths,
            self._spec(client_latents(self.cfg, covered)),
            prompt=self.cfg.model.fixed_prompt,
        )

    def _conditions(
        self, batch: Mapping[str, Any], geometry: Mapping[str, torch.Tensor], text, sampler: Sampler
    ):
        vc = W.video_conditioning(batch, device=self.device, dtype=self.dtype)
        material = PlayerStates.from_batch(batch, device=self.device)
        visible = GTLabelVisibility(device=self.device)(batch, material)
        cond = dict(vc.actions)
        cond.update(peer_conditions(material, visible, obs_signals=batch))
        cond.update(geometry)
        return vc, {**text(text_prompts=vc.prompts), **cond}

    # -- session ------------------------------------------------------------------------------------------------
    def start(
        self,
        round_spec: RoundSpec | None = None,
        player_slot: int | None = None,
        *,
        row: RoundIndexRow | None = None,
        window: ClientWindow | None = None,
        lockstep: bool | None = None,
        only_clients: bool = False,
    ) -> dict[str, Any]:
        """Open a session for ``player_slot`` of ``round_spec`` (or an index ``row``, or a loaded ``window``).

        Seeds, writes latent 0 into the KV cache and starts decoding it; the first :meth:`step` yields frame 0
        followed by the first block's frames. Returns a summary (media id, latents, blocks, peers).

        Args:
            lockstep: this session's peer sync (default: ``RealtimeConfig.lockstep``).
            only_clients: the round's seats that run no client are dead for this session (nobody plays them, so
                no recorded player enters the player state field); default: every seat as recorded (paper).
        """
        self.stop()
        cfg = self.cfg
        if window is None:
            if row is None:
                if round_spec is None or player_slot is None:
                    raise ValueError("start needs (round_spec, player_slot), a row or a window")
                row = round_spec.index_row(int(player_slot), MediaIndex.load(cfg.paths.media_index))
            window = self.load_window(row)
        self._session = _Session(
            self,
            window,
            lockstep=self.rt.lockstep if lockstep is None else bool(lockstep),
            only_clients=only_clients,
        )
        return self._session.summary()

    def step(self, controls: Any = None, *, not_before: float | None = None) -> Iterator[Frame]:
        """Generate the next block and yield its frames as they are ready (the first call also yields frame 0).

        Args:
            controls: :class:`~worldcast.engine.realtime.controls.BlockControls`; ``None`` for the recorded controls; or a
                callable returning them (e.g. ``LiveControls.sample``), which the generation thread calls at the
                last moment, right before the block's ladder.
            not_before: ``time.monotonic()`` time before which the ladder must not start (pacing: take the input as
                late as the display allows).

        After the last block of the recording :attr:`finished` is True and a step yields nothing.
        """
        if self._session is None:
            raise RuntimeError("call start() first")
        return self._session.step(controls, not_before)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait until every submitted job has finished (with ``early_prefill``: the next block's prefill)."""
        return self._session is not None and self._session.wait_idle(timeout)

    def receive(self, message: Any) -> None:
        """Deliver a peer's message (any thread)."""
        if self._session is None:
            raise RuntimeError("call start() first")
        self._session.receive(message)

    @property
    def finished(self) -> bool:
        return self._session is None or self._session.finished

    @property
    def latents(self) -> torch.Tensor:
        """The session's output latents ``[1, N, 48, 24, 42]`` (bf16 buffer, as ``latents.npy``)."""
        return self._session.output

    def stop(self, status: str = "ok") -> None:
        """Finish the session: wait for pending work, mark the pool done."""
        if self._session is not None:
            self._session.close(status)
            self._session = None


class _Session:
    """One engine session (the state of :meth:`worldcast.engine.inference.client.Client._rollout`, step by step)."""

    def __init__(
        self, engine: Engine, window: ClientWindow, *, lockstep: bool, only_clients: bool
    ) -> None:
        self.e, self.window = engine, window
        cfg, rt, dev = engine.cfg, engine.rt, engine.device
        row, spec = window.row, window.spec
        self.row, self.me, self.n = row, row.media_id, spec.latent_frames
        self.my_slot = window.observer.player_slot
        self.stride = spec.skip_frame * BLOCK
        self.recent = cfg.window.recent
        self.n_plain = cfg.window.plain_prefix_latents
        latent_shape = (cfg.model.latent_channels, cfg.model.latent_height, cfg.model.latent_width)
        self.timer = engine.timer
        self._closed = False

        # recorded inputs: the round batch, the observer's fields of view, continuous columns, own track
        self.own = own_state_source(engine.own_state_name, window, self.n, cfg)
        own_material = window.player_frames[self.my_slot]
        tan_h, tan_v = half_angle_tangents(cfg.data.hfov_degrees)
        own_obs = window.item.obs.as_dict()
        own_tans = torch.as_tensor(
            scoped_query_tans(
                own_obs,
                own_material.weapon_ids,
                range(self.n),
                tan_h=tan_h,
                tan_v=tan_v,
                weapon_names=OPENCS2_WEAPONS,
            )[0],
            dtype=torch.float32,
        )
        own_c2w = self.own.window_c2w()
        self.round_batch = {
            k: (v.unsqueeze(0) if torch.is_tensor(v) else [v])
            for k, v in window.item.batch_dict().items()
        }
        self.round_batch.update(wp_own_c2w=own_c2w[None].float(), wp_own_tans=own_tans[None])
        if only_clients:
            self._drop_unplayed(set(row.group_slots) | {self.my_slot})

        # pool, closed loop, scene state, peer blocks, slot material
        if engine.pool_dir:
            self.pool: PoolBackend = LocalDirPool(
                engine.pool_dir,
                client=self.me,
                stride=self.stride,
                poll_s=rt.poll_s,
                on_timeout="fatal" if cfg.pool.fail_on_timeout else "degrade",
                cell=self.me,
            )
        else:
            self.pool = MessagePool(
                client=self.me,
                stride=self.stride,
                outbox=self._emit,
                poll_s=rt.poll_s,
                on_timeout="fatal" if cfg.pool.fail_on_timeout else "degrade",
            )
        self.recorded = self.round_batch["player_states"]
        self.closed: ClosedLoop | None = (
            self._closed_loop() if engine.own_state_name == "state_model" else None
        )
        self.continuous = self._continuous()
        self.scene = SceneState(
            client=self.me,
            tans=[[tan_h, tan_v]],
            depth_fn=engine.models.depth.depth_grid,
            peer_latents=self.pool.peer_latents,
            bound=cfg.memory.bound,
        )
        self.blocks = PeerBlocks(
            ego_media=self.me,
            ego_slot=self.my_slot,
            sources={m.media_id: s for s, m in window.round_slots.items() if s != self.my_slot},
            candidate_at=self._candidate_at,
        )
        self.slot_material = MemorySlotFrames(
            own_media=self.me,
            own_frames=own_material,
            own_obs=window.item.obs,
            round_slots=window.round_slots,
            tick_tables=window.tick_tables,
            spec=spec,
            obs_signal_label_root=cfg.paths.obs_signal_label_root,
            hfov_degrees=cfg.data.hfov_degrees,
        )
        if lockstep and rt.sub_block:
            raise ValueError(
                "the sub-block modes publish whole blocks late: run them without lock-step"
            )
        self.peers = row.lockstep_peers() if lockstep else ()

        # sampler, text, entry noise, sink, stores
        self.sampler = engine._sampler()
        self.text = FixedPromptEncoder(cfg.model.fixed_prompt, engine.models.prompt_embeds)
        self.noise = entry_noise(
            cfg.run.seed, cfg.run.latents, self.n, latent_shape, device=dev, dtype=engine.dtype
        )
        sink = window.first_latent[None].to(dev, engine.dtype)
        self.store = torch.zeros((self.n,) + latent_shape, dtype=torch.float32)
        self.store[0] = sink[0, 0].float().cpu()
        self.output = torch.zeros((1, self.n) + latent_shape, dtype=engine.dtype, device=dev)
        self.output[:, :1] = sink

        # frames: decoder, encoder, delivery queue, decode thread
        self.frames: queue.Queue[Any] = queue.Queue()
        self.decoder = (
            None
            if rt.decoder == "none"
            else make_decoder(rt.decoder, wan_vae=engine.models.wan_vae, tiny=engine.models.tiny)
        )
        self.encoder = self._make_encoder()
        self.next_frame = 0
        # the current CUDA device is per thread: a worker that did not set it would launch on cuda:0
        bind = dict(initializer=_bind_device, initargs=(_device_index(dev),))
        self.gen = ThreadPoolExecutor(max_workers=1, thread_name_prefix="worldcast-gen", **bind)
        self.dec = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="worldcast-dec", **bind)
            if rt.decode_overlap
            else None
        )
        self.dec_stream = (
            torch.cuda.Stream(device=dev) if (rt.decode_overlap and dev.type == "cuda") else None
        )
        self.pending: list[Future] = []
        self._lock = threading.Lock()
        self.prepared: dict[int, dict[str, Any]] = {}
        self.next_s = 1
        self.commit_n, self.target_n = int(rt.commit_latents), int(rt.target_latents)
        self.published_upto = 1  # first own 4-latent block not published yet
        self.block_cameras: dict[int, torch.Tensor] = {}

        # the plain prefix starts: seed, sink write; latent 0 is the first frame
        self.prefix_merged: dict[str, Any] | None = None
        self._submit(self._open)

    def _drop_unplayed(self, clients) -> None:
        """Seats outside ``clients`` are dead and unseen for the whole session."""
        b = self.round_batch
        states = b["player_states"].clone()
        visible = b["observer_visibility"].clone()
        for slot in range(int(states.shape[1])):
            if slot not in clients:
                states[:, slot, :, 5] = 0.0
                visible[:, slot] = 0
        b["player_states"], b["observer_visibility"] = states, visible

    # -- closed loop (player_state.source = predicted), as worldcast.engine.inference.client.Client runs it -------------------------
    def _closed_loop(self) -> ClosedLoop:
        """The client's predicted states; rewrites the round batch's player-state table."""
        e, cfg, window = self.e, self.e.cfg, self.window
        p = cfg.paths
        p.require("state_model", "state_model_cells", "state_model_map_norm", "physics_prior")
        if e._state_model is None:
            e._state_model = load_state_model(
                p.state_model,
                StateTables.load(p.state_model_cells, p.state_model_map_norm),
                device=e.device,
            )
        observer = window.observer
        reader = StateReader(
            e._state_model,
            read_control_ticks(Path(p.dataset_root) / observer.ticks_path),
            media_id=self.me,
            fps=observer.fps,
            start_frame=self.row.start_frame,
            device=e.device,
        )
        exchange = (
            StateExchange(e.pool_dir, self.me, poll_s=e.rt.poll_s, fatal=cfg.pool.fail_on_timeout)
            if e.pool_dir
            else MessageStateExchange(self.pool, fatal=cfg.pool.fail_on_timeout)
        )
        return ClosedLoop.build(
            self.round_batch,
            reader=reader,
            me=self.me,
            my_slot=self.my_slot,
            round_media={s: m.media_id for s, m in window.round_slots.items()},
            clients=self.row.lockstep_peers(),
            n_latents=self.n,
            start_frame=self.row.start_frame,
            motion_kwargs=dict(
                camera_delta_scale=cfg.model.player_field.camera_delta_scale,
                channels=prior_channels(cfg.data.action_buttons),
                prior=PhysicsPrior.load(p.physics_prior),
            ),
            eye_height=cfg.data.eye_height,
            depth_fn=e.models.depth.depth_grid,
            pose_radius=cfg.player_state.pose_radius_u,
            exchange=exchange,
        )

    def _continuous(self) -> dict[str, torch.Tensor]:
        return {
            CONTINUOUS_ROWS_KEY: continuous_row_state(
                self.round_batch,
                camera_delta_scale=self.e.cfg.model.player_field.camera_delta_scale,
            )
        }

    def _candidate_at(self, slot, media_id, ws, f0):
        w, cfg = self.window, self.e.cfg
        block, reason = published_block_candidate(
            slot=slot,
            media=w.round_slots[slot],
            table=w.tick_tables[slot],
            window_start=ws,
            f0=f0,
            spec=w.spec,
            eye_height=cfg.data.eye_height,
        )
        if self.closed is not None:  # keyed where its publisher drew it
            block = self.closed.rekey(
                block, self.recorded, start_frame=self.row.start_frame, stride=self.stride
            )
        return block, reason

    def _prefix_conditions(self) -> dict[str, Any]:
        """The whole round's conditions for the plain prefix (own cameras as predicted so far in the closed loop)."""
        batch = dict(self.round_batch, latents=self.store[None].clone())
        if self.closed is not None:
            batch["wp_own_c2w"] = self.closed.own_cameras.as_predicted()[None].float()
        return self.e._conditions(
            batch, W.ordinary_window_conditions(batch, self.e.device), self.text, self.sampler
        )[1]

    def _denoise_retested(
        self, noisy: torch.Tensor, conditions, *, start: int, retest
    ) -> torch.Tensor:
        """The ladder whose rungs after the first run under ``retest(x0 of rung 1)`` (predicted visibility re-tested
        on the block's own first estimate, Sec. 3.2)."""
        sampler = self.sampler
        current = {"conditions": conditions, "retested": False}

        def call(x, t):
            flow, x0 = sampler.call(
                x, t, current["conditions"], start=start, num_frames=int(noisy.shape[1])
            )
            if not current["retested"]:
                current.update(conditions=retest(x0), retested=True)
            return flow, x0

        return ladder_denoise(call, noisy, sampler.ladder, sampler.scheduler, rng=sampler.rng)

    @property
    def finished(self) -> bool:
        """Every block of the recording has been requested."""
        return self.next_s + BLOCK > self.n

    def span(self, s: int) -> int:
        """Latents committed by the generation at ``s``: a whole block on the paper path; ``commit_latents`` in the
        sub-block modes, and the rest of the target when no later generation fits (the window needs 4 target
        latents)."""
        if s <= self.n_plain or self.commit_n == BLOCK:
            return BLOCK
        if s + self.commit_n + BLOCK > self.n:
            return min(self.target_n, self.n - s)
        return self.commit_n

    def summary(self) -> dict[str, Any]:
        return dict(
            media_id=self.me,
            slot=self.my_slot,
            latents=self.n,
            blocks=(self.n - 1) // BLOCK,
            peers=list(self.row.lockstep_peers()),
            lockstep=bool(self.peers),
            decoder=self.e.rt.decoder,
        )

    # -- threads ------------------------------------------------------------------------------------------------
    def _submit(self, fn, *args, executor=None) -> Future:
        fut = (executor or self.gen).submit(fn, *args)
        with self._lock:
            self.pending.append(fut)
        return fut

    def _check(self) -> None:
        with self._lock:
            done = [f for f in self.pending if f.done()]
            self.pending = [f for f in self.pending if not f.done()]
        for fut in done:
            fut.result()

    def _emit(self, message: Any) -> None:
        if self.e.on_message is not None:
            self.e.on_message(message)

    def receive(self, message: Any) -> None:
        if not isinstance(self.pool, MessagePool):
            raise RuntimeError("this session exchanges blocks through a pool directory")
        self.pool.receive(message)

    # -- generation thread --------------------------------------------------------------------------------------
    def _open(self) -> None:
        cfg = self.e.cfg
        with torch.no_grad():
            set_seed(cfg.run.seed)
            self.prefix_merged = self._prefix_conditions()
            self.sampler.cache.reset()
            self._label("sink")
            merged = self.prefix_merged
            if self.closed is not None:
                merged = _with_visible(
                    merged, 0, self.closed.visibility.prefix_labels(0, 1, self.output, "drawn")
                )
            self.sampler.commit(self.output[:, :1], merged, start=0)
        self._deliver(0, BlockRecord(s=0, kind="sink"))

    def _block(self, s: int, controls: Any, t_controls: float, not_before: float | None) -> None:
        rec = BlockRecord(
            s=s, kind="plain" if s <= self.n_plain else "reconstituted", t_controls=t_controls
        )
        with torch.no_grad():
            if rec.kind == "reconstituted" and s not in self.prepared:
                self._prepare(s)
            if rec.kind == "plain" and self.closed is not None:
                self._exchange_prefix(s)
            if not_before is not None and not_before > time.monotonic():
                time.sleep(not_before - time.monotonic())
            rec.t_sample = time.monotonic() if callable(controls) else t_controls
            if callable(controls):
                controls = controls()
            rec.t_ladder = time.monotonic()
            if rec.kind == "plain":
                x0 = self._plain(s, controls)
            else:
                x0 = self._reconstituted(s, controls, rec)
            span = self.span(s)
            x0 = x0[:, :span]  # sub-block modes keep the first latents of the target
            self.own.observe(s, x0)
            self.output[:, s : s + span] = x0
            ev = torch.cuda.Event() if self.e.device.type == "cuda" else None
            if ev is not None:
                ev.record()
            if rec.kind == "plain":
                self._label("commit")
                merged = self.prefix_merged
                if self.closed is not None:
                    merged = _with_visible(
                        merged,
                        s,
                        self.closed.visibility.prefix_labels(s, BLOCK, self.output, "drawn"),
                    )
                self.sampler.commit(x0, merged, start=s)
                self.store[s : s + span] = self.output[0, s : s + span].float().cpu()
            else:
                self.store[s : s + span] = x0[0].float().cpu()
            rec.t_x0 = time.monotonic()
            self._publish_completed(s + span, rec.kind, controls)
        rec.gpu_ms.update(self.timer.collect())
        self.e.records.append(rec)
        self._deliver(s, rec, ev, span)
        nxt = s + span
        if self.e.rt.early_prefill and nxt > self.n_plain and nxt + BLOCK <= self.n:
            with torch.no_grad():
                self._prepare(nxt)

    def _exchange_prefix(self, s: int) -> None:
        """Closed loop, plain prefix: publish the own position, wait for the peers' (lock-step only), read them,
        rebuild the round's conditions."""
        closed, t = self.closed, self.orig(s)
        closed.publish_own(s, t, self.output[0, :s].float().cpu())
        closed.exchange.wait(
            closed.peers if self.peers else (),
            t,
            max_wait_s=self.e.cfg.pool.wait_s,
            is_done=self.pool.is_done,
        )
        closed.read_peers(s, t)
        self.prefix_merged = self._prefix_conditions()

    def _plain(self, s: int, controls: BlockControls | None) -> torch.Tensor:
        merged = self.prefix_merged
        if controls is not None:
            merged = _with_controls(merged, controls, first_row=ROWS * s - (ROWS - 1))
        noisy = self.noise[:, s - 1 : s - 1 + BLOCK]
        self._label("step")
        with self.timer.span("ladder"):
            if self.closed is None:
                return self.sampler.denoise(noisy, merged, start=s)
            vis = self.closed.visibility
            target = _with_visible(merged, s, vis.prefix_labels(s, BLOCK, self.output, "target"))
            depth = self.e.models.depth.depth_grid
            return self._denoise_retested(
                noisy,
                target,
                start=s,
                retest=lambda x0: _with_visible(
                    target, s, vis.relabel_prefix(s, block_depth_rows(depth, x0[0]))
                ),
            )

    def _prepare(self, s: int) -> None:
        """Steps 1-8 of a reconstituted block: scene state, window, conditions, KV prefill."""
        cfg, pool, scene, blocks, closed = (
            self.e.cfg,
            self.pool,
            self.scene,
            self.blocks,
            self.closed,
        )
        host: dict[str, float] = {}
        t = self.orig(s)
        t0 = time.monotonic()
        scene.ingest_own(blocks.blocks, t_target=t, own_latents=self.store)  # 1 own write
        withdrawn, resident = scene.drain_own_step()
        if closed is not None:
            closed.publish_own(s, t, self.store)  # own position
        pool.publish_step(t_target=t, withdrawn=withdrawn, resident=resident)  # 2 step record
        t1 = time.monotonic()
        n_admitted = blocks.admit(
            pool, t_target=t, peers=self.peers, max_wait_s=cfg.pool.wait_s
        )  # 3 wait, 4 admit
        t2 = time.monotonic()
        cameras = self.own.window_c2w()
        if closed is not None:  # peers' positions
            cameras = closed.own_cameras.for_block(s)
            closed.read_peers(s, t)
            self.continuous = self._continuous()
        scene.ingest_peers(blocks.blocks, t_target=t)
        steps = pool if self.peers else _EarlierSteps(pool, t)
        scene.follow_withdrawals(
            steps, peers=blocks.peers, t_target=t, skipped=blocks.skipped
        )  # 5 follow
        query = cameras[s : s + BLOCK] if closed is not None else self.own.query_c2w(s)
        read = scene.retrieve(
            query_c2w=query,
            recent_c2w=cameras[s - self.recent : s],  # 6 retrieve
            recent_latents=self.store[s - self.recent : s],
            t_target=t,
        )
        entry = scene.entry_block(read)
        t3 = time.monotonic()
        batch = dict(
            self.round_batch,
            latents=self.store[None].clone(),
            wp_target_start=torch.tensor([s]),  # 7 window
            wp_own_c2w=cameras[None].float(),
        )
        if entry is not None:
            if entry not in blocks.eligible(t_target=t):
                raise RuntimeError(f"block {s}: retrieved block {entry} is outside the causal cut")
            b = blocks.blocks[entry]
            f0 = int(b["f0"])
            latents = (
                self.store[f0 : f0 + BLOCK]
                if b["media_id"] == self.me
                else pool.fetch_block(
                    media_id=b["media_id"], window_start=b["window_start"], f0=f0, t_target=t
                )
            )
            material = self.slot_material(b, latents)
            if closed is not None:
                material = closed.slot_rows(
                    material, b, start_frame=self.row.start_frame, stride=self.stride
                )
            batch.update({"wp_slot_" + k: v.unsqueeze(0) for k, v in material.items()})
        batch.update(self.continuous)
        if closed is not None:
            batch = closed.visibility.for_block(s, batch, self.store, self.recent)
        compacted, geom, geometry = W.prepare_window(
            batch, recent=self.recent, with_slot=entry is not None, device=self.e.device
        )
        vc, merged = self.e._conditions(compacted, geometry, self.text, self.sampler)
        ctx = geom.num_frames - BLOCK
        t4 = time.monotonic()
        self._label("prefill")
        with self.timer.span("prefill"):
            self.sampler.cache.reset()  # 8 KV prefill
            self.sampler.prefill_context(
                vc.clean_latent[:, :ctx],
                merged,
                ranges=W.context_block_ranges(geom.num_frames),
                block_noise=paired_cache_noise(
                    cfg.run.seed,
                    s,
                    len(W.context_block_ranges(geom.num_frames)),
                    self.recent // BLOCK,
                ),
            )
        host.update(
            scene_write_ms=1e3 * (t1 - t0),
            lockstep_ms=1e3 * (t2 - t1),
            retrieve_ms=1e3 * (t3 - t2),
            window_ms=1e3 * (t4 - t3),
            prefill_launch_ms=1e3 * (time.monotonic() - t4),
        )
        self.prepared[s] = dict(
            merged=merged,
            ctx=ctx,
            host=host,
            admitted=n_admitted,
            window=int(geom.num_frames),
            read=read,
            cameras=cameras,
            compacted=compacted,
            geometry=geometry,
        )

    def _reconstituted(
        self, s: int, controls: BlockControls | None, rec: BlockRecord
    ) -> torch.Tensor:
        p = self.prepared.pop(s)
        self.block_cameras[s] = p["cameras"]
        merged = (
            p["merged"]
            if controls is None
            else _with_controls(p["merged"], controls, first_row=-ROWS * BLOCK)
        )
        rec.host_ms.update(p["host"])
        read = p["read"]
        rec.read = dict(
            window=p["window"],
            admitted=p["admitted"],
            score=int(read.score),
            holes=int(read.n_hole),
            entry=None if read.entry is None else [read.entry.client, int(read.entry.t_first)],
        )
        noisy = self.noise[:, s - 1 : s - 1 + self.target_n]
        self._label("step")
        with self.timer.span("ladder"):
            if self.closed is None:
                return self.sampler.denoise(noisy, merged, start=p["ctx"])
            depth = self.e.models.depth.depth_grid

            def retest(x0):
                relabelled = self.closed.visibility.relabel_target(
                    p["compacted"], s, block_depth_rows(depth, x0[0])
                )
                out = self.e._conditions(relabelled, p["geometry"], self.text, self.sampler)[1]
                return (
                    out
                    if controls is None
                    else _with_controls(out, controls, first_row=-ROWS * BLOCK)
                )

            return self._denoise_retested(noisy, merged, start=p["ctx"], retest=retest)

    def _publish_completed(self, end: int, kind: str, controls: BlockControls | None) -> None:
        """Publish every own 4-latent block that ends before latent ``end``. In the closed loop the plain prefix is
        published after its last block, keyed at the cameras as predicted then (as the release client does).
        """
        if self.closed is not None and kind == "plain" and end <= self.n_plain:
            return
        while self.published_upto + BLOCK <= end:
            f0 = self.published_upto
            if self.closed is None:
                cameras = self.own.window_c2w()
            elif f0 <= self.n_plain:
                cameras = self.closed.own_cameras.as_predicted()
            else:
                cameras = self.block_cameras[f0]
            self._publish(f0, kind if f0 > self.n_plain else "plain", controls, cameras)
            self.published_upto += BLOCK

    def _publish(
        self, s: int, kind: str, controls: BlockControls | None, cameras: torch.Tensor
    ) -> None:
        own = dict(
            media_id=self.me,
            slot=self.my_slot,
            window_start=self.row.start_frame,
            f0=s,
            orig_first=self.orig(s),
            orig_last=self.orig(s + BLOCK - 1),
            c2w=cameras[s : s + BLOCK],
        )
        self.pool.publish_block(
            window_start=self.row.start_frame,
            f0=s,
            latents=self.store[s : s + BLOCK],
            orig_first=self.orig(s),
            orig_last=self.orig(s + BLOCK - 1),
            extra={"mode": kind},
        )
        self.blocks.add_own(own)
        if isinstance(self.pool, MessagePool):
            rows = slice(ROWS * s - (ROWS - 1), ROWS * s - (ROWS - 1) + ROWS * BLOCK)
            ctl = (
                {k: self.round_batch[k][0, rows].numpy() for k in CONTROL_KEYS}
                if controls is None
                else dict(
                    button_condition=controls.buttons,
                    camera_condition=controls.camera,
                    weapon_condition=controls.weapon,
                )
            )
            states = (
                self.own.player_state(s) if self.closed is None else _state_rows(self.closed, s)
            )
            self.pool.publish_player_state(
                PlayerStateMessage(
                    media_id=self.me,
                    slot=self.my_slot,
                    f0=s,
                    orig_first=self.orig(s),
                    states=states,
                    controls=ctl,
                )
            )

    def _label(self, name: str) -> None:
        gen = self.sampler.generator
        if isinstance(gen, _TimedCalls):
            gen.set_label(name)

    def orig(self, f: int) -> int:
        return self.row.start_frame + self.stride * int(f)

    # -- frames -------------------------------------------------------------------------------------------------
    def _make_encoder(self):
        rt = self.e.rt
        if rt.encoder == "jpeg":
            device = rt.jpeg_device if self.e.device.type == "cuda" else "cpu"
            return JpegEncoder(rt.jpeg_quality, device=device)
        if rt.encoder == "h264":
            cfg = self.e.cfg
            return H264Encoder(
                cfg.model.latent_width * cfg.model.vae_spatial_compression,
                cfg.model.latent_height * cfg.model.vae_spatial_compression,
            )
        return None

    def _deliver(self, s: int, rec: BlockRecord, ev=None, n_lat: int = 1) -> None:
        if self.dec is None:
            self._decode(s, n_lat, rec, ev)
        else:
            self._submit(self._decode, s, n_lat, rec, ev, executor=self.dec)

    def _decode(self, s: int, n_lat: int, rec: BlockRecord, ev) -> None:
        stream = self.dec_stream
        ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
        with ctx, torch.no_grad():
            if stream is not None and ev is not None:
                stream.wait_event(ev)
            for i in range(s, s + n_lat):
                if self.decoder is None:
                    images, t_ready = [self.output[0, i].float().cpu().numpy()], time.monotonic()
                else:
                    frames = self.decoder.decode(self.output[0, i : i + 1])
                    images = encode_frames(self.encoder, frames)
                    t_ready = time.monotonic()
                for img in images:
                    f = Frame(index=self.next_frame, block=s, image=img, t_ready=t_ready)
                    self.next_frame += 1
                    if not rec.t_first_frame:
                        rec.t_first_frame = t_ready
                    rec.t_last_frame = t_ready
                    if self.e.on_frame is not None:
                        self.e.on_frame(f)
                    self.frames.put(f)
        self.frames.put(("end", s))

    # -- caller thread ------------------------------------------------------------------------------------------
    def step(self, controls: Any, not_before: float | None) -> Iterator[Frame]:
        if self.finished:
            return iter(())  # the recording is over: no more frames
        s, t_controls = self.next_s, time.monotonic()
        self.next_s += self.span(s)
        self._submit(self._block, s, controls, t_controls, not_before)
        return self._drain(s)

    def wait_idle(self, timeout: float | None) -> bool:
        with self._lock:
            pending = [f for f in self.pending]
        deadline = None if timeout is None else time.monotonic() + timeout
        for fut in pending:
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                fut.result(timeout=left)
            except FutureTimeout:
                return False
        return True

    def _drain(self, s: int) -> Iterator[Frame]:
        while True:
            try:
                item = self.frames.get(timeout=0.05)
            except queue.Empty:
                self._check()
                continue
            if isinstance(item, tuple):
                if item[1] == s:
                    self._check()
                    return
                continue
            yield item

    def close(self, status: str) -> None:
        if self._closed:
            return
        self._closed = True
        failed = None
        with self._lock:
            pending = list(self.pending)
        for fut in pending:
            try:
                fut.result()
            except BaseException as exc:  # noqa: BLE001 - reported below, the pool is told first
                failed = failed or exc
        self.gen.shutdown(wait=True)
        if self.dec is not None:
            self.dec.shutdown(wait=True)
        self.pool.mark_done(
            status="failed" if failed else status, note="" if failed is None else repr(failed)
        )
        if failed is not None:
            raise failed


# ----------------------------------------------------------------------------------------------- helpers
class _EarlierSteps:
    """The peers' step records before ``t_target`` only. Without lock-step a peer's record for this very block can
    arrive between the admission and the follow, listing a block not admitted yet; its withdrawals are applied at the
    next block and the copy check (a lock-step invariant) is skipped."""

    def __init__(self, pool: PoolBackend, t_target: int) -> None:
        self.pool, self.t_target = pool, int(t_target)

    def read_steps(self, media_id: str, *, upto: int):
        return self.pool.read_steps(media_id, upto=min(int(upto), self.t_target - 1))


def _device_index(device: torch.device) -> int | None:
    """The CUDA device index ``device`` means in the calling thread (``cuda`` alone: its current device)."""
    if device.type != "cuda":
        return None
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _bind_device(index: int | None) -> None:
    if index is not None:
        torch.cuda.set_device(index)


def _with_visible(merged: Mapping[str, Any], start: int, labels: torch.Tensor) -> dict[str, Any]:
    """``merged`` with ``peer_visible`` of latents ``start ..`` replaced by predicted ``labels`` ``[n, P]``."""
    visible = merged["peer_visible"].clone()
    visible[:, start : start + labels.shape[0]] = labels.to(visible.device, visible.dtype)
    return dict(merged, peer_visible=visible)


def _state_rows(closed: ClosedLoop, s: int) -> np.ndarray:
    """``[4, 6]`` own state rows of block ``s`` as the closed loop extrapolated them when the block started."""
    return np.asarray(closed.own_cameras.block_rows(s), dtype=np.float32)


def _with_controls(
    merged: Mapping[str, Any], controls: BlockControls, *, first_row: int
) -> dict[str, Any]:
    """``merged`` with the block's 16 control rows replaced (``first_row`` may count from the end)."""
    out = dict(merged)
    n = int(controls.buttons.shape[0])
    for key, value in (
        ("button_condition", controls.buttons),
        ("camera_condition", controls.camera),
        ("weapon_condition", controls.weapon),
    ):
        t = merged[key].clone()
        start = first_row if first_row >= 0 else int(t.shape[1]) + first_row
        t[0, start : start + n] = torch.as_tensor(value, device=t.device, dtype=t.dtype)
        out[key] = t
    return out


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
