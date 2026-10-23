"""The WorldCast client (Sec. 3.1): one player's generator and state model, resident on its GPU.

:class:`Client` loads the models once. :meth:`Client.start` starts the rollout of one player of
one round; each :meth:`Client.step` generates the next block from its controls and yields the
block's frames as they are decoded. :func:`run_client` runs it on the recorded controls (the
offline client of Table 3), the web demo on live input. With the default settings, a round's
recorded length (``run.latent_frames``) and the default :class:`ServingOptions` it reproduces the
paper's latents bit for bit.

One generation thread does all model and scene-state work in block order
(:class:`~worldcast.engine.inference.rollout.Rollout`); the frames are decoded latent frame by
latent frame on the caller's thread as it takes them. One client per process, as the denoising
steps draw from the device's global RNG.

A live client with ``live.refresh`` keeps its newest block open: :meth:`Client.refresh`
denoises it again from newer controls and yields the new version's frames from the first frame the
caller has not sent yet, decoded from the latent frame that holds it on the decoder's state after
the latent frame before it in the version on screen. The next :meth:`Client.step` commits the
newest version first (the block's context, memory entry and ``live`` part), then prepares and
denoises its own block. With ``ServingOptions.refresh_devices`` the generator has replicas on more
cards (:mod:`~worldcast.engine.inference.replicas`): each version of an open block is cut in order
on the generation thread's card and denoised on a card that is free, the newest version that is
done is the one shown, and the next step commits the newest cut once every version is done.
"""

import contextlib
import json
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldcast.config.inference import InferenceConfig
from worldcast.data.latents import (
    BLOCK,
    FIRST_TARGET,
    VIDEO_FRAMES_PER_LATENT,
    last_video_frame,
)
from worldcast.data.recordings import RoundIndexRow, load_round_index_row
from worldcast.data.window import DataPaths
from worldcast.modeling.wan22.attention import (
    AttentionFn,
    attention_kernel,
    fa3_attention,
    fa3_available,
    flash_attention,
)
from worldcast.player_state import field_builder
from worldcast.sampling.sampler import Sampler
from worldcast.sampling.window import KV_CACHE_LATENTS
from worldcast.scene_state import FollowStats
from worldcast.utils.precision import enable_tf32, generator_dtype

from .controls import BlockControls
from .decode import WanFrameDecoder, pixels_to_uint8_frames
from .directory import DirectoryWorldState
from .fast import FastGenerator
from .field_rows import FieldRows
from .live_rollout import PAGE_BLOCKS, BlockCut, LiveRollout
from .loading import ClientModels, load_window
from .messages import LIVE_STATES_PER_ENTRY, MessagePart, MessageWorldState
from .replicas import GeneratorReplica, card_device, rng_state
from .rollout import BlockRead, Rollout, context_noise
from .serving import ServingOptions
from .taehv import StreamingTinyDecoder
from .world_state import DONE_FAILED, DONE_OK, WorldState

__all__ = [
    "BlockRecord",
    "Client",
    "Frame",
    "RefreshRecord",
    "StepControls",
    "Switch",
    "client_world_state",
    "run_client",
]

log = logging.getLogger(__name__)

#: A block's controls as :meth:`Client.step` takes them: the controls, a callable that returns them
#: (called on the generation thread right before the block's denoising), or ``None`` for the
#: recorded ones.
StepControls = BlockControls | Callable[[], BlockControls | None] | None
#: What a block's decode takes for a latent frame, as a refresh's switch looks ahead by it, moves
#: this share of the way to each one's.
LATENT_TIME_FOLLOW = 0.2
#: The graphs of ``torch.compile`` a card keeps per part of the DiT block (dynamo's default limit
#: for one card).
_RECOMPILES_PER_CARD = 8
#: Where a refresh's frames start (:meth:`Client.refresh`): called with the seconds its decoder
#: still needs, it returns the first video frame of the block that will not have been sent by then;
#: the frames before it stay as they were.
Switch = Callable[[float], int]


@dataclass
class Frame:
    """One video frame of a rollout.

    Attributes:
        index (int): its index in the rollout's video; frame 0 is latent frame 0.
        f0 (int): the first latent frame of its block (0 for frame 0).
        image (np.ndarray): ``[H, W, 3]`` uint8; without a decoder the latent ``[48, 24, 42]``
            float32.
        t_ready (float): the ``time.monotonic()`` when it was decoded.
        version (int): which version of its block: 0 the block's first, then one per
            :meth:`Client.refresh`.
        rows (FieldRows | None): for a block that may be refreshed, the rows the player state
            field read at the frame's latent frame in this version; ``None``:
            :meth:`Client.field_rows` has them.
        camera_pitch (float | None): likewise, the pitch of the client's camera at the frame
            (:meth:`Client.camera_pitch`).
        controls (Any): likewise, the controls its version was cut on, as the call's
            ``controls()`` returned them (a frame of a refresh can come from the step's iterator,
            :meth:`Client.refresh`).
        preview (bool): a frame of a refresh's preview: decoded from the x0 of an earlier
            denoising step than the last, which the version's own frames replace.
    """

    index: int
    f0: int
    image: np.ndarray
    t_ready: float
    version: int = 0
    rows: FieldRows | None = None
    camera_pitch: float | None = None
    controls: Any = None
    preview: bool = False


@dataclass
class RefreshRecord:
    """One refresh of a block (:meth:`Client.refresh`).

    Attributes:
        t_start (float): the ``time.monotonic()`` when the generation thread began it.
        t_controls (float): when its controls were taken.
        t_cut (float): when its cut was done and its denoising steps began.
        t_latents (float): when its latents were handed to the decoder.
        t_first_frame (float): when its first frame was decoded (0: none, every frame was sent).
        t_last_frame (float): when its last frame was decoded.
        first_frame (int): the first video frame it yielded (-1: none).
        denoise_ms (float): its four denoising steps on the generator's device, ms (CUDA events;
            on the CPU the time they took).
        card (int): the card it was denoised on, an index of ``refresh_devices`` (0: the
            client's own).
        t_preview (float): when the x0 of its preview step was handed to the decoder (0: none,
            :meth:`Client.refresh`).
        t_preview_frame (float): when the preview's first frame was decoded (0: none).
        preview_frame (int): the first video frame of the preview (-1: none).
    """

    t_start: float = 0.0
    t_controls: float = 0.0
    t_cut: float = 0.0
    t_latents: float = 0.0
    t_first_frame: float = 0.0
    t_last_frame: float = 0.0
    first_frame: int = -1
    denoise_ms: float = 0.0
    card: int = 0
    t_preview: float = 0.0
    t_preview_frame: float = 0.0
    preview_frame: int = -1


@dataclass
class BlockRecord:
    """One block of a rollout.

    Attributes:
        f0 (int): its first latent frame.
        t_controls (float): the ``time.monotonic()`` when its controls were taken, right before its
            denoising.
        t_latents (float): the ``time.monotonic()`` when its latents were handed to the decoder.
        t_first_frame (float): the ``time.monotonic()`` when its first frame was decoded.
        t_last_frame (float): the ``time.monotonic()`` when its last frame was decoded.
        read (BlockRead | None): what it read from the shared world state; ``None`` for the first
            six blocks, and for an open block until it is committed.
        t_start (float): the ``time.monotonic()`` when the generation thread began it: with a
            block before it open, the commit of that block and the prepare of this one come
            first.
        refreshes (list[RefreshRecord]): its refreshes, in order.
    """

    f0: int
    t_controls: float = 0.0
    t_latents: float = 0.0
    t_first_frame: float = 0.0
    t_last_frame: float = 0.0
    read: BlockRead | None = None
    t_start: float = 0.0
    refreshes: list[RefreshRecord] = field(default_factory=list)


class Client:
    """A resident WorldCast client (see the module docstring).

    Args:
        cfg (InferenceConfig): ``paths`` and ``run`` (``run.index_row`` is not read);
            ``model.attention`` is the kernel, ``player_state.source`` chooses the GT states, the
            closed loop or live controls (:class:`~worldcast.engine.inference.live_rollout.
            LiveRollout`).
        serving (ServingOptions): how to decode and wait (default: the paper's).
        models (ClientModels | None): loaded weights to share; default: loaded here.
        on_message (Callable | None): receives every :class:`MessagePart` the client sends the
            other clients when its rollout exchanges messages.

    Attributes:
        records (deque[BlockRecord]): the blocks of the rollout started last; a live client keeps
            its newest :data:`~worldcast.engine.inference.live_rollout.PAGE_BLOCKS` only.
    """

    def __init__(
        self,
        cfg: InferenceConfig,
        serving: ServingOptions = ServingOptions(),
        *,
        models: ClientModels | None = None,
        on_message: Callable[[MessagePart], None] | None = None,
    ) -> None:
        self.cfg, self.serving, self.on_message = cfg, serving, on_message
        # with its index: the threads of other cards resolve a bare ``cuda`` to their own
        self.device = card_device(cfg.run.device)
        cuda = self.device.type == "cuda"
        self.dtype = generator_dtype(self.device)
        enable_tf32()
        attention = self._attention()  # before the weights load: an unknown name fails here
        if models is None:
            models = ClientModels.load(cfg, serving)
        else:
            models.load_missing(cfg, serving)
        self.models = models
        self.generator = FastGenerator(
            models.generator,
            field_builder(),
            input_dtype=self.dtype,
            attention=attention,
            cuda_graphs=serving.cuda_graphs and cuda,
            compile=serving.compile,
        )
        # one KV cache for every rollout: the CUDA graphs captured in one replay into the next
        self.cache = models.generator.allocate_kv_cache(KV_CACHE_LATENTS)
        bind = {}  # the current CUDA device is per thread: the generation thread takes the client's
        if cuda:
            index = self.device.index
            bind = dict(initializer=torch.cuda.set_device, initargs=(index,))
            if index is None:
                bind["initargs"] = (torch.cuda.current_device(),)
        self.thread = ThreadPoolExecutor(1, "worldcast-gen", **bind)
        self.stream = torch.cuda.Stream(device=self.device) if cuda else None
        self.replicas = self._replicas(attention)
        # with replicas: the thread that cuts a replica's refresh in its turn, so that the
        # replica's own thread takes only denoisings, behind its block's prefill
        self.cuts = ThreadPoolExecutor(1, "worldcast-cuts", **bind) if self.replicas else None
        # with replicas the denoising on the client's card runs on a stream of its own, so that the
        # cut of another card's refresh (on the card's current stream) never waits behind it
        self.denoise_stream = (
            torch.cuda.Stream(device=self.device) if cuda and self.replicas else None
        )
        #: Called from the thread of a card when it has no job left (with replicas: a refresh may
        #: start there at once).
        self.card_freed: Callable[[], None] | None = None
        #: Per comparison of a replica's x0 with the client's card's on the same inputs (the
        #: warm-up's: the denoising of the replay's window blocks again on every replica), whether
        #: it was equal.
        self.replicas_equal: list[bool] = []
        self.records: deque[BlockRecord] = deque()
        self._rollout: Rollout | None = None
        self._running: _RunningRollout | None = None

    def _attention(self) -> AttentionFn:
        """The kernel of ``model.attention`` on the client's device: SDPA off CUDA, and
        flash-attention 2 for a FlashAttention-3 that is not installed."""
        kernel = attention_kernel(self.cfg.model.attention, self.device)
        if kernel is not fa3_attention:
            return kernel
        if fa3_available():
            log.info("attention: FlashAttention-3")
            return kernel
        log.warning("FlashAttention-3 is not installed: attention falls back to flash-attention 2")
        return flash_attention

    def _replicas(self, attention: AttentionFn) -> tuple[GeneratorReplica, ...]:
        """The generator's replicas on the cards of ``refresh_devices`` after the client's own.

        Raises:
            ValueError: the cards are not those of live refreshes, or the first is not the
                client's.
        """
        devices = self.serving.refresh_devices
        if not devices:
            return ()
        if self.cfg.player_state.source != "live" or not self.cfg.live.refresh:
            raise ValueError(
                "refresh_devices: the cards of a live client's refreshes (player_state.source"
                " live, live.refresh true)"
            )
        if card_device(devices[0]) != card_device(self.device):
            raise ValueError(
                f"refresh_devices starts with the client's card, run.device {self.cfg.run.device};"
                f" got {devices[0]}"
            )
        if self.serving.compile:
            _raise_recompile_limit(len(devices))
        return tuple(
            GeneratorReplica(
                self.cfg,
                self.serving,
                device,
                attention=attention,
                dtype=self.dtype,
                context_noise=context_noise(self.cfg),
            )
            for device in devices[1:]
        )

    def disable_replicas(self) -> None:
        """Run on the client's card alone from the next rollout on, the replicas let go (a replica
        that gave other results than the client's card: ``replicas_equal``)."""
        if not self.replicas:
            return
        log.warning(
            "the replicas of the generator on %s are let go: refreshes run on %s alone",
            ", ".join(str(replica.device) for replica in self.replicas),
            self.device,
        )
        for replica in self.replicas:
            replica.thread.shutdown(wait=True)
        self.cuts.shutdown(wait=True)
        self.replicas, self.cuts, self.denoise_stream = (), None, None
        if self._rollout is not None:
            self._rollout.replicas = ()

    def start(self, row: RoundIndexRow, *, world_state: WorldState | None = None) -> None:
        """Start a rollout: the client of a round-index ``row``.

        Seeds, writes latent frame 0 (the first frame) into the KV cache and decodes it: the first
        :meth:`step` yields frame 0, then the first block's frames.

        Args:
            row (RoundIndexRow): the client's row; its group is the lockstep group.
            world_state (WorldState | None): the shared world state, which the caller opens and
                marks done (the offline client: a :class:`DirectoryWorldState`); default: messages
                through ``on_message`` and :meth:`receive`.
        """
        self.stop()
        live = self.cfg.player_state.source == "live"
        self.records = deque(maxlen=PAGE_BLOCKS if live else None)
        self._running = _RunningRollout(self, row, world_state)
        self._rollout = self._running.rollout

    def _require_running(self) -> "_RunningRollout":
        if self._running is None:
            raise RuntimeError("no rollout is running: call start() first")
        return self._running

    def _require_started(self) -> Rollout:
        if self._rollout is None:
            raise RuntimeError("no rollout was started: call start() first")
        return self._rollout

    def step(self, controls: StepControls = None) -> Iterator[Frame]:
        """Generate the next block; yield its frames as they are decoded (the first: frame 0 too).

        The block is requested when this is called; its frames are decoded as the iterator is
        taken, on the thread that takes it. The next block may be requested before the frames of
        this one are taken to the end (the demo's worker does, with
        :attr:`ServingOptions.decoder_device <worldcast.engine.inference.serving.ServingOptions>`:
        the decode of a block overlaps the generation of the next), if the iterators are taken in
        order.

        Args:
            controls (BlockControls | Callable | None): the block's controls; ``None``: the recorded
                ones; a callable is called on the generation thread right before the block's
                denoising.

        Returns:
            Iterator[Frame]: the block's frames, to be taken to the end; nothing once
            :attr:`finished`.
        """
        return self._require_running().step(controls)

    @property
    def can_refresh(self) -> bool:
        """Whether the newest block stepped can be refreshed (:meth:`refresh`): a live rollout with
        ``live.refresh``, a block after the first six."""
        running = self._running
        return running is not None and running.can_refresh

    @property
    def refresh_cards(self) -> int:
        """The cards a refresh may run on: the client's and its replicas'."""
        return 1 + len(self.replicas)

    @property
    def refresh_card_free(self) -> bool:
        """Whether a card has no job now (none generates a block's first version or a refresh),
        so that a refresh asked for starts at once."""
        running = self._running
        return running is not None and running.free_card() is not None

    def refresh(
        self,
        controls: StepControls,
        switch: Switch,
        generated: Callable[[], None] | None = None,
        *,
        preview_step: int = 0,
        card: int | None = None,
    ) -> Iterator[Frame]:
        """Generate the newest block again from newer controls (:meth:`~worldcast.engine.
        inference.live_rollout.LiveRollout.refresh`); yield the new version's frames from the
        first frame ``switch`` names on, as they are decoded. That frame's latent frame is decoded
        on the decoder's state after the latent frame before it in the version whose frames were
        taken before, and its frames before the switch are dropped. The frames are taken on the
        thread that takes the step's, after them; the next :meth:`step` commits the newest
        version. With the decoder on the generator's device its denoising waits until the block's
        first latent frame is decoded (a refresh may be asked for before), so as not to slow the
        block's first frame down.

        With replicas (:attr:`refresh_cards` > 1) the refresh is cut once the versions asked for
        before it are, and denoised on ``card`` or a free card (:attr:`refresh_card_free`): its
        frames are taken only if it is the newest version done, and the next :meth:`step` commits
        the newest cut once every refresh is done. Its denoising does not wait for the block's
        first denoising on another card.

        Args:
            controls (StepControls): the block's controls again: called on the generation thread
                (with replicas: the card's thread) right before the denoising.
            switch (Switch): called once the new version's latents are ready, with the seconds
                one latent frame takes the decoder: the first video frame of the block not sent
                by then; one past the block's last yields nothing.
            generated (Callable | None): called once the new version's latents are ready, before
                the decoder takes them (with replicas: once its card is free too).
            preview_step (int): also hand the x0 of this denoising step (from 1; 0: none) to the
                decoder as soon as it is there, a preview of the version: its frames (marked
                ``preview``) come from the switch on as the version's do (``switch`` is called for
                each), and the version's own frames take their places.
            card (int | None): the card to denoise on, an index of ``refresh_devices``; ``None``:
                a free card, the replicas' first, or the one with the fewest jobs.

        Returns:
            Iterator[Frame]: the frames, to be taken to the end; ``version`` counts the block's
            refreshes.

        Raises:
            RuntimeError: the newest block cannot be refreshed (:attr:`can_refresh`).
        """
        return self._require_running().refresh(controls, switch, generated, preview_step, card)

    def receive(self, part: MessagePart) -> None:
        """Deliver a part of another client's message (any thread).

        Raises:
            RuntimeError: no rollout is running, or its world state is the caller's.
        """
        messages = self._require_running().messages
        if messages is None:
            raise RuntimeError("the rollout runs on the caller's world state: it takes no messages")
        messages.receive(part)

    @property
    def finished(self) -> bool:
        """Every block of the rollout has been requested; never, for a live rollout."""
        return self._running is None or self._running.finished

    @property
    def latents(self) -> torch.Tensor:
        """The latents ``[1, N, 48, 24, 42]`` of the rollout started last (``latents.npy`` holds
        them as float32); kept after :meth:`stop`. A live rollout has no last block: its round of
        ``N`` latent frames holds the first frame, the first six blocks, then the newest block."""
        return self._require_started().output

    @property
    def follow_stats(self) -> FollowStats:
        """How the scene state of the rollout started last followed the other clients'
        withdrawals; kept after :meth:`stop`."""
        return self._require_started().scene.follow_stats

    def field_rows(self, latent: int) -> FieldRows:
        """Every player's row as the player state field read it at latent frame ``latent`` of the
        rollout started last (:meth:`~worldcast.engine.inference.rollout.Rollout.field_rows`), on
        any thread; kept after :meth:`stop`.

        Args:
            latent (int): latent frame 0 or a latent frame of a block in :attr:`records`: that of
                every frame :meth:`step` has yielded (a decoder's video frame ``i`` is latent frame
                ``ceil(i / 4)``). A live client keeps the rows of its newest
                :data:`~worldcast.engine.inference.live_rollout.PAGE_BLOCKS` blocks only, each as
                it was when its block was committed.

        Raises:
            RuntimeError: no rollout was started.
            ValueError: the block of ``latent`` has not been generated: its rows may still change;
                or a live client no longer keeps it.
        """
        rollout = self._require_started()
        generated = self._latents_generated()
        if int(latent) >= generated:
            raise ValueError(
                f"latent frame {latent} has not been generated: latent frames 0-{generated - 1}"
                " have"
            )
        return rollout.field_rows(latent)

    def camera_pitch(self, frame: int) -> float:
        """The pitch of the client's own camera at video frame ``frame`` of the rollout started
        last, degrees (:meth:`~worldcast.engine.inference.rollout.Rollout.camera_pitch`), on any
        thread; kept after :meth:`stop`.

        Args:
            frame (int): a video frame :meth:`step` has yielded; a live client keeps the pitch of
                its newest :data:`~worldcast.engine.inference.live_rollout.PAGE_BLOCKS` blocks
                only.

        Raises:
            RuntimeError: no rollout was started.
            ValueError: the block of ``frame`` has not been generated: its rows may still change;
                or a live client no longer keeps it.
        """
        rollout = self._require_started()
        last = last_video_frame(self._latents_generated() - 1)
        if int(frame) > last:
            raise ValueError(
                f"video frame {frame} has not been generated: video frames 0-{last} have"
            )
        return rollout.camera_pitch(frame)

    def _latents_generated(self) -> int:
        return self.records[-1].f0 + BLOCK if self.records else 1

    def stop(self) -> None:
        """End the rollout: wait for its work; mark its world state done if the rollout opened it.

        Raises the error of a job that failed, unless :meth:`step` raised it already.
        """
        running, self._running = self._running, None
        if running is not None:
            running.close()


def client_world_state(cfg: InferenceConfig, media_id: str) -> DirectoryWorldState:
    """The session's world-state directory ``paths.world_state_dir``, as client ``media_id`` sees
    it."""
    return DirectoryWorldState(
        cfg.paths.world_state_dir, client=media_id, poll_s=cfg.world_state.poll_s
    )


def run_client(cfg: InferenceConfig) -> dict[str, Any]:
    """Run the client of ``run.index_row`` on its recorded controls, in lockstep with the round's
    other clients; writes ``latents.npy`` (the output buffer, float32) and ``client.json`` (what
    each block read, the withdrawals followed and the lockstep waits).

    A client that fails marks itself failed in the world state, so the other clients stop with an
    error.

    Args:
        cfg (InferenceConfig): ``paths`` (data, weights, ``world_state_dir``, ``out_dir``) and
            ``run.index_row``.

    Returns:
        dict[str, Any]: ``client.json``: ``media_id``, ``index_row``, ``seed``,
        ``latent_frames``, ``blocks`` (per block that read the scene state: its ``f0`` and its
        :class:`~worldcast.engine.inference.rollout.BlockRead`), ``follow``
        (:class:`~worldcast.scene_state.state.FollowStats`) and the lockstep ``wait``
        (:class:`~worldcast.engine.inference.world_state.WaitStats`).
    """
    if cfg.player_state.source == "live":
        raise ValueError(
            "player_state.source live plays a player's live controls (the live demo); the offline"
            " client runs the recorded ones: gt or predicted"
        )
    cfg.paths.require("round_index", "world_state_dir", "out_dir", *DataPaths.names())
    if cfg.run.index_row is None:
        raise ValueError("run.index_row is not set: one client renders one row of the round index")
    row = load_round_index_row(cfg.paths.round_index, int(cfg.run.index_row))
    world_state = client_world_state(cfg, row.media_id)
    try:
        client = Client(cfg, ServingOptions(decoder="none"))
        client.start(row, world_state=world_state)
        while not client.finished:
            deque(client.step(), maxlen=0)
        client.stop()
    except BaseException as exc:
        world_state.mark_done(status=DONE_FAILED, note=_failure_note(exc))
        raise
    world_state.mark_done(status=DONE_OK)

    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latents = client.latents[0].float().cpu().numpy()
    np.save(out_dir / "latents.npy", latents)
    wait = world_state.wait_stats
    summary = dict(
        media_id=row.media_id,
        index_row=cfg.run.index_row,
        seed=cfg.run.seed,
        latent_frames=len(latents),
        blocks=[dict(f0=r.f0, **asdict(r.read)) for r in client.records if r.read is not None],
        follow=asdict(client.follow_stats),
        wait=dict(n_waits=wait.n_waits, seconds_total=round(wait.seconds_total, 3)),
    )
    (out_dir / "client.json").write_text(json.dumps(summary, indent=1))
    return summary


def _failure_note(error: BaseException) -> str:
    """Why a client failed, as its done marker says it."""
    return f"{type(error).__name__}: {error}"


@dataclass
class _Delivery:
    """Latent frames ``first .. first + count - 1`` handed from the generation thread to the
    caller's, with their block's record and, on CUDA, the event that marks them written; and for
    a rollout without a last block, its ``x0`` rounded to the output's dtype ``[1, 4, 48, 24,
    42]`` (the caller's thread decodes the block's frames from it, not from the output, whose
    latent frames such a rollout writes again with every block). Without it, as for the first
    frame and every block of the paper's rollouts, the frames are the output's.

    A block that may be refreshed comes with what the page reads of it (``page``,
    :meth:`~worldcast.engine.inference.live_rollout.LiveRollout.page`), its version and the
    controls it was cut on; a refresh with its ``switch`` and its record, and whether the block's
    first version's frames took it over or a newer version came first (``taken``: its own iterator
    yields none). A refresh's ``preview`` (the x0 of an earlier denoising step) comes before its
    own frames, which take it over; the refresh's iterator takes both."""

    first: int
    count: int
    record: BlockRecord
    ready: Any = None
    latents: torch.Tensor | None = None
    page: tuple[dict, dict] | None = None
    version: int = 0
    switch: Switch | None = None
    refresh: RefreshRecord | None = None
    started: Any = None
    controls: Any = None
    taken: bool = False
    preview: bool = False


class _RunningRollout:
    """A rollout as the client runs it: its blocks generated on the client's generation thread, its
    frames decoded on the thread that takes them.

    With replicas of the generator (refresh on, ``ServingOptions.refresh_devices``) each version
    of the open block is cut on the generation thread's card in the order it was asked for
    (:meth:`LiveRollout.cut`) and denoised on its card's thread; a version is shown when it is the
    newest done (by the order of the cuts), and the next step commits the newest cut once every
    version of the block is done. The denoising on the client's card runs on a stream of its own,
    so that the cut of another card's refresh never waits behind it. The warm-up's replay
    (``mirror``) denoises each window block again on every replica and compares.
    """

    #: A live rollout with ``live.refresh``: its newest window block stays open.
    refreshes = False
    #: The open block: its f0, record and newest x0 (the generation thread's).
    open: tuple[int, BlockRecord, torch.Tensor | None] | None = None
    #: The client's generator on other cards; with them but no refresh (the warm-up's replay) each
    #: window block is denoised again on each (``mirror``).
    replicas: tuple[GeneratorReplica, ...] = ()
    mirror = False

    def __init__(self, client: Client, row: RoundIndexRow, world_state: WorldState | None) -> None:
        cfg = client.cfg
        self.client = client
        self.refreshes = cfg.player_state.source == "live" and cfg.live.refresh
        #: The world state over messages the rollout opened; ``None`` on the caller's world state.
        self.messages: MessageWorldState | None = None
        if world_state is None:
            # a live session has no end: its world state keeps the newest blocks of each client, as
            # many as hold the cameras of every entry the bank keeps; the first six entries are
            # published together, after the sixth block
            retain = None
            if cfg.player_state.source == "live":
                retain = max(LIVE_STATES_PER_ENTRY * cfg.scene_state.bound, FIRST_TARGET // BLOCK)
            self.messages = world_state = MessageWorldState(
                client=row.media_id,
                poll_s=cfg.world_state.poll_s,
                outbox=client.on_message,
                retain=retain,
            )
        self.rollout = (LiveRollout if cfg.player_state.source == "live" else Rollout)(
            cfg,
            client.models,
            Sampler.create(client.generator, client.cache, context_noise=context_noise(cfg)),
            load_window(cfg, row),
            world_state,
            lockstep=client.serving.lockstep,
            dtype=client.dtype,
        )
        self.decoder = None
        if client.serving.decoder == "wan":
            self.decoder = WanFrameDecoder(client.models.wan_vae)
        elif client.serving.decoder == "taehv":
            self.decoder = StreamingTinyDecoder(client.models.tiny)
        self.next_f0 = 1
        self.next_frame = 0
        self.delivered: queue.Queue[_Delivery] = queue.Queue()
        self.jobs: list[Future] = []
        self.failed: BaseException | None = None
        self._lock = threading.Lock()
        # the decoder's state before each latent frame of the open block in the version on screen,
        # and after its last; whether the decoder is there (the caller's thread)
        self.chain: list[Any] = []
        self.chain_end: Any = None
        self.at_end = True
        self.latent_s = 0.0  # what the decoder took for a latent frame, smoothed, s
        # set once the newest block's first latent frame is decoded: with the decoder on the
        # generator's device a refresh's denoising waits for it, so as not to slow that frame down
        self.first_decoded = threading.Event()
        self.first_decoded.set()
        # the generator on more cards: every block's context is prefilled on them too
        self.replicas = client.replicas
        self.rollout.replicas = self.replicas
        self.mirror = bool(self.replicas) and not self.refreshes
        self._pending = [0] * (1 + len(self.replicas))  # jobs not done, per card
        self._broken = False  # a job failed: those that wait for another stop
        # with replicas: the block and version cut next, the refreshes asked per block, the
        # (version, own frames) delivered newest of the open block, its first version delivered,
        # and the refreshes on the replicas' threads per block
        self._cuts = threading.Condition()
        self._turn = (0, 0)
        self._asked: dict[int, int] = {}
        self._newest = (-1, 0)
        self._opened = threading.Event()
        self._opened.set()
        self._replica_jobs: dict[int, list[Future]] = {}
        self._submit(self._open)

    @property
    def finished(self) -> bool:
        return not self.rollout.open_ended and self.next_f0 + BLOCK > self.rollout.latent_frames

    @property
    def can_refresh(self) -> bool:
        return self.refreshes and self.next_f0 - BLOCK >= FIRST_TARGET

    def free_card(self) -> int | None:
        """A card with no job, the replicas' first (the decoder shares the client's): an index
        of ``refresh_devices``; ``None``: none is free."""
        with self._lock:
            order = [*range(1, len(self._pending)), 0]
            return next((card for card in order if not self._pending[card]), None)

    # ------------------------------------------------------------------- the generation thread
    def _open(self) -> None:
        with torch.no_grad():
            self.rollout.open()
        self.delivered.put(_Delivery(0, 1, BlockRecord(f0=0)))

    def _block(self, f0: int, controls: StepControls) -> None:
        record = BlockRecord(f0=f0, t_start=time.monotonic())
        taken: list[Any] = [None]

        def take() -> BlockControls | None:
            record.t_controls = time.monotonic()
            taken[0] = controls() if callable(controls) else controls
            return taken[0]

        with torch.no_grad():
            self._commit_open()
            if self.replicas and self.refreshes and self.rollout.opens(f0):
                self._open_on_cards(f0, record, take, taken)
                return
            rng = rng_state(self.client.device) if self.mirror and f0 >= FIRST_TARGET else None
            x0 = self.rollout.denoise(f0, take)
            latents = x0.to(self.rollout.output.dtype) if self.rollout.open_ended else None
            ready = torch.cuda.Event() if self.client.stream is not None else None
            if ready is not None:
                ready.record()
            if self.refreshes and self.rollout.can_refresh(f0):
                # the block stays open: committed by the next step, as its newest version
                record.t_latents = time.monotonic()
                self.client.records.append(record)
                self.open = f0, record, x0
                page = self.rollout.page(f0)
                if self.decoder is not None and self.client.serving.decoder_device is None:
                    self.first_decoded.clear()
                delivery = _Delivery(f0, BLOCK, record, ready, latents, page, controls=taken[0])
                self.delivered.put(delivery)
                return
            if rng is not None:
                self._compare_replicas(f0, x0, rng)
            record.read = self.rollout.commit(f0, x0)
            record.t_latents = time.monotonic()
            self.client.records.append(record)
            self.delivered.put(_Delivery(f0, BLOCK, record, ready, latents))
            self.rollout.prepare_next(f0)

    def _compare_replicas(self, f0: int, x0: torch.Tensor, rng: torch.Tensor) -> None:
        """The warm-up's check: block ``f0`` (its recorded controls, not yet committed) denoised
        again on each replica from the same re-noise state, against ``x0``."""

        def again(replica: GeneratorReplica) -> torch.Tensor:
            with torch.no_grad():
                return self.rollout.denoise_on(replica, f0, rng)

        for replica in self.replicas:
            try:
                other = replica.thread.submit(again, replica).result().to(x0.device)
            except Exception:  # the check's: a replica that fails is no replica
                log.exception("the replica on %s failed on block %d", replica.device, f0)
                self.client.replicas_equal.append(False)
                continue
            equal = torch.equal(other, x0)
            self.client.replicas_equal.append(equal)
            if not equal:
                log.warning(
                    "the replica on %s denoised block %d otherwise than %s (max |dx0| %.3g): its"
                    " kernels differ",
                    replica.device,
                    f0,
                    self.client.device,
                    float((other - x0).abs().max()),
                )

    def _commit_open(self) -> None:
        """Commit the open block's newest version and prepare the block after it; with replicas,
        once every refresh of the block is done."""
        if self.open is None:
            return
        f0, record, x0 = self.open
        with self._lock:
            futures = self._replica_jobs.pop(f0, [])
        for future in futures:
            future.result().result()  # its cut, then its denoising
        if self.replicas:  # the newest version is the newest cut once all are done
            f0, record, x0 = self.open
        self.open = None
        record.read = self.rollout.commit(f0, x0)
        self.rollout.prepare_next(f0)

    def _refresh(
        self,
        f0: int,
        controls: StepControls,
        switch: Switch,
        generated: Callable[[], None] | None,
        preview_step: int,
    ) -> None:
        _, record, _ = self.open
        refresh = RefreshRecord(t_start=time.monotonic())
        taken: list[Any] = [None]

        def take() -> BlockControls | None:
            refresh.t_controls = time.monotonic()
            taken[0] = controls() if callable(controls) else controls
            return taken[0]

        cuda = self.client.stream is not None
        self.first_decoded.wait()
        version = len(record.refreshes) + 1
        on_step = self._previewer(
            preview_step, f0, record, refresh, version, switch, taken, lambda: self.rollout.page(f0)
        )
        with torch.no_grad():
            started = torch.cuda.Event(enable_timing=True) if cuda else None
            if started is not None:
                started.record()
            x0 = self.rollout.refresh(f0, take, on_step)
            refresh.t_cut = self.rollout.t_cut
            latents = x0.to(self.rollout.output.dtype)
            ready = torch.cuda.Event(enable_timing=True) if cuda else None
            if ready is not None:
                ready.record()
            refresh.t_latents = time.monotonic()
            refresh.denoise_ms = 0.0 if cuda else 1e3 * (refresh.t_latents - refresh.t_cut)
            self.open = f0, record, x0
            record.refreshes.append(refresh)
            delivery = _Delivery(
                f0, BLOCK, record, ready, latents, self.rollout.page(f0), len(record.refreshes)
            )
            delivery.switch, delivery.refresh, delivery.started = switch, refresh, started
            delivery.controls = taken[0]
            self.delivered.put(delivery)
            if generated is not None:
                generated()

    # --------------------------------------------------------- the generation on more cards
    def _open_on_cards(
        self, f0: int, record: BlockRecord, take: Callable[[], Any], taken: list[Any]
    ) -> None:
        """The first version of block ``f0`` with replicas: cut first of the block's versions,
        denoised on the client's card."""
        self._opened.clear()
        try:
            with self._cuts:
                cut = self.rollout.cut(f0, take)
                self._turn, self._newest = (f0, 1), (-1, 0)
                with self._lock:
                    for block in [block for block in self._asked if block < f0]:
                        del self._asked[block]
                self.open = f0, record, None
                self._cuts.notify_all()
            if self.decoder is not None and self.client.serving.decoder_device is None:
                self.first_decoded.clear()
            x0, latents, ready, _ = self._denoise_cut(cut, 0, None)
            record.t_latents = time.monotonic()
            self.client.records.append(record)
            with self._lock:
                self._newest = (0, 1)
                self.open = f0, record, x0
                self.delivered.put(
                    _Delivery(f0, BLOCK, record, ready, latents, cut.page, controls=taken[0])
                )
        finally:
            self._opened.set()

    def _refresh_on(
        self,
        card: int,
        f0: int,
        version: int,
        controls: StepControls,
        switch: Switch,
        preview_step: int,
    ) -> None:
        """Refresh ``version`` of the open block ``f0`` on the client's card, on its thread: its
        cut in its turn, then its denoising."""
        refresh, taken, take = self._version(card, controls)
        with torch.no_grad():
            cut, record = self._cut_version(f0, version, take, refresh)
            self.first_decoded.wait()
            self._denoise_version(
                card, f0, version, cut, record, refresh, switch, taken, preview_step
            )

    def _refresh_on_replica(
        self,
        card: int,
        f0: int,
        version: int,
        controls: StepControls,
        switch: Switch,
        generated: Callable[[], None] | None,
        preview_step: int,
    ) -> Future:
        """Refresh ``version`` of the open block ``f0`` on replica ``card``: cut in its turn on
        the cuts' thread, then denoised on the replica's thread, behind the block's prefill;
        ``generated()`` after. Returns the future of the cut, whose result is the denoising's."""
        refresh, taken, take = self._version(card, controls)

        def cut_then_denoise() -> Future:
            try:
                with torch.no_grad():
                    cut, record = self._cut_version(f0, version, take, refresh)
            except BaseException:
                self._broken = True
                self._release(card)
                raise
            args = (card, f0, version, cut, record, refresh, switch, taken, preview_step)
            thread = self.replicas[card - 1].thread
            future = thread.submit(self._run, card, self._denoise_version, generated, *args)
            with self._lock:
                self.jobs.append(future)
            return future

        with self._lock:
            self._pending[card] += 1
        future = self.client.cuts.submit(cut_then_denoise)
        with self._lock:
            self.jobs.append(future)
        return future

    @staticmethod
    def _version(
        card: int, controls: StepControls
    ) -> tuple[RefreshRecord, list[Any], Callable[[], Any]]:
        """A refresh's record, its controls as taken and what takes them."""
        refresh = RefreshRecord(t_start=time.monotonic(), card=card)
        taken: list[Any] = [None]

        def take() -> BlockControls | None:
            refresh.t_controls = time.monotonic()
            taken[0] = controls() if callable(controls) else controls
            return taken[0]

        return refresh, taken, take

    def _cut_version(
        self, f0: int, version: int, take: Callable[[], Any], refresh: RefreshRecord
    ) -> tuple[BlockCut, BlockRecord]:
        """Cut ``version`` of the open block ``f0`` once the versions before it are (the cut is
        the client card's work: a bare ``cuda`` in it is that card, on any thread)."""
        cuda = self.client.stream is not None
        on_card = torch.cuda.device(self.client.device) if cuda else contextlib.nullcontext()
        with self._cuts:
            while self._turn != (f0, version):
                self._waiting()
                if self._turn > (f0, version):
                    raise RuntimeError(f"refresh {version} of block {f0} missed its turn")
                self._cuts.wait(0.05)
            try:
                with on_card:
                    cut = self.rollout.cut(f0, take)
            finally:
                self._turn = (f0, version + 1)
                self._cuts.notify_all()
            record = self.open[1]
            record.refreshes.append(refresh)
        refresh.t_cut = cut.t_cut
        return cut, record

    def _denoise_version(
        self,
        card: int,
        f0: int,
        version: int,
        cut: BlockCut,
        record: BlockRecord,
        refresh: RefreshRecord,
        switch: Switch,
        taken: list[Any],
        preview_step: int,
    ) -> None:
        """Denoise a cut version on ``card``; its frames are shown if it is the newest done."""
        with torch.no_grad():
            on_step = self._previewer(
                preview_step, f0, record, refresh, version, switch, taken, lambda: cut.page
            )
            x0, latents, ready, refresh.denoise_ms = self._denoise_cut(cut, card, on_step)
        refresh.t_latents = time.monotonic()
        while not self._opened.wait(0.05):
            self._waiting()
        with self._lock:
            newest = (version, 1) > self._newest
            if newest:
                self._newest = (version, 1)
                self.open = f0, record, x0
            delivery = _Delivery(
                f0, BLOCK, record, ready, latents, cut.page, version, switch, refresh
            )
            delivery.controls, delivery.taken = taken[0], not newest
            self.delivered.put(delivery)

    def _waiting(self) -> None:
        """A job that waits for another stops once a job failed."""
        if self._broken:
            raise RuntimeError("an earlier job of the rollout failed")

    def _denoise_cut(
        self, cut: BlockCut, card: int, on_step: Callable[[int, torch.Tensor], None] | None
    ) -> tuple[torch.Tensor, torch.Tensor, Any, float]:
        """A cut denoised on ``card``: x0 on the client's card, it in the output's dtype, the
        event once both are there (CUDA), and the denoising's ms; done when it returns."""
        client, dtype = self.client, self.rollout.output.dtype
        replica = self.replicas[card - 1] if card else None
        if client.stream is None:
            t0 = time.monotonic()
            x0 = self.rollout.denoise_cut(cut, replica, on_step)
            return x0, x0.to(dtype), None, 1e3 * (time.monotonic() - t0)
        started, ended = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        if replica is None:
            client.denoise_stream.wait_stream(torch.cuda.current_stream(client.device))
            with torch.cuda.stream(client.denoise_stream):
                started.record()
                x0 = self.rollout.denoise_cut(cut, None, on_step)
                latents = x0.to(dtype)
                ended.record()
            ended.synchronize()
            return x0, latents, ended, started.elapsed_time(ended)
        started.record()
        x0 = self.rollout.denoise_cut(cut, replica, on_step)
        ended.record()
        x0 = x0.to(client.device)
        latents = x0.to(dtype)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(client.device))
        ready.synchronize()
        return x0, latents, ready, started.elapsed_time(ended)

    def _previewer(
        self,
        step: int,
        f0: int,
        record: BlockRecord,
        refresh: RefreshRecord,
        version: int,
        switch: Switch,
        taken: list[Any],
        page: Callable[[], tuple[dict, dict]],
    ) -> Callable[[int, torch.Tensor], None] | None:
        """What hands a refresh's preview to the decoder: the x0 of its denoising step ``step``
        (from 1; 0: no preview), on the client's card in the output's dtype. With replicas none
        goes before the block's first version, nor after a newer version."""
        if not step:
            return None
        client, dtype = self.client, self.rollout.output.dtype

        def on_step(i: int, x0: torch.Tensor) -> None:
            if i != step - 1 or not self._opened.is_set():
                return
            latents = x0.to(dtype).to(client.device)
            ready = None
            if client.stream is not None:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(client.device))
            refresh.t_preview = time.monotonic()
            delivery = _Delivery(
                f0, BLOCK, record, ready, latents, page(), version, switch, refresh
            )
            delivery.controls, delivery.preview = taken[0], True
            with self._lock:
                if self.replicas:
                    if (version, 0) <= self._newest:
                        return
                    self._newest = (version, 0)
                self.delivered.put(delivery)

        return on_step

    def _submit(
        self,
        job: Callable[..., None],
        *args: Any,
        card: int = 0,
        then: Callable[[], None] | None = None,
    ) -> Future:
        """Run ``job`` on ``card``'s thread; ``then()`` after it is done and its card counted
        free."""
        with self._lock:
            self._pending[card] += 1
        thread = self.replicas[card - 1].thread if card else self.client.thread
        future = thread.submit(self._run, card, job, then, *args)
        with self._lock:
            self.jobs.append(future)
        return future

    def _run(
        self, card: int, job: Callable[..., None], then: Callable[[], None] | None, *args: Any
    ) -> None:
        try:
            job(*args)
        except BaseException:
            self._broken = True
            raise
        finally:
            self._release(card)
        if then is not None:
            then()

    def _release(self, card: int) -> None:
        """A job of ``card`` is done: the card is free when it was its last."""
        with self._lock:
            self._pending[card] -= 1
            free = not self._pending[card]
        if free and self.client.card_freed is not None:
            self.client.card_freed()

    def _check(self) -> None:
        """Raise the error of a finished job, once."""
        with self._lock:
            done = [f for f in self.jobs if f.done()]
            self.jobs = [f for f in self.jobs if not f.done()]
        for future in done:
            if future.exception() is not None:
                self.failed = self.failed or future.exception()
                raise future.exception()

    # ----------------------------------------------------------------------- the caller's thread
    def step(self, controls: StepControls) -> Iterator[Frame]:
        if self.finished:
            return iter(())
        f0, self.next_f0 = self.next_f0, self.next_f0 + BLOCK
        self._submit(self._block, f0, controls)
        return self._drain(f0)

    def refresh(
        self,
        controls: StepControls,
        switch: Switch,
        generated: Callable[[], None] | None,
        preview_step: int,
        card: int | None,
    ) -> Iterator[Frame]:
        if not self.can_refresh:
            raise RuntimeError(
                "the newest block cannot be refreshed: a live rollout with live.refresh true"
                " refreshes the blocks after the first six"
            )
        f0 = self.next_f0 - BLOCK
        if not self.replicas:
            if card:
                raise ValueError(f"card {card}: the client has no replica")
            self._submit(self._refresh, f0, controls, switch, generated, preview_step)
            return self._drain(f0)
        if card is None:
            card = self.free_card()
        if card is None:  # every card busy: the one with the fewest jobs, the replicas' first
            with self._lock:
                card = min(range(len(self._pending)), key=lambda c: (self._pending[c], not c))
        with self._lock:
            version = self._asked[f0] = self._asked.get(f0, 0) + 1
        if card:
            future = self._refresh_on_replica(
                card, f0, version, controls, switch, generated, preview_step
            )
            with self._lock:
                self._replica_jobs.setdefault(f0, []).append(future)
        else:
            args = (card, f0, version, controls, switch, preview_step)
            self._submit(self._refresh_on, *args, card=card, then=generated)
        return self._drain(f0)

    def _drain(self, f0: int) -> Iterator[Frame]:
        """The frames of the blocks delivered up to block ``f0`` (a refresh's: up to its own
        frames, after its preview), decoded on the caller's thread."""
        stream, output = self.client.stream, self.rollout.output
        while True:
            try:
                delivery = self.delivered.get(timeout=0.05)
            except queue.Empty:
                self._check()
                continue
            if delivery.ready is not None:
                stream.wait_event(delivery.ready)
            block = output[0, delivery.first :] if delivery.latents is None else delivery.latents[0]
            if delivery.switch is None:
                yield from self._frames(delivery, block)
            else:
                yield from self._refreshed(delivery, block)
            if delivery.first == f0 and not delivery.preview:
                self._check()
                return

    def _decode(self, block: torch.Tensor, i: int) -> tuple[list[np.ndarray], float]:
        """The frames of latent frame ``i`` of ``block`` and when they were ready."""
        stream = self.client.stream
        on_stream = contextlib.nullcontext() if stream is None else torch.cuda.stream(stream)
        with on_stream, torch.no_grad():
            if self.decoder is None:
                images = [block[i].float().cpu().numpy()]
            else:
                pixels = self.decoder.decode(block[i : i + 1])
                images = pixels_to_uint8_frames(pixels)
        return images, time.monotonic()

    def _frames(self, delivery: _Delivery, block: torch.Tensor) -> Iterator[Frame]:
        """The frames of a delivery in order: the first frame, or a block's first version; for a
        block that may be refreshed, as :meth:`_versions` decodes them (a refresh delivered
        meanwhile takes over)."""
        if delivery.page is not None and self.decoder is not None:
            if not self.at_end:  # the block before ended on a version that stopped short
                self.decoder.restore(self.chain_end)
                self.at_end = True
            self.chain = [None] * delivery.count
            first = last_video_frame(delivery.first - 1) + 1
            yield from self._versions(delivery, block, 0, first, opening=True)
            return
        for i in range(delivery.count):
            images, t_ready = self._decode(block, i)
            record = delivery.record
            record.t_first_frame = record.t_first_frame or t_ready
            record.t_last_frame = t_ready
            for image in images:
                yield self._frame(self.next_frame, delivery, image, t_ready)
                self.next_frame += 1

    def _refreshed(self, delivery: _Delivery, block: torch.Tensor) -> Iterator[Frame]:
        """A refresh's frames (or its preview's) from the switch on (:meth:`Client.refresh`), as
        :meth:`_versions` decodes them; none when the block's first version took it over, or a
        newer refresh of the block (or the refresh's own frames, after its preview) is already
        delivered: its frames would take their places."""
        refresh = delivery.refresh
        if not delivery.taken and self._queued_refresh(delivery.first) is None:
            per = VIDEO_FRAMES_PER_LATENT if self.decoder is not None else 1
            first = last_video_frame(delivery.first - 1) + 1 if per > 1 else delivery.first
            row = max(int(delivery.switch(self.latent_s)) - first, 0)
            if self.decoder is not None and row < delivery.count * per:
                self.decoder.restore(self.chain[row // per])
                self.at_end = False
                yield from self._versions(delivery, block, row // per, first + row, opening=False)
            elif self.decoder is None:
                for i in range(row, delivery.count):
                    images, t_ready = self._decode(block, i)
                    _stamp(delivery, t_ready)
                    _stamp_first(delivery, first + i)
                    yield self._frame(first + i, delivery, images[0], t_ready)
        if delivery.started is not None:
            delivery.ready.synchronize()
            refresh.denoise_ms = delivery.started.elapsed_time(delivery.ready)

    def _versions(
        self, delivery: _Delivery, block: torch.Tensor, latent: int, skip: int, *, opening: bool
    ) -> Iterator[Frame]:
        """Decode ``block``, a version of a block that may be refreshed, from latent frame
        ``latent`` on, the decoder's state the one before it in the version on screen, and yield
        its frames from frame ``skip`` on; the decoder's state before each latent frame is kept.

        At each latent frame, a newer refresh of the block delivered meanwhile ends the decode,
        so that the decoder spends no time on frames a newer version replaces. The block's first
        version (``opening``: its frames are not all decoded) hands over to it there: from the
        latent frame of its switch on, but not past the first frame not yet decoded (whose frames
        the refresh's give). A refresh's own decode stops (from its second latent frame), and the
        newer refresh's iterator decodes it.
        """
        per = VIDEO_FRAMES_PER_LATENT
        base = last_video_frame(delivery.first - 1) + 1
        decoded = latent if opening else delivery.count  # latent frames with frames on screen
        segment = latent  # the first latent frame of this version's decode
        i = latent
        while i < delivery.count:
            self.chain[i] = self.decoder.snapshot()
            if opening:
                newer = self._take_over(delivery.first)
                if newer is not None:
                    row = max(int(newer.switch(self.latent_s)) - base, 0)
                    j = min(row // per, decoded)
                    if j == delivery.count:  # all decoded, and none of it to decode
                        self.decoder.restore(self.chain_end)
                        self.at_end = True
                        return
                    skip = base + min(row, decoded * per)
                    if j != i:
                        self.decoder.restore(self.chain[j])
                    if newer.ready is not None:
                        self.client.stream.wait_event(newer.ready)
                    delivery, block, i = newer, newer.latents[0], j
                    segment = j
                    continue
            elif i > segment and self._queued_refresh(delivery.first) is not None:
                self.at_end = False
                return
            started = time.monotonic()
            images, t_ready = self._decode(block, i)
            if opening and not i:
                self.first_decoded.set()
            if i > segment:  # the first one of a version also waits for its generation
                took = t_ready - started
                follow = LATENT_TIME_FOLLOW if self.latent_s else 1.0
                self.latent_s += follow * (took - self.latent_s)
            self.at_end = i == delivery.count - 1
            if self.at_end:
                self.chain_end = self.decoder.snapshot()
            _stamp(delivery, t_ready)
            for k, image in enumerate(images):
                index = base + i * per + k
                if index < skip:
                    continue
                _stamp_first(delivery, index)
                self.next_frame = max(self.next_frame, index + 1)
                yield self._frame(index, delivery, image, t_ready)
            i += 1
            decoded = max(decoded, i)

    def _queued_refresh(self, f0: int) -> _Delivery | None:
        """A refresh of block ``f0`` delivered and not yet taken by its iterator, the newest (one
        that yields none, taken over or older than a version shown, is none)."""
        with self.delivered.mutex:
            found = [
                d
                for d in self.delivered.queue
                if d.switch is not None and d.first == f0 and not d.taken
            ]
        return found[-1] if found else None

    def _take_over(self, f0: int) -> _Delivery | None:
        """The newest refresh of block ``f0`` delivered that the block's first version's frames
        did not take over yet, taken over with those before it (their iterators yield none)."""
        with self.delivered.mutex:
            found = [
                d
                for d in self.delivered.queue
                if d.switch is not None and d.first == f0 and not d.taken
            ]
            for d in found:
                d.taken = True
        return found[-1] if found else None

    def _frame(self, index: int, delivery: _Delivery, image: np.ndarray, t_ready: float) -> Frame:
        if delivery.page is None:
            return Frame(index, delivery.first, image, t_ready)
        rows, pitch = delivery.page
        latent = -(-index // VIDEO_FRAMES_PER_LATENT) if self.decoder is not None else index
        return Frame(
            index,
            delivery.first,
            image,
            t_ready,
            delivery.version,
            rows[latent],
            pitch.get(index) if self.decoder is not None else None,
            delivery.controls,
            delivery.preview,
        )

    def close(self) -> None:
        """Wait for the pending jobs, the commit of an open block among them, and mark an own world
        state done; raise a new error."""
        self.first_decoded.set()  # a refresh that waits for a frame nobody takes goes on
        if self.refreshes and self.failed is None:
            self._submit(self._commit_open)
        with self._lock:
            jobs, self.jobs = self.jobs, []
        error = next((f.exception() for f in jobs if f.exception() is not None), None)
        failed = self.failed or error
        if self.messages is not None:
            note = "" if failed is None else _failure_note(failed)
            self.messages.mark_done(status=DONE_FAILED if failed else DONE_OK, note=note)
        if error is not None:
            raise error


def _stamp(delivery: _Delivery, t_ready: float) -> None:
    """A latent frame of ``delivery`` decoded at ``t_ready``: its record's times (a preview's
    first frame in its refresh's record)."""
    if delivery.preview:
        delivery.refresh.t_preview_frame = delivery.refresh.t_preview_frame or t_ready
        return
    times = delivery.record if delivery.refresh is None else delivery.refresh
    times.t_first_frame = times.t_first_frame or t_ready
    times.t_last_frame = t_ready


def _stamp_first(delivery: _Delivery, index: int) -> None:
    """Frame ``index`` of ``delivery`` yielded: the first of a refresh (or of its preview)."""
    refresh = delivery.refresh
    if refresh is None:
        return
    if delivery.preview:
        refresh.preview_frame = index if refresh.preview_frame < 0 else refresh.preview_frame
    elif refresh.first_frame < 0:
        refresh.first_frame = index


def _raise_recompile_limit(cards: int) -> None:
    """Let ``torch.compile`` keep the graphs of the DiT block's parts on ``cards`` cards: dynamo's
    cache per code object holds each card's guards, past which it would run a card's parts
    eagerly (slower, and not the other cards' kernels)."""
    import torch._dynamo.config as dynamo

    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo, name):
            setattr(dynamo, name, max(getattr(dynamo, name), _RECOMPILES_PER_CARD * cards))
