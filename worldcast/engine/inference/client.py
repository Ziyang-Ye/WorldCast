"""One WorldCast client: one player's view of one round, on one GPU, in lock-step with the round's other clients.

This module only orchestrates; every computation lives in the area modules. The order is the paper client's
(``tools/eval/memory_deploy.py`` + ``wan_utils/worldplay_rollout.reconstituted_rollout`` on the paper path;
docs/inference.md, "The block loop")::

    load config, TF32 on                                   worldcast.config.inference, worldcast.utils.precision
    generator, depth head, prompt embedding                worldcast.modeling.{loading,depth_head,text}
    client window (index row, ticks, labels, latent 0)     worldcast.data
    entry noise (CPU generator), sink = recorded latent 0  worldcast.sampling.schedulers.entry_noise
    seed, plain prefix: latents 1-24, absolute positions   Sampler.rollout_prefix
    for s = 25, 29, ...  (t = start + 8 s):
      1  own write            scene.ingest_own             (the own block that ended before t)
      2  step record          pool.publish_step            (own withdrawals, own resident entries)
      3  lock-step wait       pool.wait_for_peers          (inside PeerBlocks.admit)
      4  admit peers' blocks  PeerBlocks.admit, scene.ingest_peers
      5  follow withdrawals   scene.follow_withdrawals
      6  retrieve <= 1 entry  scene.retrieve               (fill rule, k = 1; None = abstain)
      7  compacted window     window.prepare_window        (sink | slot 4 | recent 12 | target 4; 17 on abstain)
      8  KV prefill           Sampler.generate_block       (context re-noised to t = 16)
      9  4-step ladder        Sampler.generate_block       (field built at the injection point by the adapter)
     10  publish the block    pool.publish_block, PeerBlocks.add_own
    latents.npy (float32), DONE marker in the pool

Numerics kept on purpose: TF32 on; bf16 parameters and a bf16 cast of every generator input,
timesteps included (``CausalGeneratorAdapter``); context noise label 16 at level t ~ 14.82; the ladder re-noise from
the global RNG of the device after :func:`set_seed`; entry noise from a CPU generator; per-(seed, s, role) CPU
generators for the context writes; ``camera_encoding = noclip`` passed explicitly; the observer-signal root from the
config. The store the scene state, the window and the pool read keeps the unrounded float32 x0 of reconstituted
blocks; ``latents.npy`` is the bf16 output buffer written as float32, as the paper client wrote it. See
docs/inference.md, "Numerics that the paper's numbers depend on".

With ``player_state.source = predicted`` (closed-loop deployment, docs/inference.md, "Closed loop") the client
also publishes its own estimated position before each block (``worldcast.player_state.closed_loop``), draws every other
client where that client published itself, uses its own estimated cameras for the rays, the retrieval and the keys
of its memory entries, and predicts visibility; the plain prefix then runs block by block with the same exchange.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldcast.config.inference import InferenceConfig
from worldcast.data.actions import OPENCS2_WEAPONS, read_control_ticks
from worldcast.data.index import RoundIndexRow, load_round_index_row
from worldcast.data.item import ClientWindow, DataPaths, load_client_window
from worldcast.data.media import MediaIndex
from worldcast.data.memory_slot import MemorySlotFrames, published_block_candidate
from worldcast.data.player_frames import WindowSpec, covered_frames
from worldcast.data.ticks import read_player_ticks
from worldcast.modeling.build import (
    generator_config_from_inference,
    generator_config_from_snapshot,
    load_generator,
)
from worldcast.modeling.depth_head import PictureDepth
from worldcast.modeling.state_model import StateTables, load_state_model
from worldcast.modeling.wan22.attention import AttentionFn
from worldcast.modeling.wan22.model import (
    CausalGeneratorAdapter,
    GeneratorConfig,
    KVCache,
    WorldCastGenerator,
)
from worldcast.player_state.closed_loop import ClosedLoop, StateExchange, StateReader
from worldcast.player_state.extrapolate import PhysicsPrior, prior_channels
from worldcast.player_state.field import FieldConfig, build_field
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

from .pool import LocalDirPool, PeerBlocks, WaitStats

__all__ = ["Client", "ClientResult", "run_client", "client_latents", "FixedPromptEncoder"]

BLOCK = W.BLOCK


class FixedPromptEncoder:
    """The text encoder the sampler calls, answering the fixed prompt with one precomputed embedding.

    The paper client re-encoded the fixed prompt with umT5 every block; the result is the same tensor every time,
    so it is encoded (or loaded) once. ``embeds``: ``[1, 512, 4096]`` bf16.
    """

    def __init__(self, prompt: str, embeds: torch.Tensor) -> None:
        self.prompt, self.embeds = str(prompt), embeds

    def __call__(self, *, text_prompts) -> dict[str, torch.Tensor]:
        if list(text_prompts) != [self.prompt]:
            raise ValueError(
                f"this client encodes only the fixed prompt {self.prompt!r}, got"
                f" {list(text_prompts)}"
            )
        return {"prompt_embeds": self.embeds}


def client_latents(cfg: InferenceConfig, observer_covered: int) -> int:
    """Latents N this client generates: ``run.latents`` clipped to the observer's coverage and to ``run.max_blocks``,
    rounded down to ``1 + 4k`` (``memory_deploy.py:1119-1125``). Raises if fewer than one reconstituted block fits.
    """
    first = cfg.window.first_target
    n = min(int(cfg.run.latents), 1 + max(0, int(observer_covered) - 1) // BLOCK)
    if cfg.run.max_blocks:
        n = min(n, first + BLOCK * int(cfg.run.max_blocks))
    n = 1 + ((n - 1) // BLOCK) * BLOCK
    if n < first + BLOCK:
        raise ValueError(
            f"the observer covers {observer_covered} pixel frames ({n} latents): shorter than the"
            f" plain prefix plus one block ({first + BLOCK} latents)"
        )
    return n


def field_config(cfg: InferenceConfig) -> FieldConfig:
    """The player state field settings (``worldcast.player_state.field``) from the release config."""
    pf = cfg.model.player_field
    return FieldConfig.from_action_buttons(
        cfg.data.action_buttons,
        pf.action_signals,
        grid_h=cfg.model.latent_height // cfg.model.patch_size[1],
        grid_w=cfg.model.latent_width // cfg.model.patch_size[2],
        camera_delta_scale=pf.camera_delta_scale,
        weapon_channels=pf.weapon_channels,
        splat_temperature=pf.splat_temperature,
        splat_sigma_floor=pf.splat_sigma_floor,
        splat_topk=pf.splat_topk,
        confidence_floor=pf.confidence_floor,
        confidence_smooth=pf.confidence_smooth,
        num_frame_per_block=cfg.window.block,
        independent_first_frame=cfg.window.independent_first_frame,
    )


@dataclass
class ClientResult:
    """What one client produced: ``latents_path`` ([N, 48, 24, 42] float32), the per-block reads, the lock-step
    waits and the counters of following the peers' withdrawals (copy checks)."""

    media_id: str
    latents_path: Path
    num_latents: int
    blocks: list[dict[str, Any]] = field(default_factory=list)
    wait: WaitStats = field(default_factory=WaitStats)
    follow: dict[str, int] = field(default_factory=dict)


class Client:
    """One client (see the module docstring for the order of work).

    Args:
        config: the release config; ``paths`` and ``run.index_row`` must be set.
        backbone: backbone dimensions (default: the Wan2.2-TI2V-5B snapshot's ``config.json`` when
            ``paths.wan22_root`` holds one, else the 5B defaults). The WorldCast additions and the KV-cache size
            always come from ``config`` (:func:`worldcast.modeling.build.generator_config_from_inference`).
        attention: attention kernel (default: flash-attention, as deployed; ``sdpa_attention`` for CPU tests).
    """

    def __init__(
        self,
        config: InferenceConfig,
        *,
        backbone: GeneratorConfig | None = None,
        attention: AttentionFn | None = None,
    ) -> None:
        self.cfg = config
        self.backbone = backbone
        self.attention = attention
        self.device = torch.device(config.run.device)
        self.dtype = getattr(torch, config.sampler.model_input_dtype)

    # ------------------------------------------------------------------------------------------------ set-up
    def _row(self) -> RoundIndexRow:
        if self.cfg.run.index_row is None:
            raise ValueError(
                "run.index_row is not set: one client renders one row of the round index"
            )
        return load_round_index_row(self.cfg.paths.round_index, int(self.cfg.run.index_row))

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
        """The client window at the length the observer's recording supports."""
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

    def load_models(self):
        """``(generator, prompt embedding [1, 512, 4096], depth head)`` on the client's device."""
        p = self.cfg.paths
        base = self.backbone
        if base is None:
            snapshot = Path(p.wan22_root or "") / "config.json"
            base = (
                generator_config_from_snapshot(snapshot)
                if p.wan22_root and snapshot.is_file()
                else GeneratorConfig()
            )
        gen_cfg = generator_config_from_inference(self.cfg, base)
        generator = load_generator(
            p.checkpoint, gen_cfg, device=self.device, attention=self.attention
        )
        if (
            self.dtype != torch.bfloat16
        ):  # float32: CPU tests only (CPU has no CUDA autocast for bf16)
            generator = generator.to(self.dtype)
        if p.prompt_embedding:
            from worldcast.modeling.wan22.text_encoder import PromptEmbedding

            embeds = PromptEmbedding.load(
                p.prompt_embedding, prompt=self.cfg.model.fixed_prompt, device=self.device
            ).embeds
        else:
            from worldcast.modeling.wan22.text_encoder import TextEncoder

            if not p.wan22_root:
                raise ValueError(
                    "set paths.prompt_embedding, or paths.wan22_root to encode the prompt with umT5"
                )
            encoder = TextEncoder.from_pretrained(
                p.wan22_root, device=self.device, dtype=torch.bfloat16
            )
            embeds = encoder.encode_prompt(self.cfg.model.fixed_prompt).embeds
            del encoder
        depth = PictureDepth.load(p.depth_head, p.depth_readout, device=self.device)
        return generator, embeds, depth

    def _sampler(self, generator: WorldCastGenerator) -> Sampler:
        fcfg = field_config(self.cfg)

        def field_builder(c, weapon_weight, frame_offset, num_frames):
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

        adapter = CausalGeneratorAdapter(generator, field_builder, input_dtype=self.dtype)
        gc = generator.config
        cache = KVCache.allocate(
            num_blocks=len(generator.blocks),
            num_heads=gc.num_heads,
            head_dim=gc.dim // gc.num_heads,
            capacity_latents=self.cfg.window.kv_cache_latents,
            frame_seq_length=self.cfg.model.frame_seq_length,
            batch_size=1,
            dtype=self.dtype,
            device=self.device,
        )
        return Sampler.from_config(self.cfg, adapter, cache)

    def _conditions(
        self,
        batch: Mapping[str, Any],
        geometry: Mapping[str, torch.Tensor],
        text: FixedPromptEncoder,
        sampler: Sampler,
    ):
        """Generator conditions of a window (or of the whole round for the prefix): controls, peer table and
        visibility gate (GT labels), observer signals, camera geometry and the prompt (old ``rollout_conditions``
        + ``_prepare_conditions``). Returns ``(VideoConditioning, merged conditions)``."""
        vc = W.video_conditioning(batch, device=self.device, dtype=self.dtype)
        material = PlayerStates.from_batch(batch, device=self.device)
        visible = GTLabelVisibility(device=self.device)(batch, material)
        cond = dict(vc.actions)
        cond.update(peer_conditions(material, visible, obs_signals=batch))
        cond.update(geometry)
        return vc, {**text(text_prompts=vc.prompts), **cond}

    # ------------------------------------------------------------------------------------------------ run
    def run(self) -> ClientResult:
        """Run the client; writes ``<out_dir>/latents.npy`` and ``<out_dir>/client.json``."""
        cfg = self.cfg
        cfg.paths.require(
            "checkpoint",
            "depth_head",
            "depth_readout",
            "round_index",
            "media_index",
            "dataset_root",
            "latent_cache_root",
            "visibility_label_root",
            "obs_signal_label_root",
            "live_pool_dir",
            "out_dir",
        )
        if cfg.sampler.tf32:
            enable_tf32()
        row = self._row()
        pool = LocalDirPool(
            cfg.paths.live_pool_dir,
            client=row.media_id,
            stride=cfg.data.skip_frame * BLOCK,
            poll_s=cfg.pool.poll_s,
            on_timeout="fatal" if cfg.pool.fail_on_timeout else "degrade",
            cell=row.media_id,
        )
        try:
            generator, embeds, depth = self.load_models()
            window = self.load_window(row)
            with torch.no_grad():
                output, blocks, follow = self._rollout(window, generator, embeds, depth, pool)
        except BaseException as exc:
            pool.mark_done(
                status="failed", note=f"{type(exc).__name__}: {exc}"
            )  # peers stop waiting
            raise
        pool.mark_done(status="ok")
        out_dir = Path(cfg.paths.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        latents_path = out_dir / "latents.npy"
        np.save(latents_path, output[0].float().cpu().numpy())
        result = ClientResult(
            media_id=row.media_id,
            latents_path=latents_path,
            num_latents=int(output.shape[1]),
            blocks=blocks,
            wait=pool.wait_stats,
            follow=follow,
        )
        summary = dict(
            media_id=row.media_id,
            index_row=cfg.run.index_row,
            seed=cfg.run.seed,
            num_latents=result.num_latents,
            blocks=blocks,
            follow=follow,
            wait=dict(
                n_waits=pool.wait_stats.n_waits,
                n_timeouts=pool.wait_stats.n_timeouts,
                seconds_total=round(pool.wait_stats.seconds_total, 3),
            ),
        )
        (out_dir / "client.json").write_text(json.dumps(summary, indent=1))
        return result

    def _closed_loop(
        self, window: ClientWindow, round_batch: dict[str, Any], depth, peers
    ) -> ClosedLoop:
        """The client's predicted states (``player_state.source = predicted``); rewrites ``round_batch``'s table."""
        cfg, p, row = self.cfg, self.cfg.paths, window.row
        p.require("state_model", "state_model_cells", "state_model_map_norm", "physics_prior")
        model = load_state_model(
            p.state_model,
            StateTables.load(p.state_model_cells, p.state_model_map_norm),
            device=self.device,
        )
        observer = window.observer
        reader = StateReader(
            model,
            read_control_ticks(Path(p.dataset_root) / observer.ticks_path),
            media_id=row.media_id,
            fps=observer.fps,
            start_frame=row.start_frame,
            device=self.device,
        )
        return ClosedLoop.build(
            round_batch,
            reader=reader,
            me=row.media_id,
            my_slot=observer.player_slot,
            round_media={s: m.media_id for s, m in window.round_slots.items()},
            clients=peers,
            n_latents=window.spec.latent_frames,
            start_frame=row.start_frame,
            motion_kwargs=dict(
                camera_delta_scale=cfg.model.player_field.camera_delta_scale,
                channels=prior_channels(cfg.data.action_buttons),
                prior=PhysicsPrior.load(p.physics_prior),
            ),
            eye_height=cfg.data.eye_height,
            depth_fn=depth.depth_grid,
            pose_radius=cfg.player_state.pose_radius_u,
            exchange=StateExchange(
                p.live_pool_dir,
                row.media_id,
                poll_s=cfg.pool.poll_s,
                fatal=cfg.pool.fail_on_timeout,
            ),
        )

    def _closed_prefix(
        self,
        closed: ClosedLoop,
        sampler: Sampler,
        text: FixedPromptEncoder,
        depth,
        prefix_batch: dict[str, Any],
        noise: torch.Tensor,
        sink: torch.Tensor,
        pool: LocalDirPool,
        orig,
    ) -> torch.Tensor:
        """The plain prefix under predicted states: :meth:`Sampler.rollout_prefix` block by block, each block opened
        by the position exchange and conditioned on the table and own cameras as they are then, with predicted
        visibility (re-tested after the first rung). Returns ``[1, 1 + n, C, H, W]`` in ``noise.dtype``.
        """
        dev, n = self.device, int(noise.shape[1])
        out = torch.zeros([1, n + 1, *noise.shape[2:]], device=dev, dtype=noise.dtype)
        out[:, :1] = sink
        vis = closed.visibility

        def conditions():
            batch = dict(prefix_batch, wp_own_c2w=closed.own_cameras.as_predicted()[None].float())
            return self._conditions(batch, W.ordinary_window_conditions(batch, dev), text, sampler)[
                1
            ]

        def with_visible(merged, start: int, labels: torch.Tensor):
            visible = merged["peer_visible"].clone()
            visible[:, start : start + labels.shape[0]] = labels.to(visible.device, visible.dtype)
            return dict(merged, peer_visible=visible)

        def retest(target, s: int):
            return lambda x0: with_visible(
                target, s, vis.relabel_prefix(s, block_depth_rows(depth.depth_grid, x0[0]))
            )

        sampler.cache.reset()
        merged = conditions()
        sampler.commit(
            sink, with_visible(merged, 0, vis.prefix_labels(0, 1, out, "drawn")), start=0
        )
        for s in range(1, n + 1, BLOCK):
            t = orig(s)
            closed.publish_own(s, t, out[0, :s].float().cpu())
            closed.exchange.wait(
                closed.peers, t, max_wait_s=self.cfg.pool.wait_s, is_done=pool.is_done
            )
            closed.read_peers(s, t)
            merged = conditions()
            target = with_visible(merged, s, vis.prefix_labels(s, BLOCK, out, "target"))
            x0 = self._denoise_retested(
                sampler, noise[:, s - 1 : s - 1 + BLOCK], target, start=s, retest=retest(target, s)
            )
            out[:, s : s + BLOCK] = x0
            sampler.commit(
                x0, with_visible(merged, s, vis.prefix_labels(s, BLOCK, out, "drawn")), start=s
            )
        return out

    @staticmethod
    def _denoise_retested(
        sampler: Sampler, noisy: torch.Tensor, conditions, *, start: int, retest
    ) -> torch.Tensor:
        """:meth:`Sampler.denoise` whose rungs after the first run under ``retest(x0 of rung 1)``: the visibility
        re-tested on the block's own x0 estimate (Sec. 3.2)."""
        current = {"conditions": conditions, "retested": False}

        def call(x, t):
            flow, x0 = sampler.call(
                x, t, current["conditions"], start=start, num_frames=int(noisy.shape[1])
            )
            if not current["retested"]:
                current.update(conditions=retest(x0), retested=True)
            return flow, x0

        return ladder_denoise(call, noisy, sampler.ladder, sampler.scheduler, rng=sampler.rng)

    def _rollout(self, window: ClientWindow, generator, embeds, depth, pool: LocalDirPool):
        cfg, dev = self.cfg, self.device
        row, spec, me = window.row, window.spec, window.row.media_id
        n = spec.latent_frames
        my_slot = window.observer.player_slot
        stride = spec.skip_frame * BLOCK
        recent = cfg.window.recent
        latent_shape = (cfg.model.latent_channels, cfg.model.latent_height, cfg.model.latent_width)

        def orig(f: int) -> int:  # source frame (32 fps) of latent f
            return row.start_frame + stride * int(f)

        # ---- recorded inputs: own cameras (oracle) and fields of view, the round batch, its continuous columns
        own = window.player_frames[my_slot]
        own_c2w = c2w_from_state_rows(
            np.asarray(own.states)[BLOCK * np.arange(n)], eye_height=cfg.data.eye_height
        )
        tan_h, tan_v = half_angle_tangents(cfg.data.hfov_degrees)
        own_obs = window.item.obs.as_dict()
        own_tans = torch.as_tensor(
            scoped_query_tans(
                own_obs,
                own.weapon_ids,
                range(n),
                tan_h=tan_h,
                tan_v=tan_v,
                weapon_names=OPENCS2_WEAPONS,
            )[0],
            dtype=torch.float32,
        )
        round_batch = {
            k: v.unsqueeze(0) if torch.is_tensor(v) else [v]
            for k, v in window.item.batch_dict().items()
        }
        round_batch.update(wp_own_c2w=own_c2w[None].float(), wp_own_tans=own_tans[None])
        peers = row.lockstep_peers()
        recorded = round_batch["player_states"]
        closed = (
            self._closed_loop(window, round_batch, depth, peers)
            if cfg.player_state.source == "predicted"
            else None
        )

        def continuous() -> dict[str, torch.Tensor]:
            return {
                CONTINUOUS_ROWS_KEY: continuous_row_state(
                    round_batch, camera_delta_scale=cfg.model.player_field.camera_delta_scale
                )
            }

        row_context = continuous()

        # ---- scene state, peer blocks, slot material
        scene = SceneState(
            client=me,
            tans=[[tan_h, tan_v]],
            depth_fn=depth.depth_grid,
            peer_latents=pool.peer_latents,
            bound=cfg.memory.bound,
        )

        def candidate_at(slot, media_id, ws, f0):
            block, reason = published_block_candidate(
                slot=slot,
                media=window.round_slots[slot],
                table=window.tick_tables[slot],
                window_start=ws,
                f0=f0,
                spec=spec,
                eye_height=cfg.data.eye_height,
            )
            if closed is not None:  # keyed where its publisher drew it
                block = closed.rekey(block, recorded, start_frame=row.start_frame, stride=stride)
            return block, reason

        blocks = PeerBlocks(
            ego_media=me,
            ego_slot=my_slot,
            sources={
                m.media_id: s for s, m in window.round_slots.items() if s != my_slot
            },  # pool scope: all
            candidate_at=candidate_at,
        )
        slot_material = MemorySlotFrames(
            own_media=me,
            own_frames=own,
            own_obs=window.item.obs,
            round_slots=window.round_slots,
            tick_tables=window.tick_tables,
            spec=spec,
            obs_signal_label_root=cfg.paths.obs_signal_label_root,
            hfov_degrees=cfg.data.hfov_degrees,
        )

        # ---- generator, sampler, text; entry noise and sink
        sampler = self._sampler(generator)
        text = FixedPromptEncoder(cfg.model.fixed_prompt, embeds)
        noise = entry_noise(
            cfg.run.seed, cfg.run.latents, n, latent_shape, device=dev, dtype=self.dtype
        )
        sink = window.first_latent[None].to(dev, self.dtype)  # [1, 1, C, H, W]
        store = torch.zeros((n,) + latent_shape, dtype=torch.float32)  # clean latents, fp32 (CPU)
        store[0] = sink[0, 0].float().cpu()
        output = torch.zeros((1, n) + latent_shape, dtype=self.dtype, device=dev)
        output[:, :1] = sink

        def own_block(s: int, cameras: torch.Tensor) -> dict:
            return dict(
                media_id=me,
                slot=my_slot,
                window_start=row.start_frame,
                f0=s,
                orig_first=orig(s),
                orig_last=orig(s + BLOCK - 1),
                c2w=cameras[s : s + BLOCK],
            )

        def publish(s: int, mode: str, cameras: torch.Tensor) -> None:
            pool.publish_block(
                window_start=row.start_frame,
                f0=s,
                latents=store[s : s + BLOCK],
                orig_first=orig(s),
                orig_last=orig(s + BLOCK - 1),
                extra={"mode": mode},
            )
            blocks.add_own(own_block(s, cameras))

        # ---- plain prefix: latents 1 .. 24 on the absolute positions, no memory
        set_seed(cfg.run.seed)
        n_plain = cfg.window.plain_prefix_latents
        prefix_batch = dict(round_batch, latents=store[None].clone())
        if closed is None:
            _, merged = self._conditions(
                prefix_batch, W.ordinary_window_conditions(prefix_batch, dev), text, sampler
            )
            prefix = sampler.rollout_prefix(noise[:, :n_plain], sink, merged)
            prefix_c2w = own_c2w
        else:
            prefix = self._closed_prefix(
                closed, sampler, text, depth, prefix_batch, noise[:, :n_plain], sink, pool, orig
            )
            prefix_c2w = closed.own_cameras.as_predicted()
        output[:, : n_plain + 1] = prefix
        store[: n_plain + 1] = prefix[0].float().cpu()
        for s in range(1, n_plain + 1, BLOCK):
            publish(s, "plain", prefix_c2w)

        # ---- reconstituted blocks
        rows: list[dict[str, Any]] = []
        for s in range(n_plain + 1, n - BLOCK + 1, BLOCK):
            t = orig(s)
            scene.ingest_own(blocks.blocks, t_target=t, own_latents=store)  # 1 own write
            withdrawn, resident = scene.drain_own_step()
            if closed is not None:
                closed.publish_own(s, t, store)  # own position
            pool.publish_step(t_target=t, withdrawn=withdrawn, resident=resident)  # 2 step record
            n_admitted = blocks.admit(
                pool, t_target=t, peers=peers, max_wait_s=cfg.pool.wait_s
            )  # 3 wait, 4 admit
            cameras = own_c2w
            if closed is not None:  # peers' positions
                cameras = closed.own_cameras.for_block(s)
                closed.read_peers(s, t)
                row_context = continuous()
            scene.ingest_peers(blocks.blocks, t_target=t)
            scene.follow_withdrawals(
                pool, peers=blocks.peers, t_target=t, skipped=blocks.skipped
            )  # 5 follow
            read = scene.retrieve(
                query_c2w=cameras[s : s + BLOCK],
                recent_c2w=cameras[s - recent : s],  # 6 retrieve
                recent_latents=store[s - recent : s],
                t_target=t,
            )
            entry = scene.entry_block(read)

            batch = dict(
                round_batch,
                latents=store[None].clone(),
                wp_target_start=torch.tensor([s]),  # 7 window
                wp_own_c2w=cameras[None].float(),
            )
            if entry is not None:
                if entry not in blocks.eligible(t_target=t):
                    raise RuntimeError(
                        f"block {s}: retrieved block {entry} is outside the causal cut"
                    )
                b = blocks.blocks[entry]
                f0 = int(b["f0"])
                latents = (
                    store[f0 : f0 + BLOCK]
                    if b["media_id"] == me
                    else pool.fetch_block(
                        media_id=b["media_id"], window_start=b["window_start"], f0=f0, t_target=t
                    )
                )
                material = slot_material(b, latents)
                if closed is not None:
                    material = closed.slot_rows(
                        material, b, start_frame=row.start_frame, stride=stride
                    )
                batch.update({"wp_slot_" + k: v.unsqueeze(0) for k, v in material.items()})
            batch.update(row_context)
            if closed is not None:
                batch = closed.visibility.for_block(s, batch, store, recent)
            compacted, geom, geometry = W.prepare_window(
                batch, recent=recent, with_slot=entry is not None, device=dev
            )
            vc, merged = self._conditions(compacted, geometry, text, sampler)
            ctx = geom.num_frames - BLOCK
            noisy = noise[:, s - 1 : s - 1 + BLOCK]
            block_noise = paired_cache_noise(
                cfg.run.seed, s, len(W.context_block_ranges(geom.num_frames)), recent // BLOCK
            )
            if closed is None:
                x0 = sampler.generate_block(
                    vc.clean_latent[:, :ctx],
                    noisy,
                    merged,  # 8 prefill, 9 ladder
                    block_noise=block_noise,
                )
            else:
                sampler.cache.reset()
                sampler.prefill_context(
                    vc.clean_latent[:, :ctx],
                    merged,
                    ranges=W.context_block_ranges(geom.num_frames),
                    block_noise=block_noise,
                )

                def retest(x0, _s=s, _compacted=compacted, _geometry=geometry):
                    relabelled = closed.visibility.relabel_target(
                        _compacted, _s, block_depth_rows(depth.depth_grid, x0[0])
                    )
                    return self._conditions(relabelled, _geometry, text, sampler)[1]

                x0 = self._denoise_retested(sampler, noisy, merged, start=ctx, retest=retest)
            output[:, s : s + BLOCK] = x0
            store[s : s + BLOCK] = x0[0].float().cpu()
            publish(s, "reconstituted", cameras)  # 10 publish
            rows.append(
                dict(
                    s=s,
                    t_target=t,
                    admitted=n_admitted,
                    window=int(geom.num_frames),
                    entry=(
                        None if read.entry is None else [read.entry.client, int(read.entry.t_first)]
                    ),
                    score=int(read.score),
                    holes=int(read.n_hole),
                    candidates=int(read.n_candidates),
                )
            )
        return output, rows, asdict(scene.follow_stats)


def run_client(config: InferenceConfig, **kwargs) -> ClientResult:
    """Run one client with ``config`` (see :class:`Client` for the keyword arguments)."""
    return Client(config, **kwargs).run()
