"""One client's rollout of a round, block by block (docs/inference.md, "The block loop").

The first six blocks (latent frames 1-24) are generated on the recent context alone, at their
positions in the round. From the 25th latent frame on (App. E, "Retrieval") a block reads the scene
state and is generated from its window: first frame | memory frames | recent context | target
frames. Such a block splits at its controls::

    prepare   the exchange with the shared world state (the own memory entry, the step record, the
              lockstep wait, admit, follow, retrieve), the window, its conditions and the KV
              prefill; needs no controls of the block, so it runs as soon as the previous block
              is done
    denoise   the target frames in 4 denoising steps; needs the block's 16 rows of controls
    commit    the block into the latent store, published as a memory entry

All of it runs on one thread, in block order: the denoising steps draw from the device's global
RNG.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from worldcast.config.inference import InferenceConfig
from worldcast.data.camera import half_angle_tangents
from worldcast.data.latent_cache import load_first_latent
from worldcast.data.latents import (
    BLOCK,
    FIRST_TARGET,
    LATENT_SHAPE,
    RECENT,
    VIDEO_FRAMES_PER_BLOCK,
    source_frame,
    video_frames_of,
)
from worldcast.data.memory_frames import (
    MEMORY_FRAMES_PREFIX,
    MemoryBlock,
    MemoryFrames,
    batch_keys,
    recorded_block,
)
from worldcast.data.window import ClientWindow, collate_windows
from worldcast.engine.generator import (
    WindowInputs,
    client_cameras,
    window_conditions,
    window_inputs,
)
from worldcast.modeling.ray_embedding import RayConditions
from worldcast.player_state import (
    block_depth_frames,
    continuous_row_state,
    memory_continuous_columns,
)
from worldcast.sampling.sampler import Sampler
from worldcast.sampling.schedulers import entry_noise, paired_context_noise
from worldcast.sampling.window import (
    MEMORY_CONTINUOUS_COLUMNS_KEY,
    ROUND_CONTINUOUS_COLUMNS_KEY,
    WindowLayout,
    gather_window,
)
from worldcast.scene_state import RetrieveResult, SceneState
from worldcast.utils.seed import set_seed

from .controls import BlockControls
from .loading import ClientModels, load_closed_loop
from .memory_blocks import MemoryBlocks
from .world_state import StepRecord, WorldState

__all__ = ["BlockRead", "PreparedBlock", "Rollout"]


@dataclass
class BlockRead:
    """What a block read from the shared world state.

    Attributes:
        t_target (int): the source frame of the block's first latent frame.
        admitted (int): the other clients' blocks admitted before it.
        window_frames (int): latent frames of its window, 21, or 17 when nothing was retrieved.
        entry (tuple[str, int] | None): the retrieved memory entry: its owner and ``t_first``.
        coverage (int): missing pixels the entry covers.
        missing (int): pixels of the block no recent latent frame reaches.
        candidates (int): memory entries completed before the block.
    """

    t_target: int
    admitted: int
    window_frames: int
    entry: tuple[str, int] | None
    coverage: int
    missing: int
    candidates: int


@dataclass
class PreparedBlock:
    """A block that read the scene state, ready for its controls.

    Attributes:
        conditions (dict[str, Any]): the generator's conditions of its window; the KV cache holds
            the window's context.
        context (int): latent frames of the window before the target frames.
        cameras (Tensor): ``[N, 4, 4]`` the client's cameras as known when the block started.
        batch (dict[str, Any]): the window's batch.
        rays (dict[str, Tensor]): the ray embedding's conditions of the window.
        read (BlockRead): what the block read.
    """

    conditions: dict[str, Any]
    context: int
    cameras: torch.Tensor
    batch: dict[str, Any]
    rays: dict[str, torch.Tensor]
    read: BlockRead


class Rollout:
    """One client's rollout of a round.

    :meth:`open` once; then per block, in block order, :meth:`denoise`, :meth:`commit` and
    :meth:`prepare_next` (which prepares the block after it). A block is named by ``f0``, its
    first latent frame.

    Args:
        cfg (InferenceConfig): the client's config.
        models (ClientModels): the client's weights.
        sampler (Sampler): the block-causal sampler on the client's generator and KV cache.
        window (ClientWindow): the client's window of the recorded round.
        world_state (WorldState): the shared world state.
        lockstep (bool): wait for the other clients before each block.
        dtype (torch.dtype): the generator's dtype.
    """

    def __init__(
        self,
        cfg: InferenceConfig,
        models: ClientModels,
        sampler: Sampler,
        window: ClientWindow,
        world_state: WorldState,
        *,
        lockstep: bool,
        dtype: torch.dtype,
    ) -> None:
        self.cfg, self.models, self.sampler = cfg, models, sampler
        self.device, self.dtype = torch.device(cfg.run.device), dtype
        self.recording, self.world_state = window, world_state
        self.media_id, self.start_frame = window.row.media_id, window.row.start_frame
        self.slot, self.latent_frames = window.media.player_slot, window.spec.latent_frames
        self.others = window.row.other_clients() if lockstep else ()

        # the recorded round: the client's cameras, every player's states and controls
        client, signals = window.client_frames, window.item.signals
        batch = collate_windows([window.item.batch_dict()])
        batch.update(client_cameras(client.states, signals, client.weapon_ids, self.latent_frames))
        self.client_c2w = batch["window_c2w"][0]
        self.round_batch, self.recorded_states = batch, batch["player_states"]

        # the shared world state: the closed loop, the scene state, the known blocks
        self.closed = None
        if cfg.player_state.source == "predicted":
            self.closed = load_closed_loop(cfg, window, batch, models, world_state)
        self.continuous_rows = continuous_row_state(batch)
        self.scene = SceneState(
            client=self.media_id,
            tans=half_angle_tangents(),
            depth_fn=models.depth.log_depth,
            block_latents=self._block_latents,
            bound=cfg.scene_state.bound,
        )
        self.known = MemoryBlocks(
            client=self.media_id,
            owners={m.media_id: s for s, m in window.round_slots.items() if s != self.slot},
            block_at=self._block_at,
        )
        self.memory_frames = MemoryFrames(
            client_media=self.media_id,
            client_frames=client,
            client_signals=signals,
            round_slots=window.round_slots,
            tick_tables=window.tick_tables,
            spec=window.spec,
            observer_signal_label_root=cfg.paths.observer_signal_label_root,
        )

        # the entry noise and the first frame; the store keeps the clean latents the scene state
        # and the windows read
        shape = LATENT_SHAPE
        self.noise = entry_noise(
            cfg.run.seed,
            cfg.run.latent_frames,
            self.latent_frames,
            shape,
            device=self.device,
            dtype=dtype,
        )
        first_frame = load_first_latent(
            cfg.paths.latent_cache_root, self.media_id, self.start_frame
        )
        first_frame = first_frame[None].to(self.device, dtype)
        self.store = torch.zeros((self.latent_frames, *shape), dtype=torch.float32)
        self.store[0] = first_frame[0, 0].float().cpu()
        self.output = torch.zeros((1, self.latent_frames, *shape), dtype=dtype, device=self.device)
        self.output[:, :1] = first_frame
        self.round_conditions: dict[str, Any] = {}
        self.prepared: PreparedBlock | None = None

    def source_frame(self, latent: int) -> int:
        """The source frame of latent frame ``latent`` of the rollout."""
        return source_frame(self.start_frame, latent)

    # ------------------------------------------------------------------------------- the blocks
    def open(self) -> None:
        """Seed, then write the first frame (latent frame 0) into the KV cache."""
        set_seed(self.cfg.run.seed)
        self.round_conditions = self._round_conditions()
        self.sampler.cache.reset()
        self.sampler.write_context(self.output[:, :1], self._drawn(0, 1), frame_offset=0)

    def denoise(self, f0: int, controls: Callable[[], BlockControls | None]) -> torch.Tensor:
        """Generate block ``f0`` (latent frames ``f0 .. f0 + 3``) into :attr:`output`.

        Args:
            f0 (int): the block's first latent frame.
            controls (Callable): called right before the denoising for the block's controls;
                ``None`` keeps the recorded ones.

        Returns:
            Tensor: the block's x0 ``[1, 4, 48, 24, 42]`` float32.

        Raises:
            RuntimeError: the rollout is not open, or the block reads the scene state and was not
                prepared (:meth:`prepare_next` of the block before it).
        """
        if not self.round_conditions:
            raise RuntimeError("the rollout is not open: call open() first")
        reads_scene = f0 >= FIRST_TARGET
        if reads_scene and self.prepared is None:
            raise RuntimeError(f"block {f0} was not prepared: call prepare_next({f0 - BLOCK})")
        if not reads_scene and self.closed is not None:
            self._exchange_positions(f0)
        taken = controls()
        if taken is not None:
            self._take_controls(f0, taken)
        if reads_scene:
            self.prepared = self._at_cut(self.prepared, taken)
            x0 = self._denoise_on_window(f0, self.prepared)
        else:
            x0 = self._denoise_on_round(f0)
        self.output[:, f0 : f0 + BLOCK] = x0
        return x0

    def commit(self, f0: int, x0: torch.Tensor) -> BlockRead | None:
        """Keep block ``f0`` in the latent store and publish it; returns what it read from the
        scene state (``None`` for the first six blocks)."""
        prepared, self.prepared = self.prepared, None
        # as in the paper's runs, the store keeps the first six blocks bf16-rounded and later
        # blocks' x0 in fp32
        if prepared is None:
            self.sampler.write_context(x0, self._drawn(f0, BLOCK), frame_offset=f0)
            self.store[f0 : f0 + BLOCK] = self.output[0, f0 : f0 + BLOCK].float().cpu()
        else:
            self.store[f0 : f0 + BLOCK] = x0[0].float().cpu()
        self._publish(f0, prepared)
        return None if prepared is None else prepared.read

    def prepare_next(self, f0: int) -> None:
        """Prepare the block after block ``f0`` if it reads the scene state: before its
        controls."""
        following = f0 + BLOCK
        if FIRST_TARGET <= following <= self.latent_frames - BLOCK:
            self.prepared = self._prepare(following)

    # --------------------------------------------------------------------------- the conditions
    def _inputs(self, batch: Mapping[str, Any], rays: Mapping[str, torch.Tensor]) -> WindowInputs:
        return window_inputs(
            batch,
            prompt_embeds=self.models.prompt_embeds,
            device=self.device,
            dtype=self.dtype,
            rays=rays,
        )

    def _conditions(
        self, batch: Mapping[str, Any], rays: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        return window_conditions(
            batch,
            prompt_embeds=self.models.prompt_embeds,
            device=self.device,
            dtype=self.dtype,
            rays=rays,
        )

    def _round_conditions(self) -> dict[str, Any]:
        """The whole round's conditions, for the first six blocks (closed loop: the client's
        cameras as predicted so far)."""
        batch = dict(self.round_batch)
        if self.closed is not None:
            batch["window_c2w"] = self.closed.cameras.as_predicted()[None].float()
        rays = RayConditions.contiguous(batch["window_c2w"], batch["window_tans"])
        return self._conditions(batch, rays.conditions(self.device))

    def _drawn(self, f0: int, n: int) -> dict[str, Any]:
        """The round's conditions of a context write (closed loop: with the visibility of the
        generated latent frames)."""
        if self.closed is None:
            return self.round_conditions
        labels = self.closed.visibility.generated_visible(f0, n, self.output[0])
        return _with_visible(self.round_conditions, f0, labels)

    def _take_controls(self, f0: int, controls: BlockControls) -> None:
        """Write the block's controls into the round's history, which later windows read."""
        controls.check_block()
        first = video_frames_of(f0, BLOCK)[0]
        for key, rows in controls.conditions().items():
            history = self.round_batch[key]
            history[0, first : first + VIDEO_FRAMES_PER_BLOCK] = torch.as_tensor(
                rows, dtype=history.dtype
            )
        if f0 < FIRST_TARGET:
            self.round_conditions = _with_controls(self.round_conditions, controls, first_row=first)

    # -------------------------------------------------------------------- the first six blocks
    def _exchange_positions(self, f0: int) -> None:
        """Closed loop, the first six blocks: publish the client's position, wait for the other
        clients' (with lockstep), read them and rebuild the round's conditions."""
        closed, t = self.closed, self.source_frame(f0)
        closed.publish_position(f0, self.output[0, :f0].float().cpu())
        self.world_state.wait_for_positions(
            closed.others if self.others else (),
            t_target=t,
            max_wait_s=self.cfg.world_state.wait_s,
        )
        closed.read_positions(f0)
        self.round_conditions = self._round_conditions()

    def _denoise_on_round(self, f0: int) -> torch.Tensor:
        """One of the first six blocks: at its position in the round, on the round's conditions,
        without memory frames."""
        noisy = self._entry_noise(f0)
        if self.closed is None:
            return self.sampler.denoise(noisy, self.round_conditions, frame_offset=f0)
        visibility, depth = self.closed.visibility, self.models.depth.log_depth
        visible = visibility.target_visible(f0, self.output[0])
        target = _with_visible(self.round_conditions, f0, visible)

        def retest(x0: torch.Tensor) -> dict[str, Any]:
            labels = visibility.retested_visible(f0, block_depth_frames(depth, x0[0]))
            return _with_visible(target, f0, labels)

        return self.sampler.denoise(noisy, target, frame_offset=f0, retest=retest)

    # ------------------------------------------------------- the blocks that read the scene state
    def _block_at(self, slot: int, window_start: int, f0: int) -> MemoryBlock | None:
        """Another player's block at a published key, with its recorded cameras (closed loop:
        where its owner generated it)."""
        recording = self.recording
        block = recorded_block(
            slot,
            recording.round_slots[slot],
            recording.tick_tables[slot],
            window_start,
            f0,
            recording.spec,
        )
        if block is not None and self.closed is not None:
            block = self.closed.as_generated(block, self.recorded_states)
        return block

    def _block_latents(self, block: MemoryBlock, t: int) -> np.ndarray:
        """The latents of another client's admitted block, as the scene state copies them."""
        return self.world_state.block_latents(block, t)

    def _entry_noise(self, f0: int) -> torch.Tensor:
        """The entry noise of block ``f0``, ``[1, 4, 48, 24, 42]``: its frames of the draw for the
        requested length."""
        return self.noise[:, f0 - 1 : f0 - 1 + BLOCK]

    def _exchange(self, f0: int, t: int) -> tuple[torch.Tensor, int, RetrieveResult]:
        """The exchange with the shared world state before block ``f0`` at source frame ``t``
        (App. B): the client's cameras, the number of blocks admitted and
        what retrieval returned."""
        world_state, scene, known, closed = self.world_state, self.scene, self.known, self.closed
        scene.publish_own(known.blocks, t_target=t, own_latents=self.store)
        withdrawn, resident = scene.drain_withdrawals()
        if closed is not None:
            closed.publish_position(f0, self.store)
        world_state.publish_step(t_target=t, withdrawn=withdrawn, resident=resident)
        admitted = known.admit(
            world_state, t_target=t, others=self.others, max_wait_s=self.cfg.world_state.wait_s
        )
        cameras = self.client_c2w
        if closed is not None:
            cameras = closed.cameras.for_block(f0)
            closed.read_positions(f0)
            # the round's continuous columns, read again from the table the closed loop wrote
            self.continuous_rows = continuous_row_state(self.round_batch)
        scene.publish_others(known.blocks, t_target=t)
        scene.follow_withdrawals(
            world_state if self.others else _EarlierSteps(world_state, t),
            others=known.owners,
            t_target=t,
            skipped=known.skipped,
        )
        retrieved = scene.retrieve(
            next_c2w=cameras[f0 : f0 + BLOCK],
            recent_c2w=cameras[f0 - RECENT : f0],
            recent_latents=self.store[f0 - RECENT : f0],
            t_target=t,
        )
        return cameras, admitted, retrieved

    def _prepare(self, f0: int) -> PreparedBlock:
        """Everything of block ``f0`` before its controls: the exchange with the shared world
        state, the window, its conditions and the KV prefill."""
        t = self.source_frame(f0)
        cameras, admitted, retrieved = self._exchange(f0, t)
        entry = retrieved.entry
        batch = dict(
            self.round_batch,
            latents=self.store[None].clone(),
            window_target_start=torch.tensor([f0]),
            window_c2w=cameras[None].float(),
        )
        if entry is not None:
            batch.update(self._retrieved_frames(self.known.blocks[entry.block], t))
        batch[ROUND_CONTINUOUS_COLUMNS_KEY] = self.continuous_rows
        if self.closed is not None:
            batch = self.closed.visibility.for_block(f0, batch, self.store)
        layout = WindowLayout(RECENT, with_memory=entry is not None)
        window, cameras_of_window = gather_window(batch, layout)
        rays = cameras_of_window.conditions(self.device)
        inputs = self._inputs(window, rays)
        context = layout.target_positions[0]
        ranges = layout.context_ranges
        self.sampler.cache.reset()
        self.sampler.prefill_context(
            inputs.latents[:, :context],
            inputs.conditions,
            ranges=ranges,
            context_write_noise=paired_context_noise(
                self.cfg.run.seed, f0, len(ranges), RECENT // BLOCK
            ),
        )
        read = BlockRead(
            t_target=t,
            admitted=admitted,
            window_frames=layout.num_frames,
            entry=None if entry is None else (entry.owner, int(entry.t_first)),
            coverage=int(retrieved.coverage),
            missing=int(retrieved.n_missing),
            candidates=int(retrieved.n_candidates),
        )
        return PreparedBlock(inputs.conditions, context, cameras, window, rays, read)

    def _memory_frames(self, block: MemoryBlock, t: int) -> dict[str, torch.Tensor]:
        """The retrieved ``block``'s latents, cameras and inputs (:class:`MemoryFrames`)."""
        if block.media_id == self.media_id:
            latents = self.store[block.f0 : block.f0 + BLOCK]
        else:
            latents = torch.from_numpy(self.world_state.block_latents(block, t))
        frames = self.memory_frames(block, latents)
        if self.closed is not None:
            frames = self.closed.inputs_as_generated(frames, block)
        return frames

    def _retrieved_frames(self, block: MemoryBlock, t: int) -> dict[str, torch.Tensor]:
        """The memory frames: the retrieved ``block`` at window positions 1-4, as the window's
        gather reads them."""
        frames = self._memory_frames(block, t)
        inputs = {key: value.unsqueeze(0) for key, value in batch_keys(frames).items()}
        inputs[MEMORY_CONTINUOUS_COLUMNS_KEY] = memory_continuous_columns(
            self.continuous_rows,
            inputs[MEMORY_FRAMES_PREFIX + "states"],
            self.round_batch["client_slot"],
        )
        return inputs

    def _at_cut(self, prepared: PreparedBlock, controls: BlockControls | None) -> PreparedBlock:
        """The prepared block at its controls' cut: the block's ``controls`` (``None``: the recorded
        ones, already in the window) written into the rows of its target frames, in the window's
        batch and in its conditions."""
        if controls is None:
            return prepared
        rows = -VIDEO_FRAMES_PER_BLOCK
        return replace(
            prepared,
            batch=_with_controls(prepared.batch, controls, first_row=rows),
            conditions=_with_controls(prepared.conditions, controls, first_row=rows),
        )

    def _denoise_on_window(self, f0: int, prepared: PreparedBlock) -> torch.Tensor:
        """The 4 denoising steps of a prepared window at its cut (:meth:`_at_cut`)."""

        def retest(x0: torch.Tensor) -> Mapping[str, Any]:
            relabelled = self.closed.visibility.relabel_target(
                prepared.batch, f0, block_depth_frames(self.models.depth.log_depth, x0[0])
            )
            return self._conditions(relabelled, prepared.rays)

        return self.sampler.denoise(
            self._entry_noise(f0),
            prepared.conditions,
            frame_offset=prepared.context,
            retest=None if self.closed is None else retest,
        )

    # ------------------------------------------------------------------------------- publishing
    def _publish(self, f0: int, prepared: PreparedBlock | None) -> None:
        """Publish the finished block as a memory entry.

        In the closed loop the first six blocks are published after the last of them, with the
        cameras as predicted then.
        """
        if prepared is not None:
            self._publish_block(f0, prepared.cameras)
        elif self.closed is None:
            self._publish_block(f0, self.client_c2w)
        elif f0 + BLOCK >= FIRST_TARGET:
            cameras = self.closed.cameras.as_predicted()
            for first in range(1, FIRST_TARGET, BLOCK):
                self._publish_block(first, cameras)

    def _publish_block(self, f0: int, cameras: torch.Tensor) -> None:
        self.world_state.publish_block(
            window_start=self.start_frame, f0=f0, latents=self.store[f0 : f0 + BLOCK]
        )
        block = MemoryBlock.at(
            self.media_id, self.slot, self.start_frame, f0, cameras[f0 : f0 + BLOCK]
        )
        self.known.add_own(block)


class _EarlierSteps:
    """The other clients' step records before ``t_target`` only. Without lockstep a record for
    this very block can arrive between the admission and the follow, listing a block not admitted
    yet; its withdrawals are followed at the next block."""

    def __init__(self, world_state: WorldState, t_target: int) -> None:
        self.world_state, self.t_target = world_state, int(t_target)

    def read_steps(self, media_id: str, *, upto: int) -> list[StepRecord]:
        return self.world_state.read_steps(media_id, upto=min(int(upto), self.t_target - 1))


def _with_visible(
    conditions: Mapping[str, Any], start: int, labels: torch.Tensor
) -> dict[str, Any]:
    """``conditions`` with ``player_visible`` from latent frame ``start`` on replaced by ``labels``
    ``[n, P]``."""
    visible = conditions["player_visible"].clone()
    visible[:, start : start + labels.shape[0]] = labels.to(visible.device, visible.dtype)
    return dict(conditions, player_visible=visible)


def _with_controls(
    conditions: Mapping[str, Any], controls: BlockControls, *, first_row: int
) -> dict[str, Any]:
    """``conditions`` with a block's controls from row ``first_row`` on (negative: from the
    end)."""
    out = dict(conditions)
    for key, rows in controls.conditions().items():
        history = conditions[key].clone()
        start = first_row if first_row >= 0 else int(history.shape[1]) + first_row
        history[0, start : start + VIDEO_FRAMES_PER_BLOCK] = torch.as_tensor(
            rows, device=history.device, dtype=history.dtype
        )
        out[key] = history
    return out
