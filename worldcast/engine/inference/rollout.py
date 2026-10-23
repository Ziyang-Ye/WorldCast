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

The generator trained without scene state (``model.scene_state: false``; not the paper's client)
runs the same loop without the scene state: a block exchanges only the player states, its window is
the ordinary window of the generator's training (first frame | the latent frames before the block,
at most 36 | target frames, :meth:`~worldcast.sampling.window.WindowLayout.ordinary`), its context
is written clean, and nothing is published as a memory entry.
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
    last_video_frame,
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
from worldcast.player_state.states import CONT_PITCH_INDEX, CONT_YAW_INDEX, STATE_BASE_DIM
from worldcast.sampling.sampler import CONTEXT_NOISE, Sampler
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
from .field_rows import FieldRows
from .loading import ClientModels, load_closed_loop
from .memory_blocks import MemoryBlocks
from .replicas import GeneratorReplica
from .world_state import StepRecord, WorldState

__all__ = ["BlockRead", "PreparedBlock", "Rollout", "context_noise"]


def context_noise(cfg: InferenceConfig) -> int:
    """The context noise of the client's generator (``model.scene_state``): :data:`CONTEXT_NOISE`
    with scene state; without, 0, a clean context, as that stage was trained (a clean context copy
    under teacher forcing, a clean cache in the distillation's rollout)."""
    return CONTEXT_NOISE if cfg.model.stage.scene_state else 0


@dataclass
class BlockRead:
    """What a block read from the shared world state.

    Attributes:
        t_target (int): the source frame of the block's first latent frame.
        admitted (int): the other clients' blocks admitted before it.
        window_frames (int): latent frames of its window, 21, or 17 when nothing was retrieved
            (without scene state: 29 to 41, :meth:`WindowLayout.ordinary`).
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
        rays (dict[str, Tensor]): the ray embedding's conditions of the window (none without
            scene state).
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

    Raises:
        ValueError: the sampler writes the context at another noise than the generator's
            (:func:`context_noise`).
    """

    #: Whether the rollout has no last block (live play); the rest end with their round.
    open_ended = False
    #: The client's generator on other cards (``ServingOptions.refresh_devices``), on which every
    #: block's context is prefilled too (:class:`~worldcast.engine.inference.replicas.
    #: GeneratorReplica`); the client sets them.
    replicas: tuple[GeneratorReplica, ...] = ()

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
        if sampler.context_noise != context_noise(cfg):
            raise ValueError(
                f"the sampler writes the context at {sampler.context_noise}, the generator of"
                f" model.scene_state {cfg.model.scene_state} at {context_noise(cfg)}"
            )
        self.cfg, self.models, self.sampler = cfg, models, sampler
        self.scene_state = cfg.model.scene_state
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
        self.noise = (
            None
            if self.open_ended
            else entry_noise(
                cfg.run.seed,
                cfg.run.latent_frames,
                self.latent_frames,
                shape,
                device=self.device,
                dtype=dtype,
            )
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

    def denoise(
        self,
        f0: int,
        controls: Callable[[], BlockControls | None],
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Generate block ``f0`` (latent frames ``f0 .. f0 + 3``) into :attr:`output`.

        Args:
            f0 (int): the block's first latent frame.
            controls (Callable): called right before the denoising for the block's controls;
                ``None`` keeps the recorded ones.
            on_step (Callable | None): a block that reads the scene state: called with each
                denoising step's index and x0 (:meth:`~worldcast.sampling.sampler.Sampler.denoise`).

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
            self.prepared = self._at_cut(f0, self.prepared, taken)
            x0 = self._denoise_on_window(f0, self.prepared, on_step)
        else:
            x0 = self._denoise_on_round(f0)
        self.output[:, f0 : f0 + BLOCK] = x0
        return x0

    def commit(self, f0: int, x0: torch.Tensor) -> BlockRead | None:
        """Keep block ``f0`` in the latent store and publish it; returns what it read from the
        scene state, or its window without (``None`` for the first six blocks)."""
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

    # ------------------------------------------------------------------ the rows the field read
    def field_rows(self, latent: int) -> FieldRows:
        """Every player's row as the player state field read it at latent frame ``latent``.

        The rows are the round's player-state table at the latent frame's last video frame, with
        the yaw and pitch of its continuous columns (latent frames 1-24 read a 6-wide table, which
        integrates the same turns without the yaw wrap and the pitch clamp). A dead player's row is
        the table's, with alive 0; the field draws its corpse at the x, y, z of its last alive row,
        which the rows do not carry. Once a block is committed its rows are never rewritten, so
        another thread may read them while the rollout runs, without a lock; before, they are the
        rows known so far (under ``predicted``, the positions read for the block rewrite them).

        Args:
            latent (int): a latent frame of the rollout.

        Returns:
            FieldRows: the rows; the client's slot is :attr:`FieldRows.OWN`, a slot without media
            in the round :attr:`FieldRows.HIDDEN`, every other :attr:`FieldRows.RECORDED`;
            ``drawn`` is ``None``.

        Raises:
            ValueError: ``latent`` is not a latent frame of the rollout.
        """
        latent = int(latent)
        if not 0 <= latent < self.latent_frames:
            raise ValueError(
                f"latent frame {latent} of a rollout of {self.latent_frames} latent frames"
            )
        table = self.round_batch["player_states"][0, :, last_video_frame(latent)]
        columns = self.continuous_rows[0, latent].numpy()
        states = table.numpy().astype(np.float64)
        states[:, 3] = columns[:, CONT_YAW_INDEX - STATE_BASE_DIM]
        states[:, 4] = columns[:, CONT_PITCH_INDEX - STATE_BASE_DIM]
        slots = self.recording.round_slots
        source = np.array(
            [FieldRows.RECORDED if s in slots else FieldRows.HIDDEN for s in range(len(states))],
            dtype=np.int8,
        )
        source[self.slot] = FieldRows.OWN
        return FieldRows(states, source, None, self.cfg.player_state.source)

    def camera_pitch(self, frame: int) -> float:
        """The pitch of the client's own camera at video frame ``frame``, degrees (> 0 looks down):
        the own row of the round's player-state table at that frame, where :meth:`field_rows` gives
        a latent frame's rows at its last video frame only. Another thread may read it as it reads
        :meth:`field_rows`: a committed block's rows are never rewritten.

        Raises:
            ValueError: ``frame`` is not a video frame of the rollout.
        """
        frame, states = int(frame), self.round_batch["player_states"]
        frames = int(states.shape[2])
        if not 0 <= frame < frames:
            raise ValueError(f"video frame {frame} of a rollout of {frames} video frames")
        return float(states[0, self.slot, frame, 4])

    # --------------------------------------------------------------------------- the conditions
    def _inputs(self, batch: Mapping[str, Any], rays: Mapping[str, torch.Tensor]) -> WindowInputs:
        """The window's latents and conditions; without scene state no rays and no observer
        signals, as that generator was trained."""
        return window_inputs(
            batch,
            prompt_embeds=self.models.prompt_embeds,
            device=self.device,
            dtype=self.dtype,
            observer_signals=self.scene_state,
            rays=rays if self.scene_state else None,
        )

    def _conditions(
        self, batch: Mapping[str, Any], rays: Mapping[str, torch.Tensor] | None
    ) -> dict[str, Any]:
        return window_conditions(
            batch,
            prompt_embeds=self.models.prompt_embeds,
            device=self.device,
            dtype=self.dtype,
            rays=rays if self.scene_state else None,
            observer_signals=self.scene_state,
        )

    def _round_conditions(self) -> dict[str, Any]:
        """The whole round's conditions, for the first six blocks (closed loop: the client's
        cameras as predicted so far; under live controls ``LiveRollout._round_target`` writes a
        block's latent frames of them again by this rule)."""
        batch = dict(self.round_batch)
        if not self.scene_state:
            return self._conditions(batch, None)
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

    def _own_latents(self) -> Any:
        """The client's own latents, ``[N, 48, 24, 42]``: what the scene state reads an own block's
        four latent frames from, ``latents[f0 : f0 + 4]``."""
        return self.store

    def _refresh_continuous_rows(self, f0: int) -> None:
        """The round's continuous columns, read again from the table the closed loop wrote for
        block ``f0``."""
        self.continuous_rows = continuous_row_state(self.round_batch)

    def _block_key(self, f0: int) -> int:
        """The name of block ``f0`` in the keys of its context-write noise."""
        return f0

    def _recent(self, f0: int) -> int:
        """The latent frames of recent context in the window of block ``f0``: :data:`RECENT`, or
        without scene state those of :meth:`WindowLayout.ordinary`."""
        return RECENT if self.scene_state else WindowLayout.ordinary(f0).recent

    def _entry_noise(self, f0: int) -> torch.Tensor:
        """The entry noise of block ``f0``, ``[1, 4, 48, 24, 42]``: its frames of the draw for the
        requested length."""
        return self.noise[:, f0 - 1 : f0 - 1 + BLOCK]

    def _exchange(self, f0: int, t: int) -> tuple[torch.Tensor, int, RetrieveResult]:
        """The exchange with the shared world state before block ``f0`` at source frame ``t``
        (App. B, "Deployment protocol"): the client's cameras, the number of blocks admitted and
        what retrieval returned."""
        world_state, scene, known, closed = self.world_state, self.scene, self.known, self.closed
        scene.publish_own(known.blocks, t_target=t, own_latents=self._own_latents())
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
            self._refresh_continuous_rows(f0)
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

    def _exchange_states(self, f0: int, t: int) -> torch.Tensor:
        """The exchange before block ``f0`` at source frame ``t`` of a client without scene state:
        the player states only. In the closed loop (and under live controls) the client publishes
        its position, waits for the other clients' (with lockstep) and reads them. Returns the
        client's cameras."""
        closed = self.closed
        if closed is None:
            return self.client_c2w
        closed.publish_position(f0, self.store)
        self.world_state.wait_for_positions(
            closed.others if self.others else (),
            t_target=t,
            max_wait_s=self.cfg.world_state.wait_s,
        )
        cameras = closed.cameras.for_block(f0)
        closed.read_positions(f0)
        self._refresh_continuous_rows(f0)
        return cameras

    def _prepare(self, f0: int) -> PreparedBlock:
        """Everything of block ``f0`` before its controls: the exchange with the shared world
        state, the window, its conditions and the KV prefill."""
        t = self.source_frame(f0)
        retrieved = None
        if self.scene_state:
            cameras, admitted, retrieved = self._exchange(f0, t)
            entry = retrieved.entry
            layout = WindowLayout(RECENT, with_memory=entry is not None)
        else:
            cameras, admitted, entry = self._exchange_states(f0, t), 0, None
            layout = WindowLayout.ordinary(f0)
        batch = dict(
            self.round_batch,
            latents=self.store[None],
            window_target_start=torch.tensor([f0]),
            window_c2w=cameras[None].float(),
        )
        if entry is not None:
            batch.update(self._retrieved_frames(self.known.blocks[entry.block], t))
        batch[ROUND_CONTINUOUS_COLUMNS_KEY] = self.continuous_rows
        if self.closed is not None:
            batch = self.closed.visibility.for_block(f0, batch, self.store, recent=layout.recent)
        window, cameras_of_window = gather_window(batch, layout)
        rays = cameras_of_window.conditions(self.device) if self.scene_state else {}
        inputs = self._inputs(window, rays)
        context = layout.target_positions[0]
        ranges = layout.context_ranges
        self.sampler.cache.reset()
        noise = None  # a clean context draws none
        if self.sampler.context_noise:
            noise = paired_context_noise(
                self.cfg.run.seed, self._block_key(f0), len(ranges), RECENT // BLOCK
            )
        for replica in self.replicas:  # in parallel, from the same inputs and noise
            replica.prefill(inputs.latents[:, :context], inputs.conditions, ranges, noise)
        self.sampler.prefill_context(
            inputs.latents[:, :context], inputs.conditions, ranges=ranges, context_write_noise=noise
        )
        read = BlockRead(
            t_target=t,
            admitted=admitted,
            window_frames=layout.num_frames,
            entry=None if entry is None else (entry.owner, int(entry.t_first)),
            coverage=0 if retrieved is None else int(retrieved.coverage),
            missing=0 if retrieved is None else int(retrieved.n_missing),
            candidates=0 if retrieved is None else int(retrieved.n_candidates),
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

    def _at_cut(
        self, f0: int, prepared: PreparedBlock, controls: BlockControls | None
    ) -> PreparedBlock:
        """The prepared block at its controls' cut: the block's ``controls`` (``None``: the recorded
        ones, already in the window) written into the rows of its target frames, in the window's
        batch and in its conditions. What a live block changes besides its controls is
        ``LiveRollout._at_cut``'s."""
        if controls is None:
            return prepared
        rows = -VIDEO_FRAMES_PER_BLOCK
        return replace(
            prepared,
            batch=_with_controls(prepared.batch, controls, first_row=rows),
            conditions=_with_controls(prepared.conditions, controls, first_row=rows),
        )

    def _denoise_on_window(
        self,
        f0: int,
        prepared: PreparedBlock,
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
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
            on_step=on_step,
        )

    def denoise_on(self, replica: GeneratorReplica, f0: int, rng: torch.Tensor) -> torch.Tensor:
        """Block ``f0`` of the recorded controls (no closed loop), denoised and not yet committed,
        denoised again on ``replica`` from the re-noise state ``rng`` its denoising began with: on
        the same inputs, the same x0 where the cards run the same kernels (the warm-up's check of a
        replica, ``Client.replicas_equal``).

        Returns:
            Tensor: x0 ``[1, 4, 48, 24, 42]`` float32 on the replica's card.
        """
        if self.closed is not None or self.prepared is None:
            raise RuntimeError("only a block of the recorded controls, before its commit")
        replica.prefilled()
        replica.rng.set_state(rng)
        return replica.sampler.denoise(
            self._entry_noise(f0).to(replica.device),
            replica.conditions(self.prepared.conditions),
            frame_offset=self.prepared.context,
        )

    # ------------------------------------------------------------------------------- publishing
    def _publish(self, f0: int, prepared: PreparedBlock | None) -> None:
        """Publish the finished block as a memory entry (with scene state).

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
        if not self.scene_state:
            return
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
