"""One client's rollout under live controls (``player_state.source: live``, the web demo; not the
paper's).

:class:`LiveRollout` is the block loop of :class:`~worldcast.engine.inference.rollout.Rollout` with
a :class:`~worldcast.player_state.live.LiveLoop` in the closed loop's place: the client's own
camera (the player state field's, the rays', retrieval's and its memory entries' keys) follows the
player's live controls, and the other live players are drawn where they published themselves; no
other player is in the world. A block that reads the scene state is prepared before its controls
with the last controls held; at the cut only its target frames' conditions are written again:
their rows, controls, continuous columns, observer signals, cameras and visibility labels. The
anchor of the rays stays the camera the window's context was written with.

A live session has no last block. Its round is the smallest the loader allows (the first frame, the
first six blocks and one more: :data:`LIVE_ROUND_LATENTS` latent frames), which slides by one block
after each block from the seventh on (:func:`~worldcast.sampling.window.slide_window_entries`), so
that every window block is generated at the same latent frame of the round, from the window a
longer round would give it. Everything the session counts (the blocks' names, the times of the
memory entries, the context-noise keys, the page's frames) is the session's: a round's latent
frame plus the :attr:`LiveRollout.shift` the round has slid by. A block's memory entry keeps its
latents and its ``live`` part for as long as the memory bank holds it. The entry noise is the
paper's draw (:class:`~worldcast.sampling.schedulers.EntryNoiseStream`), taken block by block.

While its player is idle a client keeps its context as ``live.idle_context`` says
(:class:`~worldcast.engine.inference.idle_context.IdleContext`): what the store and the memory
bank keep of such a block is decided when the block is published, and what is shown of it is what
the generator made of it either way.

With ``live.refresh`` a block that reads the scene state stays open after its denoising until
it is committed: :meth:`LiveRollout.refresh` cuts it again from newer controls and runs its four
denoising steps again, on the same prefilled context (the steps write only the target frames' keys
and values), the same entry noise and the same draws of the re-noise, so that each version is a
block as the generator was trained to make it; the version committed last is the block's. Its
``live`` part is published when it is committed. The first six blocks are not refreshed.

The generator trained without scene state (``model.scene_state: false``) plays the same way without
the scene state: its window is the ordinary window of its training, up to 36 latent frames of recent
context (:meth:`~worldcast.sampling.window.WindowLayout.ordinary`), so its round holds the first
frame, ten blocks and one more (:func:`live_round_latents`: 45 latent frames) and slides from the
eleventh block on; it reads its turns ``clip``-encoded (the input path's ``noclip`` codes clamped to
[-1, 1], which is that encoding of the same turn), and its idle clip is as long as its recent
context.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from worldcast.config.inference import InferenceConfig
from worldcast.data.controls import encode_turn
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.latents import (
    BLOCK,
    FIRST_TARGET,
    LATENT_SHAPE,
    RECENT,
    VIDEO_FRAMES_PER_BLOCK,
    VIDEO_FRAMES_PER_LATENT,
    last_video_frame,
    video_frames_of,
)
from worldcast.data.memory_frames import MemoryBlock
from worldcast.data.window import ClientWindow
from worldcast.engine.generator import window_conditions
from worldcast.modeling.ray_embedding import RAY_CONDITION_KEYS
from worldcast.player_state import (
    PlayerState,
    TargetGeometry,
    block_depth_frames,
    continuous_block_state,
    continuous_row_state,
    player_state_conditions,
    visible_latent_frames,
)
from worldcast.sampling.sampler import Sampler
from worldcast.sampling.schedulers import EntryNoiseStream
from worldcast.sampling.window import (
    CONTINUOUS_COLUMNS_KEY,
    KEEP_LATENTS,
    ORDINARY_RECENT,
    roll,
    slide_window_entries,
)

from .controls import BlockControls, LiveBlockControls
from .field_rows import FieldRows
from .idle_context import IdleContext, is_idle
from .loading import ClientModels, load_live_loop
from .memory_blocks import LiveMemoryBlocks
from .replicas import GeneratorReplica, rng_state, set_rng_state
from .rollout import BlockRead, PreparedBlock, Rollout
from .world_state import WorldState

__all__ = ["LIVE_ROUND_LATENTS", "PAGE_BLOCKS", "BlockCut", "LiveRollout", "live_round_latents"]

#: The latent frames of the round a live client keeps: the first frame, the first six blocks and
#: one more, the shortest round the loader allows.
LIVE_ROUND_LATENTS = FIRST_TARGET + BLOCK
#: The blocks of the page's rows and pitches a live rollout keeps: the page decodes the block the
#: generation has just finished, a block behind it at the most.
PAGE_BLOCKS = 8

#: The entries of a gathered window per player and video frame, which the target frames' rows
#: write again at the cut.
_PLAYER_ENTRIES = (
    "player_states",
    "player_weapon_ids",
    "player_control_substeps",
    "player_control_substep_valid",
)
#: The player-state conditions a block's rows leave as they are at its cut: the round's constants,
#: and the labels, written with its denoising.
_KEPT_AT_CUT = ("client_slot", "player_team_ids", "player_visible")


def live_round_latents(scene_state: bool) -> int:
    """The latent frames of the round a live client of the generator of ``model.scene_state``
    keeps: :data:`LIVE_ROUND_LATENTS`; without scene state the first frame, ten blocks and one more
    (45): from the eleventh block on every block is generated at latent frame 41 and its window
    reads latent frames 5-40, which a slide (it keeps latent frames 0 and 1 in place) leaves to
    it."""
    if scene_state:
        return LIVE_ROUND_LATENTS
    return 1 + ORDINARY_RECENT + 2 * BLOCK


@dataclass
class BlockCut:
    """A cut of an open block (:meth:`LiveRollout.cut`): what its denoising on any card reads.

    Attributes:
        f0 (int): the block's first latent frame, the session's.
        prepared (PreparedBlock): the window at the cut (its conditions on the client's card).
        noise (Tensor): the block's entry noise ``[1, 4, 48, 24, 42]`` on the client's card.
        rng (Tensor): the re-noise state the block's first denoising began with.
        geometry (TargetGeometry): what the target frames are re-tested with.
        page (tuple[dict, dict]): what the page reads of the block (:meth:`LiveRollout.page`).
        t_cut (float): the ``time.monotonic()`` when the cut was done.
    """

    f0: int
    prepared: PreparedBlock
    noise: torch.Tensor
    rng: torch.Tensor
    geometry: TargetGeometry
    page: tuple[dict, dict]
    t_cut: float


@dataclass
class _OpenBlock:
    """The newest block of a rollout that refreshes, until it is committed: what its first cut
    started from (the prepared window, the live loop's state, the continuous columns, the RNG
    state of its denoising) and its entry noise."""

    f0: int
    prepared: PreparedBlock
    cut: dict[str, Any]
    continuous: torch.Tensor
    rng: torch.Tensor
    noise: torch.Tensor | None = None


class _OwnBlocks:
    """The latents of the client's own blocks as the scene state reads them: ``blocks[f0 : f0 +
    4]`` is the four latent frames of the session's block ``f0``."""

    def __init__(self, latents: dict[tuple[str, int], torch.Tensor], media_id: str) -> None:
        self.latents, self.media_id = latents, media_id

    def __getitem__(self, frames: slice) -> torch.Tensor:
        return self.latents[(self.media_id, frames.start)]


class LiveRollout(Rollout):
    """One client's rollout under live controls (see the module docstring).

    Each block takes its controls as :class:`~worldcast.engine.inference.controls.
    LiveBlockControls`; a turn that would take the own pitch beyond [-89, 89] is cut there
    (:meth:`~worldcast.player_state.live.LiveLoop.clamp_pitch`), and its rows' pitch view controls
    are encoded again (the yaw keeps the code the input path fed). With ``live.wall_keys`` a block's
    movement keys are cleared from the row its camera is stopped at
    (:meth:`~worldcast.player_state.live.LiveLoop.release_keys`): the controls the generator, the
    state model and the ``live`` part read. The arguments are those of
    :class:`Rollout`; the world state is a
    :class:`~worldcast.engine.inference.messages.MessageWorldState`, and ``run.latent_frames``
    is the length of the round the client keeps (:func:`live_round_latents` at the least; a
    longer one gives the same blocks). The generator trained without scene state reads the clip
    encoding of each row's view controls: those of the input path clamped to [-1, 1].

    The methods a client calls (:meth:`denoise`, :meth:`commit`, :meth:`prepare_next`,
    :meth:`field_rows`, :meth:`camera_pitch`, :meth:`source_frame`) take the session's numbers:
    a block's first latent frame, or a latent or video frame, of the session; everything else is
    indexed by the round's.

    Attributes:
        shift (int): the latent frames the round has slid by.
        refreshes (bool): ``live.refresh`` (:meth:`refresh`).
        t_cut (float): the ``time.monotonic()`` when the newest cut was done and its denoising
            steps began.

    Raises:
        ValueError: the round is shorter than :func:`live_round_latents` of the generator.
    """

    open_ended = True

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
        super().__init__(cfg, models, sampler, window, world_state, lockstep=lockstep, dtype=dtype)
        needed = live_round_latents(self.scene_state)
        if self.latent_frames < needed:
            raise ValueError(
                f"a live client of model.scene_state {self.scene_state} keeps a round of {needed}"
                f" latent frames at the least (run.latent_frames), got {self.latent_frames}"
            )
        self.closed = load_live_loop(cfg, window, self.round_batch, models, world_state)
        self.continuous_rows = continuous_row_state(self.round_batch)
        self.shift = 0
        self.known = LiveMemoryBlocks(
            client=self.media_id, owners=self.known.owners, block_at=self.known.block_at
        )
        self.entry_noise = EntryNoiseStream(
            cfg.run.seed, LATENT_SHAPE, device=self.device, dtype=dtype
        )
        self._rows: dict[int, FieldRows] = {}
        self._pitch: dict[int, float] = {}
        self.idle_context = None
        if cfg.live.idle_context == "freeze":
            recent = RECENT if self.scene_state else ORDINARY_RECENT
            self.idle_context = IdleContext(recent // BLOCK)
        self._idle, self._weapons = False, ()
        self.refreshes = cfg.live.refresh
        self._open: _OpenBlock | None = None
        self.t_cut = 0.0

    def source_frame(self, latent: int) -> int:
        """The source frame of latent frame ``latent`` of the session."""
        return super().source_frame(latent + self.shift)

    def open(self) -> None:
        """:meth:`Rollout.open`, and the rows of the first frame for the page."""
        super().open()
        self._snapshot(0)

    def denoise(
        self,
        f0: int,
        controls: Callable[[], BlockControls | None],
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """:meth:`Rollout.denoise` of the session's block ``f0`` on the block's live controls, with
        the pitch cut and the movement keys released where ``live.wall_keys`` says.

        Raises:
            ValueError: ``controls`` gives no :class:`LiveBlockControls` (``None`` included: live
                play has no recorded controls).
        """
        if self.opens(f0):
            self._open_block(f0)
        return super().denoise(f0 - self.shift, lambda: self._live(f0, controls), on_step)

    def _live(self, f0: int, controls: Callable[[], BlockControls | None]) -> LiveBlockControls:
        """``controls()`` of the session's block ``f0`` as live play takes them: the pitch cut, the
        movement keys released, the clip encoding's view controls."""
        taken = controls()
        if not isinstance(taken, LiveBlockControls):
            raise ValueError(
                "player_state.source live generates each block from its player's live"
                f" controls (LiveBlockControls); got {type(taken).__name__}"
            )
        turn = self.closed.clamp_pitch(taken.turn)
        changed: dict[str, Any] = {}
        buttons = self.closed.release_keys(f0 - self.shift, taken.buttons, turn)
        if buttons is not taken.buttons:
            changed["buttons"] = buttons
        encoding = self.recording.spec.camera_encoding
        clamped = np.flatnonzero((turn != taken.turn).any(axis=1))
        clipped = encoding == "clip" and bool((np.abs(taken.view_deltas) > 1.0).any())
        if len(clamped) or clipped:
            view_deltas = taken.view_deltas.copy()
            if clipped:  # the clip code of a turn is its noclip code clamped to [-1, 1]
                np.clip(view_deltas, -1.0, 1.0, out=view_deltas)
            for row in clamped:
                view_deltas[row, 0] = encode_turn(turn[row], encoding)[0]
            changed["view_deltas"] = view_deltas
        if len(clamped):
            changed["turn"] = turn
        return replace(taken, **changed) if changed else taken

    def opens(self, f0: int) -> bool:
        """Whether a cut of the session's block ``f0`` opens it: a block of a window, prepared,
        with ``live.refresh``, not open yet."""
        return (
            self.refreshes
            and f0 - self.shift >= FIRST_TARGET
            and self.prepared is not None
            and (self._open is None or self._open.f0 != f0)
        )

    def _open_block(self, f0: int) -> None:
        self._open = _OpenBlock(
            f0,
            self.prepared,
            self.closed.save_cut(),
            self.continuous_rows.clone(),
            rng_state(self.device),
        )

    def can_refresh(self, f0: int) -> bool:
        """Whether the session's block ``f0`` is open: denoised, not committed, and refreshed by
        :meth:`refresh` (``live.refresh``; never one of the first six blocks)."""
        return self._open is not None and self._open.f0 == int(f0)

    def refresh(
        self,
        f0: int,
        controls: Callable[[], BlockControls | None],
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Denoise the open block ``f0`` (the session's) again from newer controls: the live loop,
        the continuous columns and the prepared window go back to before its first cut, the cut is
        made with ``controls()``, and the four denoising steps run on the same prefilled context
        with the same entry noise and the same re-noise draws (the RNG is set back to where the
        first denoising began). The same controls give the first version's x0 bit for bit.

        Args:
            f0 (int): the block's first latent frame, the session's.
            controls (Callable): as :meth:`denoise`'s.
            on_step (Callable | None): as :meth:`denoise`'s.

        Returns:
            Tensor: the block's x0 ``[1, 4, 48, 24, 42]`` float32, also in :attr:`output`.

        Raises:
            RuntimeError: the block is not open (:meth:`can_refresh`).
        """
        self._reopen(f0)
        set_rng_state(self.device, self._open.rng)
        return self.denoise(f0, controls, on_step)

    def _reopen(self, f0: int) -> None:
        """Take the open block ``f0`` back to before its first cut."""
        if not self.can_refresh(f0):
            raise RuntimeError(f"block {int(f0)} is not open: only the newest block is refreshed")
        block = self._open
        self.closed.restore_cut(block.cut)
        self.continuous_rows.copy_(block.continuous)
        self.prepared = block.prepared

    # ------------------------------------------------------------------ refresh on more cards
    def cut(self, f0: int, controls: Callable[[], BlockControls | None]) -> BlockCut:
        """Cut the session's block ``f0`` on ``controls()`` for :meth:`denoise_cut`, its denoising
        on any card: the block's first cut, which opens it, or a refresh's, from before its first
        cut (``live.refresh`` with :attr:`replicas`). The cut is :meth:`denoise`'s up to the
        denoising steps; what those read besides the window is kept with it, as the cut made it
        (the entry noise, the re-noise state of the block's first denoising, the geometry of the
        re-test of its target frames, the page's rows), so that its denoising may run while a later
        cut is made. One cut at a time, in order; the newest cut's version is the one to commit.

        Raises:
            RuntimeError: the block can be neither opened nor refreshed.
        """
        if self.opens(f0):
            self._open_block(f0)
        else:
            self._reopen(f0)
        local = f0 - self.shift
        taken = self._live(f0, controls)
        self._take_controls(local, taken)
        self.prepared = self._at_cut(local, self.prepared, taken)
        return BlockCut(
            f0=f0,
            prepared=self.prepared,
            noise=self._entry_noise(local),
            rng=self._open.rng,
            geometry=self.closed.visibility.target_geometry(local),
            page=self.page(f0),
            t_cut=time.monotonic(),
        )

    def denoise_cut(
        self,
        cut: BlockCut,
        replica: GeneratorReplica | None = None,
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """The four denoising steps of a :meth:`cut` on the client's card or on ``replica``,
        from the re-noise state of the block's first denoising, re-tested on the cut's geometry:
        on the client's card the x0 of :meth:`denoise` of the same controls. The output and the
        commit take the newest version later (:meth:`commit`).

        Args:
            cut (BlockCut): the cut.
            replica (GeneratorReplica | None): the card; ``None``: the client's own.
            on_step (Callable | None): as :meth:`denoise`'s.

        Returns:
            Tensor: x0 ``[1, 4, 48, 24, 42]`` float32 on the card.
        """
        prepared = cut.prepared
        if replica is None:
            set_rng_state(self.device, cut.rng)
            sampler, depth, prompt, device = (
                self.sampler,
                self.models.depth,
                self.models.prompt_embeds,
                self.device,
            )
            conditions, rays, noise = prepared.conditions, prepared.rays, cut.noise
        else:
            replica.prefilled()
            replica.rng.set_state(cut.rng)
            sampler, depth, prompt, device = (
                replica.sampler,
                replica.depth,
                replica.prompt_embeds,
                replica.device,
            )
            conditions = replica.conditions(prepared.conditions)
            rays = {key: conditions[key] for key in prepared.rays}
            noise = cut.noise.to(device)

        def retest(x0: torch.Tensor) -> Mapping[str, Any]:
            depth_frames = block_depth_frames(depth.log_depth, x0[0])
            return window_conditions(
                cut.geometry.relabel(prepared.batch, depth_frames),
                prompt_embeds=prompt,
                device=device,
                dtype=self.dtype,
                rays=rays if self.scene_state else None,
                observer_signals=self.scene_state,
            )

        return sampler.denoise(
            noise, conditions, frame_offset=prepared.context, retest=retest, on_step=on_step
        )

    def commit(self, f0: int, x0: torch.Tensor) -> BlockRead | None:
        """:meth:`Rollout.commit` of the session's block ``f0``; the page's rows of the block are
        taken now. An open block publishes its ``live`` part first, with the rows of its newest
        cut, the version committed; with :attr:`replicas` its x0 goes into :attr:`output` here
        (:meth:`denoise_cut` leaves it)."""
        if self.can_refresh(f0):
            self.closed.publish_controls()
            self._open = None
            if self.replicas:
                local = f0 - self.shift
                self.output[:, local : local + BLOCK] = x0.to(self.output.device)
        read = super().commit(f0 - self.shift, x0)
        self._snapshot(f0)
        return read

    def prepare_next(self, f0: int) -> None:
        """:meth:`Rollout.prepare_next` after the session's block ``f0``. If the block after it does
        not fit the round, the round slides by a block first (:meth:`_slide`); what the memory bank
        no longer holds is forgotten (:meth:`LiveMemoryBlocks.keep`)."""
        local = f0 - self.shift
        if local + BLOCK > self.latent_frames - BLOCK:
            self._slide()
            local -= BLOCK
        if self.scene_state:
            resident = {entry.block for entry in self.scene.bank.resident()}
            self.known.keep(self.world_state, resident, self.scene.watermark)
        super().prepare_next(local)

    def _slide(self) -> None:
        """The newest block is committed and the next does not fit: move the round up by a block,
        its first frame and latent frame 1 staying (:func:`~worldcast.sampling.window.
        slide_window_entries`). The output holds the first frame, the first six blocks and then the
        newest block; nothing reads it, the frames are decoded from each block's x0."""
        slide_window_entries(self.round_batch)
        roll(self.store, 0, BLOCK, KEEP_LATENTS)
        roll(self.continuous_rows, 1, BLOCK, KEEP_LATENTS)
        self.closed.slide()
        self.shift += BLOCK

    # ------------------------------------------------------------------------------------ the page
    def field_rows(self, latent: int) -> FieldRows:
        """The rows the player state field read at the session's latent frame ``latent``, as its
        block was committed (:meth:`Rollout.field_rows` under live controls); the page's thread
        reads no array of the round, which slides.

        A row is :attr:`FieldRows.LIVE` where another live player published it,
        :attr:`FieldRows.HIDDEN` where no player is in the world (a seat nobody plays, a seat a
        human took before its first ``live`` part, a player whose client is done or silent for
        3 s); ``drawn`` holds the players labelled visible at the block's first denoising step
        (``None`` for latent frame 0), and ``status`` the own position's estimator status.

        Raises:
            ValueError: ``latent`` is of a block older than the newest :data:`PAGE_BLOCKS`.
        """
        return self._kept(self._rows, latent, "latent frame")

    def camera_pitch(self, frame: int) -> float:
        """The pitch of the client's own camera at the session's video frame ``frame``, as its block
        was committed (:meth:`Rollout.camera_pitch`).

        Raises:
            ValueError: ``frame`` is of a block older than the newest :data:`PAGE_BLOCKS`.
        """
        return self._kept(self._pitch, frame, "video frame")

    @staticmethod
    def _kept(kept: dict[int, Any], frame: int, what: str) -> Any:
        if int(frame) not in kept:
            held = f"{what}s {min(kept)}-{max(kept)}" if kept else "none"
            raise ValueError(
                f"{what} {int(frame)} is not kept: a live rollout keeps what the page reads of the"
                f" newest {PAGE_BLOCKS} blocks ({held})"
            )
        return kept[int(frame)]

    def page(self, f0: int) -> tuple[dict[int, FieldRows], dict[int, float]]:
        """What the page reads of the session's block ``f0`` (the first frame for ``f0`` 0) as its
        newest cut wrote it: the rows of its latent frames (:meth:`field_rows`) and the pitch of
        its video frames (:meth:`camera_pitch`), by the session's numbers; an open block's frames
        carry them, its own rows not kept before it is committed."""
        local, count = f0 - self.shift, BLOCK if f0 else 1
        kept_rows, pitch = {}, {}
        for latent in range(local, local + count):
            rows = Rollout.field_rows(self, latent)
            source = np.where(self.closed.in_world(latent), FieldRows.LIVE, FieldRows.HIDDEN)
            source = source.astype(rows.source.dtype)
            source[self.slot] = FieldRows.OWN
            drawn = self.closed.visibility.drawn(latent)
            kept_rows[latent + self.shift] = FieldRows(
                rows.states, source, drawn, self.closed.status
            )
        for frame in video_frames_of(local, count):
            pitch[frame + VIDEO_FRAMES_PER_LATENT * self.shift] = Rollout.camera_pitch(self, frame)
        return kept_rows, pitch

    def _snapshot(self, f0: int) -> None:
        """Keep the rows of the session's block ``f0`` (the first frame for ``f0`` 0) and the pitch
        of its video frames, for :data:`PAGE_BLOCKS` blocks."""
        rows, pitch = self.page(f0)
        self._rows.update(rows)
        self._pitch.update(pitch)
        newest = f0 + (BLOCK if f0 else 1) - 1
        for kept, last, per_block in (
            (self._rows, newest, BLOCK),
            (self._pitch, last_video_frame(newest), VIDEO_FRAMES_PER_BLOCK),
        ):
            for key in [k for k in kept if k <= last - PAGE_BLOCKS * per_block]:
                del kept[key]

    # ------------------------------------------------------------------------------------ the cut
    def _take_controls(self, f0: int, controls: LiveBlockControls) -> None:
        """Write the block's controls into the round's history, then its rows at the cut
        (:meth:`~worldcast.player_state.live.LiveLoop.take_controls`) and the block's continuous
        columns (a later latent frame's are written with its own block); one of the first six
        blocks writes its latent frames of the round's conditions again
        (:meth:`_round_target`)."""
        super()._take_controls(f0, controls)
        self.closed.take_controls(
            f0,
            controls.buttons,
            controls.turn,
            controls.weapon,
            controls.times,
            publish=not self.can_refresh(f0 + self.shift),
        )
        self._refresh_continuous_rows(f0)
        if f0 < FIRST_TARGET:
            self.round_conditions = self._round_target(f0)

    def _refresh_continuous_rows(self, f0: int) -> None:
        """The continuous columns of block ``f0`` from the rows written, taken on from the block
        before: no column of the whole round is computed again."""
        continuous_block_state(self.continuous_rows, self.round_batch, f0, self.shift)

    def _block_key(self, f0: int) -> int:
        return f0 + self.shift

    def _entry_noise(self, f0: int) -> torch.Tensor:
        if not self.can_refresh(f0 + self.shift):
            return self.entry_noise.block(f0 + self.shift)
        if self._open.noise is None:  # drawn once: each version of the block starts from it
            self._open.noise = self.entry_noise.block(f0 + self.shift)
        return self._open.noise

    def _denoise_on_window(
        self,
        f0: int,
        prepared: PreparedBlock,
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        self.t_cut = time.monotonic()
        return super()._denoise_on_window(f0, prepared, on_step)

    def _exchange_positions(self, f0: int) -> None:
        """:meth:`Rollout._exchange_positions`, then the surface of the block's context, read before
        its controls for its labels
        (:meth:`~worldcast.player_state.live.LiveVisibility.prepare_target`)."""
        super()._exchange_positions(f0)
        self.closed.visibility.prepare_target(f0, self.output[0])

    def _round_target(self, f0: int) -> dict[str, Any]:
        """The round's conditions with block ``f0``'s latent frames as the cut wrote them: every
        player's rows and controls, the observer signals and the client's cameras (its controls
        are :meth:`Rollout._take_controls`'s), by the rules of :meth:`Rollout._round_conditions`.
        A block reads its own latent frames and those before them, which the cut leaves as they
        were."""
        # the video frames of latent frames f0 - 1 .. f0 + 3, folded as their own round: its latent
        # frames 1 .. 4 are the block's
        rows = slice(last_video_frame(f0 - 1), last_video_frame(f0 + BLOCK - 1) + 1)
        part = {key: self.round_batch[key][:, :, rows] for key in _PLAYER_ENTRIES}
        part.update((key, self.round_batch[key]) for key in ("client_slot", "player_team_ids"))
        states = PlayerState.from_batch(part, device=self.device)
        signals = None
        if self.scene_state:
            signals = {
                key: self.round_batch[key][:, f0 - 1 : f0 + BLOCK] for key in OBSERVER_SIGNAL_KEYS
            }
        written = player_state_conditions(
            states, torch.zeros_like(states.alive), observer_signals=signals
        )
        out = dict(self.round_conditions)
        for key, block in written.items():
            if key in _KEPT_AT_CUT:
                continue
            value = out[key].clone()
            value[:, f0 : f0 + BLOCK] = block[:, 1:]
            out[key] = value
        if not self.scene_state:
            return out
        frame_key = RAY_CONDITION_KEYS[0]  # the cameras of the round's latent frames
        cameras = out[frame_key].clone()
        cameras[:, f0 : f0 + BLOCK] = self.closed.cameras.as_predicted()[f0 : f0 + BLOCK].to(
            cameras
        )
        out[frame_key] = cameras
        return out

    def _block_at(self, slot: int, window_start: int, f0: int) -> MemoryBlock | None:
        """Another player's block, keyed at the cameras of its ``live`` part
        (:meth:`~worldcast.player_state.live.LiveLoop.block_at`)."""
        return self.closed.block_at(slot, window_start, f0)

    # ----------------------------------------------------------------------------- memory entries
    def _publish_block(self, f0: int, cameras: torch.Tensor) -> None:
        """Publish the block (the round's ``f0``) under its session number; its latents and ``live``
        part stay with the known blocks. Without scene state the block is only kept, as the idle
        context says."""
        session = f0 + self.shift
        if self.idle_context is not None:
            self.idle_context.keep(
                self.store, f0, cameras[f0 : f0 + BLOCK], self._weapons, idle=self._idle
            )
        if not self.scene_state:
            return
        self.world_state.publish_block(
            window_start=self.start_frame, f0=session, latents=self.store[f0 : f0 + BLOCK]
        )
        # the cameras are a view of the round's, which slides
        block = MemoryBlock.at(
            self.media_id, self.slot, self.start_frame, session, cameras[f0 : f0 + BLOCK].clone()
        )
        self.known.add_own(block)
        key = (self.media_id, session)
        self.known.latents[key] = self.store[f0 : f0 + BLOCK].clone()
        self.known.parts[key] = self.world_state.live_part(self.media_id, session)

    def _own_latents(self) -> _OwnBlocks:
        return _OwnBlocks(self.known.latents, self.media_id)

    def _block_latents(self, block: MemoryBlock, t: int) -> np.ndarray:
        latents = super()._block_latents(block, t)
        self.known.latents[(block.media_id, block.f0)] = torch.from_numpy(latents)
        return latents

    def _memory_frames(self, block: MemoryBlock, t: int) -> dict[str, torch.Tensor]:
        """The retrieved block's latents and its inputs as its player generated them, from what the
        known blocks kept of it (the client's own blocks too)."""
        key = (block.media_id, block.f0)
        frames = {"latents": self.known.latents[key].float(), "c2w": torch.as_tensor(block.c2w)}
        return self.closed.inputs_as_generated(frames, block, self.known.parts[key])

    def _at_cut(
        self, f0: int, prepared: PreparedBlock, controls: BlockControls | None
    ) -> PreparedBlock:
        """:meth:`Rollout._at_cut` of the window, with its target frames as the cut wrote them
        (:meth:`_live_target`), which the block's commit then publishes."""
        cut = self._live_target(f0, super()._at_cut(f0, prepared, controls))
        if self.idle_context is not None:
            self._idle = self._is_idle_at(f0, cut, controls)
            self._weapons = tuple(int(w) for w in np.unique(controls.weapon))
        return cut

    def _is_idle_at(self, f0: int, prepared: PreparedBlock, controls: LiveBlockControls) -> bool:
        """Whether block ``f0`` is idle for :class:`IdleContext`: its controls are
        (:func:`is_idle`) and nobody else is in its window. With ``live.idle_with_peers`` that is no
        player labelled visible on a latent frame of the recent context or the target frames (the
        player state field of every one of those frames is then empty, as a client alone in its
        world has it); otherwise nobody else in the world over those latent frames."""
        if not is_idle(controls):
            return False
        recent = self._recent(f0)
        if self.cfg.live.idle_with_peers:
            return not visible_latent_frames(prepared.batch)[0, -(recent + BLOCK) :].any()
        return not any(self.closed.in_world(f).any() for f in range(f0 - recent, f0 + BLOCK))

    def _live_target(self, f0: int, prepared: PreparedBlock) -> PreparedBlock:
        """``prepared`` with its target frames' rows, controls, observer signals and continuous
        columns from the round as the cut wrote them, their visibility labels tested again with the
        cameras they are now (against the same context), and their cameras; the anchor of the rays
        stays the camera of the first target frame when the window was prepared."""
        # the window's entries are the gather's (CPU tensors); they are copied through numpy
        first = video_frames_of(f0, BLOCK)[0]
        window = dict(prepared.batch)
        for key in _PLAYER_ENTRIES:
            value = window[key].numpy().copy()
            value[:, :, -VIDEO_FRAMES_PER_BLOCK:] = self.round_batch[key].numpy()[
                :, :, first : first + VIDEO_FRAMES_PER_BLOCK
            ]
            window[key] = torch.from_numpy(value)
        latent_entries = {key: self.round_batch[key] for key in OBSERVER_SIGNAL_KEYS}
        latent_entries[CONTINUOUS_COLUMNS_KEY] = self.continuous_rows
        for key, rows in latent_entries.items():
            value = window[key].numpy().copy()
            value[:, -BLOCK:] = rows.numpy()[:, f0 : f0 + BLOCK]
            window[key] = torch.from_numpy(value)
        window = self.closed.visibility.retest_target(window, f0)
        cameras = self.closed.cameras.with_block(prepared.cameras, f0)
        rays = dict(prepared.rays)
        if self.scene_state:
            frame_key = RAY_CONDITION_KEYS[0]  # the cameras of the window's frames
            rays[frame_key] = rays[frame_key].clone()
            rays[frame_key][:, -BLOCK:] = cameras[f0 : f0 + BLOCK].to(rays[frame_key])
        conditions = self._conditions(window, rays)
        return replace(prepared, conditions=conditions, cameras=cameras, batch=window, rays=rays)
