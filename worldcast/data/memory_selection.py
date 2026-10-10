"""Choosing the memory frames of a training window (App. "Scene state in detail", Training).

With probability 0.8 a training sample includes GT memory frames: the four latent frames of a
same-round teammate's view that end no later than the last target frame and show most of what the
recent context lacks, as counted by m_k (:mod:`worldcast.data.memory_mask`). For one training
window this module chooses the target block and the teammate block:

* **candidates** (:func:`teammate_blocks`): every block of every cached window of a living
  teammate, its latent frames with known flash and scope labels (:func:`known_clear`);
* **memory frames** (:func:`best_block`): the complete teammate block that covers the most of the
  target block's unseen surface;
* **acceptance**, beyond the paper's text and as the training windows were selected: at least
  :data:`MIN_TOKENS` unseen and :data:`MIN_TOKENS` covered token-times, a second complete block
  that shares no frame with the memory frames (:func:`least_covering_block`), and the surface
  patch of at least one covered token looks the same in the target frame and in a memory frame
  (:func:`~worldcast.data.appearance.appearance_check`);
* one accepted target block is drawn with probability proportional to its m_k count
  (:func:`draw_target`).
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from worldcast.config.training import WINDOW_LATENT_FRAMES

from .appearance import appearance_check
from .camera import SCOPE_ZOOM_FOV, c2w_from_state_rows, camera_tans, window_cameras
from .controls import OPENCS2_WEAPONS, AlignmentError
from .labels import FlashCurve, ScopeCurve, load_observer_signal_curves, observer_signal_rows
from .latent_cache import cached_window_starts, load_block_latents
from .latents import (
    BLOCK,
    FIRST_TARGET,
    RECENT,
    SOURCE_FRAMES_PER_LATENT,
    SOURCE_FRAMES_PER_VIDEO_FRAME,
    block_starts,
    last_video_frame,
    source_frame,
    video_frames_of,
)
from .map_mesh import MapMesh, MeshLibrary
from .memory_frames import MemoryBlock, batch_keys, block_memory_frames, recorded_block
from .memory_mask import (
    TargetSurface,
    behind_players,
    coverage,
    project_points,
    rgb_quality_mask,
    token_mask,
    unseen_surface,
)
from .recordings import MediaRecord, player_rows
from .video import VideoFrames
from .window import ClientWindow, PlayerFrames, WindowSpec, frame_endpoint_rows

__all__ = [
    "BIDIRECTIONAL",
    "BLOCK_CAUSAL",
    "MIN_TOKENS",
    "TARGET_SEED_TAG",
    "MemoryFrameConfig",
    "MemoryFrameSelection",
    "MemoryFrameSource",
    "TargetBlock",
    "TeammateFrame",
    "best_block",
    "complete_blocks",
    "draw_target",
    "known_clear",
    "least_covering_block",
    "memory_frame_item",
    "select_memory_frames",
    "teammate_blocks",
]

#: Acceptance: at least this many unseen and this many covered token-times in the target block.
MIN_TOKENS = 20
#: Third word of the seed of the target draw, ``default_rng([dataset index, start frame, tag])``.
TARGET_SEED_TAG = 0xE0911


@dataclass(frozen=True)
class MemoryFrameConfig:
    """Where a window's target block may lie, and what its unseen surface is judged against.

    Attributes:
        recent (int): recent context latent frames before the target frames.
        first_target (int): the first latent frame a target block may start at.
        first_frame_only (bool): the unseen surface is judged against the first frame alone (the
            bidirectional stage 2s noises its recent context with the target).
    """

    recent: int
    first_target: int
    first_frame_only: bool = False


#: Stage 2s: the 41 latent frames are the first frame, the memory frames' place, 32 recent context
#: frames and the target block, the window's last.
BIDIRECTIONAL = MemoryFrameConfig(
    recent=WINDOW_LATENT_FRAMES - 1 - 2 * BLOCK,
    first_target=WINDOW_LATENT_FRAMES - BLOCK,
    first_frame_only=True,
)
#: Stages 3 and 4: the window of inference, 12 recent context frames and target blocks from latent
#: frame 25.
BLOCK_CAUSAL = MemoryFrameConfig(recent=RECENT, first_target=FIRST_TARGET)


class MemoryFrameSource:
    """What the memory frames of a window are read from: the recordings, the latent cache, the
    observer-signal labels and the collision meshes. It caches window lists and label curves (one
    instance per loader worker).

    Args:
        dataset_root (str | Path): the recordings.
        latent_cache_root (str | Path): the cached windows of every player.
        observer_signal_label_root (str | Path): flash and scope labels.
        meshes (MeshLibrary): the collision mesh per map.
        spec (WindowSpec): the cached windows' sampling.
        config (MemoryFrameConfig): :data:`BIDIRECTIONAL` or :data:`BLOCK_CAUSAL`.
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        latent_cache_root: str | Path,
        observer_signal_label_root: str | Path,
        meshes: MeshLibrary,
        spec: WindowSpec,
        config: MemoryFrameConfig,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.latent_cache_root = Path(latent_cache_root)
        self.observer_signal_label_root = Path(observer_signal_label_root)
        self.meshes = meshes
        self.spec = spec
        self.config = config
        self._window_starts: dict[str, np.ndarray] = {}
        self._curves: dict[str, tuple[FlashCurve, ScopeCurve]] = {}

    def window_starts(self, media_id: str) -> np.ndarray:
        """Sorted source-frame starts of the cached windows of ``media_id``."""
        if media_id not in self._window_starts:
            starts = cached_window_starts(self.latent_cache_root, media_id)
            self._window_starts[media_id] = starts
        return self._window_starts[media_id]

    def curves(self, media: MediaRecord) -> tuple[FlashCurve, ScopeCurve]:
        """The flash and scope curves of ``media``; a missing file reads as unknown."""
        if media.media_id not in self._curves:
            self._curves[media.media_id] = load_observer_signal_curves(
                self.observer_signal_label_root,
                media.media_id,
                source_frames=media.source_frames,
                missing_ok=True,
            )
        return self._curves[media.media_id]

    def window_signals(self, media: MediaRecord, start_frame: int) -> dict[str, np.ndarray]:
        """``{key: [L]}`` observer signals of the cached window of ``media`` at ``start_frame``."""
        flash, scope = self.curves(media)
        return observer_signal_rows(flash, scope, start_frame, self.spec.latent_frames)


# ================================================================================ label status
def known_clear(
    signals: Mapping[str, np.ndarray], latent_frames: Iterable[int], weapon_ids: np.ndarray
) -> np.ndarray:
    """Which latent frames of a window have a known, unflashed view.

    Args:
        signals (Mapping[str, np.ndarray]): the player's per-latent observer signals, ``[L]`` each.
        latent_frames (Iterable[int]): the latent frames to report.
        weapon_ids (np.ndarray): the player's weapon id per video frame.

    Returns:
        np.ndarray: ``[n]`` bool: the flash and scope labels are known, the frame is not flashed,
        and a scoped frame holds a weapon with a known zoom (so its field of view is known).
    """
    ids = np.asarray(list(latent_frames), int)
    clear = (
        (np.asarray(signals["obs_scope_valid"])[ids] > 0)
        & (np.asarray(signals["obs_flash_valid"])[ids] > 0)
        & (np.asarray(signals["obs_flash_flag"])[ids] == 0)
    )
    weapons = np.asarray(weapon_ids)
    for j, k in enumerate(ids):
        if np.asarray(signals["obs_scope_on"])[k] > 0:
            weapon = OPENCS2_WEAPONS[int(weapons[last_video_frame(k)])]
            level = int(np.asarray(signals["obs_scope_level"])[k])
            clear[j] &= weapon in SCOPE_ZOOM_FOV and level > 0
    return clear


def _video_frame_tans(
    flash: FlashCurve,
    scope: ScopeCurve,
    start_frame: int,
    video_frames: Sequence[int],
    weapon_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Field of view and label status at single video frames, ``[n, 2]`` float32 and ``[n]`` bool;
    each is read as a one-latent window starting at its own source frame."""
    tans, clear = [], []
    for v in video_frames:
        start = int(start_frame) + SOURCE_FRAMES_PER_VIDEO_FRAME * int(v)
        signals = observer_signal_rows(flash, scope, start, 1)
        weapons = np.asarray(weapon_ids)[int(v) : int(v) + 1]
        tans.append(camera_tans(signals, weapons, [0])[0])
        clear.append(bool(known_clear(signals, [0], weapons)[0]))
    return np.asarray(tans, np.float32), np.asarray(clear, bool)


# ================================================================= the teammate blocks and frames
@dataclass(frozen=True)
class TeammateFrame:
    """Latent frame ``j`` of a teammate block.

    Attributes:
        block (MemoryBlock): the block.
        j (int): the latent frame's index in the block, 0-3.
        frame (int): its source frame.
        tans (np.ndarray): ``[2]`` float32 ``(tan_h, tan_v)`` of the teammate's camera.
        players (np.ndarray): ``[P, 6]`` every player's state at that frame.
    """

    block: MemoryBlock
    j: int
    frame: int
    tans: np.ndarray
    players: np.ndarray


def teammate_blocks(source: MemoryFrameSource, window: ClientWindow) -> list[MemoryBlock]:
    """Every block ``f0 = 1, 5, ..., 37`` of every cached window of every teammate of the client
    during which the teammate lives, ordered by slot, window start and ``f0``.

    A block is listed once (by its media and its first source frame, first cached window first); a
    block past the teammate's ticks is skipped.
    """
    spec = source.spec
    client_slot = window.media.player_slot
    team = window.player_frames[client_slot].team_id
    out: list[MemoryBlock] = []
    seen = set()
    for slot, media in sorted(window.round_slots.items()):
        if slot == client_slot or window.player_frames[slot].team_id != team:
            continue
        table = window.tick_tables[slot]
        for start_frame in source.window_starts(media.media_id).tolist():
            endpoints = frame_endpoint_rows(table.t, float(media.fps), start_frame, spec)
            for f0 in block_starts(spec.latent_frames):
                key = (media.media_id, source_frame(start_frame, f0) // SOURCE_FRAMES_PER_LATENT)
                if key in seen:
                    continue
                block = recorded_block(
                    slot, media, table, start_frame, f0, spec, endpoints=endpoints
                )
                if block is not None:
                    seen.add(key)
                    out.append(block)
    return out


def _teammate_frames(
    source: MemoryFrameSource, blocks: Sequence[MemoryBlock], window: ClientWindow
) -> list[TeammateFrame]:
    """The known-clear latent frames of the teammate blocks."""
    spec, fps = source.spec, float(window.media.fps)
    windows: dict[tuple[str, int], tuple | None] = {}
    out = []
    for block in blocks:
        key = (block.media_id, block.window_start)
        if key not in windows:
            media, ticks = window.round_slots[block.slot], window.tick_tables[block.slot]
            try:
                frames = PlayerFrames.from_ticks(ticks, media, block.window_start, spec)
            except AlignmentError:
                windows[key] = None
                continue
            signals = source.window_signals(media, block.window_start)
            tans = camera_tans(signals, frames.weapon_ids, range(spec.latent_frames))
            windows[key] = (frames, signals, tans)
        if windows[key] is None:
            continue
        frames, signals, tans = windows[key]
        for j in range(BLOCK):
            f = block.f0 + j
            if known_clear(signals, [f], frames.weapon_ids)[0]:
                frame = source_frame(block.window_start, f)
                players = player_rows(window.tick_tables, frame, fps)
                out.append(TeammateFrame(block, j, frame, tans[f], players))
    return out


def _block_key(frame: TeammateFrame) -> tuple[str, int, int]:
    return (str(frame.block.media_id), int(frame.block.window_start), int(frame.block.f0))


def complete_blocks(frames: Sequence[TeammateFrame]) -> dict[tuple[str, int, int], list[int]]:
    """``block key -> [index of latent frame j = 0, 1, 2, 3]`` of the blocks all four of whose
    frames are among ``frames``."""
    groups: dict[tuple[str, int, int], dict[int, int]] = {}
    for i, frame in enumerate(frames):
        groups.setdefault(_block_key(frame), {})[int(frame.j)] = i
    return {
        key: [group[j] for j in range(BLOCK)]
        for key, group in groups.items()
        if set(group) == set(range(BLOCK))
    }


def best_block(
    masks: Sequence[np.ndarray],
    frames: Sequence[TeammateFrame],
    unseen: np.ndarray,
    quality: Callable[[int], np.ndarray],
) -> tuple[list[int], np.ndarray]:
    """The complete block whose four frames cover the most unseen points, RGB quality applied.

    Blocks are visited in decreasing geometric coverage (an upper bound; ties by block key), the
    RGB quality ``quality(i)`` of frame ``i`` is measured only when needed, and the search stops
    once no block left can beat the best. Frames of different blocks are never combined.

    Args:
        masks (Sequence[np.ndarray]): per frame, ``[M]`` bool, the surface points it covers.
        frames (Sequence[TeammateFrame]): the candidate frames.
        unseen (np.ndarray): ``[M]`` bool, the unseen surface points.
        quality (Callable[[int], np.ndarray]): frame index -> ``[M]`` bool, the points whose RGB
            patch can be measured in that frame.

    Returns:
        tuple[list[int], np.ndarray]: the frame indices of the block (``[]`` when none covers
        anything) and the covered unseen points ``[M]`` bool.
    """
    unseen = np.asarray(unseen, dtype=bool)
    candidates = []
    for key, ids in complete_blocks(frames).items():
        upper = np.logical_or.reduce([masks[i] for i in ids]) & unseen
        candidates.append((int(upper.sum()), key, ids))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    selected, best, best_count = [], np.zeros_like(unseen), 0
    measured: dict[int, np.ndarray] = {}
    for upper_count, _, ids in candidates:
        if upper_count <= best_count:
            break
        covered = np.zeros_like(unseen)
        for i in ids:
            if i not in measured:
                measured[i] = np.asarray(quality(i), dtype=bool)
            covered |= np.asarray(masks[i], dtype=bool) & measured[i] & unseen
        if int(covered.sum()) > best_count:
            selected, best, best_count = ids, covered, int(covered.sum())
    return selected, best


def least_covering_block(
    masks: Sequence[np.ndarray],
    frames: Sequence[TeammateFrame],
    unseen: np.ndarray,
    *,
    exclude: Sequence[tuple[str, int]],
) -> list[int] | None:
    """The complete block covering the fewest unseen points that shares no source frame
    ``(media_id, frame)`` with ``exclude``; ``None`` when there is none."""
    unseen = np.asarray(unseen, dtype=bool)
    banned = {(str(m), int(f)) for m, f in exclude}
    scored = []
    for key, ids in complete_blocks(frames).items():
        if any((str(frames[i].block.media_id), int(frames[i].frame)) in banned for i in ids):
            continue
        covered = np.logical_or.reduce([np.asarray(masks[i], dtype=bool) for i in ids]) & unseen
        scored.append((int(covered.sum()), key, ids))
    return min(scored, key=lambda x: (x[0], x[1]))[2] if scored else None


# ========================================================== the memory frames of a training window
@dataclass
class TargetBlock:
    """An accepted target block and its memory frames.

    Attributes:
        start (int): its first latent frame.
        memory_mask (np.ndarray): m_k, ``[4, 12, 21]`` bool.
        frames (list[TeammateFrame]): the memory frames, ``j = 0 .. 3``.
        matching_tokens (int): the tokens of m_k that pass the appearance check.
    """

    start: int
    memory_mask: np.ndarray
    frames: list[TeammateFrame]
    matching_tokens: int = 0

    @property
    def block(self) -> MemoryBlock:
        """The teammate block of the memory frames."""
        return self.frames[0].block

    @property
    def tans(self) -> np.ndarray:
        """``[4, 2]`` float32 fields of view of the memory frames."""
        return np.asarray([frame.tans for frame in self.frames], dtype=np.float32)


@dataclass
class MemoryFrameSelection:
    """The accepted target blocks of one window, with the client's cameras.

    Attributes:
        accepted (list[TargetBlock]): the accepted target blocks, ascending.
        window_c2w (torch.Tensor): ``[L, 4, 4]`` float32 cameras of the window's latent frames.
        window_tans (np.ndarray): ``[L, 2]`` float32 their ``(tan_h, tan_v)``.
    """

    accepted: list[TargetBlock]
    window_c2w: torch.Tensor
    window_tans: np.ndarray


@dataclass
class _Search:
    """What every target block of one window is judged with."""

    source: MemoryFrameSource
    window: ClientWindow
    window_c2w: torch.Tensor
    window_tans: np.ndarray
    mesh: MapMesh
    video: VideoFrames
    candidates: list[TeammateFrame]


def _clear_targets(source: MemoryFrameSource, window: ClientWindow) -> list[int]:
    """The target blocks whose first frame, recent context and target frames are known clear."""
    config = source.config
    signals, client = window.item.signals, window.client_frames
    return [
        s
        for s in block_starts(source.spec.latent_frames)
        if s >= config.first_target
        and known_clear(signals, [0, *range(s - config.recent, s + BLOCK)], client.weapon_ids).all()
    ]


def _unseen_surface(search: _Search, s: int) -> TargetSurface | None:
    """The unseen surface of target block ``s`` (context, living players and RGB quality applied),
    or None when the recent context is not known clear or fewer than :data:`MIN_TOKENS` tokens are
    unseen."""
    config, window = search.source.config, search.window
    client, client_slot = window.client_frames, window.media.player_slot
    fps, media_id = float(window.media.fps), window.media.media_id
    recent_frames = video_frames_of(s - config.recent, config.recent)
    flash, scope = search.source.curves(window.media)
    recent_tans, recent_clear = _video_frame_tans(
        flash, scope, window.row.start_frame, recent_frames, client.weapon_ids
    )
    if not recent_clear.all():
        return None
    recent_c2w = c2w_from_state_rows(client.states[recent_frames])
    if config.first_frame_only:
        recent_c2w, recent_tans = np.zeros((0, 4, 4)), np.zeros((0, 2))
    window_c2w, window_tans = search.window_c2w, search.window_tans
    surface = unseen_surface(
        search.mesh,
        window_c2w[s : s + BLOCK],
        window_tans[s : s + BLOCK],
        recent_c2w,
        recent_tans,
        window_c2w[0],
        window_tans[0],
    )
    for qi, f in enumerate(range(s, s + BLOCK)):
        rows = surface.frame == qi
        frame = source_frame(window.row.start_frame, f)
        players = player_rows(window.tick_tables, frame, fps)
        eye = window_c2w[f, :3, 3].numpy()
        surface.unseen[rows] &= ~behind_players(
            surface.points[rows], eye, players, viewer_slot=client_slot
        )
        uv, _ = project_points(surface.points[rows], window_c2w[f], window_tans[f])
        surface.unseen[rows] &= rgb_quality_mask(search.video.get(media_id, frame), uv)
    return surface if surface.unseen.sum() >= MIN_TOKENS else None


def _propose(search: _Search, s: int) -> TargetBlock | None:
    """Target block ``s`` with the teammate block that covers the most of its unseen surface, or
    None when it fails the acceptance of the module docstring."""
    surface = _unseen_surface(search, s)
    if surface is None:
        return None
    mesh, video = search.mesh, search.video
    last = source_frame(search.window.row.start_frame, s + BLOCK - 1)
    earlier = [c for c in search.candidates if c.block.t_last <= last]
    masks = [
        coverage(
            mesh, surface, c.block.c2w[c.j], c.tans, players=c.players, viewer_slot=c.block.slot
        )
        for c in earlier
    ]
    if not masks or not np.logical_or.reduce(masks).any():
        return None

    def quality(i: int) -> np.ndarray:
        c = earlier[i]
        uv, _ = project_points(surface.points, c.block.c2w[c.j], c.tans)
        return rgb_quality_mask(video.get(c.block.media_id, c.frame), uv)

    selected, covered = best_block(masks, earlier, surface.unseen, quality)
    memory_mask = token_mask(surface, covered)
    if memory_mask.sum() < MIN_TOKENS:
        return None
    frames = sorted((earlier[i] for i in selected), key=lambda c: (c.frame, c.block.media_id))
    # as the training windows were selected: a second complete block, sharing no frame with the
    # memory frames, must exist (the block itself is not used)
    exclude = [(c.block.media_id, c.frame) for c in frames]
    if least_covering_block(masks, earlier, surface.unseen, exclude=exclude) is None:
        return None
    return TargetBlock(s, memory_mask, frames)


def _matching_tokens(search: _Search, target: TargetBlock) -> int:
    """The tokens of the target's m_k that pass the appearance check."""
    window = search.window
    matching = appearance_check(
        search.mesh,
        search.video,
        client_media=window.media,
        tick_tables=window.tick_tables,
        start_frame=window.row.start_frame,
        target_start=target.start,
        memory_mask=target.memory_mask,
        window_c2w=np.asarray(search.window_c2w.numpy(), dtype=np.float64),
        window_tans=np.asarray(search.window_tans, dtype=np.float64),
        block=target.block,
        block_tans=target.tans,
    )
    return int(matching.sum())


def select_memory_frames(source: MemoryFrameSource, window: ClientWindow) -> MemoryFrameSelection:
    """Every acceptable target block of one training window, with its memory frames and m_k.

    The memory frames may end no later than the target block's last latent frame.

    Args:
        source (MemoryFrameSource): what the memory frames are read from.
        window (ClientWindow): the training window (the cached window of its player).

    Returns:
        MemoryFrameSelection: the accepted target blocks (none when the window has none).
    """
    spec, client = source.spec, window.client_frames
    signals = window.item.signals
    window_c2w, window_tans = window_cameras(
        client.states, signals, client.weapon_ids, spec.latent_frames
    )
    targets = _clear_targets(source, window)
    if not targets:
        return MemoryFrameSelection([], window_c2w, window_tans)
    last = source_frame(window.row.start_frame, max(targets) + BLOCK - 1)
    blocks = [b for b in teammate_blocks(source, window) if b.t_last <= last]
    media = window.media
    mesh = source.meshes.get(media.map_name)
    recordings = {**{m.media_id: m for m in window.round_slots.values()}, media.media_id: media}
    with VideoFrames(source.dataset_root, recordings) as video:
        candidates = _teammate_frames(source, blocks, window)
        search = _Search(source, window, window_c2w, window_tans, mesh, video, candidates)
        proposals = [target for s in targets if (target := _propose(search, s)) is not None]
        accepted = []
        for target in proposals:
            target.matching_tokens = _matching_tokens(search, target)
            if target.matching_tokens:
                accepted.append(target)
    return MemoryFrameSelection(accepted, window_c2w, window_tans)


def draw_target(
    selection: MemoryFrameSelection, *, dataset_index: int, start_frame: int
) -> TargetBlock | None:
    """One accepted target block, with probability proportional to its m_k count, from
    ``np.random.default_rng([dataset_index, start_frame, TARGET_SEED_TAG])``; ``None`` when none
    was accepted."""
    if not selection.accepted:
        return None
    rng = np.random.default_rng([int(dataset_index), int(start_frame), TARGET_SEED_TAG])
    weight = np.asarray([t.memory_mask.sum() for t in selection.accepted], dtype=np.float64)
    return selection.accepted[int(rng.choice(len(selection.accepted), p=weight / weight.sum()))]


def memory_frame_item(
    source: MemoryFrameSource,
    window: ClientWindow,
    selection: MemoryFrameSelection,
    target: TargetBlock,
) -> dict[str, torch.Tensor]:
    """The item keys of a window's memory frames.

    Returns:
        dict[str, Tensor]: ``window_target_start`` (int64), ``window_c2w`` ``[L, 4, 4]`` and
        ``window_tans`` ``[L, 2]`` float32, ``window_memory_mask`` (m_k, ``[4, 12, 21]`` bool), and
        the memory frames' ``memory_frames_*``
        (:func:`~worldcast.data.memory_frames.block_memory_frames`).
    """
    block = target.block
    media, ticks = window.round_slots[block.slot], window.tick_tables[block.slot]
    latents = load_block_latents(
        source.latent_cache_root, media.media_id, block.window_start, block.f0
    )
    frames = PlayerFrames.from_ticks(ticks, media, block.window_start, source.spec)
    signals = source.window_signals(media, block.window_start)
    return {
        "window_target_start": torch.tensor(int(target.start)),
        "window_c2w": selection.window_c2w.float(),
        "window_tans": torch.from_numpy(np.asarray(selection.window_tans)),
        "window_memory_mask": torch.from_numpy(np.asarray(target.memory_mask, dtype=bool)),
        **batch_keys(block_memory_frames(block, latents, frames, signals)),
    }
