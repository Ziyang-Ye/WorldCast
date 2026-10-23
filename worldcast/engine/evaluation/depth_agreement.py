"""Depth agreement between clients: whether two views show the same scene at the same instant.

The depth head's depth of one view's latent frame is back-projected with that view's camera and
reprojected into another view's camera; a pixel of the other view that the reprojection reaches
agrees if the reprojected depth and the view's own depth there lie within the larger of 24 u and 5%
(App. "Scene state in detail", Table 19; App. "Training stages and scaling", Table 10; the Depth
row of Table 4a). The measure is kept as rows of counts, one per ordered pair of views and latent
frame, of three kinds:

* between clients (:data:`BETWEEN_CLIENTS`): two views at the same latent frame;
* own next latent frame (:data:`OWN_NEXT_FRAME`): a view and its own next latent frame;
* chance (:data:`CHANCE`): a between-clients row with the first view's depth taken from another
  instant (:func:`chance_frames`: 10 s away in a round, the most distant instant in a ten-second
  window) and placed at its current camera.

A pair of views is scored at a latent frame when both players are alive and the overlap gate passes
on the recorded cameras, for every arm, so all arms share one set of rows. A unit (a
round, or a ten-second window) scores the ratio of the agreeing to the reached pixels summed over
its rows, per segment (:func:`segment_ratios`); a table averages the units, and Table 19 keeps only
the pairs that are co-visible in the recorded video (:func:`covisibility_mask`).

The arithmetic is the paper's, so its rows are reproduced bit for bit: the rays in float32, camera
rotations fitted to them by least squares, depths in float32 and a float32 tolerance, the depth
head's log depth clamped to ``[0, log 4096]`` and rounded to float16
(:meth:`~worldcast.modeling.depth_head.DepthPredictor.log_depth` with ``paper_numerics``). The
paper's float32 ``exp`` was numpy's on x86-64; another platform's can differ by an ulp, which moves
a pixel in a few rows in ten thousand.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from worldcast.data.camera import NEAR_U, half_angle_tangents
from worldcast.data.controls import FRAME_TIME_SLACK, count_until
from worldcast.data.game import CROUCHED_EYE_HEIGHT, EYE_HEIGHT
from worldcast.data.latents import (
    BLOCK,
    FPS,
    LATENT_GRID,
    SOURCE_FRAMES_PER_LATENT,
    SOURCE_FRAMES_PER_VIDEO_FRAME,
    TOKEN_GRID,
    VIDEO_FRAMES_PER_LATENT,
    WINDOW_SOURCE_FRAMES,
    WINDOW_VIDEO_FRAMES,
    block_starts,
)
from worldcast.data.recordings import TickTable
from worldcast.scene_state.geometry import FAR_U, NPIX, TOLERANCE_ABS_U, TOLERANCE_REL

__all__ = [
    "BETWEEN_CLIENTS",
    "CHANCE",
    "CHANCE_LAG",
    "GATE_DEPTHS_U",
    "GATE_MIN_RAYS",
    "HIT_U",
    "OWN_NEXT_FRAME",
    "RECORDED_GAP_SECONDS",
    "RECORDED_WINDOW_OFFSET",
    "RECORDED_WINDOW_STRIDE",
    "ROW_COLUMNS",
    "SCENE_SPLIT_SECONDS",
    "VIEW_SPLIT_SECONDS",
    "WIDE_TOLERANCE_REL",
    "Cameras",
    "block_log_depth",
    "chance_frames",
    "compare_depths",
    "covisibility_mask",
    "frame_rotations",
    "latent_depth",
    "overlap_gate",
    "pair_rows",
    "recorded_cameras",
    "recorded_depth",
    "recorded_ticks",
    "recorded_windows",
    "segment_ratios",
]

#: Row kind: two views at the same latent frame.
BETWEEN_CLIENTS = 0
#: Row kind: a view and its own next latent frame.
OWN_NEXT_FRAME = 1
#: Row kind: a between-clients row with the first view's depth from another instant.
CHANCE = 2
#: The int32 columns of a row: the unit, the two views, the first view's latent frame, the kind,
#: then the counts of :func:`compare_depths` (reached pixels; those that agree; those within the
#: wider tolerance; those where the view's own depth is farther, and nearer, than the tolerance).
ROW_COLUMNS = (
    "unit",
    "a",
    "b",
    "frame",
    "kind",
    "reached",
    "agree",
    "agree_wide",
    "farther",
    "nearer",
)
#: A pixel at or beyond this depth has no surface to reproject, u.
HIT_U = 2048.0
#: The relative part of the wider tolerance of ``agree_wide``, ``max(24 u, 15% d)``.
WIDE_TOLERANCE_REL = 0.15
#: The overlap gate places the first view's rays at these depths, u ...
GATE_DEPTHS_U = (128.0, 256.0, 512.0, 1024.0)
#: ... and passes when at least this many of them land in the second camera at one of the depths.
GATE_MIN_RAYS = 50
#: The chance row's offset, latent frames: 10 s.
CHANCE_LAG = 40
#: A recorded player counts as alive up to this long after the last tick of its table, s.
RECORDED_GAP_SECONDS = 0.5
#: Table 19's segments of a round: the first 30 s and after 30 s.
SCENE_SPLIT_SECONDS = 30.0
#: Table 10's segments of a ten-second window: the first 5 s and the last 5 s.
VIEW_SPLIT_SECONDS = 5.0
#: The recorded depth of a round reads windows that start every this many source frames, from 0
#: and then from :data:`RECORDED_WINDOW_OFFSET` (:func:`recorded_windows`): the grid of the latent
#: cache, a window's video frame count apart.
RECORDED_WINDOW_STRIDE = WINDOW_VIDEO_FRAMES
#: The first start of the recorded depth's second run of windows, source frames: the latent
#: cache's midpoints.
RECORDED_WINDOW_OFFSET = RECORDED_WINDOW_STRIDE // 2

_TAN_H, _TAN_V = half_angle_tangents()
_REACHED, _AGREE = ROW_COLUMNS.index("reached"), ROW_COLUMNS.index("agree")


def _float32_rays(grid: tuple[int, int]) -> np.ndarray:
    """``[h w, 3]`` rays ``(1, x tan_h, y tan_v)`` through the cell centres, row 0 at the top,
    computed in float32 as the paper's rows were, as float64."""
    h, w = grid
    one, half, two = np.float32(1.0), np.float32(0.5), np.float32(2.0)
    x = (np.arange(w, dtype=np.float32) + half) / np.float32(w) * two - one
    y = -((np.arange(h, dtype=np.float32) + half) / np.float32(h) * two - one)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rays = np.stack([np.ones_like(xx), xx * np.float32(_TAN_H), yy * np.float32(_TAN_V)], -1)
    return rays.reshape(-1, 3).astype(np.float64)


def _float64_rays(grid: tuple[int, int]) -> np.ndarray:
    h, w = grid
    x = (np.arange(w, dtype=np.float64) + 0.5) / w * 2 - 1
    y = -((np.arange(h, dtype=np.float64) + 0.5) / h * 2 - 1)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.stack([np.ones_like(xx), xx * _TAN_H, yy * _TAN_V], -1).reshape(-1, 3)


#: The rays of the depth grid (one per pixel) and of the token grid, float32 values.
_PIXEL_RAYS = _float32_rays(LATENT_GRID)
_TOKEN_RAYS = _float32_rays(TOKEN_GRID)
#: The token grid's rays in float64: the targets of the rotation fit.
_TOKEN_RAYS_FLOAT64 = _float64_rays(TOKEN_GRID)


def frame_rotations(yaw_degrees: np.ndarray, pitch_degrees: np.ndarray) -> np.ndarray:
    """The rotation of each latent frame's camera, ``world = camera @ R``.

    Camera axes are forward, right and up (rays ``(1, x tan_h, y tan_v)``); pitch > 0 looks down.
    Each ``R`` is the least-squares fit of the float32 token rays to their world directions, as
    the paper's rows were computed; it differs from the exact rotation by about 1e-8.

    Args:
        yaw_degrees (np.ndarray): ``[N]`` yaw, degrees.
        pitch_degrees (np.ndarray): ``[N]`` pitch, degrees.

    Returns:
        np.ndarray: ``[N, 3, 3]`` float64.
    """
    yaw = np.radians(np.asarray(yaw_degrees, dtype=np.float64))
    pitch = np.radians(np.asarray(pitch_degrees, dtype=np.float64))
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(-pitch)
    forward = np.stack([cy * cp, sy * cp, sp], -1)[:, None]
    right = np.stack([sy, -cy, np.zeros_like(sy)], -1)[:, None]
    up = np.stack([-cy * sp, -sy * sp, cp], -1)[:, None]
    rays = _TOKEN_RAYS_FLOAT64
    world = rays[:, 0:1] * forward + rays[:, 1:2] * right + rays[:, 2:3] * up
    return np.stack([np.linalg.lstsq(_TOKEN_RAYS, w, rcond=None)[0] for w in world])


@dataclass(frozen=True)
class Cameras:
    """One view's cameras at its latent frames.

    The closed loop reprojects with the published positions: ``dataclasses.replace(cameras,
    positions=...)`` of the recorded cameras, whose orientations equal the controls' integral.

    Attributes:
        positions (np.ndarray): ``[N, 3]`` float64 player origins (feet), u.
        eye_heights (np.ndarray): ``[N]`` float64 camera heights above them, u.
        rotations (np.ndarray): ``[N, 3, 3]`` float64 :func:`frame_rotations`.
        alive (np.ndarray): ``[N]`` bool, the player is alive.
    """

    positions: np.ndarray
    eye_heights: np.ndarray
    rotations: np.ndarray
    alive: np.ndarray

    def __len__(self) -> int:
        return len(self.alive)

    @property
    def eyes(self) -> np.ndarray:
        """``[N, 3]`` float64 camera positions, u."""
        lift = np.zeros_like(self.positions)
        lift[:, 2] = self.eye_heights
        return self.positions + lift


def _latent_times(start_frame: int, latent_frames: int, source_fps: float) -> np.ndarray:
    frames = int(start_frame) + SOURCE_FRAMES_PER_LATENT * np.arange(int(latent_frames))
    return frames.astype(np.float64) / float(source_fps)


def recorded_ticks(
    table: TickTable, start_frame: int, latent_frames: int, source_fps: float
) -> np.ndarray:
    """The tick that each latent frame of a window of a recording reads.

    Latent frame ``k`` is source frame ``start_frame + 8 k``; it reads the last tick at or before
    that time (the first tick before the table starts).

    Args:
        table (TickTable): the player's ticks.
        start_frame (int): the window's first source frame (0 for a whole round).
        latent_frames (int): latent frames ``N`` of the window.
        source_fps (float): source frames per second of the recording.

    Returns:
        np.ndarray: ``[N]`` int rows of ``table``.
    """
    times = _latent_times(start_frame, latent_frames, source_fps)
    return np.clip(count_until(table.t, times) - 1, 0, len(table) - 1)


def recorded_cameras(
    table: TickTable, start_frame: int, latent_frames: int, source_fps: float
) -> Cameras:
    """A recorded player's cameras at the latent frames of a window of its recording.

    Each latent frame reads its tick (:func:`recorded_ticks`), and the player is alive there if the
    tick says so and the latent frame is at most :data:`RECORDED_GAP_SECONDS` after the table's
    last tick. The eye is 64 u above the origin, 46 u while crouched.

    Args:
        table (TickTable): the player's ticks.
        start_frame (int): the window's first source frame (0 for a whole round).
        latent_frames (int): latent frames ``N`` of the window.
        source_fps (float): source frames per second of the recording.

    Returns:
        Cameras: ``N`` cameras.
    """
    times = _latent_times(start_frame, latent_frames, source_fps)
    rows = recorded_ticks(table, start_frame, latent_frames, source_fps)
    states = table.states(rows)
    covered = times <= table.t[-1] + RECORDED_GAP_SECONDS + FRAME_TIME_SLACK
    crouched = np.array(["duck" in (table.active[row] or ()) for row in rows], dtype=bool)
    return Cameras(
        positions=states[:, :3].astype(np.float64),
        eye_heights=np.where(crouched, CROUCHED_EYE_HEIGHT, EYE_HEIGHT),
        rotations=frame_rotations(states[:, 3], states[:, 4]),
        alive=(states[:, 5] > 0) & covered,
    )


def _axial(log_depth: np.ndarray) -> np.ndarray:
    depth = np.exp(np.asarray(log_depth, dtype=np.float32))
    return np.clip(depth, np.float32(1.0), np.float32(FAR_U))


def latent_depth(log_depth: np.ndarray) -> np.ndarray:
    """Axial depth of each latent frame's last video frame, ``[N, 4, 24, 42]`` log axial depth ->
    ``[N, 24, 42]`` float32 in ``[1, 4096]`` u; NaN stays NaN."""
    return _axial(np.asarray(log_depth)[:, -1])


def recorded_windows(source_frames: int) -> list[int]:
    """The start source frames of the windows the recorded depth of a recording's round reads, in
    the order :func:`recorded_depth` breaks its ties by: every :data:`RECORDED_WINDOW_STRIDE`
    from 0, then from :data:`RECORDED_WINDOW_OFFSET`, each window inside the recording's
    ``source_frames``."""
    last = int(source_frames) - WINDOW_SOURCE_FRAMES
    return [
        start
        for first in (0, RECORDED_WINDOW_OFFSET)
        for start in range(first, last + 1, RECORDED_WINDOW_STRIDE)
    ]


def recorded_depth(windows: Mapping[int, np.ndarray], latent_frames: int) -> np.ndarray:
    """The recorded video's depth at the latent frames of a round, from overlapping windows.

    Latent frame ``k`` of the round is source frame ``8 k``, a video frame of each window that
    starts an even number of source frames before it and reaches it. Its depth comes from a
    window where that video frame is the last of a latent frame, if one is, and among those (or
    else among all) from the one where that latent frame is nearest the window's middle latent
    frame; on a tie, from the first in ``windows``.

    Args:
        windows (Mapping[int, np.ndarray]): start source frame -> log axial depth ``[L, 4, 24,
            42]`` of the window's latent frames (the depth head run once on the whole window), in
            the order of :func:`recorded_windows`.
        latent_frames (int): latent frames ``N`` of the round.

    Returns:
        np.ndarray: ``[N, 24, 42]`` float32 axial depth, u; NaN where no window has the frame (the
        end of a round that runs past the recording's last window).
    """
    count = int(latent_frames)
    out = np.full((count, *LATENT_GRID), np.nan, np.float32)
    best = np.full(count, np.inf)
    frames = SOURCE_FRAMES_PER_LATENT * np.arange(count)
    for start, log_depth in windows.items():
        log_depth = np.asarray(log_depth)
        offset = frames - int(start)
        video = offset // SOURCE_FRAMES_PER_VIDEO_FRAME
        found = (offset % SOURCE_FRAMES_PER_VIDEO_FRAME == 0) & (video >= 1)
        found &= video <= VIDEO_FRAMES_PER_LATENT * (len(log_depth) - 1)
        latent = (video - 1) // VIDEO_FRAMES_PER_LATENT + 1
        column = (video - 1) % VIDEO_FRAMES_PER_LATENT
        score = np.abs(latent - (len(log_depth) - 1) // 2).astype(np.float64)
        score += np.where(column == VIDEO_FRAMES_PER_LATENT - 1, 0, len(log_depth))
        better = found & (score < best)
        best[better] = score[better]
        out[better] = _axial(log_depth[latent[better], column[better]])
    return out


def block_log_depth(
    depth_fn: Callable[[np.ndarray], np.ndarray], latents: np.ndarray
) -> np.ndarray:
    """Log axial depth of a client's latent frames as it generated them: latent frame 0 alone, then
    each block of four on its own.

    Args:
        depth_fn (Callable): ``[F, 48, 24, 42]`` latents -> ``[F, 4, 24, 42]`` log axial depth.
        latents (np.ndarray): ``[N, 48, 24, 42]``, ``N = 1 + 4 b``.

    Returns:
        np.ndarray: ``[N, 4, 24, 42]`` in ``depth_fn``'s dtype.
    """
    if (len(latents) - 1) % BLOCK:
        raise ValueError(f"{len(latents)} latent frames; expected 1 + {BLOCK} b")
    parts = [depth_fn(latents[:1])]
    parts += [depth_fn(latents[f0 : f0 + BLOCK]) for f0 in block_starts(len(latents))]
    return np.concatenate(parts)


def chance_frames(frames: np.ndarray, count: int, *, most_distant: bool = False) -> np.ndarray:
    """The latent frame whose depth a chance row places at each latent frame ``k`` of a unit of
    ``count`` latent frames.

    ``k + 40`` (10 s later) when the unit has it. Otherwise, in a round, ``k - 40`` (10 s earlier,
    near the round's end), or ``-1`` (no chance row) when that is before latent frame 1; with
    ``most_distant``, the most distant of ``1 .. count - 1``, the later one on a tie (Table 10's
    ten-second windows, which never have ``k + 40``).

    Args:
        frames (np.ndarray): ``[n]`` latent frames ``k >= 1``.
        count (int): latent frames of the unit, of the shorter view of a pair.
        most_distant (bool): the unit is a ten-second window.

    Returns:
        np.ndarray: ``[n]`` int latent frames; ``-1`` where a round has no chance row.
    """
    k = np.asarray(frames)
    if most_distant:
        otherwise = np.where(count - 1 - k >= k - 1, count - 1, 1)
    else:
        otherwise = np.where(k - CHANCE_LAG >= 1, k - CHANCE_LAG, -1)
    return np.where(k + CHANCE_LAG < count, k + CHANCE_LAG, otherwise)


def _project(
    depth: np.ndarray,
    eyes_a: np.ndarray,
    rotations_a: np.ndarray,
    eyes_b: np.ndarray,
    inverses_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``[T, 1008]`` depths of view A -> their axial depth in camera B and the B pixel each lands
    on (-1 outside B's frustum), both ``[T, 1008]``."""
    height, width = LATENT_GRID
    points = eyes_a[:, None, :] + (depth[..., None] * _PIXEL_RAYS[None]) @ rotations_a
    local = np.einsum("tnj,tjk->tnk", points - eyes_b[:, None, :], inverses_b)
    forward = local[..., 0]
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(forward) & (forward > NEAR_U)
        safe = np.where(inside, forward, 1.0)
        x, y = local[..., 1] / safe, local[..., 2] / safe
        inside &= (np.abs(x) < _TAN_H) & (np.abs(y) < _TAN_V)
        j = np.clip(((x / _TAN_H + 1) / 2 * width).astype(int), 0, width - 1)
        i = np.clip(((1 - y / _TAN_V) / 2 * height).astype(int), 0, height - 1)
    return forward, np.where(inside, i * width + j, -1)


def overlap_gate(
    eyes_a: np.ndarray, rotations_a: np.ndarray, eyes_b: np.ndarray, rotations_b: np.ndarray
) -> np.ndarray:
    """Whether view A at ``T`` instants overlaps view B: at least :data:`GATE_MIN_RAYS` of A's
    rays, placed at one of :data:`GATE_DEPTHS_U`, land in B's frustum.

    Args:
        eyes_a (np.ndarray): ``[T, 3]`` float64 camera positions of A, u.
        rotations_a (np.ndarray): ``[T, 3, 3]`` float64 :func:`frame_rotations` of A.
        eyes_b (np.ndarray): ``[T, 3]`` of B.
        rotations_b (np.ndarray): ``[T, 3, 3]`` of B.

    Returns:
        np.ndarray: ``[T]`` bool.
    """
    inverses_b = np.linalg.inv(rotations_b)
    landed = np.zeros((len(eyes_a), NPIX), bool)
    for depth in GATE_DEPTHS_U:
        rays = np.full((len(eyes_a), NPIX), depth)
        landed |= _project(rays, eyes_a, rotations_a, eyes_b, inverses_b)[1] >= 0
    return landed.sum(1) >= GATE_MIN_RAYS


def compare_depths(
    depth_a: np.ndarray,
    eyes_a: np.ndarray,
    rotations_a: np.ndarray,
    depth_b: np.ndarray,
    eyes_b: np.ndarray,
    rotations_b: np.ndarray,
) -> np.ndarray:
    """View A's depth reprojected into view B, against B's own depth, at ``T`` instants.

    Each pixel of A nearer than :data:`HIT_U` is back-projected with A's camera and projected into
    B's (strictly inside the frustum, more than 1 u ahead); a B pixel is reached when a point lands
    on it (the nearest counts) and B has a depth there. A reached pixel agrees when the two depths
    lie within ``max(24 u, 5% d_B)``, in float32 as the paper computed it.

    Args:
        depth_a (np.ndarray): ``[T, 24, 42]`` axial depth of A, u (:func:`latent_depth`).
        eyes_a (np.ndarray): ``[T, 3]`` float64 camera positions of A, u.
        rotations_a (np.ndarray): ``[T, 3, 3]`` float64 :func:`frame_rotations` of A.
        depth_b (np.ndarray): ``[T, 24, 42]`` axial depth of B, u.
        eyes_b (np.ndarray): ``[T, 3]`` of B.
        rotations_b (np.ndarray): ``[T, 3, 3]`` of B.

    Returns:
        np.ndarray: ``[T, 5]`` int64, the count columns of :data:`ROW_COLUMNS`.
    """
    count = len(depth_a)
    source = np.asarray(depth_a, dtype=np.float32).reshape(count, NPIX)
    source = np.where(source < HIT_U, source, np.float32(np.nan))
    forward, pixel = _project(source, eyes_a, rotations_a, eyes_b, np.linalg.inv(rotations_b))
    flat = np.where(pixel >= 0, np.arange(count)[:, None] * NPIX + pixel, -1)
    landed = flat >= 0
    nearest = np.full(count * NPIX, np.inf)
    np.minimum.at(nearest, flat[landed], forward[landed])
    nearest = nearest.reshape(count, NPIX)
    target = np.asarray(depth_b, dtype=np.float32).reshape(count, NPIX)
    reached = np.isfinite(nearest) & np.isfinite(target)
    gap = np.abs(nearest - target)
    # float32 tolerances against a float64 gap: as the paper's rows were counted
    tolerance = np.maximum(np.float32(TOLERANCE_ABS_U), np.float32(TOLERANCE_REL) * target)
    wide = np.maximum(np.float32(TOLERANCE_ABS_U), np.float32(WIDE_TOLERANCE_REL) * target)
    counts = (
        reached,
        reached & (gap <= tolerance),
        reached & (gap <= wide),
        reached & (target > nearest + tolerance),
        reached & (target < nearest - tolerance),
    )
    return np.stack([c.sum(1) for c in counts], 1)


def _rows(
    head: tuple[int, int, int],
    kind: int,
    depth_a: np.ndarray,
    camera_a: Cameras,
    frames_a: np.ndarray,
    depth_b: np.ndarray,
    camera_b: Cameras,
    frames_b: np.ndarray,
) -> np.ndarray:
    counts = compare_depths(
        depth_a,
        camera_a.eyes[frames_a],
        camera_a.rotations[frames_a],
        depth_b,
        camera_b.eyes[frames_b],
        camera_b.rotations[frames_b],
    )
    keys = np.tile(np.array(head), (len(frames_a), 1))
    return np.column_stack([keys, frames_a, np.full(len(frames_a), kind), counts]).astype(np.int32)


def pair_rows(
    depths: Sequence[np.ndarray],
    cameras: Sequence[Cameras],
    *,
    gate: Sequence[Cameras] | None = None,
    unit: int = 0,
    most_distant_chance: bool = False,
) -> np.ndarray:
    """The rows of one unit (a round or a ten-second window) of one arm.

    For each view A and each other view B (in order), the between-clients rows at latent frames
    ``1 .. min(N_A, N_B) - 1`` and then their chance rows (:func:`chance_frames`); then A's
    own-next-frame rows at ``1 .. N_A - 2``. A frame is scored when both players are alive and
    :func:`overlap_gate` passes, both on the ``gate`` cameras.

    Args:
        depths (Sequence[np.ndarray]): per view, ``[N, 24, 42]`` axial depth, u
            (:func:`latent_depth`, :func:`recorded_depth`).
        cameras (Sequence[Cameras]): per view, the cameras its depth is reprojected with.
        gate (Sequence[Cameras] | None): per view, the recorded cameras that select the frames;
            default ``cameras``.
        unit (int): the unit's index, column ``unit`` of the rows.
        most_distant_chance (bool): the unit is a ten-second window (Table 10).

    Returns:
        np.ndarray: ``[n, 10]`` int32 rows, columns :data:`ROW_COLUMNS`.
    """
    gate = cameras if gate is None else gate
    views = range(len(depths))
    out = [np.zeros((0, len(ROW_COLUMNS)), np.int32)]
    for a in views:
        for b in [*(v for v in views if v != a), None]:
            if b is None:
                b, kind, count = a, OWN_NEXT_FRAME, len(depths[a])
                frames_a = np.arange(1, count - 1)
                frames_b = frames_a + 1
            else:
                kind, count = BETWEEN_CLIENTS, min(len(depths[a]), len(depths[b]))
                frames_a = frames_b = np.arange(1, count)
            keep = gate[a].alive[frames_a] & gate[b].alive[frames_b]
            frames_a, frames_b = frames_a[keep], frames_b[keep]
            keep = overlap_gate(
                gate[a].eyes[frames_a],
                gate[a].rotations[frames_a],
                gate[b].eyes[frames_b],
                gate[b].rotations[frames_b],
            )
            frames_a, frames_b = frames_a[keep], frames_b[keep]
            head, camera_a, camera_b = (unit, a, b), cameras[a], cameras[b]
            depth_a, depth_b = depths[a][frames_a], depths[b][frames_b]
            out.append(_rows(head, kind, depth_a, camera_a, frames_a, depth_b, camera_b, frames_b))
            if kind == BETWEEN_CLIENTS:
                shifted = chance_frames(frames_a, count, most_distant=most_distant_chance)
                has = shifted >= 1
                depth_a, frames_a, frames_b = depths[a][shifted[has]], frames_a[has], frames_b[has]
                depth_b = depth_b[has]
                out.append(
                    _rows(head, CHANCE, depth_a, camera_a, frames_a, depth_b, camera_b, frames_b)
                )
    return np.concatenate(out)


def covisibility_mask(rows: np.ndarray, recorded_rows: np.ndarray) -> np.ndarray:
    """The between-clients and chance rows whose pair and latent frame are co-visible in the
    recorded video: its between-clients row there has an agreeing pixel (Table 19).

    Args:
        rows (np.ndarray): ``[n, 10]`` rows of an arm.
        recorded_rows (np.ndarray): ``[n, 10]`` the recorded video's rows of the same units.

    Returns:
        np.ndarray: ``[n]`` bool.

    Raises:
        ValueError: the two hold other ``(unit, a, b, frame, kind)`` keys.
    """
    rows, recorded = np.asarray(rows), np.asarray(recorded_rows)
    if rows.shape != recorded.shape or not np.array_equal(rows[:, :5], recorded[:, :5]):
        raise ValueError("the rows and the recorded video's rows hold other keys")
    head = rows[:, :4].astype(np.int64)
    keys = np.ravel_multi_index(head.T, tuple(head.max(0, initial=0) + 1))
    covisible = (recorded[:, 4] == BETWEEN_CLIENTS) & (recorded[:, _AGREE] > 0)
    return np.isin(keys, keys[covisible]) & np.isin(rows[:, 4], (BETWEEN_CLIENTS, CHANCE))


def segment_ratios(rows: np.ndarray, kind: int, units: int, split_seconds: float) -> np.ndarray:
    """Per unit and segment, the share of reached pixels that agree, over the rows of one kind.

    Latent frame ``k`` is at ``k / 4`` s; the segments are before ``split_seconds`` and from it on
    (:data:`SCENE_SPLIT_SECONDS`, :data:`VIEW_SPLIT_SECONDS`).

    Args:
        rows (np.ndarray): ``[n, 10]`` rows (:data:`ROW_COLUMNS`), units ``0 .. units - 1``.
        kind (int): :data:`BETWEEN_CLIENTS`, :data:`OWN_NEXT_FRAME` or :data:`CHANCE`.
        units (int): units in the table.
        split_seconds (float): the boundary of the segments, s.

    Returns:
        np.ndarray: ``[units, 2]`` float64 ratios of sums; NaN where no pixel is reached.
    """
    rows = np.asarray(rows)
    rows = rows[rows[:, 4] == kind]
    later = rows[:, 3] * (VIDEO_FRAMES_PER_LATENT / FPS) >= split_seconds
    cell = rows[:, 0].astype(np.int64) * 2 + later
    agree, reached = (
        np.bincount(cell, weights=rows[:, column], minlength=2 * units).reshape(units, 2)
        for column in (_AGREE, _REACHED)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(reached > 0, agree / reached, np.nan)
