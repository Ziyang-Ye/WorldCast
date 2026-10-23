"""The player box of the entity criterion: where a player should appear in another player's view.

A box is the engine's standing player seen by the recordings' camera (horizontal field of view
106.26 degrees, the eye 64 u above the origin) with a vertical field of view of 73.74 degrees:
``0.45 h`` wide and centred horizontally on the projected head (the origin + 72 u), ``h`` the
vertical distance of the projected head and feet. The box is dropped when the feet or the head lie
at most 1 u in front of the camera, or when it lies entirely outside the frame grown by 15% on each
side; it is not clipped to the frame. Both players must be alive (App. "Evaluation protocols and
long horizons in detail", The entity criterion).

Video frame ``i`` of a window from source frame ``s`` shows source frame ``s + 2 i`` (32 fps). A
player's state there is that of the last tick at or before the frame's time, and the player is
known while that time is at most 1.5 ticks past the player's last tick.

A box list holds, per video frame, ``[slot, x0, y0, x1, y1, depth]`` of every other player with a
box, in slot order (pixels of the frame; ``depth`` of the feet, u); the detection dumps store the
recorded players' boxes this way (:mod:`worldcast.engine.evaluation.detector`).
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np

from worldcast.data.game import BODY_HEIGHT, EYE_HEIGHT, HFOV_DEGREES, TICK_RATE, VFOV_DEGREES
from worldcast.data.latents import SOURCE_FPS
from worldcast.data.recordings import ALIVE_INDEX, TickTable

__all__ = [
    "BOX_MARGIN",
    "BOX_WIDTH_RATIO",
    "MIN_DEPTH",
    "TICK_GAP",
    "TICK_SLACK",
    "clip_box",
    "clipped_side",
    "frame_states",
    "player_box",
    "table_boxes",
    "tick_rows",
    "tick_states",
    "view_boxes",
]

#: Width of a box over its height.
BOX_WIDTH_RATIO = 0.45
#: A box is dropped only if it lies entirely outside the frame grown by this share of the frame's
#: width and height on each side.
BOX_MARGIN = 0.15
#: Feet and head must lie more than this far in front of the camera, along its view axis, u.
MIN_DEPTH = 1.0
#: A player is known up to this long after its last tick, s.
TICK_GAP = 1.5 / TICK_RATE
#: Slack of the frame-time lookup, s.
TICK_SLACK = 1e-7

_TAN_H = math.tan(math.radians(HFOV_DEGREES / 2.0))
_TAN_V = math.tan(math.radians(VFOV_DEGREES / 2.0))


def _axes(yaw_degrees: float, pitch_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(float(yaw_degrees))
    pitch = math.radians(float(pitch_degrees))
    cosine_pitch = math.cos(pitch)
    forward = np.array(
        [math.cos(yaw) * cosine_pitch, math.sin(yaw) * cosine_pitch, -math.sin(pitch)],
        dtype=np.float64,
    )
    right = np.array([forward[1], -forward[0], 0.0], dtype=np.float64)
    right /= np.linalg.norm(right) + 1e-9
    return forward, right, np.cross(right, forward)


def _project(
    observer: np.ndarray, point: np.ndarray, width: int, height: int
) -> tuple[float, float, float] | None:
    forward, right, up = _axes(observer[3], observer[4])
    eye = observer[:3].copy()
    eye[2] += EYE_HEIGHT
    delta = point - eye
    depth = float(delta @ forward)
    if depth <= MIN_DEPTH:
        return None
    u = ((delta @ right / depth) / _TAN_H * 0.5 + 0.5) * int(width)
    v = (0.5 - (delta @ up / depth) / _TAN_V * 0.5) * int(height)
    return float(u), float(v), depth


def player_box(
    observer: np.ndarray, target: np.ndarray, width: int, height: int
) -> tuple[float, float, float, float, float] | None:
    """The box of ``target`` in the view of ``observer``.

    Args:
        observer (np.ndarray): ``[6]`` the observer's state ``x, y, z, yaw, pitch, alive`` (u,
            degrees).
        target (np.ndarray): ``[6]`` the other player's state.
        width (int): frame width, px.
        height (int): frame height, px.

    Returns:
        tuple[float, float, float, float, float] | None: ``(x0, y0, x1, y1, depth)`` in pixels,
        ``depth`` of the feet in u; ``None`` if either player is dead, the feet or the head is not
        in front of the camera, or the box lies entirely outside the grown frame.
    """
    observer = np.asarray(observer, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if observer[ALIVE_INDEX] <= 0.5 or target[ALIVE_INDEX] <= 0.5:
        return None
    head_xyz = target[:3].copy()
    head_xyz[2] += BODY_HEIGHT
    feet = _project(observer, target[:3], width, height)
    head = _project(observer, head_xyz, width, height)
    if feet is None or head is None:
        return None
    _, feet_v, depth = feet
    head_u, head_v, _ = head
    box_width = abs(feet_v - head_v) * BOX_WIDTH_RATIO
    box = (
        head_u - box_width / 2.0,
        min(head_v, feet_v),
        head_u + box_width / 2.0,
        max(head_v, feet_v),
        depth,
    )
    margin_x, margin_y = BOX_MARGIN * width, BOX_MARGIN * height
    if (
        box[2] < -margin_x
        or box[0] > width + margin_x
        or box[3] < -margin_y
        or box[1] > height + margin_y
    ):
        return None
    return box


def tick_states(table: TickTable) -> tuple[np.ndarray, np.ndarray]:
    """A player's tick times ``[n]`` (s) and states ``[n, 6]`` (``x, y, z, yaw, pitch, alive``),
    float64, from the player's tick table (:func:`~worldcast.data.recordings.read_player_ticks`)."""
    columns = (table.x, table.y, table.z, table.yaw, table.pitch, table.is_alive)
    states = np.stack([np.asarray(column, dtype=np.float64) for column in columns], axis=1)
    return np.asarray(table.t, dtype=np.float64), states


def tick_rows(times: np.ndarray, source_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The tick of a player at each source frame: the last at or before the frame's time.

    Args:
        times (np.ndarray): ``[n]`` the player's tick times, s, increasing.
        source_frames (np.ndarray): ``[T]`` source frames (32 fps).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``rows`` ``[T]`` int64 tick rows (clamped to the table) and
        ``known`` ``[T]`` bool: the frame's time is at most :data:`TICK_GAP` past the last tick.
    """
    times = np.asarray(times, dtype=np.float64)
    seconds = np.asarray(source_frames, dtype=np.float64) / SOURCE_FPS
    rows = np.searchsorted(times, seconds + TICK_SLACK, side="right") - 1
    known = seconds <= times[-1] + TICK_GAP + TICK_SLACK
    return np.clip(rows, 0, len(times) - 1), known


def frame_states(times: np.ndarray, states: np.ndarray, source_frames: np.ndarray) -> np.ndarray:
    """A player's state at each source frame, dead where the player is not known.

    Args:
        times (np.ndarray): ``[n]`` the player's tick times, s.
        states (np.ndarray): ``[n, 6]`` the player's ``x, y, z, yaw, pitch, alive`` per tick.
        source_frames (np.ndarray): ``[T]`` source frames (32 fps).

    Returns:
        np.ndarray: ``[T, 6]`` float64 (:func:`tick_rows`), ``alive`` 0 where not known.
    """
    rows, known = tick_rows(times, source_frames)
    out = np.asarray(states, dtype=np.float64)[rows, :6].copy()
    out[~known, ALIVE_INDEX] = 0.0
    return out


def view_boxes(
    observer: np.ndarray,
    players: Mapping[int, np.ndarray],
    observer_slot: int,
    width: int,
    height: int,
) -> list[list[list[float]]]:
    """The box list of one view: every other living player's :func:`player_box` per frame.

    Args:
        observer (np.ndarray): ``[T, 6]`` the observer's state per video frame.
        players (Mapping[int, np.ndarray]): slot -> ``[T, 6]`` state per video frame; the
            observer's own slot is skipped.
        observer_slot (int): the observer's slot.
        width (int): frame width, px.
        height (int): frame height, px.

    Returns:
        list[list[list[float]]]: ``T`` frames of ``[slot, x0, y0, x1, y1, depth]``; no box where
        the observer or the player is dead (``alive < 0.5``).
    """
    out = []
    for i in range(len(observer)):
        frame = []
        if observer[i][ALIVE_INDEX] >= 0.5:
            for slot in sorted(players):
                state = players[slot][i]
                if slot == observer_slot or state[ALIVE_INDEX] < 0.5:
                    continue
                box = player_box(observer[i], state, width, height)
                if box is not None:
                    frame.append([int(slot)] + [float(v) for v in box])
        out.append(frame)
    return out


def table_boxes(
    states: np.ndarray, first: int, count: int, observer_slot: int, width: int, height: int
) -> list[list[list[float]]]:
    """The box list of one view from a table of player states per video frame of a rollout (the
    states given to the generator); rows past the table have no boxes.

    Args:
        states (np.ndarray): ``[P, R, 6]`` every player's state per video frame of the rollout.
        first (int): the table row of the view's first frame.
        count (int): frames of the view.
        observer_slot (int): the observer's slot.
        width (int): frame width, px.
        height (int): frame height, px.

    Returns:
        list[list[list[float]]]: as :func:`view_boxes`.
    """
    states = np.asarray(states, dtype=np.float64)
    rows = int(first) + np.arange(int(count))
    picked = states[:, np.minimum(rows, states.shape[1] - 1)].copy()
    picked[:, rows >= states.shape[1], ALIVE_INDEX] = 0.0
    players = {slot: picked[slot] for slot in range(picked.shape[0])}
    return view_boxes(picked[observer_slot], players, observer_slot, width, height)


def clip_box(
    box: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """``(x0, y0, x1, y1)`` clipped to the frame, ``None`` if nothing of it is inside."""
    x0 = max(0.0, min(float(box[0]), width))
    y0 = max(0.0, min(float(box[1]), height))
    x1 = max(0.0, min(float(box[2]), width))
    y1 = max(0.0, min(float(box[3]), height))
    return None if (x1 <= x0 or y1 <= y0) else (x0, y0, x1, y1)


def clipped_side(box: Sequence[float], width: int, height: int) -> float:
    """``sqrt`` of the area of the box clipped to the frame, px (0 outside)."""
    clipped = clip_box(box, width, height)
    if clipped is None:
        return 0.0
    return math.sqrt((clipped[2] - clipped[0]) * (clipped[3] - clipped[1]))
