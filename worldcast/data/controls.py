"""Controls from 64 Hz tick tables: buttons, pitch and yaw turns, weapon, ordered substeps (the
players' controls of the generator, the state model's controls).

A window that starts at source frame ``s`` has a frame every ``stride`` source frames: frame ``f``
sits at ``t_f = (s + f stride) / source_fps`` and owns the ticks in ``(t_f - dt, t_f]``, ``dt =
stride / source_fps``. A video frame (stride 2 of the 32 fps recordings) owns four ticks, a latent
frame (stride 8) sixteen.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .game import TICK_RATE
from .latents import (
    FPS,
    SOURCE_FRAMES_PER_LATENT,
    SOURCE_FRAMES_PER_VIDEO_FRAME,
    VIDEO_FRAMES_PER_LATENT,
)

__all__ = [
    "CAMERA_BIN_DEGREES",
    "CAMERA_DELTA_SCALE",
    "CAMERA_ENCODINGS",
    "CAMERA_MAX_DEGREES",
    "CAMERA_MU",
    "CONTROL_BUTTONS",
    "DEFAULT_CAMERA_ENCODING",
    "FRAME_TIME_SLACK",
    "IGNORED_BUTTONS",
    "OPENCS2_BUTTONS",
    "OPENCS2_WEAPONS",
    "STATE_MODEL_CONTROL_DIM",
    "STATE_MODEL_MAX_TICK_GAP_SECONDS",
    "STATE_MODEL_SUBSTEPS",
    "SUBSTEPS_PER_VIDEO_FRAME",
    "AlignmentError",
    "FrameControls",
    "SubstepControls",
    "align_ticks_to_ordered_substeps",
    "align_ticks_to_video_frames",
    "check_camera_encoding",
    "check_weapon_ids",
    "count_until",
    "encode_turn",
    "encode_weapon",
    "frame_tick_ranges",
    "normalize_weapon_name",
    "quantize_camera_delta",
    "state_model_controls",
]

#: A frame reads the ticks up to its own time plus this slack, seconds (the 64 Hz time base is not
#: exact in floating point).
FRAME_TIME_SLACK = 1e-7

#: The OpenCS2 button enum, in its published order.
OPENCS2_BUTTONS = (
    "attack",
    "attack2",
    "back",
    "duck",
    "forward",
    "jump",
    "look_at_weapon",
    "move_left",
    "move_right",
    "reload",
    "score",
    "speed",
    "use",
)
#: Keyboard-turn binds of the recordings' ``active`` column; the recorded pitch and yaw deltas
#: already hold the turn, so they are skipped.
IGNORED_BUTTONS = frozenset(("turn_left", "turn_right"))
#: The 11 button channels of the generator's controls, in order: the OpenCS2 buttons without the
#: scoreboard and use keys (App. "The player state field in detail", Injection).
CONTROL_BUTTONS = (
    "forward",
    "back",
    "move_left",
    "move_right",
    "jump",
    "duck",
    "speed",
    "attack",
    "attack2",
    "reload",
    "look_at_weapon",
)

#: The 52-way weapon vocabulary, indexed by normalised name (``input_weapon_id`` merges two knives).
OPENCS2_WEAPONS = (
    "<none>",
    "<unk>",
    "ak47",
    "aug",
    "awp",
    "bayonet",
    "bizon",
    "c4",
    "cz75a",
    "deagle",
    "decoy",
    "elite",
    "famas",
    "fiveseven",
    "flashbang",
    "g3sg1",
    "galilar",
    "glock",
    "hegrenade",
    "hkp2000",
    "incgrenade",
    "inferno",
    "knife",
    "knife_butterfly",
    "knife_karambit",
    "knife_m9_bayonet",
    "knife_stiletto",
    "m249",
    "m4a1",
    "m4a1_silencer",
    "mac10",
    "mag7",
    "molotov",
    "mp5sd",
    "mp7",
    "mp9",
    "negev",
    "nova",
    "p250",
    "p90",
    "planted_c4",
    "revolver",
    "sawedoff",
    "scar20",
    "sg556",
    "smokegrenade",
    "ssg08",
    "taser",
    "tec9",
    "ump45",
    "usp_silencer",
    "xm1014",
)
_WEAPON_INDEX = MappingProxyType({name: index for index, name in enumerate(OPENCS2_WEAPONS)})
_WEAPON_ALIASES = MappingProxyType(
    {
        "c4explosive": "c4",
        "highexplosivegrenade": "hegrenade",
        "weapon_c4": "c4",
        "m9bayonet": "knife_m9_bayonet",
        "shadowdaggers": "knife",
    }
)

#: Turn mu-law: turns in degrees, +-20 degree range, 0.5 degree buckets, mu = 2.7.
CAMERA_MAX_DEGREES = 20.0
CAMERA_BIN_DEGREES = 0.5
CAMERA_MU = 2.7
#: The encodings of a video frame's turn: ``noclip`` keeps turns beyond +-20 degrees, ``clip`` cuts
#: them. The released four-step generator and every stage with scene state read ``noclip``, the
#: stages without scene state ``clip`` (:data:`worldcast.config.training.STAGES`).
CAMERA_ENCODINGS = ("noclip", "clip")
DEFAULT_CAMERA_ENCODING = "noclip"
#: Degrees per unit substep turn: substeps hold the turn divided by it, the camera integral of the
#: player states multiplies it back.
CAMERA_DELTA_SCALE = 5.0
#: Ordered substeps per video frame of the players' controls: one per engine tick.
SUBSTEPS_PER_VIDEO_FRAME = round(TICK_RATE / FPS)

#: State-model controls: per latent frame 16 substeps (one per engine tick), each the 13 OpenCS2
#: buttons, the pitch and yaw turns divided by :data:`CAMERA_DELTA_SCALE` and a valid flag. A latent
#: frame needs a tick within 1.5 ticks of its end.
STATE_MODEL_SUBSTEPS = VIDEO_FRAMES_PER_LATENT * SUBSTEPS_PER_VIDEO_FRAME
STATE_MODEL_CONTROL_DIM = len(OPENCS2_BUTTONS) + 3
STATE_MODEL_MAX_TICK_GAP_SECONDS = 1.5 / TICK_RATE


def normalize_weapon_name(value: object) -> str:
    """Name of a raw ``input_weapon`` in :data:`OPENCS2_WEAPONS`, ``"<none>"`` or ``"<unk>"``."""
    if value is None:
        return "<none>"
    name = str(value).strip().lower()
    if name in {"", "nan", "none", "null", "<none>"}:
        return "<none>"
    name = name.removeprefix("weapon_")
    name = _WEAPON_ALIASES.get(name, name)
    return name if name in _WEAPON_INDEX else "<unk>"


def encode_weapon(value: object) -> int:
    """Weapon id (index into :data:`OPENCS2_WEAPONS`) of a raw ``input_weapon`` value."""
    return _WEAPON_INDEX[normalize_weapon_name(value)]


def check_weapon_ids(weapon: np.ndarray) -> None:
    """Raise unless every id of ``weapon`` indexes :data:`OPENCS2_WEAPONS` (the generator's
    weapon embedding has one row per id)."""
    if weapon.size and (weapon.min() < 0 or weapon.max() >= len(OPENCS2_WEAPONS)):
        raise ValueError(f"weapon ids must lie in [0, {len(OPENCS2_WEAPONS)})")


def check_camera_encoding(camera_encoding: str) -> str:
    """Return ``camera_encoding`` if it is one of :data:`CAMERA_ENCODINGS`; raise otherwise."""
    if camera_encoding in CAMERA_ENCODINGS:
        return camera_encoding
    raise ValueError(f"camera_encoding must be one of {CAMERA_ENCODINGS}, got {camera_encoding!r}")


def quantize_camera_delta(turn: Sequence[float], *, clip: bool) -> np.ndarray:
    """mu-law encode one frame's summed turn ``[pitch, yaw]`` (degrees) to ``[2]`` float32.

    ``v = sign(x/20) log1p(2.7 |x/20|) / log1p(2.7)``, rounded to 0.5 degree buckets, in float32. A
    turn within +-20 degrees encodes to [-1, 1]; without ``clip`` a larger turn gives ``|v| > 1``.
    """
    turn = np.asarray(turn, dtype=np.float32)
    if not np.all(np.isfinite(turn)):
        raise ValueError("a turn must be finite")
    clipped = np.clip(turn, -CAMERA_MAX_DEGREES, CAMERA_MAX_DEGREES) if clip else turn
    normalized = clipped / CAMERA_MAX_DEGREES
    encoded = np.sign(normalized) * (np.log1p(CAMERA_MU * np.abs(normalized)) / np.log1p(CAMERA_MU))
    encoded = encoded * CAMERA_MAX_DEGREES
    num_buckets = int(CAMERA_MAX_DEGREES / CAMERA_BIN_DEGREES)
    discretized = np.round((encoded + CAMERA_MAX_DEGREES) / CAMERA_BIN_DEGREES).astype(np.int64)
    return ((discretized - num_buckets) / num_buckets).astype(np.float32)


def encode_turn(
    turn: Sequence[float], camera_encoding: str = DEFAULT_CAMERA_ENCODING
) -> np.ndarray:
    """:func:`quantize_camera_delta` of one frame's turn ``[pitch, yaw]`` under ``camera_encoding``
    (one of :data:`CAMERA_ENCODINGS`): the one place that turns the name of an encoding into the
    quantizer's ``clip``."""
    return quantize_camera_delta(turn, clip=check_camera_encoding(camera_encoding) == "clip")


class AlignmentError(ValueError):
    """A tick table cannot serve the requested window."""


@dataclass(frozen=True)
class FrameControls:
    """The controls of one player per video frame.

    Attributes:
        buttons (np.ndarray): ``[T, 11]`` float32 in {0, 1}, :data:`CONTROL_BUTTONS`.
        view_deltas (np.ndarray): ``[T, 2]`` float32 mu-law ``[pitch, yaw]`` turn.
        weapon (np.ndarray): ``[T]`` int64 weapon id held at the frame's last tick.
    """

    buttons: np.ndarray
    view_deltas: np.ndarray
    weapon: np.ndarray


@dataclass(frozen=True)
class SubstepControls:
    """The controls of one player per frame, split into ``S`` ordered substeps.

    Attributes:
        values (np.ndarray): ``[F, S, B + 2]`` float32, per substep the ``B`` held buttons and the
            summed pitch and yaw turns divided by :data:`CAMERA_DELTA_SCALE` (unquantised).
        valid (np.ndarray): ``[F, S]`` bool, True where a tick fell into the substep.
    """

    values: np.ndarray
    valid: np.ndarray


def _tick_columns(
    timestamps, delta_pitch, delta_yaw, *other_columns
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate raw tick columns; return ``timestamps`` float64 and the two turn columns float32."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1:
        raise AlignmentError(f"tick timestamps must be 1-D, got {timestamps.shape}")
    if not len(timestamps):
        raise AlignmentError("tick table is empty")
    if not np.all(np.isfinite(timestamps)):
        raise AlignmentError("tick timestamps must be finite")
    if np.any(np.diff(timestamps) <= 0):
        raise AlignmentError("tick timestamps must be strictly increasing")
    delta_pitch = np.asarray(delta_pitch, dtype=np.float32)
    delta_yaw = np.asarray(delta_yaw, dtype=np.float32)
    if len({len(timestamps), len(delta_pitch), len(delta_yaw), *map(len, other_columns)}) != 1:
        raise AlignmentError("tick columns have inconsistent lengths")
    if delta_pitch.ndim != 1 or delta_yaw.ndim != 1:
        raise AlignmentError("tick turn columns must be 1-D")
    if not np.all(np.isfinite(delta_pitch)) or not np.all(np.isfinite(delta_yaw)):
        raise AlignmentError("tick turn columns must be finite")
    return timestamps, delta_pitch, delta_yaw


def _button_channels(
    row: Iterable[str] | None, channel_of: dict[str, int], unknown: set
) -> list[int]:
    """Channels of the buttons held in one tick; names outside the enum go to ``unknown``."""
    channels = []
    for button in row or ():
        name = str(button)
        if name in IGNORED_BUTTONS:
            continue
        if name not in OPENCS2_BUTTONS:
            unknown.add("<null>" if button is None else name)
        elif name in channel_of:
            channels.append(channel_of[name])
    return channels


def _frame_times(start_frame: int, num_frames: int, stride: int, source_fps: float) -> np.ndarray:
    frames = int(start_frame) + np.arange(num_frames, dtype=np.float64) * int(stride)
    return frames / float(source_fps)


def _substeps(timestamps: np.ndarray, frame_time: float, interval: float, num_substeps: int):
    """Substep of each tick of the frame interval ``(frame_time - interval, frame_time]``: ``clip(
    ceil((t - left) / interval * S) - 1, 0, S - 1)``."""
    relative = (timestamps - (frame_time - interval)) / interval
    return np.clip(np.ceil(relative * num_substeps).astype(np.int64) - 1, 0, num_substeps - 1)


def count_until(times: np.ndarray, time: float | np.ndarray) -> np.ndarray:
    """How many of the ascending ``times`` lie at or before ``time`` (a scalar or an array), within
    :data:`FRAME_TIME_SLACK`: one past the last tick a frame at ``time`` reads."""
    return np.searchsorted(times, time + FRAME_TIME_SLACK, side="right")


def frame_tick_ranges(
    *,
    timestamps: np.ndarray,
    start_frame: int,
    num_frames: int,
    source_fps: float,
    stride: int,
    max_tick_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tick rows ``[begin[f], end[f])`` of every frame of a window.

    Args:
        timestamps (np.ndarray): ``[n]`` float64 tick times, seconds, strictly increasing.
        start_frame (int): source frame of frame 0.
        num_frames (int): frames.
        source_fps (float): source frames per second.
        stride (int): source frames per frame (2 for video frames, 8 for latent frames).
        max_tick_gap_seconds (float): largest tolerated tick gap and frame-endpoint lag, seconds.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(begin, end)``, each ``[num_frames]`` int64.

    Raises:
        AlignmentError: a frame has no tick, its last tick lags it by more than
            ``max_tick_gap_seconds``, or two ticks of the window are further apart.
    """
    if start_frame < 0 or num_frames <= 0 or stride <= 0 or source_fps <= 0:
        raise ValueError("invalid frame window")
    if max_tick_gap_seconds <= 0:
        raise ValueError("max_tick_gap_seconds must be positive")

    frame_times = _frame_times(start_frame, num_frames, stride, source_fps)
    begin = np.searchsorted(timestamps, frame_times - stride / source_fps, side="right")
    end = count_until(timestamps, frame_times)
    if np.any(end <= begin):
        bad = int(np.flatnonzero(end <= begin)[0])
        raise AlignmentError(f"no tick rows align to frame {bad}")

    endpoint_lag = frame_times - timestamps[end - 1]
    if np.any(endpoint_lag > max_tick_gap_seconds):
        bad = int(np.flatnonzero(endpoint_lag > max_tick_gap_seconds)[0])
        raise AlignmentError(
            f"a tick gap leaves the end of frame {bad} uncovered by"
            f" {float(endpoint_lag[bad]):.6f}s",
        )

    gaps = np.diff(timestamps[max(0, int(begin[0]) - 1) : int(end[-1])])
    if gaps.size and float(gaps.max()) > max_tick_gap_seconds:
        raise AlignmentError(
            f"a tick gap of {float(gaps.max()):.6f}s exceeds {max_tick_gap_seconds:.6f}s inside"
            " the requested window",
        )
    return begin, end


def _forward_fill_weapons(weapons: Sequence[object]) -> list[object]:
    result, previous = [], None
    for weapon in weapons:
        name = "" if weapon is None else str(weapon).strip().lower()
        if name and name != "nan":
            previous = weapon
        result.append(previous)
    return result


def align_ticks_to_video_frames(
    *,
    timestamps: Sequence[float],
    held_buttons: Sequence[Iterable[str] | None],
    delta_pitch: Sequence[float],
    delta_yaw: Sequence[float],
    input_weapons: Sequence[object],
    start_frame: int,
    num_frames: int,
    source_fps: float,
    max_tick_gap_seconds: float,
    camera_encoding: str,
) -> FrameControls:
    """Aggregate one player's ticks into the controls of each video frame (the client's own
    controls).

    Per video frame (:func:`frame_tick_ranges`) the buttons are OR'd, the turn is summed in float32
    and encoded by :func:`quantize_camera_delta`, and the weapon is the one held at the last tick.

    Args:
        timestamps (Sequence[float]): ``[n]`` tick times, seconds.
        held_buttons (Sequence[Iterable[str] | None]): ``[n]`` the names of the buttons held at
            each tick (the recordings' ``active`` column).
        delta_pitch (Sequence[float]): ``[n]`` pitch turn per tick, degrees.
        delta_yaw (Sequence[float]): ``[n]`` yaw turn per tick, degrees.
        input_weapons (Sequence[object]): ``[n]`` raw weapon names.
        start_frame (int): source frame of video frame 0.
        num_frames (int): video frames ``T``.
        source_fps (float): source frames per second.
        max_tick_gap_seconds (float): largest tolerated tick gap and frame-endpoint lag, seconds.
        camera_encoding (str): one of :data:`CAMERA_ENCODINGS`.

    Returns:
        FrameControls: ``T`` video frames.

    Raises:
        AlignmentError: see :func:`frame_tick_ranges`; also for an unknown button.
    """
    check_camera_encoding(camera_encoding)
    timestamps, delta_pitch, delta_yaw = _tick_columns(
        timestamps, delta_pitch, delta_yaw, held_buttons, input_weapons
    )
    begin, end = frame_tick_ranges(
        timestamps=timestamps,
        start_frame=start_frame,
        num_frames=num_frames,
        source_fps=source_fps,
        stride=SOURCE_FRAMES_PER_VIDEO_FRAME,
        max_tick_gap_seconds=max_tick_gap_seconds,
    )

    unknown = set()
    channel_of = {name: channel for channel, name in enumerate(CONTROL_BUTTONS)}
    buttons = np.zeros((num_frames, len(CONTROL_BUTTONS)), dtype=np.float32)
    view_deltas = np.zeros((num_frames, 2), dtype=np.float32)
    turns = np.zeros((num_frames, 2), dtype=np.float32)
    weapon_ids = np.zeros((num_frames,), dtype=np.int64)
    filled_weapons = _forward_fill_weapons(input_weapons)
    for frame, (lo, hi) in enumerate(zip(begin.tolist(), end.tolist())):
        for row in held_buttons[lo:hi]:
            buttons[frame, _button_channels(row, channel_of, unknown)] = 1.0
        # Summed per frame in float32, one frame at a time: this op order is part of the numbers.
        turns[frame] = [delta_pitch[lo:hi].sum(), delta_yaw[lo:hi].sum()]
        view_deltas[frame] = encode_turn(turns[frame], camera_encoding)
        weapon_ids[frame] = encode_weapon(filled_weapons[hi - 1])

    if unknown:
        raise AlignmentError(f"unknown OpenCS2 buttons: {sorted(unknown)}")
    return FrameControls(buttons=buttons, view_deltas=view_deltas, weapon=weapon_ids)


def align_ticks_to_ordered_substeps(
    *,
    timestamps: Sequence[float],
    held_buttons: Sequence[Iterable[str] | None],
    delta_pitch: Sequence[float],
    delta_yaw: Sequence[float],
    start_frame: int,
    num_frames: int,
    source_fps: float,
    max_tick_gap_seconds: float,
    stride: int = SOURCE_FRAMES_PER_VIDEO_FRAME,
    num_substeps: int = SUBSTEPS_PER_VIDEO_FRAME,
    buttons: Sequence[str] = CONTROL_BUTTONS,
) -> SubstepControls:
    """Split each frame's ticks into ordered substeps.

    A tick goes to the substep of its time within the frame interval (:func:`_substeps`); per
    substep the buttons are OR'd and the turns divided by :data:`CAMERA_DELTA_SCALE` are summed,
    tick by tick. The defaults are the players' controls of the generator (video frames, four
    substeps, :data:`CONTROL_BUTTONS`); :func:`state_model_controls` reads latent frames.

    Args:
        timestamps (Sequence[float]): ``[n]`` tick times, seconds.
        held_buttons (Sequence[Iterable[str] | None]): ``[n]`` the names of the buttons held at
            each tick (the recordings' ``active`` column).
        delta_pitch (Sequence[float]): ``[n]`` pitch turn per tick, degrees.
        delta_yaw (Sequence[float]): ``[n]`` yaw turn per tick, degrees.
        start_frame (int): source frame of frame 0.
        num_frames (int): frames ``F``.
        source_fps (float): source frames per second.
        max_tick_gap_seconds (float): largest tolerated tick gap and frame-endpoint lag, seconds.
        stride (int): source frames per frame.
        num_substeps (int): substeps ``S`` per frame.
        buttons (Sequence[str]): the ``B`` button channels, names of :data:`OPENCS2_BUTTONS`.

    Returns:
        SubstepControls: ``values`` ``[F, S, B + 2]`` and ``valid`` ``[F, S]``.

    Raises:
        AlignmentError: see :func:`frame_tick_ranges`; also for an unknown button.
    """
    timestamps, delta_pitch, delta_yaw = _tick_columns(
        timestamps, delta_pitch, delta_yaw, held_buttons
    )
    begin, end = frame_tick_ranges(
        timestamps=timestamps,
        start_frame=start_frame,
        num_frames=num_frames,
        source_fps=source_fps,
        stride=stride,
        max_tick_gap_seconds=max_tick_gap_seconds,
    )

    channel_of = {name: channel for channel, name in enumerate(buttons)}
    values = np.zeros((num_frames, num_substeps, len(channel_of) + 2), dtype=np.float32)
    valid = np.zeros((num_frames, num_substeps), dtype=np.bool_)
    unknown = set()
    interval = float(stride) / float(source_fps)
    frame_times = _frame_times(start_frame, num_frames, stride, source_fps)
    # float32 throughout: a substep that holds two ticks (ticks off the 64 Hz grid) adds their
    # turns in float32, as trained
    scale = np.float32(CAMERA_DELTA_SCALE)
    for frame, (lo, hi) in enumerate(zip(begin.tolist(), end.tolist())):
        bins = _substeps(timestamps[lo:hi], frame_times[frame], interval, num_substeps)
        for row_index, substep in enumerate(bins.tolist(), start=lo):
            valid[frame, substep] = True
            held = _button_channels(held_buttons[row_index], channel_of, unknown)
            values[frame, substep, held] = 1.0
            values[frame, substep, -2] += delta_pitch[row_index] / scale
            values[frame, substep, -1] += delta_yaw[row_index] / scale

    if unknown:
        raise AlignmentError(f"unknown OpenCS2 buttons: {sorted(unknown)}")
    return SubstepControls(values=values, valid=valid)


def state_model_controls(
    *,
    timestamps: Sequence[float],
    held_buttons: Sequence[Iterable[str] | None],
    delta_pitch: Sequence[float],
    delta_yaw: Sequence[float],
    start_frame: int,
    source_fps: float,
    latent_frames: int,
) -> np.ndarray:
    """The state model's controls of one window: per latent frame 16 ordered substeps
    (:func:`align_ticks_to_ordered_substeps`), each the 13 OpenCS2 buttons, the pitch and yaw turns
    divided by :data:`CAMERA_DELTA_SCALE` and the valid flag.

    Args:
        timestamps (Sequence[float]): ``[n]`` tick times of the player's recorded tick stream,
            seconds.
        held_buttons (Sequence[Iterable[str] | None]): ``[n]`` its recorded buttons, without jump
            recall (as the state model was trained).
        delta_pitch (Sequence[float]): ``[n]`` pitch turn per tick, degrees.
        delta_yaw (Sequence[float]): ``[n]`` yaw turn per tick, degrees.
        start_frame (int): source frame (32 fps) of the window's first latent frame.
        source_fps (float): source frames per second.
        latent_frames (int): latent frames of the window.

    Returns:
        np.ndarray: ``[latent_frames, 16, 16]`` float32.

    Raises:
        AlignmentError: a latent frame without fresh ticks, or an unknown button in the window.
    """
    substeps = align_ticks_to_ordered_substeps(
        timestamps=timestamps,
        held_buttons=held_buttons,
        delta_pitch=delta_pitch,
        delta_yaw=delta_yaw,
        start_frame=start_frame,
        num_frames=latent_frames,
        source_fps=source_fps,
        max_tick_gap_seconds=STATE_MODEL_MAX_TICK_GAP_SECONDS,
        stride=SOURCE_FRAMES_PER_LATENT,
        num_substeps=STATE_MODEL_SUBSTEPS,
        buttons=OPENCS2_BUTTONS,
    )
    return np.concatenate([substeps.values, substeps.valid[..., None]], axis=-1, dtype=np.float32)
