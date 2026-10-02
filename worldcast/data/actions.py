"""Controls from 64 Hz tick tables: buttons, camera turn, weapon, substeps, state-model controls.

With ``dt = skip_frame / video_fps``, output frame ``f`` of a window starting at source frame ``s``
sits at ``t_f = s / video_fps + f dt`` and owns the ticks in ``(t_f - dt, t_f]``. The paper reads
the 32 fps recordings with ``skip_frame = 2``: frames are 16 fps and own four ticks each.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

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
_OPENCS2_INDEX = MappingProxyType({name: index for index, name in enumerate(OPENCS2_BUTTONS)})
#: Keyboard-turn binds; the recorded camera deltas already hold the turn, so they are skipped.
IGNORED_ACTIVE_ACTIONS = frozenset(("turn_left", "turn_right"))
#: The 11 button channels of the model, in order (config ``data.action_buttons``).
PAPER_ACTION_BUTTONS = (
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

#: Camera mu-law: turns in degrees, +-20 degree range, 0.5 degree buckets, mu = 2.7.
CAMERA_MAX_DEGREES = 20.0
CAMERA_BIN_DEGREES = 0.5
CAMERA_MU = 2.7
#: ``noclip`` (the paper) keeps per-frame turns beyond +-20 degrees; ``clip`` cuts them.
PAPER_CAMERA_ENCODING = "noclip"
CAMERA_ENCODINGS = ("noclip", "clip")
_UNIMPLEMENTED_CAMERA_ENCODINGS = ("carry", "hybrid", "spread", "exact")

#: State-model controls: per latent frame (8 source frames) 16 substeps, each the 13 OpenCS2
#: buttons, the pitch and yaw deltas divided by 5 and a valid flag. A latent frame needs a tick
#: within 1.5 ticks of its end.
STATE_MODEL_SUBSTEPS = 16
STATE_MODEL_CONTROL_DIM = len(OPENCS2_BUTTONS) + 3
_STATE_MODEL_SOURCE_FRAMES = 8
_STATE_MODEL_DELTA_SCALE = 5.0
_STATE_MODEL_MAX_TICK_GAP_S = 1.5 / 64.0


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


def resolve_button_indexer(button_names: Sequence[str]) -> dict[str, int]:
    """``name -> channel`` for a non-empty, duplicate-free subset of :data:`OPENCS2_BUTTONS`."""
    names = tuple(button_names)
    if not names:
        raise ValueError("button_names must be non-empty")
    unknown = [name for name in names if name not in _OPENCS2_INDEX]
    if unknown:
        raise ValueError(f"button_names has non-OpenCS2 buttons: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("button_names must not contain duplicates")
    return {name: index for index, name in enumerate(names)}


def check_camera_encoding(camera_encoding: str) -> str:
    """Return ``camera_encoding`` if it is one of :data:`CAMERA_ENCODINGS`; raise otherwise."""
    if camera_encoding in CAMERA_ENCODINGS:
        return camera_encoding
    if camera_encoding in _UNIMPLEMENTED_CAMERA_ENCODINGS:
        raise NotImplementedError(
            f"camera_encoding {camera_encoding!r} is not implemented (the paper uses 'noclip')"
        )
    raise ValueError(f"camera_encoding must be one of {CAMERA_ENCODINGS}, got {camera_encoding!r}")


def quantize_camera_delta(camera: Sequence[float], *, clip: bool) -> np.ndarray:
    """mu-law encode one frame's summed turn ``[pitch, yaw]`` (degrees) to ``[2]`` float32.

    ``v = sign(x/20) log1p(2.7 |x/20|) / log1p(2.7)``, rounded to 0.5 degree buckets, in float32. A
    turn within +-20 degrees encodes to [-1, 1]; without ``clip`` a larger turn gives ``|v| > 1``.
    """
    camera = np.asarray(camera, dtype=np.float32)
    if not np.all(np.isfinite(camera)):
        raise ValueError("Camera delta must contain only finite values")
    clipped = np.clip(camera, -CAMERA_MAX_DEGREES, CAMERA_MAX_DEGREES) if clip else camera
    normalized = clipped / CAMERA_MAX_DEGREES
    encoded = np.sign(normalized) * (np.log1p(CAMERA_MU * np.abs(normalized)) / np.log1p(CAMERA_MU))
    encoded = encoded * CAMERA_MAX_DEGREES
    num_buckets = int(CAMERA_MAX_DEGREES / CAMERA_BIN_DEGREES)
    discretized = np.round((encoded + CAMERA_MAX_DEGREES) / CAMERA_BIN_DEGREES).astype(np.int64)
    return ((discretized - num_buckets) / num_buckets).astype(np.float32)


class AlignmentError(ValueError):
    """A tick table cannot serve the requested window; ``reason`` is a stable short code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class FrameActions:
    """Per-frame controls of one player.

    Attributes:
        buttons (np.ndarray): ``[F, B]`` float32 in {0, 1}.
        camera (np.ndarray): ``[F, 2]`` float32 mu-law ``[pitch, yaw]`` turn.
        weapon (np.ndarray): ``[F]`` int64 weapon id held at the frame's last tick.
    """

    buttons: np.ndarray
    camera: np.ndarray
    weapon: np.ndarray


@dataclass(frozen=True)
class SubstepActions:
    """Per-frame controls split into ordered substeps.

    Attributes:
        values (np.ndarray): ``[F, S, B + 2]`` float32, per substep the held buttons and the summed
            ``delta / camera_delta_scale`` of pitch and yaw (unquantised).
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
        raise AlignmentError(
            "invalid_timestamp_shape", f"tick timestamps must be 1-D, got {timestamps.shape}"
        )
    if not len(timestamps):
        raise AlignmentError("empty_ticks", "tick table is empty")
    if not np.all(np.isfinite(timestamps)):
        raise AlignmentError("nonfinite_timestamps", "tick timestamps must be finite")
    if np.any(np.diff(timestamps) <= 0):
        raise AlignmentError(
            "nonmonotonic_timestamps", "tick timestamps must be strictly increasing"
        )
    delta_pitch = np.asarray(delta_pitch, dtype=np.float32)
    delta_yaw = np.asarray(delta_yaw, dtype=np.float32)
    if len({len(timestamps), len(delta_pitch), len(delta_yaw), *map(len, other_columns)}) != 1:
        raise AlignmentError("inconsistent_tick_columns", "tick columns have inconsistent lengths")
    if delta_pitch.ndim != 1 or delta_yaw.ndim != 1:
        raise AlignmentError("invalid_camera_shape", "tick camera columns must be 1-D")
    if not np.all(np.isfinite(delta_pitch)) or not np.all(np.isfinite(delta_yaw)):
        raise AlignmentError("nonfinite_camera", "tick camera columns must be finite")
    return timestamps, delta_pitch, delta_yaw


def _button_channels(row: Iterable[str] | None, indexer: dict[str, int], unknown: set) -> list[int]:
    """Channels of the buttons held in one tick; names outside the enum go to ``unknown``."""
    channels = []
    for action in row or ():
        name = str(action)
        if name in IGNORED_ACTIVE_ACTIONS:
            continue
        if name not in _OPENCS2_INDEX:
            unknown.add("<null>" if action is None else name)
        elif name in indexer:
            channels.append(indexer[name])
    return channels


def _frame_times(
    start_frame: int, num_frames: int, skip_frame: int, video_fps: float
) -> np.ndarray:
    frames = int(start_frame) + np.arange(num_frames, dtype=np.float64) * int(skip_frame)
    return frames / float(video_fps)


def _substeps(timestamps: np.ndarray, frame_time: float, interval: float, num_substeps: int):
    relative = (timestamps - (frame_time - interval)) / interval
    return np.clip(np.ceil(relative * num_substeps).astype(np.int64) - 1, 0, num_substeps - 1)


def frame_tick_ranges(
    *,
    timestamps: np.ndarray,
    start_frame: int,
    num_frames: int,
    video_fps: float,
    max_tick_gap_seconds: float,
    skip_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tick rows ``[begin[f], end[f])`` of every output frame.

    Args:
        timestamps (np.ndarray): ``[n]`` float64 tick times, seconds, strictly increasing.
        start_frame (int): source frame of output frame 0.
        num_frames (int): output frames.
        video_fps (float): source frame rate.
        max_tick_gap_seconds (float): largest tolerated tick gap and frame-endpoint lag, seconds.
        skip_frame (int): source frames per output frame.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(begin, end)``, each ``[num_frames]`` int64.

    Raises:
        AlignmentError: a frame has no tick (``frame_without_ticks``), its last tick lags it by more
            than ``max_tick_gap_seconds`` (``endpoint_uncovered``), or two ticks of the window are
            further apart (``tick_gap``).
    """
    if start_frame < 0 or num_frames <= 0 or skip_frame <= 0 or video_fps <= 0:
        raise ValueError("Invalid frame window")
    if max_tick_gap_seconds <= 0:
        raise ValueError("max_tick_gap_seconds must be positive")

    frame_times = _frame_times(start_frame, num_frames, skip_frame, video_fps)
    begin = np.searchsorted(timestamps, frame_times - skip_frame / video_fps, side="right")
    end = np.searchsorted(timestamps, frame_times + 1e-7, side="right")
    if np.any(end <= begin):
        bad = int(np.flatnonzero(end <= begin)[0])
        raise AlignmentError("frame_without_ticks", f"No tick rows align to output frame {bad}")

    endpoint_lag = frame_times - timestamps[end - 1]
    if np.any(endpoint_lag > max_tick_gap_seconds):
        bad = int(np.flatnonzero(endpoint_lag > max_tick_gap_seconds)[0])
        raise AlignmentError(
            "endpoint_uncovered",
            f"Tick gap leaves output frame {bad} endpoint uncovered by"
            f" {float(endpoint_lag[bad]):.6f}s",
        )

    gaps = np.diff(timestamps[max(0, int(begin[0]) - 1) : int(end[-1])])
    if gaps.size and float(gaps.max()) > max_tick_gap_seconds:
        raise AlignmentError(
            "tick_gap",
            f"Tick gap {float(gaps.max()):.6f}s exceeds {max_tick_gap_seconds:.6f}s inside the"
            " requested window",
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
    active_actions: Sequence[Iterable[str] | None],
    delta_pitch: Sequence[float],
    delta_yaw: Sequence[float],
    input_weapons: Sequence[object],
    start_frame: int,
    num_frames: int,
    video_fps: float,
    skip_frame: int,
    max_tick_gap_seconds: float,
    button_names: Sequence[str],
    camera_encoding: str = PAPER_CAMERA_ENCODING,
) -> FrameActions:
    """Aggregate one player's ticks into per-frame controls (the client's own control input).

    Per frame (:func:`frame_tick_ranges`) the buttons are OR'd, the turn is summed in float32 and
    encoded by :func:`quantize_camera_delta`, and the weapon is the one held at the last tick.

    Args:
        timestamps, active_actions, delta_pitch, delta_yaw, input_weapons: the tick columns, ``[n]``
            each (seconds, held button names, degrees, degrees, raw weapon names).
        start_frame, num_frames, video_fps, skip_frame, max_tick_gap_seconds: the window, as in
            :func:`frame_tick_ranges`.
        button_names (Sequence[str]): the button channels (config ``data.action_buttons``).
        camera_encoding (str): ``noclip`` (the paper) or ``clip``.

    Returns:
        FrameActions: ``num_frames`` frames.

    Raises:
        AlignmentError: see :func:`frame_tick_ranges`; ``unknown_active`` for an unknown button.
    """
    indexer = resolve_button_indexer(button_names)
    clip = check_camera_encoding(camera_encoding) == "clip"
    timestamps, delta_pitch, delta_yaw = _tick_columns(
        timestamps, delta_pitch, delta_yaw, active_actions, input_weapons
    )
    begin, end = frame_tick_ranges(
        timestamps=timestamps,
        start_frame=start_frame,
        num_frames=num_frames,
        video_fps=video_fps,
        max_tick_gap_seconds=max_tick_gap_seconds,
        skip_frame=skip_frame,
    )

    unknown = set()
    buttons = np.zeros((num_frames, len(indexer)), dtype=np.float32)
    camera = np.zeros((num_frames, 2), dtype=np.float32)
    weapon_ids = np.zeros((num_frames,), dtype=np.int64)
    filled_weapons = _forward_fill_weapons(input_weapons)
    for frame, (lo, hi) in enumerate(zip(begin.tolist(), end.tolist())):
        for row in active_actions[lo:hi]:
            buttons[frame, _button_channels(row, indexer, unknown)] = 1.0
        # Summed per frame in float32, one frame at a time: this op order is part of the numbers.
        raw_turn = np.asarray([delta_pitch[lo:hi].sum(), delta_yaw[lo:hi].sum()], dtype=np.float32)
        camera[frame] = quantize_camera_delta(raw_turn, clip=clip)
        weapon_ids[frame] = encode_weapon(filled_weapons[hi - 1])

    if unknown:
        raise AlignmentError("unknown_active", f"Unknown OpenCS2 active actions: {sorted(unknown)}")
    return FrameActions(buttons=buttons, camera=camera, weapon=weapon_ids)


def align_ticks_to_ordered_substeps(
    *,
    timestamps: Sequence[float],
    active_actions: Sequence[Iterable[str] | None],
    delta_pitch: Sequence[float],
    delta_yaw: Sequence[float],
    start_frame: int,
    num_frames: int,
    video_fps: float,
    skip_frame: int,
    max_tick_gap_seconds: float,
    button_names: Sequence[str],
    num_substeps: int,
    camera_delta_scale: float,
) -> SubstepActions:
    """Split each frame's ticks into ``num_substeps`` ordered substeps (the peers' controls).

    Tick ``t`` of the frame interval ``(left, left + interval]`` goes to substep
    ``clip(ceil((t - left) / interval * S) - 1, 0, S - 1)``; per substep the buttons are OR'd and
    ``delta / camera_delta_scale`` is summed. The paper uses S = 4 and scale 5 (config
    ``model.player_field.*``). The other arguments are those of :func:`align_ticks_to_video_frames`.
    """
    indexer = resolve_button_indexer(button_names)
    timestamps, delta_pitch, delta_yaw = _tick_columns(
        timestamps, delta_pitch, delta_yaw, active_actions
    )
    begin, end = frame_tick_ranges(
        timestamps=timestamps,
        start_frame=start_frame,
        num_frames=num_frames,
        video_fps=video_fps,
        max_tick_gap_seconds=max_tick_gap_seconds,
        skip_frame=skip_frame,
    )

    values = np.zeros((num_frames, num_substeps, len(indexer) + 2), dtype=np.float32)
    valid = np.zeros((num_frames, num_substeps), dtype=np.bool_)
    unknown = set()
    interval = float(skip_frame) / float(video_fps)
    frame_times = _frame_times(start_frame, num_frames, skip_frame, video_fps)
    for frame, (lo, hi) in enumerate(zip(begin.tolist(), end.tolist())):
        bins = _substeps(timestamps[lo:hi], frame_times[frame], interval, num_substeps)
        for row_index, substep in enumerate(bins.tolist(), start=lo):
            valid[frame, substep] = True
            values[
                frame, substep, _button_channels(active_actions[row_index], indexer, unknown)
            ] = 1.0
            values[frame, substep, -2] += delta_pitch[row_index] / float(camera_delta_scale)
            values[frame, substep, -1] += delta_yaw[row_index] / float(camera_delta_scale)

    if unknown:
        raise AlignmentError("unknown_active", f"Unknown OpenCS2 active actions: {sorted(unknown)}")
    return SubstepActions(values=values, valid=valid)


# ----------------------------------------------------------------------------- state-model controls
@dataclass(frozen=True)
class ControlTicks:
    """One player's raw tick stream, as the state model reads it.

    Attributes:
        t (np.ndarray): ``[n]`` float64 tick times, seconds.
        buttons (np.ndarray): ``[n, 13]`` float32 held buttons, OpenCS2 order.
        delta_pitch (np.ndarray): ``[n]`` float32 pitch turn per tick, degrees.
        delta_yaw (np.ndarray): ``[n]`` float32 yaw turn per tick, degrees.
        unknown_rows (np.ndarray): ``[k]`` int64 sorted rows holding a button name outside the enum.
    """

    t: np.ndarray
    buttons: np.ndarray
    delta_pitch: np.ndarray
    delta_yaw: np.ndarray
    unknown_rows: np.ndarray


def read_control_ticks(path: str | Path) -> ControlTicks:
    """Read a ticks parquet without jump recall (the state model was trained on the raw stream)."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=["t", "active", "delta_pitch", "delta_yaw"])
    active = table.column("active").to_pylist()
    buttons = np.zeros((len(active), len(OPENCS2_BUTTONS)), np.float32)
    unknown_rows = []
    for row, names in enumerate(active):
        unknown = set()
        buttons[row, _button_channels(names, _OPENCS2_INDEX, unknown)] = 1.0
        if unknown:
            unknown_rows.append(row)
    return ControlTicks(
        t=table.column("t").to_numpy().astype(np.float64),
        buttons=buttons,
        delta_pitch=table.column("delta_pitch").to_numpy().astype(np.float32),
        delta_yaw=table.column("delta_yaw").to_numpy().astype(np.float32),
        unknown_rows=np.asarray(unknown_rows, np.int64),
    )


def state_model_controls(
    ticks: ControlTicks, start_frame: int, fps: float, num_latents: int
) -> np.ndarray:
    """The state model's control input of one window.

    Per substep (placed by tick time, :func:`align_ticks_to_ordered_substeps`) the held buttons are
    OR'd, the pitch and yaw deltas divided by 5 are summed, and the valid flag is set.

    Args:
        ticks (ControlTicks): the player's raw tick stream.
        start_frame (int): source frame (32 fps) of the window's first latent frame.
        fps (float): source frame rate.
        num_latents (int): latent frames of the window.

    Returns:
        np.ndarray: ``[num_latents, 16, 16]`` float32.

    Raises:
        ValueError: a latent frame without fresh ticks, or an unknown button in the window.
    """
    begin, end = frame_tick_ranges(
        timestamps=ticks.t,
        start_frame=start_frame,
        num_frames=num_latents,
        video_fps=fps,
        max_tick_gap_seconds=_STATE_MODEL_MAX_TICK_GAP_S,
        skip_frame=_STATE_MODEL_SOURCE_FRAMES,
    )
    rows = ticks.unknown_rows
    if np.any(np.searchsorted(rows, end) > np.searchsorted(rows, begin)):
        raise ValueError("unknown OpenCS2 action in the window")
    nb = len(OPENCS2_BUTTONS)
    span = _STATE_MODEL_SOURCE_FRAMES / float(fps)
    frame_times = _frame_times(start_frame, num_latents, _STATE_MODEL_SOURCE_FRAMES, fps)
    u = np.zeros((num_latents, STATE_MODEL_SUBSTEPS, STATE_MODEL_CONTROL_DIM), np.float32)
    for f in range(num_latents):
        lo, hi = int(begin[f]), int(end[f])
        sub = _substeps(ticks.t[lo:hi], frame_times[f], span, STATE_MODEL_SUBSTEPS)
        np.maximum.at(u[f, :, :nb], sub, ticks.buttons[lo:hi])
        np.add.at(u[f, :, nb], sub, ticks.delta_pitch[lo:hi] / _STATE_MODEL_DELTA_SCALE)
        np.add.at(u[f, :, nb + 1], sub, ticks.delta_yaw[lo:hi] / _STATE_MODEL_DELTA_SCALE)
        u[f, sub, nb + 2] = 1.0
    return u
