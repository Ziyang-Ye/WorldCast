"""Per-player tick tables: read, verify and repair one OpenCS2 ``ticks.parquet``.

The file size must equal the media row's ``ticks_file_size`` and, when the row records
``ticks_sha256``, the file's sha256 must equal it: the digest keys the jump-recall draws
(:func:`apply_jump_recall`), so a re-exported parquet with other bytes would change the controls.
The mtime is not checked (a plain copy changes it). Formats: docs/data.md.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import AlignmentError
from .media import MediaRecord

#: Engine tick rate of OpenCS2 captures (Hz).
TICK_RATE = 64.0
#: Parquet columns read (the files hold more; see docs/data.md, "Formats").
TICK_COLUMNS = (
    "tick",
    "t",
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "is_alive",
    "active",
    "delta_pitch",
    "delta_yaw",
    "input_weapon",
    "team_num",
)

# Jump recall. A `+jump` bound to the mouse wheel presses and releases inside one tick, so the
# recorded button often misses it (it never fires falsely). Recall keeps every recorded bit and adds
# a run of `jump` ticks at each takeoff that has none nearby. A takeoff starts a free-flight segment
# (vertical acceleration = -sv_gravity for >= 5 ticks) whose first rise exceeds 3 units/tick (an
# impulse, not a step-off). The run's onset offset and length are drawn from the measured histograms
# below by inverse CDF, keyed by sha256("<ticks sha256>|<takeoff tick>"), so one tick file always
# yields the same labels.

#: Gravity of the engine (sv_gravity = 800 u/s^2) in units per tick^2, and its tolerance.
CS2_GRAVITY_UNITS_PER_TICK2 = 800.0 / (64.0**2)
AIRBORNE_GRAVITY_TOLERANCE = 0.06
#: Shortest free-flight run, and the smallest first rise (units/tick) of a jump takeoff.
AIRBORNE_MIN_TICKS = 5
AIRBORNE_TAKEOFF_MIN_RISE = 3.0
#: (onset offset in ticks from the first airborne tick, count), measured on real +jump runs.
JUMP_RECALL_ONSET_OFFSETS = (
    (-6, 32),
    (-5, 33),
    (-4, 30),
    (-3, 35),
    (-2, 47),
    (-1, 65),
    (0, 1009),
    (1, 47),
    (2, 1),
    (3, 2),
    (4, 1),
    (5, 1),
    (6, 1),
    (8, 1),
    (9, 1),
)
#: (run length in ticks, count), measured on real +jump runs (truncated at 39 ticks).
JUMP_RECALL_RUN_TICKS = (
    (1, 19),
    (2, 13),
    (3, 8),
    (4, 38),
    (5, 69),
    (6, 143),
    (7, 166),
    (8, 168),
    (9, 162),
    (10, 130),
    (11, 97),
    (12, 74),
    (13, 59),
    (14, 28),
    (15, 25),
    (16, 33),
    (17, 15),
    (18, 5),
    (19, 14),
    (20, 9),
    (21, 8),
    (22, 6),
    (23, 9),
    (24, 4),
    (25, 4),
    (26, 5),
    (27, 5),
    (28, 6),
    (29, 6),
    (30, 1),
    (31, 3),
    (32, 4),
    (33, 2),
    (34, 1),
    (36, 2),
    (37, 1),
    (38, 3),
    (39, 1),
)
#: A takeoff with a recorded jump bit in ``[start - 6, start + 12)`` is left alone.
_JUMP_ONSET_WINDOW = (-6, 12)


@dataclass(frozen=True)
class TickTable:
    """One player's ticks, ``n`` rows, with the columns of docs/data.md ("Formats").

    Float columns are float64 arrays holding the parquet's float32 values exactly; ``tick`` and
    ``team_num`` are int64, ``is_alive`` bool. ``active`` (lists of held buttons) has jump recall
    applied, keyed by ``sha256``, the file digest.
    """

    media_id: str
    sha256: str
    tick: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    is_alive: np.ndarray
    active: list[list | None]
    delta_pitch: np.ndarray
    delta_yaw: np.ndarray
    input_weapon: list[str | None]
    team_num: np.ndarray

    def __len__(self) -> int:
        return int(self.t.shape[0])


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_ticks_file(media: MediaRecord, ticks_path: Path) -> str:
    """Check size (and sha256 when recorded) of a tick parquet; return its sha256 (hex).

    Raises:
        FileNotFoundError: the file is missing.
        RuntimeError: a size or digest mismatch, or a malformed recorded digest.
    """
    ticks_path = Path(ticks_path)
    if not ticks_path.is_file():
        raise FileNotFoundError(f"tick table of media {media.media_id} not found: {ticks_path}")
    size = ticks_path.stat().st_size
    if size != media.ticks_file_size:
        raise RuntimeError(
            f"tick table of media {media.media_id} has {size} bytes, the media index says "
            f"{media.ticks_file_size}: {ticks_path}"
        )
    digest = sha256_file(ticks_path)
    expected = media.ticks_sha256
    if expected is None:
        return digest
    if len(expected) != 64 or not all(c in "0123456789abcdefABCDEF" for c in expected):
        raise RuntimeError(f"media {media.media_id} carries a malformed ticks_sha256 {expected!r}")
    if digest != expected:
        raise RuntimeError(
            f"tick table of media {media.media_id} does not match the media index sha256 "
            f"(file {digest}, index {expected}): {ticks_path}"
        )
    return expected


def read_player_ticks(media: MediaRecord, dataset_root: str | Path) -> TickTable:
    """Read, verify and repair the tick table ``dataset_root / media.ticks_path`` (needs pyarrow).

    Raises:
        RuntimeError: the row count differs from the index, the table is empty, ticks or times are
            not strictly increasing, the time base does not start at 0, or
            ``t != (tick - tick[0]) / 64`` (tolerance 1e-6 s).
    """
    import pyarrow.parquet as pq

    ticks_path = Path(dataset_root) / media.ticks_path
    digest = verify_ticks_file(media, ticks_path)
    table = pq.read_table(ticks_path, columns=list(TICK_COLUMNS)).to_pydict()
    n = len(table["t"])
    if n != media.ticks_rows:
        raise RuntimeError(
            f"tick table has {n} rows, the media index says {media.ticks_rows}: {ticks_path}"
        )
    if n == 0:
        raise RuntimeError(f"tick table is empty: {ticks_path}")
    tick = np.asarray(table["tick"], dtype=np.int64)
    timestamps = np.asarray(table["t"], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise RuntimeError(f"tick timestamps are not strictly increasing in {ticks_path}")
    if np.any(np.diff(tick) <= 0):
        raise RuntimeError(f"tick numbers are not strictly increasing in {ticks_path}")
    if abs(float(timestamps[0])) > 1e-6:
        raise RuntimeError(f"tick time base does not start at 0 in {ticks_path}")
    expected_timestamps = (tick - tick[0]).astype(np.float64) / TICK_RATE
    if not np.allclose(timestamps, expected_timestamps, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"tick and time columns disagree in {ticks_path}")
    if any(value is None for value in table["is_alive"]):
        raise RuntimeError(f"tick table has null is_alive values: {ticks_path}")

    def floats(name: str) -> np.ndarray:
        return np.asarray(table[name], dtype=np.float64)

    return TickTable(
        media_id=media.media_id,
        sha256=digest,
        tick=tick,
        t=timestamps,
        x=floats("x"),
        y=floats("y"),
        z=floats("z"),
        yaw=floats("yaw"),
        pitch=floats("pitch"),
        is_alive=np.asarray(table["is_alive"], dtype=np.bool_),
        active=apply_jump_recall(table["active"], table["z"], key=digest),
        delta_pitch=floats("delta_pitch"),
        delta_yaw=floats("delta_yaw"),
        input_weapon=list(table["input_weapon"]),
        team_num=np.asarray(table["team_num"], dtype=np.int64),
    )


# -------------------------------------------------------------------------------------- jump recall
def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of every run of True in a 1-D bool array."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def airborne_tick_mask(heights: Sequence[float]) -> np.ndarray:
    """``[n]`` bool, True on ticks in free flight, from the origin height ``z`` (engine units).

    A run of at least :data:`AIRBORNE_MIN_TICKS` second differences equal to ``-800 / 64^2`` marks
    the run plus its two trailing ticks. Non-finite heights raise
    :class:`~worldcast.data.actions.AlignmentError` (``nonfinite_height``).
    """
    z = np.asarray(heights, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise AlignmentError(
            "nonfinite_height", "tick heights must be finite to derive the airborne signal"
        )
    airborne = np.zeros(len(z), dtype=np.bool_)
    if len(z) < AIRBORNE_MIN_TICKS + 3:
        return airborne
    free = np.abs(np.diff(z, n=2) + CS2_GRAVITY_UNITS_PER_TICK2) < AIRBORNE_GRAVITY_TOLERANCE
    for start, end in _true_runs(free):
        if end - start >= AIRBORNE_MIN_TICKS:
            airborne[start : end + 2] = True
    return airborne


def airborne_takeoff_starts(heights: Sequence[float]) -> np.ndarray:
    """``[k]`` int64: first tick of every airborne segment whose first rise is a jump impulse."""
    z = np.asarray(heights, dtype=np.float64)
    starts = [
        start
        for start, _ in _true_runs(airborne_tick_mask(z))
        if start + 1 < len(z) and float(z[start + 1] - z[start]) > AIRBORNE_TAKEOFF_MIN_RISE
    ]
    return np.asarray(starts, dtype=np.int64)


def _inverse_cdf(table: Sequence[tuple[int, int]], unit: float) -> int:
    """Value of a ``((value, count), ...)`` histogram at quantile ``unit`` in [0, 1)."""
    target = float(unit) * sum(count for _, count in table)
    cumulative = 0
    for value, count in table:
        cumulative += count
        if target < cumulative:
            return value
    return table[-1][0]


def event_unit_draws(key: str, tick_index: int, draws: int) -> list[float]:
    """``draws`` (<= 8) uniforms in [0, 1) from ``sha256(f"{key}|{tick_index}")``, 4 bytes each."""
    digest = hashlib.sha256(f"{key}|{int(tick_index)}".encode()).digest()
    return [int.from_bytes(digest[4 * i : 4 * i + 4], "big") / 4294967296.0 for i in range(draws)]


def apply_jump_recall(
    active_actions: Sequence[list[str] | None], heights: Sequence[float], *, key: str
) -> list[list[str] | None]:
    """Copy of the per-tick ``active`` column with recalled ``jump`` ticks added.

    Args:
        active_actions (Sequence[list[str] | None]): ``[n]`` held button names per tick.
        heights (Sequence[float]): ``[n]`` origin ``z`` per tick, engine units.
        key (str): the tick file's sha256 (hex), which seeds the draws.

    Returns:
        list[list[str] | None]: rows without a recalled jump are the same objects; a recalled row is
        ``list(row) + ["jump"]``.
    """
    if not key:
        raise ValueError("jump recall needs a non-empty per-media key")
    rows = list(active_actions)
    recorded = np.fromiter(("jump" in (row or ()) for row in rows), dtype=np.bool_, count=len(rows))
    recalled = np.zeros(len(rows), dtype=np.bool_)
    onset_low, onset_high = _JUMP_ONSET_WINDOW
    for start in airborne_takeoff_starts(heights).tolist():
        if recorded[max(0, start + onset_low) : min(len(rows), start + onset_high)].any():
            continue
        offset_unit, length_unit = event_unit_draws(key, start, 2)
        offset = _inverse_cdf(JUMP_RECALL_ONSET_OFFSETS, offset_unit)
        length = _inverse_cdf(JUMP_RECALL_RUN_TICKS, length_unit)
        begin = min(max(0, start + offset), len(rows) - 1)
        recalled[begin : begin + length] = True
    for index in np.flatnonzero(recalled & ~recorded).tolist():
        row = rows[index]
        rows[index] = ([] if row is None else list(row)) + ["jump"]
    return rows
