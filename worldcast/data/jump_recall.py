"""Jump recall: the ``jump`` button of a tick table, repaired from the player's height.

A ``+jump`` bound to the mouse wheel presses and releases inside one tick, so the recorded button
often misses it (it never fires falsely). Recall keeps every recorded bit and adds a run of ``jump``
ticks at each takeoff that has none nearby. A takeoff starts a free-flight segment (vertical
acceleration = -sv_gravity for >= 5 ticks) whose first rise exceeds 3 units/tick (an impulse, not
a step-off). The run's onset offset and length are drawn from the measured histograms below by
inverse CDF, keyed by sha256("<ticks sha256>|<takeoff tick>"), so one tick file always yields the
same labels.
"""

import hashlib
from collections.abc import Sequence

import numpy as np

from .controls import AlignmentError
from .game import GRAVITY_U_PER_S2, TICK_RATE

__all__ = [
    "AIRBORNE_GRAVITY_TOLERANCE",
    "AIRBORNE_MIN_TICKS",
    "AIRBORNE_TAKEOFF_MIN_RISE",
    "GRAVITY_U_PER_TICK2",
    "JUMP_RECALL_ONSET_OFFSETS",
    "JUMP_RECALL_RUN_TICKS",
    "airborne_takeoff_starts",
    "airborne_tick_mask",
    "apply_jump_recall",
    "takeoff_draws",
]

#: Gravity of the engine in units per tick^2, and its tolerance.
GRAVITY_U_PER_TICK2 = GRAVITY_U_PER_S2 / TICK_RATE**2
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


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of every run of True in a 1-D bool array."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def airborne_tick_mask(heights: Sequence[float]) -> np.ndarray:
    """``[n]`` bool, True on ticks in free flight, from the origin height ``z`` (engine units).

    A run of at least :data:`AIRBORNE_MIN_TICKS` second differences equal to ``-800 / 64^2`` marks
    the run plus its two trailing ticks. Non-finite heights raise
    :class:`~worldcast.data.controls.AlignmentError`.
    """
    z = np.asarray(heights, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise AlignmentError("tick heights must be finite to derive the airborne signal")
    airborne = np.zeros(len(z), dtype=np.bool_)
    if len(z) < AIRBORNE_MIN_TICKS + 3:
        return airborne
    free = np.abs(np.diff(z, n=2) + GRAVITY_U_PER_TICK2) < AIRBORNE_GRAVITY_TOLERANCE
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


def takeoff_draws(key: str, takeoff: int) -> tuple[float, float]:
    """The two uniforms in [0, 1) of the takeoff at tick row ``takeoff`` (the run's onset offset
    and its length): the first two big-endian 4-byte words of ``sha256(f"{key}|{takeoff}")``, over
    2^32."""
    digest = hashlib.sha256(f"{key}|{int(takeoff)}".encode()).digest()
    onset, length = (int.from_bytes(digest[4 * i : 4 * i + 4], "big") / 2**32 for i in range(2))
    return onset, length


def apply_jump_recall(
    held_buttons: Sequence[list[str] | None], heights: Sequence[float], *, key: str
) -> list[list[str] | None]:
    """Copy of the per-tick ``active`` column with recalled ``jump`` ticks added.

    Args:
        held_buttons (Sequence[list[str] | None]): ``[n]`` held button names per tick.
        heights (Sequence[float]): ``[n]`` origin ``z`` per tick, engine units.
        key (str): the tick file's sha256 (hex), which seeds the draws.

    Returns:
        list[list[str] | None]: rows without a recalled jump are the same objects; a recalled row is
        ``list(row) + ["jump"]``.
    """
    if not key:
        raise ValueError("jump recall needs a non-empty per-media key")
    rows = list(held_buttons)
    recorded = np.fromiter(("jump" in (row or ()) for row in rows), dtype=np.bool_, count=len(rows))
    recalled = np.zeros(len(rows), dtype=np.bool_)
    onset_low, onset_high = _JUMP_ONSET_WINDOW
    for start in airborne_takeoff_starts(heights).tolist():
        if recorded[max(0, start + onset_low) : min(len(rows), start + onset_high)].any():
            continue
        offset_unit, length_unit = takeoff_draws(key, start)
        offset = _inverse_cdf(JUMP_RECALL_ONSET_OFFSETS, offset_unit)
        length = _inverse_cdf(JUMP_RECALL_RUN_TICKS, length_unit)
        begin = min(max(0, start + offset), len(rows) - 1)
        recalled[begin : begin + length] = True
    for index in np.flatnonzero(recalled & ~recorded).tolist():
        row = rows[index]
        rows[index] = ([] if row is None else list(row)) + ["jump"]
    return rows
