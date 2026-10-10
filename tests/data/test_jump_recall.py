"""Jump recall: a takeoff the buttons missed gets its jump."""

import numpy as np
import pytest

from worldcast.data.jump_recall import (
    GRAVITY_U_PER_TICK2,
    airborne_takeoff_starts,
    apply_jump_recall,
)


def _jump_heights(takeoff: int, ticks: int = 120) -> np.ndarray:
    """Heights of a player that stands, jumps at tick ``takeoff`` (4.1 u/tick up) and lands."""
    z = np.zeros(ticks)
    velocity, height = 4.1, 0.0
    for i in range(takeoff, ticks):
        height += velocity
        velocity -= GRAVITY_U_PER_TICK2
        if height <= 0.0:
            break
        z[i] = height
    return z


def test_jump_recall_adds_a_jump_at_a_takeoff_the_buttons_missed():
    heights = _jump_heights(40)
    assert GRAVITY_U_PER_TICK2 == 800.0 / 64.0**2
    # the last tick on the ground starts the free flight
    assert airborne_takeoff_starts(heights).tolist() == [39]
    rows = [["forward"]] * len(heights)
    recalled = apply_jump_recall(rows, heights, key="a" * 64)
    jumps = [i for i, row in enumerate(recalled) if "jump" in row]
    # the run this key draws from the measured histograms: onset offset 0, eight ticks
    assert jumps == [39, 40, 41, 42, 43, 44, 45, 46]
    assert all(row[0] == "forward" for row in recalled)
    assert apply_jump_recall(rows, heights, key="b" * 64) != recalled  # another file, another run
    assert recalled == apply_jump_recall(rows, heights, key="a" * 64)  # keyed by the file digest
    kept = [row for i, row in enumerate(recalled) if i not in jumps]
    assert all(row is rows[0] for row in kept)  # the other rows are the same objects


def test_jump_recall_keeps_a_recorded_jump():
    heights = _jump_heights(40)
    rows = [["forward"]] * len(heights)
    rows[39] = ["forward", "jump"]
    assert apply_jump_recall(rows, heights, key="a" * 64) == rows
    with pytest.raises(ValueError):
        apply_jump_recall(rows, heights, key="")
