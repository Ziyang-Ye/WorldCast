"""A block's controls."""

import numpy as np
import pytest

from worldcast.engine.inference.controls import BlockControls


def _controls(rows: int = 16, buttons: int = 11) -> BlockControls:
    return BlockControls(
        buttons=np.zeros((rows, buttons)), view_deltas=np.zeros((rows, 2)), weapon=np.zeros(rows)
    )


def test_a_blocks_controls_are_16_rows_of_11_buttons():
    controls = _controls()
    assert controls.buttons.dtype == controls.view_deltas.dtype == np.float32
    assert controls.weapon.dtype == np.int64
    controls.check_block()
    for other in (_controls(rows=4), _controls(buttons=10)):
        with pytest.raises(ValueError, match="16 rows"):
            other.check_block()


def test_the_controls_are_the_generators_three_control_conditions():
    controls = BlockControls(
        buttons=np.ones((16, 11)), view_deltas=np.full((16, 2), 0.5), weapon=np.full(16, 7)
    )
    conditions = controls.conditions()
    assert list(conditions) == ["buttons", "view_deltas", "weapon"]
    assert conditions["buttons"] is controls.buttons
    assert conditions["view_deltas"] is controls.view_deltas
    assert conditions["weapon"] is controls.weapon


def test_controls_must_be_consistent():
    with pytest.raises(ValueError, match="controls must be"):
        BlockControls(
            buttons=np.zeros((16, 11)), view_deltas=np.zeros((15, 2)), weapon=np.zeros(16)
        )
    with pytest.raises(ValueError, match="controls must be"):
        BlockControls(buttons=np.zeros((16, 11)), view_deltas=np.zeros((16, 2)), weapon=np.zeros(4))
    for weapon in (-1, 52):
        with pytest.raises(ValueError, match="weapon ids must lie in \\[0, 52\\)"):
            BlockControls(
                buttons=np.zeros((16, 11)),
                view_deltas=np.zeros((16, 2)),
                weapon=np.full(16, weapon),
            )
