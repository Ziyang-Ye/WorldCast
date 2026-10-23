"""A block's controls, and those of a player's live input."""

import numpy as np
import pytest

from worldcast.engine.inference.controls import BlockControls, LiveBlockControls


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


def _live(**changed) -> LiveBlockControls:
    fields = dict(
        buttons=np.ones((16, 11)),
        view_deltas=np.full((16, 2), 0.125),
        weapon=np.full(16, 7),
        turn=np.full((16, 2), 1.5),
        times=10.0 + np.arange(16) / 16,
    )
    return LiveBlockControls(**{**fields, **changed})


def test_live_controls_carry_the_raw_turn_and_the_session_times():
    controls = _live()
    assert isinstance(controls, BlockControls)
    controls.check_block()
    assert controls.turn.dtype == np.float32 and controls.turn.tolist() == [[1.5, 1.5]] * 16
    assert controls.times.dtype == np.float64
    assert controls.times[[0, -1]].tolist() == [10.0, 10.9375]
    # the generator reads the same three conditions as from any block's controls
    conditions = controls.conditions()
    assert list(conditions) == ["buttons", "view_deltas", "weapon"]
    assert conditions["buttons"] is controls.buttons
    assert conditions["view_deltas"] is controls.view_deltas
    assert conditions["weapon"] is controls.weapon


def test_live_controls_have_a_turn_and_a_time_per_row():
    for turn, times in [
        (np.zeros((15, 2)), np.zeros(16)),
        (np.zeros((16, 3)), np.zeros(16)),
        (np.zeros((16, 2)), np.zeros((16, 1))),
        (np.zeros((16, 2)), np.zeros(4)),
    ]:
        with pytest.raises(ValueError, match=r"live controls of 16 rows have turn \[16, 2\]"):
            _live(turn=turn, times=times)
    with pytest.raises(ValueError, match="controls must be"):
        _live(view_deltas=np.zeros((15, 2)))
    with pytest.raises(ValueError, match="16 rows"):
        _live(
            buttons=np.ones((4, 11)),
            view_deltas=np.zeros((4, 2)),
            weapon=np.zeros(4),
            turn=np.zeros((4, 2)),
            times=np.zeros(4),
        ).check_block()
