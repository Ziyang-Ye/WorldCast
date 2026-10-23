"""A recording's controls as the blocks of a live session (``RecordedControls``)."""

import numpy as np
import pytest

import tests.engine.inference.support as sw
from worldcast.data.controls import encode_turn
from worldcast.data.latents import video_frames_of
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference.controls import BlockControls, LiveBlockControls
from worldcast.engine.inference.loading import load_window
from worldcast.engine.inference.replay import RecordedControls


@pytest.fixture
def recorded(synthetic, monkeypatch):
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    cfg = sw.config(
        synthetic["world"], synthetic["weights"], synthetic["tmp"] / "out", max_blocks=2
    )
    window = load_window(cfg, load_round_index_row(cfg.paths.round_index, 0))
    return window, RecordedControls(window)


def test_a_block_is_the_recordings_rows_and_the_turn_they_encode(recorded):
    window, controls = recorded
    block = controls.block(5)
    assert isinstance(block, LiveBlockControls) and isinstance(block, BlockControls)
    rows, frames = video_frames_of(5, 4), window.client_frames
    assert np.array_equal(block.buttons, frames.buttons[rows])
    assert np.array_equal(block.view_deltas, frames.view_deltas[rows])
    assert np.array_equal(block.weapon, frames.weapon_ids[rows])
    assert np.array_equal(block.turn, frames.turn[rows]) and block.turn.any()
    assert np.array_equal(block.view_deltas, np.stack([encode_turn(t) for t in block.turn]))
    # the session clock of the video frames, 16 a second
    assert block.times.tolist() == [v / 16.0 for v in rows]


def test_only_the_blocks_of_the_recordings_window_are_played(recorded):
    window, controls = recorded
    last = window.spec.latent_frames - 4
    assert controls.block(last).buttons.shape == (16, 11)
    for f0 in (0, 2, last + 4):
        with pytest.raises(ValueError, match=f"block {f0} is not a block of the recording's 33"):
            controls.block(f0)
