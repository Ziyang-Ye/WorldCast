"""The training window of a micro-batch: the ordinary window, or the memory window."""

import pytest
import torch

from tests.engine.training.support import make_item
from worldcast.config.training import paper_config
from worldcast.data.window import collate_windows
from worldcast.engine.training.window import (
    MemoryWindowConfig,
    memory_frame_config,
    memory_this_step,
    memory_windows,
    training_window,
)


def _batch(target_starts: list[int]) -> dict:
    return collate_windows([make_item(i, target_start=s) for i, s in enumerate(target_starts)])


@pytest.mark.parametrize("bidirectional", [False, True])
def test_training_window(bidirectional):
    batch = _batch([37, 37] if bidirectional else [25, 37])
    window = training_window(batch, step=0, seed=0, memory=None, bidirectional=bidirectional)
    assert not window.has_memory and window.batch is batch and window.pinned_frames == 1
    assert window.rays is None and window.loss_mask is None and window.memory_mask is None
    recent = 32 if bidirectional else 12
    memory = MemoryWindowConfig(prob=1.0, recent=recent)
    window = training_window(batch, step=0, seed=0, memory=memory, bidirectional=bidirectional)
    frames = 1 + 4 + recent + 4
    assert window.has_memory and window.batch["latents"].shape[1] == frames
    assert window.pinned_frames == (5 if bidirectional else 1)
    assert window.loss_mask.tolist() == [[0.0] * (frames - 4) + [1.0] * 4] * 2
    assert torch.equal(window.memory_mask, batch["window_memory_mask"])
    assert window.ray_conditions("cpu")["ray_frame_c2w"].shape == (2, frames, 4, 4)


def test_the_memory_draw_is_one_per_step_and_seed():
    # the same on every rank and micro-batch: it depends on the seed and the step alone
    draws = [memory_this_step(0, step, 0.8) for step in range(1000)]
    assert draws[:8] == [False, True, True, True, False, True, False, False]
    assert sum(draws) == 796
    assert draws != [memory_this_step(1, step, 0.8) for step in range(1000)]
    assert memory_this_step(0, 3, 1.0) and not memory_this_step(0, 3, 0.0)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("2", None),
        ("3_noscene", None),
        ("2s", MemoryWindowConfig(prob=0.8, recent=32, continuous_camera=False)),
        ("3", MemoryWindowConfig(prob=0.8, recent=12, continuous_camera=True)),
    ],
)
def test_the_memory_windows_of_a_stage(stage, expected):
    cfg = paper_config(stage)
    assert memory_windows(cfg) == expected
    if expected is not None:
        assert memory_frame_config(cfg).recent == expected.recent
