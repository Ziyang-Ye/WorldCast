"""The context block ranges of a window."""

import pytest

from worldcast.sampling import window as Wn


@pytest.mark.parametrize("num_frames", [21, 17])
def test_context_block_ranges(num_frames):
    ctx_frames = num_frames - 4
    ranges = [(0, 1)] + [(st, 4) for st in range(1, ctx_frames, 4)]
    assert Wn.context_block_ranges(num_frames) == ranges
