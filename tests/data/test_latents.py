"""The latent frames: their shape, their time grid and the layout of a block's context."""

import pytest

from worldcast.data.latents import (
    BLOCK,
    FIRST_TARGET,
    FPS,
    FRAME_SIZE,
    RECENT,
    SOURCE_FRAMES_PER_LATENT,
    TOKEN_GRID,
    VIDEO_FRAMES_PER_BLOCK,
    block_span,
    last_video_frames,
    latent_frame_count,
    source_frame,
    video_frame_count,
    video_frames_of,
)


def test_a_block_is_a_second_of_video():
    assert (BLOCK, VIDEO_FRAMES_PER_BLOCK, FPS) == (4, 16, 16.0)
    assert len(video_frames_of(1, BLOCK)) == VIDEO_FRAMES_PER_BLOCK
    assert FRAME_SIZE == (384, 672) and TOKEN_GRID == (12, 21)
    # three blocks of recent context; six blocks come before the first block with scene state
    assert (RECENT, FIRST_TARGET) == (12, 25)


def test_the_causal_vae_layout():
    assert video_frame_count(41) == 161 and latent_frame_count(161) == 41
    with pytest.raises(ValueError):
        latent_frame_count(160)
    assert video_frames_of(0) == [0] and video_frames_of(3) == [9, 10, 11, 12]
    assert video_frames_of(0, 2) == [0, 1, 2, 3, 4]  # latent frame 0 owns video frame 0 alone
    assert video_frames_of(5, BLOCK) == list(range(17, 33))
    assert last_video_frames(2, 3).tolist() == [8, 12, 16]
    with pytest.raises(ValueError):
        video_frames_of(-1)


def test_a_latent_frame_spans_eight_source_frames():
    assert SOURCE_FRAMES_PER_LATENT == 8
    assert source_frame(16, 5) == 56
    assert block_span(16, 5) == (56, 80)
