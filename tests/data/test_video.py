"""Recorded video: a stage-1 window and single frames of a recording."""

import dataclasses

import numpy as np
import pytest
import torch

from tests.data.support import RECORDING_FRAMES as FRAMES
from tests.data.support import RECORDING_LEVELS as LEVELS
from tests.data.support import media_record, tick_table
from worldcast.data.video import VideoFrames, load_video_window


def test_a_window_reads_every_second_frame(recording):
    video = recording / "m0.mp4"
    frames = load_video_window(video, 2, video_frames=3, source_frames=FRAMES)
    assert frames.shape == (3, 3, 384, 672) and frames.dtype == torch.float32
    levels = (frames.mean(dim=(0, 2, 3)) / 2.0 + 0.5) * 255.0
    assert levels.tolist() == pytest.approx([LEVELS[2], LEVELS[4], LEVELS[6]], abs=4.0)
    with pytest.raises(RuntimeError, match="12 frames, expected 13"):
        load_video_window(video, 0, video_frames=3, source_frames=FRAMES + 1)
    with pytest.raises(RuntimeError):  # source frames 9, 11, 13 run past the recording
        load_video_window(video, 9, video_frames=3, source_frames=FRAMES)


def test_single_frames_are_cached_at_the_frame_size(recording):
    media = {"m0": dataclasses.replace(media_record(tick_table()), video_path="m0.mp4")}
    with VideoFrames(recording, media) as video:
        frame = video.get("m0", 5)
        assert frame.shape == (384, 672, 3) and frame.dtype == np.uint8
        assert float(frame.mean()) == pytest.approx(LEVELS[5], abs=4.0)
        assert video.get("m0", 5) is frame
        with pytest.raises(RuntimeError, match="cannot decode"):
            video.get("m0", FRAMES + 3)
    assert not video.readers
