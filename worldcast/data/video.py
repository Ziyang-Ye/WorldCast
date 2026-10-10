"""Recorded video: the frames of a stage-1 window, and single frames of a round's recordings.

Optional dependencies, imported when used: decord (when installed) and OpenCV.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .latents import FRAME_SIZE, SOURCE_FRAMES_PER_VIDEO_FRAME, source_span
from .recordings import MediaRecord

__all__ = ["VideoFrames", "load_video_window"]


def load_video_window(
    video_path: str | Path, start_frame: int, *, video_frames: int, source_frames: int
) -> torch.Tensor:
    """The video frames of a window (every second source frame of a recording from ``start_frame``
    on), ``[3, T, 384, 672]`` float32 in [-1, 1].

    Decoded with decord when it is installed, else OpenCV; resized bilinearly (no corner alignment)
    when the video has another size, then mapped ``(x / 255 - 0.5) * 2``.

    Args:
        video_path (str | Path): the recording.
        start_frame (int): the window's first source frame.
        video_frames (int): ``T``, the window's video frames.
        source_frames (int): the recording's frame count, checked against the decoder's.
    """
    span = source_span(video_frames)
    try:
        import decord
    except ImportError:
        decord = None
    if decord is not None:
        reader = decord.VideoReader(str(video_path), num_threads=1)
        count = len(reader)
    else:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if count != source_frames or start_frame + span > source_frames:
        raise RuntimeError(f"{video_path}: {count} frames, expected {source_frames}")
    if decord is not None:
        indices = list(range(start_frame, start_frame + span, SOURCE_FRAMES_PER_VIDEO_FRAME))
        frames = torch.from_numpy(reader.get_batch(indices).asnumpy())
    else:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        buffer = []
        for offset in range(span):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode {video_path} frame {start_frame + offset}")
            if offset % SOURCE_FRAMES_PER_VIDEO_FRAME == 0:
                buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        frames = torch.from_numpy(np.stack(buffer))
    frames = frames.float().permute(0, 3, 1, 2)
    if tuple(frames.shape[-2:]) != FRAME_SIZE:
        frames = F.interpolate(frames, size=FRAME_SIZE, mode="bilinear", align_corners=False)
    return ((frames / 255.0 - 0.5) * 2.0).permute(1, 0, 2, 3).contiguous()


class VideoFrames:
    """Decoded RGB frames of a round's recordings, cached per ``(media_id, source frame)``; a
    context manager that releases its readers.

    Decoded with OpenCV and resized with ``INTER_AREA`` to 384 x 672, the resolution of every RGB
    measurement of the memory frames.

    Args:
        dataset_root (str | Path): the recordings.
        media (Mapping[str, MediaRecord]): the round's media rows by media id.
    """

    def __init__(self, dataset_root: str | Path, media: Mapping[str, MediaRecord]) -> None:
        self.dataset_root = Path(dataset_root)
        self.media = media
        self.readers: dict = {}
        self.frames: dict[tuple[str, int], np.ndarray] = {}

    def __enter__(self) -> "VideoFrames":
        return self

    def __exit__(self, *exc) -> None:
        for reader in self.readers.values():
            reader.release()
        self.readers.clear()

    def get(self, media_id: str, frame: int) -> np.ndarray:
        """``[H, W, 3]`` uint8 RGB of source frame ``frame`` of ``media_id``."""
        import cv2

        key = (str(media_id), int(frame))
        if key not in self.frames:
            path = self.dataset_root / str(self.media[media_id].video_path)
            if media_id not in self.readers:
                self.readers[media_id] = cv2.VideoCapture(str(path))
            capture = self.readers[media_id]
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, bgr = capture.read()
            if not ok or abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - (int(frame) + 1)) > 0.5:
                raise RuntimeError(f"cannot decode {path} frame {frame}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = FRAME_SIZE
            self.frames[key] = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        return self.frames[key]
