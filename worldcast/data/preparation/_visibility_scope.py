"""The frames of a recording seen through a sniper scope, from its video.

A scope narrows the field of view and draws a black surround, so the frame's four corners go to
black while its middle shows the scene. The corners' mean luminance over the frame's is about 1 for
an ordinary frame and 0 for a scoped one; a frame is scoped below :data:`SCOPE_CORNER_RATIO`. The
labels leave out every scoped frame and its neighbours.
"""

from pathlib import Path

import numpy as np

__all__ = ["SCOPE_CORNER_RATIO", "corner_ratio", "scoped_frames"]

#: A frame whose corners' luminance is below this fraction of its mean is scoped.
SCOPE_CORNER_RATIO = 0.25
#: A corner is a sixth of the frame's height and an eighth of its width.
_CORNER_HEIGHT_DIVISOR = 6
_CORNER_WIDTH_DIVISOR = 8
#: Luminance weights of the channels (BGR, as OpenCV decodes).
_LUMA_BGR = np.array([0.114, 0.587, 0.299])


def corner_ratio(frame: np.ndarray) -> float:
    """The corners' mean luminance over the frame's.

    Args:
        frame (np.ndarray): ``[H, W, 3]`` uint8, BGR.
    """
    import cv2

    height, width = frame.shape[:2]
    corner_h = max(1, height // _CORNER_HEIGHT_DIVISOR)
    corner_w = max(1, width // _CORNER_WIDTH_DIVISOR)
    corners = (
        frame[:corner_h, :corner_w],
        frame[:corner_h, -corner_w:],
        frame[-corner_h:, :corner_w],
        frame[-corner_h:, -corner_w:],
    )
    corner_mean = sum(np.array(cv2.mean(patch)[:3]) for patch in corners) / len(corners)
    frame_luma = float(np.array(cv2.mean(frame)[:3]) @ _LUMA_BGR)
    return float(corner_mean @ _LUMA_BGR) / max(frame_luma, 1e-6)


def scoped_frames(video_path: str | Path, frames: int) -> np.ndarray:
    """``[frames]`` bool: the frames of a video seen through a scope, and each one's neighbours.

    Decoded with OpenCV (the optional dependency of the video readers).

    Args:
        video_path (str | Path): the recording.
        frames (int): its frame count; the video must hold exactly this many.

    Raises:
        RuntimeError: the video cannot be opened or does not hold ``frames`` frames.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open the video {video_path}")
    raw = np.zeros(int(frames), dtype=bool)
    try:
        for index in range(int(frames)):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"{video_path} ends at frame {index}, expected {frames}")
            raw[index] = corner_ratio(frame) < SCOPE_CORNER_RATIO
        if capture.read()[0]:
            raise RuntimeError(f"{video_path} holds more than {frames} frames")
    finally:
        capture.release()
    scoped = raw.copy()
    scoped[1:] |= raw[:-1]
    scoped[:-1] |= raw[1:]
    return scoped
