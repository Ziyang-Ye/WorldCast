"""The latent frames: their shape, their time grid and the layout of a block's context.

A recording has 32 source frames per second; every second one is a video frame (16 fps). Latent
frame 0 holds video frame 0 and latent frame ``f >= 1`` the video frames ``4f - 3 .. 4f`` (the
causal VAE); a latent frame takes its state (camera, positions, depth) at its last video frame,
``4f``. A block is four latent frames, one second of video.
"""

import numpy as np

__all__ = [
    "BLOCK",
    "FIRST_TARGET",
    "FPS",
    "FRAME_SIZE",
    "FRAME_TOKENS",
    "LATENT_CHANNELS",
    "LATENT_GRID",
    "LATENT_HEIGHT",
    "LATENT_SHAPE",
    "LATENT_WIDTH",
    "PIXELS_PER_CELL",
    "RECENT",
    "SOURCE_FRAMES_PER_LATENT",
    "SOURCE_FRAMES_PER_VIDEO_FRAME",
    "TOKEN_GRID",
    "VIDEO_FRAMES_PER_BLOCK",
    "VIDEO_FRAMES_PER_LATENT",
    "block_span",
    "block_starts",
    "last_video_frame",
    "last_video_frames",
    "latent_frame_count",
    "source_frame",
    "source_span",
    "video_frame_count",
    "video_frames_of",
    "window_key",
]

#: Source frames (32 fps) per video frame (16 fps).
SOURCE_FRAMES_PER_VIDEO_FRAME = 2
#: Video frames per second.
FPS = 16.0
#: Video frames per latent frame.
VIDEO_FRAMES_PER_LATENT = 4
#: Source frames per latent frame: the stride of the latent frames' time grid.
SOURCE_FRAMES_PER_LATENT = SOURCE_FRAMES_PER_VIDEO_FRAME * VIDEO_FRAMES_PER_LATENT
#: Latent frames per block (the unit of generation; one memory entry, the memory frames).
BLOCK = 4
#: Video frames per block: 16, a second of video.
VIDEO_FRAMES_PER_BLOCK = VIDEO_FRAMES_PER_LATENT * BLOCK
#: Latent frames of the recent context of a block: the client's ``s - 12 .. s - 1`` (Sec. 3.3).
RECENT = 12
#: The first latent frame generated with the scene state: a client's latent frame 25 (App. "Scene
#: state in detail", Retrieval); latent frames 1-24, six blocks, come before.
FIRST_TARGET = 1 + 6 * BLOCK
#: Wan2.2 VAE latent channels and grid for 384x672 video.
LATENT_CHANNELS = 48
LATENT_HEIGHT = 24
LATENT_WIDTH = 42
#: The latent grid ``(height, width)``: also the depth head's grid, one cell per 16 x 16 pixels.
LATENT_GRID = (LATENT_HEIGHT, LATENT_WIDTH)
#: One latent frame ``[48, 24, 42]``.
LATENT_SHAPE = (LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)
#: Pixels per latent cell along each axis: the VAE's spatial stride.
PIXELS_PER_CELL = 16
#: The video frames' ``(height, width)``.
FRAME_SIZE = (PIXELS_PER_CELL * LATENT_HEIGHT, PIXELS_PER_CELL * LATENT_WIDTH)
#: The generator's token grid per latent frame (the latent grid in 2 x 2 patches) and its 252
#: tokens.
TOKEN_GRID = (LATENT_HEIGHT // 2, LATENT_WIDTH // 2)
FRAME_TOKENS = TOKEN_GRID[0] * TOKEN_GRID[1]


def video_frame_count(latent_frames: int) -> int:
    """``T = 1 + 4 (F - 1)`` video frames of ``F`` latent frames."""
    return 1 + VIDEO_FRAMES_PER_LATENT * (int(latent_frames) - 1)


def source_span(video_frames: int) -> int:
    """Source frames from the first to the last of ``T`` video frames, ``1 + 2 (T - 1)``."""
    return 1 + SOURCE_FRAMES_PER_VIDEO_FRAME * (int(video_frames) - 1)


def latent_frame_count(video_frames: int) -> int:
    """``F`` for ``T = 1 + 4 (F - 1)`` video frames; raises if ``T`` is not of that form."""
    video_frames = int(video_frames)
    if (video_frames - 1) % VIDEO_FRAMES_PER_LATENT:
        raise ValueError(f"{video_frames} video frames; expected 1 + 4(F - 1)")
    return 1 + (video_frames - 1) // VIDEO_FRAMES_PER_LATENT


def last_video_frame(latent_frame: int) -> int:
    """The last video frame, ``4 f``, of latent frame ``f``: where its state is read."""
    return VIDEO_FRAMES_PER_LATENT * int(latent_frame)


def last_video_frames(first: int, count: int) -> np.ndarray:
    """``[count]`` int64: :func:`last_video_frame` of the latent frames ``first .. first + count -
    1``."""
    return VIDEO_FRAMES_PER_LATENT * np.arange(int(first), int(first) + int(count))


def video_frames_of(first: int, count: int = 1) -> list[int]:
    """The video frames of the latent frames ``first .. first + count - 1``, ascending: ``[0]`` for
    latent frame 0, ``4f - 3 .. 4f`` for a latent frame ``f >= 1`` (16 for a block)."""
    first, count = int(first), int(count)
    if first < 0 or count < 1:
        raise ValueError(f"latent frames {first} .. {first + count - 1}")
    start = max(0, VIDEO_FRAMES_PER_LATENT * (first - 1) + 1)
    return list(range(start, VIDEO_FRAMES_PER_LATENT * (first + count - 1) + 1))


def block_starts(latent_frames: int) -> range:
    """The first latent frame of every complete block of a window of ``latent_frames`` latent
    frames: 1, 5, 9, ... (latent frame 0 is the first frame)."""
    return range(1, int(latent_frames) - BLOCK + 1, BLOCK)


def source_frame(start_frame: int, latent_frame: int) -> int:
    """The source frame of ``latent_frame`` (of its last video frame) in a window that starts at
    source frame ``start_frame``."""
    return int(start_frame) + SOURCE_FRAMES_PER_LATENT * int(latent_frame)


def block_span(start_frame: int, f0: int) -> tuple[int, int]:
    """The source frames ``(t_first, t_last)`` of the first and last latent frame of the block at
    latent frame ``f0`` of a window that starts at source frame ``start_frame``."""
    return source_frame(start_frame, f0), source_frame(start_frame, int(f0) + BLOCK - 1)


def window_key(start_frame: int) -> str:
    """Latent-cache member name of the window that starts at source frame ``start_frame``."""
    return f"win_{int(start_frame):06d}"
