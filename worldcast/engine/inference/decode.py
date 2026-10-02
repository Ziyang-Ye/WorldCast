"""Decode a client's latents to uint8 frames and a 16 fps, 672 x 384 mp4, as the paper's videos were made.

``latents.npy`` ``[N, 48, 24, 42]`` (``N = 1 + 4 k``) is stream-decoded by the Wan2.2 VAE (CUDA, bf16) in chunks of
:data:`DEFAULT_DECODE_CHUNK` latents with the conv cache carried across chunks, so the frames equal one decode of the
whole sequence. The mp4 bytes depend on the ffmpeg/x264 build: compare latents or uint8 frames instead.
"""

import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

from worldcast.modeling.wan22.vae import Wan22StreamingDecoder, Wan22VAE, latent_scale

__all__ = [
    "DEFAULT_DECODE_CHUNK",
    "FPS",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "LATENT_BLOCK",
    "MP4_WRITER_KWARGS",
    "count_mp4_frames",
    "decode_frames",
    "decode_to_mp4",
    "expected_frame_count",
    "load_latents",
    "pixels_to_uint8_frames",
    "stream_decode_pixels",
    "vae_decode_scale",
    "write_mp4",
]

#: Latent frames per streamed decode call (paper: 8). Keep 8 on GPU: cuDNN may pick another algorithm for another
#: chunk shape in bf16.
DEFAULT_DECODE_CHUNK = 8
#: Output video frame rate (pixel frames per second).
FPS = 16.0
#: Output frame size in pixels (latent grid 24 x 42 times 16).
FRAME_HEIGHT, FRAME_WIDTH = 384, 672
#: Latents per generated block; a rollout has ``1 + LATENT_BLOCK * k`` latents.
LATENT_BLOCK = 4
#: imageio writer arguments of the paper videos.
MP4_WRITER_KWARGS = dict(
    fps=FPS, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error"
)


# ------------------------------------------------------------------------------------- streamed decode
#: ``[mean, 1/std]`` of the decode, in the VAE's dtype (a fp32 scale would promote a bf16 chunk).
vae_decode_scale = latent_scale


def stream_decode_pixels(
    decoder: Wan22StreamingDecoder,
    latents: torch.Tensor,
    *,
    chunk: int = DEFAULT_DECODE_CHUNK,
    scale: Sequence[torch.Tensor],
) -> Iterator[torch.Tensor]:
    """Yield pixels ``[T, 3, H, W]`` float32 in [0, 1], one latent chunk at a time (``1 + 4 (N - 1)`` in total).

    ``latents`` ``[1, N, C, h, w]`` is one video on the VAE's device and dtype; the decoder is reset first.
    """
    chunk = int(chunk)
    if chunk < 1:
        raise ValueError(f"decode chunk must be >= 1, got {chunk}")
    if int(latents.shape[0]) != 1:
        raise ValueError(f"streaming decode takes one video, got batch {latents.shape[0]}")
    decoder.reset()
    for start in range(0, int(latents.shape[1]), chunk):
        piece = latents[:, start : start + chunk]
        pixels = decoder.decode_chunk(piece.permute(0, 2, 1, 3, 4), scale).float().clamp_(-1, 1)
        yield (pixels.permute(0, 2, 1, 3, 4)[0] * 0.5 + 0.5).clamp(0, 1)


def pixels_to_uint8_frames(pixels: torch.Tensor) -> np.ndarray:
    """``[T, 3, H, W]`` in [0, 1] -> ``[T, H, W, 3]`` uint8, as the mp4 receives them.

    ``round`` is torch's round-half-to-even on ``x * 255`` in float32.
    """
    frames = (pixels.float().clamp(0, 1) * 255).round().to(torch.uint8)
    return frames.permute(0, 2, 3, 1).cpu().numpy()


# ------------------------------------------------------------------------------------- decode pipeline
def expected_frame_count(num_latents: int) -> int:
    """Pixel frames decoded from ``num_latents`` latents: ``1 + 4 (N - 1)``."""
    return 1 + 4 * (int(num_latents) - 1)


def load_latents(path: str | os.PathLike) -> torch.Tensor:
    """``latents.npy`` ``[N, 48, h, w]`` -> ``[1, N, 48, h, w]`` as stored; ``N`` must be ``1 + 4 k``."""
    latents = torch.from_numpy(np.load(str(path)))[None]
    if latents.ndim != 5:
        raise ValueError(f"{path}: expected latents [N, C, h, w], got {tuple(latents.shape[1:])}")
    n = int(latents.shape[1])
    if n < 1 or (n - 1) % LATENT_BLOCK:
        raise ValueError(f"{path}: {n} latents is not 1 + {LATENT_BLOCK} k")
    return latents


def decode_frames(
    vae: Wan22VAE,
    latents: torch.Tensor,
    *,
    chunk: int = DEFAULT_DECODE_CHUNK,
) -> Iterator[np.ndarray]:
    """Yield uint8 frames ``[H, W, 3]`` of ``latents [1, N, C, h, w]``, cast to the VAE's device and dtype first."""
    param = next(vae.parameters())
    device, dtype = param.device, param.dtype
    scale = vae_decode_scale(device, dtype)
    decoder = Wan22StreamingDecoder(vae)
    with torch.no_grad():
        for pixels in stream_decode_pixels(
            decoder, latents.to(device=device, dtype=dtype), chunk=chunk, scale=scale
        ):
            yield from pixels_to_uint8_frames(pixels)
            del pixels


def write_mp4(frames: Iterable[np.ndarray], path: str | os.PathLike, *, fps: float = FPS) -> int:
    """Write uint8 ``[H, W, 3]`` frames to ``path`` with the paper's libx264 settings; returns the frame count.

    The file is written under a hidden temporary name in the same directory and renamed when complete.
    """
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}")
    kwargs = dict(MP4_WRITER_KWARGS)
    kwargs["fps"] = fps
    writer = imageio.get_writer(str(partial), **kwargs)
    written = 0
    try:
        for frame in frames:
            writer.append_data(frame)
            written += 1
    finally:
        writer.close()
    os.replace(partial, path)
    return written


def decode_to_mp4(
    vae: Wan22VAE,
    latents: torch.Tensor,
    path: str | os.PathLike,
    *,
    chunk: int = DEFAULT_DECODE_CHUNK,
) -> int:
    """Decode ``latents [1, N, 48, 24, 42]`` with ``vae`` (paper: CUDA, bf16) into a 16 fps mp4 at ``path``.

    Returns the number of frames written, ``1 + 4 (N - 1)``; raises if the decode produced a different count.
    """
    expected = expected_frame_count(int(latents.shape[1]))
    written = write_mp4(decode_frames(vae, latents, chunk=chunk), path)
    if written != expected:
        raise RuntimeError(
            f"decoded {written} frames, expected {expected} (1 + 4 (N - 1) for N ="
            f" {latents.shape[1]})"
        )
    return written


def count_mp4_frames(path: str | os.PathLike) -> int:
    """Count the frames of a video file with OpenCV (needs ``opencv-python``)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n
