"""Decode a client's latents into uint8 frames and a 16 fps 672 x 384 mp4 with the paper's settings.

The Wan2.2 VAE decodes a video a chunk of latents at a time with its causal conv cache carried
across chunks: the frames are those of one decode of the whole sequence, up to the convolution
kernels cuDNN picks for a chunk's shape (the paper: CUDA, bf16, chunks of
:data:`DEFAULT_DECODE_CHUNK`). The mp4 bytes depend on the ffmpeg / x264 build: compare latents or
uint8 frames instead. An mp4 is named only once ffmpeg decodes it without a decoder error to every
frame written (:func:`write_mp4`).
"""

import os
import subprocess
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldcast.data.latents import BLOCK, FPS, video_frame_count
from worldcast.modeling.wan22.vae import Wan22StreamingDecoder, Wan22VAE, latent_scale

__all__ = [
    "DEFAULT_DECODE_CHUNK",
    "MP4_WRITER_KWARGS",
    "WanFrameDecoder",
    "decode_frames",
    "decode_pixels",
    "decode_to_mp4",
    "load_latents",
    "mp4_decoded_frames",
    "pixels_to_uint8_frames",
    "stream_decode_pixels",
    "write_mp4",
]

#: Latents per decode call. Keep 8 on the GPU: cuDNN may pick another algorithm for another chunk
#: shape in bf16.
DEFAULT_DECODE_CHUNK = 8
#: imageio writer arguments of the paper's videos.
MP4_WRITER_KWARGS = dict(
    fps=FPS, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error"
)


def _placement(vae: Wan22VAE) -> tuple[torch.device, torch.dtype, list[torch.Tensor]]:
    """The VAE's device and dtype, and its latent scale there."""
    param = next(vae.parameters())
    return param.device, param.dtype, latent_scale(param.device, param.dtype)


def _unit_range(decoded: torch.Tensor) -> torch.Tensor:
    """The VAE's output ``[1, 3, t, H, W]`` -> ``[t, 3, H, W]`` float32 in [0, 1]: clamped to
    [-1, 1], then mapped."""
    pixels = decoded.float().clamp_(-1, 1)
    return (pixels.permute(0, 2, 1, 3, 4)[0] * 0.5 + 0.5).clamp(0, 1)


def _pixels(stream: Wan22StreamingDecoder, latents: torch.Tensor, scale) -> torch.Tensor:
    """The next chunk ``[1, n, C, h, w]`` of a stream -> ``[t, 3, H, W]`` float32 in [0, 1]."""
    return _unit_range(stream.decode_chunk(latents.permute(0, 2, 1, 3, 4), scale))


def decode_pixels(vae: Wan22VAE, latents: torch.Tensor) -> torch.Tensor:
    """``[1, F, 48, h, w]`` latents -> ``[1 + 4 (F - 1), 3, 16 h, 16 w]`` float32 frames in [0, 1]:
    one decode of the whole sequence, in the VAE's dtype."""
    device, dtype, scale = _placement(vae)
    return _unit_range(vae.decode(latents.to(device, dtype).permute(0, 2, 1, 3, 4), scale))


def pixels_to_uint8_frames(pixels: torch.Tensor) -> np.ndarray:
    """``[t, 3, H, W]`` in [0, 1] -> ``[t, H, W, 3]`` uint8, as the mp4 receives them: torch's
    round-half-to-even of ``x * 255`` in float32.

    C-contiguous, laid out on the pixels' device: a strided view would cost a slow copy under the
    GIL in ``PIL.Image.fromarray``."""
    frames = (pixels.float().clamp(0, 1) * 255).round().to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def stream_decode_pixels(
    decoder: Wan22StreamingDecoder,
    latents: torch.Tensor,
    *,
    chunk: int = DEFAULT_DECODE_CHUNK,
    scale: Sequence[torch.Tensor],
) -> Iterator[torch.Tensor]:
    """Yield the pixels of one video, one chunk of latents at a time.

    Args:
        decoder (Wan22StreamingDecoder): reset first.
        latents (Tensor): ``[1, N, C, h, w]`` latents on the VAE's device, in its dtype.
        chunk (int): latents per call.
        scale (Sequence[Tensor]): ``[mean, 1/std]`` of :func:`latent_scale`, in the VAE's dtype.

    Yields:
        Tensor: ``[t, 3, H, W]`` float32 in [0, 1]; ``1 + 4 (N - 1)`` frames in total.
    """
    if int(chunk) < 1:
        raise ValueError(f"decode chunk must be >= 1, got {chunk}")
    if int(latents.shape[0]) != 1:
        raise ValueError(f"streaming decode takes one video, got batch {latents.shape[0]}")
    decoder.reset()
    for start in range(0, int(latents.shape[1]), int(chunk)):
        yield _pixels(decoder, latents[:, start : start + int(chunk)], scale)


class WanFrameDecoder:
    """The Wan2.2 VAE decoding one video as its latents arrive.

    ``decode(latents [n, C, h, w])`` returns the chunk's pixels ``[t, 3, H, W]`` float32 in [0, 1]:
    one frame for latent 0, four for every later latent. The frames do not depend on the chunking,
    up to the kernel choice of the convolutions.
    """

    def __init__(self, vae: Wan22VAE) -> None:
        self.device, self.dtype, self.scale = _placement(vae)
        self.stream = Wan22StreamingDecoder(vae)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """The pixels of the next latents ``[n, C, h, w]`` of the video."""
        return _pixels(
            self.stream, latents.to(device=self.device, dtype=self.dtype)[None], self.scale
        )


def load_latents(path: str | os.PathLike) -> torch.Tensor:
    """``latents.npy`` ``[N, 48, h, w]`` -> ``[1, N, 48, h, w]`` as stored; N is ``1 + 4 k``."""
    latents = torch.from_numpy(np.load(str(path)))[None]
    if latents.ndim != 5:
        raise ValueError(f"{path}: expected latents [N, C, h, w], got {tuple(latents.shape[1:])}")
    n = int(latents.shape[1])
    if (n - 1) % BLOCK:
        raise ValueError(f"{path}: {n} latents is not 1 + 4 k")
    return latents


def decode_frames(
    vae: Wan22VAE, latents: torch.Tensor, *, chunk: int = DEFAULT_DECODE_CHUNK
) -> Iterator[np.ndarray]:
    """Yield the uint8 frames ``[H, W, 3]`` of ``latents [1, N, C, h, w]``."""
    device, dtype, scale = _placement(vae)
    latents = latents.to(device=device, dtype=dtype)
    with torch.no_grad():
        for pixels in stream_decode_pixels(
            Wan22StreamingDecoder(vae), latents, chunk=chunk, scale=scale
        ):
            yield from pixels_to_uint8_frames(pixels)


def write_mp4(
    frames: Iterable[np.ndarray],
    path: str | os.PathLike,
    *,
    writer: Mapping[str, Any] = MP4_WRITER_KWARGS,
) -> int:
    """Write uint8 ``[H, W, 3]`` frames to ``path`` with the paper's x264 settings, or ``writer``'s.

    The file is written in one pass, its index after its frames, under a hidden name in the same
    directory, and named ``path`` once it decodes without a decoder error to every frame written
    (:func:`mp4_decoded_frames`); else, or when the frames stop with an exception, it is removed.

    Args:
        frames (Iterable[np.ndarray]): uint8 ``[H, W, 3]``.
        path (str | os.PathLike): the mp4.
        writer (Mapping[str, Any]): the imageio writer's arguments, by default the paper's.

    Returns:
        int: the frames written.

    Raises:
        RuntimeError: the file does not decode without a decoder error to every frame written.
    """
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}")
    written = 0
    try:
        out = imageio.get_writer(str(partial), **writer)
        try:
            for frame in frames:
                out.append_data(frame)
                written += 1
        finally:
            out.close()
        decoded, error = mp4_decoded_frames(partial)
        if error or decoded != written:
            raise RuntimeError(
                f"{path} not written: it decodes to {decoded} of its {written} frames"
                + (f"; ffmpeg: {error}" if error else "")
                + " (the video file is incomplete)"
            )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return written


def mp4_decoded_frames(path: str | os.PathLike) -> tuple[int, str]:
    """Decode the first video stream of the mp4 ``path`` with ffmpeg, up to its first decoder
    error.

    The decode stops at that error (``-xerror``, which also fails a frame the decoder had to
    conceal): a count of the frames alone misses most losses of a page of the file.

    Returns:
        tuple[int, str]: the frames decoded and the first line of ffmpeg's error, ``""`` when the
        file decodes without one.
    """
    import imageio_ffmpeg

    done = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            *("-nostdin", "-v", "error", "-xerror", "-progress", "pipe:1", "-nostats"),
            *("-i", str(path), "-map", "0:v:0", "-f", "null", "-"),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    counts = [line[6:] for line in done.stdout.splitlines() if line.startswith("frame=")]
    errors = done.stderr.splitlines()
    error = errors[0] if errors else f"exit status {done.returncode}" if done.returncode else ""
    return int(counts[-1]) if counts else 0, error


def decode_to_mp4(
    vae: Wan22VAE,
    latents: torch.Tensor,
    path: str | os.PathLike,
    *,
    chunk: int = DEFAULT_DECODE_CHUNK,
) -> int:
    """Decode ``latents [1, N, 48, 24, 42]`` into a 16 fps mp4 at ``path``; returns the frames."""
    expected = video_frame_count(int(latents.shape[1]))
    written = write_mp4(decode_frames(vae, latents, chunk=chunk), path)
    if written != expected:
        raise RuntimeError(f"decoded {written} frames, expected {expected} = 1 + 4 (N - 1)")
    return written
