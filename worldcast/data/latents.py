"""The latent cache: a window's first latent frame (the sink) and, for training, whole windows.

``<latent_cache_root>/<media_id>.npz`` (config ``paths.latent_cache_root``) holds one member
``win_<start:06d>`` per cached window, float16 ``[1, 41, 48, 24, 42]`` (Wan2.2 VAE latents of 161
frames at 16 fps, 384x672). Inference reads only latent 0 of the window that starts at the client's
``start_frame``. Values are widened to float32 exactly.
"""

from pathlib import Path

import numpy as np
import torch

#: Latent frames per cached window (161 pixel frames).
CACHE_LATENT_FRAMES = 41
#: Wan2.2 VAE latent channels and grid for 384x672 video.
LATENT_CHANNELS = 48
LATENT_HEIGHT = 24
LATENT_WIDTH = 42


def window_key(start_frame: int) -> str:
    """Latent-cache member name of the window starting at source frame ``start_frame``."""
    return f"win_{int(start_frame):06d}"


def _archive(latent_cache_root: str | Path, media_id: str) -> Path:
    path = Path(latent_cache_root) / f"{media_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"latent cache entry missing: {path}")
    return path


def load_window_latents(
    latent_cache_root: str | Path,
    media_id: str,
    start_frame: int,
    *,
    latent_frames: int = CACHE_LATENT_FRAMES,
    height: int = LATENT_HEIGHT,
    width: int = LATENT_WIDTH,
) -> torch.Tensor:
    """All latent frames of the cached window ``win_<start_frame>`` of ``media_id``.

    Returns:
        torch.Tensor: ``[latent_frames, 48, height, width]`` float32.

    Raises:
        FileNotFoundError: no archive. KeyError: no such window. ValueError: a member that is not
        ``[1, latent_frames, 48, height, width]``.
    """
    path = _archive(latent_cache_root, media_id)
    key = window_key(start_frame)
    with np.load(path) as archive:
        if key not in archive:
            raise KeyError(f"latent window {key!r} missing in {path}")
        latent = torch.from_numpy(np.asarray(archive[key]))
    expected = (1, int(latent_frames), LATENT_CHANNELS, int(height), int(width))
    if tuple(latent.shape) != expected:
        raise ValueError(f"latent cache shape {tuple(latent.shape)} != {expected} at {path}")
    return latent[0].float()


def load_first_latent(
    latent_cache_root: str | Path, media_id: str, start_frame: int
) -> torch.Tensor:
    """Latent 0 of cached window ``win_<start_frame>`` (the sink), ``[1, 48, 24, 42]`` float32."""
    return load_window_latents(latent_cache_root, media_id, start_frame)[:1].clone()


def cached_window_starts(latent_cache_root: str | Path, media_id: str) -> np.ndarray:
    """Sorted source-frame starts of every cached window of ``media_id`` (``[n]`` int64).

    A media without an archive has no cached window: the cache covers the media that own training
    windows, not every player of every round, and such a player contributes no memory block.
    """
    path = Path(latent_cache_root) / f"{media_id}.npz"
    if not path.is_file():
        return np.zeros((0,), dtype=np.int64)
    with np.load(path) as archive:
        starts = sorted(int(name[4:]) for name in archive.files if name.startswith("win_"))
    return np.asarray(starts, dtype=np.int64)


def load_block_latents(
    latent_cache_root: str | Path,
    media_id: str,
    window_start: int,
    first_latent: int,
    length: int = 4,
) -> torch.Tensor:
    """``length`` latent frames from ``first_latent`` on of cached window ``win_<window_start>``.

    Returns:
        torch.Tensor: ``[length, 48, H, W]`` float32.
    """
    path = _archive(latent_cache_root, media_id)
    with np.load(path) as archive:
        block = np.asarray(
            archive[window_key(window_start)][
                0, int(first_latent) : int(first_latent) + int(length)
            ]
        )
    if int(block.shape[0]) != int(length):
        raise RuntimeError(
            f"{media_id}@{window_start}: latents {first_latent}..{first_latent + length - 1} "
            "are not all in the cached window"
        )
    return torch.from_numpy(block).float()
