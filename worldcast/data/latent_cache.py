"""The latent cache.

``<latent_cache_root>/<media_id>.npz`` (config ``paths.latent_cache_root``) holds one member
``win_<start:06d>`` per cached window (:func:`~worldcast.data.latents.window_key`), float16 ``[1,
41, 48, 24, 42]`` (Wan2.2 VAE latents of 161 video frames at 16 fps, 384x672). Inference reads only
latent frame 0 of the window that starts at the client's ``start_frame`` (the first frame). Values
are widened to float32 exactly.
"""

from pathlib import Path

import numpy as np
import torch

from worldcast.config.training import WINDOW_LATENT_FRAMES

from .latents import BLOCK, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH, window_key

__all__ = [
    "cached_window_starts",
    "load_block_latents",
    "load_first_latent",
    "load_window_latents",
]

#: The members of an archive are named ``win_<start_frame:06d>``.
_KEY_PREFIX = window_key(0)[:-6]


def _archive(latent_cache_root: str | Path, media_id: str) -> Path:
    return Path(latent_cache_root) / f"{media_id}.npz"


def _window(latent_cache_root: str | Path, media_id: str, start_frame: int) -> np.ndarray:
    """The cached window ``win_<start_frame>`` of ``media_id``, ``[41, 48, 24, 42]`` as stored."""
    path = _archive(latent_cache_root, media_id)
    if not path.is_file():
        raise FileNotFoundError(f"latent cache entry missing: {path}")
    key = window_key(start_frame)
    with np.load(path) as archive:
        if key not in archive:
            raise KeyError(f"latent window {key!r} missing in {path}")
        window = np.asarray(archive[key])
    expected = (1, WINDOW_LATENT_FRAMES, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)
    if window.shape != expected:
        raise ValueError(f"latent cache shape {window.shape} != {expected} at {path}")
    return window[0]


def load_window_latents(
    latent_cache_root: str | Path, media_id: str, start_frame: int
) -> torch.Tensor:
    """All latent frames of the cached window ``win_<start_frame>`` of ``media_id``.

    Returns:
        torch.Tensor: ``[41, 48, 24, 42]`` float32.

    Raises:
        FileNotFoundError: no archive.
        KeyError: no such window.
        ValueError: a member that is not ``[1, 41, 48, 24, 42]``.
    """
    return torch.from_numpy(_window(latent_cache_root, media_id, start_frame)).float()


def load_first_latent(
    latent_cache_root: str | Path, media_id: str, start_frame: int
) -> torch.Tensor:
    """The first frame: latent frame 0 of the cached window ``win_<start_frame>``, ``[1, 48, 24,
    42]`` float32."""
    return load_window_latents(latent_cache_root, media_id, start_frame)[:1].clone()


def load_block_latents(
    latent_cache_root: str | Path, media_id: str, start_frame: int, f0: int
) -> torch.Tensor:
    """The block at latent frame ``f0`` of the cached window ``win_<start_frame>``.

    Returns:
        torch.Tensor: ``[4, 48, 24, 42]`` float32.

    Raises:
        RuntimeError: the block runs past the window. Otherwise as :func:`load_window_latents`.
    """
    block = _window(latent_cache_root, media_id, start_frame)[int(f0) : int(f0) + BLOCK]
    if len(block) != BLOCK:
        raise RuntimeError(
            f"{media_id}@{start_frame}: latent frames {f0}..{f0 + BLOCK - 1} are not all in the"
            " cached window"
        )
    return torch.from_numpy(block).float()


def cached_window_starts(latent_cache_root: str | Path, media_id: str) -> np.ndarray:
    """Sorted source-frame starts of every cached window of ``media_id`` (``[n]`` int64).

    A media without an archive has no cached window: the cache covers the media that own training
    windows, not every player of every round, and such a player contributes no memory block.
    """
    path = _archive(latent_cache_root, media_id)
    if not path.is_file():
        return np.zeros((0,), dtype=np.int64)
    with np.load(path) as archive:
        names = [name for name in archive.files if name.startswith(_KEY_PREFIX)]
    return np.asarray(sorted(int(name[len(_KEY_PREFIX) :]) for name in names), dtype=np.int64)
