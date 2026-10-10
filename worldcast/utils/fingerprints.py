"""Digests of latents: how a run is compared with the reference runs (docs/inference.md) and how a
published block is checked.

The module loads numpy only: a reader of the shared world state checks a block without torch.
"""

import hashlib

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["fingerprint", "float32_array", "sha256_float32"]


def float32_array(array: ArrayLike) -> np.ndarray:
    """A tensor or an array as a contiguous float32 numpy array (a view where it already is
    one)."""
    if hasattr(array, "detach"):  # a torch tensor, on any device and in any dtype (bf16 too)
        array = array.detach().float().cpu().numpy()
    return np.ascontiguousarray(array, dtype=np.float32)


def sha256_float32(array: ArrayLike) -> str:
    """The sha256 (64 hex digits) of a tensor's or an array's float32 bytes."""
    return hashlib.sha256(float32_array(array).tobytes()).hexdigest()


def fingerprint(array: ArrayLike) -> str:
    """The first 32 hex digits of :func:`sha256_float32`."""
    return sha256_float32(array)[:32]
