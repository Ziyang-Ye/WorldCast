"""A causal generator that records its calls, and a tiny sampler on it."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from worldcast.modeling.wan22.dit import KVCache
from worldcast.sampling.sampler import CONTEXT_NOISE, FRAMEWISE_KEYS, Sampler

#: A tiny latent frame ``[C, H, W]`` and its tokens.
LATENT = (3, 2, 2)
TOKENS = 3


@dataclass
class Call:
    """One generator call as the generator received it.

    Attributes:
        frame_offset (int): window position of the call's first latent frame.
        x (Tensor): ``[B, n, C, H, W]`` the noisy latents.
        timestep (Tensor): ``[B, n]``.
        conditions (Mapping[str, Any]): the call's conditions.
        cached (int): latent frames the cache held before the call.
    """

    frame_offset: int
    x: torch.Tensor
    timestep: torch.Tensor
    conditions: Mapping[str, Any]
    cached: int

    @property
    def summary(self) -> tuple[int, int, float]:
        """``(frame_offset, latent frames, timestep)``; the frames of a call share one timestep."""
        (timestep,) = self.timestep.float().unique().tolist()
        return self.frame_offset, self.x.shape[1], timestep


class RecordingGenerator:
    """A ``CausalGenerator`` with bf16 inputs: records every call, moves the cache's end as the
    generator does and returns :meth:`flow`."""

    input_dtype = torch.bfloat16

    def __init__(self) -> None:
        self.calls: list[Call] = []

    @staticmethod
    def flow(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.float() + timestep.float()[..., None, None, None] / 2000.0

    def __call__(self, noisy, timestep, conditions, *, kv_cache, frame_offset):
        self.calls.append(Call(frame_offset, noisy, timestep, conditions, kv_cache.end // TOKENS))
        kv_cache.end = kv_cache.span(frame_offset * TOKENS, noisy.shape[1] * TOKENS)
        return self.flow(noisy, timestep)


def tiny_sampler(
    seed: int = 0, *, context_noise: int = CONTEXT_NOISE
) -> tuple[Sampler, RecordingGenerator]:
    """The paper's sampler on a :class:`RecordingGenerator`, 25 latent frames of cache and a seeded
    stream."""
    generator = RecordingGenerator()
    cache = KVCache.allocate(
        num_dit_blocks=1,
        num_heads=1,
        head_dim=2,
        latent_frames=25,
        frame_tokens=TOKENS,
        dtype=torch.float32,
    )
    rng = torch.Generator().manual_seed(seed)
    return Sampler.create(generator, cache, context_noise=context_noise, rng=rng), generator


def indexed_conditions(frames: int) -> dict[str, torch.Tensor]:
    """Conditions of a window of ``frames`` latent frames: the frame-wise ones ``[1, frames, 3]``
    holding the frame index, and one that spans the window."""
    index = torch.arange(frames, dtype=torch.float32).view(1, frames, 1).expand(1, frames, 3)
    conditions = {key: index.clone() for key in FRAMEWISE_KEYS}
    conditions["view_deltas"] = torch.zeros(1, 1 + 4 * (frames - 1), 2)
    return conditions


def latents(frames: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[1, frames, *LATENT]`` seeded noise."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, frames, *LATENT, generator=g).to(dtype)
