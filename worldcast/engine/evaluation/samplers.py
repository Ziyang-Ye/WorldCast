"""The samplers of the evaluation protocols, on one 41-latent window with its recorded first frame.

* :func:`sample_bidirectional`: stages 1_long, 2 and 2s; the whole window in one block, the first
  frame pinned clean at t = 0, 20 UniPC steps.
* :func:`sample_block_causal`: stage 3; the first frame and then each block of 4 latents written
  into the KV cache, each block denoised by 20 UniPC steps and written clean (t = 0).
* :func:`sample_four_step`: stage 4; as the block-causal one with the four denoising steps,
  re-noising between them from a generator seeded per window.

The two block-causal samplers run on a :class:`~worldcast.sampling.sampler.Sampler` whose context
noise is 0: the context is written clean.
"""

from collections.abc import Callable, Mapping
from typing import Any

import torch

from worldcast.data.latents import BLOCK
from worldcast.sampling.sampler import Sampler
from worldcast.sampling.unipc import FlowUniPCSolver

__all__ = ["FlowFn", "sample_bidirectional", "sample_block_causal", "sample_four_step"]

#: ``flow_fn(noisy, timestep, conditions) -> flow``: one whole-window forward (``[B, F, C, H, W]``,
#: ``[B, F]``), e.g. the trainer's bidirectional model.
FlowFn = Callable[[torch.Tensor, torch.Tensor, Mapping[str, Any]], torch.Tensor]


def _unipc(
    flow_of: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    latents: torch.Tensor,
    steps: int,
    *,
    first_frame: torch.Tensor | None = None,
) -> torch.Tensor:
    """``steps`` UniPC steps on ``latents`` ``[B, F, C, H, W]`` with ``flow_of(x, timestep)``; with
    ``first_frame`` frame 0 is that frame, pinned clean at t = 0."""
    solver = FlowUniPCSolver(steps, device=latents.device)
    for t in solver.timesteps:
        timestep = t * torch.ones(latents.shape[:2], device=latents.device, dtype=torch.float32)
        if first_frame is not None:
            timestep[:, :1] = 0
        stepped = solver.step(flow_of(latents, timestep), latents)
        if first_frame is not None:
            stepped = torch.cat([first_frame, stepped[:, 1:]], dim=1)
        latents = stepped
    return latents


def sample_bidirectional(
    flow_fn: FlowFn,
    noise: torch.Tensor,
    first_frame: torch.Tensor,
    conditions: Mapping[str, Any],
    *,
    steps: int,
) -> torch.Tensor:
    """Denoise the whole window at once; frame 0 stays the recorded first frame (t = 0).

    Args:
        flow_fn (FlowFn): the generator on the window.
        noise (Tensor): ``[B, F - 1, C, H, W]`` noise of the frames after the first.
        first_frame (Tensor): ``[B, 1, C, H, W]`` the recorded first frame.
        conditions (Mapping[str, Any]): the window's conditions.
        steps (int): UniPC steps.

    Returns:
        Tensor: ``[B, F, C, H, W]`` in ``noise.dtype``.
    """
    first_frame = first_frame.to(device=noise.device, dtype=noise.dtype)
    return _unipc(
        lambda x, timestep: flow_fn(x, timestep, conditions),
        torch.cat([first_frame, noise], dim=1),
        steps,
        first_frame=first_frame,
    )


def _block_causal(
    sampler: Sampler,
    noise: torch.Tensor,
    first_frame: torch.Tensor,
    conditions: Mapping[str, Any],
    denoise: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """The first frame written into the emptied cache, then per block ``denoise(noise,
    frame_offset)`` and the write of its result. Returns the window in ``noise.dtype``."""
    window = torch.cat([first_frame.to(noise), torch.zeros_like(noise)], dim=1)
    sampler.cache.reset()
    sampler.write_context(window[:, :1], conditions, frame_offset=0)
    for start in range(1, window.shape[1], BLOCK):
        block = slice(start, start + BLOCK)
        window[:, block] = denoise(noise[:, start - 1 : start - 1 + BLOCK], start)
        sampler.write_context(window[:, block], conditions, frame_offset=start)
    return window


def sample_block_causal(
    sampler: Sampler,
    noise: torch.Tensor,
    first_frame: torch.Tensor,
    conditions: Mapping[str, Any],
    *,
    steps: int,
) -> torch.Tensor:
    """The teacher-forced model block by block, each block by ``steps`` UniPC steps.

    Args:
        sampler (Sampler): the generator on a KV cache with room for the window, context noise 0.
        noise (Tensor): ``[B, F - 1, C, H, W]`` noise of the frames after the first.
        first_frame (Tensor): ``[B, 1, C, H, W]`` the recorded first frame.
        conditions (Mapping[str, Any]): the window's conditions.
        steps (int): UniPC steps per block.

    Returns:
        Tensor: ``[B, F, C, H, W]`` in ``noise.dtype``.
    """

    def denoise(noisy: torch.Tensor, frame_offset: int) -> torch.Tensor:
        def flow_of(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return sampler.predict_flow(x, timestep, conditions, frame_offset=frame_offset)

        return _unipc(flow_of, noisy, steps)

    return _block_causal(sampler, noise, first_frame, conditions, denoise)


def sample_four_step(
    sampler: Sampler,
    noise: torch.Tensor,
    first_frame: torch.Tensor,
    conditions: Mapping[str, Any],
) -> torch.Tensor:
    """The four-step generator block by block (1000, 937.5, 833.3, 625), re-noising x0 between the
    steps with draws from the sampler's ``rng``.

    Args:
        sampler (Sampler): the generator on a KV cache with room for the window, context noise 0;
            its ``rng`` is the window's generator, on the device of ``noise``.
        noise (Tensor): ``[B, F - 1, C, H, W]`` noise of the frames after the first.
        first_frame (Tensor): ``[B, 1, C, H, W]`` the recorded first frame.
        conditions (Mapping[str, Any]): the window's conditions.

    Returns:
        Tensor: ``[B, F, C, H, W]`` in ``noise.dtype``.
    """

    def denoise(noisy: torch.Tensor, frame_offset: int) -> torch.Tensor:
        return sampler.denoise(noisy, conditions, frame_offset=frame_offset)

    return _block_causal(sampler, noise, first_frame, conditions, denoise)
