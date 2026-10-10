"""The block-causal sampler of one client: generator calls on the KV cache.

A rollout is three operations on the cache. :meth:`Sampler.write_context` writes clean latents into
it at the context noise, :meth:`Sampler.prefill_context` writes a window's context range by range,
and :meth:`Sampler.denoise` runs the denoising steps on a block. A client denoises and writes its
first six blocks one after another at their positions in the round; for every later block it
empties the cache, prefills the context of the block's window and denoises the target frames.
docs/inference.md ("The block loop", "Numerics") gives the order of the RNG draws and the dtypes.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from worldcast.modeling.wan22.dit import KVCache
from worldcast.utils.precision import cast_floating_tensors

from .schedulers import (
    ContextWriteNoise,
    FlowMatchScheduler,
    denoise_block,
    flow_to_x0,
    noise_context,
    shift_denoising_steps,
)

__all__ = [
    "CONTEXT_NOISE",
    "DENOISING_STEPS",
    "FRAMEWISE_KEYS",
    "CausalGenerator",
    "Sampler",
    "framewise_conditions",
]

#: The four denoising steps on the unshifted schedule; their timesteps
#: (:func:`~worldcast.sampling.schedulers.shift_denoising_steps`) are 1000, 937.5, 833.3, 625.
DENOISING_STEPS = (1000, 750, 500, 250)
#: The timestep label of every context write (the generator was trained with context noise in
#: [16, 32)); the level snaps to t ~ 14.82.
CONTEXT_NOISE = 16
#: Conditions cut to a call's latent frames (axis 1), as the reference runs passed them; every
#: other condition spans the window.
FRAMEWISE_KEYS = ("player_alive", "player_visible")


def framewise_conditions(
    conditions: Mapping[str, Any], *, frame_offset: int, num_frames: int
) -> dict[str, Any]:
    """The conditions of one call: those of :data:`FRAMEWISE_KEYS` it has, narrowed to the call's
    latent frames ``[frame_offset, frame_offset + num_frames)`` (views)."""
    narrowed = dict(conditions)
    for key in FRAMEWISE_KEYS:
        if key in narrowed:
            narrowed[key] = narrowed[key].narrow(1, frame_offset, num_frames)
    return narrowed


class CausalGenerator(Protocol):
    """One causal DiT call on the KV cache, for example
    :class:`~worldcast.modeling.wan22.model.CausalGeneratorAdapter`.

    ``generator(noisy, timestep, conditions, kv_cache=, frame_offset=)`` takes ``noisy``
    ``[B, F, C, H, W]`` and ``timestep`` ``[B, F]`` in ``input_dtype``. Its ``F`` latent frames sit
    at the window positions ``frame_offset ..`` (RoPE and control history): it attends to the cache
    before them, writes their keys and values there and moves ``kv_cache.end`` to the end of its
    write. It returns the flow ``[B, F, C, H, W]`` float32 and draws no random numbers.
    """

    input_dtype: torch.dtype | None

    def __call__(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        kv_cache: KVCache,
        frame_offset: int,
    ) -> torch.Tensor: ...


@dataclass
class Sampler:
    """The block-causal sampler of one client: generator, flow-matching table, denoising steps, KV
    cache.

    Attributes:
        generator (CausalGenerator): one causal DiT call.
        scheduler (FlowMatchScheduler): the flow-matching table.
        denoising_timesteps (Tensor): ``[K]`` float32 timesteps of the denoising steps
            (:func:`~worldcast.sampling.schedulers.shift_denoising_steps`).
        cache (KVCache): the KV cache.
        context_noise (int): the timestep label of every context write (16); 0: the context is
            written clean, at t = 0 (the generator trained without scene state, and the scoring
            of stages 3 and 4).
        rng (torch.Generator | None): the stream of the re-noise between denoising steps and of the
            context writes without a given noise, a generator on the latents' device; ``None`` is
            the global RNG of that device. A client uses one stream for its whole rollout.
    """

    generator: CausalGenerator
    scheduler: FlowMatchScheduler
    denoising_timesteps: torch.Tensor
    cache: KVCache
    context_noise: int = CONTEXT_NOISE
    rng: torch.Generator | None = None

    @classmethod
    def create(
        cls,
        generator: CausalGenerator,
        cache: KVCache,
        *,
        context_noise: int = CONTEXT_NOISE,
        rng: torch.Generator | None = None,
    ) -> "Sampler":
        """The paper's sampler: Wan2.2's table and the timesteps of :data:`DENOISING_STEPS`.

        Args:
            generator (CausalGenerator): one causal DiT call.
            cache (KVCache): the KV cache.
            context_noise (int): see :class:`Sampler`.
            rng (torch.Generator | None): see :class:`Sampler`.
        """
        scheduler = FlowMatchScheduler()
        return cls(
            generator=generator,
            scheduler=scheduler,
            denoising_timesteps=shift_denoising_steps(DENOISING_STEPS, scheduler),
            cache=cache,
            context_noise=context_noise,
            rng=rng,
        )

    def predict_flow(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        frame_offset: int,
    ) -> torch.Tensor:
        """The generator on ``F`` latent frames at window positions ``frame_offset ..
        frame_offset + F - 1``: the conditions of :data:`FRAMEWISE_KEYS` cut to those frames
        (:func:`framewise_conditions`), the frames' keys and values written into the cache.

        Args:
            x (Tensor): ``[B, F, C, H, W]`` latents.
            timestep (Tensor): ``[B, F]`` timesteps.
            conditions (Mapping[str, Any]): the window's conditions.
            frame_offset (int): window position of the first frame.

        Returns:
            Tensor: what the generator returns, the flow ``[B, F, C, H, W]`` float32.
        """
        return self.generator(
            x,
            timestep,
            framewise_conditions(conditions, frame_offset=frame_offset, num_frames=x.shape[1]),
            kv_cache=self.cache,
            frame_offset=frame_offset,
        )

    def predict_x0(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        frame_offset: int,
    ) -> torch.Tensor:
        """One generator call (:meth:`predict_flow`) and the x0 it implies.

        Args:
            x (Tensor): ``[B, F, C, H, W]`` latents x_t, cast to the generator's ``input_dtype``.
            timestep (Tensor): ``[B, F]`` timesteps, cast likewise.
            conditions (Mapping[str, Any]): the window's conditions.
            frame_offset (int): window position of the first frame.

        Returns:
            Tensor: x0 ``[B, F, C, H, W]`` float32, converted from the cast inputs, the generator's
            own view of x_t and t.
        """
        if self.generator.input_dtype is not None:
            x, timestep = cast_floating_tensors((x, timestep), self.generator.input_dtype)
        flow = self.predict_flow(x, timestep, conditions, frame_offset=frame_offset)
        return flow_to_x0(flow, x, timestep, self.scheduler)

    # Attribution (write_context, prefill_context): rerunning a finished block through the
    # generator to fill the KV cache is the scheme of CausVid (github.com/tianweiy/CausVid at
    # fab2440f, MIT) and Self Forcing (github.com/guandeh17/Self-Forcing, Apache-2.0; context
    # noise).
    def write_context(
        self,
        latents: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        frame_offset: int,
        noise: torch.Tensor | None = None,
    ) -> None:
        """Write clean latents into the KV cache at the context noise (clean when it is 0); the
        output is discarded.

        Args:
            latents (Tensor): ``[B, F, C, H, W]`` clean latents at window positions
                ``frame_offset ..``.
            conditions (Mapping[str, Any]): the window's conditions.
            frame_offset (int): window position of the first frame.
            noise (Tensor | None): ``[B F, C, H, W]`` re-noise; ``None`` draws it from ``rng``.
        """
        noised, timestep = noise_context(
            latents, self.scheduler, context_noise=self.context_noise, rng=self.rng, noise=noise
        )
        self.predict_x0(noised, timestep, conditions, frame_offset=frame_offset)

    def denoise(
        self,
        noisy: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        frame_offset: int,
        retest: Callable[[torch.Tensor], Mapping[str, Any]] | None = None,
    ) -> torch.Tensor:
        """Denoise one block at window positions ``frame_offset ..`` through the denoising steps.

        Args:
            noisy (Tensor): ``[B, F, C, H, W]`` entry noise.
            conditions (Mapping[str, Any]): the window's conditions.
            frame_offset (int): window position of the first frame.
            retest (Callable | None): the conditions of the denoising steps after the first, from
                the first step's x0: the predicted visibility re-tested on the block's own first
                estimate (Sec. 3.4).

        Returns:
            Tensor: x0 ``[B, F, C, H, W]`` float32.
        """

        def call(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            nonlocal conditions, retest
            x0 = self.predict_x0(x, t, conditions, frame_offset=frame_offset)
            if retest is not None:
                conditions, retest = retest(x0), None
            return x0

        return denoise_block(call, noisy, self.denoising_timesteps, self.scheduler, rng=self.rng)

    def prefill_context(
        self,
        context: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        ranges: Sequence[tuple[int, int]],
        context_write_noise: ContextWriteNoise | None = None,
    ) -> None:
        """Write the window's context range by range; each range attends to the ranges before it
        and to itself.

        Args:
            context (Tensor): ``[B, F, C, H, W]`` clean first frame | memory frames | recent
                context: every latent frame of the ranges (17 of the paper's window, 13 without
                memory frames).
            conditions (Mapping[str, Any]): the window's conditions.
            ranges (Sequence[tuple[int, int]]): the window positions ``(start, end)`` of each
                write, the frames ``start .. end - 1``
                (:attr:`~worldcast.sampling.window.WindowLayout.context_ranges`).
            context_write_noise (ContextWriteNoise | None): the noise of each write,
                :func:`~worldcast.sampling.schedulers.paired_context_noise` on the paper path;
                ``None`` draws from ``rng``.
        """
        covered = max(end for _, end in ranges)
        if context.shape[1] < covered:
            raise ValueError(
                f"the ranges cover {covered} latent frames, the context has {context.shape[1]}"
            )
        for i, (start, end) in enumerate(ranges):
            piece = context[:, start:end]
            noise = None
            if context_write_noise is not None:
                noise = context_write_noise(i, (len(piece) * (end - start), *piece.shape[2:]))
            self.write_context(piece, conditions, frame_offset=start, noise=noise)
