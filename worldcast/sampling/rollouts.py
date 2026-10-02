"""The block-causal sampler of one client: the plain prefix and the reconstituted blocks.

A client rolls out the plain prefix (:meth:`Sampler.rollout_prefix`), then generates every later
block from its window (:meth:`Sampler.generate_block`). docs/inference.md ("The block loop",
"Numerics that the paper's numbers depend on") gives the order of the RNG draws and the dtypes.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from worldcast.modeling.wan22.model import KVCache
from worldcast.utils.precision import cast_floating_tensors

from .schedulers import (
    FlowMatchScheduler,
    flow_to_x0,
    ladder_denoise,
    renoise_for_cache,
    warped_ladder,
)
from .window import BLOCK, context_block_ranges

__all__ = ["FRAMEWISE_KEYS", "CausalGenerator", "Sampler", "framewise_conditions"]

#: Conditions cut to a call's latent frames (axis 1); every other condition spans the window.
FRAMEWISE_KEYS = ("peer_alive", "peer_visible")


def framewise_conditions(
    conditions: Mapping[str, Any], *, start: int, num_frames: int
) -> dict[str, Any]:
    """The conditions of one call: :data:`FRAMEWISE_KEYS` narrowed to its latent frames (views)."""
    sliced = dict(conditions)
    for key in FRAMEWISE_KEYS:
        sliced[key] = conditions[key].narrow(1, start, num_frames)
    return sliced


class CausalGenerator(Protocol):
    """One causal DiT call on the KV cache, for example
    :class:`~worldcast.modeling.wan22.model.CausalGeneratorAdapter`.

    ``generator(noisy, timestep, conditions, kv_cache=, current_start=, cache_start=)`` takes
    ``noisy`` ``[B, F, C, H, W]`` and ``timestep`` ``[B, F]`` in ``input_dtype``, writes the keys
    and values of its tokens at ``[cache_start, cache_start + 252 F)`` and attends to the cache up
    to there; ``current_start // 252`` is the window position of its first latent frame (RoPE and
    control history). It returns the flow ``[B, F, C, H, W]`` float32 and draws no random numbers.
    """

    input_dtype: torch.dtype | None

    def __call__(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        kv_cache: KVCache,
        current_start: int,
        cache_start: int,
    ) -> torch.Tensor: ...


@dataclass
class Sampler:
    """The block-causal sampler of one client: generator, flow-matching table, ladder, KV cache.

    Attributes:
        generator (CausalGenerator): one causal DiT call.
        scheduler (FlowMatchScheduler): the flow-matching table.
        ladder (Tensor): ``[K]`` float32 rung timesteps (:func:`warped_ladder`).
        cache (KVCache): the KV cache.
        frame_seq_length (int): tokens per latent frame (252).
        context_noise (int): the timestep label of every context write (16).
        rng (torch.Generator | None): the stream of the ladder's re-noise and of the prefix's
            context writes; ``None`` is the global RNG of the latents' device. A client uses one
            stream for its whole rollout.
    """

    generator: CausalGenerator
    scheduler: FlowMatchScheduler
    ladder: torch.Tensor
    cache: KVCache
    frame_seq_length: int
    context_noise: int = 16
    rng: torch.Generator | None = None

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        generator: CausalGenerator,
        cache: KVCache,
        *,
        rng: torch.Generator | None = None,
    ) -> "Sampler":
        """From a :class:`worldcast.config.inference.InferenceConfig`.

        Args:
            cfg (InferenceConfig): the inference config.
            generator (CausalGenerator): its ``input_dtype`` must be the config's
                ``sampler.model_input_dtype``.
            cache (KVCache): the KV cache.
            rng (torch.Generator | None): see :class:`Sampler`.
        """
        want = getattr(torch, cfg.sampler.model_input_dtype)
        if getattr(generator, "input_dtype", None) != want:
            raise ValueError(
                f"generator.input_dtype={getattr(generator, 'input_dtype', None)} but the config"
                f" says model_input_dtype={cfg.sampler.model_input_dtype}"
            )
        if dict(cfg.sampler.framewise_condition_axes) != {key: 1 for key in FRAMEWISE_KEYS}:
            raise NotImplementedError(
                f"sampler.framewise_condition_axes: only {FRAMEWISE_KEYS} on axis 1 are implemented"
            )
        scheduler = FlowMatchScheduler.from_config(cfg.sampler.scheduler)
        return cls(
            generator=generator,
            scheduler=scheduler,
            ladder=warped_ladder(cfg.sampler.ladder, scheduler, warp=cfg.sampler.warp_ladder),
            cache=cache,
            frame_seq_length=cfg.model.frame_seq_length,
            context_noise=cfg.sampler.context_noise,
            rng=rng,
        )

    def call(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        start: int,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One generator call on latent frames at window positions ``start .. start + n - 1``.

        The call writes the frames' keys and values into the cache.

        Args:
            x (Tensor): ``[B, n, C, H, W]`` latents.
            timestep (Tensor): ``[B, n]`` timesteps.
            conditions (Mapping[str, Any]): the window's conditions.
            start (int): window position of the first frame.
            num_frames (int): ``n``.

        Returns:
            tuple[Tensor, Tensor]: the flow and x0, both float32; x0 is converted from the cast
            inputs, the model's own view of x_t and t.
        """
        if self.generator.input_dtype is not None:
            x, timestep = cast_floating_tensors((x, timestep), self.generator.input_dtype)
        offset = int(start) * self.frame_seq_length
        flow = self.generator(
            x,
            timestep,
            framewise_conditions(conditions, start=start, num_frames=num_frames),
            kv_cache=self.cache,
            current_start=offset,
            cache_start=offset,
        )
        return flow, flow_to_x0(flow, x, timestep, self.scheduler)

    # Attribution (commit, rollout_prefix, generate_block): rerunning a finished block through the
    # generator to fill the KV cache is the scheme of CausVid (github.com/tianweiy/CausVid at
    # fab2440f, MIT) and Self Forcing (github.com/guandeh17/Self-Forcing, Apache-2.0; context
    # noise).
    def commit(
        self,
        latents: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        start: int,
        noise: torch.Tensor | None = None,
    ) -> None:
        """Write clean latents into the KV cache at the context noise; the output is discarded.

        Args:
            latents (Tensor): ``[B, n, C, H, W]`` clean latents at window positions ``start ..``.
            conditions (Mapping[str, Any]): the window's conditions.
            start (int): window position of the first frame.
            noise (Tensor | None): ``[B n, C, H, W]`` re-noise; ``None`` draws it from ``rng``.
        """
        noised, timestep = renoise_for_cache(
            latents,
            context_noise=self.context_noise,
            scheduler=self.scheduler,
            noise=noise,
            rng=self.rng,
        )
        self.call(noised, timestep, conditions, start=start, num_frames=latents.shape[1])

    def denoise(
        self, noisy: torch.Tensor, conditions: Mapping[str, Any], *, start: int
    ) -> torch.Tensor:
        """Denoise one block at window positions ``start ..`` on the ladder.

        Args:
            noisy (Tensor): ``[B, n, C, H, W]`` entry noise.
            conditions (Mapping[str, Any]): the window's conditions.
            start (int): window position of the first frame.

        Returns:
            Tensor: x0 ``[B, n, C, H, W]`` float32.
        """
        return ladder_denoise(
            lambda x, t: self.call(x, t, conditions, start=start, num_frames=noisy.shape[1]),
            noisy,
            self.ladder,
            self.scheduler,
            rng=self.rng,
        )

    def rollout_prefix(
        self, noise: torch.Tensor, initial_latent: torch.Tensor, conditions: Mapping[str, Any]
    ) -> torch.Tensor:
        """The plain prefix: write the sink, then denoise and write ``n / 4`` blocks.

        Args:
            noise (Tensor): ``[B, n, C, H, W]`` entry noise (``n = 24``).
            initial_latent (Tensor): ``[B, 1, C, H, W]`` the recorded latent 0.
            conditions (Mapping[str, Any]): the round's conditions.

        Returns:
            Tensor: ``[B, 1 + n, C, H, W]`` in ``noise.dtype``.
        """
        batch, n = noise.shape[:2]
        if n % BLOCK:
            raise ValueError(f"the prefix length {n} is not a multiple of {BLOCK}")
        self.cache.reset()
        output = torch.zeros(
            [batch, n + 1, *noise.shape[2:]], device=noise.device, dtype=noise.dtype
        )
        output[:, :1] = initial_latent[:, :1]
        self.commit(initial_latent[:, :1], conditions, start=0)
        for start in range(1, n + 1, BLOCK):
            x0 = self.denoise(noise[:, start - 1 : start - 1 + BLOCK], conditions, start=start)
            output[:, start : start + BLOCK] = x0
            self.commit(x0, conditions, start=start)
        return output

    def prefill_context(
        self,
        context: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        ranges: Sequence[tuple[int, int]],
        block_noise: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        """Write the window's context range by range.

        Args:
            context (Tensor): ``[B, ctx, C, H, W]`` clean context latents.
            conditions (Mapping[str, Any]): the window's conditions.
            ranges (Sequence[tuple[int, int]]): ``(start, n)`` of each write
                (:func:`~worldcast.sampling.window.context_block_ranges`).
            block_noise (Callable | None): the re-noise of each range,
                :func:`~worldcast.sampling.schedulers.paired_cache_noise` on the paper path;
                ``None`` draws from ``rng``.
        """
        for i, (start, n) in enumerate(ranges):
            piece = context[:, start : start + n]
            noise = None
            if block_noise is not None:
                flat = (piece.shape[0] * piece.shape[1], *piece.shape[2:])
                noise = block_noise(i, start, n, flat, piece.device, piece.dtype)
            self.commit(piece, conditions, start=start, noise=noise)

    def generate_block(
        self,
        context: torch.Tensor,
        noisy: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        block_noise: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """One reconstituted block: empty the cache, write the window's context, denoise the target.

        Args:
            context (Tensor): ``[B, 17 | 13, C, H, W]`` clean sink | memory slot | recent.
            noisy (Tensor): ``[B, 4, C, H, W]`` entry noise of the target.
            conditions (Mapping[str, Any]): the window's conditions (21 or 17 latent frames).
            block_noise (Callable | None): see :meth:`prefill_context`.

        Returns:
            Tensor: the target's x0 ``[B, 4, C, H, W]`` float32.
        """
        ctx = context.shape[1]
        self.cache.reset()
        self.prefill_context(
            context,
            conditions,
            ranges=context_block_ranges(ctx + noisy.shape[1]),
            block_noise=block_noise,
        )
        return self.denoise(noisy, conditions, start=ctx)
