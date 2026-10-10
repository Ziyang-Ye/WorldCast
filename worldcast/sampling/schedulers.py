"""Wan2.2's flow-matching table, the timesteps of the denoising steps, flow <-> x0, and the noise
of a rollout.

``flow`` is the paper's velocity ``v = eps - x0`` (Eq. (5)) under Wan2.2's name; the generator
predicts it.
"""

import hashlib
from collections.abc import Callable, Sequence
from typing import NamedTuple, TypeVar

import numpy as np
import torch

from worldcast.utils.precision import GENERATOR_DTYPE

__all__ = [
    "NUM_TRAIN_TIMESTEPS",
    "TIMESTEP_SHIFT",
    "BlockCall",
    "ContextWriteNoise",
    "DenoisingExit",
    "FlowMatchScheduler",
    "context_noise_seed",
    "denoise_block",
    "draw_noise",
    "entry_noise",
    "flow_to_x0",
    "nearest_index",
    "noise_context",
    "paired_context_noise",
    "renoise_frames",
    "run_denoising_steps",
    "shift_denoising_steps",
    "shift_sigma",
    "table_sigma",
    "x0_to_flow",
]

#: Entries of Wan2.2's flow-matching table; timesteps run in ``[0, 1000]``.
NUM_TRAIN_TIMESTEPS = 1000
#: Wan2.2's timestep shift.
TIMESTEP_SHIFT = 5.0

_Sigma = TypeVar("_Sigma", torch.Tensor, np.ndarray, float)


def shift_sigma(sigma: _Sigma, shift: float) -> _Sigma:
    """The timestep shift ``shift sigma / (1 + (shift - 1) sigma)``."""
    return shift * sigma / (1 + (shift - 1) * sigma)


class FlowMatchScheduler:
    """The 1000-entry flow-matching table of Wan2.2.

    ``sigmas`` ``[1000]`` float32 is ``linspace(1, 0, 1001)[:-1]`` shifted by
    :data:`TIMESTEP_SHIFT`; ``timesteps`` is ``1000 sigmas`` (1000 .. 4.98).
    """

    def __init__(self) -> None:
        sigmas = torch.linspace(1.0, 0.0, NUM_TRAIN_TIMESTEPS + 1)[:-1]
        self.sigmas: torch.Tensor = shift_sigma(sigmas, TIMESTEP_SHIFT)
        self.timesteps: torch.Tensor = self.sigmas * NUM_TRAIN_TIMESTEPS

    # Attribution: add_noise is DiffSynth-Studio's (Apache-2.0); the batched nearest-entry argmin,
    # the [N, 1, 1, 1] sigma and type_as follow CausVid (github.com/tianweiy/CausVid at fab2440f,
    # MIT, (c) 2025-2026 Tianwei Yin) via Self Forcing (github.com/guandeh17/Self-Forcing,
    # Apache-2.0).
    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """``(1 - sigma) clean + sigma noise`` in ``noise.dtype``, sigma of the nearest table
        entry. The lookup and the sum are float32, as the reference runs re-noised
        (:func:`flow_to_x0` is the float64 one).

        Args:
            clean (Tensor): ``[N, C, H, W]``.
            noise (Tensor): ``[N, C, H, W]``.
            timestep (Tensor): ``[N]``, or ``[B, F]`` with ``B F = N``; the level snaps (t = 16
                applies sigma(997)).
        """
        if clean.ndim != 4 or timestep.numel() != clean.shape[0]:
            raise ValueError(
                f"add_noise takes [N, C, H, W] frames and N timesteps, got {tuple(clean.shape)}"
                f" and {tuple(timestep.shape)}"
            )
        sigma = table_sigma(timestep.flatten(), self, noise, dtype=torch.float32)
        return ((1 - sigma) * clean + sigma * noise).type_as(noise)


def nearest_index(
    timestep: torch.Tensor, scheduler: FlowMatchScheduler, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """The table entry nearest each timestep: the first index minimising ``|t - t_i|`` in
    ``dtype``, ``[N]`` long for the ``N`` timesteps of any shape. The bf16 timesteps the generator
    sees (1000 / 936 / 832 / 624) map to indices 0 / 255 / 502 / 751."""
    level_t = scheduler.timesteps.to(device=timestep.device, dtype=dtype)
    return (timestep.reshape(-1, 1) - level_t).abs().argmin(dim=1)


def table_sigma(
    timestep: torch.Tensor,
    scheduler: FlowMatchScheduler,
    like: torch.Tensor,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """sigma of the table entry nearest each timestep (:func:`nearest_index`), shaped to broadcast
    against ``like``; the lookup is in ``dtype``, float64 by default, as the reference runs
    converted the flow.

    Args:
        timestep (Tensor): ``[N]`` or ``[B, F]`` on ``like``'s device.
        scheduler (FlowMatchScheduler): the table.
        like (Tensor): the tensor sigma multiplies.
        dtype (torch.dtype): dtype of the lookup and of the result.

    Returns:
        Tensor: ``timestep.shape`` followed by ``like.ndim - timestep.ndim`` singleton axes.
    """
    level_sigma = scheduler.sigmas.to(device=like.device, dtype=dtype)
    nearest = nearest_index(timestep, scheduler, dtype)
    singleton_axes = (1,) * (like.ndim - timestep.ndim)
    return level_sigma[nearest].reshape(*timestep.shape, *singleton_axes)


def flow_to_x0(
    flow: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler: FlowMatchScheduler
) -> torch.Tensor:
    """x0 implied by a flow prediction, ``x0 = x_t - sigma_t flow``, in ``flow.dtype``.

    Formed in float64 and rounded once, as the reference runs did: a float32 product rounds to
    other latents.

    Args:
        flow (Tensor): ``[B, F, C, H, W]`` the flow ``eps - x0``.
        xt (Tensor): ``[B, F, C, H, W]`` as the generator saw it (after its input cast).
        timestep (Tensor): ``[B, F]`` as the generator saw it.
        scheduler (FlowMatchScheduler): the table.
    """
    sigma = table_sigma(timestep, scheduler, flow)
    x0 = xt.to(device=flow.device, dtype=torch.float64) - sigma * flow.to(torch.float64)
    return x0.to(flow.dtype)


def x0_to_flow(
    x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler: FlowMatchScheduler
) -> torch.Tensor:
    """The flow of an x0 prediction, ``(x_t - x0) / sigma_t``, in ``x0.dtype``: the inverse of
    :func:`flow_to_x0`, formed in float64 and rounded once.

    Args:
        x0 (Tensor): ``[..., C, H, W]`` the predicted clean latents.
        xt (Tensor): x_t, shaped like ``x0``, as the model saw it.
        timestep (Tensor): the leading axes of ``x0``, as the model saw it (never 0: sigma > 0).
        scheduler (FlowMatchScheduler): the table.
    """
    sigma = table_sigma(timestep, scheduler, x0)
    flow = (xt.to(device=x0.device, dtype=torch.float64) - x0.to(torch.float64)) / sigma
    return flow.to(x0.dtype)


def shift_denoising_steps(steps: Sequence[int], scheduler: FlowMatchScheduler) -> torch.Tensor:
    """The timesteps of denoising steps given on the unshifted schedule, ``step -> table[1000 -
    step]`` for steps in ``[0, 1000]``, float32 on the CPU (the paper's 1000, 750, 500, 250 give
    1000, 937.5, 833.33, 625)."""
    steps = torch.tensor(list(steps), dtype=torch.long)
    if bool(((steps < 0) | (steps > NUM_TRAIN_TIMESTEPS)).any()):
        raise ValueError(f"denoising steps lie in [0, {NUM_TRAIN_TIMESTEPS}], got {steps.tolist()}")
    timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
    return timesteps[NUM_TRAIN_TIMESTEPS - steps]


def draw_noise(like: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
    """Standard normal noise like ``like`` from the stream ``rng``, a generator on its device;
    ``None`` draws from the global RNG of that device."""
    if rng is None:
        return torch.randn_like(like)
    return torch.randn(like.shape, generator=rng, dtype=like.dtype, device=like.device)


#: ``call(x_t, t) -> x0``: one generator call on a block, ``x_t`` ``[B, F, C, H, W]``, ``t``
#: ``[B, F]``, and the x0 it implies (:func:`flow_to_x0`).
BlockCall = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class DenoisingExit(NamedTuple):
    """The exit step of a block: its input x_t ``[B, F, C, H, W]``, timestep ``[B, F]`` and x0."""

    input: torch.Tensor
    timestep: torch.Tensor
    x0: torch.Tensor


def renoise_frames(
    x0: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    rng: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Diffuse x0 ``[B, F, C, H, W]`` to ``timestep`` ``[B, F]``, in x0's dtype.

    Args:
        x0 (Tensor): the clean latents.
        timestep (Tensor): the level of every frame.
        scheduler (FlowMatchScheduler): the table.
        rng (torch.Generator | None): the stream of the one draw (``None``: the global RNG).
        noise (Tensor | None): ``[B F, C, H, W]`` noise in place of the draw.
    """
    frames = x0.flatten(0, 1)
    if noise is None:
        noise = draw_noise(frames, rng)
    else:
        noise = torch.as_tensor(noise).to(device=frames.device, dtype=frames.dtype)
        if noise.shape != frames.shape:
            raise ValueError(f"noise {tuple(noise.shape)} != frames {tuple(frames.shape)}")
    return scheduler.add_noise(frames, noise, timestep).unflatten(0, x0.shape[:2])


def run_denoising_steps(
    call: BlockCall,
    noisy: torch.Tensor,
    denoising_timesteps: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    exit_step: int,
    rng: torch.Generator | None = None,
    renoise_dtype: torch.dtype | None = None,
) -> DenoisingExit:
    """Few-step sampling of one block from denoising step 0 down to ``exit_step``.

    Step k predicts x0 from x_{t_k}; before the exit step x0 is re-noised to t_{k+1}
    (:func:`renoise_frames`), so the order is call, draw, call, draw, ...

    Args:
        call (BlockCall): the generator on the block.
        noisy (Tensor): ``[B, F, C, H, W]`` entry noise.
        denoising_timesteps (Tensor): ``[K]`` timesteps of the denoising steps
            (:func:`shift_denoising_steps`), filled per frame in their own dtype.
        scheduler (FlowMatchScheduler): the table.
        exit_step (int): the last step, in ``[0, K)``.
        rng (torch.Generator | None): the stream of the re-noise (``None``: the global RNG).
        renoise_dtype (torch.dtype | None): dtype of each re-noised input (``None``: x0's).

    Returns:
        DenoisingExit: the exit step's input, timestep and x0.
    """
    if not 0 <= exit_step < len(denoising_timesteps):
        raise ValueError(
            f"exit step {exit_step} is outside {len(denoising_timesteps)} denoising steps"
        )
    levels = denoising_timesteps.tolist()
    batch, frames = noisy.shape[:2]

    def frame_timestep(level: float) -> torch.Tensor:
        return torch.full(
            (batch, frames), level, dtype=denoising_timesteps.dtype, device=noisy.device
        )

    x_t = noisy
    for k in range(exit_step):
        x0 = call(x_t, frame_timestep(levels[k]))
        x_t = renoise_frames(x0, frame_timestep(levels[k + 1]), scheduler, rng=rng)
        if renoise_dtype is not None:
            x_t = x_t.to(renoise_dtype)
    t_exit = frame_timestep(levels[exit_step])
    return DenoisingExit(input=x_t, timestep=t_exit, x0=call(x_t, t_exit))


def denoise_block(
    call: BlockCall,
    noisy: torch.Tensor,
    denoising_timesteps: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Denoise one block through all the denoising steps, the re-noised inputs cast to
    ``noisy.dtype`` (the generator's bf16 on the paper path, as the reference runs fed them).

    Args:
        call (BlockCall): the generator on the block.
        noisy (Tensor): ``[B, F, C, H, W]`` entry noise.
        denoising_timesteps (Tensor): ``[K]`` timesteps of the denoising steps.
        scheduler (FlowMatchScheduler): the table.
        rng (torch.Generator | None): the stream of the re-noise (``None``: the global RNG).

    Returns:
        Tensor: the last x0 ``[B, F, C, H, W]`` float32.
    """
    return run_denoising_steps(
        call,
        noisy,
        denoising_timesteps,
        scheduler,
        exit_step=len(denoising_timesteps) - 1,
        rng=rng,
        renoise_dtype=noisy.dtype,
    ).x0


def noise_context(
    latents: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    context_noise: int,
    rng: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The input of a context write: clean latents re-noised to the context noise.

    Args:
        latents (Tensor): ``[B, F, C, H, W]`` clean latents.
        scheduler (FlowMatchScheduler): the table.
        context_noise (int): the timestep label (16; the level is the snapped table entry), or 0:
            a clean context, the latents themselves, without a draw.
        rng (torch.Generator | None): the stream of the draw (``None``: the global RNG).
        noise (Tensor | None): ``[B F, C, H, W]`` noise in place of a draw from ``rng``.

    Returns:
        tuple[Tensor, Tensor]: the latents to write, in their dtype, and the ``[B, F]`` float32
        timestep label, both on the latents' device.
    """
    timestep = torch.full(
        latents.shape[:2], float(context_noise), device=latents.device, dtype=torch.float32
    )
    if not context_noise:
        return latents, timestep
    return renoise_frames(latents, timestep, scheduler, rng=rng, noise=noise), timestep


#: ``noise(i, shape) -> eps``: the noise of a window's ``i``-th context write, ``shape``
#: ``[B n, C, H, W]``; :func:`renoise_frames` moves it to the latents' device and dtype.
ContextWriteNoise = Callable[[int, Sequence[int]], torch.Tensor]


def context_noise_seed(seed: int, block_start: int, role: str) -> int:
    """The seed of one context write's noise: ``int(sha256(f"{seed}|{block_start}|{role}")[:16],
    16) % (2**63 - 1)``, as the reference runs keyed it (:func:`paired_context_noise`)."""
    key = f"{int(seed)}|{int(block_start)}|{role}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**63 - 1)


def paired_context_noise(
    seed: int, block_start: int, num_ranges: int, num_recent_blocks: int
) -> ContextWriteNoise:
    """The context-write noise of one block generated from its window, keyed on ``(seed, s,
    role)``.

    Each range has a role: the first frame (range 0), the recent context (the last
    ``num_recent_blocks`` ranges) and the memory frames (the others), named ``sink``, ``recent0 ..``
    and ``slot{i - 1}`` in the key, as the reference runs keyed them; so the first frame and the
    recent context get the same noise with or without a memory entry. Each role draws float32 on
    the CPU from a generator seeded with its :func:`context_noise_seed`.

    Args:
        seed (int): the rollout's seed.
        block_start (int): ``s``, the first latent frame of the block.
        num_ranges (int): context writes of the window
            (:attr:`~worldcast.sampling.window.WindowLayout.context_ranges`).
        num_recent_blocks (int): blocks of recent context (3).

    Returns:
        ContextWriteNoise: float32 noise on the CPU, for
        :meth:`~worldcast.sampling.sampler.Sampler.prefill_context`.
    """

    def role_of(i: int) -> str:
        if i == 0:
            return "sink"
        back = num_ranges - i  # 1 is the last range
        if back <= num_recent_blocks:
            return f"recent{num_recent_blocks - back}"
        return f"slot{i - 1}"

    def noise(i: int, shape: Sequence[int]) -> torch.Tensor:
        role_seed = context_noise_seed(seed, block_start, role_of(int(i)))
        g = torch.Generator(device="cpu").manual_seed(role_seed)
        return torch.randn(tuple(shape), generator=g, dtype=torch.float32)

    return noise


def entry_noise(
    seed: int,
    requested_num_latents: int,
    num_latents: int,
    latent_shape: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = GENERATOR_DTYPE,
) -> torch.Tensor:
    """The rollout's entry noise ``[1, num_latents - 1, C, H, W]``; block s starts from
    ``noise[:, s - 1 : s + 3]``.

    Drawn from ``torch.Generator("cpu").manual_seed(seed)`` for the requested length, then cut to
    the effective one. The CPU normal kernel is platform dependent: the paper's draw does not
    reproduce on arm64.

    Args:
        seed (int): the rollout's seed.
        requested_num_latents (int): latent frames the run asked for (the length of the draw).
        num_latents (int): latent frames the rollout generates, at most the requested ones.
        latent_shape (Sequence[int]): ``(C, H, W)`` of a latent frame.
        device (torch.device | str | None): where the noise goes.
        dtype (torch.dtype): its dtype, the generator's.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    shape = (1, int(requested_num_latents) - 1, *[int(v) for v in latent_shape])
    full = torch.randn(shape, generator=g)
    return full[:, : int(num_latents) - 1].to(device, dtype)
