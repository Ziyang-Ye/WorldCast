"""Wan2.2's flow-matching table, the few-step ladder, flow -> x0, and the noise of a rollout."""

import hashlib
from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch

__all__ = [
    "BlockCall",
    "FlowMatchScheduler",
    "LadderExit",
    "draw_noise",
    "entry_noise",
    "flow_to_x0",
    "ladder_denoise",
    "paired_cache_noise",
    "renoise_for_cache",
    "renoise_frames",
    "run_ladder",
    "shift_sigma",
    "table_sigma",
    "warped_ladder",
]


def shift_sigma(sigma, shift: float):
    """The timestep shift ``shift sigma / (1 + (shift - 1) sigma)`` of a tensor or a float."""
    return shift * sigma / (1 + (shift - 1) * sigma)


class FlowMatchScheduler:
    """The 1000-entry flow-matching table of Wan2.2.

    ``sigmas`` ``[1000]`` float32 is ``linspace(1, 0, 1001)[:-1]`` shifted by :func:`shift_sigma`;
    ``timesteps`` is ``1000 sigmas`` (1000 .. 4.99).

    Args:
        num_train_timesteps (int): entries of the table.
        shift (float): the timestep shift (Wan2.2: 5).
    """

    def __init__(self, *, num_train_timesteps: int = 1000, shift: float = 5.0) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        sigmas = torch.linspace(1.0, 0.0, num_train_timesteps + 1)[:-1]
        self.sigmas: torch.Tensor = shift_sigma(sigmas, shift)
        self.timesteps: torch.Tensor = self.sigmas * num_train_timesteps

    @classmethod
    def from_config(cls, cfg) -> "FlowMatchScheduler":
        """From a :class:`worldcast.config.inference.SchedulerConfig`."""
        return cls(num_train_timesteps=cfg.num_train_timesteps, shift=cfg.shift)

    # Attribution (timestep_index, add_noise): add_noise is DiffSynth-Studio's (Apache-2.0); the
    # batched nearest-entry argmin, the [N, 1, 1, 1] sigma and type_as follow CausVid
    # (github.com/tianweiy/CausVid at fab2440f, MIT, (c) 2025-2026 Tianwei Yin) via Self Forcing
    # (github.com/guandeh17/Self-Forcing, Apache-2.0).
    def timestep_index(self, timestep: torch.Tensor) -> torch.Tensor:
        """Index of the table entry nearest each timestep, ``[N]`` -> ``[N]`` long (in float32)."""
        timesteps = self.timesteps.to(timestep.device)
        return torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """``(1 - sigma) clean + sigma noise`` in ``noise.dtype``, sigma of the nearest table entry.

        Args:
            clean (Tensor): ``[N, C, H, W]``.
            noise (Tensor): ``[N, C, H, W]``.
            timestep (Tensor): ``[N]`` or ``[B, F]``; the level snaps (t = 16 applies sigma(997)).
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        sigma = self.sigmas.to(noise.device)[self.timestep_index(timestep)].reshape(-1, 1, 1, 1)
        return ((1 - sigma) * clean + sigma * noise).type_as(noise)


def table_sigma(
    timestep: torch.Tensor,
    scheduler: FlowMatchScheduler,
    like: torch.Tensor,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """sigma of the table entry nearest each timestep, shaped to broadcast against ``like``.

    The nearest entry is the first index minimising ``|t - t_i|`` in ``dtype``; the bf16 timesteps
    the generator sees (1000 / 936 / 832 / 624) map to indices 0 / 255 / 502 / 751.

    Args:
        timestep (Tensor): ``[N]`` or ``[B, F]`` on ``like``'s device.
        scheduler (FlowMatchScheduler): the table.
        like (Tensor): the tensor sigma multiplies.
        dtype (torch.dtype): dtype of the lookup and of the result.

    Returns:
        Tensor: ``timestep.shape`` followed by ``like.ndim - timestep.ndim`` singleton axes.
    """
    level_t = scheduler.timesteps.to(device=like.device, dtype=dtype)
    level_sigma = scheduler.sigmas.to(device=like.device, dtype=dtype)
    nearest = (timestep.reshape(-1, 1) - level_t).abs().argmin(dim=1)
    singleton_axes = (1,) * (like.ndim - timestep.ndim)
    return level_sigma[nearest].reshape(*timestep.shape, *singleton_axes)


def flow_to_x0(
    flow: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler: FlowMatchScheduler
) -> torch.Tensor:
    """x0 implied by a velocity prediction, ``x0 = x_t - sigma_t v``, in ``flow.dtype``.

    Formed in float64 and rounded once.

    Args:
        flow (Tensor): ``[B, F, C, H, W]`` the velocity ``v = eps - x0``.
        xt (Tensor): ``[B, F, C, H, W]`` as the generator saw it (after its input cast).
        timestep (Tensor): ``[B, F]`` as the generator saw it.
        scheduler (FlowMatchScheduler): the table.
    """
    sigma = table_sigma(timestep, scheduler, flow)
    x0 = xt.to(device=flow.device, dtype=torch.float64) - sigma * flow.to(torch.float64)
    return x0.to(flow.dtype)


def warped_ladder(
    rungs: Sequence[int], scheduler: FlowMatchScheduler, *, warp: bool = True
) -> torch.Tensor:
    """The ladder's timesteps, ``rung -> table[1000 - rung]``, float32 on the CPU (the paper's:
    1000, 937.5, 833.33, 625); only the warped ladder is implemented."""
    if not warp:
        raise NotImplementedError(
            "only the warped ladder (warp_denoising_step: true) is implemented"
        )
    timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
    return timesteps[scheduler.num_train_timesteps - torch.tensor(list(rungs), dtype=torch.long)]


def draw_noise(like: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Standard normal noise like ``like``; ``None`` draws from the global RNG of its device."""
    if generator is None:
        return torch.randn_like(like)
    return torch.randn(like.shape, generator=generator, dtype=like.dtype, device=like.device)


#: ``call(x_t, t) -> (v, x0)``: one generator call on a block, ``x_t`` ``[B, F, C, H, W]``, ``t``
#: ``[B, F]``.
BlockCall = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class LadderExit(NamedTuple):
    """The exit rung of a block: its input x_t ``[B, F, C, H, W]``, timestep ``[B, F]`` and x0."""

    input: torch.Tensor
    timestep: torch.Tensor
    x0: torch.Tensor


def renoise_frames(
    x0: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: FlowMatchScheduler,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Diffuse x0 ``[B, F, C, H, W]`` to ``timestep`` ``[B, F]`` with one draw in x0's dtype."""
    frames = x0.flatten(0, 1)
    eps = draw_noise(frames, rng)
    return scheduler.add_noise(frames, eps, timestep).unflatten(0, x0.shape[:2])


def run_ladder(
    call: BlockCall,
    noisy: torch.Tensor,
    ladder: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    exit_rung: int,
    rng: torch.Generator | None = None,
    renoise_dtype: torch.dtype | None = None,
) -> LadderExit:
    """Few-step sampling of one block from rung 0 down to ``exit_rung``.

    Rung k predicts x0 from x_{t_k}; before the exit rung x0 is re-noised to t_{k+1}
    (:func:`renoise_frames`), so the order is call, draw, call, draw, ...

    Args:
        call (BlockCall): the generator on the block.
        noisy (Tensor): ``[B, F, C, H, W]`` entry noise.
        ladder (Tensor): ``[K]`` rung timesteps, filled per frame in their own dtype.
        scheduler (FlowMatchScheduler): the table.
        exit_rung (int): the last rung, in ``[0, K)``.
        rng (torch.Generator | None): the stream of the re-noise (``None``: the global RNG).
        renoise_dtype (torch.dtype | None): dtype of each re-noised input (``None``: x0's).
    """
    if not 0 <= exit_rung < len(ladder):
        raise ValueError(f"exit rung {exit_rung} is outside a {len(ladder)}-rung ladder")
    levels = ladder.tolist()
    batch, frames = noisy.shape[:2]

    def frame_timestep(level) -> torch.Tensor:
        return torch.full((batch, frames), level, dtype=ladder.dtype, device=noisy.device)

    x_t = noisy
    for rung in range(exit_rung):
        _, x0 = call(x_t, frame_timestep(levels[rung]))
        x_t = renoise_frames(x0, frame_timestep(levels[rung + 1]), scheduler, rng)
        if renoise_dtype is not None:
            x_t = x_t.to(renoise_dtype)
    t_exit = frame_timestep(levels[exit_rung])
    _, x0 = call(x_t, t_exit)
    return LadderExit(input=x_t, timestep=t_exit, x0=x0)


def ladder_denoise(
    call: BlockCall,
    noisy: torch.Tensor,
    ladder: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Denoise one block on the whole ladder, the re-noised inputs cast to ``noisy.dtype``.

    Returns:
        Tensor: the last x0 ``[B, F, C, H, W]`` float32.
    """
    return run_ladder(
        call,
        noisy,
        ladder,
        scheduler,
        exit_rung=len(ladder) - 1,
        rng=rng,
        renoise_dtype=noisy.dtype,
    ).x0


def renoise_for_cache(
    latents: torch.Tensor,
    *,
    context_noise: int,
    scheduler: FlowMatchScheduler,
    noise: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The input of a context write: clean latents re-noised to ``context_noise``.

    Args:
        latents (Tensor): ``[B, F, C, H, W]`` clean latents.
        context_noise (int): the timestep label (16); the level is the snapped table entry.
        scheduler (FlowMatchScheduler): the table.
        noise (Tensor | None): ``[B F, C, H, W]`` noise in place of a draw from ``rng``.
        rng (torch.Generator | None): the stream of the draw (``None``: the global RNG).

    Returns:
        tuple[Tensor, Tensor]: the re-noised latents in their dtype, and the ``[B, F]`` float32
        timestep label, both on the latents' device.
    """
    if not context_noise:
        raise NotImplementedError(
            "a clean (t = 0) context write is not implemented; the paper's is 16"
        )
    timestep = torch.full(
        [latents.shape[0], latents.shape[1]],
        float(context_noise),
        device=latents.device,
        dtype=torch.float32,
    )
    flat = latents.flatten(0, 1)
    if noise is None:
        eps = draw_noise(flat, rng)
    else:
        eps = torch.as_tensor(noise).to(device=flat.device, dtype=flat.dtype)
        if eps.shape != flat.shape:
            raise ValueError(f"cache noise {tuple(eps.shape)} != block {tuple(flat.shape)}")
    noised = scheduler.add_noise(flat, eps, timestep.flatten(0, 1)).unflatten(0, latents.shape[:2])
    return noised.to(latents.dtype), timestep


def paired_cache_noise(
    seed: int, block_start: int, n_ranges: int, n_recent_blocks: int
) -> Callable[..., torch.Tensor]:
    """The context-write noise of one reconstituted block, keyed on ``(seed, s, role)``.

    Range 0 is ``sink``, the last ``n_recent_blocks`` ranges are ``recent0 ..``, the others
    ``slot{i - 1}``, so the sink and recent rows get the same noise with or without a memory entry.
    Each role draws float32 on the CPU from a generator seeded with
    ``int(sha256(f"{seed}|{s}|{role}")[:16], 16) % (2**63 - 1)``.

    Returns:
        Callable: ``block_noise(i, start, n, shape, device, dtype)`` for
        :meth:`~worldcast.sampling.rollouts.Sampler.prefill_context`.
    """

    def role_of(i: int) -> str:
        if i == 0:
            return "sink"
        back = n_ranges - i  # 1 is the last range
        if back <= n_recent_blocks:
            return f"recent{n_recent_blocks - back}"
        return f"slot{i - 1}"

    def block_noise(i, start, n, shape, device, dtype):
        key = f"{int(seed)}|{int(block_start)}|{role_of(int(i))}"
        h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**63 - 1)
        g = torch.Generator(device="cpu").manual_seed(h)
        return torch.randn(tuple(shape), generator=g, dtype=torch.float32).to(
            device=device, dtype=dtype
        )

    return block_noise


def entry_noise(
    seed: int,
    requested_latents: int,
    latents: int,
    latent_shape: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """The rollout's entry noise ``[1, latents - 1, C, H, W]``; block s starts from
    ``noise[:, s - 1 : s + 3]``.

    Drawn from ``torch.Generator("cpu").manual_seed(seed)`` for the requested length, then cut to
    the effective one. The CPU normal kernel is platform dependent: the paper's draw does not
    reproduce on arm64.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    shape = (1, int(requested_latents) - 1, *[int(v) for v in latent_shape])
    full = torch.randn(shape, generator=g)
    return full[:, : int(latents) - 1].to(device, dtype)
