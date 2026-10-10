"""The training losses: the weighted flow-matching loss of Eq. (5) and the distribution matching
losses of stage 4.

**Flow matching** (stages 1-3). One sample (:func:`sample_flow_matching`) draws, in order, the
noise ``eps`` and one timestep per block (one per window in the bidirectional stages; stage 1 draws
it from a ``[B, 1]`` draw); ``x_t = (1 - sigma) x0 + sigma eps`` at the bf16 timestep, the target is
``v = eps - x0``, and the leading clean frames (the first frame, and under bidirectional attention
the memory frames) are pinned at ``t = 0``. With context noise (stage 3) the context copy is noised
per block to a timestep in [16, 32) with draws of its own, the first frame and the memory frames
included. The loss is ``mean(w (v_theta - v)^2)`` over every element with

    ``w = W(t) * m_frame * c_k * alpha_k / mean(alpha)``   (:func:`compose_weight`)

* ``W(t)``: Wan's training weight of the timestep (:func:`training_weight`); not in Eq. (5).
* ``m_frame``: a memory window supervises its target frames only (:func:`apply_frame_loss_mask`).
* ``c_k``: the memory weight ``1/2 (1 + m_k / mean_f(m))``, unit mean per target frame
  (:func:`memory_weight`).
* ``alpha_k``: the foreground weight (:func:`foreground_weight_map`), divided by its mean over the
  micro-batch.

**Distribution matching** (stage 4): :func:`distribution_matching_gradient`,
:func:`distribution_matching_loss`, :func:`critic_loss` and the score timestep
:func:`sample_score_timestep`, written from DMD and DMD2 and the flow-matching identities.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from worldcast.player_state import (
    PlayerState,
    integrate_camera_angles,
    latent_frame_rows,
    pack_substeps,
    project_players,
)
from worldcast.sampling.schedulers import (
    NUM_TRAIN_TIMESTEPS,
    TIMESTEP_SHIFT,
    FlowMatchScheduler,
    draw_noise,
    nearest_index,
    renoise_frames,
    shift_sigma,
    x0_to_flow,
)
from worldcast.sampling.window import CONTINUOUS_COLUMNS_KEY
from worldcast.utils.precision import GENERATOR_DTYPE

__all__ = [
    "BETA_MAX",
    "BETA_RADIUS",
    "CONTEXT_BAND",
    "FOREGROUND_LAMBDA",
    "FOREGROUND_MIN_SIGMA",
    "SCORE_TIMESTEP_RANGE",
    "FlowMatchingSample",
    "apply_frame_loss_mask",
    "compose_weight",
    "context_band_index_range",
    "critic_loss",
    "distribution_matching_gradient",
    "distribution_matching_loss",
    "distribution_matching_mask",
    "flow_matching_loss",
    "foreground_angles",
    "foreground_weight",
    "foreground_weight_map",
    "memory_weight",
    "pin_clean_frames",
    "sample_flow_matching",
    "sample_score_timestep",
    "sample_timestep_index",
    "sample_window_timestep_index",
    "training_weight",
    "training_weight_table",
]

#: The context noise of stage 3: timesteps in [16, 32).
CONTEXT_BAND = (16, 32)
#: lambda of the foreground weight: the weight at a visible player's centre before beta_p.
FOREGROUND_LAMBDA = 3.0
#: Floor of the foreground Gaussian's widths, latent cells.
FOREGROUND_MIN_SIGMA = 0.6
#: beta_p = min(BETA_MAX, max(1, BETA_RADIUS / r_p)), the boost of a distant player.
BETA_RADIUS, BETA_MAX = 1.5, 3.0
#: The score timestep is clamped to DMD's [0.02, 0.98] of the schedule.
SCORE_TIMESTEP_RANGE = (20, 980)


# ======================================================================================== timesteps
def sample_window_timestep_index(
    batch_size: int, num_frames: int, *, high: int, device: torch.device | None = None
) -> torch.Tensor:
    """Table indices ``[B, F]`` long, one per window: one ``randint`` of shape ``[B, 1]``, uniform
    on ``[0, high)`` from the global RNG of ``device``, repeated over the frames."""
    index = torch.randint(0, high, [batch_size, 1], device=device, dtype=torch.long)
    return index.repeat(1, num_frames)


def sample_timestep_index(
    batch_size: int,
    num_frames: int,
    *,
    frames_per_timestep: int,
    first_frame_alone: bool,
    low: int = 0,
    high: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Table indices ``[B, F]`` long, uniform on ``[low, high)``, from the global RNG of
    ``device``: one per run of ``frames_per_timestep`` latent frames.

    One ``randint`` of shape ``[B, F]``, whose first value of each run is copied over the run;
    with ``first_frame_alone`` frame 0 is a run of its own. The draw's shape is part of the RNG
    stream. Index ``i`` is the timestep ``scheduler.timesteps[i]`` (1000 down to about 5).

    Args:
        batch_size (int): B.
        num_frames (int): F.
        frames_per_timestep (int): latent frames that share a timestep (the whole window when
            bidirectional, a block of 4 under teacher forcing).
        first_frame_alone (bool): the first frame has a timestep of its own (teacher forcing).
        low (int): smallest index.
        high (int): one past the largest index.
        device (torch.device | None): the device of the draw.
    """
    timestep = torch.randint(low, high, [batch_size, num_frames], device=device, dtype=torch.long)
    # Attribution: copying each block's first index over the block (the reshape / [:, :, 1:] = [:,
    # :, 0:1] lines below) follows CausVid (github.com/tianweiy/CausVid at fab2440f, MIT, (c)
    # 2025-2026 Tianwei Yin) via Self Forcing (github.com/guandeh17/Self-Forcing, Apache-2.0).
    if first_frame_alone:
        from_second = timestep[:, 1:]
        from_second = from_second.reshape(from_second.shape[0], -1, frames_per_timestep)
        from_second[:, :, 1:] = from_second[:, :, 0:1]
        from_second = from_second.reshape(from_second.shape[0], -1)
        return torch.cat([timestep[:, 0:1], from_second], dim=1)
    timestep = timestep.reshape(timestep.shape[0], -1, frames_per_timestep)
    timestep[:, :, 1:] = timestep[:, :, 0:1]
    return timestep.reshape(timestep.shape[0], -1)


def training_weight_table(scheduler: FlowMatchScheduler) -> torch.Tensor:
    """Wan's training weight per table entry, ``[N]`` float32.

    ``y = exp(-2 ((t - N/2) / N)^2)`` over the table timesteps; ``W = (y - min y) N / sum(y - min
    y)``, so W averages to 1 over the table, peaks near t = 500 and is 0 at t = 1000.
    """
    n = int(scheduler.timesteps.numel())
    y = torch.exp(-2 * ((scheduler.timesteps - n / 2) / n) ** 2)
    y_shifted = y - y.min()
    return y_shifted * (n / y_shifted.sum())


def training_weight(timestep: torch.Tensor, scheduler: FlowMatchScheduler) -> torch.Tensor:
    """``W(t)`` (:func:`training_weight_table`) at the table entry nearest each timestep, compared
    in float32 as trained: ``[B, F]`` or ``[N]`` -> ``[B F]`` float32."""
    index = nearest_index(timestep, scheduler, dtype=torch.float32)
    return training_weight_table(scheduler).to(timestep.device)[index]


# =========================================================================== flow-matching inputs
def pin_clean_frames(
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    clean: torch.Tensor,
    target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``noisy``, ``timestep`` and ``target`` with the leading ``clean.shape[1]`` frames pinned:
    the clean latents copied in, timestep 0, target 0 (copies; the inputs are not modified).

    Args:
        noisy (Tensor): ``[B, F, C, H, W]`` x_t.
        timestep (Tensor): ``[B, F]``.
        clean (Tensor): ``[B, n, C, H, W]`` the clean latents of the leading ``n`` frames.
        target (Tensor | None): ``[B, F, C, H, W]`` the flow target, when there is one.
    """
    n_pin = int(clean.shape[1])
    noisy, timestep = noisy.clone(), timestep.clone()
    noisy[:, :n_pin] = clean
    timestep[:, :n_pin] = 0
    if target is not None:
        target = target.clone()
        target[:, :n_pin] = 0
    return noisy, timestep, target


def context_band_index_range(scheduler: FlowMatchScheduler) -> tuple[int, int]:
    """The half-open table-index range ``(lo, hi)`` of the timesteps in :data:`CONTEXT_BAND`
    (994-996 on the shift-5 table: t = 29.3, 24.5, 19.7)."""
    low, high = CONTEXT_BAND
    timesteps = [float(t) for t in scheduler.timesteps]
    below_high = sum(1 for t in timesteps if t < high)
    below_low = sum(1 for t in timesteps if t < low)
    return len(timesteps) - below_high, len(timesteps) - below_low


@dataclass
class FlowMatchingSample:
    """The inputs and the target of one flow-matching step.

    Attributes:
        noisy (Tensor): ``[B, F, C, H, W]`` x_t (the clean latents' dtype), the pinned frames
            clean.
        timestep (Tensor): ``[B, F]`` (bf16 on the paper path), 0 on the pinned frames.
        target (Tensor): ``[B, F, C, H, W]`` ``v = eps - x0``, 0 on the pinned frames.
        n_pin (int): leading frames pinned clean.
        noise (Tensor): ``[B, F, C, H, W]`` eps.
        context (Tensor): ``[B, F, C, H, W]`` the context copy of teacher forcing: x0, or x0
            noised to the context noise.
        context_timestep (Tensor | None): ``[B, F]`` its timesteps; ``None`` for a clean copy.
    """

    noisy: torch.Tensor
    timestep: torch.Tensor
    target: torch.Tensor
    n_pin: int
    noise: torch.Tensor
    context: torch.Tensor
    context_timestep: torch.Tensor | None


def sample_flow_matching(
    clean: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    n_pin: int,
    frames_per_timestep: int,
    first_frame_alone: bool,
    window_timestep: bool = False,
    timestep_dtype: torch.dtype = GENERATOR_DTYPE,
    context_noise: bool = False,
) -> FlowMatchingSample:
    """Noise, timesteps, x_t, target and the context copy of one step (module docstring).

    Args:
        clean (Tensor): ``[B, F, C, H, W]`` x0 (bf16 on the paper path).
        scheduler (FlowMatchScheduler): the shift-5 table.
        n_pin (int): frames pinned clean.
        frames_per_timestep (int): latent frames that share a timestep (41 bidirectional, 4 under
            teacher forcing).
        first_frame_alone (bool): frame 0 has a timestep of its own (teacher forcing).
        window_timestep (bool): one timestep per window from a ``[B, 1]`` draw (stage 1).
        timestep_dtype (torch.dtype): the trainer's compute dtype.
        context_noise (bool): the context copy is noised in :data:`CONTEXT_BAND`, its first frame
            and memory frames included (stage 3); otherwise it is clean.

    Returns:
        FlowMatchingSample: the draws ``eps``, the timestep indices and, with context noise, the
        context's indices and noise, in this order from the global RNG.
    """
    batch_size, num_frames = int(clean.shape[0]), int(clean.shape[1])
    high = NUM_TRAIN_TIMESTEPS
    runs = dict(
        frames_per_timestep=frames_per_timestep,
        first_frame_alone=first_frame_alone,
        device=clean.device,
    )
    noise = draw_noise(clean)
    if window_timestep:
        index = sample_window_timestep_index(batch_size, num_frames, high=high, device=clean.device)
    else:
        index = sample_timestep_index(batch_size, num_frames, high=high, **runs)
    timesteps = scheduler.timesteps.to(clean.device)
    timestep = timesteps[index].to(dtype=timestep_dtype, device=clean.device)
    noisy = renoise_frames(clean, timestep, scheduler, noise=noise.flatten(0, 1))
    noisy, timestep, target = pin_clean_frames(noisy, timestep, clean[:, :n_pin], noise - clean)
    if not context_noise:
        return FlowMatchingSample(noisy, timestep, target, n_pin, noise, clean.clone(), None)
    low, high = context_band_index_range(scheduler)
    context_index = sample_timestep_index(batch_size, num_frames, low=low, high=high, **runs)
    context_timestep = timesteps[context_index].to(dtype=timestep_dtype, device=clean.device)
    context = renoise_frames(clean, context_timestep, scheduler)
    return FlowMatchingSample(noisy, timestep, target, n_pin, noise, context, context_timestep)


# ========================================================================================== weights
def apply_frame_loss_mask(weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``weight`` ``[B, F, 1, 1, 1]`` times the frame mask ``[B, F]`` (0/1), scaled by ``F /
    n_supervised`` per sample, so that a memory window trains at the per-frame scale of an
    ordinary window."""
    m = mask.to(device=weight.device, dtype=weight.dtype)
    scale = m * (float(weight.shape[1]) / m.sum(dim=1))[:, None]
    return weight * scale.view(*scale.shape, 1, 1, 1)


def foreground_weight(
    uv: torch.Tensor, radius: torch.Tensor, visible: torch.Tensor, *, grid: tuple[int, int]
) -> torch.Tensor:
    """alpha_k = ``1 + (lambda - 1) max_p g_p(k) beta_p`` on the latent grid, ``[B, F, 1, h, w]``
    float32 (App. C, "Foreground weight").

    ``g_p`` is a unit-peak Gaussian centred ``r_p`` above the player's feet, with widths ``max(r_p
    / 2, 0.6)`` across and ``max(r_p, 0.6)`` along the grid's rows; only visible players count;
    ``beta_p = min(3, max(1, 1.5 / r_p))``.

    Args:
        uv (Tensor): ``[B, F, P, 2]`` feet in latent cells (column, row).
        radius (Tensor): ``[B, F, P]`` projected half body height ``r_p``, latent rows, at least
            half a row (:func:`worldcast.player_state.projection.project_players`).
        visible (Tensor): ``[B, F, P]`` bool, the GT label and in front of the camera.
        grid (tuple[int, int]): the latent grid ``(24, 42)``.
    """
    grid_h, grid_w = grid
    cols = torch.arange(grid_w, device=uv.device, dtype=torch.float32) + 0.5
    rows = torch.arange(grid_h, device=uv.device, dtype=torch.float32) + 0.5
    centre_u = uv[..., 0].float()
    centre_v = uv[..., 1].float() - radius.float()
    sigma_v = radius.float().clamp(min=FOREGROUND_MIN_SIGMA)
    sigma_u = (0.5 * radius.float()).clamp(min=FOREGROUND_MIN_SIGMA)
    du = (cols[None, None, None, None, :] - centre_u[..., None, None]) / sigma_u[..., None, None]
    dv = (rows[None, None, None, :, None] - centre_v[..., None, None]) / sigma_v[..., None, None]
    gaussian = torch.exp(-0.5 * (du * du + dv * dv))
    gaussian = gaussian * visible.to(gaussian.dtype)[..., None, None]
    beta = (BETA_RADIUS / radius.float().clamp(min=1e-4)).clamp(min=1.0, max=BETA_MAX)
    gaussian = gaussian * beta[..., None, None]
    return (1.0 + (FOREGROUND_LAMBDA - 1.0) * gaussian.amax(dim=2)).unsqueeze(2)


def foreground_angles(
    batch: Mapping[str, torch.Tensor], *, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(yaw, pitch)`` ``[B, F, P]`` float32 degrees of every player at the latent frames.

    A memory window with the continuous camera carries the angles (the gathered columns 0 and 1);
    otherwise they are integrated from video frame 0
    (:func:`~worldcast.player_state.states.integrate_camera_angles`), the pitch clamped as a sum to
    [-89, 89] (as trained; the player state field clamps it row by row), and read at the latent
    frames' rows.
    """
    states = batch["player_states"]
    continuous = batch.get(CONTINUOUS_COLUMNS_KEY)
    if continuous is not None:
        continuous = continuous.to(states.device).float()
        yaw, pitch = continuous[..., 0].contiguous(), continuous[..., 1].contiguous()
    else:
        substeps = pack_substeps(
            batch["player_control_substeps"], batch["player_control_substep_valid"]
        )
        yaw, pitch = integrate_camera_angles(states[:, :, 0], substeps, clamp_pitch_sum=True)
        rows = latent_frame_rows(int(states.shape[2])).to(states.device)
        yaw = yaw.index_select(2, rows).permute(0, 2, 1).contiguous()
        pitch = pitch.index_select(2, rows).permute(0, 2, 1).contiguous()
    if device is not None:
        yaw, pitch = yaw.to(device).float(), pitch.to(device).float()
    return yaw, pitch


def foreground_weight_map(
    player_state: PlayerState,
    visible: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    grid: tuple[int, int],
) -> torch.Tensor:
    """alpha_k of a window, ``[B, F, 1, h, w]`` float32, before its normalisation.

    Every player is projected through the client's camera (the projection of the player state
    field, on the latent grid); a player counts where the GT label marks it visible in any of the
    latent frame's video frames and it is in front of the camera.

    Args:
        player_state (PlayerState): every player's latent-rate states.
        visible (Tensor): ``[B, F, P]`` bool, the GT visibility labels at the latent frames
            (:func:`~worldcast.player_state.visibility.visible_latent_frames`).
        batch (Mapping[str, Tensor]): the window, for the angles (:func:`foreground_angles`).
        grid (tuple[int, int]): the latent grid ``(24, 42)``.
    """
    device = player_state.xyz.device
    yaw, pitch = foreground_angles(batch, device=device)
    xyz, num_frames = player_state.xyz, int(player_state.xyz.shape[1])
    slot = player_state.client_slot.long()
    client_xyz = xyz.gather(2, slot.view(-1, 1, 1, 1).expand(-1, num_frames, 1, 3)).squeeze(2)
    pick = slot.view(-1, 1, 1).expand(-1, num_frames, 1)
    client_yaw = yaw.gather(2, pick).squeeze(-1)
    client_pitch = pitch.gather(2, pick).squeeze(-1)
    uv, radius, in_front, _ = project_players(
        xyz, client_xyz, client_yaw, client_pitch, grid_h=grid[0], grid_w=grid[1]
    )
    return foreground_weight(uv, radius, visible & in_front, grid=grid)


def memory_weight(
    memory_mask: torch.Tensor, num_frames: int, *, grid: tuple[int, int]
) -> torch.Tensor:
    """c_k of a memory window on the latent grid, ``[B, F, 1, h, w]`` float32.

    On the last ``T`` (target) frames ``c_k = 1/2 (1 + m_k / mean_f(m))``, ``mean_f`` the frame's
    share of marked tokens, and 1 on a frame without marked tokens: unit mean per frame. 1
    elsewhere.

    Args:
        memory_mask (Tensor): ``[B, T, gh, gw]`` m_k on the token grid.
        num_frames (int): the window's latent frames.
        grid (tuple[int, int]): the latent grid; each token covers an exact block of it.
    """
    batch, frames, token_h, token_w = memory_mask.shape
    m = memory_mask.to(torch.float32)
    share = m.flatten(2).mean(dim=2)[:, :, None, None]
    weight = (1.0 + m / share.clamp_min(1e-8)) / 2.0
    weight = torch.where(share > 0, weight, torch.ones_like(weight))
    weight = weight.repeat_interleave(grid[0] // token_h, dim=-2)
    weight = weight.repeat_interleave(grid[1] // token_w, dim=-1)
    out = torch.ones(batch, num_frames, 1, *grid, dtype=torch.float32, device=memory_mask.device)
    out[:, num_frames - frames :, 0] = weight
    return out


def compose_weight(
    timestep: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    n_pin: int,
    frame_loss_mask: torch.Tensor | None = None,
    c_k: torch.Tensor | None = None,
    alpha_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """The per-element weight ``W(t) * m_frame * c_k * alpha_k / mean(alpha)``.

    Args:
        timestep (Tensor): ``[B, F]`` the training timesteps, after pinning.
        scheduler (FlowMatchScheduler): the table W is read from.
        n_pin (int): the leading pinned frames, weight 0.
        frame_loss_mask (Tensor | None): ``[B, F]`` of a memory window.
        c_k (Tensor | None): the memory weight ``[B, F, 1, H, W]`` (:func:`memory_weight`).
        alpha_k (Tensor | None): the foreground weight ``[B, F, 1, H, W]``
            (:func:`foreground_weight_map`), divided here by its mean over the whole tensor.

    Returns:
        Tensor: ``[B, F, 1, 1, 1]`` float32 without the maps, ``[B, F, 1, H, W]`` with.
    """
    batch_size, num_frames = int(timestep.shape[0]), int(timestep.shape[1])
    weight = training_weight(timestep, scheduler).unflatten(0, (batch_size, num_frames))
    weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).clone()
    weight[:, :n_pin] = 0
    if frame_loss_mask is not None:
        weight = apply_frame_loss_mask(weight, frame_loss_mask)
    if c_k is not None:
        weight = weight * c_k.to(weight.dtype)
    if alpha_k is not None:
        normalised = alpha_k / alpha_k.mean().clamp_min(1e-8)
        weight = weight * normalised.to(weight.dtype)
    return weight


def flow_matching_loss(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """``mean(weight * (pred - target)^2)`` in float32 over every element, 0-dim."""
    loss = (pred.float() - target.float()).pow(2)
    return (loss if weight is None else loss * weight).mean()


# =========================================================================== distillation (stage 4)
def sample_score_timestep(
    batch_size: int, num_frames: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """The timestep the rollout is diffused to for the teacher and the critic, one per sample,
    ``[B, F]`` float32.

    An integer ``s`` uniform on [0, 1000) (one ``randint`` of shape ``[B, 1]``), warped by the
    schedule's shift, ``t = 1000 shift_sigma(s / 1000)``, and clamped to
    :data:`SCORE_TIMESTEP_RANGE`. ``add_noise`` snaps ``t`` to the nearest table entry.
    """
    draw = sample_window_timestep_index(
        batch_size, num_frames, high=NUM_TRAIN_TIMESTEPS, device=device
    )
    shifted = shift_sigma(draw / NUM_TRAIN_TIMESTEPS, TIMESTEP_SHIFT) * NUM_TRAIN_TIMESTEPS
    return shifted.clamp(*SCORE_TIMESTEP_RANGE)


def distribution_matching_mask(rollout: torch.Tensor) -> torch.Tensor:
    """Bool, shaped like the rollout, False on the first frame (recorded: it gets no gradient)."""
    mask = torch.ones_like(rollout, dtype=torch.bool)
    mask[:, :1] = False
    return mask


def distribution_matching_gradient(
    critic_x0: torch.Tensor, teacher_x0: torch.Tensor, rollout: torch.Tensor
) -> torch.Tensor:
    """The distribution matching gradient g, in x0 form, normalised per sample (DMD).

    ``g = (critic_x0 - teacher_x0) / max(mean_k |x_k - teacher_x0_k|, 1e-8)``, the mean over every
    axis but the batch; non-finite entries are mapped by ``torch.nan_to_num``.

    Args:
        critic_x0 (Tensor): ``[B, F, C, H, W]`` the critic's x0 at x_t.
        teacher_x0 (Tensor): ``[B, F, C, H, W]`` the frozen teacher's x0 at x_t (no CFG).
        rollout (Tensor): ``[B, F, C, H, W]`` the generator's rollout x (bf16).

    Returns:
        Tensor: g in the promoted dtype (float32).
    """
    distance = (teacher_x0 - rollout).abs()
    scale = distance.mean(dim=list(range(1, distance.ndim)), keepdim=True).clamp(min=1e-8)
    return torch.nan_to_num((critic_x0 - teacher_x0) / scale)


def distribution_matching_loss(
    rollout: torch.Tensor, gradient: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """The generator's loss ``1/2 mean_k (x_k - sg[x_k - g_k])^2`` over the masked elements, whose
    gradient with respect to the rollout is ``g_k / N``; 0-dim float64.

    The target ``x - g`` is formed in the inputs' dtype and detached; both sides are cast to
    float64 for the squared error.
    """
    target = (rollout - gradient).detach().to(torch.float64)
    prediction = rollout.to(torch.float64)
    if mask is not None:
        prediction, target = prediction[mask], target[mask]
    return 0.5 * torch.nn.functional.mse_loss(prediction, target)


def critic_loss(
    rollout: torch.Tensor,
    critic_x0: torch.Tensor,
    noise: torch.Tensor,
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: FlowMatchScheduler,
) -> torch.Tensor:
    """The critic's flow-matching loss on the rollout without its first frame: its flow ``(x_t -
    x0_hat) / sigma_t`` (:func:`~worldcast.sampling.schedulers.x0_to_flow`) regresses onto ``eps -
    x``, the plain mean over every (sample, frame) pair.

    Args:
        rollout (Tensor): ``[B, F, C, H, W]`` the rollout x without gradient (bf16).
        critic_x0 (Tensor): ``[B, F, C, H, W]`` the critic's x0 at ``noisy``.
        noise (Tensor): ``[B, F, C, H, W]`` eps.
        noisy (Tensor): ``[B, F, C, H, W]`` x_t as the critic saw it.
        timestep (Tensor): ``[B, F]`` the score timestep, the first frame not pinned.
        scheduler (FlowMatchScheduler): the table.

    Returns:
        Tensor: 0-dim, float32 on the paper path.
    """

    def scored(frames: torch.Tensor) -> torch.Tensor:
        return frames[:, 1:].flatten(0, 1)

    velocity = scored(noise) - scored(rollout)
    predicted = x0_to_flow(scored(critic_x0), scored(noisy), scored(timestep), scheduler)
    return (predicted - velocity).square().mean()
