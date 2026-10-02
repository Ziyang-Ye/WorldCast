"""The player state field (Sec. 3.2, Eq. (2); App. "The player state field in detail").

For each latent frame the field places every other player on the generator's 12 x 21 token grid,
``[B, F, 23, 12, 21]`` float32 (channel-major per frame, zero where nobody is written).

Splatting (:func:`splat_weights`): each written player gets the isotropic Gaussian kernel of ATI,
``w_p(u) = e_p exp(-||x_u - mu_p||^2 / 2 sigma^2)``, centred on the body (the projected feet raised
by the projected radius), with ``e_p`` the visibility confidence in ``[0.3, 1]``
(``worldcast.player_state.visibility``). The kernel is ``exp(-d^2 T)`` in ATI's
short-edge-normalised coordinates; the configured ``T = 220`` is capped by the 0.5-token sigma floor
to ``T = 72`` on 12 x 21, which is exactly sigma = half a token (:func:`splat_temperature`).

Composition (:func:`compose_field`): at each token the two players with the largest depth-discounted
weight ``w_p(u) exp(-z~_p)`` (``z~`` the log-compressed depth) are merged:
``c(u) = min(1, sum_{p in S(u)} w_p(u))`` and ``F_a(u) = sum_{p in S(u)} w_p(u) a(s_p)``.

=====  ===========  ===================================================  ========================
index  name         value per player                                     merge at token u
=====  ===========  ===================================================  ========================
0      coverage                                                          ``min(1, sum w)``
1      depth        ``log1p(max(z, 0)) / log1p(4096)`` clamped to        top-2 weighted sum
                    [0, 1], z the forward distance from the eye (u)
2, 3   sin_yaw,     sine and cosine of the player's yaw minus the        top-2 weighted sum
       cos_yaw      client's
4      team         +1 enemy, -1 teammate                                top-2 weighted sum
5      headcount    ``min(sum_p max_u w_p(u), 4) / 4``, one per frame    broadcast
6-14   controls     share of the latent frame's 16 valid substeps with   top-2 weighted sum
                    the key held: forward, back, move_left, move_right,
                    jump, duck, speed, attack, reload
15-18  weapon_0..3  the learned 4-d embedding of the held weapon         top-2 weighted sum
19     dying        ``exp(-tsd / 5)`` on corpse frames (tsd in latent    ``max_p w^c_p dying_p``
                    frames since death)
20     corpse                                                            ``max_p w^c_p``
21     live_id      0.20 + 0.02 seat (teammate), 0.50 + 0.02 seat        top-1 value x ``c(u)``
                    (enemy)
22     corpse_id    0.80 + 0.02 seat                                     arg-max value x
                                                                         ``max_p w^c_p``
=====  ===========  ===================================================  ========================

``w^c_p`` are the corpse weights: the same kernel at the frozen position (the last alive frame),
with half the projected radius, no confidence and no visibility gate (only ``depth > 1``, a known
death site and not the client). docs/inference.md, "Paper vs code", lists where this is more
specific than the paper's text (items 8-11, 15).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .attributes import (
    CORPSE_RADIUS_FACTOR,
    FIELD_CONTROLS,
    PAPER_ACTION_BUTTONS,
    WEAPON_CHANNELS,
    control_fractions,
    control_indices,
    corpse_id_plane,
    death_planes,
    duck_index,
    enemy_mask,
    headcount,
    live_id_values,
    team_sign,
    weapon_values,
)
from .projection import project_view
from .tables import (
    CAMERA_DELTA_SCALE,
    STATE_BASE_DIM,
    STATE_CONTINUOUS_DIM,
    check_block_alive_agreement,
    death_state_for_table,
)
from .visibility import CONFIDENCE_FLOOR, CONFIDENCE_SMOOTH, live_eligibility, visibility_confidence

__all__ = [
    "DEPTH_LOG_SCALE",
    "FIELD_CHANNELS",
    "CHANNEL_NAMES",
    "FieldConfig",
    "splat_temperature",
    "splat_weights",
    "log_depth",
    "compose_field",
    "build_field",
]

#: Depth channel: ``log1p(depth) / log1p(4096)``, so depths up to a map's diagonal land in [0, 1].
DEPTH_LOG_SCALE = math.log1p(4096.0)

CHANNEL_NAMES = (
    ("coverage", "depth", "sin_yaw", "cos_yaw", "team", "headcount")
    + FIELD_CONTROLS
    + tuple(f"weapon_{k}" for k in range(WEAPON_CHANNELS))
    + ("dying", "corpse", "live_id", "corpse_id")
)
FIELD_CHANNELS = len(CHANNEL_NAMES)


@dataclass(frozen=True)
class FieldConfig:
    """Field settings; the defaults are the paper's (config ``model.player_field``).

    Attributes:
        grid_h (int): token rows (12: the 24 x 42 latent grid patchified by two).
        grid_w (int): token columns (21).
        camera_delta_scale (float): degrees per unit camera delta.
        duck_button_index (int): crouch column of the substep buttons (drives the body height).
        control_indices (tuple[int, ...]): button columns of the control channels, in channel order.
        weapon_channels (int): width of the weapon embedding.
        splat_temperature (float): configured ATI temperature (220; capped by the sigma floor).
        splat_sigma_floor (float): smallest kernel sigma, tokens (0.5).
        splat_topk (int): players merged per token (2).
        confidence_floor (float): lowest confidence ``e_p`` of a written player.
        confidence_smooth (float): EMA alpha of the confidence.
        num_frame_per_block (int): latent frames per block (the confidence resets at block starts).
        independent_first_frame (bool): frame 0 is its own block (the sink).
    """

    grid_h: int = 12
    grid_w: int = 21
    camera_delta_scale: float = CAMERA_DELTA_SCALE
    duck_button_index: int = duck_index(PAPER_ACTION_BUTTONS)
    control_indices: tuple[int, ...] = control_indices(PAPER_ACTION_BUTTONS, FIELD_CONTROLS)
    weapon_channels: int = WEAPON_CHANNELS
    splat_temperature: float = 220.0
    splat_sigma_floor: float = 0.5
    splat_topk: int = 2
    confidence_floor: float = CONFIDENCE_FLOOR
    confidence_smooth: float = CONFIDENCE_SMOOTH
    num_frame_per_block: int = 4
    independent_first_frame: bool = True

    @property
    def channels(self) -> int:
        """Field width: 6 geometry + controls + weapon + 2 death + 2 identity (23 for the paper)."""
        return 6 + len(self.control_indices) + int(self.weapon_channels) + 2 + 2

    @classmethod
    def from_action_buttons(
        cls, action_buttons: Sequence[str], controls: Sequence[str] = FIELD_CONTROLS, **overrides
    ) -> "FieldConfig":
        """The config for a button layout (config ``data.action_buttons``) and control channels."""
        return cls(
            duck_button_index=duck_index(action_buttons),
            control_indices=control_indices(action_buttons, controls),
            **overrides,
        )


# -------------------------------------------------------------------------------------------- splat
def splat_temperature(
    grid_h: int, grid_w: int, *, temperature: float = 220.0, sigma_floor: float = 0.5
) -> float:
    """The ATI temperature after the sigma floor, ``min(T, (min(H, W) / 2)^2 / (2 floor^2))``.

    One token is ``2 / min(H, W)`` ATI units, so ``sigma_tokens = min(H, W) / (2 sqrt(2 T))``; 72.0
    on 12 x 21. A floor of 0 leaves ``T`` as configured.
    """
    if sigma_floor <= 0.0:
        return float(temperature)
    short_edge = float(min(int(grid_h), int(grid_w)))
    cap = (short_edge / 2.0) ** 2 / (2.0 * sigma_floor**2)
    return float(min(float(temperature), cap))


def splat_weights(
    uv: torch.Tensor,
    radius: torch.Tensor,
    eligible: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    temperature: float,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """ATI splat weights ``w_p(u)`` of every player on the token grid.

    The kernel is isotropic; the radius only raises its centre from the feet to the body.

    Args:
        uv (torch.Tensor): ``[B, F, P, 2]`` feet position, token units.
        radius (torch.Tensor): ``[B, F, P]`` projected radius, token rows.
        eligible (torch.Tensor): ``[B, F, P]`` gate (bool or 0/1).
        grid_h (int): token rows.
        grid_w (int): token columns.
        temperature (float): from :func:`splat_temperature`.
        confidence (torch.Tensor | None): ``[B, F, P]`` ``e_p``, or None.

    Returns:
        torch.Tensor: ``[B, F, P, H, W]`` float32 in ``[0, 1]``.
    """
    device = uv.device
    m = float(min(int(grid_h), int(grid_w)))
    cols = torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5
    rows = torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5
    grid_x = (2.0 * cols / float(grid_w) - 1.0) * (float(grid_w) / m)
    grid_y = (2.0 * rows / float(grid_h) - 1.0) * (float(grid_h) / m)
    centre_u = uv[..., 0].float()
    centre_v = uv[..., 1].float() - radius.float()
    px = (2.0 * centre_u / float(grid_w) - 1.0) * (float(grid_w) / m)
    py = (2.0 * centre_v / float(grid_h) - 1.0) * (float(grid_h) / m)
    dx = grid_x[None, None, None, None, :] - px[..., None, None]
    dy = grid_y[None, None, None, :, None] - py[..., None, None]
    dist2 = dx * dx + dy * dy
    weight = torch.exp(-dist2 * temperature)
    weight = weight * eligible.to(weight.dtype)[..., None, None]
    if confidence is not None:
        weight = weight * confidence.to(weight.dtype)[..., None, None]
    return weight


def log_depth(depth: torch.Tensor) -> torch.Tensor:
    """Log-compressed depth ``z~ = clamp(log1p(max(z, 0)) / log1p(4096), 0, 1)``."""
    return (torch.log1p(depth.clamp(min=0.0)) / DEPTH_LOG_SCALE).clamp(0.0, 1.0)


# -------------------------------------------------------------------------------------- composition
def compose_field(
    weights: torch.Tensor,
    depth: torch.Tensor,
    relative_yaw: torch.Tensor,
    enemy: torch.Tensor,
    controls: torch.Tensor | None,
    weapons: torch.Tensor,
    *,
    dying_plane: torch.Tensor,
    corpse_plane: torch.Tensor,
    corpse_ids: torch.Tensor,
    topk: int = 2,
) -> torch.Tensor:
    """Merge the players' splats and attributes into the field (top-2 depth-discounted merge).

    Args:
        weights (torch.Tensor): ``[B, F, P, H, W]`` live splat weights.
        depth (torch.Tensor): ``[B, F, P]`` forward depth, u.
        relative_yaw (torch.Tensor): ``[B, F, P]`` radians.
        enemy (torch.Tensor): ``[B, F, P]`` bool, eligible and not on the client's team.
        controls (torch.Tensor | None): ``[B, F, P, K]`` held fractions, or None (no control
            channels).
        weapons (torch.Tensor): ``[B, F, P, C]`` weapon embedding values.
        dying_plane (torch.Tensor): ``[B, F, H, W]`` (:mod:`worldcast.player_state.attributes`).
        corpse_plane (torch.Tensor): ``[B, F, H, W]``.
        corpse_ids (torch.Tensor): ``[B, F, H, W]``.
        topk (int): players merged per token.

    Returns:
        torch.Tensor: ``[B, F, 6 + K + C + 4, H, W]`` in the channel order of the module docstring.
    """
    scaled = log_depth(depth)
    k = min(int(topk), int(weights.shape[2]))
    rank_key = weights * torch.exp(-scaled)[..., None, None]
    _, index = torch.topk(rank_key, k=k, dim=2)
    picked = weights.gather(2, index)
    coverage = picked.sum(dim=2).clamp(0.0, 1.0)

    def resolve(per_player: torch.Tensor) -> torch.Tensor:
        spread = per_player.to(weights.dtype)[..., None, None].expand_as(weights)
        return (spread.gather(2, index) * picked).sum(dim=2)

    def resolve_winner(per_player: torch.Tensor) -> torch.Tensor:
        # the top-ranked player's value times the coverage: a blend of two identities names nobody
        spread = per_player.to(weights.dtype)[..., None, None].expand_as(weights)
        return spread.gather(2, index[:, :, :1]).squeeze(2) * coverage

    count = headcount(weights)
    planes = [
        coverage,
        resolve(scaled),
        resolve(torch.sin(relative_yaw)),
        resolve(torch.cos(relative_yaw)),
        resolve(team_sign(enemy, weights.dtype)),
        count[..., None, None].expand_as(coverage),
    ]
    if controls is not None:
        planes += [resolve(controls[..., k_]) for k_ in range(int(controls.shape[-1]))]
    planes += [resolve(weapons[..., k_]) for k_ in range(int(weapons.shape[-1]))]
    planes += [dying_plane.to(weights.dtype), corpse_plane.to(weights.dtype)]
    planes += [resolve_winner(live_id_values(enemy)), corpse_ids.to(weights.dtype)]
    return torch.stack(planes, dim=2)


# -------------------------------------------------------------------------------------------- field
def build_field(
    peer_states: torch.Tensor,
    peer_actions: torch.Tensor,
    observer_slot: torch.Tensor,
    team_ids: torch.Tensor,
    alive: torch.Tensor,
    visible: torch.Tensor,
    weapons: torch.Tensor,
    weapon_embedding: torch.Tensor,
    *,
    frame_offset: int = 0,
    video_frames: int | None = None,
    config: FieldConfig = FieldConfig(),
) -> torch.Tensor:
    """The player state field of the decoded frames, from the player-state conditions.

    The field is computed over the whole window the table spans (the camera integral and the
    confidence EMA run along it) and the decoded frames ``[frame_offset, frame_offset +
    video_frames)`` are returned. ``alive`` and ``visible`` may be sliced to those frames (the
    sampler slices them per block); they are zero-padded back.

    Args:
        peer_states (torch.Tensor): ``[B, F, P, 6 | 13]`` peer state table (whole window).
        peer_actions (torch.Tensor): ``[B, F, P, 16, 14]`` packed substeps (whole window).
        observer_slot (torch.Tensor): ``[B]`` the client's seat.
        team_ids (torch.Tensor): ``[B, P]`` engine team ids.
        alive (torch.Tensor): ``[B, f, P]`` 0/1, ``f`` = ``video_frames`` or ``F``.
        visible (torch.Tensor): ``[B, f, P]`` 0/1 visibility gate.
        weapons (torch.Tensor): ``[B, F, P]`` weapon ids (whole window).
        weapon_embedding (torch.Tensor): ``[52, 4]`` the injector's weapon embedding weight.
        frame_offset (int): first decoded frame.
        video_frames (int | None): decoded frames (default ``F - frame_offset``).
        config (FieldConfig): the field settings.

    Returns:
        torch.Tensor: ``[B, video_frames, 23, 12, 21]`` float32 (paper config).
    """
    device = peer_states.device
    peer_actions = peer_actions.to(device=device)
    if peer_states.ndim != 4 or peer_states.shape[-1] not in (STATE_BASE_DIM, STATE_CONTINUOUS_DIM):
        raise ValueError(
            f"peer_states must be [B, F, P, {STATE_BASE_DIM}] (contiguous window) or "
            f"[B, F, P, {STATE_CONTINUOUS_DIM}] (gathered window), got {tuple(peer_states.shape)}"
        )
    if peer_actions.ndim != 5 or peer_actions.shape[:3] != peer_states.shape[:3]:
        raise ValueError(
            "peer_actions must be [B, F, P, Q, A] aligned with peer_states, got"
            f" {tuple(peer_actions.shape)} vs {tuple(peer_states.shape)}"
        )
    if int(weapon_embedding.shape[-1]) != int(config.weapon_channels):
        raise ValueError(
            f"weapon embedding width {int(weapon_embedding.shape[-1])} != config.weapon_channels "
            f"{config.weapon_channels}"
        )
    num_frames = int(peer_states.shape[1])
    frame_offset = int(frame_offset)
    video_frames = num_frames - frame_offset if video_frames is None else int(video_frames)
    if frame_offset < 0 or video_frames < 0 or frame_offset + video_frames > num_frames:
        raise ValueError(
            f"decoded frames [{frame_offset}, {frame_offset + video_frames}) lie outside the"
            f" {num_frames}-frame window the peer table covers"
        )
    grid = dict(grid_h=int(config.grid_h), grid_w=int(config.grid_w))
    camera = dict(
        duck_button_index=int(config.duck_button_index),
        camera_delta_scale=float(config.camera_delta_scale),
    )

    # live players: projection, gate, confidence, team
    view = project_view(peer_states, peer_actions, observer_slot, **grid, **camera)
    gate = live_eligibility(
        view.in_front,
        alive=alive,
        visible=visible,
        observer_slot=observer_slot,
        frame_offset=frame_offset,
    )
    confidence = visibility_confidence(
        gate.visible_window,
        num_frame_per_block=config.num_frame_per_block,
        independent_first_frame=config.independent_first_frame,
        floor=config.confidence_floor,
        smooth=config.confidence_smooth,
    )
    enemy = enemy_mask(team_ids, observer_slot, gate.eligible)

    # corpses: frozen position, half the radius, no visibility gate, no confidence
    death = death_state_for_table(peer_states)
    check_block_alive_agreement(
        death.alive, gate.alive_window, frame_offset=frame_offset, num_frames=int(alive.shape[1])
    )
    frozen = project_view(death.frozen_states, peer_actions, observer_slot, **grid, **camera)
    corpse_eligible = frozen.in_front & death.corpse.to(device) & ~gate.is_self
    temperature = splat_temperature(
        config.grid_h,
        config.grid_w,
        temperature=config.splat_temperature,
        sigma_floor=config.splat_sigma_floor,
    )
    corpse_weights = splat_weights(
        frozen.uv,
        frozen.radius * CORPSE_RADIUS_FACTOR,
        corpse_eligible,
        **grid,
        temperature=temperature,
    )
    dying_plane, corpse_plane = death_planes(corpse_weights, death.dying)
    corpse_ids = corpse_id_plane(corpse_weights)

    weights = splat_weights(
        view.uv, view.radius, gate.eligible, **grid, temperature=temperature, confidence=confidence
    )
    field = compose_field(
        weights,
        view.depth,
        view.relative_yaw,
        enemy,
        control_fractions(peer_actions, config.control_indices),
        weapon_values(weapons.to(device), weapon_embedding),
        dying_plane=dying_plane,
        corpse_plane=corpse_plane,
        corpse_ids=corpse_ids,
        topk=config.splat_topk,
    )
    return field[:, frame_offset : frame_offset + video_frames]
