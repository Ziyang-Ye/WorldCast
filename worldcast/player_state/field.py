"""The player state field (Sec. 3.2; App. "The player state field in detail", Splatting and
Composition).

For each latent frame the field places every other player on the generator's 12 x 21 token grid,
``[B, F, 23, 12, 21]`` float32 (channel-major per frame, zero where nobody is written).

Splatting (:func:`splat`): each written player gets the isotropic Gaussian kernel of ATI,
``w_p(u) = e_p exp(-||x_u - mu_p||^2 / 2 sigma^2)``, centred on the body (the projected feet raised
by the projected radius), with ``e_p`` the visibility confidence in ``[0.3, 1]``
(``worldcast.player_state.visibility``). The kernel is ``exp(-d^2 T)`` in ATI's
short-edge-normalised coordinates with ``T = 72``, which is sigma = half a token on 12 x 21.

Composition (:func:`compose`, Eq. (6)): at each token the two players with the largest
depth-discounted weight ``w_p(u) exp(-z~_p)`` (``z~`` the log-compressed depth) are merged:
``c(u) = min(1, sum_{p in S(u)} w_p(u))`` and ``F_a(u) = sum_{p in S(u)} w_p(u) a(s_p)``.

=====  ===========  ===================================================  ========================
index  channel      value per player                                     merge at token u
=====  ===========  ===================================================  ========================
0      coverage                                                          ``min(1, sum w)``
1      depth        ``log1p(max(z, 0)) / log1p(4096)`` clamped to        top-2 weighted sum
                    [0, 1], z the forward distance from the eye (u)
2, 3   yaw          sine and cosine of the player's yaw minus the        top-2 weighted sum
                    client's
4      team         +1 enemy, -1 teammate                                top-2 weighted sum
5      headcount    ``min(sum_p max_u w_p(u), 4) / 4``, one per frame    broadcast
6-14   controls     share of the latent frame's 16 valid substeps with   top-2 weighted sum
                    the key held: forward, back, move_left, move_right,
                    jump, duck, speed, attack, reload
15-18  weapon       the learned 4-d embedding of the held weapon         top-2 weighted sum
19     dying        ``exp(-tsd / 5)`` on corpse frames (tsd in latent    ``max_p w^c_p dying_p``
                    frames since death)
20     corpse                                                            ``max_p w^c_p``
21     identity     live: 0.20 + 0.02 slot (teammate), 0.50 + 0.02       top-1 value x ``c(u)``
                    slot (enemy)
22     identity     corpse: 0.80 + 0.02 slot                             arg-max value x
                                                                         ``max_p w^c_p``
=====  ===========  ===================================================  ========================

``w^c_p`` are the corpse weights: the same kernel at the frozen position (the last alive frame),
with half the projected radius, no confidence and no visibility gate (only ``depth > 1``, a known
death site and not the client).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from worldcast.data.latents import BLOCK, TOKEN_GRID
from worldcast.modeling.state_injector import (
    FIELD_CHANNEL_NAMES,
    FIELD_CONTROLS,
    WEAPON_CHANNELS,
    log_compressed_depth,
)
from worldcast.modeling.wan22.model import FieldBuilder

from .attributes import (
    CORPSE_RADIUS_FACTOR,
    control_fractions,
    corpse_id_plane,
    death_planes,
    enemy_mask,
    headcount,
    live_id_values,
    team_sign,
    weapon_values,
)
from .projection import project_view
from .states import (
    FIELD_CONDITION_KEYS,
    DeathState,
    check_block_alive_agreement,
    death_state_for_table,
    has_continuous_columns,
)
from .visibility import CONFIDENCE_FLOOR, live_eligibility, visibility_confidence

__all__ = [
    "SPLAT_TEMPERATURE",
    "TOP_K",
    "PlayerStateFieldConfig",
    "compose",
    "corpse_weights",
    "field_builder",
    "player_state_field",
    "splat",
]

#: The temperature ``T`` of the kernel ``exp(-d^2 T)``, ``d`` in units of half the grid's short
#: edge: sigma = ``1 / sqrt(2 T)`` = 1 / 12 of them, half a token on the 12 x 21 token grid.
SPLAT_TEMPERATURE = 72.0
#: Players merged per token.
TOP_K = 2


# ------------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class PlayerStateFieldConfig:
    """Settings of the player state field; the defaults are the paper's (App. "The player state
    field in detail").

    Attributes:
        confidence_floor (float): lowest confidence ``e_p`` of a written player.
        frames_per_block (int): latent frames per block of the generator's attention: the
            confidence restarts at block starts.
        first_frame_alone (bool): latent frame 0 (the first frame) is a block of its own.
    """

    confidence_floor: float = CONFIDENCE_FLOOR
    frames_per_block: int = BLOCK
    first_frame_alone: bool = True


# -------------------------------------------------------------------------------------------- splat
def splat(
    uv: torch.Tensor,
    radius: torch.Tensor,
    eligible: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
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
    weight = torch.exp(-dist2 * SPLAT_TEMPERATURE)
    weight = weight * eligible.to(weight.dtype)[..., None, None]
    if confidence is not None:
        weight = weight * confidence.to(weight.dtype)[..., None, None]
    return weight


# -------------------------------------------------------------------------------------- composition
def compose(
    weights: torch.Tensor,
    depth: torch.Tensor,
    relative_yaw: torch.Tensor,
    enemy: torch.Tensor,
    controls: torch.Tensor,
    weapons: torch.Tensor,
    *,
    dying_plane: torch.Tensor,
    corpse_plane: torch.Tensor,
    corpse_ids: torch.Tensor,
) -> torch.Tensor:
    """Merge the players' splats and attributes into the field (top-2 depth-discounted merge).

    Args:
        weights (torch.Tensor): ``[B, F, P, H, W]`` live splat weights.
        depth (torch.Tensor): ``[B, F, P]`` forward depth, u.
        relative_yaw (torch.Tensor): ``[B, F, P]`` radians.
        enemy (torch.Tensor): ``[B, F, P]`` bool, eligible and not on the client's team.
        controls (torch.Tensor): ``[B, F, P, 9]`` held fractions (``attributes.control_fractions``).
        weapons (torch.Tensor): ``[B, F, P, 4]`` weapon embedding values.
        dying_plane (torch.Tensor): ``[B, F, H, W]`` (``attributes.death_planes``).
        corpse_plane (torch.Tensor): ``[B, F, H, W]``.
        corpse_ids (torch.Tensor): ``[B, F, H, W]`` (``attributes.corpse_id_plane``).

    Returns:
        torch.Tensor: ``[B, F, 23, H, W]``, the channels of :data:`FIELD_CHANNEL_NAMES`.
    """
    scaled = log_compressed_depth(depth)
    k = min(TOP_K, int(weights.shape[2]))
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

    planes = {
        "coverage": coverage,
        "depth": resolve(scaled),
        "yaw_sin": resolve(torch.sin(relative_yaw)),
        "yaw_cos": resolve(torch.cos(relative_yaw)),
        "team": resolve(team_sign(enemy, weights.dtype)),
        "headcount": headcount(weights)[..., None, None].expand_as(coverage),
        **{f"controls_{name}": resolve(controls[..., i]) for i, name in enumerate(FIELD_CONTROLS)},
        **{f"weapon_{i}": resolve(weapons[..., i]) for i in range(WEAPON_CHANNELS)},
        "dying": dying_plane.to(weights.dtype),
        "corpse": corpse_plane.to(weights.dtype),
        "identity_live": resolve_winner(live_id_values(enemy)),
        "identity_corpse": corpse_ids.to(weights.dtype),
    }
    return torch.stack([planes[name] for name in FIELD_CHANNEL_NAMES], dim=2)


# -------------------------------------------------------------------------------------------- field
def _check_table(
    player_state_table: torch.Tensor, player_controls: torch.Tensor, weapon_embedding: torch.Tensor
) -> None:
    if player_state_table.ndim != 4:
        raise ValueError(
            f"player_state_table must be [B, F, P, 6 | 13], got {tuple(player_state_table.shape)}"
        )
    has_continuous_columns(player_state_table)  # raises for another width
    if player_controls.ndim != 5 or player_controls.shape[:3] != player_state_table.shape[:3]:
        raise ValueError(
            "player_controls must be [B, F, P, Q, A] aligned with player_state_table, got"
            f" {tuple(player_controls.shape)} vs {tuple(player_state_table.shape)}"
        )
    if int(weapon_embedding.shape[-1]) != WEAPON_CHANNELS:
        raise ValueError(
            f"weapon embedding width {int(weapon_embedding.shape[-1])} != {WEAPON_CHANNELS}"
        )


def _call_frames(window_frames: int, frame_offset: int, num_frames: int | None) -> tuple[int, int]:
    """The call's ``(frame_offset, num_frames)`` in a window of ``window_frames`` latent frames."""
    frame_offset = int(frame_offset)
    num_frames = window_frames - frame_offset if num_frames is None else int(num_frames)
    if frame_offset < 0 or num_frames < 0 or frame_offset + num_frames > window_frames:
        raise ValueError(
            f"the call's latent frames [{frame_offset}, {frame_offset + num_frames}) lie outside"
            f" the {window_frames}-frame window the player state table covers"
        )
    return frame_offset, num_frames


def corpse_weights(
    death: DeathState,
    player_controls: torch.Tensor,
    client_slot: torch.Tensor,
    is_client: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """The corpse weights ``w^c_p``: the kernel at the frozen position with half the projected
    radius, no confidence and no visibility gate.

    Args:
        death (DeathState): death state of the player state table.
        player_controls (torch.Tensor): ``[B, F, P, 16, 14]`` packed substeps.
        client_slot (torch.Tensor): ``[B]`` the client's slot.
        is_client (torch.Tensor): ``[B, F, P]`` bool, the client's column.
        grid_h (int): token rows.
        grid_w (int): token columns.

    Returns:
        torch.Tensor: ``[B, F, P, H, W]`` float32.
    """
    frozen = project_view(
        death.frozen_states, player_controls, client_slot, grid_h=grid_h, grid_w=grid_w
    )
    eligible = frozen.in_front & death.corpse.to(is_client.device) & ~is_client
    return splat(
        frozen.uv, frozen.radius * CORPSE_RADIUS_FACTOR, eligible, grid_h=grid_h, grid_w=grid_w
    )


def player_state_field(
    player_state_table: torch.Tensor,
    player_controls: torch.Tensor,
    client_slot: torch.Tensor,
    player_team_ids: torch.Tensor,
    player_alive: torch.Tensor,
    player_visible: torch.Tensor,
    player_weapons: torch.Tensor,
    weapon_embedding: torch.Tensor,
    *,
    frame_offset: int = 0,
    num_frames: int | None = None,
    config: PlayerStateFieldConfig = PlayerStateFieldConfig(),
) -> torch.Tensor:
    """The player state field of the latent frames of one generator call, from the player-state
    conditions (:func:`~worldcast.player_state.states.player_state_conditions`).

    The field is computed over the whole window the table spans (the camera integral and the
    confidence run along it) and the call's latent frames ``[frame_offset, frame_offset +
    num_frames)`` are returned. ``player_alive`` and ``player_visible`` may be sliced to those
    frames (the sampler slices them per block); they are zero-padded back.

    The client's camera: a 13-column table carries its yaw and pitch per latent frame (columns 6,
    7). A 6-column table gives them at frame 0 only; later frames add the turns of
    ``player_controls``, and columns 3, 4 of later frames are not read.

    Args:
        player_state_table (torch.Tensor): ``[B, F, P, 6 | 13]`` the player states per latent
            frame (whole window).
        player_controls (torch.Tensor): ``[B, F, P, 16, 14]`` packed substeps (whole window).
        client_slot (torch.Tensor): ``[B]`` the client's slot.
        player_team_ids (torch.Tensor): ``[B, P]`` engine team ids.
        player_alive (torch.Tensor): ``[B, f, P]`` 0/1, ``f`` = ``num_frames`` or ``F``.
        player_visible (torch.Tensor): ``[B, f, P]`` 0/1 visibility gate.
        player_weapons (torch.Tensor): ``[B, F, P]`` weapon ids (whole window).
        weapon_embedding (torch.Tensor): ``[52, 4]`` the injector's weapon embedding weight.
        frame_offset (int): the call's first latent frame in the window.
        num_frames (int | None): the call's latent frames (default ``F - frame_offset``).
        config (PlayerStateFieldConfig): the field settings.

    Returns:
        torch.Tensor: ``[B, num_frames, 23, 12, 21]`` float32.
    """
    device = player_state_table.device
    player_controls = player_controls.to(device=device)
    _check_table(player_state_table, player_controls, weapon_embedding)
    frame_offset, num_frames = _call_frames(
        int(player_state_table.shape[1]), frame_offset, num_frames
    )
    grid_h, grid_w = TOKEN_GRID

    view = project_view(
        player_state_table, player_controls, client_slot, grid_h=grid_h, grid_w=grid_w
    )
    gate = live_eligibility(
        view.in_front,
        alive=player_alive,
        visible=player_visible,
        client_slot=client_slot,
        frame_offset=frame_offset,
    )
    confidence = visibility_confidence(
        gate.visible_window,
        frames_per_block=config.frames_per_block,
        first_frame_alone=config.first_frame_alone,
        floor=config.confidence_floor,
    )
    death = death_state_for_table(player_state_table)
    check_block_alive_agreement(
        death.alive,
        gate.alive_window,
        frame_offset=frame_offset,
        num_frames=int(player_alive.shape[1]),
    )
    fallen = corpse_weights(
        death, player_controls, client_slot, gate.is_client, grid_h=grid_h, grid_w=grid_w
    )
    dying_plane, corpse_plane = death_planes(fallen, death.dying)
    field = compose(
        splat(
            view.uv, view.radius, gate.eligible, grid_h=grid_h, grid_w=grid_w, confidence=confidence
        ),
        view.depth,
        view.relative_yaw,
        enemy_mask(player_team_ids, client_slot, gate.eligible),
        control_fractions(player_controls),
        weapon_values(player_weapons.to(device), weapon_embedding),
        dying_plane=dying_plane,
        corpse_plane=corpse_plane,
        corpse_ids=corpse_id_plane(fallen),
    )
    return field[:, frame_offset : frame_offset + num_frames]


@dataclass(frozen=True)
class _FieldBuilder:
    config: PlayerStateFieldConfig
    visibility_gate: bool
    condition_keys: tuple[str, ...] = FIELD_CONDITION_KEYS

    def __call__(
        self,
        conditions: Mapping[str, Any],
        weapon_embedding: torch.Tensor,
        frame_offset: int,
        num_frames: int,
    ) -> torch.Tensor:
        table, controls, client_slot, team_ids, alive, visible, weapons = (
            conditions[key] for key in self.condition_keys
        )
        return player_state_field(
            table,
            controls,
            client_slot,
            team_ids,
            alive,
            visible if self.visibility_gate else torch.ones_like(alive),
            weapons,
            weapon_embedding,
            frame_offset=frame_offset,
            num_frames=num_frames,
            config=self.config,
        )


def field_builder(
    config: PlayerStateFieldConfig = PlayerStateFieldConfig(), *, visibility_gate: bool = True
) -> FieldBuilder:
    """The generator's field builder: :func:`player_state_field` of its player-state conditions
    (:func:`~worldcast.player_state.states.player_state_conditions`).

    Args:
        config (PlayerStateFieldConfig): the field settings.
        visibility_gate (bool): gate by ``player_visible``; without it (Table 5, "Trained without
            visibility") every living player in front of the camera is written.

    Returns:
        FieldBuilder: ``(conditions, weapon_embedding, frame_offset, num_frames) -> field``, the
        field ``[B, num_frames, 23, 12, 21]`` of the call's latent frames; it reads
        :data:`~worldcast.player_state.states.FIELD_CONDITION_KEYS`.
    """
    return _FieldBuilder(config, visibility_gate)
