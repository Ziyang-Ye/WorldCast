"""Attributes of the player state field (App. "The player state field in detail", Channels).

The per-player values the field writes besides geometry: team, the held share of each control, the
learned weapon embedding, the identity bands, and the headcount, dying and corpse planes. As
trained, where the paper's text says less: the headcount is capped at four and divided by four; the
identity channels are resolved winner-take-all (live) and by arg max (corpse); the death planes are
a max over players, not the weighted sum of the other channels.
"""

import torch
import torch.nn.functional as F

from worldcast.data.controls import CONTROL_BUTTONS
from worldcast.modeling.state_injector import FIELD_CONTROLS

from .states import held_fraction

__all__ = [
    "CONTROL_COLUMNS",
    "CORPSE_RADIUS_FACTOR",
    "HEADCOUNT_CAP",
    "ID_BAND_CORPSE_BASE",
    "ID_BAND_ENEMY_BASE",
    "ID_BAND_SLOTS",
    "ID_BAND_STEP",
    "ID_BAND_TEAMMATE_BASE",
    "control_fractions",
    "corpse_id_plane",
    "death_planes",
    "enemy_mask",
    "headcount",
    "live_id_values",
    "team_sign",
    "weapon_values",
]

#: Column of each field control in the substep buttons: 0, 1, 2, 3, 4, 5, 6, 7, 9.
CONTROL_COLUMNS = tuple(CONTROL_BUTTONS.index(name) for name in FIELD_CONTROLS)
#: The headcount channel saturates at this many players.
HEADCOUNT_CAP = 4.0
#: Reserved identity bands: slot p reads 0.20 + 0.02 p for a teammate of the client, 0.50 + 0.02 p
#: for an enemy and 0.80 + 0.02 p for a corpse; 0 keeps meaning "nobody".
ID_BAND_SLOTS = 10
ID_BAND_STEP = 0.02
ID_BAND_TEAMMATE_BASE = 0.20
ID_BAND_ENEMY_BASE = 0.50
ID_BAND_CORPSE_BASE = 0.80
#: A downed body is splatted with this fraction of the live projected radius (a lower centre).
CORPSE_RADIUS_FACTOR = 0.5


def control_fractions(controls: torch.Tensor) -> torch.Tensor:
    """``[B, F, P, 9]`` share of each latent frame's valid substeps with a field control held, from
    the packed substeps ``[B, F, P, Q, A + 1]``."""
    return torch.stack([held_fraction(controls, column) for column in CONTROL_COLUMNS], dim=-1)


def enemy_mask(
    team_ids: torch.Tensor, client_slot: torch.Tensor, eligible: torch.Tensor
) -> torch.Tensor:
    """``[B, F, P]`` bool: eligible players not on the client's team.

    Args:
        team_ids (torch.Tensor): ``[B, P]`` engine team ids.
        client_slot (torch.Tensor): ``[B]`` the client's slot.
        eligible (torch.Tensor): ``[B, F, P]`` bool, the live gate.
    """
    device = eligible.device
    teams = team_ids.to(device).long()
    slot = client_slot.to(device).long()
    own = teams.gather(1, slot.view(-1, 1))
    same = (teams == own)[:, None, :].expand_as(eligible)
    return eligible & ~same


def team_sign(enemy: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Team channel value per player: +1 enemy, -1 teammate (0 means "nobody" after the merge)."""
    return enemy.to(dtype) * 2.0 - 1.0


def weapon_values(weapons: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    """The learned weapon embedding of each player's weapon.

    Args:
        weapons (torch.Tensor): ``[B, F, P]`` 52-way weapon ids.
        embedding (torch.Tensor): ``[52, 4]`` the injector's weapon embedding weight
            (``worldcast.modeling.state_injector``).

    Returns:
        torch.Tensor: ``[B, F, P, 4]`` in the embedding's dtype.
    """
    return F.embedding(weapons.long(), embedding)


def headcount(weights: torch.Tensor) -> torch.Tensor:
    """``[B, F]`` headcount ``min(sum_p max_u w_p(u), 4) / 4`` of ``[B, F, P, H, W]`` weights."""
    return weights.amax(dim=(3, 4)).sum(dim=2).clamp(max=HEADCOUNT_CAP) / HEADCOUNT_CAP


def live_id_values(enemy: torch.Tensor) -> torch.Tensor:
    """``[B, F, P]`` float32 identity of each slot: teammate ``0.20 + 0.02 p``, enemy ``0.50 + 0.02
    p``.

    Ineligible slots read a teammate value; they carry zero weight, so the merge never picks them
    where an eligible player covers, and the coverage zeroes the plane where nobody does.
    """
    slots = int(enemy.shape[-1])
    if slots > ID_BAND_SLOTS:
        raise ValueError(f"identity bands reserve values for {ID_BAND_SLOTS} slots, got {slots}")
    offsets = ID_BAND_STEP * torch.arange(slots, device=enemy.device, dtype=torch.float32)
    bases = ID_BAND_TEAMMATE_BASE + (ID_BAND_ENEMY_BASE - ID_BAND_TEAMMATE_BASE) * enemy.to(
        torch.float32
    )
    return bases + offsets


def corpse_id_plane(corpse_weights: torch.Tensor) -> torch.Tensor:
    """Corpse identity plane: ``0.80 + 0.02 p`` of the arg-max corpse ``p``, times its weight.

    Args:
        corpse_weights (torch.Tensor): ``[B, F, P, H, W]`` corpse splat weights.

    Returns:
        torch.Tensor: ``[B, F, H, W]``; divided by the corpse plane it decodes the exact band value.
    """
    slots = int(corpse_weights.shape[2])
    if slots > ID_BAND_SLOTS:
        raise ValueError(f"identity bands reserve values for {ID_BAND_SLOTS} slots, got {slots}")
    values = ID_BAND_CORPSE_BASE + ID_BAND_STEP * torch.arange(
        slots, device=corpse_weights.device, dtype=corpse_weights.dtype
    )
    coverage, winner = corpse_weights.max(dim=2)
    return values[winner] * coverage


def death_planes(
    corpse_weights: torch.Tensor, dying: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dying and corpse planes, written where each player fell.

    Args:
        corpse_weights (torch.Tensor): ``[B, F, P, H, W]`` corpse splat weights.
        dying (torch.Tensor): ``[B, F, P]`` ``exp(-tsd / 5)`` on corpse frames.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``max_p w_p dying_p`` and ``max_p w_p``, each
        ``[B, F, H, W]``.
    """
    dying_plane = (corpse_weights * dying.to(corpse_weights.dtype)[..., None, None]).amax(dim=2)
    corpse_plane = corpse_weights.amax(dim=2)
    return dying_plane, corpse_plane
