"""Per-player attributes of the player state field besides geometry: controls, team, weapon, death.

Paper (App. "The player state field in detail", Channels): team; headcount; nine control channels,
each the fraction of the latent frame during which a key is held; four weapon-embedding channels;
dying and corpse channels, written where the player fell; two identity channels. The values here are
per player (``[B, F, P, ...]``) or, for the death and corpse-identity planes, already per token;
``worldcast.player_state.field`` places them on the grid.

As trained, where the paper's text says less (docs/inference.md, "Paper vs code"): the headcount is
capped at four and divided by four; the identity channels are resolved winner-take-all (live) and by
arg max (corpse), and the death planes are a max over players, not the weighted sum of the attribute
channels.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from worldcast.data.actions import OPENCS2_WEAPONS, PAPER_ACTION_BUTTONS

from .tables import SUBSTEP_VALID_INDEX

__all__ = [
    "PAPER_ACTION_BUTTONS",
    "FIELD_CONTROLS",
    "WEAPON_VOCAB",
    "WEAPON_CHANNELS",
    "HEADCOUNT_CAP",
    "ID_BAND_SEATS",
    "ID_BAND_STEP",
    "ID_BAND_ALLY_BASE",
    "ID_BAND_ENEMY_BASE",
    "ID_BAND_CORPSE_BASE",
    "CORPSE_RADIUS_FACTOR",
    "control_indices",
    "duck_index",
    "held_fraction",
    "control_fractions",
    "enemy_mask",
    "team_sign",
    "weapon_values",
    "headcount",
    "live_id_values",
    "corpse_id_plane",
    "death_planes",
]

#: The nine control channels of the field, in channel order (config
#: ``model.player_field.action_signals``).
FIELD_CONTROLS = (
    "forward",
    "back",
    "move_left",
    "move_right",
    "jump",
    "duck",
    "speed",
    "attack",
    "reload",
)
WEAPON_VOCAB = len(OPENCS2_WEAPONS)
#: Width of the learned weapon embedding (config ``model.player_field.weapon_channels``).
WEAPON_CHANNELS = 4
HEADCOUNT_CAP = 4.0

#: Reserved identity bands: seat p of the client's team reads 0.20 + 0.02 p, an opponent 0.50 + 0.02
#: p, a corpse 0.80 + 0.02 p; 0 keeps meaning "nobody".
ID_BAND_SEATS = 10
ID_BAND_STEP = 0.02
ID_BAND_ALLY_BASE = 0.20
ID_BAND_ENEMY_BASE = 0.50
ID_BAND_CORPSE_BASE = 0.80

#: A downed body is splatted with this fraction of the live projected radius (a lower centre).
CORPSE_RADIUS_FACTOR = 0.5


# ----------------------------------------------------------------------------------------- controls
def control_indices(
    action_buttons: Sequence[str] = PAPER_ACTION_BUTTONS, controls: Sequence[str] = FIELD_CONTROLS
) -> tuple[int, ...]:
    """Column of each field control in the substep buttons (paper: 0, 1, 2, 3, 4, 5, 6, 7, 9)."""
    buttons = [str(b) for b in action_buttons]
    names = [str(n) for n in controls]
    missing = [n for n in names if n not in buttons]
    if missing:
        raise ValueError(f"field controls {missing} are not in action_buttons {buttons}")
    return tuple(buttons.index(n) for n in names)


def duck_index(action_buttons: Sequence[str] = PAPER_ACTION_BUTTONS) -> int:
    """Column of the crouch button (paper: 5); it drives the body height."""
    buttons = [str(b) for b in action_buttons]
    if "duck" not in buttons:
        raise ValueError("action_buttons has no 'duck'")
    return buttons.index("duck")


def held_fraction(actions: torch.Tensor, button_index: int) -> torch.Tensor:
    """Share of a latent frame's valid substeps during which a button is held.

    Args:
        actions (torch.Tensor): ``[B, F, P, Q, A + 1]`` packed substeps (16 per latent frame).
        button_index (int): button column.

    Returns:
        torch.Tensor: ``[B, F, P]`` in ``[0, 1]``, ``actions``' dtype (0 where no substep is valid).
    """
    valid = (actions[..., SUBSTEP_VALID_INDEX] > 0.5).to(actions.dtype)
    held = (actions[..., button_index] > 0.5).to(actions.dtype) * valid
    return held.sum(-1) / valid.sum(-1).clamp(min=1.0)


def control_fractions(actions: torch.Tensor, indices: Sequence[int]) -> torch.Tensor | None:
    """``[B, F, P, K]`` held fractions of the ``K`` field controls (None for no ``indices``)."""
    if not indices:
        return None
    return torch.stack([held_fraction(actions, int(i)) for i in indices], dim=-1)


# --------------------------------------------------------------------------------------------- team
def enemy_mask(
    team_ids: torch.Tensor, observer_slot: torch.Tensor, eligible: torch.Tensor
) -> torch.Tensor:
    """``[B, F, P]`` bool: eligible players not on the client's team.

    Args:
        team_ids (torch.Tensor): ``[B, P]`` engine team ids.
        observer_slot (torch.Tensor): ``[B]`` the client's seat.
        eligible (torch.Tensor): ``[B, F, P]`` bool, the live gate.
    """
    device = eligible.device
    teams = team_ids.to(device).long()
    slot = observer_slot.to(device).long()
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


# ----------------------------------------------------------------------------------------- identity
def live_id_values(enemy: torch.Tensor) -> torch.Tensor:
    """``[B, F, P]`` float32 identity of each seat: ally ``0.20 + 0.02 p``, enemy ``0.50 + 0.02 p``.

    Ineligible seats read an ally value; they carry zero weight, so the merge never picks them where
    an eligible player covers, and the coverage zeroes the plane where nobody does.
    """
    seats = int(enemy.shape[-1])
    if seats > ID_BAND_SEATS:
        raise ValueError(f"identity bands reserve values for {ID_BAND_SEATS} seats, got {seats}")
    offsets = ID_BAND_STEP * torch.arange(seats, device=enemy.device, dtype=torch.float32)
    bases = ID_BAND_ALLY_BASE + (ID_BAND_ENEMY_BASE - ID_BAND_ALLY_BASE) * enemy.to(torch.float32)
    return bases + offsets


def corpse_id_plane(corpse_weights: torch.Tensor) -> torch.Tensor:
    """Corpse identity plane: ``0.80 + 0.02 p`` of the arg-max corpse ``p``, times its weight.

    Args:
        corpse_weights (torch.Tensor): ``[B, F, P, H, W]`` corpse splat weights.

    Returns:
        torch.Tensor: ``[B, F, H, W]``; divided by the corpse plane it decodes the exact band value.
    """
    seats = int(corpse_weights.shape[2])
    if seats > ID_BAND_SEATS:
        raise ValueError(f"identity bands reserve values for {ID_BAND_SEATS} seats, got {seats}")
    values = ID_BAND_CORPSE_BASE + ID_BAND_STEP * torch.arange(
        seats, device=corpse_weights.device, dtype=corpse_weights.dtype
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
