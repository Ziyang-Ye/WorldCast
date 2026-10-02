"""The visibility gate of the player state field and its confidence ``e_p`` (Sec. 3.2).

Another player is written into the field only if it is visible. With GT states (Table 3) the gate
reads the GT visibility labels (:class:`GTLabelVisibility`); in the closed loop the labels come from
:class:`worldcast.player_state.predicted_visibility.PredictedVisibility`. A latent frame is visible
if any of its four pixel rows ``4f-3 .. 4f`` is.

Gate (:func:`live_eligibility`): ``in_front & alive & not the client & visible``; with labels, the
frustum and occlusion tests are inside the label and the only geometric test is ``depth > 1``.
Corpses skip the gate (``worldcast.player_state.field``). Confidence
(:func:`visibility_confidence`): a causal EMA (alpha 0.5) of the label over the latent frames of a
block, reset at every block start, mapped to ``0.3 + 0.7 ema``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch

from .tables import PlayerStates

__all__ = [
    "CONFIDENCE_FLOOR",
    "CONFIDENCE_SMOOTH",
    "GTLabelVisibility",
    "Eligibility",
    "latent_visibility_components",
    "latent_visibility_confirmed",
    "pad_framewise_to_window",
    "live_eligibility",
    "visibility_confidence",
]

#: Lowest confidence of a written player (config ``model.player_field.confidence_floor``).
CONFIDENCE_FLOOR = 0.3
#: EMA alpha of the confidence (config ``model.player_field.confidence_smooth``).
CONFIDENCE_SMOOTH = 0.5


def latent_visibility_components(
    observer_visibility: torch.Tensor,
    observer_visibility_valid: torch.Tensor,
    latent_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold pixel-row labels to latent frames, keeping visible and valid apart.

    Args:
        observer_visibility (torch.Tensor): ``[B, P, T]`` bool, row-wise ``visible & valid``.
        observer_visibility_valid (torch.Tensor): ``[B, P, T]`` bool.
        latent_rows (torch.Tensor): ``[F]`` pixel row of each latent frame (``4 f``).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(visible, valid)``, each ``[B, F, P]`` bool: visible in
        any of the latent frame's four rows, valid in all of them.
    """
    if observer_visibility.shape != observer_visibility_valid.shape:
        raise ValueError("visibility and its validity mask must have one shape")
    if observer_visibility.ndim != 3:
        raise ValueError("observer_visibility must be [B, P, T]")

    rows = latent_rows.to(observer_visibility.device).long()
    window = (rows[:, None] + torch.arange(-3, 1, device=rows.device)[None, :]).clamp_min(0)
    flat = window.reshape(-1)

    vis = observer_visibility.bool().index_select(2, flat)
    val = observer_visibility_valid.bool().index_select(2, flat)
    batch, players = vis.shape[:2]
    frames = rows.numel()
    vis = vis.reshape(batch, players, frames, 4)
    val = val.reshape(batch, players, frames, 4)
    visible = vis.any(-1).permute(0, 2, 1).contiguous()
    valid = val.all(-1).permute(0, 2, 1).contiguous()
    return visible, valid


def latent_visibility_confirmed(
    observer_visibility: torch.Tensor,
    observer_visibility_valid: torch.Tensor,
    latent_rows: torch.Tensor,
) -> torch.Tensor:
    """``[B, P, T] -> [B, F, P]`` bool: visible in any of the latent frame's four pixel rows.

    Occluded and unknown both fold to False. Raises if a row is visible where it is not valid, i.e.
    if the labels are not row-wise ``visible & valid``.
    """
    if observer_visibility.shape != observer_visibility_valid.shape:
        raise ValueError("visibility and its validity mask must have one shape")
    leaked = observer_visibility.bool() & ~observer_visibility_valid.bool()
    if bool(leaked.any()):
        raise ValueError(
            "observer_visibility marks rows visible where observer_visibility_valid is False"
            f" ({int(leaked.sum())} of {leaked.numel()} rows); expected row-wise visible & valid"
            " labels"
        )
    visible, _ = latent_visibility_components(
        observer_visibility, observer_visibility_valid, latent_rows
    )
    return visible


@dataclass(frozen=True)
class GTLabelVisibility:
    """The visibility gate from the GT labels (Table 3).

    Reads the batch's ``observer_visibility`` / ``observer_visibility_valid`` (``[B, P, T]`` bool,
    the client's labels of every seat, from ``worldcast.data``) and folds them with
    :func:`latent_visibility_confirmed`.

    Attributes:
        device (torch.device | None): where to compute (default: the states' device).
    """

    device: torch.device | None = None

    def __call__(self, batch: Mapping[str, torch.Tensor], states: PlayerStates) -> torch.Tensor:
        """``[B, F, P]`` bool; raises if the batch carries no labels (never a silent "unknown")."""
        visibility = batch.get("observer_visibility")
        valid = batch.get("observer_visibility_valid")
        if visibility is None or valid is None:
            raise ValueError(
                "the batch carries no visibility labels; set the visibility label root in the"
                " config"
            )
        device = states.latent_rows.device if self.device is None else torch.device(self.device)
        return latent_visibility_confirmed(
            visibility.to(device), valid.to(device), states.latent_rows.to(device)
        )


def pad_framewise_to_window(
    value: torch.Tensor | None, *, window: torch.Tensor, frame_offset: int
) -> torch.Tensor | None:
    """Put a block-sliced condition (``peer_alive``, ``peer_visible``) back on the whole window.

    The sampler slices these two to the decoded block; zero outside it means "not alive" / "not
    visible", so no player is written there (and those rows are cut off the field anyway).

    Args:
        value (torch.Tensor | None): ``[B, f, P]`` (``f <= F``) or None.
        window (torch.Tensor): ``[B, F, P]`` reference (shape and device).
        frame_offset (int): first frame of the block.

    Returns:
        torch.Tensor | None: ``[B, F, P]`` in ``value``'s dtype, or None.
    """
    if value is None:
        return None
    value = value.to(window.device)
    if value.shape[1] == window.shape[1]:
        return value
    padded = value.new_zeros(window.shape)
    end = min(frame_offset + value.shape[1], window.shape[1])
    padded[:, frame_offset:end] = value[:, : end - frame_offset]
    return padded


class Eligibility(NamedTuple):
    """Output of :func:`live_eligibility`; all ``[B, F, P]`` over the whole window.

    Attributes:
        eligible (torch.Tensor): bool, the live players written into the field.
        is_self (torch.Tensor): bool, the client's own seat.
        alive_window (torch.Tensor): ``peer_alive`` padded to the window.
        visible_window (torch.Tensor): ``peer_visible`` padded to the window.
    """

    eligible: torch.Tensor
    is_self: torch.Tensor
    alive_window: torch.Tensor
    visible_window: torch.Tensor


def live_eligibility(
    in_front: torch.Tensor,
    *,
    alive: torch.Tensor,
    visible: torch.Tensor,
    observer_slot: torch.Tensor,
    frame_offset: int = 0,
) -> Eligibility:
    """The gate of the live field: ``in_front & alive & ~client & visible``.

    Args:
        in_front (torch.Tensor): ``[B, F, P]`` bool, ``depth > 1`` (whole window).
        alive (torch.Tensor): ``[B, f, P]`` ``peer_alive`` (block-sliced or whole window).
        visible (torch.Tensor): ``[B, f, P]`` ``peer_visible`` (block-sliced or whole window).
        observer_slot (torch.Tensor): ``[B]`` the client's seat.
        frame_offset (int): first frame of the block ``alive`` and ``visible`` cover.

    Returns:
        Eligibility: the gate and its parts.
    """
    device = in_front.device
    num_frames = int(in_front.shape[1])
    alive_window = pad_framewise_to_window(alive, window=in_front, frame_offset=int(frame_offset))
    eligible = in_front & (alive_window > 0.5)
    slot = observer_slot.to(device).long()
    is_self = torch.zeros_like(eligible)
    is_self.scatter_(2, slot.view(-1, 1, 1).expand(-1, num_frames, 1), True)
    eligible = eligible & ~is_self
    visible_window = pad_framewise_to_window(
        visible, window=in_front, frame_offset=int(frame_offset)
    )
    eligible = eligible & (visible_window > 0.5)
    return Eligibility(
        eligible=eligible, is_self=is_self, alive_window=alive_window, visible_window=visible_window
    )


def visibility_confidence(
    visible_window: torch.Tensor,
    *,
    num_frame_per_block: int = 4,
    independent_first_frame: bool = True,
    floor: float = CONFIDENCE_FLOOR,
    smooth: float = CONFIDENCE_SMOOTH,
) -> torch.Tensor:
    """``e_p``: the block-reset causal EMA of the visibility label, mapped to ``[floor, 1]``.

    Block starts are frame 0 and ``first, first + 4, ...`` with ``first = 1`` when the first frame
    is its own block (the sink); on a 21-frame window that is ``{0, 1, 5, 9, 13, 17}``. Resetting at
    block starts makes a block's confidence independent of the zero padding outside it.

    Args:
        visible_window (torch.Tensor): ``[B, F, P]`` label (``visible & valid``) padded to the
            window.
        num_frame_per_block (int): latent frames per block (4).
        independent_first_frame (bool): frame 0 is its own block.
        floor (float): lowest confidence of a written player (0.3).
        smooth (float): EMA alpha (0.5).

    Returns:
        torch.Tensor: ``[B, F, P]`` float32 in ``[floor, 1]``.
    """
    floor = float(floor)
    values = visible_window.float().clamp(0.0, 1.0)
    frames = int(values.shape[1])
    alpha = float(smooth)
    first = 1 if independent_first_frame else 0
    starts = {0, *range(first, frames, max(1, int(num_frame_per_block)))}
    smoothed = []
    previous = None
    for frame in range(frames):
        current = values[:, frame]
        if frame in starts:
            state = current
        else:
            state = (1.0 - alpha) * previous + alpha * current
        smoothed.append(state)
        previous = state
    stacked = torch.stack(smoothed, dim=1)
    return (floor + (1.0 - floor) * stacked).clamp(floor, 1.0)
