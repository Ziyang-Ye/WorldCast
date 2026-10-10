"""The visibility gate of the player state field and its confidence ``e_p`` (Sec. 3.4, Depth and
visibility; App. "The player state field in detail", Visibility).

Another player is written into the field only if it is visible. The gate reads the client's
visibility labels of the batch (``client_visibility``, per video frame): the GT labels of the
recording (Table 3) or, with predicted states, the labels the client predicts from its depth head
(:mod:`worldcast.player_state.predicted_visibility`). A latent frame is visible if any of its video
frames is (:func:`visible_latent_frames`).

Gate (:func:`live_eligibility`): ``in_front & alive & not the client & visible``; the frustum and
occlusion tests are inside the label and the only geometric test is ``depth > 1``. Corpses skip the
gate (``worldcast.player_state.field``). Confidence (:func:`visibility_confidence`): a causal EMA
(alpha 0.5) of the label over the latent frames of a block, reset at every block start, mapped to
``0.3 + 0.7 ema``.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from worldcast.data.latents import BLOCK, latent_frame_count, video_frames_of

__all__ = [
    "CONFIDENCE_FLOOR",
    "CONFIDENCE_SMOOTH",
    "Eligibility",
    "fold_video_frames",
    "latent_visibility",
    "live_eligibility",
    "pad_framewise_to_window",
    "visibility_confidence",
    "visible_latent_frames",
]

#: Lowest confidence of a written player.
CONFIDENCE_FLOOR = 0.3
#: EMA alpha of the confidence.
CONFIDENCE_SMOOTH = 0.5


# ------------------------------------------------------------------------------------------- labels
def fold_video_frames(labels: torch.Tensor, first: int, count: int) -> torch.Tensor:
    """Labels per video frame as labels per latent frame: True if any of its video frames is.

    Args:
        labels (torch.Tensor): ``[R, ...]`` bool, one row per video frame of the latent frames
            ``first .. first + count - 1`` (:func:`~worldcast.data.latents.video_frames_of`).
        first (int): the first latent frame.
        count (int): latent frames.

    Returns:
        torch.Tensor: ``[count, ...]`` bool.
    """
    sizes = [len(video_frames_of(f)) for f in range(int(first), int(first) + int(count))]
    return torch.stack([frames.any(0) for frames in labels.split(sizes)])


def latent_visibility(
    client_visibility: torch.Tensor, client_visibility_valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold the labels of a window's video frames to its latent frames, keeping visible and valid
    apart.

    Args:
        client_visibility (torch.Tensor): ``[B, P, T]`` in {0, 1} (bool or float32), ``visible &
            valid`` per video frame.
        client_visibility_valid (torch.Tensor): ``[B, P, T]`` bool, the label is defined.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(visible, valid)``, each ``[B, F, P]`` bool: visible in
        any of the latent frame's video frames, valid in all of them.
    """
    if client_visibility.shape != client_visibility_valid.shape:
        raise ValueError("visibility and its validity mask must have one shape")
    if client_visibility.ndim != 3:
        raise ValueError("client_visibility must be [B, P, T]")
    frames = latent_frame_count(int(client_visibility.shape[2]))
    visible = fold_video_frames(client_visibility.bool().movedim(2, 0), 0, frames)
    invalid = fold_video_frames(~client_visibility_valid.bool().movedim(2, 0), 0, frames)
    return visible.permute(1, 0, 2).contiguous(), ~invalid.permute(1, 0, 2).contiguous()


def visible_latent_frames(
    batch: Mapping[str, torch.Tensor], *, device: torch.device | None = None
) -> torch.Tensor:
    """The visibility gate per latent frame, from the batch's labels.

    Occluded and unknown both read as not visible.

    Args:
        batch (Mapping[str, torch.Tensor]): ``client_visibility`` and ``client_visibility_valid``
            (``[B, P, T]``, the client's labels of every slot per video frame): the GT labels of
            ``worldcast.data``, or the predicted ones.
        device (torch.device | None): where to compute (default: the labels' device).

    Returns:
        torch.Tensor: ``[B, F, P]`` bool.

    Raises:
        ValueError: the batch carries no labels (never a silent "unknown"), or a video frame is
            marked visible where its label is not valid.
    """
    visibility, valid = batch.get("client_visibility"), batch.get("client_visibility_valid")
    if visibility is None or valid is None:
        raise ValueError(
            "the batch carries no visibility labels; set the visibility label root in the config"
        )
    if device is not None:
        visibility, valid = visibility.to(device), valid.to(device)
    leaked = visibility.bool() & ~valid.bool()
    if bool(leaked.any()):
        raise ValueError(
            "client_visibility marks video frames visible where client_visibility_valid is False"
            f" ({int(leaked.sum())} of {leaked.numel()}); expected visible & valid labels"
        )
    return latent_visibility(visibility, valid)[0]


# --------------------------------------------------------------------- the gate and its confidence
def pad_framewise_to_window(
    value: torch.Tensor, *, window: torch.Tensor, frame_offset: int
) -> torch.Tensor:
    """Put a block-sliced condition (``player_alive``, ``player_visible``) back on the whole window.

    The sampler slices these two to the latent frames of a generator call; zero outside them means
    "not alive" / "not visible", so no player is written there (and those frames are cut off the
    field anyway).

    Args:
        value (torch.Tensor): ``[B, f, P]`` (``f <= F``).
        window (torch.Tensor): ``[B, F, P]`` reference (shape and device).
        frame_offset (int): the call's first latent frame.

    Returns:
        torch.Tensor: ``[B, F, P]`` in ``value``'s dtype.
    """
    value = value.to(window.device)
    if value.shape[1] == window.shape[1]:
        return value
    padded = value.new_zeros(window.shape)
    end = min(frame_offset + value.shape[1], window.shape[1])
    padded[:, frame_offset:end] = value[:, : end - frame_offset]
    return padded


@dataclass(frozen=True)
class Eligibility:
    """Output of :func:`live_eligibility`; all ``[B, F, P]`` over the whole window.

    Attributes:
        eligible (torch.Tensor): bool, the live players written into the field.
        is_client (torch.Tensor): bool, the client's slot.
        alive_window (torch.Tensor): ``player_alive`` padded to the window.
        visible_window (torch.Tensor): ``player_visible`` padded to the window.
    """

    eligible: torch.Tensor
    is_client: torch.Tensor
    alive_window: torch.Tensor
    visible_window: torch.Tensor


def live_eligibility(
    in_front: torch.Tensor,
    *,
    alive: torch.Tensor,
    visible: torch.Tensor,
    client_slot: torch.Tensor,
    frame_offset: int = 0,
) -> Eligibility:
    """The gate of the live field: ``in_front & alive & ~client & visible``.

    Args:
        in_front (torch.Tensor): ``[B, F, P]`` bool, ``depth > 1`` (whole window).
        alive (torch.Tensor): ``[B, f, P]`` ``player_alive`` (block-sliced or whole window).
        visible (torch.Tensor): ``[B, f, P]`` ``player_visible`` (block-sliced or whole window).
        client_slot (torch.Tensor): ``[B]`` the client's slot.
        frame_offset (int): first latent frame of the block ``alive`` and ``visible`` cover.

    Returns:
        Eligibility: the gate and its parts.
    """
    device = in_front.device
    num_frames = int(in_front.shape[1])
    alive_window = pad_framewise_to_window(alive, window=in_front, frame_offset=int(frame_offset))
    eligible = in_front & (alive_window > 0.5)
    slot = client_slot.to(device).long()
    is_client = torch.zeros_like(eligible)
    is_client.scatter_(2, slot.view(-1, 1, 1).expand(-1, num_frames, 1), True)
    eligible = eligible & ~is_client
    visible_window = pad_framewise_to_window(
        visible, window=in_front, frame_offset=int(frame_offset)
    )
    eligible = eligible & (visible_window > 0.5)
    return Eligibility(
        eligible=eligible,
        is_client=is_client,
        alive_window=alive_window,
        visible_window=visible_window,
    )


def visibility_confidence(
    visible_window: torch.Tensor,
    *,
    frames_per_block: int = BLOCK,
    first_frame_alone: bool = True,
    floor: float = CONFIDENCE_FLOOR,
) -> torch.Tensor:
    """``e_p``: the block-reset causal EMA of the visibility label, mapped to ``[floor, 1]``.

    Block starts are frame 0 and ``first, first + 4, ...`` with ``first = 1`` when the first frame
    is a block of its own; on a 21-frame window that is ``{0, 1, 5, 9, 13, 17}``. Resetting at
    block starts makes a block's confidence independent of the zero padding outside it.

    Args:
        visible_window (torch.Tensor): ``[B, F, P]`` label (``visible & valid``) padded to the
            window.
        frames_per_block (int): latent frames per block (4).
        first_frame_alone (bool): latent frame 0 is a block of its own.
        floor (float): lowest confidence of a written player (0.3).

    Returns:
        torch.Tensor: ``[B, F, P]`` float32 in ``[floor, 1]``.
    """
    floor = float(floor)
    values = visible_window.float().clamp(0.0, 1.0)
    frames = int(values.shape[1])
    first = 1 if first_frame_alone else 0
    starts = {0, *range(first, frames, max(1, int(frames_per_block)))}
    smoothed = []
    previous = None
    for frame in range(frames):
        current = values[:, frame]
        if frame in starts:
            state = current
        else:
            state = (1.0 - CONFIDENCE_SMOOTH) * previous + CONFIDENCE_SMOOTH * current
        smoothed.append(state)
        previous = state
    stacked = torch.stack(smoothed, dim=1)
    return (floor + (1.0 - floor) * stacked).clamp(floor, 1.0)
