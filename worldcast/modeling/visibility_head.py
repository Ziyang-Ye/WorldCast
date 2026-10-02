"""The visibility probe of training stages 2s and 3 (not in the paper; off at inference).

Per latent frame and player it pools the tokens under the player's footprint after one DiT block,
appends five geometry features and predicts an engine-confirmed line of sight; its BCE is detached
from the backbone. It is kept for the strict keys of the 2s and 3 checkpoints and because building
it draws random numbers (the initialisation order).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "DEPTH_LOG_SCALE",
    "GEOMETRY_FEATURES",
    "POSITIVE_WEIGHT_CAP",
    "VisibilityHead",
    "VisibilityHeadConfig",
    "VisibilityInputs",
    "visibility_bce",
]

#: Depth normalisation of the geometry feature (``worldcast.player_state.field.DEPTH_LOG_SCALE``).
DEPTH_LOG_SCALE = math.log1p(4096.0)

#: depth, sin(relative yaw), cos(relative yaw), coverage, in front.
GEOMETRY_FEATURES = 5

#: Ceiling of the BCE's positive weight (labels are ~3.6% positive: full balance would weight 27).
POSITIVE_WEIGHT_CAP = 10.0


@dataclass(frozen=True)
class VisibilityHeadConfig:
    """Shape of the probe; the defaults are those of the paper's runs."""

    #: Index of the DiT block the probe reads after.
    block: int = 20
    hidden: int = 256
    #: A pure probe: no gradient reaches the backbone.
    detach: bool = True


@dataclass
class VisibilityInputs:
    """What the head reads besides the tokens, over the window (``F`` frames, ``P`` players).

    Built by the trainer with ``worldcast.player_state.projection.project_view`` and
    ``worldcast.player_state.field.splat_weights``.

    Attributes:
        footprint (Tensor): ``[B, F, P, h, w]`` splat weight of each player on the token grid (in
            the frustum, alive, not the observer; no visibility gate).
        depth (Tensor): ``[B, F, P]`` camera-space depth, Source units.
        relative_yaw (Tensor): ``[B, F, P]`` the player's yaw minus the observer's, radians.
        in_front (Tensor): ``[B, F, P]`` bool, depth > 1.
    """

    footprint: torch.Tensor
    depth: torch.Tensor
    relative_yaw: torch.Tensor
    in_front: torch.Tensor


class VisibilityHead(nn.Module):
    """``norm`` LayerNorm(dim) on the pooled tokens, then ``mlp`` ``Linear(dim + 5, hidden) -> SiLU
    -> Linear(hidden, hidden) -> SiLU -> Linear(hidden, 1)`` on ``[pooled | 5 geometry]``."""

    def __init__(self, dim: int, config: VisibilityHeadConfig = VisibilityHeadConfig()) -> None:
        super().__init__()
        self.block = config.block
        self.detach = config.detach
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim + GEOMETRY_FEATURES, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        inputs: VisibilityInputs,
        *,
        grid_h: int,
        grid_w: int,
        frame_offset: int,
        video_frames: int,
        teacher_forcing: bool,
    ) -> torch.Tensor:
        """Visibility logits of the frames ``[frame_offset, frame_offset + video_frames)``.

        Args:
            hidden (Tensor): ``[B, tokens, dim]`` output of block :attr:`block` (``[clean | noisy]``
                under teacher forcing; the noisy half is read).
            inputs (VisibilityInputs): the window's inputs.
            grid_h (int): token rows.
            grid_w (int): token columns.
            frame_offset (int): window index of the first frame.
            video_frames (int): frames read.
            teacher_forcing (bool): ``hidden`` holds both copies.

        Returns:
            Tensor: ``[B, video_frames, P]`` float32 logits.
        """
        batch = hidden.shape[0]
        tokens = hidden
        if teacher_forcing:
            tokens = tokens[:, tokens.shape[1] // 2 :]
        if self.detach:
            tokens = tokens.detach()
        frames = int(video_frames)
        window = slice(int(frame_offset), int(frame_offset) + frames)
        occupancy = inputs.footprint[:, window].to(tokens.dtype)  # [B, f, P, h, w]
        grid = tokens.view(batch, frames, grid_h * grid_w, -1)  # [B, f, hw, dim]
        weights = occupancy.flatten(3)  # [B, f, P, hw]
        coverage = weights.sum(-1)  # [B, f, P]
        pooled = (
            torch.einsum("bfpk,bfkd->bfpd", weights, grid) / coverage.clamp_min(1e-6)[..., None]
        )
        pooled = self.norm(pooled.float())
        scaled_depth = (
            torch.log1p(inputs.depth[:, window].float().clamp(min=0.0)) / DEPTH_LOG_SCALE
        ).clamp(0.0, 1.0)
        relative_yaw = inputs.relative_yaw[:, window].float()
        geometry = torch.stack(
            [
                scaled_depth,
                torch.sin(relative_yaw),
                torch.cos(relative_yaw),
                coverage.float().clamp(max=4.0) / 4.0,
                inputs.in_front[:, window].float(),
            ],
            dim=-1,
        )
        return self.mlp(torch.cat([pooled, geometry], dim=-1)).squeeze(-1)


def visibility_bce(
    logits: torch.Tensor, visible: torch.Tensor, valid: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Masked, class-balanced BCE of the head.

    Positives are weighted ``min(negatives / positives, POSITIVE_WEIGHT_CAP)``; unknown labels get
    zero weight.

    Args:
        logits (Tensor): ``[B, F, P]`` logits.
        visible (Tensor): ``[B, F, P]`` labels.
        valid (Tensor): ``[B, F, P]`` confirmed labels (visible or occluded).

    Returns:
        dict[str, Tensor]: ``loss`` (the only one with a gradient), ``accuracy``, ``recall``,
        ``specificity``, ``predicted_positive_rate``, ``label_positive_rate``,
        ``positive_weight``, ``valid_labels``, ``positive_labels``.
    """
    if logits.shape != visible.shape or logits.shape != valid.shape:
        raise ValueError("logits, visible and valid must share one shape")
    mask = valid.detach().to(device=logits.device, dtype=torch.bool)
    target = visible.detach().to(device=logits.device, dtype=logits.dtype)
    per_position = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    )
    positive = mask & (target > 0.5)
    negative = mask & (target <= 0.5)
    positive_count = positive.sum().float().clamp_min(1.0)
    negative_count = negative.sum().float().clamp_min(1.0)
    positive_weight = (negative_count / positive_count).clamp(max=POSITIVE_WEIGHT_CAP)
    weight = torch.where(
        positive,
        positive_weight.expand_as(per_position),
        torch.where(negative, torch.ones_like(per_position), torch.zeros_like(per_position)),
    )
    with torch.no_grad():
        predicted = logits.detach() > 0
        confirmed = mask.sum().float().clamp_min(1.0)
        correct = (predicted == (target > 0.5)) & mask
        accuracy = correct.sum().float() / confirmed
        recall = (predicted & positive).sum().float() / positive_count
        specificity = ((~predicted) & negative).sum().float() / negative_count
        predicted_positive_rate = (predicted & mask).sum().float() / confirmed
        label_positive_rate = positive.sum().float() / confirmed
    return {
        "loss": (per_position * weight).sum() / weight.sum().clamp_min(1e-12),
        "accuracy": accuracy,
        "recall": recall,
        "specificity": specificity,
        "predicted_positive_rate": predicted_positive_rate,
        "label_positive_rate": label_positive_rate,
        "positive_weight": positive_weight.detach(),
        "valid_labels": mask.sum().detach(),
        "positive_labels": positive.sum().detach(),
    }
