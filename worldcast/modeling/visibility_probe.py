"""The visibility probe (not in the paper; off at inference): the stages with scene state build
it, stages 2s and 3 train it.

Per latent frame and player it pools the tokens under the player's footprint after one DiT block,
appends five geometry features and predicts an engine-confirmed line of sight. It reads a detached
hidden state, so its BCE trains the probe only. It is kept for the strict keys of those stages'
checkpoints and because building it draws random numbers (the initialisation order).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from worldcast.modeling.state_injector import log_compressed_depth

__all__ = [
    "GEOMETRY_FEATURES",
    "POSITIVE_WEIGHT_CAP",
    "VisibilityProbe",
    "VisibilityProbeConfig",
    "VisibilityProbeInputs",
    "visibility_bce",
]

#: depth, sin(relative yaw), cos(relative yaw), coverage, in front.
GEOMETRY_FEATURES = 5

#: Ceiling of the BCE's positive weight (labels are ~3.6% positive: full balance would weight 27).
POSITIVE_WEIGHT_CAP = 10.0
#: The coverage feature is a player's summed footprint, clamped at this many tokens and divided by
#: it.
_COVERAGE_CAP = 4.0


@dataclass(frozen=True)
class VisibilityProbeConfig:
    """Shape of the probe; the defaults are those of the paper's runs."""

    #: Index of the DiT block the probe reads after, from 0.
    dit_block: int = 20
    #: Width of the MLP.
    hidden: int = 256


@dataclass(eq=False)
class VisibilityProbeInputs:
    """What the probe reads besides the tokens, over the window (``F`` frames, ``P`` players).

    Built by the trainer with ``worldcast.player_state.projection.project_view`` and
    ``worldcast.player_state.field.splat``.

    Attributes:
        footprint (Tensor): ``[B, F, P, h, w]`` splat weight of each player on the token grid (in
            front, alive, not the client's own player; no visibility gate).
        depth (Tensor): ``[B, F, P]`` camera-space depth, u.
        relative_yaw (Tensor): ``[B, F, P]`` the player's yaw minus the client's, radians.
        in_front (Tensor): ``[B, F, P]`` bool, depth > 1.
    """

    footprint: torch.Tensor
    depth: torch.Tensor
    relative_yaw: torch.Tensor
    in_front: torch.Tensor


class VisibilityProbe(nn.Module):
    """``norm`` LayerNorm(dim) on the pooled tokens, then ``mlp`` ``Linear(dim + 5, hidden) -> SiLU
    -> Linear(hidden, hidden) -> SiLU -> Linear(hidden, 1)`` on ``[pooled | 5 geometry]``.

    Args:
        dim (int): model width.
        config (VisibilityProbeConfig): the DiT block it reads after and the MLP's width.
    """

    def __init__(self, dim: int, config: VisibilityProbeConfig = VisibilityProbeConfig()) -> None:
        super().__init__()
        self.dit_block = config.dit_block
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim + GEOMETRY_FEATURES, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 1),
        )

    def forward(self, hidden: torch.Tensor, inputs: VisibilityProbeInputs) -> torch.Tensor:
        """Visibility logits of every latent frame of the window.

        Args:
            hidden (Tensor): ``[B, F h w, dim]`` output of DiT block :attr:`dit_block` (under
                teacher forcing, of the noisy copy); read detached.
            inputs (VisibilityProbeInputs): the window's inputs; the footprint gives the token
                grid ``(h, w)``.

        Returns:
            Tensor: ``[B, F, P]`` logits, float32, or the dtype of the generator's CUDA autocast
            (bf16 on the paper path) inside it.
        """
        batch, frames, _, grid_h, grid_w = inputs.footprint.shape
        if hidden.shape[:2] != (batch, frames * grid_h * grid_w):
            raise ValueError(
                f"the hidden state {tuple(hidden.shape)} is not [B, F h w, dim] of the footprint"
                f" {tuple(inputs.footprint.shape)}"
            )
        tokens = hidden.detach().reshape(batch, frames, grid_h * grid_w, -1)  # [B, F, hw, dim]
        weights = inputs.footprint.to(tokens.dtype).flatten(3)  # [B, F, P, hw]
        coverage = weights.sum(-1)  # [B, F, P]
        pooled = (
            torch.einsum("bfpk,bfkd->bfpd", weights, tokens) / coverage.clamp_min(1e-6)[..., None]
        )
        pooled = self.norm(pooled.float())
        relative_yaw = inputs.relative_yaw.float()
        geometry = torch.stack(
            [
                log_compressed_depth(inputs.depth.float()),
                torch.sin(relative_yaw),
                torch.cos(relative_yaw),
                coverage.float().clamp(max=_COVERAGE_CAP) / _COVERAGE_CAP,
                inputs.in_front.float(),
            ],
            dim=-1,
        )
        return self.mlp(torch.cat([pooled, geometry], dim=-1)).squeeze(-1)


def visibility_bce(
    logits: torch.Tensor, visible: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Masked, class-balanced BCE of the probe, a scalar.

    Positives are weighted ``min(negatives / positives, POSITIVE_WEIGHT_CAP)`` (at most 10);
    unknown labels get zero weight.

    Args:
        logits (Tensor): ``[B, F, P]`` logits.
        visible (Tensor): ``[B, F, P]`` labels.
        valid (Tensor): ``[B, F, P]`` confirmed labels (visible or occluded).
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
    return (per_position * weight).sum() / weight.sum().clamp_min(1e-12)
