"""Observer signals: whether the observer is flashed, and whether and how far it is scoped in.

Not in the paper (docs/inference.md, "Paper vs code"). Each signal has a validity flag; an invalid
row reads a learned "unknown" embedding. The embedding of a latent frame is added to every token of
it after the patch embedding.
"""

import torch
import torch.nn as nn

__all__ = ["OBS_SIGNAL_KEYS", "SCOPE_LEVELS", "ObserverSignalBranch"]

#: Condition names, in the order :meth:`ObserverSignalBranch.forward` takes them.
OBS_SIGNAL_KEYS = (
    "obs_flash_flag",
    "obs_flash_valid",
    "obs_scope_on",
    "obs_scope_level",
    "obs_scope_valid",
)
#: Scope zoom levels: 0 unscoped, 1 first zoom, 2 second zoom.
SCOPE_LEVELS = 3
#: Width of each signal's embedding.
_EMBED = 8


def _embed(table: nn.Embedding, unknown: torch.Tensor, index: torch.Tensor, valid: torch.Tensor):
    known = table(index)
    return torch.where(valid.bool().unsqueeze(-1), known, unknown.expand_as(known))


class ObserverSignalBranch(nn.Module):
    """``[B, F]`` flash and scope signals -> ``[B, F, dim]``: three 8-d embeddings, an MLP and a
    zero-initialised output layer."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.flash_known = nn.Embedding(2, _EMBED)
        self.scope_on_known = nn.Embedding(2, _EMBED)
        self.level_known = nn.Embedding(SCOPE_LEVELS, _EMBED)
        self.flash_unknown = nn.Parameter(torch.randn(_EMBED) * 0.02)
        self.scope_on_unknown = nn.Parameter(torch.randn(_EMBED) * 0.02)
        self.level_unknown = nn.Parameter(torch.randn(_EMBED) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(3 * _EMBED, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        flash_flag: torch.Tensor,
        flash_valid: torch.Tensor,
        scope_on: torch.Tensor,
        scope_level: torch.Tensor,
        scope_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Embed one window's signals, each ``[B, F]`` read as integers; ``*_valid = 0`` reads
        "unknown"."""
        flash = _embed(
            self.flash_known, self.flash_unknown, flash_flag.long().clamp(0, 1), flash_valid
        )
        scope_on = _embed(
            self.scope_on_known, self.scope_on_unknown, scope_on.long().clamp(0, 1), scope_valid
        )
        level = _embed(
            self.level_known,
            self.level_unknown,
            scope_level.long().clamp(0, SCOPE_LEVELS - 1),
            scope_valid,
        )
        return self.out(self.mlp(torch.cat([flash, scope_on, level], dim=-1)))
