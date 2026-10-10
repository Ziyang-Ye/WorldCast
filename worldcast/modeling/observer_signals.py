"""Observer signals: whether the client's player is flashed, and whether and how far it is scoped
in.

Not in the paper's description of the generator. Each signal has a validity flag; an invalid entry
reads a learned "unknown" embedding. The embedding of a latent frame is added to every token of it
after the patch embedding.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from worldcast.data.labels import OBSERVER_SIGNAL_KEYS, SCOPE_LEVELS

__all__ = ["ObserverSignalConfig", "ObserverSignalEmbedding", "ObserverSignals"]

#: Width of each signal's embedding.
_EMBED = 8


@dataclass(frozen=True)
class ObserverSignalConfig:
    """Shape of the embedding; the default is that of the paper's runs."""

    #: Width of the MLP.
    hidden: int = 128


@dataclass(eq=False)
class ObserverSignals:
    """The observer signals of one window, each ``[B, F_window]`` and read as integers; in a
    condition dict, the entries ``worldcast.data.labels.OBSERVER_SIGNAL_KEYS``: ``obs_`` and the
    field names.

    Attributes:
        flash_flag (Tensor): 1 while the client's player is flashed, else 0.
        flash_valid (Tensor): 1 where ``flash_flag`` is known; 0 reads the "unknown" embedding.
        scope_on (Tensor): 1 while it is scoped in, else 0.
        scope_level (Tensor): the zoom level: 0, 1 or 2 (``SCOPE_LEVELS`` = 3).
        scope_valid (Tensor): 1 where ``scope_on`` and ``scope_level`` are known.
    """

    flash_flag: torch.Tensor
    flash_valid: torch.Tensor
    scope_on: torch.Tensor
    scope_level: torch.Tensor
    scope_valid: torch.Tensor

    def __post_init__(self) -> None:
        shapes = {name: tuple(signal.shape) for name, signal in vars(self).items()}
        if len(set(shapes.values())) != 1 or self.flash_flag.ndim != 2:
            raise ValueError(f"the observer signals share one shape [B, F], got {shapes}")

    @classmethod
    def from_conditions(cls, conditions: Mapping[str, Any]) -> "ObserverSignals | None":
        """The signals of a condition dict; ``None`` when it has none of them (they come all
        together or not at all)."""
        given = [conditions.get(key) is not None for key in OBSERVER_SIGNAL_KEYS]
        if not any(given):
            return None
        if not all(given):
            raise ValueError(
                f"the observer signals arrive together or not at all: {OBSERVER_SIGNAL_KEYS}"
            )
        return cls(**{key.removeprefix("obs_"): conditions[key] for key in OBSERVER_SIGNAL_KEYS})

    def to(self, device: torch.device | str) -> "ObserverSignals":
        """The signals on ``device``."""
        moved = {name: signal.to(device=device) for name, signal in vars(self).items()}
        return ObserverSignals(**moved)


def _embed(
    table: nn.Embedding, unknown: torch.Tensor, index: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    known = table(index)
    return torch.where(valid.bool().unsqueeze(-1), known, unknown.expand_as(known))


class ObserverSignalEmbedding(nn.Module):
    """:class:`ObserverSignals` ``[B, F]`` -> ``[B, F, dim]``: three 8-d embeddings (flash, scope,
    zoom level), an MLP and a zero-initialised output layer.

    Args:
        dim (int): model width.
        config (ObserverSignalConfig): width of the MLP.
    """

    def __init__(self, dim: int, config: ObserverSignalConfig = ObserverSignalConfig()) -> None:
        super().__init__()
        hidden = config.hidden
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

    def forward(self, signals: ObserverSignals) -> torch.Tensor:
        """The embedding ``[B, F, dim]`` of one window's signals; a value outside its range is
        clamped into it, as trained."""
        flash = _embed(
            self.flash_known,
            self.flash_unknown,
            signals.flash_flag.long().clamp(0, 1),
            signals.flash_valid,
        )
        scope_on = _embed(
            self.scope_on_known,
            self.scope_on_unknown,
            signals.scope_on.long().clamp(0, 1),
            signals.scope_valid,
        )
        level = _embed(
            self.level_known,
            self.level_unknown,
            signals.scope_level.long().clamp(0, SCOPE_LEVELS - 1),
            signals.scope_valid,
        )
        return self.out(self.mlp(torch.cat([flash, scope_on, level], dim=-1)))
