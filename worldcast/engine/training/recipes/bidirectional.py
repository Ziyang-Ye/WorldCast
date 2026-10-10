"""Stages 1 and 2: flow matching on one bidirectional window (41 latent frames; 21 in stage 1).

The step of :class:`~worldcast.engine.training.recipes.flow_matching.FlowMatchingTrainer` with every
token attending to every token of the window: one timestep per window, the first frame pinned clean
at t = 0. With scene state (stage 2s) the memory window is first frame | memory frames 4 | recent
context 32 | target frames 4, its memory frames pinned clean with the first frame.
"""

from typing import Any

from ..losses import FlowMatchingSample
from .flow_matching import FlowMatchingTrainer

__all__ = ["BidirectionalTrainer"]


class BidirectionalTrainer(FlowMatchingTrainer):
    """Stages 1, 1_long, 2 and 2s (module docstring)."""

    bidirectional = True

    def forward_kwargs(self, sample: FlowMatchingSample) -> dict[str, Any]:
        """Nothing besides the noisy window."""
        return {}
