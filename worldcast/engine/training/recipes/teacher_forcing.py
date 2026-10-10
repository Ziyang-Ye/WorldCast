"""Stage 3: teacher forcing under block-causal attention, with and without scene state.

The step of :class:`~worldcast.engine.training.recipes.flow_matching.FlowMatchingTrainer` on the
sequence ``[clean context copy | noisy window]``: one timestep per block (the first frame, then 10
blocks of 4 latent frames), a noisy token of block g attending to the clean tokens of the blocks
before g and to the noisy tokens of block g. With scene state the context copy is noised per block
to a timestep in [16, 32), the first frame and the memory frames included, and on 80 % of the
steps the window is the memory window: first frame | memory frames 4 | recent context 12 | target
frames 4. Without scene state the context copy is clean, and the player state field is added to it
as well as to the noisy window, as trained.
"""

from typing import Any

from ..losses import FlowMatchingSample
from .flow_matching import FlowMatchingTrainer

__all__ = ["TeacherForcingTrainer"]


class TeacherForcingTrainer(FlowMatchingTrainer):
    """Stage 3 (module docstring)."""

    bidirectional = False

    def forward_kwargs(self, sample: FlowMatchingSample) -> dict[str, Any]:
        """The context copy, its timesteps and whether the field is added to it."""
        return {
            "context_latents": sample.context,
            "context_timestep": sample.context_timestep,
            "field_on_context": not self.cfg.stage.scene_state,
        }
