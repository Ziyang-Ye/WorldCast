"""The training window of one micro-batch: the ordinary window, or the memory window.

:func:`memory_this_step` decides per optimizer step, the same on every rank, whether the step trains
on memory windows (App. E, "Training": with probability 0.8), and :func:`training_window` builds
what the generator trains on.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from worldcast.config.training import TrainConfig
from worldcast.data.memory_frames import MEMORY_FRAMES_PREFIX
from worldcast.data.memory_selection import BIDIRECTIONAL, BLOCK_CAUSAL, MemoryFrameConfig
from worldcast.modeling.ray_embedding import RayConditions
from worldcast.player_state import continuous_row_state, memory_continuous_columns
from worldcast.sampling.window import (
    MEMORY_CONTINUOUS_COLUMNS_KEY,
    ROUND_CONTINUOUS_COLUMNS_KEY,
    WindowLayout,
    gather_window,
)

__all__ = [
    "MEMORY_PROB",
    "MemoryWindowConfig",
    "TrainingWindow",
    "memory_frame_config",
    "memory_this_step",
    "memory_windows",
    "training_window",
]

#: The memory windows' share of the optimizer steps, with scene state.
MEMORY_PROB = 0.8
#: The key of the per-step memory draw, as trained.
_MEMORY_DRAW_KEY = "worldplay-windowout"


def memory_this_step(seed: int, step: int, prob: float) -> bool:
    """Whether optimizer step ``step`` trains on memory windows: a Bernoulli(``prob``) draw keyed
    on ``(seed, step)``, the same on every rank and every micro-batch, and on resume."""
    if prob >= 1.0 or prob <= 0.0:
        return prob >= 1.0
    key = f"{_MEMORY_DRAW_KEY}:{int(seed)}:{int(step)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64) < prob


@dataclass(frozen=True)
class MemoryWindowConfig:
    """The memory windows of a stage.

    Attributes:
        prob (float): share of the optimizer steps that train on memory windows.
        recent (int): recent context latent frames of a memory window.
        continuous_camera (bool): the player state table of a memory window is integrated on the
            whole window and then cut, rather than integrated on the cut rows (stage 3).
    """

    prob: float
    recent: int
    continuous_camera: bool = False


def memory_frame_config(cfg: TrainConfig) -> MemoryFrameConfig:
    """The memory frames of a stage's windows: those of a bidirectional or of a block-causal
    generator."""
    return BIDIRECTIONAL if cfg.stage.recipe == "bidirectional" else BLOCK_CAUSAL


def memory_windows(cfg: TrainConfig) -> MemoryWindowConfig | None:
    """The memory windows of a flow-matching stage; ``None`` without scene state."""
    if not cfg.stage.scene_state:
        return None
    teacher_forcing = cfg.stage.recipe != "bidirectional"
    return MemoryWindowConfig(MEMORY_PROB, memory_frame_config(cfg).recent, teacher_forcing)


@dataclass
class TrainingWindow:
    """What the generator trains on for one micro-batch.

    Attributes:
        batch (dict[str, Any]): the ordinary window (the collated batch itself) or the memory
            window: first frame | memory frames | recent context | target frames.
        has_memory (bool): which one.
        rays (RayConditions | None): the memory window's cameras.
        pinned_frames (int): leading frames trained clean at t = 0 without a loss: the first frame,
            and under bidirectional attention the memory frames too.
        loss_mask (Tensor | None): ``[B, F]`` float32, 1 on the target frames of a memory window.
        memory_mask (Tensor | None): m_k ``[B, 4, 12, 21]`` bool of a memory window.
    """

    batch: dict[str, Any]
    has_memory: bool = False
    rays: RayConditions | None = None
    pinned_frames: int = 1
    loss_mask: torch.Tensor | None = None
    memory_mask: torch.Tensor | None = None

    def ray_conditions(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        """The ray embedding's camera inputs: the memory window's, or an ordinary window's (no
        memory frames, anchored at latent frame 0)."""
        rays = self.rays
        if rays is None:
            rays = RayConditions.contiguous(self.batch["window_c2w"], self.batch["window_tans"])
        return rays.conditions(device)


def training_window(
    batch: dict[str, Any],
    *,
    step: int,
    seed: int,
    memory: MemoryWindowConfig | None,
    bidirectional: bool,
) -> TrainingWindow:
    """This micro-batch's training window: the ordinary window, or, on a step that
    :func:`memory_this_step` draws, the memory window around the batch's target block.

    Args:
        batch (dict[str, Any]): the collated batch.
        step (int): the optimizer step.
        seed (int): the run's base seed.
        memory (MemoryWindowConfig | None): the stage's memory windows, ``None`` without.
        bidirectional (bool): the memory frames are pinned clean with the first frame.
    """
    if memory is None or not memory_this_step(seed, step, memory.prob):
        return TrainingWindow(batch)
    source = batch
    if memory.continuous_camera:
        columns = continuous_row_state(batch)
        source = dict(batch)
        source[ROUND_CONTINUOUS_COLUMNS_KEY] = columns
        source[MEMORY_CONTINUOUS_COLUMNS_KEY] = memory_continuous_columns(
            columns, batch[MEMORY_FRAMES_PREFIX + "states"], batch["client_slot"]
        )
    layout = WindowLayout(memory.recent, with_memory=True)
    gathered, rays = gather_window(source, layout)
    loss_mask = torch.zeros(len(batch["latents"]), layout.num_frames)
    loss_mask[:, layout.target_positions] = 1.0
    return TrainingWindow(
        batch=gathered,
        has_memory=True,
        rays=rays,
        pinned_frames=1 + len(layout.memory_positions) if bidirectional else 1,
        loss_mask=loss_mask,
        memory_mask=batch["window_memory_mask"].bool().clone(),
    )
