"""The control embedding of ``a_n`` in Eq. (1): buttons, pitch / yaw deltas and held weapon.

The controls enter every DiT block through adaptive layer normalization (App. "Injection"). Latent
frame ``j`` reads the 20 video frames that end at video frame ``4 j`` (11 buttons, the pitch and
yaw deltas and a 32-d weapon embedding each: 900 numbers) and maps them to one ``[dim]`` embedding.
Every DiT block, and the head, adds a zero-initialised low-rank adapter of it
(:func:`~worldcast.modeling.wan22.dit.adaln_adapter`) to its AdaLN modulation.
"""

from dataclasses import dataclass

import torch
from torch import nn

from worldcast.data.controls import CONTROL_BUTTONS, OPENCS2_WEAPONS
from worldcast.data.latents import VIDEO_FRAMES_PER_LATENT

__all__ = [
    "BUTTONS",
    "CONTROL_KEYS",
    "HISTORY_FRAMES",
    "VIEW_DELTAS",
    "WEAPONS",
    "WEAPON_EMBEDDING_DIM",
    "ControlConfig",
    "ControlEmbedding",
]

#: Batch and condition-dict keys of the client's controls, the arguments of
#: :meth:`ControlEmbedding.forward`: ``[B, T, 11]`` buttons, ``[B, T, 2]`` pitch and yaw deltas and
#: ``[B, T]`` weapon ids over the ``T`` video frames of a window.
CONTROL_KEYS = ("buttons", "view_deltas", "weapon")
#: Buttons of a video frame's controls (11).
BUTTONS = len(CONTROL_BUTTONS)
#: The pitch delta and the yaw delta.
VIEW_DELTAS = 2
#: Weapons a weapon embedding distinguishes (52).
WEAPONS = len(OPENCS2_WEAPONS)
#: Width of the control embedding's weapon embedding.
WEAPON_EMBEDDING_DIM = 32
#: Video frames of control history a latent frame reads.
HISTORY_FRAMES = 20


@dataclass(frozen=True)
class ControlConfig:
    """Widths of the control embedding; the defaults are the paper's."""

    #: Hidden width of the embedding's MLP.
    hidden: int = 1024
    #: Rank of the AdaLN adapters of the DiT blocks and the head.
    adaln_rank: int = 128


class ControlEmbedding(nn.Module):
    """Controls and weapon over a 20-frame history -> one ``[dim]`` embedding per latent frame.

    ``weapon_embedding`` is ``Embedding(52, 32)``; ``net`` is ``LayerNorm(900) -> Linear(900, 1024)
    -> SiLU -> Linear(1024, dim)``.

    Args:
        dim (int): model width.
        config (ControlConfig): the MLP's hidden width.
    """

    def __init__(self, dim: int, config: ControlConfig = ControlConfig()) -> None:
        super().__init__()
        self.weapon_embedding = nn.Embedding(WEAPONS, WEAPON_EMBEDDING_DIM)
        in_dim = HISTORY_FRAMES * (BUTTONS + VIEW_DELTAS + WEAPON_EMBEDDING_DIM)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, dim),
        )

    def group_history(
        self, controls: torch.Tensor, *, frame_offset: int, num_frames: int
    ) -> torch.Tensor:
        """Each latent frame's control history.

        Args:
            controls (Tensor): ``[B, T, A]`` per-video-frame features in window order,
                ``T >= 1 + 4 (frame_offset + num_frames - 1)``.
            frame_offset (int): window index of the call's first latent frame.
            num_frames (int): latent frames of the call.

        Returns:
            Tensor: ``[B, num_frames, 20 A]``; latent frame ``j`` of the call reads the video
            frames ``4 (frame_offset + j) - 19 .. 4 (frame_offset + j)``, clamped at video frame 0.
        """
        batch_size, video_frames, width = controls.shape
        last = frame_offset + num_frames - 1
        if video_frames < 1 + VIDEO_FRAMES_PER_LATENT * last:
            raise ValueError(
                f"{video_frames} video frames of controls do not reach latent frame {last}"
            )
        endpoints = VIDEO_FRAMES_PER_LATENT * (
            frame_offset + torch.arange(num_frames, device=controls.device, dtype=torch.long)
        )
        history = torch.arange(1 - HISTORY_FRAMES, 1, device=controls.device, dtype=torch.long)
        indices = (endpoints[:, None] + history[None]).clamp_min(0)
        return controls[:, indices].reshape(batch_size, num_frames, HISTORY_FRAMES * width)

    def forward(
        self,
        buttons: torch.Tensor,
        view_deltas: torch.Tensor,
        weapon: torch.Tensor,
        *,
        frame_offset: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Control embedding ``[B, num_frames, dim]`` of the call's latent frames.

        Args:
            buttons (Tensor): ``[B, T, 11]`` held buttons per video frame (0 / 1), floating, in
                the dtype of ``net``'s parameters (any float dtype inside CUDA autocast).
            view_deltas (Tensor): ``[B, T, 2]`` pitch and yaw deltas per video frame, floating.
            weapon (Tensor): ``[B, T]`` integer weapon ids in ``[0, 52)``.
            frame_offset (int): window index of the call's first latent frame.
            num_frames (int): latent frames of the call.
        """
        if weapon.ndim != 2 or weapon.is_floating_point() or weapon.dtype == torch.bool:
            raise ValueError(
                f"weapon must hold integer ids [B, T], got {weapon.dtype} {tuple(weapon.shape)}"
            )
        floating = (("buttons", buttons, BUTTONS), ("view_deltas", view_deltas, VIEW_DELTAS))
        for name, value, width in floating:
            if tuple(value.shape) != (*weapon.shape, width) or not value.is_floating_point():
                raise ValueError(
                    f"{name} must be floating [B, T, {width}] over the T = {weapon.shape[1]} video"
                    f" frames of weapon, got {value.dtype} {tuple(value.shape)}"
                )
        if weapon.numel() and (weapon.min() < 0 or weapon.max() >= WEAPONS):
            raise ValueError("weapon contains an out-of-range id")
        controls = torch.cat(
            [buttons, view_deltas, self.weapon_embedding(weapon.long()).to(buttons.dtype)], dim=-1
        )
        history = self.group_history(controls, frame_offset=frame_offset, num_frames=num_frames)
        return self.net(history)
