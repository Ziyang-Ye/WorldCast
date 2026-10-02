"""The observer's controls ``a_n`` of Eq. (1): buttons, camera deltas and weapon, via AdaLN.

Latent frame ``j`` reads the 20 pixel frames that end at pixel frame ``4 j`` (11 buttons, 2 camera
deltas and a 32-d weapon embedding each: 900 numbers) and maps them to one ``[dim]`` embedding.
Every DiT block, and the head, adds a zero-initialised low-rank adapter of it to its AdaLN
modulation.
"""

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["ActionConfig", "ControlConditioner", "adaln_adapter"]


@dataclass(frozen=True)
class ActionConfig:
    """Shape of the control conditioner; the defaults are the paper's."""

    button_dim: int = 11
    camera_dim: int = 2
    weapon_vocab_size: int = 52
    weapon_embedding_dim: int = 32
    vae_time_compression_ratio: int = 4
    history_frames: int = 20
    hidden_dim: int = 1024
    adaln_rank: int = 128


class ControlConditioner(nn.Module):
    """Controls and weapon over a 20-frame history -> one ``[dim]`` embedding per latent frame.

    ``weapon_embedding`` is ``Embedding(52, 32)``; ``net`` is ``LayerNorm(900) -> Linear(900, 1024)
    -> SiLU -> Linear(1024, dim)``.
    """

    def __init__(self, dim: int, config: ActionConfig = ActionConfig()) -> None:
        super().__init__()
        self.weapon_vocab_size = config.weapon_vocab_size
        self.vae_time_compression_ratio = config.vae_time_compression_ratio
        self.history_frames = config.history_frames
        self.weapon_embedding = nn.Embedding(config.weapon_vocab_size, config.weapon_embedding_dim)
        in_dim = config.history_frames * (
            config.button_dim + config.camera_dim + config.weapon_embedding_dim
        )
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, dim),
        )

    def group_history(
        self, actions: torch.Tensor, latent_frames: int, *, latent_start: int
    ) -> torch.Tensor:
        """Each latent frame's control history.

        Args:
            actions (Tensor): ``[B, T, A]`` per-pixel-frame features in window order,
                ``T >= 1 + 4 (latent_start + latent_frames - 1)``.
            latent_frames (int): latent frames of the call.
            latent_start (int): window index of the call's first latent frame.

        Returns:
            Tensor: ``[B, latent_frames, 20 A]``; latent ``j`` reads the rows
            ``4 (latent_start + j) - 19 .. 4 (latent_start + j)``, clamped at row 0.
        """
        batch_size, pixel_frames, action_dim = actions.shape
        last = latent_start + latent_frames - 1
        if pixel_frames < 1 + self.vae_time_compression_ratio * last:
            raise ValueError(
                f"{pixel_frames} pixel frames of controls do not reach latent frame {last}"
            )
        endpoints = self.vae_time_compression_ratio * (
            latent_start + torch.arange(latent_frames, device=actions.device, dtype=torch.long)
        )
        history = torch.arange(1 - self.history_frames, 1, device=actions.device, dtype=torch.long)
        indices = (endpoints[:, None] + history[None]).clamp_min(0)
        return actions[:, indices].reshape(
            batch_size, latent_frames, self.history_frames * action_dim
        )

    def forward(
        self,
        buttons: torch.Tensor,
        camera: torch.Tensor,
        weapon: torch.Tensor,
        *,
        latent_frames: int,
        latent_start: int,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Control embedding ``[B, latent_frames, dim]`` of the call's latent frames.

        Args:
            buttons (Tensor): ``[B, T, 11]`` held buttons per pixel frame (0 / 1).
            camera (Tensor): ``[B, T, 2]`` camera deltas per pixel frame.
            weapon (Tensor): ``[B, T]`` weapon ids in ``[0, 52)``.
            latent_frames (int): latent frames of the call.
            latent_start (int): window index of the call's first latent frame.
            dtype (torch.dtype | None): output dtype (the generator passes its token dtype).
        """
        if weapon.numel() and (weapon.min() < 0 or weapon.max() >= self.weapon_vocab_size):
            raise ValueError("weapon condition contains an out-of-range id")
        actions = torch.cat(
            [buttons, camera, self.weapon_embedding(weapon.long()).to(buttons.dtype)], dim=-1
        )
        output = self.net(self.group_history(actions, latent_frames, latent_start=latent_start))
        return output if dtype is None else output.to(dtype=dtype)


def adaln_adapter(dim: int, multiplier: int, rank: int) -> nn.Sequential:
    """Zero-initialised low-rank AdaLN adapter: ``SiLU -> Linear(dim, rank, no bias) -> SiLU ->
    Linear(rank, multiplier * dim)``; 6 vectors for a DiT block, 2 for the head."""
    module = nn.Sequential(
        nn.SiLU(),
        nn.Linear(dim, rank, bias=False),
        nn.SiLU(),
        nn.Linear(rank, multiplier * dim, bias=True),
    )
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module
