"""The state injector ``Conv_0`` of Eq. (2): ``h_{f,u} <- h_{f,u} + Conv_0(F_{n,f})(u)``.

``F_{n,f}`` is the player state field of latent frame ``f``, 23 channels on the 12 x 21 token grid,
built by ``worldcast.player_state.field`` and added after the second DiT block. This module owns
the field's channel layout, which the stem was trained on, and the field's learned
``Embedding(52, 4)`` of the held weapon, which was trained with the stem: the checkpoint loads in
one place, and the field builder reads the embedding from the loaded generator.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from worldcast.modeling.controls import WEAPONS

__all__ = [
    "DEPTH_LOG_SCALE",
    "FIELD_CHANNELS",
    "FIELD_CHANNEL_NAMES",
    "FIELD_CONTROLS",
    "WEAPON_CHANNELS",
    "ZERO_INIT_CHANNELS",
    "StateInjector",
    "StateInjectorConfig",
    "log_compressed_depth",
]

#: The nine controls the field carries, in channel order (App. "Channels"), under the recordings'
#: button names: ``move_left`` / ``move_right`` strafe, ``duck`` crouches, ``speed`` walks.
FIELD_CONTROLS = (
    "forward",
    "back",
    "move_left",
    "move_right",
    "jump",
    "duck",
    "speed",
    "attack",
    "reload",
)
#: Channels of the field's learned weapon embedding.
WEAPON_CHANNELS = 4
#: The channels of the player state field, in order (App. "Channels").
FIELD_CHANNEL_NAMES = (
    "coverage",
    "depth",
    "yaw_sin",
    "yaw_cos",
    "team",
    "headcount",
    *(f"controls_{name}" for name in FIELD_CONTROLS),
    *(f"weapon_{k}" for k in range(WEAPON_CHANNELS)),
    "dying",
    "corpse",
    "identity_live",
    "identity_corpse",
)
#: Channels of the field: 23.
FIELD_CHANNELS = len(FIELD_CHANNEL_NAMES)
#: The last four channels: their stem input columns start at zero, as trained (the stem was drawn
#: over the other 19 and widened by zero columns).
ZERO_INIT_CHANNELS = FIELD_CHANNEL_NAMES[-4:]
#: The scale of the field's depth channel, ``log1p(depth) / DEPTH_LOG_SCALE``: 4096 u maps to 1
#: and a larger depth clamps there. A constant of the channel as trained, apart from the depth
#: head's far plane (``worldcast.scene_state.geometry.FAR_U``, 4096 u as well).
DEPTH_LOG_SCALE = math.log1p(4096.0)


def log_compressed_depth(depth: torch.Tensor) -> torch.Tensor:
    """The field's depth channel: ``log1p(depth) / log1p(4096)``, clamped to [0, 1]; ``depth`` in
    u, any shape."""
    return (torch.log1p(depth.clamp(min=0.0)) / DEPTH_LOG_SCALE).clamp(0.0, 1.0)


@dataclass(frozen=True)
class StateInjectorConfig:
    """Shape of the injector; the defaults are the paper's."""

    #: Index of the DiT block the field is added after, from 0: 1, the second DiT block.
    dit_block: int = 1
    #: Channels between ``stem`` and ``proj``.
    hidden: int = 32


class StateInjector(nn.Module):
    """``Conv_0``: ``stem`` ``Conv2d(23, 32, 3, padding=1)`` -> SiLU -> ``proj`` ``Conv2d(32, dim,
    1)`` (zero-initialised), frame by frame; and the field's ``weapon_embedding`` ``Embedding(52,
    4)``.

    A fresh injector draws, in order, the weapon embedding, a stem over the first 19 channels (then
    widened by zero input columns for :data:`ZERO_INIT_CHANNELS`) and ``proj``, as trained: a
    23-channel draw would shift every module initialised after it. Loaded weights replace all of
    it.

    Args:
        dim (int): model width.
        config (StateInjectorConfig): the DiT block the field is added after and the stem's
            channels.
    """

    def __init__(self, dim: int, config: StateInjectorConfig = StateInjectorConfig()) -> None:
        super().__init__()
        self.dit_block = config.dit_block
        self.weapon_embedding = nn.Embedding(WEAPONS, WEAPON_CHANNELS)
        zero = len(ZERO_INIT_CHANNELS)
        self.stem = nn.Conv2d(FIELD_CHANNELS - zero, config.hidden, kernel_size=3, padding=1)
        weight = self.stem.weight.detach()
        zeros = weight.new_zeros(weight.shape[0], zero, *weight.shape[2:])
        self.stem.weight = nn.Parameter(torch.cat([weight, zeros], dim=1))
        self.stem.in_channels = FIELD_CHANNELS
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(config.hidden, dim, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def delta(self, field: torch.Tensor) -> torch.Tensor:
        """``Conv_0`` of each frame: ``[B, F, 23, h, w]`` -> ``[B, F h w, dim]`` in token order."""
        if field.ndim != 5 or field.shape[2] != FIELD_CHANNELS:
            raise ValueError(
                f"the field must be [B, F, {FIELD_CHANNELS}, h, w], got {tuple(field.shape)}"
            )
        batch, frames, channels, grid_h, grid_w = field.shape
        flat = field.reshape(batch * frames, channels, grid_h, grid_w)
        out = self.proj(self.act(self.stem(flat)))
        out = out.reshape(batch, frames, -1, grid_h * grid_w)
        return out.permute(0, 1, 3, 2).reshape(batch, frames * grid_h * grid_w, -1)

    def forward(
        self, hidden: torch.Tensor, field: torch.Tensor, *, copies: int = 1
    ) -> torch.Tensor:
        """``hidden + Conv_0(field)``.

        Args:
            hidden (Tensor): ``[B, copies F h w, dim]`` tokens after DiT block :attr:`dit_block`.
            field (Tensor): ``[B, F, 23, h, w]`` the field of the call's ``F`` latent frames, in
                the dtype of the injector's parameters (any float dtype inside CUDA autocast).
            copies (int): copies of the call's frames back to back in ``hidden``, each receiving
                the same delta (2: the teacher-forcing ``[context | noisy]`` of the runs without
                scene state).
        """
        delta = self.delta(field)
        if copies < 1 or (delta.shape[0], copies * delta.shape[1]) != hidden.shape[:2]:
            raise ValueError(
                f"{copies} copies of the field {tuple(field.shape)} do not cover the hidden state"
                f" {tuple(hidden.shape)}, [B, copies F h w, dim]"
            )
        if copies != 1:
            delta = torch.cat([delta] * copies, dim=1)
        return hidden + delta
