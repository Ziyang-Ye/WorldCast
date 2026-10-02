"""The state injector ``Conv_0`` of Eq. (2): ``h_{f,u} <- h_{f,u} + Conv_0(F_{n,f})(u)``.

``F_{n,f}`` is the player state field of latent frame ``f``, 23 channels on the 12 x 21 token grid,
built by ``worldcast.player_state.field`` and added after the second DiT block. Its four weapon
channels come from a learned ``Embedding(52, 4)`` trained with this stem; it lives here so that the
checkpoint loads in one place, and the field builder reads it from the loaded generator.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = [
    "FIELD_CHANNELS",
    "FIELD_CHANNEL_NAMES",
    "ZERO_INIT_LANES",
    "StateInjector",
    "StateInjectorConfig",
]

#: Channels of the player state field (the paper's appendix on the field, "Channels").
FIELD_CHANNEL_NAMES = (
    "coverage",
    "log_depth",
    "sin_relative_yaw",
    "cos_relative_yaw",
    "team",
    "headcount",
    "ctl_forward",
    "ctl_back",
    "ctl_move_left",
    "ctl_move_right",
    "ctl_jump",
    "ctl_duck",
    "ctl_speed",
    "ctl_attack",
    "ctl_reload",
    "weapon_0",
    "weapon_1",
    "weapon_2",
    "weapon_3",
    "dying",
    "corpse",
    "live_identity",
    "corpse_identity",
)
#: 23.
FIELD_CHANNELS = len(FIELD_CHANNEL_NAMES)
#: Lanes whose stem input columns start at zero (the death and identity lanes, last in the layout),
#: as trained: the stem was drawn over the other 19 lanes and widened by zero columns.
ZERO_INIT_LANES = ("dying", "corpse", "live_identity", "corpse_identity")


@dataclass(frozen=True)
class StateInjectorConfig:
    """Shape of the injector; the defaults are the paper's."""

    #: Index of the DiT block the field is added after: 1, the second block.
    write_block: int = 1
    field_channels: int = FIELD_CHANNELS
    hidden: int = 32
    weapon_vocab: int = 52
    weapon_channels: int = 4


class StateInjector(nn.Module):
    """``Conv_0``: ``stem`` ``Conv2d(23, 32, 3, padding=1)`` -> SiLU -> ``proj`` ``Conv2d(32, dim,
    1)`` (zero-initialised), frame by frame; and the field's ``weapon_embedding`` ``Embedding(52,
    4)``.

    A fresh stem is built as trained: the default ``Conv2d`` initialisation over the first 19 lanes,
    then zero input columns for :data:`ZERO_INIT_LANES`. Loaded weights replace all of it.
    """

    def __init__(self, dim: int, config: StateInjectorConfig = StateInjectorConfig()) -> None:
        super().__init__()
        self.write_block = config.write_block
        self.field_channels = config.field_channels
        self.weapon_embedding = nn.Embedding(config.weapon_vocab, config.weapon_channels)
        zero_lanes = len(ZERO_INIT_LANES) if config.field_channels == FIELD_CHANNELS else 0
        self.stem = nn.Conv2d(
            config.field_channels - zero_lanes, config.hidden, kernel_size=3, padding=1
        )
        if zero_lanes:
            weight = self.stem.weight.detach()
            zeros = weight.new_zeros(weight.shape[0], zero_lanes, *weight.shape[2:])
            self.stem.weight = nn.Parameter(torch.cat([weight, zeros], dim=1))
            self.stem.in_channels = config.field_channels
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(config.hidden, dim, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def delta(self, field: torch.Tensor) -> torch.Tensor:
        """``Conv_0`` of each frame: ``[B, F, C, h, w]`` -> ``[B, F h w, dim]`` in token order."""
        batch, frames, channels, grid_h, grid_w = field.shape
        flat = field.reshape(batch * frames, channels, grid_h, grid_w)
        out = self.proj(self.act(self.stem(flat)))
        out = out.reshape(batch, frames, -1, grid_h * grid_w)
        return out.permute(0, 1, 3, 2).reshape(batch, frames * grid_h * grid_w, -1)

    def forward(
        self, hidden: torch.Tensor, field: torch.Tensor, *, frame_offset: int, num_frames: int
    ) -> torch.Tensor:
        """``hidden + Conv_0`` of the field rows ``[frame_offset, frame_offset + num_frames)``.

        Args:
            hidden (Tensor): ``[B, num_frames h w, dim]`` tokens after block ``write_block``.
            field (Tensor): ``[B, F_window, 23, h, w]``, or the call's rows
                ``[B, num_frames, 23, h, w]``; any float dtype.
            frame_offset (int): window index of the call's first latent frame.
            num_frames (int): latent frames of the call.
        """
        if field.shape[1] != num_frames:
            field = field[:, frame_offset : frame_offset + num_frames]
        return hidden + self.delta(field)
