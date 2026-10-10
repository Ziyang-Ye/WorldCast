"""The depth head and its read-out (Sec. 3.4, App. "Depth head"), frozen at inference.

RGB latents ``[F, 48, 24, 42]`` -> log axial depth ``[F, 4, 24, 42]`` (engine units) of each
latent frame's four video frames, one value per 16 x 16-pixel patch; channel ``-1`` is the latent
frame's last video frame. Each latent frame is seen with its temporal neighbours, edge-replicated
inside the given frames, so a block's depth depends on its own latents only. The scene state and
the predicted visibility read it. It runs in float32; the reference runs allow TF32 for it
(:func:`worldcast.utils.precision.enable_tf32`). The short parameter names (``inp``, ``dblocks``,
``outn``, ``n1``, ``c1``, ...) are the keys of the released depth files.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from worldcast.data.latents import LATENT_CHANNELS, LATENT_GRID
from worldcast.utils.weights import (
    indexed_count,
    leading_size,
    module_from_state,
    read_state_dict,
)

__all__ = [
    "DepthFn",
    "DepthHead",
    "DepthPredictor",
    "DepthReadout",
    "ResidualBlock",
    "load_depth_predictor",
]

#: A depth function, as the scene state and the predicted visibility take it: latent frames ``[F,
#: 48, 24, 42]`` (a tensor or an array) -> log axial depth ``[F, 4, 24, 42]`` float32
#: (:meth:`DepthPredictor.log_depth`).
DepthFn = Callable[[torch.Tensor | np.ndarray], np.ndarray]
#: The read-out predicts the log depth relative to 200 u.
_LOG_DEPTH_OFFSET = float(np.log(200.0))


class ResidualBlock(nn.Module):
    """Residual ``GroupNorm(8) -> SiLU -> Conv3x3``, twice."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.n1 = nn.GroupNorm(8, channels)
        self.c1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, channels)
        self.c2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[N, C, h, w]`` -> the same shape."""
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class DepthHead(nn.Module):
    """The depth head: RGB latents -> the depth-video latents of the same frames, ``[B, F, 48, h,
    w]`` -> ``[B, F, 48, h, w]``, each frame from three latent frames of context.

    A residual network with a half-resolution branch; the defaults are the paper's (44M
    parameters).

    Args:
        width (int): channels of the residual blocks.
        blocks (int): residual blocks at full resolution.
        half_blocks (int): residual blocks of the half-resolution branch.
    """

    def __init__(self, width: int = 384, blocks: int = 10, half_blocks: int = 5) -> None:
        super().__init__()
        self.inp = nn.Conv2d(3 * LATENT_CHANNELS, width, 3, 1, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.down = nn.Conv2d(width, width, 3, 2, 1)
        self.dblocks = nn.Sequential(*[ResidualBlock(width) for _ in range(half_blocks)])
        self.up = nn.ConvTranspose2d(width, width, 4, 2, 1)
        self.outn = nn.GroupNorm(8, width)
        self.out = nn.Conv2d(width, LATENT_CHANNELS, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RGB latents ``[B, F, 48, h, w]`` -> depth-video latents ``[B, F, 48, h, w]``."""
        if x.ndim != 5 or x.shape[2] != LATENT_CHANNELS:
            raise ValueError(
                f"the depth head reads latents [B, F, {LATENT_CHANNELS}, h, w], got"
                f" {tuple(x.shape)}"
            )
        batch, frames = x.shape[:2]
        # each frame with its neighbours (f-1, f, f+1), edge-replicated, stacked on channels
        padded = torch.cat([x[:, :1], x, x[:, -1:]], 1)
        context = torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], 2).flatten(0, 1)
        h = self.blocks(self.inp(context))
        d = self.dblocks(self.down(h))
        h = h + self.up(d)[..., : h.shape[-2], : h.shape[-1]]
        return self.out(F.silu(self.outn(h))).view(batch, frames, -1, *x.shape[-2:])


class DepthReadout(nn.Module):
    """The read-out: a depth-video latent -> 4 log axial depths per latent frame, one per video
    frame, without VAE decoding (0.21M parameters).

    Latent frame ``j >= 1`` covers video frames ``4j-3 .. 4j``; latent frame 0 (one video frame)
    has that frame's depth on all four.

    Args:
        width (int): channels of its two hidden convolutions.
    """

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, width, 3, 1, 1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, 1, 1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, 4, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Depth-video latents ``[..., 48, h, w]`` (the output of :class:`DepthHead`) -> log axial
        depth ``[..., 4, h, w]``."""
        out = self.net(x.reshape(-1, *x.shape[-3:])) + _LOG_DEPTH_OFFSET
        return out.reshape(*x.shape[:-3], *out.shape[-3:])


class DepthPredictor:
    """The frozen depth head with its read-out: :meth:`log_depth` maps latents to log axial depth.

    Args:
        head (DepthHead): the depth head.
        readout (DepthReadout): its read-out.
        device (torch.device | str): where both run.
    """

    def __init__(
        self, head: DepthHead, readout: DepthReadout, device: torch.device | str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.head = head.to(self.device).eval().requires_grad_(False)
        self.readout = readout.to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def log_depth(self, latents: torch.Tensor | np.ndarray) -> np.ndarray:
        """Log axial depth of RGB latents, in float32 on :attr:`device`.

        Args:
            latents (Tensor | ndarray): ``[F, 48, 24, 42]`` latents, any float dtype.

        Returns:
            ndarray: ``[F, 4, 24, 42]`` float32 on the CPU.
        """
        x = latents if torch.is_tensor(latents) else torch.from_numpy(np.asarray(latents))
        x = x.detach().to(self.device, torch.float32)
        if x.ndim != 4 or tuple(x.shape[1:]) != (LATENT_CHANNELS, *LATENT_GRID):
            raise ValueError(
                f"latents must be [F, {LATENT_CHANNELS}, {LATENT_GRID[0]}, {LATENT_GRID[1]}], got"
                f" {tuple(x.shape)}"
            )
        features = self.head(x[None])[0]
        return self.readout(features.float()).float().cpu().numpy()


def load_depth_predictor(
    head_checkpoint: str | Path,
    readout_checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> DepthPredictor:
    """The depth head and its read-out of two weight files, each sized from its file, in float32.

    Args:
        head_checkpoint (str | Path): the depth head's weights (``depth_head.safetensors``).
        readout_checkpoint (str | Path): the read-out's (``depth_readout.safetensors``).
        device (torch.device | str): where both run.
    """
    state = read_state_dict(head_checkpoint)
    head = module_from_state(
        lambda: DepthHead(
            width=leading_size(state, "inp.weight"),
            blocks=indexed_count(state, "blocks."),
            half_blocks=indexed_count(state, "dblocks."),
        ),
        state,
    )
    state = read_state_dict(readout_checkpoint)
    readout = module_from_state(lambda: DepthReadout(leading_size(state, "net.0.weight")), state)
    return DepthPredictor(head.float(), readout.float(), device)
