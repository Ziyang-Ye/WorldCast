"""Picture depth: the frozen latent depth head and its read-out, used by the scene state.

RGB latents ``[T, 48, 24, 42]`` -> log axial depth ``[T, 4, 24, 42]`` (engine units) of each
latent's four pixel frames on the 16 x 16-pixel cell grid; channel ``-1`` is the latent's last pixel
frame. Each latent is seen with its temporal neighbours, edge-replicated inside the given frames, so
a block's depth depends on its own latents only. Runs in float32 (TF32 on CUDA).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GRID", "LATENT_CHANNELS", "Block", "DepthLat", "PictureDepth", "Readout"]

#: The read-out's 16 x 16-pixel cell grid of a 384 x 672 frame (the latent grid).
GRID = (24, 42)
#: Channels of a Wan2.2 latent.
LATENT_CHANNELS = 48


class Block(nn.Module):
    """Residual ``GroupNorm(8) -> SiLU -> Conv3x3``, twice."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.n1 = nn.GroupNorm(8, c)
        self.c1 = nn.Conv2d(c, c, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class DepthLat(nn.Module):
    """The latent depth head: ``[B, T, cin, h, w]`` -> ``[B, T, cin, h, w]`` features, one U-step.

    The paper's checkpoint has width 384, 10 blocks and 5 down blocks (44.2M parameters).

    Args:
        cin (int): latent channels.
        w (int): width.
        nb (int): residual blocks at full resolution.
        ndb (int | None): residual blocks at half resolution (default ``max(1, nb // 2)``).
    """

    def __init__(
        self, cin: int = LATENT_CHANNELS, w: int = 256, nb: int = 8, ndb: int | None = None
    ) -> None:
        super().__init__()
        self.inp = nn.Conv2d(3 * cin, w, 3, 1, 1)
        self.blocks = nn.Sequential(*[Block(w) for _ in range(nb)])
        self.down = nn.Conv2d(w, w, 3, 2, 1)
        self.dblocks = nn.Sequential(*[Block(w) for _ in range(ndb or max(1, nb // 2))])
        self.up = nn.ConvTranspose2d(w, w, 4, 2, 1)
        self.outn = nn.GroupNorm(8, w)
        self.out = nn.Conv2d(w, cin, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        # each frame with its neighbours (t-1, t, t+1), edge-replicated, stacked on channels
        padded = torch.cat([x[:, :1], x, x[:, -1:]], 1)
        context = torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], 2).flatten(0, 1)
        h = self.blocks(self.inp(context))
        d = self.dblocks(self.down(h))
        h = h + self.up(d)[..., : h.shape[-2], : h.shape[-1]]
        return self.out(F.silu(self.outn(h))).view(B, T, -1, x.shape[-2], x.shape[-1])


class Readout(nn.Module):
    """48 depth-head channels -> 4 log axial depths, one per pixel frame of the latent.

    Latent ``j >= 1`` covers pixel frames ``4j-3 .. 4j``; latent 0 (one pixel frame) has that
    frame's depth on all four. The output is offset by ``log(200)``.
    """

    def __init__(self, w: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(48, w, 3, 1, 1),
            nn.GroupNorm(8, w),
            nn.SiLU(),
            nn.Conv2d(w, w, 3, 1, 1),
            nn.GroupNorm(8, w),
            nn.SiLU(),
            nn.Conv2d(w, 4, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + float(np.log(200.0))


def _load_state_dict(path: str | os.PathLike) -> dict[str, torch.Tensor]:
    """A flat state dict from ``.safetensors`` or a torch ``.pt`` file (tensors only)."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


class PictureDepth:
    """The frozen depth head and read-out: :meth:`depth_grid` maps latents to log axial depth.

    Args:
        head (DepthLat): the depth head.
        readout (Readout): the read-out.
        device (torch.device | str): where both run.
    """

    def __init__(
        self, head: DepthLat, readout: Readout, device: torch.device | str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.head = head.to(self.device).eval().requires_grad_(False)
        self.readout = readout.to(self.device).eval().requires_grad_(False)

    @classmethod
    def load(
        cls,
        head_ckpt: str | os.PathLike,
        readout_ckpt: str | os.PathLike,
        device: torch.device | str = "cpu",
    ) -> "PictureDepth":
        """Load the head and read-out state dicts (strict, float32), sized from the state dicts.

        The modules are built on ``meta``, so loading never advances the global RNG.
        """
        sd = _load_state_dict(head_ckpt)
        nb = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        ndb = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("dblocks."))
        with torch.device("meta"):
            head = DepthLat(w=int(sd["inp.weight"].shape[0]), nb=nb, ndb=ndb)
        head = head.to_empty(device="cpu")
        head.load_state_dict(sd)
        rsd = _load_state_dict(readout_ckpt)
        with torch.device("meta"):
            readout = Readout(int(rsd["net.0.weight"].shape[0]))
        readout = readout.to_empty(device="cpu")
        readout.load_state_dict(rsd)
        return cls(head, readout, device)

    @torch.no_grad()
    def depth_grid(self, latents: torch.Tensor | np.ndarray) -> np.ndarray:
        """Log axial depth of RGB latents, in float32 on :attr:`device`.

        Args:
            latents (Tensor | ndarray): ``[T, 48, 24, 42]`` latents, any float dtype.

        Returns:
            ndarray: ``[T, 4, 24, 42]`` float32 on the CPU.
        """
        x = latents if torch.is_tensor(latents) else torch.from_numpy(np.asarray(latents))
        x = x.detach().to(self.device, torch.float32)
        if x.ndim != 4 or x.shape[1] != LATENT_CHANNELS or tuple(x.shape[-2:]) != GRID:
            raise ValueError(
                f"latents must be [T, {LATENT_CHANNELS}, {GRID[0]}, {GRID[1]}], got"
                f" {tuple(x.shape)}"
            )
        features = self.head(x[None])[0]
        return self.readout(features.float()).float().cpu().numpy()
