"""Plücker ray code of every latent frame, relative to the first target frame (Sec. 3.3, Eq. (3)).

``r_{f,u} = (d_{f,u}, (o_f / l) x d_{f,u})`` with ``d`` the unit direction of token ``u`` in frame
``f`` (from the frame's own field of view), ``o_f`` the camera centre and ``l = 420`` Source units;
``h_{f,u} += phi(r_{f,u})`` before the first DiT block. Camera space is x right, y down, z forward;
``c2w`` is camera-to-world in Source units.
"""

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["RayConditions", "RayEmbedding", "relative_c2w", "se3_inverse"]


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Batched rigid inverse of ``[..., 4, 4]`` transforms."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = R_inv
    out[..., :3, 3:] = -R_inv @ t
    out[..., 3, 3] = 1.0
    return out


def relative_c2w(memory_c2w: torch.Tensor, query_c2w: torch.Tensor) -> torch.Tensor:
    """Cameras ``[K, 4, 4]`` in the frame of ``query_c2w`` ``[4, 4]`` (Source units)."""
    return se3_inverse(query_c2w)[None] @ memory_c2w


@dataclass
class RayConditions:
    """Cameras of one window (condition keys ``state_wp_*``).

    Attributes:
        frame_c2w (Tensor): ``[B, F_window, 4, 4]`` camera of every latent frame, in window order.
        frame_tans (Tensor): ``[B, F_window, 2]`` ``(tan(hfov/2), tan(vfov/2))`` of each frame.
        anchor_c2w (Tensor): ``[B, 4, 4]`` camera of the first target frame.
        memory_c2w (Tensor): ``[B, K, 4, 4]`` cameras of the memory slot (``K`` is 4 or 0); they
            are ``frame_c2w`` at ``memory_frames``.
        memory_frames (Tensor): ``[B, K]`` long window positions of the memory slot.
    """

    frame_c2w: torch.Tensor
    frame_tans: torch.Tensor
    anchor_c2w: torch.Tensor
    memory_c2w: torch.Tensor
    memory_frames: torch.Tensor


class RayEmbedding(nn.Module):
    """``phi`` of Eq. (3): ``Linear(6, dim) -> SiLU -> Linear(dim, dim)``, zero-initialised."""

    def __init__(self, dim: int, *, ray_unit_u: float = 420.0) -> None:
        super().__init__()
        self.ray_unit_u = float(ray_unit_u)
        self.ray_mlp = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.ray_mlp[2].weight)
        nn.init.zeros_(self.ray_mlp[2].bias)

    def ray_code(
        self,
        frame_c2w: torch.Tensor,
        anchor_c2w: torch.Tensor,
        frame_tans: torch.Tensor,
        grid_h: int,
        grid_w: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """``phi(r)`` of ``n`` frames, in frame-major, row-major token order.

        Runs inside the generator's bf16 autocast: on CUDA the pose matmuls are bf16, as trained.

        Args:
            frame_c2w (Tensor): ``[B, n, 4, 4]`` camera of each frame.
            anchor_c2w (Tensor): ``[B, 4, 4]`` camera of the first target frame.
            frame_tans (Tensor): ``[B, n, 2]`` ``(tan(hfov/2), tan(vfov/2))`` of each frame.
            grid_h (int): token rows.
            grid_w (int): token columns.
            dtype (torch.dtype): dtype of the rays fed to ``phi`` (the tokens').

        Returns:
            Tensor: ``[B, n grid_h grid_w, dim]``.
        """
        device = frame_c2w.device
        tans = frame_tans.to(device=device, dtype=torch.float32)
        ys, xs = torch.meshgrid(
            torch.arange(grid_h, device=device).float() + 0.5,
            torch.arange(grid_w, device=device).float() + 0.5,
            indexing="ij",
        )
        x = ((xs - grid_w / 2) / (grid_w / 2))[None, None] * tans[..., 0, None, None]
        y = ((ys - grid_h / 2) / (grid_h / 2))[None, None] * tans[..., 1, None, None]
        directions = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        directions = directions / directions.norm(dim=-1, keepdim=True)  # [B, n, h, w, 3]
        relative = torch.stack(
            [
                relative_c2w(frame_c2w[b].float(), anchor_c2w[b].float())
                for b in range(frame_c2w.shape[0])
            ]
        )  # [B, n, 4, 4], in the anchor's frame
        d = directions.flatten(2, 3) @ relative[..., :3, :3].transpose(-1, -2)  # [B, n, hw, 3]
        o = relative[..., :3, 3, None].transpose(-1, -2).expand_as(d)
        rays = torch.cat([d, torch.linalg.cross(o / self.ray_unit_u, d)], dim=-1)
        return self.ray_mlp(rays.flatten(1, 2).to(dtype=dtype))

    def forward(
        self,
        rays: RayConditions,
        *,
        frame_offset: int,
        num_frames: int,
        grid: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """The ray code of a call's frames ``[frame_offset, frame_offset + num_frames)``.

        Args:
            rays (RayConditions): the window's cameras; the memory slot is checked against
                ``frame_c2w`` and must sit at the same positions for the whole batch.
            frame_offset (int): window index of the call's first frame.
            num_frames (int): latent frames in the call.
            grid (tuple[int, int]): token grid ``(h, w)``.
            dtype (torch.dtype): dtype of the rays fed to ``phi`` (the tokens').

        Returns:
            Tensor: ``[B, num_frames h w, dim]``.
        """
        device = self.ray_mlp[0].weight.device
        frame_c2w = rays.frame_c2w.to(device=device)
        memory_frames = rays.memory_frames
        if not torch.equal(memory_frames, memory_frames[:1].expand_as(memory_frames)):
            raise ValueError("memory_frames must be the same for the whole batch")
        slots = memory_frames[0].to(device=device, dtype=torch.long)
        if not torch.equal(frame_c2w[:, slots].float(), rays.memory_c2w.to(device=device).float()):
            raise ValueError("frame_c2w at the memory slot differs from memory_c2w")
        frames = slice(frame_offset, frame_offset + num_frames)
        return self.ray_code(
            frame_c2w[:, frames],
            rays.anchor_c2w.to(device=device),
            rays.frame_tans[:, frames],
            *grid,
            dtype=dtype,
        )
