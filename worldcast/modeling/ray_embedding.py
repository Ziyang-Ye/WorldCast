"""The ray embedding (Sec. 3.3, Eq. (3); App. "Ray embedding"): Plücker coordinates of every token.

``r_{f,u} = (d_{f,u}, (o_f / l) x d_{f,u})`` relative to an anchor camera, with ``d`` the unit
viewing direction of token ``u`` in latent frame ``f`` (from the frame's own field of view), ``o_f``
the camera centre and the length scale ``l = 420`` u; ``h_{f,u} += phi(r_{f,u})`` before the first
DiT block. The anchor is the first target frame's camera in a gathered window, with or without
memory frames (``worldcast.sampling.window.gather_window``), and the camera of latent frame 0 in a
contiguous window (the first six blocks of a rollout, a training window without memory frames).
Camera space is x right, y down, z forward; ``c2w`` is camera-to-world in u.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

__all__ = [
    "LENGTH_SCALE_U",
    "RAY_CONDITION_KEYS",
    "RayConditions",
    "RayEmbedding",
    "plucker_rays",
    "relative_c2w",
    "se3_inverse",
]

#: The length scale ``l`` of Eq. (3), u.
LENGTH_SCALE_U = 420.0
#: Condition-dict keys of the cameras: ``ray_`` and the field names of :class:`RayConditions`.
RAY_CONDITION_KEYS = ("ray_frame_c2w", "ray_frame_tans", "ray_anchor_c2w")


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


def relative_c2w(c2w: torch.Tensor, anchor_c2w: torch.Tensor) -> torch.Tensor:
    """Cameras ``[K, 4, 4]`` in the frame of the anchor camera ``[4, 4]`` (u)."""
    return se3_inverse(anchor_c2w)[None] @ c2w


@dataclass(eq=False)
class RayConditions:
    """The cameras of one window; in a condition dict, the entries :data:`RAY_CONDITION_KEYS`.

    Attributes:
        frame_c2w (Tensor): ``[B, F_window, 4, 4]`` camera of every latent frame, in window order
            (a memory frame: the camera its memory entry was generated from).
        frame_tans (Tensor): ``[B, F_window, 2]`` ``(tan(hfov/2), tan(vfov/2))`` of each frame.
        anchor_c2w (Tensor): ``[B, 4, 4]`` the anchor: the camera of the first target frame in a
            gathered window, of latent frame 0 in a contiguous one.
    """

    frame_c2w: torch.Tensor
    frame_tans: torch.Tensor
    anchor_c2w: torch.Tensor

    def __post_init__(self) -> None:
        c2w, tans, anchor = self.frame_c2w, self.frame_tans, self.anchor_c2w
        if c2w.ndim != 4 or tuple(c2w.shape[-2:]) != (4, 4):
            raise ValueError(f"frame_c2w must be [B, F, 4, 4], got {tuple(c2w.shape)}")
        if tuple(tans.shape) != (*c2w.shape[:2], 2):
            raise ValueError(
                f"frame_tans must be [B, F, 2] of the cameras, got {tuple(tans.shape)}"
            )
        if tuple(anchor.shape) != (c2w.shape[0], 4, 4):
            raise ValueError(f"anchor_c2w must be [B, 4, 4], got {tuple(anchor.shape)}")

    @classmethod
    def contiguous(cls, c2w: torch.Tensor, tans: torch.Tensor) -> "RayConditions":
        """The cameras of a contiguous window (the first six blocks of a rollout, a training
        window without memory frames): the anchor is the camera of latent frame 0.

        Args:
            c2w (Tensor): ``[B, F, 4, 4]`` camera of every latent frame of the window.
            tans (Tensor): ``[B, F, 2]`` ``(tan(hfov/2), tan(vfov/2))`` of each frame.
        """
        if c2w.ndim != 4:
            raise ValueError(f"c2w must be [B, F, 4, 4], got {tuple(c2w.shape)}")
        return cls(frame_c2w=c2w, frame_tans=tans, anchor_c2w=c2w[:, 0])

    @classmethod
    def from_conditions(cls, conditions: Mapping[str, Any]) -> "RayConditions | None":
        """The cameras of a condition dict; ``None`` when it has none of the ray entries (they
        come all together or not at all)."""
        frame_c2w, frame_tans, anchor_c2w = (conditions.get(key) for key in RAY_CONDITION_KEYS)
        given = [value is not None for value in (frame_c2w, frame_tans, anchor_c2w)]
        if not any(given):
            return None
        if not all(given):
            raise ValueError(f"the cameras arrive together or not at all: {RAY_CONDITION_KEYS}")
        return cls(frame_c2w=frame_c2w, frame_tans=frame_tans, anchor_c2w=anchor_c2w)

    def conditions(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        """The entries :data:`RAY_CONDITION_KEYS` of a condition dict, float32 on ``device``."""
        cameras = (self.frame_c2w, self.frame_tans, self.anchor_c2w)
        return {
            key: value.to(device=device, dtype=torch.float32, non_blocking=True)
            for key, value in zip(RAY_CONDITION_KEYS, cameras)
        }


def plucker_rays(
    frame_c2w: torch.Tensor,
    frame_tans: torch.Tensor,
    anchor_c2w: torch.Tensor,
    grid: tuple[int, int],
) -> torch.Tensor:
    """The Plücker coordinates ``r = (d, (o / l) x d)`` of every token of ``n`` frames (Eq. (3)).

    Args:
        frame_c2w (Tensor): ``[B, n, 4, 4]`` camera of each frame.
        frame_tans (Tensor): ``[B, n, 2]`` ``(tan(hfov/2), tan(vfov/2))`` of each frame.
        anchor_c2w (Tensor): ``[B, 4, 4]`` the anchor camera (:class:`RayConditions`).
        grid (tuple[int, int]): token grid ``(h, w)``.

    Returns:
        Tensor: ``[B, n, h w, 6]``, the tokens of a frame in row-major order; float32, and bf16
        inside the generator's CUDA autocast, where the pose matmuls are bf16, as trained.
    """
    grid_h, grid_w = grid
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
    # one sample at a time, as trained; a batched product of the poses may round differently
    relative = torch.stack(
        [
            relative_c2w(frame_c2w[b].float(), anchor_c2w[b].float())
            for b in range(frame_c2w.shape[0])
        ]
    )  # [B, n, 4, 4], in the anchor's frame
    d = directions.flatten(2, 3) @ relative[..., :3, :3].transpose(-1, -2)  # [B, n, hw, 3]
    o = relative[..., :3, 3, None].transpose(-1, -2).expand_as(d)
    return torch.cat([d, torch.linalg.cross(o / LENGTH_SCALE_U, d)], dim=-1)


class RayEmbedding(nn.Module):
    """The MLP ``phi`` of Eq. (3) on :func:`plucker_rays`: ``Linear(6, dim) -> SiLU -> Linear(dim,
    dim)``, the last layer zero-initialised.

    Args:
        dim (int): model width.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        rays: RayConditions,
        *,
        frame_offset: int,
        num_frames: int,
        grid: tuple[int, int],
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """The ray embedding of a call's frames ``[frame_offset, frame_offset + num_frames)``.

        Args:
            rays (RayConditions): the window's cameras.
            frame_offset (int): window index of the call's first latent frame.
            num_frames (int): latent frames of the call.
            grid (tuple[int, int]): token grid ``(h, w)``.
            dtype (torch.dtype | None): the dtype the coordinates are cast to before ``phi``
                (the generator passes its token dtype; ``None``: as computed).

        Returns:
            Tensor: ``[B, num_frames h w, dim]``, frame-major.
        """
        if rays.frame_c2w.shape[1] < frame_offset + num_frames:
            raise ValueError(
                f"the cameras cover {rays.frame_c2w.shape[1]} latent frames of the window, the call"
                f" reads up to frame {frame_offset + num_frames - 1}"
            )
        device = self.mlp[0].weight.device
        frames = slice(frame_offset, frame_offset + num_frames)
        coordinates = plucker_rays(
            rays.frame_c2w.to(device=device)[:, frames],
            rays.frame_tans[:, frames],
            rays.anchor_c2w.to(device=device),
            grid,
        )
        coordinates = coordinates.flatten(1, 2)
        return self.mlp(coordinates if dtype is None else coordinates.to(dtype))
