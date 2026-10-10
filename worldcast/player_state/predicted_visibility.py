"""Predicted visibility labels (Sec. 3.4, Depth and visibility; App. "The player state field in
detail", Visibility).

With predicted states a client computes the visibility labels of the other players itself, from the
depth of its own generated latent frames (the depth head). A player's test point is its feet + 40
u. It is visible if the point lies in front of the camera and in the frustum, and the median depth
of the 2 x 2 depth cells at its projection is at most 24 u in front of it; eight more points on a
circle of 40 u around it are tested too (visible if any is). A video frame of a generated latent
frame is tested against its own depth; the video frames of the block about to be generated against
the recent context (the last 12 generated latent frames), back-projected to 3D and reprojected into
the video frame's camera (a player on cells no point reaches is unknown). After the first denoising
step the block's video frames are re-tested against the depth of its x0 estimate.

The reprojection here and the scene state's (:mod:`worldcast.scene_state.geometry`) are two
arithmetics on one depth grid, each as in the paper's runs; they share the near plane and the ray
grid.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from worldcast.data.camera import NEAR_U, camera_rays, camera_to_world, half_angle_tangents
from worldcast.data.latents import (
    BLOCK,
    LATENT_GRID,
    PIXELS_PER_CELL,
    RECENT,
    VIDEO_FRAMES_PER_BLOCK,
    VIDEO_FRAMES_PER_LATENT,
    video_frames_of,
)
from worldcast.data.recordings import ALIVE_INDEX
from worldcast.modeling.depth_head import DepthFn

from .states import integrate_camera_angles, pack_substeps
from .visibility import fold_video_frames

__all__ = [
    "MARGIN_U",
    "PLAYER_LIFT_U",
    "POSE_RADIUS_U",
    "PredictedVisibility",
    "block_depth_frames",
    "ztest_labels",
]

#: The margin of the visibility test: a player may lie this far behind the surface of its cells
#: and still be visible, u (Sec. 3.4).
MARGIN_U = 24.0
#: A player's test point above its feet, u.
PLAYER_LIFT_U = 40.0
#: Radius of the circle of eight more test points around a player, u.
POSE_RADIUS_U = 40.0


def _depth_frames(log_depth: torch.Tensor, first_latent: int) -> dict[int, torch.Tensor]:
    """``{video frame: axial depth [24, 42]}`` of the latent frames ``first_latent ..``, from their
    log depth ``[n, 4, 24, 42]`` (channel ``c``: the latent frame's ``c``-th video frame)."""
    out = {}
    for i in range(int(log_depth.shape[0])):
        for c, v in enumerate(video_frames_of(int(first_latent) + i)):
            # exp in float32, as in the paper's runs; the re-test of a block's x0 reads the depth
            # in float64
            out[v] = log_depth[i, c].exp()
    return out


# The three functions below are the reprojection of the predicted labels, as in the paper's runs:
# torch float64 on the CPU; a point is rotated by ``R`` itself, ``(p - t) R`` (the scene state uses
# ``inv(R).T``); every finite positive depth is back-projected (the scene state drops the far
# plane); a point lands in a cell for ``0 <= col < w`` and ``0 <= row < h`` (the scene state: ``|u|,
# |v| <= 1``), and a cell without a point is NaN. The labels are thresholded at cell borders and at
# the margin, so they follow this arithmetic and not the scene state's.


def _project(
    points: torch.Tensor, c2w: torch.Tensor, tans: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World points ``[R, M, 3]`` (or shared ``[M, 3]``) in cameras ``[R, 4, 4]``: column and row
    on the depth grid (fractional) and axial depth, each ``[R, M]`` float64."""
    tan_h, tan_v = float(tans[0]), float(tans[1])
    h, w = LATENT_GRID
    rot, t = c2w[:, :3, :3].double(), c2w[:, :3, 3].double()
    p = points.double()
    if p.ndim == 2:
        p = p.unsqueeze(0).expand(int(c2w.shape[0]), -1, -1)
    cam = torch.einsum("rmk,rkj->rmj", p - t[:, None, :], rot)
    z = cam[..., 2]
    zs = torch.where(z.abs() > 1e-9, z, torch.full_like(z, 1e-9))
    return (
        (cam[..., 0] / (zs * tan_h) + 1.0) * (w / 2.0),
        (cam[..., 1] / (zs * tan_v) + 1.0) * (h / 2.0),
        z,
    )


def _back_project(depth: torch.Tensor, c2w: torch.Tensor, tans: Sequence[float]) -> torch.Tensor:
    """Axial depth ``[K, H, W]`` seen from ``c2w [K, 4, 4]`` -> world points ``[M, 3]`` float64 of
    the cells with a finite positive depth."""
    rays = torch.from_numpy(camera_rays(tans, LATENT_GRID)).view(*LATENT_GRID, 3)
    d = depth.double()
    world = (
        torch.einsum("khwj,kij->khwi", rays[None] * d[..., None], c2w[:, :3, :3].double())
        + c2w[:, None, None, :3, 3].double()
    )
    return world[torch.isfinite(d) & (d > 0)]


def _zbuffer(points: torch.Tensor, c2w: torch.Tensor, tans: Sequence[float]) -> torch.Tensor:
    """Nearest point per cell of each camera: ``[R, H, W]`` axial depth, NaN where none lands."""
    h, w = LATENT_GRID
    n = int(c2w.shape[0])
    if int(points.shape[0]) == 0:
        return torch.full((n, h, w), float("nan"), dtype=torch.float64, device=points.device)
    out = torch.full((n, h * w), float("inf"), dtype=torch.float64, device=points.device)
    col, row, z = _project(points, c2w, tans)
    ok = (z > NEAR_U) & (col >= 0) & (col < w) & (row >= 0) & (row < h)
    idx = row.clamp(0, h - 1).floor().long() * w + col.clamp(0, w - 1).floor().long()
    out.scatter_reduce_(
        1,
        idx,
        torch.where(ok, z, torch.full_like(z, float("inf"))),
        reduce="amin",
        include_self=True,
    )
    out[~torch.isfinite(out)] = float("nan")
    return out.view(n, h, w)


def _window_median(
    depth: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Median of the 2 x 2 cells ``(cy-1..cy) x (cx-1..cx)`` (clamped at 0) that hold a depth, and
    their number ``n``, each ``[R, P]``; NaN where ``n`` is 0."""
    r, p = cx.shape
    rows = torch.stack([(cy - 1).clamp(min=0), cy], -1)
    cols = torch.stack([(cx - 1).clamp(min=0), cx], -1)
    w = int(depth.shape[-1])
    flat = (rows[..., :, None] * w + cols[..., None, :]).reshape(r, p * 4)
    vals = depth.reshape(r, -1).gather(1, flat).view(r, p, 4)
    valid = ~torch.isnan(vals)
    n = valid.sum(-1)
    # the n known depths sort first (the unknown ones as inf); their median is the mean of the two
    # middle ones, at (n - 1) // 2 and n // 2
    s = torch.where(valid, vals, torch.full_like(vals, float("inf"))).sort(-1).values
    lo, hi = (n - 1).clamp(min=0) // 2, (n // 2).clamp(max=3)
    med = 0.5 * (s.gather(-1, lo[..., None]) + s.gather(-1, hi[..., None])).squeeze(-1)
    return torch.where(n > 0, med, torch.full_like(med, float("nan"))), n


def ztest_labels(
    points: torch.Tensor,
    c2w: torch.Tensor,
    depth: torch.Tensor,
    *,
    live: torch.Tensor,
    tans: Sequence[float],
    pose_radius: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-test players against a depth map.

    Args:
        points (torch.Tensor): ``[R, P, 3]`` test points, u.
        c2w (torch.Tensor): ``[R, 4, 4]`` cameras.
        depth (torch.Tensor): ``[R, H, W]`` axial depth, u (NaN: no surface known).
        live (torch.Tensor): ``[R, P]`` bool, alive, present and not the client.
        tans (Sequence[float]): ``(tan_h, tan_v)``.
        pose_radius (float): radius of the eight extra test points, u (0: none).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(visible, valid)``, each ``[R, P]`` bool.
    """
    if pose_radius > 0:
        visible, valid = ztest_labels(points, c2w, depth, live=live, tans=tans)
        for k in range(8):
            a = k * math.pi / 4.0
            offset = points.new_tensor([pose_radius * math.cos(a), pose_radius * math.sin(a), 0.0])
            visible = visible | ztest_labels(points + offset, c2w, depth, live=live, tans=tans)[0]
        return visible, valid | visible
    h, w = LATENT_GRID
    cell = PIXELS_PER_CELL
    col, row, z = _project(points, c2w, tans)
    # the frustum test is on the pixel grid: the point must lie half a pixel inside the frame
    fc, fr = col * cell, row * cell
    in_view = (
        (z > NEAR_U) & (fc >= 0.5) & (fc <= w * cell - 0.5) & (fr >= 0.5) & (fr <= h * cell - 0.5)
    )
    cx = fc.floor().clamp(0, w * cell - 1).long() // cell
    cy = fr.floor().clamp(0, h * cell - 1).long() // cell
    med, n = _window_median(depth.double(), cx, cy)
    live = live.bool()
    hole = in_view & (n == 0) & live
    visible = live & in_view & (n > 0) & (med - z > -MARGIN_U)
    return visible, live & ~hole


def block_depth_frames(depth_fn: DepthFn, x0: torch.Tensor) -> np.ndarray:
    """Axial depth ``[16, 24, 42]`` float64 of a block's 16 video frames, from its latent frames.

    Entry ``4l + c`` is video frame ``c`` of latent frame ``l``; the four latent frames
    ``[4, 48, 24, 42]`` are read in one depth call.
    """
    grid = np.asarray(depth_fn(x0), dtype=np.float64)
    out = np.empty((VIDEO_FRAMES_PER_BLOCK, *LATENT_GRID), dtype=np.float64)
    for c in range(VIDEO_FRAMES_PER_LATENT):
        out[c::VIDEO_FRAMES_PER_LATENT] = np.exp(grid[:, c])
    return out


class PredictedVisibility:
    """The visibility labels a client computes for itself, per video frame of the round.

    ``cameras`` ``[T, 4, 4]`` (the client's table position + eye height, angles from its own view
    controls) and ``points`` ``[T, P, 3]`` (every player's table position + 40 u) follow the
    player-state table: call :meth:`update_frames` after the table changes.

    Args:
        batch (Mapping[str, torch.Tensor]): the whole round of one client, batch size 1:
            ``player_states`` ``[1, P, T, 6]``, ``player_control_substeps``,
            ``player_control_substep_valid`` and ``client_slot`` ``[1]``.
        depth_fn (DepthFn): the depth head.
    """

    def __init__(self, batch: Mapping[str, torch.Tensor], *, depth_fn: DepthFn) -> None:
        states = batch["player_states"]
        if int(states.shape[0]) != 1:
            raise ValueError(f"one client's round (batch size 1), got {int(states.shape[0])}")
        self.client_slot = int(torch.as_tensor(batch["client_slot"]).reshape(-1)[0])
        packed = pack_substeps(
            batch["player_control_substeps"], batch["player_control_substep_valid"]
        )
        yaw, pitch = integrate_camera_angles(states[:, :, 0].float(), packed)
        self._yaw = yaw[0, self.client_slot].double()
        self._pitch = pitch[0, self.client_slot].double()
        self.states = states
        self.depth_fn = depth_fn
        self.tans = half_angle_tangents()
        live = (states[0, :, :, ALIVE_INDEX] > 0.5).permute(1, 0).clone()
        live[:, self.client_slot] = False
        self.live = live
        video_frames, players = int(states.shape[2]), int(states.shape[1])
        self.cameras = torch.zeros(video_frames, 4, 4, dtype=torch.float64)
        self.points = torch.zeros(video_frames, players, 3, dtype=torch.float64)
        self.update_frames(np.arange(video_frames))

    def update_frames(self, video_frames: Sequence[int] | np.ndarray) -> None:
        """Re-read the client's cameras and the test points at ``video_frames`` from the table."""
        v = torch.as_tensor(np.asarray(video_frames, np.int64))
        st = self.states[0].float()
        self.cameras[v] = camera_to_world(
            st[self.client_slot, v, :3].double(), self._yaw[v], self._pitch[v]
        )
        pts = st[:, v, :3].permute(1, 0, 2).double().clone()
        pts[..., 2] += PLAYER_LIFT_U
        self.points[v] = pts

    def _depth(self, latents: torch.Tensor, first_latent: int) -> dict[int, torch.Tensor]:
        log_depth = torch.as_tensor(self.depth_fn(latents.detach().float())).detach().float().cpu()
        return _depth_frames(log_depth, first_latent)

    def generated_labels(
        self, video_frames: Sequence[int], depth: Mapping[int, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Video frames of generated latent frames, each against its own depth: ``(visible,
        valid)`` ``[R, P]``."""
        idx = torch.as_tensor(list(video_frames), dtype=torch.long)
        return ztest_labels(
            self.points[idx],
            self.cameras[idx],
            torch.stack([depth[v] for v in video_frames]).double(),
            live=self.live[idx],
            tans=self.tans,
            pose_radius=POSE_RADIUS_U,
        )

    def target_labels(
        self,
        video_frames: Sequence[int],
        depth: Mapping[int, torch.Tensor],
        context_frames: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Video frames of the block about to be generated, against the generated
        ``context_frames``, whose surface is reprojected into their cameras: ``(visible, valid)``
        ``[R, P]``."""
        return self._reprojected_labels(video_frames, self._context_points(depth, context_frames))

    def _context_points(
        self, depth: Mapping[int, torch.Tensor], context_frames: Sequence[int]
    ) -> torch.Tensor:
        """The surface of the generated ``context_frames`` as world points ``[M, 3]`` float64
        (:func:`_back_project`)."""
        if not context_frames:
            return torch.zeros(0, 3, dtype=torch.float64)
        cidx = torch.as_tensor(list(context_frames), dtype=torch.long)
        return _back_project(
            torch.stack([depth[v] for v in context_frames]), self.cameras[cidx], self.tans
        )

    def _reprojected_labels(
        self, video_frames: Sequence[int], context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(visible, valid)`` ``[R, P]`` of ``video_frames`` against the ``context`` points
        ``[M, 3]`` reprojected into their cameras, on the points' device."""
        idx = torch.as_tensor(list(video_frames), dtype=torch.long)
        cameras = self.cameras[idx]
        return ztest_labels(
            self.points[idx],
            cameras,
            _zbuffer(context, cameras.to(context.device), self.tans).to(cameras.device),
            live=self.live[idx],
            tans=self.tans,
            pose_radius=POSE_RADIUS_U,
        )

    # -- gathered windows: labels written into the round batch before the window gather
    def for_block(self, s: int, batch: Mapping[str, torch.Tensor], latents: torch.Tensor) -> dict:
        """The round batch with the labels the window of block ``s`` gathers.

        Args:
            s (int): the block's first latent frame.
            batch (Mapping[str, torch.Tensor]): the round batch.
            latents (torch.Tensor): ``[N, 48, 24, 42]`` the client's latent frames; ``latents[:s]``
                are read.

        Returns:
            dict: ``batch`` with ``client_visibility`` / ``client_visibility_valid`` predicted on
            the video frames of the first frame, the recent context and the target block, and
            unknown elsewhere.
        """
        s = int(s)
        template = batch["client_visibility"]
        video_frames = int(template.shape[-1])
        visible = torch.zeros(self.live.shape[1], video_frames, dtype=torch.bool)
        valid = torch.zeros_like(visible)

        recent_start = max(1, s - RECENT)
        depth_start = max(0, min(recent_start - 1, s - RECENT))
        depth = self._depth(latents[depth_start:s], depth_start)
        recent = video_frames_of(recent_start, s - recent_start)
        v, ok = self.generated_labels(recent, depth)
        visible[:, recent], valid[:, recent] = v.T, ok.T

        context_start = max(0, s - RECENT)
        context = video_frames_of(context_start, s - context_start)
        target = [f for f in video_frames_of(s, BLOCK) if f < video_frames]
        v, ok = self.target_labels(target, depth, context)
        visible[:, target], valid[:, target] = v.T, ok.T

        # the first frame (latent frame 0), against its own depth read with latent frame 1
        first_frame = self._depth(latents[0 : min(2, s)], 0)[0]
        v, ok = ztest_labels(
            self.points[:1],
            self.cameras[:1],
            first_frame[None].double(),
            live=self.live[:1],
            tans=self.tans,
            pose_radius=POSE_RADIUS_U,
        )
        visible[:, 0], valid[:, 0] = v[0], ok[0]
        out = dict(batch)
        out["client_visibility"] = visible[None].to(template.dtype)
        out["client_visibility_valid"] = valid[None].to(batch["client_visibility_valid"].dtype)
        return out

    def relabel_target(
        self, window: Mapping[str, torch.Tensor], s: int, depth_frames: np.ndarray
    ) -> dict:
        """The gathered window with its last 16 video frames (block ``s``) re-tested against
        ``depth_frames`` ``[16, 24, 42]`` (:func:`block_depth_frames` of the block's x0)."""
        frames = video_frames_of(s, BLOCK)
        visible, valid = self._retest(frames, depth_frames)
        labels, known = window["client_visibility"], window["client_visibility_valid"]
        last = int(labels.shape[-1])
        where = torch.arange(last - len(frames), last, dtype=torch.long)
        new_labels, new_known = labels.clone(), known.clone()
        new_labels[0, :, where] = visible.T.to(labels.dtype)
        new_known[0, :, where] = valid.T.to(known.dtype)
        return dict(window, client_visibility=new_labels, client_visibility_valid=new_known)

    def _retest(
        self, video_frames: Sequence[int], depth_frames: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = torch.as_tensor(np.asarray(depth_frames, dtype=np.float64))
        if bool(torch.isnan(d).any()) or bool((d <= 0).any()):
            raise ValueError("the x0 depth has NaN or non-positive cells")
        return self.generated_labels(video_frames, {v: d[i] for i, v in enumerate(video_frames)})

    # -- the first six blocks: ``player_visible`` per latent frame of the round
    def target_visible(self, start: int, latents: torch.Tensor) -> torch.Tensor:
        """``player_visible`` ``[4, P]`` of the block about to be denoised at latent frame
        ``start``, against the last 12 generated latent frames of ``latents`` ``[N, 48, 24, 42]``.
        """
        start = int(start)
        visible, _ = self.target_labels(*self._target_context(start, latents))
        return fold_video_frames(visible, start, BLOCK)

    def _target_context(
        self, start: int, latents: torch.Tensor
    ) -> tuple[list[int], dict[int, torch.Tensor], list[int]]:
        """The video frames of the block at latent frame ``start``, the depth of its context (the
        last 12 generated latent frames of ``latents``) and the context's video frames."""
        first = max(0, start - RECENT)
        depth = self._depth(latents[first:start], first)
        return video_frames_of(start, BLOCK), depth, video_frames_of(first, start - first)

    def generated_visible(self, start: int, n: int, latents: torch.Tensor) -> torch.Tensor:
        """``player_visible`` ``[n, P]`` of the generated latent frames ``start .. start + n - 1``
        of ``latents`` ``[N, 48, 24, 42]``, each against its own depth."""
        start, n = int(start), int(n)
        first = max(0, start - 1)
        depth = self._depth(latents[first : start + n], first)
        visible, _ = self.generated_labels(video_frames_of(start, n), depth)
        return fold_video_frames(visible, start, n)

    def retested_visible(self, start: int, depth_frames: np.ndarray) -> torch.Tensor:
        """``player_visible`` ``[4, P]`` of the block at latent frame ``start``, re-tested against
        the depth ``[16, 24, 42]`` of its x0 estimate."""
        visible, _ = self._retest(video_frames_of(start, BLOCK), depth_frames)
        return fold_video_frames(visible, int(start), BLOCK)
