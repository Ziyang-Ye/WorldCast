"""Predicted visibility (Sec. 3.2, App. "The player state field in detail"): the field's visibility
labels from the picture depth head instead of the recording.

A player's test point is its feet + 40 u. It is visible if it lies in front of the camera and in the
frustum, and the median depth of the 2 x 2 depth cells at its projection is at most 24 u in front of
it; with a pose radius, eight more points on a circle of that radius are tested too (visible if any
is). A pixel row whose frame is drawn is tested against its own depth; the rows of the block about
to be drawn against the last 12 drawn latent frames, lifted to 3D and splatted into the row's camera
(a player on cells nothing reaches is unknown). After the first ladder rung the block's rows are
re-tested against the depth of its x0 estimate.
"""

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

from .projection import EYE_HEIGHT, camera_to_world, half_angle_tangents
from .tables import integrate_camera_angles, pack_substeps

__all__ = ["GRID", "DepthFn", "PredictedVisibility", "ztest_labels", "block_depth_rows"]

#: The depth read-out's grid (16 x 16-pixel cells of a 384 x 672 frame); the frustum test runs on
#: the pixel grid (``FINE`` pixels per cell).
GRID = (24, 42)
FINE = 16
#: Test point above the feet, occlusion margin and near plane, u.
PEER_LIFT_U = 40.0
MARGIN_U = 24.0
NEAR_U = 1.0
#: Latent frames per block.
BLOCK = 4
#: Drawn latent frames the block about to be generated is z-tested against.
WARP_LATENTS = 12

#: Latent frames ``[T, 48, 24, 42]`` -> log axial depth ``[T, 4, 24, 42]`` (the picture depth head).
DepthFn = Callable[[torch.Tensor], np.ndarray]


def _pixel_rows(latent: int) -> list[int]:
    f = int(latent)
    return [0] if f == 0 else list(range(4 * f - 3, 4 * f + 1))


def _depth_rows(log_depth: torch.Tensor, first_latent: int) -> dict[int, torch.Tensor]:
    out = {}
    for i in range(int(log_depth.shape[0])):
        for c, r in enumerate(_pixel_rows(int(first_latent) + i)):
            out[r] = log_depth[i, c].exp()
    return out


def _project(points: torch.Tensor, c2w: torch.Tensor, tans):
    """World points ``[R, M, 3]`` (or shared ``[M, 3]``) in cameras ``[R, 4, 4]``: col, row, z."""
    tan_h, tan_v = float(tans[0]), float(tans[1])
    h, w = GRID
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


def _surface_points(depth: torch.Tensor, c2w: torch.Tensor, tans) -> torch.Tensor:
    """Axial depth ``[K, H, W]`` seen from ``c2w [K, 4, 4]`` -> world points ``[M, 3]``."""
    tan_h, tan_v = float(tans[0]), float(tans[1])
    h, w = GRID
    u = ((torch.arange(w, dtype=torch.float64) + 0.5) / w * 2 - 1) * tan_h
    v = ((torch.arange(h, dtype=torch.float64) + 0.5) / h * 2 - 1) * tan_v
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    d = depth.double()
    cam = torch.stack([uu[None] * d, vv[None] * d, d], dim=-1)
    world = (
        torch.einsum("khwj,kij->khwi", cam, c2w[:, :3, :3].double())
        + c2w[:, None, None, :3, 3].double()
    )
    return world[torch.isfinite(d) & (d > 0)]


def _splat(points: torch.Tensor, c2w: torch.Tensor, tans) -> torch.Tensor:
    """Nearest point per cell of each camera: ``[R, H, W]`` axial depth, NaN where none lands."""
    h, w = GRID
    n = int(c2w.shape[0])
    if int(points.shape[0]) == 0:
        return torch.full((n, h, w), float("nan"), dtype=torch.float64)
    out = torch.full((n, h * w), float("inf"), dtype=torch.float64)
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


def _window_median(depth: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor):
    """Median of cells ``(cy-1..cy) x (cx-1..cx)`` (clamped at 0), NaN ignored; ``(median, n)``."""
    r, p = cx.shape
    rows = torch.stack([(cy - 1).clamp(min=0), cy], -1)
    cols = torch.stack([(cx - 1).clamp(min=0), cx], -1)
    w = int(depth.shape[-1])
    flat = (rows[..., :, None] * w + cols[..., None, :]).reshape(r, p * 4)
    vals = depth.reshape(r, -1).gather(1, flat).view(r, p, 4)
    valid = ~torch.isnan(vals)
    n = valid.sum(-1)
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
    tans,
    pose_radius: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-test players against a depth map.

    Args:
        points (torch.Tensor): ``[R, P, 3]`` test points, u.
        c2w (torch.Tensor): ``[R, 4, 4]`` cameras.
        depth (torch.Tensor): ``[R, H, W]`` axial depth, u (NaN: no surface known).
        live (torch.Tensor): ``[R, P]`` bool, alive, present and not the client.
        tans (tuple[float, float]): ``(tan_h, tan_v)``.
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
    h, w = GRID
    col, row, z = _project(points, c2w, tans)
    fc, fr = col * FINE, row * FINE
    in_view = (
        (z > NEAR_U) & (fc >= 0.5) & (fc <= w * FINE - 0.5) & (fr >= 0.5) & (fr <= h * FINE - 0.5)
    )
    cx = fc.floor().clamp(0, w * FINE - 1).long() // FINE
    cy = fr.floor().clamp(0, h * FINE - 1).long() // FINE
    med, n = _window_median(depth.double(), cx, cy)
    live = live.bool()
    hole = in_view & (n == 0) & live
    visible = live & in_view & (n > 0) & (med - z > -MARGIN_U)
    return visible, live & ~hole


def block_depth_rows(depth_fn: DepthFn, x0: torch.Tensor) -> np.ndarray:
    """Axial depth ``[16, 24, 42]`` float64 of a block's 16 pixel rows, from its latent frames.

    Row ``4l + c`` is frame ``c`` of latent frame ``l``; the four latent frames
    ``[4, 48, 24, 42]`` are read in one depth call.
    """
    grid = np.asarray(depth_fn(x0), dtype=np.float64)
    out = np.empty((4 * BLOCK,) + GRID, dtype=np.float64)
    for c in range(4):
        out[c::4] = np.exp(grid[:, c])
    return out


class PredictedVisibility:
    """The visibility labels a client computes for itself, per pixel row of the round.

    ``cameras`` ``[T, 4, 4]`` (the client's table position + eye height, angles from its own view
    controls) and ``points`` ``[T, P, 3]`` (every player's table position + 40 u) follow the
    player-state table: call :meth:`update_rows` after the table changes.

    Args:
        batch (Mapping[str, torch.Tensor]): the whole round (``player_states``, the substeps,
            ``observer_slot``).
        depth_fn (DepthFn): the picture depth head.
        camera_delta_scale (float): degrees per unit camera delta.
        pose_radius (float): radius of the extra test points, u.
    """

    def __init__(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        depth_fn: DepthFn,
        camera_delta_scale: float,
        pose_radius: float,
    ) -> None:
        states = batch["player_states"]
        self.observer = int(torch.as_tensor(batch["observer_slot"]).reshape(-1)[0])
        packed = pack_substeps(
            batch["player_action_substeps"], batch["player_action_substep_valid"]
        )
        yaw, pitch = integrate_camera_angles(
            states[:, :, 0].float(), packed, camera_delta_scale=float(camera_delta_scale)
        )
        self._yaw, self._pitch = yaw[0, self.observer].double(), pitch[0, self.observer].double()
        self.states = states
        self.depth_fn = depth_fn
        self.pose_radius = float(pose_radius)
        self.tans = half_angle_tangents()
        live = (states[0, :, :, 5] > 0.5).permute(1, 0).clone()
        live[:, self.observer] = False
        self.live = live
        n_rows, n_players = int(states.shape[2]), int(states.shape[1])
        self.cameras = torch.zeros(n_rows, 4, 4, dtype=torch.float64)
        self.points = torch.zeros(n_rows, n_players, 3, dtype=torch.float64)
        self.update_rows(np.arange(n_rows))

    def update_rows(self, rows) -> None:
        """Re-read the client's cameras and the test points of pixel ``rows`` from the table."""
        r = torch.as_tensor(np.asarray(rows, np.int64))
        st = self.states[0].float()
        self.cameras[r] = camera_to_world(
            st[self.observer, r, :3].double(), self._yaw[r], self._pitch[r], eye_height=EYE_HEIGHT
        )
        pts = st[:, r, :3].permute(1, 0, 2).double().clone()
        pts[..., 2] += PEER_LIFT_U
        self.points[r] = pts

    def _depth(self, latents: torch.Tensor, first_latent: int) -> dict[int, torch.Tensor]:
        log_depth = torch.as_tensor(self.depth_fn(latents.detach().float())).detach().float().cpu()
        return _depth_rows(log_depth, first_latent)

    def drawn_labels(self, rows: Sequence[int], depth: Mapping[int, torch.Tensor]):
        """Rows whose frames are drawn, against their own depth: ``(visible, valid)`` ``[R, P]``."""
        idx = torch.as_tensor(list(rows), dtype=torch.long)
        return ztest_labels(
            self.points[idx],
            self.cameras[idx],
            torch.stack([depth[r] for r in rows]).double(),
            live=self.live[idx],
            tans=self.tans,
            pose_radius=self.pose_radius,
        )

    def target_labels(
        self, rows: Sequence[int], depth: Mapping[int, torch.Tensor], source_rows: Sequence[int]
    ):
        """Rows of the block about to be drawn, against the drawn ``source_rows``.

        The source rows' surface is splatted into the rows' cameras. Returns ``(visible, valid)``
        ``[R, P]``.
        """
        idx = torch.as_tensor(list(rows), dtype=torch.long)
        if source_rows:
            sidx = torch.as_tensor(list(source_rows), dtype=torch.long)
            pts = _surface_points(
                torch.stack([depth[r] for r in source_rows]), self.cameras[sidx], self.tans
            )
        else:
            pts = torch.zeros(0, 3, dtype=torch.float64)
        return ztest_labels(
            self.points[idx],
            self.cameras[idx],
            _splat(pts, self.cameras[idx], self.tans),
            live=self.live[idx],
            tans=self.tans,
            pose_radius=self.pose_radius,
        )

    # -- gathered windows: labels written into the round batch before the window gather
    def for_block(
        self, s: int, batch: Mapping[str, torch.Tensor], store: torch.Tensor, recent: int
    ) -> dict:
        """``batch`` with ``observer_visibility`` / ``_valid`` predicted on the rows block ``s``'s
        window gathers (the sink, the recent latent frames, the target) and unknown elsewhere. Reads
        ``store[:s]`` only."""
        s, recent = int(s), int(recent)
        ref = batch["observer_visibility"]
        n_rows = int(ref.shape[-1])
        vis = torch.zeros(self.live.shape[1], n_rows, dtype=torch.bool)
        val = torch.zeros_like(vis)
        lo = max(1, s - recent)
        first = max(0, min(lo - 1, s - WARP_LATENTS))
        depth = self._depth(store[first:s], first)
        drawn = [r for f in range(lo, s) for r in _pixel_rows(f)]
        v, va = self.drawn_labels(drawn, depth)
        vis[:, drawn], val[:, drawn] = v.T, va.T
        source = [r for f in range(max(0, s - WARP_LATENTS), s) for r in _pixel_rows(f)]
        target = [r for f in range(s, s + BLOCK) for r in _pixel_rows(f) if r < n_rows]
        v, va = self.target_labels(target, depth, source)
        vis[:, target], val[:, target] = v.T, va.T
        # the sink (latent 0), against its own depth read in the context of latent frames 0 .. 1
        sink = self._depth(store[0 : min(2, s)], 0)[0]
        v, va = ztest_labels(
            self.points[:1],
            self.cameras[:1],
            sink[None].double(),
            live=self.live[:1],
            tans=self.tans,
            pose_radius=self.pose_radius,
        )
        vis[:, 0], val[:, 0] = v[0], va[0]
        out = dict(batch)
        out["observer_visibility"] = vis[None].to(ref.dtype)
        out["observer_visibility_valid"] = val[None].to(batch["observer_visibility_valid"].dtype)
        return out

    def relabel_target(
        self, window: Mapping[str, torch.Tensor], s: int, depth_rows: np.ndarray
    ) -> dict:
        """The gathered window with its last 16 rows (block ``s``) re-tested against ``depth_rows``
        ``[16, 24, 42]``."""
        rows = [r for f in range(int(s), int(s) + BLOCK) for r in _pixel_rows(f)]
        vis, val = self._relabel(rows, depth_rows)
        ref, ok = window["observer_visibility"], window["observer_visibility_valid"]
        where = torch.arange(int(ref.shape[-1]) - 4 * BLOCK, int(ref.shape[-1]), dtype=torch.long)
        new_vis, new_val = ref.clone(), ok.clone()
        new_vis[0, :, where] = vis.T.to(ref.dtype)
        new_val[0, :, where] = val.T.to(ok.dtype)
        return dict(window, observer_visibility=new_vis, observer_visibility_valid=new_val)

    def _relabel(self, rows, depth_rows):
        d = torch.as_tensor(np.asarray(depth_rows, dtype=np.float64))
        if bool(torch.isnan(d).any()) or bool((d <= 0).any()):
            raise ValueError("the x0 depth has NaN or non-positive cells")
        return self.drawn_labels(rows, {r: d[i] for i, r in enumerate(rows)})

    # -- plain prefix: latent-level labels per block, any of a latent frame's rows
    def prefix_labels(self, start: int, n: int, output: torch.Tensor, phase: str) -> torch.Tensor:
        """``peer_visible`` ``[n, P]`` of latent frames ``start .. start+n-1``.

        ``phase='target'`` for the block about to be denoised (``output[:, :start]`` drawn),
        ``'drawn'`` for clean latent frames being written to the cache.
        """
        start, n = int(start), int(n)
        lat = list(range(start, start + n))
        rows = [r for f in lat for r in _pixel_rows(f)]
        if phase == "target":
            source = list(range(max(0, start - WARP_LATENTS), start))
            depth = self._depth(output[0, source[0] : start], source[0]) if source else {}
            v, va = self.target_labels(rows, depth, [r for f in source for r in _pixel_rows(f)])
        elif phase == "drawn":
            first = max(0, start - 1)
            v, _ = self.drawn_labels(rows, self._depth(output[0, first : start + n], first))
        else:
            raise ValueError(f"phase must be target | drawn, got {phase!r}")
        return torch.stack([v[[rows.index(r) for r in _pixel_rows(f)]].any(0) for f in lat])

    def relabel_prefix(self, start: int, depth_rows: np.ndarray) -> torch.Tensor:
        """``peer_visible`` ``[4, P]`` of the prefix block at ``start``, tested on its x0 depth."""
        rows = [r for f in range(int(start), int(start) + BLOCK) for r in _pixel_rows(f)]
        vis, _ = self._relabel(rows, depth_rows)
        return torch.stack(
            [
                vis[[rows.index(r) for r in _pixel_rows(f)]].any(0)
                for f in range(int(start), int(start) + BLOCK)
            ]
        )
