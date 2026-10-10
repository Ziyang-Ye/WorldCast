"""Scene-state geometry on the depth grid: rays, reprojection, z-buffers and coverage (Sec. 3.3).

A camera is a camera-to-world ``c2w`` ``[4, 4]`` with rotation columns right / down / forward in
engine units (u), and ``tan = (tan_h, tan_v)`` its half-angle field-of-view tangents. Depth is axial
(along the forward axis). The depth grid is the latent grid, 24 x 42, one pixel per 16 x 16-pixel
patch of a frame; pixel ``(i, j)`` has flat index ``i * 42 + j``. All arithmetic is float64 numpy on
the CPU, as in the paper's runs; the predicted visibility reprojects the same depth with torch on a
finer grid (:mod:`worldcast.player_state.predicted_visibility`), likewise as in the paper's runs.
"""

import numpy as np
from numpy.typing import ArrayLike

from worldcast.data.camera import NEAR_U, grid_rays
from worldcast.data.latents import LATENT_GRID

__all__ = [
    "FAR_U",
    "MISS_FRACTION",
    "NPIX",
    "TOLERANCE_ABS_U",
    "TOLERANCE_REL",
    "axial_from_log_depth",
    "covered",
    "per_camera_tans",
    "pixel_rays",
    "project",
    "tolerance",
    "view_points",
    "zbuffer",
]

#: Pixels of the depth grid.
NPIX = LATENT_GRID[0] * LATENT_GRID[1]
#: Far plane of the depth head's training target, u. Apart from the scale of the player state
#: field's depth channel (``worldcast.modeling.state_injector.DEPTH_LOG_SCALE``), which is
#: ``log1p`` of the same 4096 u.
FAR_U = 4096.0
#: A pixel whose depth reaches this fraction of :data:`FAR_U` has no surface (a miss).
MISS_FRACTION = 0.95
#: The depth tolerance of coverage, ``max(TOLERANCE_ABS_U, TOLERANCE_REL d)``: "the larger of 24 u
#: and 5%" (App. "Scene state in detail").
TOLERANCE_ABS_U = 24.0
TOLERANCE_REL = 0.05


def tolerance(depth: ArrayLike) -> np.ndarray:
    """The depth tolerance ``max(24 u, 0.05 d)`` at axial depth(s) ``d`` (any shape): float64, u."""
    return np.maximum(TOLERANCE_ABS_U, TOLERANCE_REL * np.asarray(depth, dtype=np.float64))


def per_camera_tans(tans: ArrayLike, n: int) -> np.ndarray:
    """``[n, 2]`` float64 tangents of ``n`` cameras, from ``[n, 2]`` or from one ``(tan_h, tan_v)``
    shared by all."""
    tans = np.asarray(tans, dtype=np.float64).reshape(-1, 2)
    if len(tans) not in (1, n):
        raise ValueError(f"{len(tans)} fields of view for {n} cameras: one each, or one for all")
    return np.repeat(tans, n, 0) if len(tans) == 1 else tans


def pixel_rays(c2w: ArrayLike, tan: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The viewing rays of one camera through the pixel centres of the depth grid.

    Args:
        c2w (ArrayLike): ``[4, 4]`` camera-to-world (any float dtype; used in float64).
        tan (ArrayLike): ``[2]`` ``(tan_h, tan_v)``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``eye`` ``[3]`` and ``rays`` ``[NPIX, 3]``, float64, in the
        world, row-major (:func:`worldcast.data.camera.camera_rays`); their forward component is 1,
        so a depth along them is axial.
    """
    c = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    t = np.asarray(tan, dtype=np.float64).reshape(2)
    return c[:3, 3].copy(), grid_rays(c, t, LATENT_GRID)


def view_points(c2w: ArrayLike, tans: ArrayLike, depth: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Back-project axial depth maps to world points.

    Args:
        c2w (ArrayLike): ``[M, 4, 4]`` cameras.
        tans (ArrayLike): ``[M, 2]``, or ``[1, 2]`` shared by all cameras.
        depth (ArrayLike): ``[M, 24, 42]`` (or ``[M, NPIX]``) axial depth, u.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``points`` ``[M, NPIX, 3]`` float64 world points (u) and
        ``hit`` ``[M, NPIX]`` bool, the pixels with a surface (``0 < d < 0.95 * 4096``). A missed
        pixel's point is the camera centre and must be ignored.
    """
    cams = np.asarray(c2w, dtype=np.float64).reshape(-1, 4, 4)
    tans = per_camera_tans(tans, len(cams))
    depth = np.asarray(depth, dtype=np.float64).reshape(len(cams), NPIX)
    hit = np.isfinite(depth) & (depth > 0) & (depth < MISS_FRACTION * FAR_U)
    points = np.zeros((len(cams), NPIX, 3))
    for m in range(len(cams)):
        eye, rays = pixel_rays(cams[m], tans[m])
        points[m] = eye[None] + np.where(hit[m], depth[m], 0.0)[:, None] * rays
    return points, hit


def project(
    points: ArrayLike, c2w: ArrayLike, tan: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into one camera.

    A point lands iff it is more than ``NEAR_U`` in front of the camera and inside the field of view
    (``|u|, |v| <= 1`` in tangent-normalised coordinates); its pixel is ``floor((v + 1) / 2 * 24)``,
    ``floor((u + 1) / 2 * 42)``, clipped to the grid.

    Args:
        points (ArrayLike): ``[n, 3]`` world points, u.
        c2w (ArrayLike): ``[4, 4]`` camera-to-world.
        tan (ArrayLike): ``(tan_h, tan_v)``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``pix`` ``[k]`` int64 flat pixel index and ``z``
        ``[k]`` float64 axial depth (u) of the ``k`` points that land, in input order, and ``ok``
        ``[n]`` bool, the mask selecting them.
    """
    c = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    t = np.asarray(tan, dtype=np.float64).reshape(2)
    # inv(R).T rather than R.T, as in the paper's runs (the pixel counts downstream follow it)
    q = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - c[:3, 3][None]) @ np.linalg.inv(
        c[:3, :3]
    ).T
    z = q[:, 2]
    zs = np.where(z > NEAR_U, z, 1.0)
    u, v = q[:, 0] / (zs * t[0]), q[:, 1] / (zs * t[1])
    ok = (z > NEAR_U) & (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
    h, w = LATENT_GRID
    i = np.clip(np.floor((v[ok] + 1.0) * 0.5 * h), 0, h - 1).astype(np.int64)
    j = np.clip(np.floor((u[ok] + 1.0) * 0.5 * w), 0, w - 1).astype(np.int64)
    return i * w + j, z[ok], ok


def zbuffer(
    points: ArrayLike,
    c2w: ArrayLike,
    tan: ArrayLike,
    group: np.ndarray | None = None,
    n_groups: int = 1,
) -> np.ndarray:
    """Project points into one camera, keeping the nearest (smallest axial depth) per pixel.

    Args:
        points (ArrayLike): ``[n, 3]`` world points, u.
        c2w (ArrayLike): ``[4, 4]`` camera-to-world.
        tan (ArrayLike): ``(tan_h, tan_v)``.
        group (np.ndarray | None): ``[n]`` ints in ``[0, n_groups)``; with it, one z-buffer per
            group of points in one pass.
        n_groups (int): number of groups (with ``group`` only).

    Returns:
        np.ndarray: float64 ``[NPIX]`` (no ``group``) or ``[n_groups, NPIX]``, axial depth in u,
        ``inf`` where no point lands.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if group is None:
        buf = np.full(NPIX, np.inf)
        if len(pts):
            pix, z, _ = project(pts, c2w, tan)
            np.minimum.at(buf, pix, z)
        return buf
    buf = np.full(int(n_groups) * NPIX, np.inf)
    if len(pts):
        pix, z, ok = project(pts, c2w, tan)
        np.minimum.at(buf, np.asarray(group, dtype=np.int64)[ok] * NPIX + pix, z)
    return buf.reshape(int(n_groups), NPIX)


def covered(zbuf: ArrayLike, depth: ArrayLike, hit: ArrayLike) -> np.ndarray:
    """Which pixels of a view a z-buffer reproduces within the tolerance (the coverage test).

    A point set covers pixel ``p`` of a view with depth ``d(p)`` iff the view has a surface there,
    the set's z-buffer has a point there, and ``|zbuf(p) - d(p)| <= max(24 u, 0.05 d(p))``.

    Args:
        zbuf (ArrayLike): the point set's z-buffer in the view's camera(s), broadcastable with
            ``depth``, u.
        depth (ArrayLike): the view's axial depth, u.
        hit (ArrayLike): the view's surface mask (:func:`view_points`).

    Returns:
        np.ndarray: bool, the shape of the broadcast inputs.
    """
    d = np.asarray(depth, dtype=np.float64)
    zb = np.asarray(zbuf, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(hit, bool) & np.isfinite(zb) & (np.abs(zb - d) <= tolerance(d))


def axial_from_log_depth(log_depth: ArrayLike) -> np.ndarray:
    """The depth head's read-out as the axial depth a memory entry stores.

    A latent frame spans four video frames; its camera and depth are those of the last one (App.
    "Scene state in detail"), channel -1 of ``log_depth`` (log axial depth ``[M, 4, 24, 42]``).

    Returns:
        np.ndarray: ``[M, 24, 42]`` float64, u.
    """
    g = np.asarray(log_depth, dtype=np.float64)
    if g.ndim != 4 or g.shape[-2:] != LATENT_GRID:
        raise ValueError(
            f"log depth must be [M, C, {LATENT_GRID[0]}, {LATENT_GRID[1]}], got {g.shape}"
        )
    return np.exp(g[:, -1])
