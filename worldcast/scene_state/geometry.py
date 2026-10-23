"""Scene-state geometry on the depth grid: rays, reprojection, z-buffers and coverage (Sec. 3.3).

A camera is a camera-to-world ``c2w`` ``[4, 4]`` with rotation columns right / down / forward in
engine units (u), and ``tan = (tan_h, tan_v)`` its half-angle field-of-view tangents. Depth is axial
(along the forward axis). The depth grid is the latent grid, 24 x 42, one pixel per 16 x 16-pixel
patch of a frame; pixel ``(i, j)`` has flat index ``i * 42 + j``. All arithmetic is float64 numpy on
the CPU, as in the paper's runs; the predicted visibility reprojects the same depth with torch on a
finer grid (:mod:`worldcast.player_state.predicted_visibility`), likewise as in the paper's runs.
"""

import functools
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from worldcast.data.camera import NEAR_U, camera_rays
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
    "zbuffers",
    "zbuffers_of",
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


@functools.cache
def _camera_rays(tan: tuple[float, float]) -> np.ndarray:
    rays = camera_rays(tan, LATENT_GRID)
    rays.flags.writeable = False
    return rays


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
    return c[:3, 3].copy(), _camera_rays((float(t[0]), float(t[1]))) @ c[:3, :3].T


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
    # each camera's rays are pixel_rays', one product per camera
    rays = np.matmul(
        np.stack([_camera_rays((float(t[0]), float(t[1]))) for t in tans]),
        cams[:, :3, :3].transpose(0, 2, 1),
    )
    return cams[:, None, :3, 3] + np.where(hit, depth, 0.0)[..., None] * rays, hit


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
    c = np.asarray(c2w, dtype=np.float64).reshape(1, 4, 4)
    t = np.asarray(tan, dtype=np.float64).reshape(2)
    ok, _, pix, z = _landing(_in_cameras(_points(points), c[:, :3, 3], _rotations(c))[0], t)
    return pix, z, ok


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
    pts = _points(points)
    groups = 1 if group is None else int(n_groups)
    buf = np.full(groups * NPIX, np.inf)
    if len(pts):
        c = np.asarray(c2w, dtype=np.float64).reshape(1, 4, 4)
        t = np.asarray(tan, dtype=np.float64).reshape(2)
        _, landed, pix, z = _landing(_in_cameras(pts, c[:, :3, 3], _rotations(c))[0], t)
        if group is not None:
            pix = np.asarray(group, dtype=np.int64)[landed] * NPIX + pix
        np.minimum.at(buf, pix, z)
    return buf if group is None else buf.reshape(groups, NPIX)


def zbuffers(points: ArrayLike, c2w: ArrayLike, tans: ArrayLike) -> np.ndarray:
    """:func:`zbuffer` of one point set in each of ``M`` cameras, in one pass.

    Args:
        points (ArrayLike): ``[n, 3]`` world points, u.
        c2w (ArrayLike): ``[M, 4, 4]`` cameras.
        tans (ArrayLike): ``[M, 2]``, or ``[1, 2]`` shared by all cameras.

    Returns:
        np.ndarray: float64 ``[M, NPIX]``, axial depth in u, ``inf`` where no point lands.
    """
    return zbuffers_of([points], c2w, tans)[0]


def zbuffers_of(point_sets: Sequence[ArrayLike], c2w: ArrayLike, tans: ArrayLike) -> np.ndarray:
    """:func:`zbuffer` of each of ``S`` point sets in each of ``M`` cameras, in one pass; each set
    is projected into each camera as :func:`project` projects it.

    Args:
        point_sets (Sequence[ArrayLike]): ``S`` sets of world points ``[n_s, 3]``, u.
        c2w (ArrayLike): ``[M, 4, 4]`` cameras.
        tans (ArrayLike): ``[M, 2]``, or ``[1, 2]`` shared by all cameras.

    Returns:
        np.ndarray: float64 ``[S, M, NPIX]``, axial depth in u, ``inf`` where no point lands.
    """
    cams = np.asarray(c2w, dtype=np.float64).reshape(-1, 4, 4)
    tans = per_camera_tans(tans, len(cams))
    sets = [_points(points) for points in point_sets]
    sizes = [len(points) for points in sets]
    m, total = len(cams), sum(sizes)
    buf = np.full(len(sets) * m * NPIX, np.inf)
    if total:
        eyes, rotations = cams[:, :3, 3], _rotations(cams)
        q = np.empty((m, total, 3))
        start = 0
        for points in sets:
            if len(points):
                _in_cameras(points, eyes, rotations, out=q[:, start : start + len(points)])
            start += len(points)
        _, landed, pix, z = _landing(q, tans[:, None, :])
        camera, row = np.divmod(landed, total)
        group = np.repeat(np.arange(len(sets)), sizes)[row]
        np.minimum.at(buf, (group * m + camera) * NPIX + pix, z)
    return buf.reshape(len(sets), m, NPIX)


def _points(points: ArrayLike) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _rotations(cams: np.ndarray) -> np.ndarray:
    """``[M, 3, 3]``: the transposed inverse of the rotation of each camera of ``cams`` ``[M, 4,
    4]``, by which a point's offset from the eye is multiplied."""
    # inv(R).T rather than R.T, as in the paper's runs: integer pixel counts downstream can flip
    # otherwise
    return np.linalg.inv(cams[:, :3, :3]).transpose(0, 2, 1)


def _in_cameras(
    points: np.ndarray, eyes: np.ndarray, rotations: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """``points`` ``[n, 3]`` in the coordinates of each of ``M`` cameras (``eyes`` ``[M, 3]``,
    :func:`_rotations`): ``[M, n, 3]``, one product per camera."""
    offsets = np.empty((len(eyes), len(points), 3))
    for k in range(3):
        np.subtract(points[None, :, k], eyes[:, None, k], out=offsets[:, :, k])
    return np.matmul(offsets, rotations, out=out)


def _landing(
    q: np.ndarray, tan: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The points in camera coordinates ``q`` ``[..., 3]`` that land (:func:`project`) in cameras of
    tangents ``tan`` (``[2]``, or one per camera broadcast against ``q``): the mask ``ok`` of
    ``q``'s shape less its last axis, the flat indices of the points that land, their pixels and
    their axial depths."""
    z = q[..., 2]
    # u and v of a point behind the near plane are never read: its division needs no guard
    with np.errstate(divide="ignore", invalid="ignore"):
        u = q[..., 0] / (z * tan[..., 0])
        v = q[..., 1] / (z * tan[..., 1])
    ok = (z > NEAR_U) & (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
    landed = np.flatnonzero(ok)
    u, v, z = (a.reshape(-1)[landed] for a in (u, v, z))
    h, w = LATENT_GRID
    i = np.clip(np.floor((v + 1.0) * 0.5 * h), 0, h - 1).astype(np.int64)
    j = np.clip(np.floor((u + 1.0) * 0.5 * w), 0, w - 1).astype(np.int64)
    return ok, landed, i * w + j, z


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
