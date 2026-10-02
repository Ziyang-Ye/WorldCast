"""Scene-state geometry on the 24 x 42 depth grid: rays, projection, z-buffers and coverage.

A camera is a camera-to-world ``c2w`` ``[4, 4]`` with rotation columns right / down / forward in
engine units (u), and ``tan = (tan_h, tan_v)`` its half-angle field-of-view tangents. Depth is axial
(along the forward axis). Pixel ``(i, j)`` has flat index ``i * 42 + j``. All arithmetic is float64
numpy on the CPU, as deployed.
"""

import numpy as np

__all__ = [
    "GRID",
    "NPIX",
    "FAR_U",
    "MISS_FRACTION",
    "NEAR_U",
    "MARGIN_ABS_U",
    "MARGIN_REL",
    "margin",
    "pixel_rays",
    "view_points",
    "project",
    "zbuffer",
    "explained",
    "axial_from_log_grid",
]

#: The depth read-out grid (rows, columns): 16 x 16-pixel patches of a 384 x 672 frame.
GRID: tuple[int, int] = (24, 42)
NPIX: int = GRID[0] * GRID[1]
#: Far plane of the depth head's training target, u; a pixel at ``>= 0.95 FAR_U`` is a miss.
FAR_U: float = 4096.0
MISS_FRACTION: float = 0.95
#: A point must lie more than this far in front of a camera (axially, u) to project into it.
NEAR_U: float = 1.0
#: Depth tolerance ``max(MARGIN_ABS_U, MARGIN_REL d)``: "the larger of 24 u and 5%" (App. scene
#: state).
MARGIN_ABS_U: float = 24.0
MARGIN_REL: float = 0.05


def margin(d) -> np.ndarray:
    """Depth tolerance ``max(24 u, 0.05 d)`` at axial depth(s) ``d`` (any shape), float64, u."""
    return np.maximum(MARGIN_ABS_U, MARGIN_REL * np.asarray(d, dtype=np.float64))


def pixel_rays(c2w, tan) -> tuple[np.ndarray, np.ndarray]:
    """The 24 x 42 viewing rays of one camera.

    Args:
        c2w (np.ndarray): ``[4, 4]`` camera-to-world (any float dtype; used in float64).
        tan (np.ndarray): ``[2]`` ``(tan_h, tan_v)``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``eye`` ``[3]`` and ``rays`` ``[NPIX, 3]``, float64: ``R [x
        tan_h, y tan_v, 1]`` at the pixel centres ``x = (j + .5) * 2 / 42 - 1``, ``y = (i + .5) * 2
        / 24 - 1``, row-major; their forward component is 1, so a depth along them is axial.
    """
    c = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    t = np.asarray(tan, dtype=np.float64).reshape(2)
    h, w = GRID
    xx, yy = np.meshgrid((np.arange(w) + 0.5) * 2 / w - 1, (np.arange(h) + 0.5) * 2 / h - 1)
    r = np.stack([xx.ravel() * t[0], yy.ravel() * t[1], np.ones(NPIX)], 1) @ c[:3, :3].T
    return c[:3, 3].copy(), r


def view_points(c2w, tans, depth) -> tuple[np.ndarray, np.ndarray]:
    """Back-project axial depth maps to world points.

    Args:
        c2w (np.ndarray): ``[M, 4, 4]`` cameras.
        tans (np.ndarray): ``[M, 2]``, or ``[1, 2]`` shared by all cameras.
        depth (np.ndarray): ``[M, 24, 42]`` (or ``[M, NPIX]``) axial depth, u.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``points`` ``[M, NPIX, 3]`` float64 world points (u) and
        ``hit`` ``[M, NPIX]`` bool, the pixels with a surface (``0 < d < 0.95 * 4096``). A missed
        pixel's point is the camera centre and must be ignored.
    """
    cams = np.asarray(c2w, dtype=np.float64).reshape(-1, 4, 4)
    tans = np.asarray(tans, dtype=np.float64).reshape(-1, 2)
    if len(tans) == 1:
        tans = np.repeat(tans, len(cams), 0)
    depth = np.asarray(depth, dtype=np.float64).reshape(len(cams), NPIX)
    hit = np.isfinite(depth) & (depth > 0) & (depth < MISS_FRACTION * FAR_U)
    points = np.zeros((len(cams), NPIX, 3))
    for m in range(len(cams)):
        eye, r = pixel_rays(cams[m], tans[m])
        points[m] = eye[None] + np.where(hit[m], depth[m], 0.0)[:, None] * r
    return points, hit


def project(points, c2w, tan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into one camera.

    A point lands iff it is more than ``NEAR_U`` in front of the camera and inside the field of view
    (``|u|, |v| <= 1`` in tangent-normalised coordinates); its pixel is ``floor((v + 1) / 2 * 24)``,
    ``floor((u + 1) / 2 * 42)``, clipped to the grid.

    Args:
        points (np.ndarray): ``[n, 3]`` world points, u.
        c2w (np.ndarray): ``[4, 4]`` camera-to-world.
        tan (np.ndarray): ``(tan_h, tan_v)``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``pix`` ``[k]`` int64 flat pixel index and ``z``
        ``[k]`` float64 axial depth (u) of the ``k`` points that land, in input order, and ``ok``
        ``[n]`` bool, the mask selecting them.
    """
    c = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    t = np.asarray(tan, dtype=np.float64).reshape(2)
    # inv(R).T rather than R.T, as deployed: integer pixel counts downstream can flip otherwise
    q = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - c[:3, 3][None]) @ np.linalg.inv(
        c[:3, :3]
    ).T
    z = q[:, 2]
    zs = np.where(z > NEAR_U, z, 1.0)
    u, v = q[:, 0] / (zs * t[0]), q[:, 1] / (zs * t[1])
    ok = (z > NEAR_U) & (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
    h, w = GRID
    i = np.clip(np.floor((v[ok] + 1.0) * 0.5 * h), 0, h - 1).astype(np.int64)
    j = np.clip(np.floor((u[ok] + 1.0) * 0.5 * w), 0, w - 1).astype(np.int64)
    return i * w + j, z[ok], ok


def zbuffer(points, c2w, tan, owner: np.ndarray | None = None, n_owner: int = 1) -> np.ndarray:
    """Splat points into one camera, keeping the nearest (smallest axial depth) per pixel.

    Args:
        points (np.ndarray): ``[n, 3]`` world points, u.
        c2w (np.ndarray): ``[4, 4]`` camera-to-world.
        tan (np.ndarray): ``(tan_h, tan_v)``.
        owner (np.ndarray | None): ``[n]`` ints in ``[0, n_owner)``; with it, one z-buffer per owner
            in one pass.
        n_owner (int): number of owners (with ``owner`` only).

    Returns:
        np.ndarray: float64 ``[NPIX]`` (no ``owner``) or ``[n_owner, NPIX]``, axial depth in u,
        ``inf`` where no point lands.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if owner is None:
        buf = np.full(NPIX, np.inf)
        if len(pts):
            pix, z, _ = project(pts, c2w, tan)
            np.minimum.at(buf, pix, z)
        return buf
    buf = np.full(int(n_owner) * NPIX, np.inf)
    if len(pts):
        pix, z, ok = project(pts, c2w, tan)
        np.minimum.at(buf, np.asarray(owner, dtype=np.int64)[ok] * NPIX + pix, z)
    return buf.reshape(int(n_owner), NPIX)


def explained(zbuf, depth, hit) -> np.ndarray:
    """Which pixels of a view a z-buffer reproduces within the tolerance (the coverage test).

    A point set covers pixel ``p`` of a view with depth ``d(p)`` iff the view has a surface there,
    the set's z-buffer has a point there, and ``|zbuf(p) - d(p)| <= max(24 u, 0.05 d(p))``.

    Args:
        zbuf (np.ndarray): the point set's z-buffer in the view's camera(s), broadcastable with
            ``depth``, u.
        depth (np.ndarray): the view's axial depth, u.
        hit (np.ndarray): the view's surface mask (:func:`view_points`).

    Returns:
        np.ndarray: bool, the shape of the broadcast inputs.
    """
    d = np.asarray(depth, dtype=np.float64)
    zb = np.asarray(zbuf, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(hit, bool) & np.isfinite(zb) & (np.abs(zb - d) <= margin(d))


def axial_from_log_grid(log_grid) -> np.ndarray:
    """The depth-head read-out as the axial depth a memory entry stores.

    A latent frame spans four video frames; its camera and depth are those of the last one (App.
    "Scene state in detail"), channel -1 of ``log_grid`` (log axial depth ``[M, 4, 24, 42]``).

    Returns:
        np.ndarray: ``[M, 24, 42]`` float64, u.
    """
    g = np.asarray(log_grid, dtype=np.float64)
    if g.ndim != 4 or g.shape[-2:] != GRID:
        raise ValueError(f"log depth grid must be [M, C, {GRID[0]}, {GRID[1]}], got {g.shape}")
    return np.exp(g[:, -1])
