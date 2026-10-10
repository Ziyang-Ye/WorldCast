"""The appearance check of a training window's memory frames: an acceptance test of
:mod:`~worldcast.data.memory_selection`, beyond the paper's text.

A token of m_k counts only if its surface patch looks the same in the target frame and in a memory
frame: the token's footprint on the tangent plane of its surface point, 17 x 17 samples, is warped
into both recorded frames and compared (:func:`matching_patches`).

Optional dependencies, imported when used: OpenCV and scipy.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from .camera import as_float64, to_camera
from .latents import BLOCK, TOKEN_GRID, source_frame
from .map_mesh import MapMesh
from .memory_frames import MemoryBlock
from .memory_mask import (
    MIN_INCIDENCE_COSINE,
    UNCLIPPED_GREY,
    TargetSurface,
    behind_players,
    coverage,
    off_hud,
    project_points,
    rgb_quality_mask,
)
from .recordings import MediaRecord, TickTable, player_rows
from .video import VideoFrames

__all__ = ["PATCH_LIMITS", "PATCH_SIDE", "appearance_check", "matching_patches"]

#: Samples per side of a token's footprint, and points per batch.
PATCH_SIDE = 17
_PATCH_BATCH = 256
#: Limits of a matching patch pair.
PATCH_LIMITS = {
    "min_incidence_cosine": MIN_INCIDENCE_COSINE,
    "min_texture_std": 0.020,
    "min_zncc": 0.85,
    "min_gradient_cosine": 0.65,
    "max_rgb_mae": 0.080,
    "max_rgb_p90": 0.150,
    "max_quadrant_mae": 0.100,
    "bad_pixel_rgb_error": 0.200,
    "max_bad_pixel_fraction": 0.050,
    "max_clipped_fraction": 0.100,
}


def _patch_project(points, camera, tans, image_shape):
    """World points ``[..., 3]`` -> pixel centres ``[..., 2]`` of an image and axial depth."""
    h, w = image_shape[:2]
    p = to_camera(points, camera)
    z = p[..., 2]
    xy = np.divide(
        p[..., :2], z[..., None], out=np.zeros_like(p[..., :2]), where=np.abs(z[..., None]) > 1e-9
    )
    return (xy / tans * 0.5 + 0.5) * np.array([w, h]) - 0.5, z


def _warp_patches(points, normals, target_c2w, source_c2w, target_tans, source_tans, shapes):
    """Pixel centres ``[N, 17, 17, 2]`` of each point's token footprint in the target image and of
    the same tangent-plane points in the source image, and the geometric validity ``[N]``."""
    q, s = as_float64(target_c2w), as_float64(source_c2w)
    qt, st = as_float64(target_tans), as_float64(source_tans)
    norm = np.linalg.norm(normals, axis=1)
    finite = np.isfinite(points).all(1) & np.isfinite(normals).all(1) & (norm > 1e-9)
    n = np.divide(normals, norm[:, None], out=np.zeros_like(normals), where=norm[:, None] > 1e-9)
    centres, qz = _patch_project(points, q, qt, shapes[0])
    h, w = shapes[0][:2]
    offset = (np.arange(PATCH_SIDE) + 0.5) / PATCH_SIDE - 0.5
    ox, oy = np.meshgrid(offset * w / TOKEN_GRID[1], offset * h / TOKEN_GRID[0])
    qxy = centres[:, None, None] + np.stack([ox, oy], axis=-1)
    uv = ((qxy + 0.5) / np.array([w, h]) * 2 - 1) * qt
    rays = np.concatenate([uv, np.ones((*uv.shape[:-1], 1))], axis=-1) @ q[:3, :3].T
    denom = np.einsum("nijk,nk->nij", rays, n)
    numerator = np.einsum("nk,nk->n", points - q[:3, 3], n)
    depth = np.divide(
        numerator[:, None, None], denom, out=np.zeros_like(denom), where=np.abs(denom) > 1e-9
    )
    world = q[:3, 3] + depth[..., None] * rays
    sxy, sz = _patch_project(world, s, st, shapes[1])
    q_cos = np.abs(denom) / np.maximum(np.linalg.norm(rays, axis=-1), 1e-9)
    source_rays = world - s[:3, 3]
    s_cos = np.abs(np.einsum("nijk,nk->nij", source_rays, n)) / np.maximum(
        np.linalg.norm(source_rays, axis=-1), 1e-9
    )
    pixel_geometry = (
        (depth > 1e-7)
        & (sz > 1e-7)
        & (q_cos >= PATCH_LIMITS["min_incidence_cosine"])
        & (s_cos >= PATCH_LIMITS["min_incidence_cosine"])
    )
    valid = (
        finite
        & (qz > 1e-7)
        & pixel_geometry.all((1, 2))
        & np.isfinite(qxy).all((1, 2, 3))
        & np.isfinite(sxy).all((1, 2, 3))
    )
    return np.nan_to_num(qxy), np.nan_to_num(sxy), valid


def _rgb_float(image) -> np.ndarray:
    a = np.asarray(image)
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32, copy=False)


def _bilinear(image, xy):
    h, w = image.shape[:2]
    x, y = np.clip(xy[..., 0], 0, w - 1), np.clip(xy[..., 1], 0, h - 1)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    ax, ay = (x - x0)[..., None], (y - y0)[..., None]
    return (
        (1 - ax) * (1 - ay) * image[y0, x0]
        + ax * (1 - ay) * image[y0, x1]
        + (1 - ax) * ay * image[y1, x0]
        + ax * ay * image[y1, x1]
    ).astype(np.float32)


def _measurable(xy, shape):
    h, w = shape[:2]
    inside = off_hud((xy[..., 0] + 0.5) / w, (xy[..., 1] + 0.5) / h)
    return (
        inside & (xy[..., 0] >= 0) & (xy[..., 0] < w - 1) & (xy[..., 1] >= 0) & (xy[..., 1] < h - 1)
    )


def _gradient_cosine(a, b):
    ad = np.stack(np.gradient(a, axis=(1, 2)), axis=-1)
    bd = np.stack(np.gradient(b, axis=(1, 2)), axis=-1)
    return (ad * bd).sum((1, 2, 3)) / np.maximum(
        np.sqrt((ad * ad).sum((1, 2, 3)) * (bd * bd).sum((1, 2, 3))), 1e-9
    )


def _matching_batch(target_rgb, source_rgb, points, normals, cameras, tans) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    qxy, sxy, valid = _warp_patches(
        points, normals, *cameras, *tans, (target_rgb.shape, source_rgb.shape)
    )
    pixels = _measurable(qxy, target_rgb.shape) & _measurable(sxy, source_rgb.shape)
    a, b = _bilinear(target_rgb, qxy), _bilinear(source_rgb, sxy)
    gray_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    ag, bg = a @ gray_weights, b @ gray_weights
    az, bz = ag - ag.mean((1, 2), keepdims=True), bg - bg.mean((1, 2), keepdims=True)
    aq, bq = np.sqrt(np.mean(az * az, axis=(1, 2))), np.sqrt(np.mean(bz * bz, axis=(1, 2)))
    zncc = np.mean(az * bz, axis=(1, 2)) / np.maximum(aq * bq, 1e-9)
    sigma = (0, 1.0, 1.0)
    gradient = _gradient_cosine(
        gaussian_filter(ag, sigma=sigma, mode="reflect", truncate=4.0),
        gaussian_filter(bg, sigma=sigma, mode="reflect", truncate=4.0),
    )
    error = np.abs(a - b).mean(-1)
    mid = PATCH_SIDE // 2
    quadrant = np.stack(
        [
            error[:, ys, xs].mean((1, 2))
            for ys in (slice(0, mid), slice(mid, None))
            for xs in (slice(0, mid), slice(mid, None))
        ],
        axis=1,
    ).max(1)
    t = PATCH_LIMITS
    low, high = UNCLIPPED_GREY
    target_clipped = ((ag < low) | (ag > high)).mean((1, 2))
    source_clipped = ((bg < low) | (bg > high)).mean((1, 2))
    rejected = (
        ~valid,
        ~pixels.all((1, 2)),
        (aq < t["min_texture_std"]) | (bq < t["min_texture_std"]),
        (target_clipped > t["max_clipped_fraction"]) | (source_clipped > t["max_clipped_fraction"]),
        np.clip(zncc, -1, 1) < t["min_zncc"],
        np.clip(gradient, -1, 1) < t["min_gradient_cosine"],
        (error.mean((1, 2)) > t["max_rgb_mae"])
        | (np.quantile(error, 0.90, axis=(1, 2)) > t["max_rgb_p90"]),
        quadrant > t["max_quadrant_mae"],
        (error > t["bad_pixel_rgb_error"]).mean((1, 2)) > t["max_bad_pixel_fraction"],
    )
    return ~np.logical_or.reduce(list(rejected))


def matching_patches(
    target_rgb: np.ndarray,
    source_rgb: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    target_c2w: np.ndarray,
    source_c2w: np.ndarray,
    target_tans: Sequence[float],
    source_tans: Sequence[float],
) -> np.ndarray:
    """Which surface points look the same in a target frame and in a memory frame.

    A point's token footprint (17 x 17 samples on its tangent plane) is read in both images; the
    patches match by texture, clipping, ZNCC >= 0.85, gradient cosine >= 0.65, and the colour and
    outlier limits (:data:`PATCH_LIMITS`).

    Args:
        target_rgb (np.ndarray): ``[H, W, 3]`` the target frame, uint8 or float in [0, 1].
        source_rgb (np.ndarray): ``[H, W, 3]`` the memory frame.
        points (np.ndarray): ``[N, 3]`` world points, u.
        normals (np.ndarray): ``[N, 3]`` their surface normals.
        target_c2w (np.ndarray): ``[4, 4]`` the target frame's camera.
        source_c2w (np.ndarray): ``[4, 4]`` the memory frame's camera.
        target_tans (Sequence[float]): ``(tan_h, tan_v)`` of the target frame.
        source_tans (Sequence[float]): ``(tan_h, tan_v)`` of the memory frame.

    Returns:
        np.ndarray: ``[N]`` bool.
    """
    target_rgb, source_rgb = _rgb_float(target_rgb), _rgb_float(source_rgb)
    points, normals = as_float64(points), as_float64(normals)
    out = np.zeros(len(points), dtype=bool)
    for lo in range(0, len(points), _PATCH_BATCH):
        hi = min(lo + _PATCH_BATCH, len(points))
        out[lo:hi] = _matching_batch(
            target_rgb,
            source_rgb,
            points[lo:hi],
            normals[lo:hi],
            (target_c2w, source_c2w),
            (target_tans, source_tans),
        )
    return out


def appearance_check(
    mesh: MapMesh,
    video: VideoFrames,
    *,
    client_media: MediaRecord,
    tick_tables: Mapping[int, TickTable],
    start_frame: int,
    target_start: int,
    memory_mask: np.ndarray,
    window_c2w: np.ndarray,
    window_tans: np.ndarray,
    block: MemoryBlock,
    block_tans: np.ndarray,
) -> np.ndarray:
    """``[4, 12, 21]`` bool: the tokens of m_k whose surface patch looks the same in the target
    frame and in at least one of the memory frames (:func:`matching_patches`).

    Args:
        mesh (MapMesh): the map.
        video (VideoFrames): the round's decoded frames.
        client_media (MediaRecord): the window's player.
        tick_tables (Mapping[int, TickTable]): every slot's ticks.
        start_frame (int): the window's first source frame.
        target_start (int): the target block's first latent frame.
        memory_mask (np.ndarray): m_k ``[4, 12, 21]`` bool.
        window_c2w (np.ndarray): ``[L, 4, 4]`` float64 cameras of the window's latent frames.
        window_tans (np.ndarray): ``[L, 2]`` float64.
        block (MemoryBlock): the memory frames.
        block_tans (np.ndarray): ``[4, 2]`` float32 fields of view of the memory frames.
    """
    fps, client_slot = float(client_media.fps), int(client_media.player_slot)
    source_frames = [source_frame(block.t_first, j) for j in range(BLOCK)]
    source_c2w = np.asarray(block.c2w.numpy(), dtype=np.float64)
    source_tans = np.asarray(block_tans, dtype=np.float32).astype(np.float64)
    images = [video.get(block.media_id, frame) for frame in source_frames]
    players = [player_rows(tick_tables, frame, fps) for frame in source_frames]
    out = np.zeros_like(memory_mask, dtype=bool)
    for qi in range(BLOCK):
        camera, tans = window_c2w[target_start + qi], window_tans[target_start + qi]
        frame = source_frame(start_frame, target_start + qi)
        target_rgb = video.get(client_media.media_id, frame)
        points, token, normals = mesh.surface_tokens(camera, tans)
        marked = memory_mask[qi].ravel()[token]
        points, token, normals = points[marked], token[marked], normals[marked]
        target_players = player_rows(tick_tables, frame, fps)
        hidden = behind_players(points, camera[:3, 3], target_players, viewer_slot=client_slot)
        uv, _ = project_points(points, camera, tans)
        surface = TargetSurface(
            points=points,
            frame=np.zeros(len(points), np.int64),
            token=token,
            normals=normals,
            unseen=~hidden & rgb_quality_mask(target_rgb, uv),
            cameras=camera[None],
            tans=tans[None],
        )
        for si in range(BLOCK):
            seen = coverage(
                mesh,
                surface,
                source_c2w[si],
                source_tans[si],
                players=players[si],
                viewer_slot=block.slot,
            )
            uv, _ = project_points(points, source_c2w[si], source_tans[si])
            seen &= rgb_quality_mask(images[si], uv)
            match = matching_patches(
                target_rgb,
                images[si],
                points,
                normals,
                camera,
                source_c2w[si],
                tans,
                source_tans[si],
            )
            out[qi].ravel()[token] |= seen & match
    return out
