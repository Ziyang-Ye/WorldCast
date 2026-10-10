"""The mask m_k of a training window (App. "Scene state in detail", Training).

m_k marks the token positions of the target frames whose surface the memory frames show and the
context does not. It is measured on the map geometry (the collision mesh), the recorded cameras and
the recorded video:

* **unseen surface** (:func:`unseen_surface`): the mesh surface the four target frames show, one ray
  per token of the 12 x 21 grid through the frame's field of view, that the context has not seen
  (z-tested from the first frame and, under block-causal attention, from every video frame of the
  recent context), minus rays that pass a living player (:func:`behind_players`) and points whose
  target RGB patch is untextured or under the HUD (:func:`rgb_quality_mask`);
* **coverage** (:func:`coverage`): the unseen points a memory frame sees, z-tested, at most 420 u
  away, at half the target's resolution or better, not grazing, not behind a living player;
* **m_k** (:func:`token_mask`): the covered unseen token-times of the target block, ``[4, 12, 21]``
  bool.

The projections are float64 numpy with the camera rotation applied as ``(p - eye) R``
(:func:`~worldcast.data.camera.to_camera`), as the training windows were selected.

Optional dependency, imported when used: OpenCV (the RGB statistics).
"""

from dataclasses import dataclass

import numpy as np

from .camera import as_array, as_float64, to_camera
from .latents import TOKEN_GRID
from .map_mesh import MapMesh
from .recordings import ALIVE_INDEX

__all__ = [
    "MAX_DISTANCE_U",
    "MIN_INCIDENCE_COSINE",
    "MIN_PATCH_STD",
    "MIN_RESOLUTION_RATIO",
    "PLAYER_CENTRE_RAISE_U",
    "PLAYER_RADIUS_U",
    "UNCLIPPED_GREY",
    "TargetSurface",
    "behind_players",
    "coverage",
    "off_hud",
    "project_points",
    "rgb_quality_mask",
    "seen_mask",
    "token_mask",
    "unseen_surface",
]

#: A memory frame sees a point at no less than this share of the target frame's linear resolution.
MIN_RESOLUTION_RATIO = 0.5
#: A frame sees a surface at an incidence cosine of at least this (not grazing).
MIN_INCIDENCE_COSINE = 0.1
#: Texture threshold: the standard deviation of the 9 x 9 grey patch around a point.
MIN_PATCH_STD = 0.015
#: Grey levels (in [0, 1]) outside this open range are clipped.
UNCLIPPED_GREY = (0.025, 0.975)
#: A memory frame's eye lies at most this far from the point, u.
MAX_DISTANCE_U = 420.0
#: A living player is a sphere of this radius around its position raised by 36 u, u.
PLAYER_RADIUS_U = 42.0
PLAYER_CENTRE_RAISE_U = 36.0


def seen_mask(
    mesh: MapMesh, points: np.ndarray, c2w: np.ndarray, tan_h: float, tan_v: float
) -> np.ndarray:
    """``[M]`` bool: a frame with camera ``c2w`` (``[4, 4]``) shows the world points ``[M, 3]`` (in
    its frustum and visible)."""
    c = np.asarray(c2w, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    camera = to_camera(points, c)
    x, y, z = camera[:, 0], camera[:, 1], camera[:, 2]
    inside = (z > 0) & (np.abs(x) < z * tan_h) & (np.abs(y) < z * tan_v)
    out = np.zeros(len(points), dtype=bool)
    if inside.any():
        out[inside] = mesh.visible(points[inside], c[:3, 3])
    return out


@dataclass
class TargetSurface:
    """The mesh surface the target frames show.

    Attributes:
        points (np.ndarray): ``[M, 3]`` float64 first hits of the token rays.
        frame (np.ndarray): ``[M]`` int64 target frame of each point.
        token (np.ndarray): ``[M]`` int64 token id within its frame.
        normals (np.ndarray): ``[M, 3]`` normals of the hit faces.
        unseen (np.ndarray): ``[M]`` bool, not seen by the context (and still eligible).
        cameras (np.ndarray): ``[F, 4, 4]`` float64 target cameras.
        tans (np.ndarray): ``[F, 2]`` float64 their ``(tan_h, tan_v)``.
    """

    points: np.ndarray
    frame: np.ndarray
    token: np.ndarray
    normals: np.ndarray
    unseen: np.ndarray
    cameras: np.ndarray
    tans: np.ndarray


def unseen_surface(
    mesh: MapMesh,
    target_c2w: np.ndarray,
    target_tans: np.ndarray,
    context_c2w: np.ndarray,
    context_tans: np.ndarray,
    first_c2w: np.ndarray,
    first_tans: np.ndarray,
) -> TargetSurface:
    """The surface the target frames show, and which of it the context has not seen.

    Args:
        mesh (MapMesh): the map.
        target_c2w (np.ndarray): ``[4, 4, 4]`` target cameras (arrays or tensors, here and below).
        target_tans (np.ndarray): ``[4, 2]`` their ``(tan_h, tan_v)``.
        context_c2w (np.ndarray): ``[R, 4, 4]`` cameras of the recent context (``R = 0``: the first
            frame alone).
        context_tans (np.ndarray): ``[R, 2]``.
        first_c2w (np.ndarray): ``[4, 4]`` the first frame's camera.
        first_tans (np.ndarray): ``[2]`` its ``(tan_h, tan_v)``.

    Returns:
        TargetSurface: before the player and RGB filters.
    """
    cameras, context = as_float64(target_c2w), as_float64(context_c2w)
    tans, context_tans = as_float64(target_tans), as_float64(context_tans)
    hits = [mesh.surface_tokens(camera, tans[i]) for i, camera in enumerate(cameras)]
    points = np.concatenate([h[0] for h in hits])
    seen = seen_mask(mesh, points, as_float64(first_c2w), *as_float64(first_tans))
    for camera, tan in zip(context, context_tans):
        seen |= seen_mask(mesh, points, camera, *tan)
    return TargetSurface(
        points=points,
        frame=np.concatenate([np.full(len(h[0]), i, np.int64) for i, h in enumerate(hits)]),
        token=np.concatenate([h[1] for h in hits]),
        normals=np.concatenate([h[2] for h in hits]),
        unseen=~seen,
        cameras=cameras,
        tans=tans,
    )


def project_points(
    points: np.ndarray, c2w: np.ndarray, tans: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World points ``[M, 3]`` in the image of the camera ``c2w`` (``[4, 4]``, ``tans = (tan_h,
    tan_v)``): ``uv`` in [0, 1] (``[M, 2]``) and axial depth (``[M]``)."""
    camera, tans = to_camera(as_float64(points), as_float64(c2w)), as_float64(tans)
    z = camera[:, 2]
    uv = np.stack(
        [
            camera[:, 0] / np.maximum(z, 1e-8) / tans[0],
            camera[:, 1] / np.maximum(z, 1e-8) / tans[1],
        ],
        1,
    )
    return uv * 0.5 + 0.5, z


def behind_players(
    points: np.ndarray, eye: np.ndarray, players: np.ndarray, *, viewer_slot: int
) -> np.ndarray:
    """``[M]`` bool: the segment from ``eye`` to the point passes within :data:`PLAYER_RADIUS_U`
    of a living player other than the viewer, short of the point.

    Args:
        points (np.ndarray): ``[M, 3]`` world points.
        eye (np.ndarray): ``[3]`` the viewer.
        players (np.ndarray): ``[P, 6]`` every player's ``x, y, z, yaw, pitch, alive``
            (:func:`~worldcast.data.recordings.player_rows`).
        viewer_slot (int): the viewer's slot.
    """
    out = np.zeros(len(points), bool)
    d = as_float64(points) - as_float64(eye)
    dd = (d * d).sum(1)
    for slot, state in enumerate(np.asarray(players)):
        if slot == viewer_slot or state[ALIVE_INDEX] <= 0:
            continue
        centre = state[:3] + np.array([0.0, 0.0, PLAYER_CENTRE_RAISE_U])
        t = np.clip(((centre - eye) * d).sum(1) / np.maximum(dd, 1e-9), 0, 1)
        out |= (((eye + t[:, None] * d - centre) ** 2).sum(1) < PLAYER_RADIUS_U**2) & (t < 0.999)
    return out


def coverage(
    mesh: MapMesh,
    surface: TargetSurface,
    c2w: np.ndarray,
    tans: np.ndarray,
    *,
    players: np.ndarray,
    viewer_slot: int,
) -> np.ndarray:
    """``[M]`` bool: the unseen points a source frame sees well enough to count.

    Visible (frustum and z-test), within :data:`MAX_DISTANCE_U`, projected area ratio ``(d_t /
    d_s)^2 (tan_t / tan_s) cos_s / max(cos_t, 0.05) >= MIN_RESOLUTION_RATIO^2``, incidence ``cos_s
    >= MIN_INCIDENCE_COSINE``, and not behind a living player (:func:`behind_players`).

    Args:
        mesh (MapMesh): the map.
        surface (TargetSurface): the target surface.
        c2w (np.ndarray): ``[4, 4]`` the source frame's camera.
        tans (np.ndarray): ``[2]`` its ``(tan_h, tan_v)``.
        players (np.ndarray): ``[P, 6]`` every player's state at the source frame.
        viewer_slot (int): the source frame's player.
    """
    # the tangents keep their dtype: their product below is float32 for the float32 tangents the
    # training windows were selected with
    points, c, tans = surface.points, as_float64(c2w), as_array(tans)
    covered = seen_mask(mesh, points, c, *tans) & surface.unseen
    source_ray = points - c[:3, 3]
    sd = np.linalg.norm(source_ray, axis=1)
    target_ray = points - surface.cameras[surface.frame, :3, 3]
    td = np.linalg.norm(target_ray, axis=1)
    s_cos = np.abs((surface.normals * source_ray / np.maximum(sd[:, None], 1e-9)).sum(1))
    t_cos = np.abs((surface.normals * target_ray / np.maximum(td[:, None], 1e-9)).sum(1))
    area = (
        (td / np.maximum(sd, 1e-9)) ** 2
        * surface.tans[surface.frame].prod(1)
        / np.prod(tans)
        * s_cos
        / np.maximum(t_cos, 0.05)
    )
    covered &= (
        (sd <= MAX_DISTANCE_U) & (area >= MIN_RESOLUTION_RATIO**2) & (s_cos >= MIN_INCIDENCE_COSINE)
    )
    return covered & ~behind_players(points, c[:3, 3], players, viewer_slot=viewer_slot)


def off_hud(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bool, like ``u``: the image point ``(u, v)`` in [0, 1] lies off the HUD and the border,
    ``u`` in (0.06, 0.94) and ``v`` in (0.12, 0.78)."""
    return (u > 0.06) & (u < 0.94) & (v > 0.12) & (v < 0.78)


def rgb_quality_mask(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """``[M]`` bool: the point's RGB patch can be measured: off the HUD and the border
    (:func:`off_hud`), not clipped (9 x 9 grey mean within :data:`UNCLIPPED_GREY`) and textured (9 x
    9 grey standard deviation >= :data:`MIN_PATCH_STD`).

    Args:
        image (np.ndarray): ``[H, W, 3]`` uint8 RGB.
        uv (np.ndarray): ``[M, 2]`` image coordinates in [0, 1].
    """
    import cv2

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean = cv2.boxFilter(gray, -1, (9, 9))
    second = cv2.boxFilter(gray * gray, -1, (9, 9))
    std = np.sqrt(np.maximum(second - mean * mean, 0))
    h, w = gray.shape
    uv = np.asarray(uv)
    x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    y = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
    low, high = UNCLIPPED_GREY
    return (
        off_hud(uv[:, 0], uv[:, 1])
        & (mean[y, x] > low)
        & (mean[y, x] < high)
        & (std[y, x] >= MIN_PATCH_STD)
    )


def token_mask(surface: TargetSurface, covered: np.ndarray) -> np.ndarray:
    """m_k on the target block's token-times, ``[4, 12, 21]`` bool: the tokens whose surface point
    is unseen and covered by the memory frames (``covered``: ``[M]`` bool per surface point)."""
    mask = np.zeros((len(surface.cameras), *TOKEN_GRID), bool)
    mask.reshape(len(mask), -1)[surface.frame, surface.token] = surface.unseen & covered
    return mask
