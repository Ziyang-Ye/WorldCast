"""The recordings' camera: engine poses as cameras, the field of view, rays and camera space.

Camera model: horizontal field of view 106.26 degrees (16:9 Hor+ framing), eye 64 u above the
engine origin (the ground between the feet; crouch ignored). World frame: z up, yaw about +z from
+x, pitch positive looking down, so the view direction is ``(cos yaw cos pitch, sin yaw cos pitch,
-sin pitch)``; camera space is x right, y down, z forward. A scoped weapon narrows the field of
view to its zoom level's.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

import numpy as np
import torch

from .controls import OPENCS2_WEAPONS
from .game import ASPECT_TAN_RATIO, EYE_HEIGHT, HFOV_DEGREES
from .latents import last_video_frame, last_video_frames

__all__ = [
    "NEAR_U",
    "SCOPE_ZOOM_FOV",
    "as_array",
    "as_float64",
    "c2w_from_state_rows",
    "camera_axes",
    "camera_rays",
    "camera_tans",
    "camera_to_world",
    "grid_rays",
    "half_angle_tangents",
    "hfov_from_source_zoom",
    "to_camera",
    "window_cameras",
]

#: A point projects into a camera only if it lies more than this far in front of it (axially), u.
NEAR_U = 1.0
#: Scoped weapons: the engine's zoom fields of view (4:3 horizontal, degrees) per zoom level.
SCOPE_ZOOM_FOV = MappingProxyType(
    {
        "awp": (40.0, 10.0),
        "ssg08": (40.0, 15.0),
        "g3sg1": (40.0, 15.0),
        "scar20": (40.0, 15.0),
        "aug": (55.0,),
        "sg556": (55.0,),
    }
)


def as_array(x: torch.Tensor | np.ndarray | Sequence) -> np.ndarray:
    """A tensor or an array as a numpy array of its own dtype."""
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)


def as_float64(x: torch.Tensor | np.ndarray | Sequence) -> np.ndarray:
    """A tensor or an array as a float64 numpy array, the dtype of the geometry."""
    return np.asarray(as_array(x), dtype=np.float64)


def camera_axes(
    yaw_degrees: torch.Tensor, pitch_degrees: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The engine's view axes ``(forward, right, up)``, each ``[..., 3]`` float32.

    ``forward = (cos y cos p, sin y cos p, -sin p)``, ``right = normalise(forward_y, -forward_x,
    0)``, ``up = right x forward``.
    """
    yaw = torch.deg2rad(yaw_degrees.float())
    pitch = torch.deg2rad(pitch_degrees.float())
    cp = torch.cos(pitch)
    forward = torch.stack([torch.cos(yaw) * cp, torch.sin(yaw) * cp, -torch.sin(pitch)], dim=-1)
    right = torch.stack(
        [forward[..., 1], -forward[..., 0], torch.zeros_like(forward[..., 0])], dim=-1
    )
    right = right / (right.norm(dim=-1, keepdim=True) + 1e-9)
    up = torch.cross(right, forward, dim=-1)
    return forward, right, up


def camera_to_world(
    xyz: torch.Tensor, yaw_degrees: torch.Tensor, pitch_degrees: torch.Tensor
) -> torch.Tensor:
    """Camera-to-world matrices of engine poses (camera x right, y down, z forward).

    Args:
        xyz (torch.Tensor): ``[F, 3]`` player origin (feet), u; the camera sits
            :data:`~worldcast.data.game.EYE_HEIGHT` above.
        yaw_degrees (torch.Tensor): ``[F]`` yaw, degrees.
        pitch_degrees (torch.Tensor): ``[F]`` pitch, degrees.

    Returns:
        torch.Tensor: ``[F, 4, 4]`` in ``xyz``'s dtype.
    """
    forward, right, up = camera_axes(yaw_degrees, pitch_degrees)
    c2w = torch.zeros(xyz.shape[0], 4, 4, dtype=xyz.dtype, device=xyz.device)
    c2w[:, :3, 0] = right
    c2w[:, :3, 1] = -up
    c2w[:, :3, 2] = forward
    c2w[:, :3, 3] = xyz + torch.tensor([0.0, 0.0, EYE_HEIGHT], dtype=xyz.dtype, device=xyz.device)
    c2w[:, 3, 3] = 1.0
    return c2w


def c2w_from_state_rows(state_rows: np.ndarray) -> torch.Tensor:
    """``[F, 6]`` player-state rows (``x y z yaw pitch alive``) -> ``[F, 4, 4]`` float32 cameras."""
    rows = torch.from_numpy(np.asarray(state_rows, dtype=np.float32))
    return camera_to_world(rows[:, :3], rows[:, 3], rows[:, 4])


def camera_rays(tans: Sequence[float], grid: tuple[int, int]) -> np.ndarray:
    """The viewing rays through the cell centres of an image grid, in camera space.

    Args:
        tans (Sequence[float]): ``(tan_h, tan_v)`` of the field of view.
        grid (tuple[int, int]): the image grid ``(h, w)``.

    Returns:
        np.ndarray: ``[h w, 3]`` float64, row-major: ``[x tan_h, y tan_v, 1]`` at ``x = (j + .5) *
        2 / w - 1``, ``y = (i + .5) * 2 / h - 1``; the forward component is 1, so a depth along a
        ray is axial.
    """
    h, w = grid
    xx, yy = np.meshgrid((np.arange(w) + 0.5) * 2 / w - 1, (np.arange(h) + 0.5) * 2 / h - 1)
    return np.stack([xx.ravel() * tans[0], yy.ravel() * tans[1], np.ones(h * w)], 1)


def grid_rays(c2w: np.ndarray, tans: Sequence[float], grid: tuple[int, int]) -> np.ndarray:
    """:func:`camera_rays` of the camera ``c2w`` (``[4, 4]`` float64) in the world, ``[h w, 3]``
    float64."""
    return camera_rays(tans, grid) @ c2w[:3, :3].T


def to_camera(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """World points ``[..., 3]`` in the camera space of ``c2w`` ``[4, 4]``: ``(p - eye) R``, float64
    for float64 inputs."""
    return (points - c2w[:3, 3]) @ c2w[:3, :3]


def half_angle_tangents(hfov_degrees: float = HFOV_DEGREES) -> tuple[float, float]:
    """``(tan(hfov / 2), tan(vfov / 2))`` of a 16:9 camera: ``(1.3333, 0.75)`` when unscoped."""
    if not (0.0 < float(hfov_degrees) < 180.0):
        raise ValueError("hfov_degrees must be in (0, 180)")
    tan_h = math.tan(math.radians(float(hfov_degrees) / 2.0))
    return tan_h, tan_h * ASPECT_TAN_RATIO


def hfov_from_source_zoom(zoom_fov_4_3: float) -> float:
    """The engine's zoom field of view (4:3 horizontal, degrees) as this data's 16:9 Hor+ one."""
    half = math.radians(float(zoom_fov_4_3) / 2.0)
    return 2.0 * math.degrees(math.atan(math.tan(half) * 4.0 / 3.0))


def camera_tans(
    signals: Mapping[str, Sequence[int]], weapon_ids: Sequence[int], latent_frames: Iterable[int]
) -> np.ndarray:
    """The field of view of a player's camera at latent frames, narrowed while it is scoped.

    A latent frame is scoped iff its scope signal is valid, on and at level >= 1 and the weapon held
    at its last video frame has a zoom entry; it then has the zoom field of view of that level.

    Args:
        signals (Mapping[str, Sequence[int]]): the player's per-latent observer signals
            (``obs_scope_on``, ``obs_scope_level``, ``obs_scope_valid``).
        weapon_ids (Sequence[int]): the player's weapon id per video frame.
        latent_frames (Iterable[int]): the latent frames to report.

    Returns:
        np.ndarray: ``[n, 2]`` float32 ``(tan(hfov / 2), tan(vfov / 2))``.
    """
    on, level, valid = (
        np.asarray(signals[key], dtype=np.int64)
        for key in ("obs_scope_on", "obs_scope_level", "obs_scope_valid")
    )
    weapons = np.asarray(weapon_ids, dtype=np.int64).reshape(-1)
    unscoped = half_angle_tangents()
    tans = []
    for k in latent_frames:
        zooms = None
        if valid[k] > 0 and on[k] > 0 and level[k] >= 1:
            zooms = SCOPE_ZOOM_FOV.get(OPENCS2_WEAPONS[weapons[last_video_frame(k)]])
        if zooms:
            zoom = zooms[min(int(level[k]), len(zooms)) - 1]
            tans.append(half_angle_tangents(hfov_from_source_zoom(zoom)))
        else:
            tans.append(unscoped)
    return np.asarray(tans, np.float32)


def window_cameras(
    states: np.ndarray,
    signals: Mapping[str, Sequence[int]],
    weapon_ids: Sequence[int],
    latent_frames: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """A player's recorded cameras over a window, at its latent frames.

    Args:
        states (np.ndarray): ``[T, 6]`` the player's state per video frame.
        signals (Mapping[str, Sequence[int]]): its per-latent observer signals.
        weapon_ids (Sequence[int]): its weapon id per video frame.
        latent_frames (int): latent frames ``F`` of the window.

    Returns:
        tuple[torch.Tensor, np.ndarray]: ``c2w`` ``[F, 4, 4]`` float32 and ``(tan(hfov / 2),
        tan(vfov / 2))`` ``[F, 2]`` float32.
    """
    c2w = c2w_from_state_rows(np.asarray(states)[last_video_frames(0, latent_frames)])
    return c2w, camera_tans(signals, weapon_ids, range(latent_frames))
