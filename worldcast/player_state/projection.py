"""The client's camera: engine poses as cameras, players on the token grid, fields of view.

Camera model: horizontal field of view 106.26 degrees, vertical 73.74 degrees (16:9 Hor+ framing),
eye 64 u above the engine origin (the ground between the feet; crouch ignored). World frame: z up,
yaw about +z from +x, pitch positive looking down, so the view direction is ``(cos yaw cos pitch,
sin yaw cos pitch, -sin pitch)``; camera space is x right, y down, z forward. Depth is the forward
distance from the eye in engine units (u). Token coordinates are on the generator's 12 x 21 grid:
``u`` in ``[0, 21]`` left to right, ``v`` in ``[0, 12]`` top to bottom, token centres at ``+0.5``.

A player's projected radius is half its body height (72 u standing, 54 u crouched, linear in the
latent's crouch fraction) over its depth, in token rows, clamped to ``[0.5, 6]``. The only geometric
gate is ``depth > 1``; the frustum and occlusion come from the visibility labels
(``worldcast.player_state.visibility``).
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import torch

from .attributes import held_fraction
from .tables import (
    CAMERA_DELTA_SCALE,
    CONT_PITCH_INDEX,
    CONT_YAW_INDEX,
    SUBSTEP_PITCH_INDEX,
    SUBSTEP_VALID_INDEX,
    SUBSTEP_YAW_INDEX,
    has_continuous_columns,
)

__all__ = [
    "HFOV_DEGREES",
    "VFOV_DEGREES",
    "EYE_HEIGHT",
    "BODY_TOP",
    "BODY_TOP_DUCKED",
    "RADIUS_MIN_TOKENS",
    "ASPECT_TAN_RATIO",
    "SCOPE_ZOOM_FOV",
    "Projection",
    "camera_axes",
    "camera_to_world",
    "c2w_from_state_rows",
    "angles_from_actions",
    "table_angles",
    "body_height",
    "project_players",
    "project_view",
    "half_angle_tangents",
    "hfov_from_source_zoom",
    "scoped_query_tans",
    "camera_tans",
]

HFOV_DEGREES = 106.26
VFOV_DEGREES = 73.74
#: Camera height above the player origin (standing eye), u (config ``data.eye_height``).
EYE_HEIGHT = 64.0
#: Body height standing and crouched, u.
BODY_TOP = 72.0
BODY_TOP_DUCKED = 54.0
#: Smallest projected radius; the largest is half the grid height (6 tokens on 12 x 21).
RADIUS_MIN_TOKENS = 0.5
#: ``tan(vfov / 2) = tan(hfov / 2) * 9 / 16`` for the cameras of the rays and of retrieval.
ASPECT_TAN_RATIO = 9.0 / 16.0
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


class Projection(NamedTuple):
    """Every player projected into the client's camera, per latent frame.

    Attributes:
        uv (torch.Tensor): ``[B, F, P, 2]`` float32 feet position, token units (``u`` column, ``v``
            row).
        radius (torch.Tensor): ``[B, F, P]`` float32 projected half body height, token rows.
        in_front (torch.Tensor): ``[B, F, P]`` bool, ``depth > 1``.
        depth (torch.Tensor): ``[B, F, P]`` float32 forward distance from the eye, u (negative
            behind the camera).
        relative_yaw (torch.Tensor): ``[B, F, P]`` float32 player yaw minus the client's yaw,
            radians.
    """

    uv: torch.Tensor
    radius: torch.Tensor
    in_front: torch.Tensor
    depth: torch.Tensor
    relative_yaw: torch.Tensor


# ------------------------------------------------------------------------------------------ cameras
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
    xyz: torch.Tensor, yaw_degrees: torch.Tensor, pitch_degrees: torch.Tensor, *, eye_height: float
) -> torch.Tensor:
    """Camera-to-world matrices of engine poses (camera x right, y down, z forward).

    Args:
        xyz (torch.Tensor): ``[F, 3]`` player origin (feet), u; the camera sits ``eye_height``
            above.
        yaw_degrees (torch.Tensor): ``[F]`` yaw, degrees.
        pitch_degrees (torch.Tensor): ``[F]`` pitch, degrees.
        eye_height (float): camera height above the feet, u.

    Returns:
        torch.Tensor: ``[F, 4, 4]`` in ``xyz``'s dtype.
    """
    forward, right, up = camera_axes(yaw_degrees, pitch_degrees)
    c2w = torch.zeros(xyz.shape[0], 4, 4, dtype=xyz.dtype, device=xyz.device)
    c2w[:, :3, 0] = right
    c2w[:, :3, 1] = -up
    c2w[:, :3, 2] = forward
    c2w[:, :3, 3] = xyz + torch.tensor(
        [0.0, 0.0, float(eye_height)], dtype=xyz.dtype, device=xyz.device
    )
    c2w[:, 3, 3] = 1.0
    return c2w


def c2w_from_state_rows(state_rows: np.ndarray, *, eye_height: float = EYE_HEIGHT) -> torch.Tensor:
    """``[F, 6]`` player-state rows (``x y z yaw pitch alive``) -> ``[F, 4, 4]`` float32 cameras."""
    rows = torch.from_numpy(np.asarray(state_rows, dtype=np.float32))
    return camera_to_world(rows[:, :3], rows[:, 3], rows[:, 4], eye_height=eye_height)


# ------------------------------------------------------------------------------------------- angles
def angles_from_actions(
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    *,
    camera_delta_scale: float = CAMERA_DELTA_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-latent yaw and pitch of every player, integrated from frame 0 of a contiguous window.

    Frame 0's deltas describe the interval before the window and are dropped. No yaw wrap and no
    pitch clamp: this is the 6-wide table of the plain prefix, exact on a contiguous window.

    Args:
        initial_state (torch.Tensor): ``[B, P, 6]`` (yaw, pitch in degrees at columns 3, 4).
        actions (torch.Tensor): ``[B, F, P, Q, A + 1]`` packed substeps.
        camera_delta_scale (float): degrees per unit delta.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(yaw, pitch)``, each ``[B, F, P]``, degrees.
    """
    valid = (actions[..., SUBSTEP_VALID_INDEX] > 0.5).to(actions.dtype)
    d_yaw = (actions[..., SUBSTEP_YAW_INDEX] * valid).sum(-1) * camera_delta_scale
    d_pitch = (actions[..., SUBSTEP_PITCH_INDEX] * valid).sum(-1) * camera_delta_scale
    d_yaw = torch.cat([torch.zeros_like(d_yaw[:, :1]), d_yaw[:, 1:]], dim=1)
    d_pitch = torch.cat([torch.zeros_like(d_pitch[:, :1]), d_pitch[:, 1:]], dim=1)
    yaw = initial_state[:, None, :, 3] + d_yaw.cumsum(dim=1)
    pitch = initial_state[:, None, :, 4] + d_pitch.cumsum(dim=1)
    return yaw, pitch


def table_angles(
    states: torch.Tensor, actions: torch.Tensor, *, camera_delta_scale: float = CAMERA_DELTA_SCALE
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(yaw, pitch)`` ``[B, F, P]`` degrees for every row of a peer state table.

    A 13-wide (gathered) table reads its continuous columns 6, 7; a 6-wide (contiguous) one
    integrates from frame 0 (:func:`angles_from_actions`), starting at the row-0 engine angles.
    """
    if has_continuous_columns(states):
        return states[..., CONT_YAW_INDEX].float(), states[..., CONT_PITCH_INDEX].float()
    positions = states[..., :3].float()
    return angles_from_actions(
        torch.cat([positions[:, 0], states[:, 0, :, 3:]], dim=-1),
        actions,
        camera_delta_scale=camera_delta_scale,
    )


# --------------------------------------------------------------------------------------- projection
def body_height(actions: torch.Tensor, duck_button_index: int) -> torch.Tensor:
    """``[B, F, P]`` body height, u: 72 standing, 54 crouched, linear in the crouch fraction."""
    crouch = held_fraction(actions, duck_button_index)
    return BODY_TOP + crouch * (BODY_TOP_DUCKED - BODY_TOP)


def project_players(
    xyz: torch.Tensor,
    observer_xyz: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    height: torch.Tensor | float = BODY_TOP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project every player into the client's token grid.

    Args:
        xyz (torch.Tensor): ``[B, F, P, 3]`` player positions (feet), u.
        observer_xyz (torch.Tensor): ``[B, F, 3]`` the client's position (feet), u.
        yaw (torch.Tensor): ``[B, F]`` the client's yaw, degrees.
        pitch (torch.Tensor): ``[B, F]`` the client's pitch, degrees.
        grid_h (int): token rows (12).
        grid_w (int): token columns (21).
        height (torch.Tensor | float): body height, u, ``[B, F, P]`` or a scalar.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ``uv``, ``radius``,
        ``in_front`` and ``depth`` as in :class:`Projection`.
    """
    forward, right, up = camera_axes(yaw, pitch)
    eye = observer_xyz.float().clone()
    eye = eye + torch.tensor([0.0, 0.0, EYE_HEIGHT], device=eye.device, dtype=eye.dtype)
    delta = xyz.float() - eye[:, :, None, :]
    depth = (delta * forward[:, :, None, :]).sum(-1)
    safe = depth.clamp(min=1.0)

    tan_h = math.tan(math.radians(HFOV_DEGREES / 2.0))
    tan_v = math.tan(math.radians(VFOV_DEGREES / 2.0))
    x = (delta * right[:, :, None, :]).sum(-1) / safe
    y = (delta * up[:, :, None, :]).sum(-1) / safe
    u = (x / tan_h * 0.5 + 0.5) * float(grid_w)
    v = (0.5 - y / tan_v * 0.5) * float(grid_h)

    radius = (height * 0.5) / (safe * tan_v) * float(grid_h) * 0.5
    radius = radius.clamp(max=float(grid_h) * 0.5)
    uv = torch.stack([u, v], dim=-1)
    return uv, radius.clamp(min=RADIUS_MIN_TOKENS), depth > 1.0, depth


def project_view(
    states: torch.Tensor,
    actions: torch.Tensor,
    observer_slot: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    duck_button_index: int,
    camera_delta_scale: float = CAMERA_DELTA_SCALE,
) -> Projection:
    """Project a peer state table into the camera of the client's own row of it.

    Args:
        states (torch.Tensor): ``[B, F, P, 6 | 13]`` peer state table
            (``worldcast.player_state.tables``).
        actions (torch.Tensor): ``[B, F, P, Q, A + 1]`` packed substeps (crouch fraction -> body
            height; the 6-wide camera integral).
        observer_slot (torch.Tensor): ``[B]`` the client's seat.
        grid_h (int): token rows.
        grid_w (int): token columns.
        duck_button_index (int): crouch column (5).
        camera_delta_scale (float): degrees per unit delta (6-wide tables only).

    Returns:
        Projection: every player in the client's camera.
    """
    positions = states[..., :3].float()
    num_frames = int(positions.shape[1])
    yaw, pitch = table_angles(states, actions, camera_delta_scale=camera_delta_scale)
    slot = observer_slot.to(positions.device).long()
    pick = slot.view(-1, 1, 1, 1).expand(-1, num_frames, 1, 3)
    observer_xyz = positions.gather(2, pick).squeeze(2)
    flat = slot.view(-1, 1, 1).expand(-1, num_frames, 1)
    observer_yaw = yaw.gather(2, flat).squeeze(-1)
    observer_pitch = pitch.gather(2, flat).squeeze(-1)
    uv, radius, in_front, depth = project_players(
        positions,
        observer_xyz,
        observer_yaw,
        observer_pitch,
        grid_h=grid_h,
        grid_w=grid_w,
        height=body_height(actions, duck_button_index),
    )
    relative_yaw = torch.deg2rad(yaw - observer_yaw[..., None])
    return Projection(
        uv=uv, radius=radius, in_front=in_front, depth=depth, relative_yaw=relative_yaw
    )


# ------------------------------------------------------------------------------------ field of view
def half_angle_tangents(
    hfov_degrees: float = HFOV_DEGREES, aspect_tan_ratio: float = ASPECT_TAN_RATIO
) -> tuple[float, float]:
    """``(tan(hfov / 2), tan(hfov / 2) * aspect_tan_ratio)``: ``(1.3333, 0.75)`` at the defaults."""
    if not (0.0 < float(hfov_degrees) < 180.0):
        raise ValueError("hfov_degrees must be in (0, 180)")
    tan_h = math.tan(math.radians(float(hfov_degrees) / 2.0))
    return tan_h, tan_h * float(aspect_tan_ratio)


def hfov_from_source_zoom(zoom_fov_4_3: float) -> float:
    """The engine's zoom field of view (4:3 horizontal, degrees) as this data's 16:9 Hor+ one."""
    half = math.radians(float(zoom_fov_4_3) / 2.0)
    return 2.0 * math.degrees(math.atan(math.tan(half) * 4.0 / 3.0))


def scoped_query_tans(
    obs_rows: Mapping[str, Sequence[int]],
    weapon_ids: Sequence[int],
    latent_indices: Iterable[int],
    *,
    tan_h: float,
    tan_v: float,
    weapon_names: Sequence[str],
) -> tuple[list[tuple[float, float]], int]:
    """Per latent frame the camera's ``(tan_h, tan_v)``, narrowed while the player is scoped.

    A latent frame is scoped iff its scope signal is valid, on and at level >= 1, and the weapon
    held at its pixel row (``4 k``) has a zoom entry; it then uses the zoom field of view of that
    level.

    Args:
        obs_rows (Mapping[str, Sequence[int]]): per-latent observer signals; reads ``obs_scope_on``,
            ``obs_scope_level`` and ``obs_scope_valid``.
        weapon_ids (Sequence[int]): the player's weapon id per pixel row.
        latent_indices (Iterable[int]): the latent frames to report.
        tan_h (float): unscoped ``tan(hfov / 2)`` (:func:`half_angle_tangents`).
        tan_v (float): unscoped ``tan(vfov / 2)``.
        weapon_names (Sequence[str]): the 52-way weapon vocabulary.

    Returns:
        tuple[list[tuple[float, float]], int]: ``(tan_h, tan_v)`` per latent frame, and how many are
        scoped.
    """
    on = np.asarray(obs_rows.get("obs_scope_on", []), dtype=np.int64)
    lev = np.asarray(obs_rows.get("obs_scope_level", []), dtype=np.int64)
    val = np.asarray(obs_rows.get("obs_scope_valid", []), dtype=np.int64)
    wid = np.asarray(weapon_ids, dtype=np.int64).reshape(-1)
    out, n_scoped = [], 0
    for k in latent_indices:
        k = int(k)
        th, tv = float(tan_h), float(tan_v)
        if (
            k < len(on)
            and k < len(val)
            and val[k] > 0
            and on[k] > 0
            and k < len(lev)
            and lev[k] >= 1
        ):
            row = 4 * k
            name = (
                weapon_names[int(wid[row])]
                if 0 <= row < len(wid) and 0 <= int(wid[row]) < len(weapon_names)
                else None
            )
            zooms = SCOPE_ZOOM_FOV.get(name)
            if zooms:
                z = zooms[min(int(lev[k]), len(zooms)) - 1]
                th = math.tan(math.radians(hfov_from_source_zoom(z) / 2.0))
                tv = th * ASPECT_TAN_RATIO
                n_scoped += 1
        out.append((th, tv))
    return out, n_scoped


def camera_tans(
    obs: Mapping[str, Sequence[int]],
    weapon_ids: Sequence[int],
    indices: Iterable[int],
    hfov: float = HFOV_DEGREES,
    *,
    weapon_names: Sequence[str],
) -> np.ndarray:
    """``[len(indices), 2]`` float32 ``(tan_h, tan_v)`` per latent frame, narrowed while scoped."""
    tan_h, tan_v = half_angle_tangents(hfov)
    tans, _ = scoped_query_tans(
        obs, weapon_ids, indices, tan_h=tan_h, tan_v=tan_v, weapon_names=weapon_names
    )
    return np.asarray(tans, np.float32)
