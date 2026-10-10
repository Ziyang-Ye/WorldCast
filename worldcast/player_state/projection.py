"""Players in the client's camera: every player projected onto a grid over the frame (Sec. 3.2).

The camera is the recordings' (:mod:`worldcast.data.camera`): horizontal field of view 106.26
degrees, vertical 73.74 degrees, eye 64 u above the engine origin. Depth is the forward distance
from the eye in engine units (u). The field projects onto the generator's 12 x 21 token grid: ``u``
in ``[0, 21]`` left to right, ``v`` in ``[0, 12]`` top to bottom, token centres at ``+0.5``; the
foreground weight of the loss projects onto the 24 x 42 latent grid; a player's box in a view
(:func:`player_boxes`) onto the frame's 672 x 384 pixels.

A player's projected radius is half its body height (72 u standing, 54 u crouched, linear in the
latent frame's crouch fraction) over its depth, in rows of the grid, clamped to ``[0.5, half the
rows]``. The only geometric gate is ``depth > 1``; the frustum and occlusion come from the
visibility labels (``worldcast.player_state.visibility``).
"""

import math
from dataclasses import dataclass

import torch

from worldcast.data.camera import NEAR_U, camera_axes, half_angle_tangents
from worldcast.data.controls import CONTROL_BUTTONS
from worldcast.data.game import EYE_HEIGHT
from worldcast.data.latents import FRAME_SIZE
from worldcast.data.recordings import ALIVE_INDEX

from .states import (
    CONT_PITCH_INDEX,
    CONT_YAW_INDEX,
    camera_turns,
    has_continuous_columns,
    held_fraction,
)
from .visibility import live_eligibility

__all__ = [
    "BODY_HEIGHT",
    "BODY_HEIGHT_CROUCHED",
    "CROUCH_COLUMN",
    "PLAYER_WIDTH",
    "RADIUS_MIN_CELLS",
    "VFOV_DEGREES",
    "PlayerBoxes",
    "Projection",
    "angles_from_controls",
    "body_height",
    "player_boxes",
    "project_players",
    "project_view",
    "table_angles",
]

#: Vertical field of view the field projects with, degrees (16:9 Hor+; the horizontal one is the
#: recordings' ``HFOV_DEGREES``). As trained: the rounded 73.74, not the ``tan(hfov / 2) * 9 / 16``
#: of the rays and of retrieval (:func:`worldcast.data.camera.half_angle_tangents`); the two
#: tangents differ by 5.6e-6.
VFOV_DEGREES = 73.74
#: Body height standing and crouched, u.
BODY_HEIGHT = 72.0
BODY_HEIGHT_CROUCHED = 54.0
#: Column of the crouch button (``duck``) in the substep buttons (5); its held share sets the body
#: height.
CROUCH_COLUMN = CONTROL_BUTTONS.index("duck")
#: Smallest projected radius, rows of the grid it is projected on; the largest is half the grid's
#: height (6 on the 12 x 21 token grid).
RADIUS_MIN_CELLS = 0.5
#: A player's width, u (the engine's player hull): its box is as wide at the player's depth.
PLAYER_WIDTH = 32.0


@dataclass(frozen=True)
class Projection:
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


@dataclass(frozen=True)
class PlayerBoxes:
    """Every player's box in a client's view, per frame (:func:`player_boxes`).

    Attributes:
        boxes (torch.Tensor): ``[F, P, 4]`` float32 ``x0, y0, x1, y1``, pixels of the 672 x 384
            frame.
        shown (torch.Tensor): ``[F, P]`` bool, another living player in front of the camera: the
            field's gate (:func:`~worldcast.player_state.visibility.live_eligibility`) with every
            player visible.
        in_view (torch.Tensor): ``[F, P]`` bool, the field's gate with the client's labels: shown
            and visible.
    """

    boxes: torch.Tensor
    shown: torch.Tensor
    in_view: torch.Tensor


# ------------------------------------------------------------------------------------------- angles
def angles_from_controls(
    initial_state: torch.Tensor, controls: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Yaw and pitch of every player per latent frame, integrated from frame 0 of the table.

    The frame-0 angles plus the turns of the latent frames ``1..f``
    (:func:`~worldcast.player_state.states.camera_turns`), without yaw wrap and without pitch clamp:
    as trained, a 6-wide table integrates the turns of its own rows, where the continuous columns
    of a 13-wide table follow the engine
    (:func:`~worldcast.player_state.states.integrate_camera_angles`).

    Args:
        initial_state (torch.Tensor): ``[B, P, 6]`` (yaw, pitch in degrees at columns 3, 4).
        controls (torch.Tensor): ``[B, F, P, Q, A + 1]`` packed substeps.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(yaw, pitch)``, each ``[B, F, P]``, degrees.
    """
    turn_pitch, turn_yaw = camera_turns(controls, frame_dim=1)
    yaw = initial_state[:, None, :, 3] + turn_yaw.cumsum(dim=1)
    pitch = initial_state[:, None, :, 4] + turn_pitch.cumsum(dim=1)
    return yaw, pitch


def table_angles(states: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(yaw, pitch)`` ``[B, F, P]`` degrees for every row of a player state table.

    A 13-wide table reads its continuous columns 6, 7; a 6-wide one integrates from frame 0
    (:func:`angles_from_controls`), starting at the row-0 engine angles.
    """
    if has_continuous_columns(states):
        return states[..., CONT_YAW_INDEX].float(), states[..., CONT_PITCH_INDEX].float()
    positions = states[..., :3].float()
    return angles_from_controls(torch.cat([positions[:, 0], states[:, 0, :, 3:]], dim=-1), controls)


# --------------------------------------------------------------------------------------- projection
def body_height(controls: torch.Tensor) -> torch.Tensor:
    """``[B, F, P]`` body height, u: 72 standing, 54 crouched, linear in the crouch fraction of the
    packed substeps ``[B, F, P, Q, A + 1]``."""
    crouch = held_fraction(controls, CROUCH_COLUMN)
    return BODY_HEIGHT + crouch * (BODY_HEIGHT_CROUCHED - BODY_HEIGHT)


def project_players(
    xyz: torch.Tensor,
    client_xyz: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    height: torch.Tensor | float = BODY_HEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project every player into the client's camera, in the units of a grid over the frame.

    Args:
        xyz (torch.Tensor): ``[B, F, P, 3]`` player positions (feet), u.
        client_xyz (torch.Tensor): ``[B, F, 3]`` the client's position (feet), u.
        yaw (torch.Tensor): ``[B, F]`` the client's yaw, degrees.
        pitch (torch.Tensor): ``[B, F]`` the client's pitch, degrees.
        grid_h (int): rows of the grid: 12 for the field's token grid (24 for the latent grid of
            the foreground weight).
        grid_w (int): its columns: 21 (42).
        height (torch.Tensor | float): body height, u, ``[B, F, P]`` or a scalar.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ``uv``, ``radius``,
        ``in_front`` and ``depth`` as in :class:`Projection`, in units of the grid.
    """
    forward, right, up = camera_axes(yaw, pitch)
    eye = client_xyz.float()
    eye = eye + torch.tensor([0.0, 0.0, EYE_HEIGHT], device=eye.device, dtype=eye.dtype)
    delta = xyz.float() - eye[:, :, None, :]
    depth = (delta * forward[:, :, None, :]).sum(-1)
    safe = depth.clamp(min=NEAR_U)

    tan_h = half_angle_tangents()[0]
    tan_v = math.tan(math.radians(VFOV_DEGREES / 2.0))
    x = (delta * right[:, :, None, :]).sum(-1) / safe
    y = (delta * up[:, :, None, :]).sum(-1) / safe
    u = (x / tan_h * 0.5 + 0.5) * float(grid_w)
    v = (0.5 - y / tan_v * 0.5) * float(grid_h)

    radius = (height * 0.5) / (safe * tan_v) * float(grid_h) * 0.5
    radius = radius.clamp(max=float(grid_h) * 0.5)
    uv = torch.stack([u, v], dim=-1)
    return uv, radius.clamp(min=RADIUS_MIN_CELLS), depth > NEAR_U, depth


def project_view(
    states: torch.Tensor,
    controls: torch.Tensor,
    client_slot: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
) -> Projection:
    """Project a player state table into the camera of the client's row of it.

    Args:
        states (torch.Tensor): ``[B, F, P, 6 | 13]`` player state table
            (:mod:`worldcast.player_state.states`). A 13-column table carries the client's yaw and
            pitch per latent frame (columns 6, 7); a 6-column table gives them at frame 0 only, and
            columns 3, 4 of its later frames are not read (:func:`table_angles`).
        controls (torch.Tensor): ``[B, F, P, Q, A + 1]`` packed substeps (crouch fraction -> body
            height; the turns a 6-column table's camera is integrated from).
        client_slot (torch.Tensor): ``[B]`` the client's slot.
        grid_h (int): token rows.
        grid_w (int): token columns.

    Returns:
        Projection: every player in the client's camera.
    """
    positions = states[..., :3].float()
    num_frames = int(positions.shape[1])
    yaw, pitch = table_angles(states, controls)
    slot = client_slot.to(positions.device).long()
    pick = slot.view(-1, 1, 1, 1).expand(-1, num_frames, 1, 3)
    client_xyz = positions.gather(2, pick).squeeze(2)
    flat = slot.view(-1, 1, 1).expand(-1, num_frames, 1)
    client_yaw = yaw.gather(2, flat).squeeze(-1)
    client_pitch = pitch.gather(2, flat).squeeze(-1)
    uv, radius, in_front, depth = project_players(
        positions,
        client_xyz,
        client_yaw,
        client_pitch,
        grid_h=grid_h,
        grid_w=grid_w,
        height=body_height(controls),
    )
    relative_yaw = torch.deg2rad(yaw - client_yaw[..., None])
    return Projection(
        uv=uv, radius=radius, in_front=in_front, depth=depth, relative_yaw=relative_yaw
    )


def player_boxes(
    states: torch.Tensor, controls: torch.Tensor, visible: torch.Tensor, client_slot: int
) -> PlayerBoxes:
    """Every player's box in a client's view, gated as the field gates the players it writes.

    The players are projected into the client's camera, its own row of ``states``, as the field
    projects them (:func:`project_players`), onto the frame's pixels: a box's bottom centre is the
    player's feet, its height the body's (:func:`body_height`), its width :data:`PLAYER_WIDTH` at
    that scale.

    Args:
        states (torch.Tensor): ``[F, P, 6]`` every player's state per frame (u, degrees; the
            client's yaw and pitch are its camera).
        controls (torch.Tensor): ``[F, P, Q, A + 1]`` their packed substeps (the crouch sets the
            body height).
        visible (torch.Tensor): ``[F, P]`` bool, the client's visibility labels (visible and
            valid).
        client_slot (int): the client's slot.

    Returns:
        PlayerBoxes: the boxes, and which of them the gate passes.
    """
    rows = states[None].float()
    client = rows[:, :, client_slot]
    height = body_height(controls[None].float())
    frame_h, frame_w = FRAME_SIZE
    uv, radius, in_front, _ = project_players(
        rows[..., :3],
        client[..., :3],
        client[..., 3],
        client[..., 4],
        grid_h=frame_h,
        grid_w=frame_w,
        height=height,
    )
    u, v = uv[0].unbind(-1)
    half_width = radius[0] * PLAYER_WIDTH / height[0]
    boxes = torch.stack([u - half_width, v - 2.0 * radius[0], u + half_width, v], dim=-1)
    alive, slot = rows[..., ALIVE_INDEX], torch.tensor([client_slot])
    shown = live_eligibility(
        in_front, alive=alive, visible=torch.ones_like(in_front), client_slot=slot
    )
    gated = live_eligibility(in_front, alive=alive, visible=visible[None], client_slot=slot)
    return PlayerBoxes(boxes=boxes, shown=shown.eligible[0], in_view=gated.eligible[0])
