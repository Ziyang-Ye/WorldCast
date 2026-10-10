"""Players in the client's camera."""

import math

import pytest
import torch

from worldcast.data.game import EYE_HEIGHT
from worldcast.player_state.projection import (
    BODY_HEIGHT,
    BODY_HEIGHT_CROUCHED,
    CROUCH_COLUMN,
    VFOV_DEGREES,
    angles_from_controls,
    body_height,
    player_boxes,
    project_players,
    project_view,
    table_angles,
)
from worldcast.player_state.states import pack_substeps

GRID = dict(grid_h=12, grid_w=21)


def _project(xyz, *, yaw=0.0, pitch=0.0, **kwargs):
    """One player at ``xyz`` seen by a client at the origin looking along ``yaw`` / ``pitch``."""
    grid = {**GRID, **kwargs}
    return project_players(
        torch.tensor([[[xyz]]]),
        torch.zeros(1, 1, 3),
        torch.tensor([[yaw]]),
        torch.tensor([[pitch]]),
        **grid,
    )


def test_a_player_straight_ahead_lands_on_the_grid_centre():
    uv, radius, in_front, depth = _project([500.0, 0.0, EYE_HEIGHT])
    assert uv[0, 0, 0].tolist() == pytest.approx([10.5, 6.0])
    assert depth.item() == pytest.approx(500.0) and bool(in_front.item())
    # half the body height over the depth, in token rows
    tan_v = math.tan(math.radians(VFOV_DEGREES / 2.0))
    assert radius.item() == pytest.approx(0.5 * BODY_HEIGHT / (500.0 * tan_v) * 6.0)
    assert radius.item() == pytest.approx(0.576, abs=1e-3)
    # on the latent grid of the foreground weight the same player is twice as large
    uv, radius, _, _ = _project([500.0, 0.0, EYE_HEIGHT], grid_h=24, grid_w=42)
    assert uv[0, 0, 0].tolist() == pytest.approx([21.0, 12.0])
    assert radius.item() == pytest.approx(2 * 0.576, abs=2e-3)


def test_left_is_left_and_up_is_up():
    uv, _, _, _ = _project([500.0, 100.0, EYE_HEIGHT + 50.0])  # engine y points to the left
    # tan(hfov / 2) = 1.3333 and tan(vfov / 2) = 0.75: 0.2 and 0.1 of the depth off the axis
    assert uv[0, 0, 0].tolist() == pytest.approx([10.5 - 0.15 * 10.5, 6.0 - 0.1333 * 6.0], abs=1e-3)
    uv, _, _, _ = _project([0.0, 500.0, EYE_HEIGHT], yaw=90.0)
    assert uv[0, 0, 0].tolist() == pytest.approx([10.5, 6.0], abs=1e-4)
    uv, _, _, _ = _project([500.0, 0.0, EYE_HEIGHT - 500.0], pitch=45.0)  # pitch looks down
    assert uv[0, 0, 0].tolist() == pytest.approx([10.5, 6.0], abs=1e-4)


def test_the_radius_is_clamped_and_a_player_behind_is_not_in_front():
    _, radius, in_front, depth = _project([-500.0, 0.0, 0.0])
    assert depth.item() == -500.0 and not bool(in_front.item())
    assert _project([5.0, 0.0, 0.0])[1].item() == 6.0  # half the grid's rows
    assert _project([1e5, 0.0, 0.0])[1].item() == 0.5
    crouched = _project([100.0, 0.0, 0.0], height=BODY_HEIGHT_CROUCHED)[1].item()
    assert crouched == pytest.approx(_project([100.0, 0.0, 0.0])[1].item() * 0.75)
    assert not bool(_project([1.0, 0.0, EYE_HEIGHT])[2].item())  # at the near plane: not in front


def test_body_height_follows_the_crouch_fraction():
    controls = torch.zeros(1, 1, 1, 16, 14)
    controls[..., 13] = 1.0
    assert body_height(controls).item() == BODY_HEIGHT == 72.0
    controls[..., :8, CROUCH_COLUMN] = 1.0
    assert body_height(controls).item() == (72.0 + 54.0) / 2.0


def _turning_controls() -> torch.Tensor:
    """Packed substeps ``[1, 4, 1, 16, 14]`` of one player over four latent frames: 10 degrees of
    yaw and -5 of pitch in frame 1, 25 of pitch in frame 2, 180 of yaw in frame 3."""
    controls = torch.zeros(1, 4, 1, 16, 14)
    controls[..., 13] = 1.0
    controls[0, 1, 0, 0, 12], controls[0, 1, 0, 5, 11] = 2.0, -1.0  # units of 5 degrees
    controls[0, 2, 0, 3, 11] = 5.0
    controls[0, 3, 0, :, 12] = 36.0 / 16.0
    controls[0, 0, 0, :, 11:13] = 9.0  # the turns of frame 0 precede the table
    return controls


def test_a_six_column_table_integrates_its_angles_from_frame_zero():
    initial = torch.tensor([[[0.0, 0.0, 0.0, 170.0, 70.0, 1.0]]])
    yaw, pitch = angles_from_controls(initial, _turning_controls())
    # as trained: the yaw is not wrapped and the pitch is not clamped
    assert yaw[0, :, 0].tolist() == pytest.approx([170.0, 180.0, 180.0, 360.0])
    assert pitch[0, :, 0].tolist() == pytest.approx([70.0, 65.0, 90.0, 90.0])
    # the table's own yaw and pitch columns are read at frame 0 only
    table = torch.zeros(1, 4, 1, 6)
    table[0, :, 0, 3], table[0, :, 0, 4] = torch.tensor([170.0, 1.0, 2.0, 3.0]), 70.0
    again = table_angles(table, _turning_controls())
    assert torch.equal(again[0], yaw) and torch.equal(again[1], pitch)


def test_a_thirteen_column_table_carries_its_angles():
    table = torch.zeros(1, 4, 1, 13)
    table[0, :, 0, 6] = torch.tensor([10.0, 20.0, 30.0, 40.0])  # continuous yaw
    table[0, :, 0, 7] = torch.tensor([1.0, 2.0, 3.0, 4.0])  # continuous pitch
    yaw, pitch = table_angles(table, _turning_controls())
    assert yaw[0, :, 0].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert pitch[0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError):
        table_angles(torch.zeros(1, 4, 1, 9), _turning_controls())


def test_project_view_looks_through_the_clients_own_row():
    # the client (slot 1) stands at the origin; slot 0 stands 500 u along +y, slot 2 along -y
    table = torch.zeros(1, 2, 3, 13)
    table[0, :, 0, :3] = torch.tensor([0.0, 500.0, 0.0])
    table[0, :, 2, :3] = torch.tensor([0.0, -500.0, 0.0])
    table[0, :, :, 5] = 1.0
    table[0, :, 1, 6] = torch.tensor([90.0, -90.0])  # the client turns from +y to -y
    table[0, :, 0, 6] = 30.0  # slot 0 looks along 30 degrees, slot 2 along 0
    controls = torch.zeros(1, 2, 3, 16, 14)
    controls[..., 13] = 1.0
    view = project_view(table, controls, torch.tensor([1]), **GRID)
    assert view.uv.shape == (1, 2, 3, 2) and view.radius.shape == (1, 2, 3)
    assert view.uv[0, 0, 0].tolist() == pytest.approx([10.5, 6.0 + 0.576 * 64.0 / 36.0], abs=1e-3)
    assert view.depth[0, :, 0].tolist() == pytest.approx([500.0, -500.0])
    assert view.depth[0, :, 2].tolist() == pytest.approx([-500.0, 500.0])
    assert view.in_front[0].tolist() == [[True, False, False], [False, False, True]]
    assert view.depth[0, :, 1].tolist() == [0.0, 0.0]  # the client itself
    assert view.relative_yaw[0, 0].tolist() == pytest.approx(
        [math.radians(30.0 - 90.0), 0.0, math.radians(0.0 - 90.0)]
    )


def test_a_box_stands_on_the_feet_and_the_gate_picks_the_bold_ones():
    # the client (slot 0) at the origin; slot 1 500 u ahead, slot 2 250 u to its right and
    # crouched, slot 3 behind it, slot 4 ahead but dead
    states = torch.zeros(2, 5, 6)
    states[..., 5] = 1.0
    states[:, 1, :3] = torch.tensor([500.0, 0.0, 0.0])
    states[:, 2, :3] = torch.tensor([500.0, -250.0, 0.0])
    states[:, 3, :3] = torch.tensor([-500.0, 0.0, 0.0])
    states[:, 4, :3] = torch.tensor([500.0, 100.0, 0.0])
    states[:, 4, 5] = 0.0
    states[1, 0, 3] = 90.0  # at frame 1 the client faces +y, where slot 1 then stands
    states[1, 1, :3] = torch.tensor([0.0, 500.0, 0.0])
    substeps = torch.zeros(2, 5, 4, 13)
    substeps[:, 2, :, CROUCH_COLUMN] = 1.0
    controls = pack_substeps(substeps, torch.ones(2, 5, 4, dtype=torch.bool))
    # the labels have slot 1 in view at frame 0 only, slot 2 never; the others' labels say
    # visible, which the gate overrules
    visible = torch.tensor([[True, True, False, True, True], [True, False, False, True, True]])
    view = player_boxes(states, controls, visible, 0)
    # 500 u ahead on a level camera, eye 64 u up: feet 32.8 px below the centre, 72 u = 36.9 px
    # tall and 32 u = 16.4 px wide
    assert view.boxes[0, 1].tolist() == pytest.approx(
        [327.808, 187.904, 344.192, 224.768], abs=1e-3
    )
    # 250 u to the right, crouched: 54 u tall, as wide
    assert view.boxes[0, 2].tolist() == pytest.approx(
        [453.809, 197.120, 470.192, 224.768], abs=1e-3
    )
    assert view.boxes[1, 1].tolist() == pytest.approx(view.boxes[0, 1].tolist(), abs=1e-3)
    # shown: neither the client, nor one behind it (slot 3; slot 2 once the client turns), nor the
    # dead; bold: shown and labelled
    assert view.shown.tolist() == [
        [False, True, True, False, False],
        [False, True, False, False, False],
    ]
    assert view.in_view.tolist() == [
        [False, True, False, False, False],
        [False, False, False, False, False],
    ]
