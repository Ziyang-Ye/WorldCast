"""The recordings' camera: poses as cameras, rays and the field of view."""

import numpy as np
import pytest
import torch

from tests.data.support import unscoped_signals
from worldcast.data.camera import (
    as_float64,
    c2w_from_state_rows,
    camera_rays,
    camera_tans,
    grid_rays,
    half_angle_tangents,
    hfov_from_source_zoom,
    to_camera,
    window_cameras,
)
from worldcast.data.controls import OPENCS2_WEAPONS


def test_a_pose_looking_along_x_is_a_right_handed_camera():
    c2w = c2w_from_state_rows(np.array([[10.0, 20.0, 30.0, 0.0, 0.0, 1.0]]))[0]
    right, down, forward, eye = c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3]
    assert torch.allclose(forward, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(right, torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)
    assert torch.allclose(down, torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert eye.tolist() == [10.0, 20.0, 94.0]  # 64 u above the feet


def test_the_unscoped_field_of_view_is_16_by_9():
    tan_h, tan_v = half_angle_tangents()
    assert tan_h == pytest.approx(4.0 / 3.0, abs=1e-3) and tan_v == pytest.approx(0.75, abs=1e-3)
    with pytest.raises(ValueError):
        half_angle_tangents(180.0)


def test_rays_go_through_the_cell_centres():
    rays = camera_rays((2.0, 1.0), (2, 4))
    assert rays.shape == (8, 3) and (rays[:, 2] == 1.0).all()
    assert (
        rays[:4, 0].tolist() == [-1.5, -0.5, 0.5, 1.5]
        and rays[:, 1].tolist() == [-0.5] * 4 + [0.5] * 4
    )
    assert np.array_equal(grid_rays(np.eye(4), (2.0, 1.0), (2, 4)), rays)
    # a camera that looks along the world's +x: its forward axis (column 2) is +x
    c2w = c2w_from_state_rows(np.array([[10.0, 20.0, 30.0, 0.0, 0.0, 1.0]]))[0]
    world = grid_rays(as_float64(c2w), (2.0, 1.0), (2, 4))
    assert world[:, 0].tolist() == pytest.approx([1.0] * 8)  # depth along a ray is axial
    assert world[0].tolist() == pytest.approx([1.0, 1.5, 0.5])  # top left: +y is left, +z is up


def test_world_points_in_camera_space():
    c2w = as_float64(c2w_from_state_rows(np.array([[10.0, 20.0, 30.0, 0.0, 0.0, 1.0]]))[0])
    assert c2w.dtype == np.float64
    # 5 u ahead of the eye (64 u above the feet), 2 u to its left, 1 u above it
    point = np.array([[15.0, 22.0, 95.0]])
    assert to_camera(point, c2w)[0].tolist() == pytest.approx([-2.0, -1.0, 5.0])


def test_a_scoped_latent_frame_has_the_zoom_field_of_view():
    signals = unscoped_signals(3)
    signals["obs_scope_on"][1:] = 1
    signals["obs_scope_level"][1:] = [1, 2]
    weapons = np.full(9, OPENCS2_WEAPONS.index("awp"))
    tans = camera_tans(signals, weapons, range(3))
    assert tans.dtype == np.float32 and tans.shape == (3, 2)
    np.testing.assert_allclose(tans[0], half_angle_tangents(), rtol=1e-6)
    for row, zoom in zip(tans[1:], (40.0, 10.0)):
        np.testing.assert_allclose(row, half_angle_tangents(hfov_from_source_zoom(zoom)), rtol=1e-6)
    weapons[:] = OPENCS2_WEAPONS.index("ak47")  # no zoom entry: unscoped
    np.testing.assert_allclose(camera_tans(signals, weapons, [2])[0], tans[0])


def test_window_cameras_are_read_at_the_latent_frames_last_video_frames():
    states = np.zeros((9, 6), np.float32)
    states[:, 0] = np.arange(9)
    c2w, tans = window_cameras(states, unscoped_signals(3), np.zeros(9, np.int64), 3)
    assert c2w.shape == (3, 4, 4) and c2w[:, 0, 3].tolist() == [0.0, 4.0, 8.0]
    np.testing.assert_allclose(tans, np.tile(half_angle_tangents(), (3, 1)), rtol=1e-6)
