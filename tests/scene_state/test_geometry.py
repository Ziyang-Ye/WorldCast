"""Scene-state geometry: rays, projection, z-buffers and coverage on the depth grid."""

import numpy as np
import pytest

from tests.scene_state import support as sw
from worldcast.data.camera import half_angle_tangents
from worldcast.scene_state import geometry as g


def test_the_depth_tolerance_is_the_larger_of_24_u_and_5_percent():
    assert (g.TOLERANCE_ABS_U, g.TOLERANCE_REL) == (24.0, 0.05)
    assert g.tolerance([100.0, 480.0, 1000.0]).tolist() == [24.0, 24.0, 50.0]


def test_one_field_of_view_serves_every_camera():
    shared = g.per_camera_tans(sw.TAN, 4)
    assert shared.shape == (4, 2) and shared.dtype == np.float64
    assert (shared == np.asarray(sw.TAN)).all()
    own = np.arange(8.0).reshape(4, 2)
    assert (g.per_camera_tans(own, 4) == own).all()
    with pytest.raises(ValueError, match="2 fields of view for 4 cameras"):
        g.per_camera_tans(own[:2], 4)


def test_rays_have_unit_forward_component():
    assert g.NPIX == 24 * 42
    eye, rays = g.pixel_rays(sw.cam([10.0, 20.0, 30.0], 0.3), sw.TAN)
    assert eye.tolist() == [10.0, 20.0, 30.0] and rays.shape == (g.NPIX, 3)
    np.testing.assert_allclose(rays, sw.rays(sw.cam([10.0, 20.0, 30.0], 0.3)), atol=1e-12)
    forward = sw.rot(0.3)[:, 2]
    np.testing.assert_allclose(rays @ forward, 1.0, atol=1e-12)  # depth along a ray is axial
    # the top-left pixel centre of an unrotated camera: (-1 + 1 / 42) tan_h, (-1 + 1 / 24) tan_v
    _, straight = g.pixel_rays(np.eye(4), (2.0, 1.0))
    assert straight[0].tolist() == pytest.approx([-2.0 + 2.0 / 42.0, -1.0 + 1.0 / 24.0, 1.0])


def test_back_projection_and_projection_are_inverse():
    camera = sw.cam([0.0, 0.0, -300.0], 0.2)
    depth = sw.room_depth(camera)
    points, hit = g.view_points(camera[None], sw.TAN, depth[None])
    assert points.shape == (1, g.NPIX, 3) and hit.shape == (1, g.NPIX)
    assert 0 < hit.sum() < g.NPIX  # the rays through the open roof miss
    pix, z, ok = g.project(points[0][hit[0]], camera, sw.TAN)
    assert ok.all() and (pix == np.flatnonzero(hit[0])).all()
    np.testing.assert_allclose(z, depth.reshape(-1)[hit[0]], rtol=1e-9)
    # a point behind the camera, and one outside the field of view, do not land
    stray = camera[:3, 3] + np.array([[0.0, 0.0, -50.0], [5000.0, 0.0, 10.0]]) @ camera[:3, :3].T
    assert not g.project(stray, camera, sw.TAN)[2].any()


def test_a_depth_at_the_far_plane_or_without_a_value_is_no_surface():
    depth = np.full((1, 24, 42), 100.0)
    depth.reshape(-1)[:5] = [0.0, -5.0, np.nan, np.inf, 0.95 * 4096.0]
    depth.reshape(-1)[5] = np.nextafter(0.95 * 4096.0, 0.0)
    _, hit = g.view_points(np.eye(4)[None], sw.TAN, depth)
    assert hit[0, :7].tolist() == [False, False, False, False, False, True, True]


def test_a_point_lands_in_front_of_the_near_plane_and_inside_the_field_of_view():
    tan = np.array([1.0, 0.5])
    points = np.array(
        [
            [0.0, 0.0, 1.0],  # on the near plane
            [0.0, 0.0, 1.0000001],
            [10.0, 0.0, 10.0],  # on the right edge
            [10.0000001, 0.0, 10.0],  # beyond it
            [0.0, -5.0, 10.0],  # on the upper edge
            [0.0, 0.0, -10.0],  # behind the camera
        ]
    )
    pix, z, ok = g.project(points, np.eye(4), tan)
    assert ok.tolist() == [False, True, True, False, True, False]
    assert pix.tolist() == [12 * 42 + 21, 12 * 42 + 41, 21] and z.tolist() == [
        1.0000001,
        10.0,
        10.0,
    ]


def test_points_are_projected_with_the_inverse_of_the_cameras_rotation():
    """As in the paper's runs. A recorded camera's rotation is built in float32, so it is not
    exactly orthonormal and ``inv(R).T`` is not ``R``: this point, 2e-9 of the image width left of
    the border between the columns 0 and 1, falls into column 1 when rotated by ``R``."""
    camera = np.array(
        [
            [0.6018149852752686, -0.1523868441581726, 0.7839623689651489, 120.0],
            [-0.7986355423927307, -0.11483171582221985, 0.5907579660415649, -340.0],
            [0.0, -0.9816271662712097, -0.1908089965581894, 124.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    point = np.array([[135.10663741045266, -90.11519692970623, 173.60177420495208]])
    pix, z, ok = g.project(point, camera, half_angle_tangents())
    assert ok.all() and pix.tolist() == [3 * 42 + 0] and z[0] == pytest.approx(150.0)


def test_a_z_buffer_keeps_the_nearest_point_per_pixel_and_per_group():
    camera = np.eye(4)
    points = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 50.0], [0.0, 0.0, 200.0]])
    centre = 12 * 42 + 21
    buf = g.zbuffer(points, camera, sw.TAN)
    assert buf.shape == (g.NPIX,) and buf[centre] == 50.0
    assert np.isinf(np.delete(buf, centre)).all()
    grouped = g.zbuffer(points, camera, sw.TAN, group=np.array([0, 1, 0]), n_groups=3)
    assert grouped.shape == (3, g.NPIX)
    assert grouped[:, centre].tolist() == [100.0, 50.0, np.inf]
    assert np.isinf(g.zbuffer(np.zeros((0, 3)), camera, sw.TAN)).all()


def test_coverage_needs_a_surface_a_point_and_a_depth_within_the_tolerance():
    depth = np.array([100.0, 100.0, 1000.0, 1000.0, 100.0, 100.0, 100.0])
    zbuf = np.array([110.0, 130.0, 1049.0, 1051.0, np.inf, 100.0, 124.0])
    hit = np.array([True, True, True, True, True, False, True])
    assert g.covered(zbuf, depth, hit).tolist() == [True, False, True, False, False, False, True]


def test_a_memory_entry_stores_the_depth_of_a_latent_frames_last_video_frame():
    log_depth = np.log(np.arange(1.0, 5.0)).reshape(1, 4, 1, 1) * np.ones((2, 4, 24, 42))
    depth = g.axial_from_log_depth(log_depth.astype(np.float32))
    assert depth.shape == (2, 24, 42) and depth.dtype == np.float64
    np.testing.assert_allclose(depth, 4.0, rtol=1e-7)
    with pytest.raises(ValueError):
        g.axial_from_log_depth(np.zeros((2, 4, 12, 21)))
