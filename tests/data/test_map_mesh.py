"""The map geometry: rays through the token grid and the occlusion test."""

import numpy as np
import pytest
import torch

from worldcast.data.camera import as_float64
from worldcast.data.map_mesh import EPS_U, MapMesh, MeshLibrary, visible_from


class _Wall:
    """An intersector whose every ray hits a surface 50 u from its origin."""

    def intersects_location(self, origins, directions, multiple_hits=False):
        return origins + 50.0 * directions, np.arange(len(origins)), None


def test_a_point_is_visible_up_to_the_first_surface():
    eye = np.zeros(3)
    points = np.array([[30.0, 0.0, 0.0], [0.0, 49.0, 0.0], [0.0, 0.0, 100.0], [1.0, 0.0, 0.0]])
    # in front of the wall; on it, within the slack; behind it; too near the eye to be a surface
    assert visible_from(_Wall(), points, eye, EPS_U).tolist() == [True, True, False, False]
    assert visible_from(_Wall(), np.zeros((0, 3)), eye, EPS_U).shape == (0,)
    assert as_float64(torch.ones(2, dtype=torch.float32)).dtype == np.float64


@pytest.fixture(scope="module")
def room() -> MapMesh:
    """A closed box of 400 u around the origin."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("trimesh.ray.ray_pyembree")
    return MapMesh(trimesh.creation.box(extents=(400.0, 400.0, 400.0)))


def test_one_ray_per_token_hits_the_room(room):
    points, tokens, normals = room.surface_tokens(np.eye(4), (4.0 / 3.0, 0.75))
    assert tokens.tolist() == list(range(12 * 21))  # row-major, every ray hits a wall
    assert points.shape == normals.shape == (252, 3)
    np.testing.assert_allclose(np.abs(points).max(1), 200.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)
    assert (points[:, 2] > 0).all()  # in front of the camera


def test_walls_occlude(room):
    inside, outside = [[150.0, 0.0, 0.0]], [[300.0, 0.0, 0.0]]
    assert room.visible(inside, np.zeros(3)).tolist() == [True]
    assert room.visible(outside, np.zeros(3)).tolist() == [False]


def test_a_library_serves_the_maps_it_was_given(room):
    library = MeshLibrary({"de_test": room})
    assert library.get("de_test") is room
    with pytest.raises(KeyError, match="collision mesh"):
        library.get("de_other")
