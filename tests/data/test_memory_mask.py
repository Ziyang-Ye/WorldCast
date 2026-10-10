"""The mask m_k of the memory frames: the unseen surface, its coverage and the RGB quality."""

import numpy as np
import pytest

from tests.data.synthetic_round import room_mesh
from worldcast.data import memory_mask as M
from worldcast.data.camera import c2w_from_state_rows, half_angle_tangents

TANS = np.asarray(half_angle_tangents())
NOBODY = np.zeros((10, 6))


class _Clear:
    """A map without walls: every point is visible from everywhere."""

    def visible(self, points, eye):
        return np.ones(len(points), bool)


def _camera(yaw: float) -> np.ndarray:
    """A camera at the room's centre looking along ``yaw`` degrees."""
    return c2w_from_state_rows(np.array([[0.0, 0.0, -64.0, yaw, 0.0, 1.0]]))[0].numpy()


def test_the_unseen_surface_is_what_neither_the_first_frame_nor_the_context_shows():
    pytest.importorskip("trimesh.ray.ray_pyembree")
    room = room_mesh()
    behind, ahead = _camera(180.0), _camera(0.0)
    target = dict(target_c2w=np.stack([behind] * 4), target_tans=np.stack([TANS] * 4))
    first = dict(first_c2w=ahead, first_tans=TANS)
    no_context = dict(context_c2w=np.zeros((0, 4, 4)), context_tans=np.zeros((0, 2)))
    surface = M.unseen_surface(room, **target, **no_context, **first)
    # one point per token of each of the four target frames, none of them shown by the first frame
    assert surface.points.shape == (4 * 252, 3) and surface.unseen.all()
    assert surface.frame.tolist() == [q for q in range(4) for _ in range(252)]
    assert surface.token.tolist() == list(range(252)) * 4
    assert (surface.points[:, 0] < 0).all()  # the half of the room behind the first frame
    # a context frame that looked the same way has seen all of it
    seen = M.unseen_surface(
        room, **target, context_c2w=behind[None], context_tans=TANS[None], **first
    )
    assert not seen.unseen.any()
    # ... and the first frame alone, when it looks there
    assert not M.unseen_surface(
        room, **target, **no_context, first_c2w=behind, first_tans=TANS
    ).unseen.any()


def test_a_living_player_hides_the_points_behind_it():
    eye = np.zeros(3)
    points = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [20.0, 0.0, 0.0]])
    players = np.zeros((10, 6))
    players[3] = [50.0, 0.0, -36.0, 0.0, 0.0, 1.0]  # its centre, 36 u above the feet, at (50, 0, 0)
    # behind the player; 50 u off the line of sight; in front of the player
    assert M.behind_players(points, eye, players, viewer_slot=0).tolist() == [True, False, False]
    assert not M.behind_players(points, eye, players, viewer_slot=3).any()  # the viewer itself
    players[3, 5] = 0.0
    assert not M.behind_players(points, eye, players, viewer_slot=0).any()  # a dead player


def _surface(points: list[list[float]]) -> M.TargetSurface:
    """Points on a wall facing a target camera at the origin."""
    return M.TargetSurface(
        points=np.array(points, float),
        frame=np.zeros(len(points), int),
        token=np.arange(len(points)),
        normals=np.tile([0.0, 0.0, -1.0], (len(points), 1)),
        unseen=np.ones(len(points), bool),
        cameras=np.eye(4)[None],
        tans=np.array([[1.0, 1.0]]),
    )


def test_a_source_frame_covers_what_it_sees_near_and_sharp_enough():
    surface = _surface([[0, 0, 10], [8, 0, 10]])
    here = np.eye(4)
    covers = dict(players=NOBODY, viewer_slot=0)
    assert M.coverage(_Clear(), surface, here, [1, 1], **covers).tolist() == [True, True]
    # a scoped source frame does not show the point off its axis
    assert M.coverage(_Clear(), surface, here, [0.25, 0.25], **covers).tolist() == [True, False]
    # from 100 u further back the surface is at less than half the target's resolution
    back = np.eye(4)
    back[2, 3] = -100
    assert not M.coverage(_Clear(), surface, back, [1, 1], **covers).any()
    # beyond 420 u a source frame does not count
    far = _surface([[0, 0, 430], [0, 0, 410]])
    assert M.coverage(_Clear(), far, here, [1, 1], **covers).tolist() == [False, True]
    # a point the context has seen is not covered, and neither is one behind a living player
    seen = _surface([[0, 0, 10], [8, 0, 10]])
    seen.unseen[0] = False
    assert M.coverage(_Clear(), seen, here, [1, 1], **covers).tolist() == [False, True]
    players = np.zeros((10, 6))
    players[4] = [0.0, 0.0, 100.0 - 36.0, 0.0, 0.0, 1.0]  # centred on the axis, 100 u ahead
    hidden = M.coverage(_Clear(), far, here, [1, 1], players=players, viewer_slot=0)
    assert hidden.tolist() == [False, False]
    assert M.coverage(_Clear(), far, here, [1, 1], players=players, viewer_slot=4).tolist() == [
        False,
        True,
    ]


def test_m_k_marks_the_unseen_and_covered_tokens_of_each_target_frame():
    surface = _surface([[0, 0, 10]] * 4)
    surface.frame = np.array([0, 0, 1, 3])
    surface.token = np.array([0, 22, 251, 5])
    surface.cameras = np.stack([np.eye(4)] * 4)
    surface.unseen = np.array([True, True, True, False])
    mask = M.token_mask(surface, np.array([True, False, True, True]))
    assert mask.shape == (4, 12, 21) and mask.dtype == bool
    assert np.argwhere(mask).tolist() == [[0, 0, 0], [1, 11, 20]]


def test_points_project_to_image_coordinates():
    uv, depth = M.project_points(
        np.array([[0.0, 0.0, 10.0], [5.0, -2.5, 10.0]]), np.eye(4), [1, 0.5]
    )
    assert uv.tolist() == [[0.5, 0.5], [0.75, 0.25]] and depth.tolist() == [10.0, 10.0]


def test_rgb_quality_needs_decoded_texture():
    pytest.importorskip("cv2")
    uv = np.array([[0.5, 0.5], [0.5, 0.95]])
    assert not M.rgb_quality_mask(np.full((100, 100, 3), 255, np.uint8), uv).any()
    textured = np.random.default_rng(7).integers(20, 220, (100, 100, 3), dtype=np.uint8)
    assert M.rgb_quality_mask(textured, uv).tolist() == [True, False]  # the second is on the HUD
    assert M.off_hud(np.array([0.5, 0.05, 0.5]), np.array([0.5, 0.5, 0.8])).tolist() == [
        True,
        False,
        False,
    ]
