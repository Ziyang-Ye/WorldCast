"""The predicted visibility labels: the z-test against the depth head's depth."""

import numpy as np
import pytest
import torch

from tests.player_state.support import flat_depth, standing_round
from worldcast.data.camera import camera_to_world, half_angle_tangents
from worldcast.data.game import EYE_HEIGHT
from worldcast.player_state import PredictedVisibility, block_depth_frames
from worldcast.player_state.predicted_visibility import MARGIN_U, PLAYER_LIFT_U, ztest_labels

TANS = half_angle_tangents()
#: The client at the origin looking along +x; slot 1 stands 500 u in front of it, slot 2 behind.
POSITIONS = [[0.0, 0.0, 0.0], [500.0, 0.0, 0.0], [-500.0, 0.0, 0.0]]


def _camera():
    """A camera at the origin's eye height looking along +x."""
    return camera_to_world(torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1), torch.zeros(1))


def test_a_player_is_visible_unless_a_surface_is_in_front_of_it():
    assert MARGIN_U == 24.0
    points = torch.tensor([[[500.0, 0.0, EYE_HEIGHT], [-500.0, 0.0, EYE_HEIGHT]]]).double()
    live = torch.tensor([[True, True]])

    def labels(depth):
        grid = torch.full((1, 24, 42), depth, dtype=torch.float64)
        return [t[0].tolist() for t in ztest_labels(points, _camera(), grid, live=live, tans=TANS)]

    assert labels(1000.0) == [[True, False], [True, True]]  # behind the camera: known, not seen
    assert labels(476.5) == [[True, False], [True, True]]  # a surface within the 24 u margin
    assert labels(476.0) == [[False, False], [True, True]]  # at the margin: hidden
    assert labels(100.0) == [[False, False], [True, True]]  # a wall in front
    assert labels(float("nan")) == [[False, False], [False, True]]  # no surface known: unknown
    dead = ztest_labels(
        points, _camera(), torch.full((1, 24, 42), 1000.0), live=~live, tans=TANS, pose_radius=40.0
    )
    assert not bool(dead[0].any()) and not bool(dead[1].any())


def test_a_player_seen_at_one_of_its_eight_ring_points_is_visible_and_known():
    points = torch.tensor([[[500.0, 0.0, EYE_HEIGHT]]]).double()
    live = torch.tensor([[True]])
    # the depth is known everywhere but in the two columns at the image centre
    depth = torch.full((1, 24, 42), 2000.0, dtype=torch.float64)
    depth[:, :, 20:22] = float("nan")
    centre = ztest_labels(points, _camera(), depth, live=live, tans=TANS)
    assert (centre[0].item(), centre[1].item()) == (False, False)  # unknown
    ring = ztest_labels(points, _camera(), depth, live=live, tans=TANS, pose_radius=40.0)
    assert (ring[0].item(), ring[1].item()) == (True, True)  # 40 u aside is 1.26 columns
    # a wall 300 u away where the depth is known: seen at no point, the centre stays unknown
    depth[:, :, :20], depth[:, :, 22:] = 300.0, 300.0
    walled = ztest_labels(points, _camera(), depth, live=live, tans=TANS, pose_radius=40.0)
    assert (walled[0].item(), walled[1].item()) == (False, False)


def test_block_depth_frames_orders_the_video_frames_latent_frame_by_latent_frame():
    log_depth = np.log(np.arange(1, 17, dtype=np.float64)).reshape(4, 4, 1, 1)
    grid = np.broadcast_to(log_depth, (4, 4, 24, 42))
    frames = block_depth_frames(lambda x0: grid, torch.zeros(4, 48, 24, 42))
    assert frames.shape == (16, 24, 42) and frames.dtype == np.float64
    assert np.allclose(frames[:, 0, 0], np.arange(1, 17))


def test_predicted_visibility_of_the_first_six_blocks():
    batch = standing_round(POSITIONS)
    latents = torch.zeros(9, 48, 24, 42)
    far = PredictedVisibility(batch, depth_fn=flat_depth(2000.0))
    assert far.client_slot == 0 and far.points[0, 1].tolist() == [500.0, 0.0, PLAYER_LIFT_U]
    # the surface 2000 u away, back-projected from the generated latent frames, is behind slot 1
    assert far.target_visible(5, latents).tolist() == [[False, True, False]] * 4
    assert far.generated_visible(0, 1, latents).tolist() == [[False, True, False]]
    assert far.generated_visible(1, 4, latents).tolist() == [[False, True, False]] * 4
    wall = PredictedVisibility(batch, depth_fn=flat_depth(100.0))
    assert not bool(wall.target_visible(5, latents).any())
    assert not bool(wall.generated_visible(1, 4, latents).any())
    near = np.full((16, 24, 42), 100.0)
    assert not bool(far.retested_visible(5, near).any())
    assert far.retested_visible(5, near * 20).tolist() == [[False, True, False]] * 4
    with pytest.raises(ValueError, match="NaN or non-positive"):
        far.retested_visible(5, near * float("nan"))
    two = {key: torch.cat([value, value]) for key, value in batch.items()}
    with pytest.raises(ValueError, match="batch size 1"):
        PredictedVisibility(two, depth_fn=flat_depth(2000.0))


def test_predicted_labels_of_a_gathered_window():
    batch = standing_round(POSITIONS, latents=33)
    predicted = PredictedVisibility(batch, depth_fn=flat_depth(2000.0))
    latents = torch.zeros(33, 48, 24, 42)
    labelled = predicted.for_block(25, batch, latents)
    visible, valid = labelled["client_visibility"], labelled["client_visibility_valid"]
    assert visible.shape == valid.shape == (1, 3, 129)
    read = [0] + list(range(49, 113))  # the first frame, the recent context, the target frames
    assert bool(visible[0, 1, read].all()) and not bool(visible[0, 2].any())
    assert not bool(valid[0, :, 1:49].any()) and not bool(valid[0, :, 113:].any())  # unknown

    window = dict(
        client_visibility=visible[:, :, 32:113], client_visibility_valid=valid[:, :, 32:113]
    )
    again = predicted.relabel_target(window, 25, np.full((16, 24, 42), 100.0))
    assert not bool(again["client_visibility"][0, :, -16:].any())
    assert torch.equal(
        again["client_visibility"][..., :-16], window["client_visibility"][..., :-16]
    )

    batch["player_states"][0, 1, :, 0] = -500.0  # the table changed: slot 1 is now behind
    predicted.update_frames(np.arange(129))
    assert not bool(predicted.target_visible(5, latents).any())


def test_the_target_block_is_tested_against_the_whole_recent_context():
    """The client looks away during the last recent latent frame: the surface in front of the
    target block's cameras is known from the eleven recent latent frames before it."""
    batch = standing_round(POSITIONS, latents=33)
    turns = batch["player_control_substeps"]
    turns[0, 0, 93, 0, 12] = 18.0  # 90 degrees at the first video frame of latent frame 24 ...
    turns[0, 0, 97, 0, 12] = -18.0  # ... and back at the first of the target block
    predicted = PredictedVisibility(batch, depth_fn=flat_depth(2000.0))
    labelled = predicted.for_block(25, batch, torch.zeros(33, 48, 24, 42))
    visible, valid = labelled["client_visibility"], labelled["client_visibility_valid"]
    assert bool(visible[0, 1, 97:113].all()) and bool(valid[0, 1, 97:113].all())
    # while it looks away, slot 1 is outside its view: known, and not visible
    assert not bool(visible[0, 1, 93:97].any()) and bool(valid[0, 1, 93:97].all())
    assert bool(visible[0, 1, 49:93].all())


def test_the_depth_of_a_generated_frame_is_exponentiated_in_float32():
    """As in the paper's runs. The log depth below is 436.0175476 u in float32 and 436.0175501 u in
    float64; the nearest test point of slot 1 (40 u nearer than its 500.5 u) is visible beyond
    436.0175489 u."""
    log_depth = np.float32(6.0776824951171875)
    batch = standing_round([[0.4824511408805847, 0.0, 0.0], [500.5, 0.0, 0.0]], latents=5)

    def depth_fn(latents):
        return np.full((int(latents.shape[0]), 4, 24, 42), log_depth, np.float32)

    predicted = PredictedVisibility(batch, depth_fn=depth_fn)
    assert (
        predicted.generated_visible(1, 4, torch.zeros(5, 48, 24, 42)).tolist()
        == [[False, False]] * 4
    )
    # the re-test of a block's x0 reads the depth in float64: there the player is visible
    frames = block_depth_frames(depth_fn, torch.zeros(4, 48, 24, 42))
    assert predicted.retested_visible(1, frames).tolist() == [[False, True]] * 4
