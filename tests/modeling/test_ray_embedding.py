"""The ray embedding: Plücker coordinates relative to the anchor camera, and their conditions."""

import math

import pytest
import torch

from tests.modeling.support import random_c2w, randomize_
from worldcast.modeling.ray_embedding import (
    LENGTH_SCALE_U,
    RAY_CONDITION_KEYS,
    RayConditions,
    RayEmbedding,
    plucker_rays,
    relative_c2w,
    se3_inverse,
)


def _conditions(frames: int = 5, anchor: int = 3, seed: int = 0) -> RayConditions:
    c2w = random_c2w(1, frames, seed=seed)
    return RayConditions(
        frame_c2w=c2w, frame_tans=torch.full((1, frames, 2), 0.5), anchor_c2w=c2w[:, anchor]
    )


def test_rigid_inverse_and_relative_cameras():
    c2w = random_c2w(1, 3, seed=1)[0]
    eye = torch.eye(4).expand(3, 4, 4)
    torch.testing.assert_close(se3_inverse(c2w) @ c2w, eye, rtol=0, atol=1e-4)
    relative = relative_c2w(c2w, c2w[1])
    torch.testing.assert_close(relative[1], torch.eye(4), rtol=0, atol=1e-4)


def test_the_embedding_starts_at_zero():
    """``phi`` is Linear-SiLU-Linear with a zero last layer (App. "Ray embedding")."""
    torch.manual_seed(0)
    embedding = RayEmbedding(16)
    out = embedding(_conditions(), frame_offset=0, num_frames=5, grid=(2, 3), dtype=torch.float32)
    assert out.shape == (1, 5 * 6, 16) and not bool(out.any())


def _rays(conditions: RayConditions, *, grid=(2, 3)) -> torch.Tensor:
    return plucker_rays(conditions.frame_c2w, conditions.frame_tans, conditions.anchor_c2w, grid)


def test_rays_of_the_anchor_camera():
    """At the anchor the camera centre is the origin, so the moment is zero and the direction of
    a token is its unit viewing direction: x right, y down, z forward, from the frame's field of
    view."""
    rays = _rays(_conditions(anchor=3), grid=(2, 2))[:, 3]
    assert rays.shape == (1, 4, 6) and rays.dtype == torch.float32
    norm = math.sqrt(0.25**2 + 0.25**2 + 1)  # token centres at +-1/2 of the half-extent tan 0.5
    want = torch.tensor([[-0.25, -0.25, 1], [0.25, -0.25, 1], [-0.25, 0.25, 1], [0.25, 0.25, 1]])
    torch.testing.assert_close(rays[0, :, :3], want / norm, rtol=0, atol=1e-5)
    torch.testing.assert_close(rays[0, :, 3:], torch.zeros(4, 3), rtol=0, atol=1e-5)


def test_rays_are_relative_to_the_anchor_and_scaled_by_the_length():
    """Moving every camera by one rigid transform changes nothing; the moment is ``(o / l) x d``."""
    conditions = _conditions()
    rays = _rays(conditions)
    assert rays.shape == (1, 5, 6, 6)
    move = random_c2w(1, 1, seed=7)[0, 0]
    moved = RayConditions(
        frame_c2w=move @ conditions.frame_c2w,
        frame_tans=conditions.frame_tans,
        anchor_c2w=move @ conditions.anchor_c2w,
    )
    torch.testing.assert_close(_rays(moved), rays, rtol=0, atol=2e-3)
    centre = relative_c2w(conditions.frame_c2w[0], conditions.anchor_c2w[0])[1, :3, 3]
    moment = torch.linalg.cross((centre / LENGTH_SCALE_U).expand(6, 3), rays[0, 1, :, :3])
    torch.testing.assert_close(rays[0, 1, :, 3:], moment, rtol=1e-4, atol=1e-5)


def test_a_call_embeds_its_frames_of_the_window():
    embedding = randomize_(RayEmbedding(8), 2)
    conditions = _conditions()
    kwargs = dict(grid=(2, 3), dtype=torch.float32)
    with torch.no_grad():
        window = embedding(conditions, frame_offset=0, num_frames=5, **kwargs)
        call = embedding(conditions, frame_offset=3, num_frames=2, **kwargs)
    assert torch.equal(call, window[:, 18:30])


def test_a_contiguous_window_is_anchored_at_its_first_frame():
    c2w, tans = random_c2w(2, 5, seed=3), torch.full((2, 5, 2), 0.5)
    conditions = RayConditions.contiguous(c2w, tans)
    assert conditions.frame_c2w is c2w and conditions.frame_tans is tans
    assert torch.equal(conditions.anchor_c2w, c2w[:, 0])
    with pytest.raises(ValueError, match=r"c2w must be \[B, F, 4, 4\]"):
        RayConditions.contiguous(c2w[0], tans)


def test_ray_conditions_check_their_shapes():
    c2w, tans = random_c2w(2, 5, seed=3), torch.full((2, 5, 2), 0.5)
    with pytest.raises(ValueError, match=r"frame_c2w must be \[B, F, 4, 4\]"):
        RayConditions(frame_c2w=c2w[..., :3, :], frame_tans=tans, anchor_c2w=c2w[:, 0])
    with pytest.raises(ValueError, match=r"frame_tans must be \[B, F, 2\]"):
        RayConditions(frame_c2w=c2w, frame_tans=tans[:, :4], anchor_c2w=c2w[:, 0])
    with pytest.raises(ValueError, match=r"anchor_c2w must be \[B, 4, 4\]"):
        RayConditions(frame_c2w=c2w, frame_tans=tans, anchor_c2w=c2w[0, 0])


def test_conditions_round_trip_through_a_condition_dict():
    conditions = _conditions()
    conditions.frame_c2w = conditions.frame_c2w.double()
    entries = conditions.conditions("cpu")
    assert tuple(entries) == RAY_CONDITION_KEYS
    assert entries["ray_frame_c2w"].dtype == torch.float32
    back = RayConditions.from_conditions({"prompt_embeds": None, **entries})
    assert torch.equal(back.anchor_c2w, conditions.anchor_c2w)
    assert RayConditions.from_conditions({"prompt_embeds": None}) is None
    del entries["ray_frame_tans"]
    with pytest.raises(ValueError, match="together"):
        RayConditions.from_conditions(entries)
