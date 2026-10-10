"""The appearance check of the memory frames: a token's surface patch in two recorded frames."""

import numpy as np
import pytest

from worldcast.data.appearance import matching_patches

TANS = (4.0 / 3.0, 0.75)


def _texture(seed: int) -> np.ndarray:
    """A smooth, well-exposed random texture, ``[384, 672, 3]`` uint8."""
    cv2 = pytest.importorskip("cv2")
    noise = np.random.default_rng(seed).uniform(0.0, 1.0, (384, 672, 3)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 6.0)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return (40 + 170 * smooth).astype(np.uint8)


def _wall():
    """Four points of a wall 300 u in front of a camera at the origin, facing it."""
    points = np.array([[x, y, 300.0] for x in (-120.0, 120.0) for y in (-40.0, 40.0)])
    normals = np.tile([0.0, 0.0, -1.0], (4, 1))
    return points, normals


def test_a_patch_matches_itself_and_nothing_else():
    pytest.importorskip("scipy")
    points, normals = _wall()
    camera, image = np.eye(4), _texture(0)

    def match(source_image, source_camera=camera):
        return matching_patches(
            image, source_image, points, normals, camera, source_camera, TANS, TANS
        )

    assert match(image).all()
    assert not match(_texture(1)).any()  # another surface
    assert not match(np.full_like(image, 128)).any()  # no texture to compare
    assert not match(np.full_like(image, 255)).any()  # clipped
    shifted = np.eye(4)
    shifted[0, 3] = 60.0  # the same image from another place shows another part of the wall
    assert not match(image, shifted).any()
    behind = np.eye(4)
    behind[2, 3] = 600.0  # the wall is behind this camera
    assert not match(image, behind).any()
