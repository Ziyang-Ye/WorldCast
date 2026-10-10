"""Fixtures of the data tests: a small recorded video and a synthetic recorded round."""

import numpy as np
import pytest

from tests.data import synthetic_round
from tests.data.support import RECORDING_LEVELS


@pytest.fixture(scope="session")
def recording(tmp_path_factory):
    """The directory of ``m0.mp4``: a 12-frame 96 x 168 recording, frame ``i`` a flat grey of
    ``RECORDING_LEVELS[i]``."""
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio

    root = tmp_path_factory.mktemp("recordings")
    writer = imageio.get_writer(
        str(root / "m0.mp4"), fps=32, codec="libx264", quality=9, macro_block_size=1
    )
    for level in RECORDING_LEVELS:
        writer.append_data(np.full((96, 168, 3), level, np.uint8))
    writer.close()
    return root


@pytest.fixture(scope="module")
def recorded_round(tmp_path_factory):
    """The synthetic round (:mod:`tests.data.synthetic_round`), its ticks and frames served from
    memory."""
    pytest.importorskip("trimesh.ray.ray_pyembree")
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    with pytest.MonkeyPatch.context() as monkeypatch:
        synthetic = synthetic_round.write_round(tmp_path_factory.mktemp("round"))
        synthetic_round.patch_readers(monkeypatch, synthetic)
        yield synthetic
