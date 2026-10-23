"""Fixtures of the preparation tests: a tiny OpenCS2 download."""

import shutil

import pytest

from tests.data.preparation.support import write_opencs2


@pytest.fixture(scope="session")
def opencs2_template(tmp_path_factory):
    """The directory of the tiny download (:data:`tests.data.preparation.support.SEATS`)."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    root = tmp_path_factory.mktemp("opencs2")
    write_opencs2(root)
    return root


@pytest.fixture
def opencs2(opencs2_template, tmp_path):
    """A copy of the tiny download that the test may damage."""
    root = tmp_path / "opencs2"
    shutil.copytree(opencs2_template, root)
    return root
