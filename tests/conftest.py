"""Fixtures shared by the tests: a tiny Wan2.2 VAE, a single-process gloo group, and an ffmpeg that
loses part of what it is given."""

import os
import shlex
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

#: The repository.
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def tiny_vae():
    """A small random ``Wan22VAE`` (:func:`tests.modeling.support.make_tiny_vae`)."""
    from tests.modeling.support import make_tiny_vae

    return make_tiny_vae()


@pytest.fixture(scope="module")
def gloo():
    """A world-size-1 gloo process group for this module (destroyed afterwards), so that the
    trainers run under real FSDP on the CPU."""
    import torch.distributed as dist

    if dist.is_initialized():
        yield dist.group.WORLD
        return
    with tempfile.TemporaryDirectory() as tmp:
        store = dist.FileStore(os.path.join(tmp, "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
        try:
            yield dist.group.WORLD
        finally:
            dist.destroy_process_group()


@pytest.fixture
def lossy_ffmpeg(tmp_path) -> Callable[[str], str]:
    """``lossy_ffmpeg(loss)``: the path of an ffmpeg that loses ``loss`` of what it is given
    (``tests/tools/stand_ins/lossy_ffmpeg.py`` over imageio-ffmpeg's), in ``tmp_path / "bin"``."""
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    stand_in = REPO / "tests" / "tools" / "stand_ins" / "lossy_ffmpeg.py"

    def make(loss: str) -> str:
        command = shlex.join([sys.executable, str(stand_in), loss, imageio_ffmpeg.get_ffmpeg_exe()])
        path = tmp_path / "bin" / f"ffmpeg-{loss}"
        path.parent.mkdir(exist_ok=True)
        # the repository for tests.tools.support, without the run.sh tests' sitecustomize (torch)
        path.write_text(
            f"#!/bin/sh\nPYTHONPATH={shlex.quote(str(REPO))}\nexport PYTHONPATH\n"
            f'exec {command} "$@"\n'
        )
        path.chmod(0o755)
        return str(path)

    return make
