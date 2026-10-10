"""The synthetic round as the tools read it: a config file and the offline client's latents."""

import pytest

import tests.engine.inference.support as sw
from tests.engine.inference.support import offline_run
from tests.tools.support import write_config


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """The synthetic round's config file, its config, and the offline client's id and latents."""
    offline = offline_run(tmp_path / "offline")
    sw.patch_ticks(monkeypatch, offline["world"]["tables"])
    config = write_config(offline["cfg"], tmp_path / "config.yaml")
    return config, offline["cfg"], offline["media_id"], offline["latents"]
