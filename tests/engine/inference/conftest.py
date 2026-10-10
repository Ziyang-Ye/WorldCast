"""Fixtures of the inference tests: one synthetic round and the offline client's latents of it."""

import numpy as np
import pytest
import torch

import tests.engine.inference.support as sw
from worldcast.engine.inference.client import run_client


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """A synthetic round with one client, and tiny random weights."""
    torch.set_num_threads(1)
    tmp = tmp_path_factory.mktemp("client")
    world = sw.make_world(tmp / "world", clients=(0,))
    weights = sw.make_weights(tmp / "weights")
    return dict(tmp=tmp, world=world, weights=weights)


@pytest.fixture
def reference(synthetic, monkeypatch):
    """The offline client's latents of the synthetic round ``[33, 48, 24, 42]``; the round's tick
    tables are served for the rest of the test."""
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    out = synthetic["tmp"] / "client"
    if not (out / "latents.npy").exists():
        cfg = sw.config(
            synthetic["world"],
            synthetic["weights"],
            out,
            max_blocks=sw.MAX_BLOCKS,
            world_state=str(synthetic["tmp"] / "world_state"),
        )
        run_client(cfg)
    return np.load(out / "latents.npy")
