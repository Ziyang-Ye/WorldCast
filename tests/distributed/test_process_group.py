"""The process group and the RNG states of a rank."""

import os
import random

import numpy as np
import torch

from worldcast.distributed import process_group as P


def test_without_torchrun_the_process_runs_alone(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    info = P.init_distributed()
    assert (info.rank, info.world_size) == (0, 1) and info.is_main
    assert not info.initialized  # no process group: the collectives below do nothing
    P.barrier()
    P.destroy_distributed()
    assert P.DistInfo(rank=3, world_size=8).is_main is False


def test_a_rank_of_a_process_group_reports_it(gloo):
    assert P.DistInfo().initialized
    P.barrier()


def test_deterministic_algorithms_are_switched_on_with_the_cublas_workspace(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    enabled = torch.are_deterministic_algorithms_enabled()
    try:
        P.use_deterministic_algorithms()
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(enabled)


def test_every_rng_state_is_restored():
    device = torch.device("cpu")
    state = P.capture_rng_state(device)
    first = (random.random(), float(np.random.rand()), float(torch.rand(())))
    P.restore_rng_state(state, device)
    assert (random.random(), float(np.random.rand()), float(torch.rand(()))) == first
    assert set(state) == {"python", "numpy", "torch_cpu"}
