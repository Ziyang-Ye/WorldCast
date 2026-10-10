"""The torchrun process group and the RNG states of every rank."""

import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

__all__ = [
    "DistInfo",
    "barrier",
    "capture_rng_state",
    "destroy_distributed",
    "init_distributed",
    "restore_rng_state",
    "use_deterministic_algorithms",
]

#: Timeout of every collective of the process group.
_COLLECTIVE_TIMEOUT = timedelta(minutes=30)


def _in_process_group() -> bool:
    return dist.is_available() and dist.is_initialized()


@dataclass(frozen=True)
class DistInfo:
    """This process: ``rank`` of ``world_size``, and its ``device``."""

    rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        """This is rank 0, which writes the run's files."""
        return self.rank == 0

    @property
    def initialized(self) -> bool:
        """The process is in a process group (started by torchrun)."""
        return _in_process_group()


def init_distributed() -> DistInfo:
    """Join torchrun's process group (NCCL on CUDA, gloo otherwise) and pick this rank's device.

    Without torchrun's environment the process runs alone, as rank 0 of 1 without a process group
    (and a trainer without FSDP: :func:`worldcast.engine.training.build.wrap`).

    Returns:
        DistInfo: this process.
    """
    use_cuda = torch.cuda.is_available()
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
        if use_cuda:
            torch.cuda.set_device(device)
        return DistInfo(device=device)
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if use_cuda:
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo",
            rank=rank,
            world_size=world_size,
            timeout=_COLLECTIVE_TIMEOUT,
        )
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    return DistInfo(rank=rank, world_size=world_size, device=device)


def destroy_distributed() -> None:
    """Leave the process group, if the process is in one."""
    if _in_process_group():
        dist.destroy_process_group()


def barrier() -> None:
    """Wait for every rank of the process group; nothing for a process alone."""
    if _in_process_group():
        dist.barrier()


def use_deterministic_algorithms() -> None:
    """Deterministic torch kernels for the rest of the process, as the distillation stage ran.

    cuBLAS refuses them without ``CUBLAS_WORKSPACE_CONFIG``, which is set to ``:4096:8`` unless
    the launcher set it. Call it before the first CUDA matmul.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Python, numpy, torch CPU and, on CUDA, the device's RNG states."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.device(device).type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    """Restore :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device=device)
