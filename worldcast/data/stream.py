"""Drawing the training windows: the resumable sampler and the per-rank batch stream.

The windows are drawn by :class:`ResumableDistributedSampler`: one global multinomial draw with
replacement over the window weights, seeded per epoch and sharded by rank.
:class:`ResumableDataStream` serves one rank's batches and resumes at the next unconsumed one.
"""

import hashlib
import itertools
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .window import collate_windows

__all__ = ["ResumableDataStream", "ResumableDistributedSampler"]


class ResumableDistributedSampler(DistributedSampler):
    """``DistributedSampler`` with an explicit rank and world size, exact resume and weights.

    With ``weights`` (one per dataset index), every rank draws the same global sequence
    ``torch.multinomial(weights, total_size, replacement=True)`` from a CPU generator seeded
    ``seed + epoch`` and takes positions ``rank, rank + world_size, ...``. Without, torch's
    shuffled order.
    ``start_index`` skips what a resumed run already consumed of the epoch.

    Args:
        dataset (Dataset): the windows.
        num_replicas (int): the world size.
        rank (int): this rank.
        seed (int): the run's base seed.
        weights (Sequence[float] | None): the window weights.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed, drop_last=True
        )
        self.start_index = 0
        self.weights = None
        self.weights_sha256 = None
        if weights is not None:
            self.weights = torch.as_tensor(weights, dtype=torch.float64).contiguous().clone()
            digest = hashlib.sha256(self.weights.numpy().tobytes(order="C"))
            self.weights_sha256 = digest.hexdigest()

    def set_start_index(self, start_index: int) -> None:
        """Skip the first ``start_index`` samples of this rank's epoch."""
        self.start_index = int(start_index)

    def __iter__(self) -> Iterator[int]:
        if self.weights is None:
            indices = super().__iter__()
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            draw = torch.multinomial(
                self.weights, self.total_size, replacement=True, generator=generator
            )
            indices = iter(draw[self.rank :: self.num_replicas].tolist())
        return itertools.islice(indices, self.start_index, None)

    def __len__(self) -> int:
        return max(0, self.num_samples - self.start_index)

    def state_dict(self) -> dict:
        """The epoch and the position in it, with what :meth:`load_state_dict` checks."""
        return {
            "epoch": int(self.epoch),
            "start_index": int(self.start_index),
            "seed": int(self.seed),
            "rank": int(self.rank),
            "num_replicas": int(self.num_replicas),
            "dataset_size": len(self.dataset),
            "weights_sha256": self.weights_sha256,
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Resume; the topology, the dataset and its weights must be the checkpoint's."""
        expected = {
            "rank": int(self.rank),
            "num_replicas": int(self.num_replicas),
            "dataset_size": len(self.dataset),
            "weights_sha256": self.weights_sha256,
        }
        changed = {k: (state.get(k), v) for k, v in expected.items() if state.get(k) != v}
        if changed:
            raise RuntimeError(f"the topology or the data changed since the checkpoint: {changed}")
        self.seed = int(state["seed"])
        self.set_epoch(int(state["epoch"]))
        self.set_start_index(int(state["start_index"]))


class ResumableDataStream:
    """The batches of one rank, resumable at the next batch the trainer has not consumed.

    The loader has its own generator, seeded ``seed + rank + 100003``: it gives the workers their
    seeds, which a loader would otherwise draw from the global RNG at every epoch. A resumed run
    serves the same windows; the workers' seeds within the resumed epoch are not part of the
    resume (the windows draw nothing from them).

    Args:
        dataset (Dataset): the windows (with ``sample_weights`` when weighted).
        rank (int): this rank.
        world_size (int): the world size.
        seed (int): the run's base seed.
        batch_size (int): samples per GPU per micro-batch.
        num_workers (int): loader workers.
        pin_memory (bool): pin the batches (CUDA).
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        rank: int,
        world_size: int,
        seed: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
    ) -> None:
        self.batch_size = int(batch_size)
        self.sampler = ResumableDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            weights=getattr(dataset, "sample_weights", None),
        )
        if self.sampler.num_samples < self.batch_size:
            raise ValueError(f"rank {rank} has fewer than batch_size = {batch_size} windows")
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(int(seed) + int(rank) + 100_003)
        self.loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=True,
            generator=self.loader_generator,
            collate_fn=collate_windows,
        )
        self.epoch = 0
        self.sample_offset = 0
        self.iterator = None

    def _new_iterator(self) -> None:
        self.sampler.set_epoch(self.epoch)
        self.sampler.set_start_index(self.sample_offset)
        self.iterator = iter(self.loader)

    def next_batch(self) -> dict[str, Any]:
        """The next batch, into the next epoch when this one is done."""
        while True:
            if self.iterator is None:
                self._new_iterator()
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.epoch += 1
                self.sample_offset = 0
                self.iterator = None
                continue
            self.sample_offset += self.batch_size
            return batch

    def state_dict(self) -> dict:
        """The sampler's state at the next unconsumed batch, and the loader generator's."""
        self.sampler.set_epoch(self.epoch)
        self.sampler.set_start_index(self.sample_offset)
        return {
            "sampler": self.sampler.state_dict(),
            "batch_size": self.batch_size,
            "loader_generator": self.loader_generator.get_state(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Resume at the batch after the last one the checkpointed run consumed."""
        if int(state["batch_size"]) != self.batch_size:
            raise RuntimeError(f"the checkpoint ran with batch_size {state['batch_size']}")
        self.sampler.load_state_dict(state["sampler"])
        self.epoch = int(self.sampler.epoch)
        self.sample_offset = int(self.sampler.start_index)
        self.loader_generator.set_state(state["loader_generator"])
        self.iterator = None
