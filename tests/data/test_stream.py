"""Drawing the training windows: the resumable sampler and the batch stream of one rank."""

import pytest
import torch
from torch.utils.data import Dataset

from worldcast.data.stream import ResumableDataStream, ResumableDistributedSampler


class _IndexDataset(Dataset):
    """``size`` items that hold their own index; with ``weights``, a weighted dataset."""

    def __init__(self, size: int, weights: list[float] | None = None) -> None:
        self.size = int(size)
        if weights is not None:
            self.sample_weights = weights

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {"latents": torch.tensor([index], dtype=torch.float32), "metadata": {"i": index}}


def test_every_rank_takes_its_share_of_one_weighted_draw():
    dataset = _IndexDataset(50)
    weights = [0.0] * 40 + [1.0] * 10  # only the last ten windows can be drawn
    ranks = [
        ResumableDistributedSampler(dataset, num_replicas=4, rank=rank, seed=3, weights=weights)
        for rank in range(4)
    ]
    draws = [list(sampler) for sampler in ranks]
    assert [len(draw) for draw in draws] == [12] * 4  # 50 // 4 per rank
    assert all(index >= 40 for draw in draws for index in draw)
    generator = torch.Generator().manual_seed(3)
    whole = torch.multinomial(
        torch.tensor(weights, dtype=torch.float64), 48, replacement=True, generator=generator
    ).tolist()
    assert draws[1] == whole[1::4]  # positions rank, rank + world size, ... of the one draw
    ranks[1].set_epoch(1)
    assert list(ranks[1]) != draws[1]  # another epoch, another draw


def test_sampler_resume_state_and_refusals():
    dataset = _IndexDataset(50)
    weights = [1.0 + (i % 5) for i in range(50)]
    sampler = ResumableDistributedSampler(dataset, num_replicas=4, rank=1, seed=3, weights=weights)
    whole = list(sampler)
    sampler.set_epoch(2)
    sampler.set_start_index(5)
    state = sampler.state_dict()
    assert (state["epoch"], state["start_index"], state["seed"], state["dataset_size"]) == (
        2,
        5,
        3,
        50,
    )
    assert len(sampler) == 12 - 5
    fresh = ResumableDistributedSampler(dataset, num_replicas=4, rank=1, seed=99, weights=weights)
    fresh.load_state_dict(state)
    assert list(fresh) == list(sampler) and len(list(fresh)) == 7 and list(fresh) != whole[5:]
    with pytest.raises(RuntimeError, match="topology or the data"):
        ResumableDistributedSampler(
            dataset, num_replicas=4, rank=2, seed=3, weights=weights
        ).load_state_dict(state)
    with pytest.raises(RuntimeError, match="topology or the data"):
        ResumableDistributedSampler(
            dataset, num_replicas=4, rank=1, seed=3, weights=[2.0] * 50
        ).load_state_dict(state)


def _stream(dataset: Dataset, batch_size: int = 2) -> ResumableDataStream:
    return ResumableDataStream(
        dataset,
        rank=0,
        world_size=1,
        seed=7,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )


def test_a_stream_resumes_at_the_next_batch_the_trainer_has_not_consumed():
    dataset = _IndexDataset(10, weights=[1.0] * 10)
    whole = _stream(dataset)
    batches = [whole.next_batch()["latents"][:, 0].tolist() for _ in range(7)]  # into epoch 1
    stream = _stream(dataset)
    assert [stream.next_batch()["latents"][:, 0].tolist() for _ in range(2)] == batches[:2]
    state = stream.state_dict()
    assert state["batch_size"] == 2
    assert (state["sampler"]["epoch"], state["sampler"]["start_index"]) == (0, 4)  # samples
    resumed = _stream(dataset)
    resumed.load_state_dict(state)
    assert [resumed.next_batch()["latents"][:, 0].tolist() for _ in range(5)] == batches[2:]
    assert (resumed.epoch, resumed.sample_offset) == (1, 4)
    assert resumed.next_batch()["metadata"] == [
        {"i": i} for i in whole.next_batch()["latents"][:, 0]
    ]
    with pytest.raises(RuntimeError, match="the checkpoint ran with batch_size 2"):
        _stream(dataset, batch_size=5).load_state_dict(state)
    with pytest.raises(ValueError, match="fewer than batch_size = 20 windows"):
        _stream(dataset, batch_size=20)
