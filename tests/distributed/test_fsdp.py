"""The FSDP helpers on a plain module and under FSDP (a world-size-1 gloo group)."""

from contextlib import nullcontext

import pytest
import torch
from torch import nn

from worldcast.distributed import fsdp as F


def _module() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _backward(module: nn.Module) -> None:
    module(torch.ones(3, 4)).sum().backward()


def test_a_plain_module_needs_no_collective():
    module = _module()
    assert not F.is_fsdp(module) and F.live_module(module) is module
    assert F.sharded_parameters(module) == [] and F.group_all_reduce(module) is None
    assert isinstance(F.no_sync(module, True), nullcontext)
    state = F.full_state_dict(module)
    assert state.keys() == module.state_dict().keys()
    with torch.no_grad():
        module[0].weight.zero_()
    assert bool(state["0.weight"].any())  # a copy, not a view of the live weights
    _backward(module)
    norm = F.clip_grad_norm(module, 1e-3)
    total = torch.sqrt(sum(p.grad.pow(2).sum() for p in module.parameters()))
    assert float(norm) > 1e-3 and float(total) <= 1e-3 * 1.001


@pytest.mark.parametrize("sharding", ["full", "hybrid_full"])
def test_fsdp_wraps_and_consolidates(gloo, monkeypatch, sharding):
    monkeypatch.setattr(F, "MIN_UNIT_PARAMS", 1)  # every layer of the small module is a unit
    module = _module()
    reference = {k: v.clone() for k, v in module.state_dict().items()}
    wrapped = F.fsdp_wrap(module, sharding=sharding, wrap="size", device=torch.device("cpu"))
    assert F.is_fsdp(wrapped) and F.live_module(wrapped) is module
    assert F.is_fsdp(module[0]) and F.is_fsdp(module[2])  # the units of the size wrap
    shapes = [tuple(p.shape) for _, p in F.live_module(wrapped).named_parameters()]
    assert shapes == [(8, 4), (8,), (2, 8), (2,)]
    for on_every_rank in (False, True):
        state = F.full_state_dict(wrapped, on_every_rank=on_every_rank)
        assert all(torch.equal(state[k], reference[k]) for k in reference)
    with F.no_sync(wrapped, True):
        _backward(wrapped)
    _backward(wrapped)
    assert float(F.clip_grad_norm(wrapped, 1.0)) > 0
    value = torch.tensor([2.0])
    F.group_all_reduce(wrapped)(value)
    assert value.item() == 2.0  # a SUM over a group of one


def test_the_root_wrap_is_one_unit(gloo):
    module = _module()
    wrapped = F.fsdp_wrap(module, sharding="full", wrap="root", device=torch.device("cpu"))
    assert F.is_fsdp(wrapped) and not F.is_fsdp(module[0])


def test_the_sharding_and_the_wrap_are_named(gloo):
    for sharding, wrap in (("shard_grad_op", "size"), ("full", "block")):
        with pytest.raises(ValueError, match="sharding is one of .* and wrap one of"):
            F.fsdp_wrap(_module(), sharding=sharding, wrap=wrap, device=torch.device("cpu"))
