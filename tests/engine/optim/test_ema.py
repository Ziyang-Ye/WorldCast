"""The sharded EMA: its fp32 start, its recursion, its weights in place of the live ones, and its
resume."""

import pytest
import torch
from torch import nn

from worldcast.engine.optim.ema import ShardedEMA


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))


def test_the_ema_starts_as_an_fp32_copy_and_follows_the_recursion():
    model = _model().double()
    ema = ShardedEMA.start(model, 0.9)
    for name, p in model.named_parameters():
        assert ema.shadow[name].dtype is torch.float32 and torch.equal(ema.shadow[name], p.float())
    start = {n: t.clone() for n, t in ema.shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for name, p in model.named_parameters():
        expected = start[name] * 0.9 + p.detach().float() * 0.1
        assert torch.allclose(ema.shadow[name], expected, atol=1e-6)


def test_the_ema_converges_to_the_weights():
    model = _model()
    ema = ShardedEMA.start(model, 0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(5.0)
    gaps = []
    for _ in range(8):
        ema.update(model)
        gaps.append(max(float((s - 5.0).abs().max()) for s in ema.shadow.values()))
    assert all(b < a for a, b in zip(gaps, gaps[1:])) and gaps[-1] < 0.05


def test_the_ema_weights_stand_in_for_the_live_ones_inside_the_block():
    model = _model()
    ema = ShardedEMA.start(model, 0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)
    live = {name: p.detach().clone() for name, p in model.named_parameters()}
    with ema.applied(model):
        assert all(torch.equal(p, ema.shadow[name]) for name, p in model.named_parameters())
    assert all(torch.equal(p, live[name]) for name, p in model.named_parameters())
    full = ema.full_state_dict(model)
    assert all(torch.equal(full[name], ema.shadow[name]) for name in ema.shadow)


def test_a_resumed_ema_is_the_saved_one():
    model = _model()
    ema = ShardedEMA.start(model, 0.9)
    ema.update(_model(seed=1))
    resumed = ShardedEMA.resume(model, 0.9, ema.state_dict())
    assert resumed.decay == 0.9 and resumed.shadow.keys() == ema.shadow.keys()
    assert all(torch.equal(resumed.shadow[name], ema.shadow[name]) for name in ema.shadow)
    with pytest.raises(RuntimeError, match="do not match this model's parameters"):
        ShardedEMA.resume(nn.Linear(4, 3), 0.9, ema.state_dict())
