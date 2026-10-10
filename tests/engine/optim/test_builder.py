"""The optimizer: the groups and their learning rates, the sharded group norms and the per-group
clip."""

import math

import pytest
import torch
from torch import nn

from worldcast.engine.optim import builder as O
from worldcast.modeling.wan22.model import WORLDCAST_MODULES


def _model():
    model = nn.Module()
    model.backbone = nn.Linear(2, 2)
    model.state_injector = nn.Linear(2, 2)
    model.ray_embedding = nn.Linear(2, 2)
    return model


def test_a_parameter_belongs_to_the_module_in_its_path():
    assert O.parameter_group("generator.blocks.3.ffn.0.weight") == "backbone"
    assert O.parameter_group("generator.controls.buttons.weight") == "backbone"
    assert O.parameter_group("state_injector.proj.bias") == "state_injector"
    # under FSDP the names carry its wrapper's attribute
    wrapped = "_fsdp_wrapped_module.generator.ray_embedding.mlp.0.weight"
    assert O.parameter_group(wrapped) == "ray_embedding"
    assert O.parameter_group("generator.visibility_probe.head.weight") == "visibility_probe"
    assert O.parameter_group("generator.observer_signals.out.weight") == "observer_signals"
    # the groups are the generator's modules but the controls, which train with the backbone
    assert set(O.GROUPS[1:]) == set(WORLDCAST_MODULES) - {"controls"}


def test_groups_and_learning_rates():
    optimizer = O.build_optimizer(_model().named_parameters(), 2.8e-5)
    groups = {g["name"]: g["lr"] for g in optimizer.param_groups}
    assert groups == {"backbone": 2.8e-5, "state_injector": 1.4e-3, "ray_embedding": 1e-4}
    assert optimizer.param_groups[0]["betas"] == (0.0, 0.999)
    assert optimizer.param_groups[0]["weight_decay"] == 0.01


def test_frozen_parameters_are_left_out():
    model = _model()
    model.state_injector.requires_grad_(False)
    optimizer = O.build_optimizer(model.named_parameters(), 1e-5)
    assert [g["name"] for g in optimizer.param_groups] == ["backbone", "ray_embedding"]


def test_the_distillation_trains_every_module_at_one_lr():
    optimizer = O.build_optimizer(_model().named_parameters(), 2e-6, module_lrs=False)
    assert {g["lr"] for g in optimizer.param_groups} == {2e-6}


def test_group_grad_norms_reduce_the_sharded_sums():
    """The sharded parameters' sums are all-reduced; the reduce sees exactly that vector."""
    a, b = nn.Parameter(torch.ones(3)), nn.Parameter(torch.ones(2))
    a.grad, b.grad = torch.full((3,), 2.0), torch.full((2,), 3.0)
    groups = [{"name": "backbone", "params": [a]}, {"name": "state_injector", "params": [b]}]
    seen = []

    def reduce(t):
        seen.append(t.clone())
        t.mul_(4)  # four shards with the same local sum

    norms = O.group_grad_norms(groups, sharded=[a], all_reduce=reduce)
    assert torch.equal(seen[0], torch.tensor([12.0, 0.0]))
    assert norms == pytest.approx({"backbone": math.sqrt(48.0), "state_injector": math.sqrt(18.0)})
    foreign = torch.optim.SGD([a], lr=0.1).param_groups
    with pytest.raises(ValueError, match="have no name: build the optimizer by build_optimizer"):
        O.group_grad_norms(foreign)


def test_each_group_is_clipped_on_its_own():
    large, small = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))
    large.grad, small.grad = torch.tensor([3.0, 4.0]), torch.tensor([0.3, 0.4])
    groups = [{"name": "backbone", "params": [large]}, {"name": "ray_embedding", "params": [small]}]
    norms = O.group_grad_norms(groups)
    assert norms == pytest.approx({"backbone": 5.0, "ray_embedding": 0.5})
    total = O.clip_per_group_(groups, norms, 1.0)
    assert total == pytest.approx(math.sqrt(25.25))  # the global norm before the clip
    # the group above the threshold is scaled onto it; the one below is untouched
    assert torch.allclose(large.grad, torch.tensor([0.6, 0.8]), atol=1e-6)
    assert small.grad.tolist() == pytest.approx([0.3, 0.4])
