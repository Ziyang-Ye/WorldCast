"""AdamW over the generator's parameter groups, and the per-group gradient clip.

Each WorldCast module trains in a group of its own, at its own learning rate (:data:`MODULE_LRS`);
the backbone and the control embedding train at the run's ``optim.lr``. The distillation stage
trains every group at one learning rate. AdamW: betas (0, 0.999), weight decay 0.01 on every
parameter (biases and norms included), a constant learning rate.
"""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

__all__ = [
    "BETAS",
    "GROUPS",
    "MAX_GRAD_NORM",
    "MODULE_LRS",
    "WEIGHT_DECAY",
    "build_optimizer",
    "clip_per_group_",
    "group_grad_norms",
    "parameter_group",
]

#: Learning rates of the WorldCast modules in stages 2 and 3, by their name in the generator.
MODULE_LRS = {
    "state_injector": 1.4e-3,
    "visibility_probe": 9.9e-5,
    "observer_signals": 1.98e-4,
    "ray_embedding": 1e-4,
}
#: Optimizer groups, in order: ``backbone`` holds every parameter outside those modules (the
#: controls train with it).
GROUPS = ("backbone", *MODULE_LRS)
#: AdamW betas of every run.
BETAS = (0.0, 0.999)
#: AdamW weight decay of every run, on every parameter.
WEIGHT_DECAY = 0.01
#: Clip threshold of the gradient norm, of each parameter group or of the whole model
#: (``optim.clip``).
MAX_GRAD_NORM = 1.0


def parameter_group(name: str) -> str:
    """The group of a parameter name (plain or FSDP-wrapped)."""
    components = name.split(".")
    return next((module for module in MODULE_LRS if module in components), "backbone")


def build_optimizer(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    lr: float,
    *,
    module_lrs: bool = True,
    foreach: bool | None = None,
) -> torch.optim.AdamW:
    """AdamW over the trainable parameters, a group per module present (each carries its ``name``).

    Args:
        named_parameters (Iterable[tuple[str, nn.Parameter]]): ``module.named_parameters()``.
        lr (float): the backbone's learning rate; with ``module_lrs=False`` every group's.
        module_lrs (bool): the WorldCast modules train at :data:`MODULE_LRS`.
        foreach (bool | None): AdamW's ``foreach``; the distillation stage passes ``True``, as
            trained; ``None`` is torch's default (the multi-tensor kernel on CUDA).

    Returns:
        torch.optim.AdamW: the optimizer.
    """
    groups: dict[str, list[nn.Parameter]] = {name: [] for name in GROUPS}
    for name, parameter in named_parameters:
        if parameter.requires_grad:
            groups[parameter_group(name)].append(parameter)
    lrs = {name: MODULE_LRS[name] if module_lrs else lr for name in GROUPS[1:]}
    lrs["backbone"] = lr
    return torch.optim.AdamW(
        [{"params": ps, "lr": lrs[name], "name": name} for name, ps in groups.items() if ps],
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
        foreach=foreach,
    )


def _group_names(param_groups: Sequence[Mapping]) -> list[str]:
    """The names of an optimizer's groups; raises for groups :func:`build_optimizer` did not
    make."""
    if any("name" not in group for group in param_groups):
        raise ValueError(
            "the parameter groups have no name: build the optimizer by build_optimizer"
        )
    return [group["name"] for group in param_groups]


def group_grad_norms(
    param_groups: Sequence[Mapping],
    *,
    sharded: Iterable[torch.Tensor] = (),
    all_reduce: Callable[[torch.Tensor], None] | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    """The L2 gradient norm of every optimizer group, ``{name: norm}``.

    Squared sums are accumulated in float32. Those of the ``sharded`` parameters (FSDP shards) are
    reduced with ``all_reduce`` (an in-place SUM over the shard group); the others count as they
    are.

    Args:
        param_groups (Sequence[Mapping]): the groups of an optimizer of :func:`build_optimizer`
            (each carries its ``name``).
        sharded (Iterable[Tensor]): the parameters whose gradients are shards.
        all_reduce (Callable | None): the reduction of the sharded sums.
        device (torch.device | None): where the sums live (default: the first parameter's device).

    Returns:
        dict[str, float]: the norm per group.
    """
    sharded_ids = {id(p) for p in sharded}
    names = _group_names(param_groups)
    if device is None:
        device = next((p.device for g in param_groups for p in g["params"]), torch.device("cpu"))
    sharded_squares = torch.zeros(len(names), device=device, dtype=torch.float32)
    local_squares = torch.zeros(len(names), device=device, dtype=torch.float32)
    for i, group in enumerate(param_groups):
        for p in group["params"]:
            if p.grad is not None:
                squares = sharded_squares if id(p) in sharded_ids else local_squares
                squares[i] += p.grad.detach().float().pow(2).sum()
    if all_reduce is not None:
        all_reduce(sharded_squares)
    norms = (sharded_squares + local_squares).sqrt().tolist()
    return dict(zip(names, norms))


def clip_per_group_(
    param_groups: Sequence[Mapping], group_norms: Mapping[str, float], max_grad_norm: float
) -> float:
    """Clip each group's gradients to ``max_grad_norm`` on its own, in place.

    ``coef = min(1, max_grad_norm / (norm + 1e-6))``; a group is scaled only when ``coef < 1``, so a
    module on another loss scale throttles only itself.

    Args:
        param_groups (Sequence[Mapping]): the groups of an optimizer of :func:`build_optimizer`.
        group_norms (Mapping[str, float]): each group's norm (:func:`group_grad_norms`).
        max_grad_norm (float): the threshold of each group.

    Returns:
        float: the global norm before the clip.
    """
    for name, group in zip(_group_names(param_groups), param_groups):
        coef = min(1.0, float(max_grad_norm) / (group_norms[name] + 1e-6))
        if coef < 1.0:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(coef)
    return math.sqrt(sum(v * v for v in group_norms.values()))
