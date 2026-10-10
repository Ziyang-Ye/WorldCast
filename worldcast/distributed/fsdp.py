"""FSDP as the paper's runs wrapped their models, and the helpers that read and reduce through it.

``use_orig_params``, ``limit_all_gathers``, and on CUDA mixed precision: bf16 parameters, fp32
gradient reduction and buffers. The root casts every floating forward input to bf16, timesteps
included. Every rank loads the same weights, so module states are not synchronised. Wrap ``size``:
every module whose unwrapped remainder holds at least :data:`MIN_UNIT_PARAMS` parameters is a unit
(on the 5B generator: the time projection, each block's FFN, then each block); wrap ``root``: the
model is one unit. Sharding ``full`` (FULL_SHARD) or ``hybrid_full`` (HYBRID_SHARD: shard within a
node, replicate across nodes). The helpers below take a plain module too, for a process alone.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

from worldcast.utils.precision import GENERATOR_DTYPE

__all__ = [
    "MIN_UNIT_PARAMS",
    "clip_grad_norm",
    "fsdp_wrap",
    "full_state_dict",
    "group_all_reduce",
    "is_fsdp",
    "live_module",
    "no_sync",
    "sharded_parameters",
]

#: Parameters of the smallest FSDP unit of the ``size`` wrap.
MIN_UNIT_PARAMS = 50_000_000
#: Module classes that never become a unit of their own.
_NEVER_UNITS = (nn.ModuleList, nn.ModuleDict)


def _size_policy() -> Callable[..., bool]:
    """The ``size`` wrap: torch's size-based policy with :data:`MIN_UNIT_PARAMS`."""
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    return partial(
        size_based_auto_wrap_policy,
        min_num_params=MIN_UNIT_PARAMS,
        exclude_wrap_modules=set(_NEVER_UNITS),
    )


def fsdp_wrap(module: nn.Module, *, sharding: str, wrap: str, device: torch.device) -> nn.Module:
    """Wrap ``module`` with FSDP (module docstring); mixed precision on CUDA only.

    Args:
        module (nn.Module): the module whose ``forward`` the trainer calls.
        sharding (str): ``full`` or ``hybrid_full``.
        wrap (str): ``size`` or ``root``.
        device (torch.device): the rank's device.

    Returns:
        nn.Module: the FSDP root.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    strategies = {"full": ShardingStrategy.FULL_SHARD, "hybrid_full": ShardingStrategy.HYBRID_SHARD}
    policies = {"size": _size_policy(), "root": None}
    if sharding not in strategies or wrap not in policies:
        raise ValueError(
            f"sharding is one of {tuple(strategies)} and wrap one of {tuple(policies)}; got"
            f" {sharding!r} and {wrap!r}"
        )
    # Attribution: this FSDP configuration (mixed precision, sharding map, size policy,
    # limit_all_gathers, sync_module_states) follows CausVid's fsdp_wrap
    # (github.com/tianweiy/CausVid at fab2440f, MIT) via Self Forcing
    # (github.com/guandeh17/Self-Forcing, Apache-2.0); the calls are PyTorch's FSDP API.
    precision = None
    if torch.device(device).type == "cuda":
        precision = MixedPrecision(
            param_dtype=GENERATOR_DTYPE,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    return FSDP(
        module,
        auto_wrap_policy=policies[wrap],
        sharding_strategy=strategies[sharding],
        mixed_precision=precision,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=False,
    )


def is_fsdp(module: nn.Module) -> bool:
    """Whether ``module`` is an FSDP root (else a plain module)."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return isinstance(module, FSDP)


def live_module(module: nn.Module) -> nn.Module:
    """The module whose ``named_parameters`` are this rank's live parameters (local shards)."""
    return module.module if is_fsdp(module) else module


def sharded_parameters(module: nn.Module) -> list[torch.Tensor]:
    """The parameters of the units that shard; empty for a plain module.

    Their squared gradient sums are reduced over the shard group, the others count as they are.
    """
    if not is_fsdp(module):
        return []
    out: list[torch.Tensor] = []
    for handle in module._all_handles:
        if handle.uses_sharded_strategy:
            out.extend(handle.flat_param._params)
    return out


def group_all_reduce(module: nn.Module) -> Callable[[torch.Tensor], None] | None:
    """In-place SUM over the module's FSDP shard group; ``None`` for a plain module."""
    if not is_fsdp(module):
        return None
    group = module.process_group
    return lambda tensor: dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)


def no_sync(module: nn.Module, enabled: bool) -> AbstractContextManager:
    """FSDP's ``no_sync`` for every micro-batch but the last of an accumulated step."""
    return module.no_sync() if enabled and is_fsdp(module) else nullcontext()


def clip_grad_norm(module: nn.Module, max_norm: float) -> torch.Tensor:
    """Clip the global gradient norm of one model; returns the norm before the clip."""
    if is_fsdp(module):
        return module.clip_grad_norm_(float(max_norm))
    params = [p for p in module.parameters() if p.grad is not None]
    return torch.nn.utils.clip_grad_norm_(params, float(max_norm))


def full_state_dict(module: nn.Module, *, on_every_rank: bool = False) -> dict[str, torch.Tensor]:
    """The consolidated state dict (a collective under FSDP).

    Args:
        module (nn.Module): a plain or FSDP-wrapped module.
        on_every_rank (bool): gather on every rank and leave the tensors on the module's device;
            by default on the CPU and, under FSDP, on rank 0 only (``{}`` elsewhere).
    """
    if not is_fsdp(module):
        state = module.state_dict()
        if on_every_rank:
            return {k: v.detach().clone() for k, v in state.items()}
        return {k: v.detach().cpu().clone() for k, v in state.items()}
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    # Attribution: the full-state-dict idiom as in CausVid's fsdp_state_dict
    # (github.com/tianweiy/CausVid at fab2440f, MIT) via Self Forcing
    # (github.com/guandeh17/Self-Forcing, Apache-2.0).
    policy = FullStateDictConfig(offload_to_cpu=not on_every_rank, rank0_only=not on_every_rank)
    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, policy):
        return module.state_dict()
