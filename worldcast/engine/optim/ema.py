"""The fp32 EMA of the generator, ``s = d s + (1 - d) p``, kept per FSDP shard."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch
from torch import nn

from worldcast.distributed.fsdp import full_state_dict, live_module

__all__ = ["ShardedEMA"]


class ShardedEMA:
    """The EMA of a module's live parameters (its local shards under FSDP), beside each parameter.

    Built by :meth:`start` (from the live weights) or :meth:`resume` (from a checkpoint's shards).

    Args:
        decay (float): ``d``.
        shadow (dict[str, Tensor]): the EMA of every live parameter, fp32, on its device.
    """

    def __init__(self, decay: float, shadow: dict[str, torch.Tensor]) -> None:
        self.decay = float(decay)
        self.shadow = shadow

    @classmethod
    @torch.no_grad()
    def start(cls, module: nn.Module, decay: float) -> "ShardedEMA":
        """An EMA whose shadow is an fp32 copy of the live parameters."""
        live = live_module(module).named_parameters()
        return cls(decay, {name: p.detach().clone().float() for name, p in live})

    @classmethod
    def resume(
        cls, module: nn.Module, decay: float, state: Mapping[str, torch.Tensor]
    ) -> "ShardedEMA":
        """The EMA of a checkpoint: ``state`` (:meth:`state_dict` of a run with the same topology)
        beside the module's parameters."""
        params = dict(live_module(module).named_parameters())
        if params.keys() != state.keys():
            raise RuntimeError("the saved EMA shards do not match this model's parameters")
        shadow = {k: v.detach().clone().float().to(params[k].device) for k, v in state.items()}
        return cls(decay, shadow)

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        """``s = d s + (1 - d) p`` per tensor."""
        d = self.decay
        for name, p in live_module(module).named_parameters():
            self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """This rank's shards on the CPU (its resume file)."""
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    @contextmanager
    def applied(self, module: nn.Module) -> Iterator[None]:
        """The EMA weights in place of the module's live parameters (its local shards) inside the
        block; the live weights are restored on exit."""
        params = dict(live_module(module).named_parameters())
        live = {name: p.detach().clone() for name, p in params.items()}
        try:
            with torch.no_grad():
                for name, p in params.items():
                    p.copy_(self.shadow[name].to(dtype=p.dtype, device=p.device))
            yield
        finally:
            with torch.no_grad():
                for name, p in params.items():
                    p.copy_(live[name])

    def full_state_dict(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """The consolidated EMA weights (:func:`worldcast.distributed.fsdp.full_state_dict`)."""
        with self.applied(module):
            return full_state_dict(module)
