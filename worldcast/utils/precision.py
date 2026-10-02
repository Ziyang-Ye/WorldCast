"""Numerics of the generator: TF32 for fp32 CUDA work, bf16 parameters and bf16 inputs.

docs/inference.md, "Numerics that the paper's numbers depend on": the generator sees every floating
input in bf16, timesteps included (936 for 937.5), and the flow -> x0 conversion uses those values.
"""

import dataclasses
from collections import OrderedDict
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import nn

__all__ = [
    "GENERATOR_DTYPE",
    "cast_floating_tensors",
    "cast_parameters_bf16",
    "enable_tf32",
    "generator_autocast",
]

#: Dtype of the generator's parameters and inputs.
GENERATOR_DTYPE = torch.bfloat16


def enable_tf32() -> None:
    """Allow TF32 for fp32 CUDA matmuls and cuDNN convolutions (process-wide)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def generator_autocast(x: torch.Tensor) -> AbstractContextManager:
    """CUDA autocast to the dtype of ``x`` when it is fp16 or bf16; a no-op otherwise."""
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=x.dtype)
    return nullcontext()


def cast_parameters_bf16(module: nn.Module) -> nn.Module:
    """Cast the floating parameters and buffers of ``module`` to bf16 in place.

    Plain tensor attributes, such as the generator's complex128 RoPE table, keep their dtype.
    """
    return module.to(dtype=GENERATOR_DTYPE)


def cast_floating_tensors(obj: Any, dtype: torch.dtype = GENERATOR_DTYPE) -> Any:
    """Cast every floating tensor inside ``obj`` to ``dtype``, rebuilding the containers.

    Walks dataclasses, (ordered) dicts, namedtuples, lists, tuples and sets as FSDP's root input
    cast does (``torch.distributed.utils._apply_to_tensors``). Any other object, and a tensor
    already in ``dtype`` or not floating, is returned as it is.
    """
    if isinstance(obj, torch.Tensor):
        if not torch.is_floating_point(obj) or obj.dtype == dtype:
            return obj
        return obj.to(dtype)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = dataclasses.fields(obj)
        return dataclasses.replace(
            obj, **{f.name: cast_floating_tensors(getattr(obj, f.name), dtype) for f in fields}
        )
    if isinstance(obj, OrderedDict):
        return obj.__class__(
            (key, cast_floating_tensors(value, dtype)) for key, value in obj.items()
        )
    if isinstance(obj, dict):
        return {key: cast_floating_tensors(value, dtype) for key, value in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(cast_floating_tensors(item, dtype) for item in obj))
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(cast_floating_tensors(item, dtype) for item in obj)
    return obj
