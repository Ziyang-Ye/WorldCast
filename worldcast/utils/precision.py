"""Numerics of the generator: TF32 for fp32 CUDA work, bf16 parameters and bf16 inputs.

docs/inference.md, "Numerics": the generator sees every floating input in bf16, timesteps included
(936 for 937.5), and the flow -> x0 conversion uses those values.
"""

import dataclasses
from collections import OrderedDict
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

__all__ = [
    "GENERATOR_DTYPE",
    "HALF_DTYPES",
    "cast_floating_tensors",
    "enable_tf32",
    "fp32_island",
    "generator_autocast",
    "generator_dtype",
]

#: Dtype of the generator's parameters and inputs on CUDA, as trained.
GENERATOR_DTYPE = torch.bfloat16
#: The half-precision dtypes: the generator's CUDA autocast runs in the one its input has, and
#: the flash kernels take no other.
HALF_DTYPES = (torch.float16, torch.bfloat16)


def generator_dtype(device: torch.device | str) -> torch.dtype:
    """The dtype the generator runs in on ``device``: bf16 on CUDA, float32 elsewhere (bf16
    parameters need CUDA autocast)."""
    return GENERATOR_DTYPE if torch.device(device).type == "cuda" else torch.float32


def enable_tf32(*, matmul: bool = True) -> None:
    """Allow TF32 for cuDNN convolutions and, with ``matmul``, for fp32 CUDA matmuls
    (process-wide), as the reference runs did: without it the generator's fp32 regions and the
    depth head round differently. ``matmul=False`` turns it off for the matmuls, as an evaluation
    protocol may ask."""
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = True


def generator_autocast(x: torch.Tensor) -> AbstractContextManager:
    """CUDA autocast to the dtype of ``x`` when it is fp16 or bf16; a no-op otherwise."""
    if x.is_cuda and x.dtype in HALF_DTYPES:
        return torch.autocast(device_type="cuda", dtype=x.dtype)
    return nullcontext()


def fp32_island() -> AbstractContextManager:
    """An fp32 region inside the generator's CUDA autocast: the matmuls in it run in fp32, as
    trained; a no-op on a machine without CUDA."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float32)
    return nullcontext()


def cast_floating_tensors(obj: Any, dtype: torch.dtype = GENERATOR_DTYPE) -> Any:
    """Cast every floating tensor inside ``obj`` to ``dtype``, rebuilding the containers.

    Walks dataclasses, (ordered) dicts, namedtuples, lists, tuples and sets, the containers the
    input cast of FSDP's mixed precision walks. Any other object, and a tensor already in ``dtype``
    or not floating, is returned as it is.
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
