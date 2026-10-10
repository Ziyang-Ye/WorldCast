"""The numerics of the generator (docs/inference.md, "Numerics"): bf16 casts and TF32.

The reference for the input cast is torch's own ``torch.distributed.utils._cast_forward_inputs``,
the cast of FSDP's root (``cast_root_forward_inputs=True``).
"""

import dataclasses
import warnings
from collections import OrderedDict, namedtuple
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.distributed.utils import _cast_forward_inputs

from worldcast.utils import precision

Pair = namedtuple("Pair", "a b")


@dataclass
class Cameras:
    c2w: torch.Tensor
    positions: torch.Tensor


@dataclass
class Conditions:
    """A record as a module receives it: tensors of several dtypes, a callable and a nested
    record."""

    prompt_embeds: torch.Tensor
    weapon: torch.Tensor
    valid: torch.Tensor
    field: Callable
    cameras: Cameras


def _same(x, y):
    """Structural equality with exact tensors (dtype included)."""
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and torch.equal(x, y)
    if dataclasses.is_dataclass(x):
        return type(x) is type(y) and all(
            _same(getattr(x, f.name), getattr(y, f.name)) for f in dataclasses.fields(x)
        )
    if isinstance(x, dict):
        return type(x) is type(y) and x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    return x is y or x == y


def _conditions() -> Conditions:
    g = torch.Generator().manual_seed(0)
    return Conditions(
        prompt_embeds=torch.randn(1, 5, 8, generator=g),
        weapon=torch.randint(0, 52, (1, 25), generator=g),
        valid=torch.ones(1, 7, dtype=torch.bool),
        field=lambda frame_offset, num_frames: torch.zeros(1, num_frames, 23, 2, 3),
        cameras=Cameras(
            c2w=torch.randn(1, 7, 4, 4, generator=g) * 300, positions=torch.tensor([[1, 2]])
        ),
    )


def test_cast_matches_torch_root_cast():
    g = torch.Generator().manual_seed(1)
    args = (
        torch.randn(1, 4, 48, 3, 5, generator=g),
        torch.tensor([[1000.0, 937.5, 833.3333, 625.0]]),
        _conditions(),
        [
            torch.randn(3, generator=g),
            torch.arange(3),
            (torch.randn(2, dtype=torch.float64, generator=g),),
        ],
        OrderedDict(a=torch.randn(2, generator=g), b=torch.ones(2, dtype=torch.complex64)),
        Pair(torch.randn(2, generator=g), 7),
    )
    kwargs = {"frame_offset": 5, "extra": {"x": torch.randn(2, generator=g).to(torch.float16)}}
    ref_args, ref_kwargs = _cast_forward_inputs(torch.bfloat16, *args, **kwargs)
    out_args, out_kwargs = precision.cast_floating_tensors((args, kwargs))
    assert _same(tuple(out_args), tuple(ref_args)) and _same(out_kwargs, ref_kwargs)
    assert out_args[2].field is args[2].field  # callables pass through untouched
    assert out_args[2].cameras.c2w.dtype == torch.bfloat16
    assert out_args[2].weapon.dtype == out_args[2].cameras.positions.dtype == torch.long


def test_bf16_timesteps_are_what_the_model_sees():
    timesteps = torch.tensor([[1000.0, 937.5, 833.3333, 625.0]])  # of the denoising steps
    t = precision.cast_floating_tensors(timesteps)
    assert t.dtype == torch.bfloat16
    assert t.float().tolist() == [[1000.0, 936.0, 832.0, 624.0]]
    assert precision.cast_floating_tensors(torch.tensor([16.0])).float().item() == 16.0


def test_a_plain_object_passes_through_the_cast():
    """What is neither a tensor, a container nor a dataclass is handed on as it is (the generator
    writes into the caller's KV cache)."""

    class Buffers:
        def __init__(self) -> None:
            self.keys = [torch.zeros(2)]

    buffers = Buffers()
    assert precision.cast_floating_tensors({"kv_cache": buffers})["kv_cache"] is buffers
    assert buffers.keys[0].dtype == torch.float32


def test_the_generator_is_bf16_on_cuda_and_float32_elsewhere():
    assert precision.generator_dtype("cuda:1") == torch.bfloat16
    assert precision.generator_dtype(torch.device("cpu")) == torch.float32


def test_generator_autocast_is_a_no_op_off_cuda():
    """bf16 parameters autocast on CUDA alone: on the CPU the generator's matmuls stay in the
    dtype of their operands."""
    with precision.generator_autocast(torch.zeros(1, dtype=torch.bfloat16)):
        assert not torch.is_autocast_enabled("cpu")
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float32


def test_fp32_island_is_silent_and_a_no_op_off_cuda():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with precision.fp32_island():
            assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float32


def test_enable_tf32():
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        precision.enable_tf32()
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        precision.enable_tf32(matmul=False)  # the convolutions keep it
        assert not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
