"""The numerics of the generator (docs/inference.md, "Numerics"): bf16 casts and TF32.

The reference for the input cast is torch's own ``torch.distributed.utils._cast_forward_inputs``,
the cast of FSDP's root (``cast_root_forward_inputs=True``).
"""

from collections import OrderedDict, namedtuple

import pytest
import torch
from torch.distributed.utils import _cast_forward_inputs

from worldcast.modeling.rays import RayConditions
from worldcast.modeling.wan22.model import GeneratorConditions, KVCache
from worldcast.utils import precision

Pair = namedtuple("Pair", "a b")


def _same(x, y):
    """Structural equality with exact tensors (dtype included)."""
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and torch.equal(x, y)
    if isinstance(x, GeneratorConditions) or isinstance(x, RayConditions):
        return type(x) is type(y) and all(
            _same(getattr(x, f), getattr(y, f)) for f in x.__dataclass_fields__
        )
    if isinstance(x, dict):
        return type(x) is type(y) and x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    return x is y or x == y


def _conditions():
    g = torch.Generator().manual_seed(0)
    c2w = torch.randn(1, 7, 4, 4, generator=g) * 300
    return GeneratorConditions(
        prompt_embeds=torch.randn(1, 5, 8, generator=g),
        buttons=torch.randint(0, 2, (1, 25, 11), generator=g).float(),
        camera=torch.randn(1, 25, 2, generator=g),
        weapon=torch.randint(0, 52, (1, 25), generator=g),
        obs_flash_flag=torch.randint(0, 2, (1, 7), generator=g),
        obs_flash_valid=torch.ones(1, 7, dtype=torch.bool),
        obs_scope_on=torch.zeros(1, 7),
        obs_scope_level=torch.zeros(1, 7, dtype=torch.long),
        obs_scope_valid=torch.ones(1, 7),
        state_field=lambda offset, frames: torch.zeros(1, 7, 23, 2, 3),
        rays=RayConditions(
            frame_c2w=c2w,
            frame_tans=torch.rand(1, 7, 2, generator=g) + 0.5,
            anchor_c2w=c2w[:, 5],
            memory_c2w=c2w[:, 1:3],
            memory_frames=torch.tensor([[1, 2]]),
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
    kwargs = {"current_start": 1260, "extra": {"x": torch.randn(2, generator=g).to(torch.float16)}}
    ref_args, ref_kwargs = _cast_forward_inputs(torch.bfloat16, *args, **kwargs)
    out_args, out_kwargs = precision.cast_floating_tensors((args, kwargs))
    assert _same(tuple(out_args), tuple(ref_args)) and _same(out_kwargs, ref_kwargs)
    assert out_args[2].state_field is args[2].state_field  # callables pass through untouched


def test_bf16_timesteps_are_what_the_model_sees():
    warped = torch.tensor([[1000.0, 937.5, 833.3333, 625.0]])  # the warped 4-step ladder
    t = precision.cast_floating_tensors(warped)
    assert t.dtype == torch.bfloat16
    assert t.float().tolist() == [[1000.0, 936.0, 832.0, 624.0]]
    assert precision.cast_floating_tensors(torch.tensor([16.0])).float().item() == 16.0


def test_kv_cache_passes_through_the_casts():
    """The generator writes into the caller's cache: an input cast must hand on the same object."""
    cache = KVCache.allocate(
        num_blocks=1, num_heads=1, head_dim=2, capacity_latents=1, frame_seq_length=2
    )
    assert precision.cast_floating_tensors({"kv_cache": cache})["kv_cache"] is cache
    _, kwargs = _cast_forward_inputs(torch.float32, kv_cache=cache)
    assert kwargs["kv_cache"] is cache


def test_enable_tf32():
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        precision.enable_tf32()
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def test_cast_parameters_bf16_keeps_the_rope_table(tiny):
    from worldcast.modeling.wan22.model import WorldCastGenerator

    model = WorldCastGenerator(tiny.tiny_config())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    freqs = model.freqs
    precision.cast_parameters_bf16(model)
    assert model.freqs is freqs and freqs.dtype == torch.complex128
    for key, value in model.state_dict().items():
        assert value.dtype == torch.bfloat16 and torch.equal(
            value, before[key].to(torch.bfloat16)
        ), key
