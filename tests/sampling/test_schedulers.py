"""The flow-matching table, the ladder, the entry noise and the input cast of a call."""

import pytest
import torch
from torch.distributed.utils import _cast_forward_inputs

from tests.sampling.fakes import assert_same
from worldcast import sampling as S
from worldcast.utils.precision import cast_floating_tensors

BF16 = torch.bfloat16


@pytest.fixture(scope="module")
def new_sched():
    return S.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)


def test_from_config_equals_default():
    from worldcast.config.inference import SchedulerConfig

    a = S.FlowMatchScheduler.from_config(SchedulerConfig())
    b = S.FlowMatchScheduler()
    assert_same(a.sigmas, b.sigmas)
    assert_same(a.timesteps, b.timesteps)


def test_snapped_levels(new_sched):
    # docs/inference.md, "Numerics": 16 -> index 997 (t ~ 14.82, sigma 0.014822); bf16 rungs
    # 936/832/624
    idx = new_sched.timestep_index(torch.tensor([1000.0, 936.0, 832.0, 624.0, 16.0], dtype=BF16))
    assert idx.tolist() == [0, 255, 502, 751, 997]
    assert abs(float(new_sched.timesteps[997]) - 14.82) < 0.01
    assert abs(float(new_sched.sigmas[997]) - 0.014822) < 1e-6


def test_warp_values(new_sched):
    lad = S.warped_ladder([1000, 750, 500, 250], new_sched)
    assert lad.tolist() == [1000.0, 937.5, pytest.approx(833.3333, abs=1e-3), 625.0]
    # the warped ladder the paper's clients recorded
    assert [round(float(v), 6) for v in lad] == [1000.0, 937.5, 833.333313, 625.0]
    with pytest.raises(NotImplementedError):
        S.warped_ladder([1000], new_sched, warp=False)


def test_renoise_clean_write_not_ported(new_sched):
    with pytest.raises(NotImplementedError):
        S.renoise_for_cache(torch.zeros(1, 1, 2, 2, 2), context_noise=0, scheduler=new_sched)


@pytest.mark.parametrize(
    "x_dtype, t",
    [
        (torch.float32, torch.tensor([[937.5, 833.3333]])),
        (BF16, torch.tensor([[16.0]])),
        (torch.float32, torch.tensor([[0, 0]], dtype=torch.int64)),
    ],
)
def test_input_cast_matches_fsdp_root(x_dtype, t):
    x = torch.randn(1, t.shape[1], 2, 2, 2).to(x_dtype)
    (want_x, want_t), _ = _cast_forward_inputs(BF16, x, t)
    got_x, got_t = cast_floating_tensors((x, t), BF16)  # the sampler's cast of a call
    assert_same(got_x, want_x)
    assert_same(got_t, want_t)
    if t.dtype == torch.float32 and t.numel() == 2:
        assert got_t.tolist() == [[936.0, 832.0]]


def test_entry_noise():
    shape = (3, 4, 6)
    seed, requested, effective = 20260917, 41, 29
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise_full = torch.randn((1, requested - 1, *shape), generator=g)
    want = noise_full[:, : effective - 1].to(None, BF16)
    got = S.entry_noise(seed, requested, effective, shape, dtype=BF16)
    assert_same(got, want)
