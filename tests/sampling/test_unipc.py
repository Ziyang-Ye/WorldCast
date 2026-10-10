"""The 20-step UniPC solver: its timesteps, an exact solve of a straight flow, and the solve of a
flow that depends on the state."""

import pytest
import torch

from worldcast.sampling.unipc import FlowUniPCSolver


def test_timesteps_of_twenty_steps():
    solver = FlowUniPCSolver(20)
    assert solver.timesteps.tolist() == [
        999, 989, 978, 965, 952, 937, 920, 902, 882, 859,
        833, 803, 768, 728, 681, 624, 555, 468, 356, 208,
    ]  # fmt: skip
    assert solver.sigmas.dtype == torch.float32 and solver.sigmas.shape == (21,)
    assert solver.sigmas[-1] == 0


def test_a_straight_flow_lands_on_its_x0():
    solver = FlowUniPCSolver(20)
    g = torch.Generator().manual_seed(3)
    x0, eps = torch.randn(1, 3, 4, 5, generator=g), torch.randn(1, 3, 4, 5, generator=g)
    x = (1 - solver.sigmas[0]) * x0 + solver.sigmas[0] * eps
    for _ in solver.timesteps:
        x = solver.step(eps - x0, x)
    assert torch.allclose(x, x0, atol=1e-5)


def test_predictor_and_corrector_on_a_state_dependent_flow():
    """Twenty steps on ``flow = tanh(x) t / 1000 + 0.3 x`` from seeded noise: each x0 estimate
    differs, so the second-order terms of the predictor and the corrector enter the result (on a
    straight flow they vanish). The values are those of diffusers' ``UniPCMultistepScheduler``
    (flow sigmas, shift 5, order 2, ``bh2``, final sigma zero) on the same flow."""
    solver = FlowUniPCSolver(20)
    x = torch.randn(1, 2, 3, 3, generator=torch.Generator().manual_seed(11))
    for t in solver.timesteps:
        x = solver.step(torch.tanh(x) * float(t) / 1000.0 + 0.3 * x, x)
    want = [
        -0.2314185, 0.4938212, -0.8565426, -0.8267643, -0.0686441, -0.5314081,
        0.6876467, 0.3176694, 0.0329923, -0.0818891, 0.3493915, 0.2751190,
        1.1057557, -0.5720267, -0.6623695, 0.5483575, 1.0341935, 0.1330088,
    ]  # fmt: skip
    torch.testing.assert_close(x.flatten(), torch.tensor(want), rtol=0, atol=1e-5)


def test_the_sample_keeps_its_dtype_and_a_solver_serves_one_solve():
    solver = FlowUniPCSolver(4)
    x = torch.randn(1, 2, 3, 3).to(torch.bfloat16)
    for _ in solver.timesteps:
        x = solver.step(torch.randn(1, 2, 3, 3), x)
        assert x.dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="a new solve needs a new solver"):
        solver.step(torch.randn(1, 2, 3, 3), x)
