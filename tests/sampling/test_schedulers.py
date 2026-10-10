"""The flow-matching table, the denoising steps, flow <-> x0, the re-noise and the entry noise."""

import numpy as np
import pytest
import torch

from worldcast.sampling import schedulers as S

BF16 = torch.bfloat16


@pytest.fixture(scope="module")
def scheduler():
    return S.FlowMatchScheduler()


def test_the_timestep_shift_of_a_float_an_array_and_a_tensor():
    assert S.shift_sigma(0.5, 5.0) == pytest.approx(5 / 6)
    assert S.shift_sigma(np.array([0.0, 1.0]), 5.0).tolist() == [0.0, 1.0]
    assert S.shift_sigma(torch.tensor(0.75), 5.0).item() == pytest.approx(0.9375)


def test_table_is_wan22s(scheduler):
    """``sigma_i = 5 s / (1 + 4 s)`` with ``s = 1 - i / 1000``; the timestep is ``1000 sigma``."""
    assert scheduler.sigmas.shape == (1000,) and scheduler.sigmas.dtype == torch.float32
    assert scheduler.sigmas[0] == 1.0 and scheduler.timesteps[0] == 1000.0
    assert float(scheduler.sigmas[500]) == pytest.approx(5 / 6, abs=1e-6)
    assert float(scheduler.timesteps[999]) == pytest.approx(1000 * 0.005 / 1.004, abs=1e-4)
    assert bool((scheduler.sigmas[1:] < scheduler.sigmas[:-1]).all())


def test_timesteps_snap_to_the_nearest_table_entry(scheduler):
    """docs/inference.md, "Numerics": the bf16 denoising steps 1000 / 936 / 832 / 624 read the
    entries 0 / 255 / 502 / 751, and the context noise 16 the entry 997 (t ~ 14.82)."""
    t = torch.tensor([1000.0, 936.0, 832.0, 624.0, 16.0], dtype=BF16)
    for dtype in (torch.float32, torch.float64):
        assert S.nearest_index(t[None], scheduler, dtype).tolist() == [0, 255, 502, 751, 997]
        sigma = S.table_sigma(t, scheduler, t, dtype=dtype)
        assert sigma.dtype == dtype
        assert torch.equal(sigma, scheduler.sigmas[[0, 255, 502, 751, 997]].to(dtype))
    assert float(scheduler.timesteps[997]) == pytest.approx(14.82, abs=0.01)
    assert float(scheduler.sigmas[997]) == pytest.approx(0.014822, abs=1e-6)
    assert S.table_sigma(t[None], scheduler, torch.zeros(1, 5, 2, 3, 4)).shape == (1, 5, 1, 1, 1)


def test_the_timesteps_of_the_denoising_steps(scheduler):
    timesteps = S.shift_denoising_steps([1000, 750, 500, 250], scheduler)
    assert timesteps.dtype == torch.float32
    # the timesteps the paper's clients recorded
    assert [round(float(v), 6) for v in timesteps] == [1000.0, 937.5, 833.333313, 625.0]
    assert S.shift_denoising_steps([0], scheduler).tolist() == [0.0]
    with pytest.raises(ValueError, match=r"lie in \[0, 1000\], got \[1001\]"):
        S.shift_denoising_steps([1001], scheduler)


def test_flow_to_x0_at_the_snapped_level(scheduler):
    """``x0 = x_t - sigma flow`` in float64, rounded once to the flow's dtype."""
    g = torch.Generator().manual_seed(0)
    flow = torch.randn(1, 2, 3, 4, 4, generator=g)
    xt = torch.randn(1, 2, 3, 4, 4, generator=g).to(BF16)
    t = torch.tensor([[936.0, 16.0]], dtype=BF16)
    sigma = scheduler.sigmas[[255, 997]].double().view(1, 2, 1, 1, 1)
    want = (xt.double() - sigma * flow.double()).float()
    assert torch.equal(S.flow_to_x0(flow, xt, t, scheduler), want)


def test_x0_to_flow_inverts_flow_to_x0(scheduler):
    g = torch.Generator().manual_seed(4)
    xt, flow = torch.randn(6, 4, 3, 5, generator=g), torch.randn(6, 4, 3, 5, generator=g)
    t = scheduler.timesteps[torch.tensor([10, 200, 400, 600, 800, 950])]
    x0 = S.flow_to_x0(flow, xt, t, scheduler)
    assert torch.allclose(S.x0_to_flow(x0, xt, t, scheduler), flow, atol=1e-4)


def test_add_noise_and_the_context_write(scheduler):
    """A context write is ``(1 - sigma_997) x + sigma_997 eps`` in the latents' dtype, labelled
    t = 16; a clean context (context noise 0) is the latents themselves at t = 0, without a
    draw."""
    g = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 2, 3, 4, 4, generator=g).to(BF16)
    noise = torch.randn(2, 3, 4, 4, generator=g)
    sigma = scheduler.sigmas[997].view(1, 1, 1, 1)  # float32: the sum is formed in float32
    want = ((1 - sigma) * latents[0] + sigma * noise.to(BF16)).to(BF16)
    noised, timestep = S.noise_context(latents, scheduler, context_noise=16, noise=noise)
    assert noised.dtype == BF16 and torch.equal(noised[0], want)
    assert timestep.dtype == torch.float32 and timestep.tolist() == [[16.0, 16.0]]
    with pytest.raises(ValueError, match="noise"):
        S.noise_context(latents, scheduler, context_noise=16, noise=noise[:1])
    with pytest.raises(ValueError, match=r"\[N, C, H, W\] frames and N timesteps"):
        scheduler.add_noise(latents, latents, timestep)  # [B, F, ...] frames are flattened first
    torch.manual_seed(0)
    state = torch.get_rng_state()
    clean, timestep = S.noise_context(latents, scheduler, context_noise=0)
    assert clean is latents and timestep.dtype == torch.float32
    assert timestep.tolist() == [[0.0, 0.0]] and torch.equal(torch.get_rng_state(), state)


def test_the_context_write_draws_once_from_its_stream(scheduler):
    latents = torch.zeros(1, 4, 2, 3, 3)
    drawn, _ = S.noise_context(
        latents, scheduler, context_noise=16, rng=torch.Generator().manual_seed(5)
    )
    eps = torch.randn(4, 2, 3, 3, generator=torch.Generator().manual_seed(5))
    assert torch.equal(drawn[0], scheduler.sigmas[997] * eps)


def test_denoising_steps_alternate_calls_and_draws(scheduler):
    """Step k predicts x0 from x_{t_k}; before the exit step x0 is re-noised to t_{k+1}: call,
    draw, call, draw, call, and the exit's input, timestep and x0 are returned."""
    steps = S.shift_denoising_steps([1000, 750, 500, 250], scheduler)
    noisy = torch.randn(1, 4, 2, 3, 3, generator=torch.Generator().manual_seed(2))
    seen = []

    def call(x_t, t):
        seen.append((x_t.clone(), t.clone()))
        return 0.5 * x_t.float()

    rng = torch.Generator().manual_seed(3)
    out = S.run_denoising_steps(call, noisy, steps, scheduler, exit_step=2, rng=rng)
    assert [t.unique().tolist() for _, t in seen] == [[1000.0], [937.5], [steps[2].item()]]
    assert torch.equal(seen[0][0], noisy)
    replay = torch.Generator().manual_seed(3)
    x_t = noisy
    for k in (1, 2):  # the re-noise of step k - 1's x0 to step k
        x_t = S.renoise_frames(0.5 * x_t, seen[k][1], scheduler, rng=replay)
        assert torch.equal(seen[k][0], x_t)
    assert torch.equal(out.input, x_t) and torch.equal(out.x0, 0.5 * x_t)
    assert torch.equal(out.timestep, seen[2][1])
    with pytest.raises(ValueError, match="exit step"):
        S.run_denoising_steps(call, noisy, steps, scheduler, exit_step=4)


def test_denoise_block_casts_the_renoised_inputs(scheduler):
    """``denoise_block`` runs every step and hands each re-noised input over in the entry noise's
    dtype."""
    steps = S.shift_denoising_steps([1000, 750, 500, 250], scheduler)
    noisy = torch.randn(1, 4, 2, 3, 3).to(BF16)
    dtypes = []

    def call(x_t, t):
        dtypes.append(x_t.dtype)
        return x_t.float()

    x0 = S.denoise_block(call, noisy, steps, scheduler, rng=torch.Generator().manual_seed(0))
    assert dtypes == [BF16] * 4 and x0.dtype == torch.float32


def test_paired_context_noise_is_keyed_by_seed_block_and_role():
    """The first frame and the recent ranges draw the same noise with or without a memory entry;
    the memory range has its own key."""
    shape = (4, 2, 3, 3)
    with_memory = S.paired_context_noise(7, 25, num_ranges=5, num_recent_blocks=3)
    without = S.paired_context_noise(7, 25, num_ranges=4, num_recent_blocks=3)
    first = with_memory(0, shape)
    assert first.shape == shape and first.dtype == torch.float32 and first.device.type == "cpu"
    assert torch.equal(first, without(0, shape))  # the first frame
    for k in range(3):  # recent0 .. recent2
        assert torch.equal(with_memory(2 + k, shape), without(1 + k, shape))
    memory = with_memory(1, shape)
    assert all(not torch.equal(memory, with_memory(i, shape)) for i in (0, 2, 3, 4))
    # the key of a role, pinned by value: another key is another noise for every later block
    assert S.context_noise_seed(0, 25, "sink") == 5127211072542556352
    assert S.context_noise_seed(7, 29, "recent0") == 2402999386936367769
    assert S.context_noise_seed(7, 29, "slot0") == 7825267613107456043
    for i, role in enumerate(("sink", "slot0", "recent0", "recent1", "recent2")):
        g = torch.Generator().manual_seed(S.context_noise_seed(7, 25, role))
        assert torch.equal(with_memory(i, shape), torch.randn(shape, generator=g)), role
    assert not torch.equal(memory, S.paired_context_noise(7, 29, 5, 3)(1, shape))
    assert not torch.equal(memory, S.paired_context_noise(8, 25, 5, 3)(1, shape))


def test_entry_noise_is_the_requested_draw_cut_to_the_rollout():
    """One draw of a CPU generator seeded with the seed, for the requested length, then cut: a
    shorter rollout keeps the first frames of the longer one's noise, and the global RNG is not
    touched."""
    shape = (3, 4, 6)
    torch.manual_seed(0)
    state = torch.get_rng_state()
    full = S.entry_noise(20260917, 41, 41, shape, dtype=BF16)
    cut = S.entry_noise(20260917, 41, 29, shape, dtype=BF16)
    assert torch.equal(torch.get_rng_state(), state)
    assert full.shape == (1, 40, *shape) and full.dtype == BF16
    assert cut.shape == (1, 28, *shape) and torch.equal(cut, full[:, :28])
    assert not torch.equal(S.entry_noise(20260918, 41, 29, shape, dtype=BF16), cut)
