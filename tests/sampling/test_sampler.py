"""The block-causal sampler: the order of its generator calls and of its draws."""

import pytest
import torch

from tests.sampling.support import (
    LATENT,
    TOKENS,
    RecordingGenerator,
    indexed_conditions,
    latents,
    tiny_sampler,
)
from worldcast.sampling.sampler import FRAMEWISE_KEYS, Sampler, framewise_conditions
from worldcast.sampling.schedulers import draw_noise, flow_to_x0, noise_context
from worldcast.sampling.window import WindowLayout

INPUT_DTYPE = RecordingGenerator.input_dtype
#: The timesteps of the denoising steps as the bf16 generator sees them.
TIMESTEPS = [1000.0, 936.0, 832.0, 624.0]


def _drew(sampler: Sampler, seed: int, *draws: tuple[int, torch.dtype]) -> bool:
    """Whether the sampler's stream, seeded with ``seed``, made exactly the draws ``(latent frames,
    dtype)``."""
    rng = torch.Generator().manual_seed(seed)
    for frames, dtype in draws:
        draw_noise(torch.empty(frames, *LATENT, dtype=dtype), rng)
    return torch.equal(sampler.rng.get_state(), rng.get_state())


def test_the_papers_sampler():
    """Four denoising steps on the shifted schedule and context writes at noise 16."""
    sampler, _ = tiny_sampler()
    timesteps = [round(float(t), 2) for t in sampler.denoising_timesteps]
    assert timesteps == [1000.0, 937.5, 833.33, 625.0]
    assert sampler.context_noise == 16


def test_framewise_conditions_are_views_of_the_calls_frames():
    conditions = indexed_conditions(9)
    call = framewise_conditions(conditions, frame_offset=5, num_frames=4)
    assert call.keys() == conditions.keys()
    for key in FRAMEWISE_KEYS:
        assert call[key][0, :, 0].tolist() == [5.0, 6.0, 7.0, 8.0]
        assert call[key].data_ptr() == conditions[key][:, 5:].data_ptr()
    assert call["view_deltas"] is conditions["view_deltas"]
    unchanged = framewise_conditions({"view_deltas": 1}, frame_offset=5, num_frames=4)
    assert unchanged == {"view_deltas": 1}


def test_predict_flow_is_the_generator_on_the_calls_frames_of_the_window():
    sampler, generator = tiny_sampler()
    sampler.cache.end = 5 * TOKENS
    x, t = latents(4, seed=1), torch.full((1, 4), 937.5)
    flow = sampler.predict_flow(x, t, indexed_conditions(9), frame_offset=5)
    (call,) = generator.calls
    assert call.summary == (5, 4, 937.5) and call.cached == 5
    assert call.x is x and call.timestep is t  # as given: only `predict_x0` casts
    assert call.conditions[FRAMEWISE_KEYS[0]].shape == (1, 4, 3)
    assert sampler.cache.end == 9 * TOKENS  # the generator moves the end
    assert torch.equal(flow, generator.flow(x, t))


def test_predict_x0_casts_its_inputs_and_converts_the_flow_at_the_cast_timestep():
    sampler, generator = tiny_sampler()
    sampler.cache.end = 5 * TOKENS
    x, t = latents(4, seed=1), torch.full((1, 4), 937.5)
    x0 = sampler.predict_x0(x, t, indexed_conditions(9), frame_offset=5)
    (call,) = generator.calls
    assert call.summary == (5, 4, 936.0)
    assert call.x.dtype == call.timestep.dtype == INPUT_DTYPE and torch.equal(
        call.x, x.to(INPUT_DTYPE)
    )
    flow = generator.flow(x.to(INPUT_DTYPE), t.to(INPUT_DTYPE))
    sigma = sampler.scheduler.sigmas[255].double()  # the entry nearest 936, not 937.5
    assert x0.dtype == torch.float32
    assert torch.equal(x0, (x.to(INPUT_DTYPE).double() - sigma * flow.double()).float())


def test_write_context_writes_clean_latents_at_the_context_noise():
    sampler, generator = tiny_sampler(seed=3)
    clean = latents(4, seed=2, dtype=INPUT_DTYPE)
    sampler.cache.end = TOKENS
    assert sampler.write_context(clean, indexed_conditions(5), frame_offset=1) is None
    want, _ = noise_context(
        clean, sampler.scheduler, context_noise=16, rng=torch.Generator().manual_seed(3)
    )
    (call,) = generator.calls
    assert call.summary == (1, 4, 16.0) and torch.equal(call.x, want)
    assert _drew(sampler, 3, (4, INPUT_DTYPE))

    # a given noise replaces the draw
    noise = latents(4, seed=4)[0]
    sampler.write_context(clean, indexed_conditions(5), frame_offset=1, noise=noise)
    want, _ = noise_context(clean, sampler.scheduler, context_noise=16, noise=noise)
    assert torch.equal(generator.calls[1].x, want)
    assert _drew(sampler, 3, (4, INPUT_DTYPE))


def test_a_clean_context_is_written_at_t_0_without_a_draw():
    sampler, generator = tiny_sampler(seed=3, context_noise=0)
    clean = latents(4, seed=2, dtype=INPUT_DTYPE)
    sampler.cache.end = TOKENS
    sampler.write_context(clean, indexed_conditions(5), frame_offset=1)
    (call,) = generator.calls
    assert call.summary == (1, 4, 0.0) and torch.equal(call.x, clean)
    assert _drew(sampler, 3)


def test_denoise_runs_the_denoising_steps_on_one_range_and_retests_after_the_first():
    sampler, generator = tiny_sampler(seed=4)
    sampler.cache.end = TOKENS
    noisy, conditions = latents(4, seed=5, dtype=INPUT_DTYPE), indexed_conditions(5)
    retested = {**conditions, "retested": True}
    first_x0 = []

    def retest(x0):
        first_x0.append(x0)
        return retested

    x0 = sampler.denoise(noisy, conditions, frame_offset=1, retest=retest)
    calls = generator.calls
    assert [call.summary for call in calls] == [(1, 4, t) for t in TIMESTEPS]
    assert [call.cached for call in calls] == [1, 5, 5, 5]  # the range is rewritten in place
    assert torch.equal(calls[0].x, noisy) and all(call.x.dtype == INPUT_DTYPE for call in calls)
    assert ["retested" in call.conditions for call in calls] == [False, True, True, True]

    def step_x0(call):
        return flow_to_x0(
            generator.flow(call.x, call.timestep), call.x, call.timestep, sampler.scheduler
        )

    assert len(first_x0) == 1 and torch.equal(first_x0[0], step_x0(calls[0]))
    assert x0.dtype == torch.float32 and torch.equal(x0, step_x0(calls[3]))
    # three re-noises of a float32 x0 between the four steps
    assert _drew(sampler, 4, *[(4, torch.float32)] * 3)


def test_prefill_writes_the_context_range_by_range():
    """The first frame, then blocks, each re-noised with its own noise; with
    ``context_write_noise`` the sampler's stream is not drawn from."""
    sampler, generator = tiny_sampler(seed=6)
    context = latents(17, seed=8)
    ranges = WindowLayout().context_ranges  # of the paper's 21 latent frames
    asked = []

    def context_write_noise(i, shape):
        asked.append((i, tuple(shape)))
        return torch.full(shape, float(i))

    sampler.prefill_context(
        context, indexed_conditions(21), ranges=ranges, context_write_noise=context_write_noise
    )
    assert [call.summary for call in generator.calls] == [
        (0, 1, 16.0),
        (1, 4, 16.0),
        (5, 4, 16.0),
        (9, 4, 16.0),
        (13, 4, 16.0),
    ]
    assert [call.cached for call in generator.calls] == [0, 1, 5, 9, 13]
    assert asked == [(0, (1, *LATENT)), *[(i, (4, *LATENT)) for i in range(1, 5)]]
    want, _ = noise_context(
        context[:, 5:9], sampler.scheduler, context_noise=16, noise=torch.full((4, *LATENT), 2.0)
    )
    assert torch.equal(generator.calls[2].x, want.to(INPUT_DTYPE))
    assert _drew(sampler, 6)
    with pytest.raises(ValueError, match="the ranges cover 17 latent frames, the context has 13"):
        sampler.prefill_context(context[:, :13], indexed_conditions(21), ranges=ranges)


def test_a_rollout_draws_once_per_context_write_and_three_times_per_block():
    """The first frame and two blocks, each denoised and written at its position: 1 + 2 (3 + 1)
    draws; the paper's first six blocks make 25."""
    sampler, generator = tiny_sampler(seed=5)
    noise, first = latents(8, seed=6, dtype=INPUT_DTYPE), latents(1, seed=7, dtype=INPUT_DTYPE)
    conditions = indexed_conditions(9)
    sampler.write_context(first, conditions, frame_offset=0)
    for start in (1, 5):
        x0 = sampler.denoise(noise[:, start - 1 : start + 3], conditions, frame_offset=start)
        sampler.write_context(x0, conditions, frame_offset=start)
    assert [call.summary for call in generator.calls] == [
        (0, 1, 16.0),
        *[(1, 4, t) for t in TIMESTEPS],
        (1, 4, 16.0),
        *[(5, 4, t) for t in TIMESTEPS],
        (5, 4, 16.0),
    ]
    assert sampler.cache.end == 9 * TOKENS
    block = [(4, torch.float32)] * 4
    assert _drew(sampler, 5, (1, INPUT_DTYPE), *block, *block)
