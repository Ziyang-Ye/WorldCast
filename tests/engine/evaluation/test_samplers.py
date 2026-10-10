"""The three samplers' call patterns, on a fake generator."""

import pytest
import torch

from worldcast.engine.evaluation import (
    sample_bidirectional,
    sample_block_causal,
    sample_four_step,
)
from worldcast.modeling.wan22.dit import KVCache
from worldcast.sampling.sampler import Sampler


class FakeCausal:
    """Records ``(frames, timestep, window position)`` of each call and writes the cache."""

    input_dtype = None

    def __init__(self):
        self.calls = []

    def __call__(self, noisy, timestep, conditions, *, kv_cache, frame_offset):
        assert conditions["player_alive"].shape[1] == noisy.shape[1]  # framewise keys cut
        kv_cache.end = kv_cache.span(frame_offset * 252, noisy.shape[1] * 252)
        self.calls.append((noisy.shape[1], timestep.clone(), frame_offset))
        return 0.3 * noisy.float() + 1e-5 * timestep.float()[..., None, None, None]


def _window(seed=4, frames=41):
    g = torch.Generator().manual_seed(seed)
    first = torch.randn(1, 1, 2, 24, 42, generator=g)
    noise = torch.randn(1, frames - 1, 2, 24, 42, generator=g)
    conditions = {
        "player_alive": torch.ones(1, frames, 3),
        "player_visible": torch.ones(1, frames, 3),
    }
    return first, noise, conditions


def _sampler(generator, rng=None) -> Sampler:
    """The evaluation's sampler: the context is written clean."""
    cache = KVCache.allocate(
        num_dit_blocks=1,
        num_heads=1,
        head_dim=1,
        latent_frames=41,
        frame_tokens=252,
        dtype=torch.float32,
    )
    return Sampler.create(generator, cache, context_noise=0, rng=rng)


def test_bidirectional_pins_the_first_frame():
    first, noise, conditions = _window()
    timesteps = []

    def flow_fn(x, t, c):
        timesteps.append(t.clone())
        return 0.5 * x

    out = sample_bidirectional(flow_fn, noise, first, conditions, steps=20)
    assert out.shape == (1, 41, 2, 24, 42) and torch.equal(out[:, :1], first)
    assert len(timesteps) == 20 and all(t.shape == (1, 41) and t[0, 0] == 0 for t in timesteps)
    assert timesteps[0][0, 1] == 999 and timesteps[-1][0, 1] == 208


def test_block_causal_writes_the_first_frame_then_each_block_clean():
    first, noise, conditions = _window()
    fake = FakeCausal()
    out = sample_block_causal(_sampler(fake), noise, first, conditions, steps=20)
    assert torch.equal(out[:, :1], first) and out.dtype == noise.dtype
    frames, timestep, position = fake.calls[0]
    assert (frames, position, timestep.tolist()) == (1, 0, [[0.0]])
    blocks = fake.calls[1:]
    assert len(blocks) == 10 * 21
    for b in range(10):
        calls = blocks[21 * b : 21 * (b + 1)]
        assert {c[2] for c in calls} == {1 + 4 * b} and all(c[0] == 4 for c in calls)
        assert calls[0][1][0, 0] == 999 and bool((calls[-1][1] == 0).all())


def test_four_step_renoises_from_the_window_generator():
    first, noise, conditions = _window()

    def run(seed):
        fake = FakeCausal()
        rng = torch.Generator().manual_seed(seed)
        out = sample_four_step(_sampler(fake, rng), noise, first, conditions)
        return out, fake

    torch.manual_seed(1234)
    out, fake = run(7)
    blocks = fake.calls[1:]
    assert len(blocks) == 10 * 5
    levels = [float(c[1][0, 0]) for c in blocks[:5]]
    assert levels[:4] == pytest.approx([1000.0, 937.5, 833.3333, 625.0]) and levels[4] == 0
    assert all(c[1].dtype == torch.float32 for c in blocks)
    torch.manual_seed(99)  # the global RNG plays no part
    again, _ = run(7)
    other, _ = run(8)
    assert torch.equal(out, again) and not torch.equal(out, other)
