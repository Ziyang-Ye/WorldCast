"""The fast generator call equals the generator's own call bit for bit (CPU, float32, tiny random
weights).

Both run under the same :class:`~worldcast.sampling.sampler.Sampler` as a client drives it: the
first blocks at their positions in the round on a growing cache, then blocks generated from their
windows (21 and 17 latent frames: the context written range by range, the target rewritten in
place on every denoising step).
"""

import pytest
import torch

from tests.modeling.support import (
    FRAME_TOKENS,
    LATENT_H,
    LATENT_W,
    TINY,
    field_builder,
    tiny_generator,
    window_conditions,
)
from worldcast.engine.inference.fast import FastGenerator, patchify_linear
from worldcast.modeling.wan22.attention import causal_blocks
from worldcast.modeling.wan22.dit import KVCache
from worldcast.modeling.wan22.model import CausalGeneratorAdapter
from worldcast.sampling.sampler import Sampler


def _sampler(generator) -> Sampler:
    cache = generator.generator.allocate_kv_cache(25, frame_tokens=FRAME_TOKENS)
    return Sampler.create(generator, cache, rng=torch.Generator().manual_seed(11))


def _pair(**overrides) -> tuple[Sampler, Sampler]:
    """The adapter and the fast path on one generator, each with its own cache and stream."""
    model = tiny_generator(**overrides)
    reference = CausalGeneratorAdapter(model, field_builder, input_dtype=None)
    fast = FastGenerator(model, field_builder, input_dtype=None)
    return _sampler(reference), _sampler(fast)


def _latents(frames: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, frames, TINY["in_dim"], LATENT_H, LATENT_W, generator=g)


def _conditions(frames: int, *, seed: int, **optional: bool) -> dict:
    """A window's conditions; ``ray_=False`` or ``obs_=False`` leaves those entries out."""
    conditions = window_conditions(frames, anchor=frames - 4, seed=seed)
    dropped = tuple(prefix for prefix, kept in optional.items() if not kept)
    return {k: v for k, v in conditions.items() if not k.startswith(dropped)}


def _first_blocks(sampler: Sampler, noise: torch.Tensor, first: torch.Tensor, conditions: dict):
    sampler.cache.reset()
    sampler.write_context(first, conditions, frame_offset=0)
    blocks = []
    for start in range(1, noise.shape[1] + 1, 4):
        noisy = noise[:, start - 1 : start + 3]
        blocks.append(sampler.denoise(noisy, conditions, frame_offset=start))
        sampler.write_context(blocks[-1], conditions, frame_offset=start)
    return torch.cat(blocks, dim=1)


def _window_block(sampler: Sampler, context: torch.Tensor, noisy: torch.Tensor, conditions: dict):
    sampler.cache.reset()
    sampler.prefill_context(context, conditions, ranges=causal_blocks(context.shape[1]))
    return sampler.denoise(noisy, conditions, frame_offset=context.shape[1])


def _assert_caches_equal(a: KVCache, b: KVCache) -> None:
    assert a.end == b.end
    for ka, kb, va, vb in zip(a.keys, b.keys, a.values, b.values):
        assert torch.equal(ka[:, : a.end], kb[:, : a.end])
        assert torch.equal(va[:, : a.end], vb[:, : a.end])


def test_the_first_blocks_are_bit_identical():
    reference, fast = _pair()
    conditions = _conditions(13, seed=5)
    noise, first = _latents(12, seed=7), _latents(1, seed=8)
    a = _first_blocks(reference, noise, first, conditions)
    b = _first_blocks(fast, noise, first, conditions)
    assert a.shape == noise.shape and torch.equal(a, b)
    _assert_caches_equal(reference.cache, fast.cache)


@pytest.mark.parametrize("with_memory", [True, False])
def test_blocks_generated_from_their_windows_are_bit_identical(with_memory):
    reference, fast = _pair()
    frames = 21 if with_memory else 17
    for block in range(2):  # the cache is emptied and refilled every block
        conditions = _conditions(frames, seed=20 + block)
        context, noisy = _latents(frames - 4, seed=30 + block), _latents(4, seed=40 + block)
        a = _window_block(reference, context, noisy, conditions)
        b = _window_block(fast, context, noisy, conditions)
        assert torch.equal(a, b), f"block {block}: max |diff| {(a - b).abs().max().item()}"
        _assert_caches_equal(reference.cache, fast.cache)


def test_a_generator_without_the_optional_modules_is_bit_identical():
    """Stage 1's generator: no state injector, observer-signal embedding or ray embedding."""
    reference, fast = _pair(state_injector=None, observer_signals=None, ray_embedding=False)
    conditions = _conditions(17, seed=3, ray_=False, obs_=False)
    context, noisy = _latents(13, seed=1), _latents(4, seed=2)
    a = _window_block(reference, context, noisy, conditions)
    assert torch.equal(a, _window_block(fast, context, noisy, conditions))


def _count_calls(monkeypatch, generator, name: str) -> list:
    """Record the calls of ``generator.<name>``."""
    calls: list = []
    method = getattr(generator, name)

    def counted(*args, **kwargs):
        calls.append(args)
        return method(*args, **kwargs)

    monkeypatch.setattr(generator, name, counted)
    return calls


def test_the_work_the_denoising_steps_share_is_done_once(monkeypatch):
    """The prologue is computed on the first denoising step and reused by the other three; the
    prompt's text embedding (and with it every block's cross-attention keys and values) is computed
    once per prompt."""
    _, fast = _pair()
    model = fast.generator.generator
    prologues = _count_calls(monkeypatch, model, "prologue")
    text_embeddings = _count_calls(monkeypatch, model, "embed_text")
    conditions = _conditions(21, seed=1)
    context = _latents(17, seed=2)
    for seed in (3, 4):
        _window_block(fast, context, _latents(4, seed=seed), conditions)
    # per block: five context writes and four denoising steps that share one prologue
    assert len(prologues) == 2 * (5 + 1) and len(text_embeddings) == 1
    other_prompt = {**conditions, "prompt_embeds": conditions["prompt_embeds"] + 1.0}
    _window_block(fast, context, _latents(4, seed=4), other_prompt)
    assert len(text_embeddings) == 2


@pytest.mark.parametrize("input_dtype", [None, torch.float32], ids=["uncast", "cast"])
def test_another_prompt_gets_its_own_cross_attention(input_dtype):
    """The prompt is identified before the input cast: a float64 prompt is cast anew on every call,
    and the cast of another prompt can land on the storage the first one's cast has left."""
    model = tiny_generator()
    reference = _sampler(CausalGeneratorAdapter(model, field_builder, input_dtype=input_dtype))
    fast = _sampler(FastGenerator(model, field_builder, input_dtype=input_dtype))
    conditions = _conditions(17, seed=1)
    prompt = conditions["prompt_embeds"]
    if input_dtype is not None:
        prompt = prompt.double()
    context = _latents(13, seed=2)
    for trial in range(8):
        conditions["prompt_embeds"] = prompt + float(trial)
        noisy = _latents(4, seed=10 + trial)
        expected = _window_block(reference, context, noisy, conditions)
        assert torch.equal(_window_block(fast, context, noisy, conditions), expected), trial


def test_the_fast_path_serves_one_client():
    _, fast = _pair()
    conditions = window_conditions(1, batch=2)
    with pytest.raises(ValueError, match="batch 1"):
        fast.generator(
            _latents(1, seed=0).expand(2, -1, -1, -1, -1),
            torch.full((2, 1), 16.0),
            conditions,
            kv_cache=fast.cache,
            frame_offset=0,
        )
    with pytest.raises(ValueError, match="needs a CUDA generator"):
        FastGenerator(tiny_generator(), field_builder, cuda_graphs=True)


def test_patchify_linear_matches_conv3d():
    conv = torch.nn.Conv3d(8, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2)).double()
    x = torch.randn(2, 8, 4, 8, 12, dtype=torch.float64)
    torch.testing.assert_close(patchify_linear(conv, x), conv(x), rtol=0, atol=1e-12)
