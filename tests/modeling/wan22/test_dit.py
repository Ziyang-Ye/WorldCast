"""The causal DiT block and what it is built from: RoPE, the KV cache, the DiT block's parts."""

import functools

import pytest
import torch

from tests.modeling.support import randomize_
from worldcast.modeling.wan22.attention import sdpa_attention
from worldcast.modeling.wan22.dit import (
    DIT_BLOCK_PARTS,
    CausalDiTBlock,
    CausalHead,
    KVCache,
    adaln_adapter,
    apply_rope,
    causal_self_attention,
    rope_frequencies,
    rope_params,
    rope_table,
    sinusoidal_embedding_1d,
)
from worldcast.utils.precision import cast_floating_tensors

EPS = 1e-6


def _cache(latent_frames: int = 3, frame_tokens: int = 2, heads: int = 1) -> KVCache:
    return KVCache.allocate(
        num_dit_blocks=2,
        num_heads=heads,
        head_dim=4,
        latent_frames=latent_frames,
        frame_tokens=frame_tokens,
        dtype=torch.float32,
    )


def test_kv_cache_advances_and_rewrites_in_place():
    cache = _cache()
    assert (cache.capacity, cache.end) == (6, 0) and len(cache.keys) == len(cache.values) == 2
    assert cache.span(0, 2) == 2
    key, value = torch.ones(1, 2, 1, 4), torch.full((1, 2, 1, 4), 2.0)
    keys, values = cache.update(0, key, value, 0, 2)
    assert torch.equal(keys, key) and torch.equal(values, value)
    cache.end = 2
    with pytest.raises(ValueError, match="must start there"):
        cache.span(4, 2)  # a gap after the end
    assert cache.span(2, 4) == 6  # a write at the end advances
    cache.end = 6
    assert cache.span(2, 4) == 6  # a write inside rewrites in place
    keys, _ = cache.update(0, 3 * torch.ones(1, 4, 1, 4), torch.zeros(1, 4, 1, 4), 2, 6)
    assert keys[0, :, 0, 0].tolist() == [1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
    with pytest.raises(ValueError, match="holds 6 tokens"):
        cache.span(6, 2)
    with pytest.raises(ValueError, match="starts at a token of the cache, not at -2"):
        cache.span(-2, 2)
    for shape in ((2, 2, 1, 4), (1, 1, 1, 4), (1, 2, 1, 2)):  # rollouts, tokens, head width
        with pytest.raises(
            ValueError, match=r"takes keys and values \[1, 2, 1, 4\] \(1 rollouts\)"
        ):
            cache.update(0, torch.ones(shape), torch.ones(shape), 0, 2)
    with pytest.raises(ValueError, match=r"got \(1, 2, 1, 4\) and \(1, 2, 1, 1\)"):
        cache.update(0, torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 1), 0, 2)
    cache.reset()
    assert cache.end == 0 and cache.keys[0][0, 0, 0, 0] == 1.0  # the buffers are kept


def test_kv_cache_passes_through_the_input_casts():
    """The cache is no container: a cast of the inputs hands on the caller's own cache, buffers
    uncast."""
    cache = _cache()
    cast = cast_floating_tensors({"kv_cache": cache, "x": torch.zeros(1)}, torch.bfloat16)
    assert cast["kv_cache"] is cache and cache.keys[0].dtype == torch.float32
    assert cast["x"].dtype == torch.bfloat16


def test_sinusoidal_embedding_is_cos_then_sin():
    embedding = sinusoidal_embedding_1d(8, torch.tensor([0.0, 1000.0]))
    assert embedding.shape == (2, 8) and embedding.dtype == torch.float64
    assert embedding[0].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert embedding[1, 0].item() == pytest.approx(torch.cos(torch.tensor(1000.0)).item())
    assert embedding[1, 4].item() == pytest.approx(torch.sin(torch.tensor(1000.0)).item())


def test_rope_rotates_by_the_window_position():
    """A token's factors are those of its window frame, row and column; frame 0, row 0, column 0
    is the identity, and a rotation keeps every pair's norm."""
    freqs = rope_frequencies(8)  # head_dim 8: two factors for the frame, one each for row, column
    assert freqs.shape == (1024, 4)
    assert torch.equal(freqs[:, :2], rope_params(1024, 4))
    assert torch.equal(freqs[:, 2:3], freqs[:, 3:]) and torch.equal(
        freqs[:, 3:], rope_params(1024, 2)
    )
    table = rope_table(freqs, (3, 2, 2), frame_offset=5)
    assert table.shape == (12, 1, 4) and table.dtype == torch.complex128
    for frame in range(3):  # a call at an offset reads the same factors as its frames alone
        alone = rope_table(freqs, (1, 2, 2), frame_offset=5 + frame)
        assert torch.equal(table[4 * frame : 4 * frame + 4], alone)
    x = torch.randn(1, 12, 2, 8, generator=torch.Generator().manual_seed(0))
    rotated = apply_rope(x, table)
    assert rotated.dtype == torch.float32 and rotated.shape == x.shape
    pairs = lambda t: t.reshape(1, 12, 2, 4, 2).norm(dim=-1)  # noqa: E731
    torch.testing.assert_close(pairs(rotated), pairs(x), rtol=1e-6, atol=1e-6)
    origin = rope_table(freqs, (1, 2, 2), frame_offset=0)
    assert torch.equal(apply_rope(x[:, :4], origin)[:, 0], x[:, 0])


def test_causal_self_attention_attends_to_what_the_cache_holds():
    """A call's keys and values join the cache and its queries attend to the whole cache: the
    ranges written before it and itself."""
    freqs = rope_params(64, 4)  # head_dim 4: two factors, both temporal
    g = torch.Generator().manual_seed(1)
    cache = _cache(latent_frames=2, frame_tokens=4, heads=2)
    first, second = ([torch.randn(1, 4, 2, 4, generator=g) for _ in range(3)] for _ in range(2))
    ropes = [rope_table(freqs, (1, 2, 2), frame_offset=frame) for frame in range(2)]
    attend = functools.partial(
        causal_self_attention, cache=cache, dit_block=1, attention=sdpa_attention
    )
    out_first = attend(*first, rope=ropes[0], start=0, end=cache.span(0, 4))
    cache.end = 4
    out_second = attend(*second, rope=ropes[1], start=4, end=cache.span(4, 4))
    rotated = [
        (apply_rope(q, r), apply_rope(k, r), v) for (q, k, v), r in zip((first, second), ropes)
    ]
    keys = torch.cat([rotated[0][1], rotated[1][1]], dim=1)
    values = torch.cat([first[2], second[2]], dim=1)
    assert torch.equal(out_first, sdpa_attention(*rotated[0]))
    assert torch.equal(out_second, sdpa_attention(rotated[1][0], keys, values))
    assert torch.equal(cache.keys[1][:, :8], keys) and bool((cache.keys[0] == 0).all())


def test_adaln_adapters_start_at_zero():
    """The low-rank adapter of a DiT block (six AdaLN vectors) and of the head (two) add nothing
    until trained."""
    torch.manual_seed(0)
    for multiplier in (6, 2):
        adapter = adaln_adapter(16, multiplier, 4)
        out = adapter(torch.randn(2, 3, 16))
        assert out.shape == (2, 3, multiplier * 16) and not bool(out.any())
        assert adapter[1].bias is None and bool(adapter[1].weight.any())


def _block(dim: int = 32) -> CausalDiTBlock:
    block = CausalDiTBlock(dim, 64, 2, EPS)
    block.attach_control_adaln(8)
    return randomize_(block, 2).eval()


def test_the_control_adapters_are_attached_after_construction():
    """The DiT block and the head are built without their control adapter, which the generator
    attaches after the backbone's initialisation (the draw order of the paper's runs)."""
    block, head = CausalDiTBlock(32, 64, 2, EPS), CausalHead(32, 8, (1, 2, 2), EPS)
    assert not hasattr(block, "control_adaln") and not hasattr(head, "control_adaln")
    block.attach_control_adaln(8)
    head.attach_control_adaln(8)
    assert block.control_adaln[-1].out_features == 6 * 32
    assert head.control_adaln[-1].out_features == 2 * 32


def test_block_is_its_three_parts_around_the_two_attentions():
    block = _block()
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 3 * 4, 32, generator=g)  # 3 frames of 4 tokens
    modulation_in = torch.randn(2, 3, 6, 32, generator=g)
    control_embedding = torch.randn(2, 3, 32, generator=g)
    text = torch.randn(2, 7, 32, generator=g)
    kwargs = dict(
        timestep_modulation=modulation_in,
        control_embedding=control_embedding,
        self_attention=sdpa_attention,
        attention=sdpa_attention,
    )
    with torch.no_grad():
        out = block(x, text=text, **kwargs)
        pre_attention, post_self_attention, post_cross_attention = DIT_BLOCK_PARTS
        qkv, modulation = pre_attention(block, x, modulation_in, control_embedding)
        assert len(modulation) == 6 and modulation[0].shape == (2, 3, 1, 32)
        hidden, query = post_self_attention(block, x, sdpa_attention(*qkv), modulation)
        key_value = block.cross_attn.key_value(text)
        attended = sdpa_attention(query, *key_value)
        assert torch.equal(out, post_cross_attention(block, hidden, attended, modulation))
        # the text's keys and values may be handed over precomputed, and the parts replaced
        assert torch.equal(out, block(x, text=key_value, **kwargs))
        calls = []

        def counted(part):
            def run(*args):
                calls.append(part.__name__)
                return part(*args)

            return run

        parts = tuple(counted(part) for part in DIT_BLOCK_PARTS)
        assert torch.equal(out, block(x, text=text, parts=parts, **kwargs))
    assert calls == ["pre_attention", "post_self_attention", "post_cross_attention"]
    assert out.shape == x.shape and out.dtype == torch.float32


def test_block_modulates_each_frame_with_its_own_timestep_and_controls():
    """AdaLN is per frame: changing one frame's modulation or control embedding leaves the
    pre-attention queries of the other frames' tokens unchanged."""
    block = _block()
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 2 * 4, 32, generator=g)
    modulation = torch.randn(1, 2, 6, 32, generator=g)
    control_embedding = torch.randn(1, 2, 32, generator=g)
    with torch.no_grad():
        (query, _, _), _ = block.pre_attention(x, modulation, control_embedding)
        modulation2, control_embedding2 = modulation.clone(), control_embedding.clone()
        modulation2[:, 1] += 1.0
        control_embedding2[:, 1] += 1.0
        (query2, _, _), _ = block.pre_attention(x, modulation2, control_embedding2)
    assert torch.equal(query[:, :4], query2[:, :4]) and not torch.equal(query[:, 4:], query2[:, 4:])
