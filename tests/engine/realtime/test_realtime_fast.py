"""The fast generator call equals the release generator call bit for bit (CPU, float32, tiny random
weights).

Both paths run through the release :class:`worldcast.sampling.rollouts.Sampler` on the same window:
the plain-prefix schedule (absolute positions, growing cache) and two reconstituted blocks (21- and
17-latent windows, context re-noised range by range, the target overwritten in place on every rung).
"""

import pytest
import torch

from worldcast.engine.realtime.fast import FastGenerator, attention_backend, patchify_linear
from worldcast.modeling.action import ActionConfig
from worldcast.modeling.wan22.attention import sdpa_attention
from worldcast.modeling.wan22.model import (
    CausalGeneratorAdapter,
    GeneratorConfig,
    KVCache,
    WorldCastGenerator,
)
from worldcast.sampling.rollouts import Sampler
from worldcast.sampling.schedulers import FlowMatchScheduler, warped_ladder

TINY = dict(
    in_dim=8,
    out_dim=8,
    dim=32,
    ffn_dim=64,
    freq_dim=16,
    text_dim=24,
    text_len=7,
    num_heads=2,
    num_layers=4,
)
LAT_H, LAT_W = 8, 12  # 4 x 6 = 24 tokens per latent
FSL = (LAT_H // 2) * (LAT_W // 2)
WINDOW = 21


def tiny_generator(seed: int = 3) -> WorldCastGenerator:
    torch.manual_seed(seed)
    config = GeneratorConfig(
        **TINY,
        action=ActionConfig(hidden_dim=64, adaln_rank=8),
        obs_signal_hidden=16,
    )
    model = WorldCastGenerator(config, attention=sdpa_attention)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return model.eval()


def field_builder(c, weapon_weight, frame_offset, num_frames):
    """A deterministic stand-in for ``build_field`` that reads the peer table and the weapon
    embedding."""
    rows = c["peer_states"][:, frame_offset : frame_offset + num_frames, :, :1]  # [B, n, P, 1]
    base = torch.tanh(rows.mean(dim=2, keepdim=True) / 100.0)[..., None]  # [B, n, 1, 1, 1]
    grid = torch.arange(23 * (LAT_H // 2) * (LAT_W // 2), dtype=torch.float32).view(
        1, 1, 23, LAT_H // 2, LAT_W // 2
    )
    return torch.sin(grid / 7.0 + base) + weapon_weight.sum() * 0.01


def random_c2w(frames: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(frames, 3, 3, generator=g))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[..., None, :]
    q = q * torch.det(q)[..., None, None]
    c2w = torch.eye(4).repeat(frames, 1, 1)
    c2w[:, :3, :3] = q
    c2w[:, :3, 3] = torch.randn(frames, 3, generator=g) * 300
    return c2w[None]


def window_conditions(frames: int, *, slot: bool, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    rows = 1 + 4 * (frames - 1)
    c2w = random_c2w(frames, seed + 1)
    target = frames - 4
    memory = [1, 2, 3, 4] if slot else []
    return dict(
        prompt_embeds=torch.randn(1, 5, TINY["text_dim"], generator=g),
        button_condition=torch.randint(0, 2, (1, rows, 11), generator=g).float(),
        camera_condition=torch.randn(1, rows, 2, generator=g),
        weapon_condition=torch.randint(0, 52, (1, rows), generator=g),
        obs_flash_flag=torch.randint(0, 2, (1, frames), generator=g),
        obs_flash_valid=torch.randint(0, 2, (1, frames), generator=g),
        obs_scope_on=torch.randint(0, 2, (1, frames), generator=g),
        obs_scope_level=torch.randint(0, 3, (1, frames), generator=g),
        obs_scope_valid=torch.randint(0, 2, (1, frames), generator=g),
        state_wp_frame_c2w=c2w,
        state_wp_frame_tans=torch.rand(1, frames, 2, generator=g) + 0.5,
        state_wp_anchor_c2w=c2w[:, target],
        state_wp_memory_c2w=c2w[:, memory],
        state_wp_memory_frames=torch.tensor([memory], dtype=torch.long),
        peer_states=torch.randn(1, frames, 4, 6, generator=g) * 100,
        peer_actions=torch.rand(1, frames, 4, 16, 14, generator=g),
        peer_observer_slot=torch.zeros(1, dtype=torch.long),
        peer_team_ids=torch.tensor([[0, 0, 1, 1]]),
        peer_alive=torch.ones(1, frames, 4),
        peer_visible=torch.randint(0, 2, (1, frames, 4), generator=g).float(),
        peer_weapons=torch.randint(0, 52, (1, frames, 4), generator=g),
    )


def sampler_for(generator) -> Sampler:
    cfg = generator.generator.config
    cache = KVCache.allocate(
        num_blocks=cfg.num_layers,
        num_heads=cfg.num_heads,
        head_dim=cfg.dim // cfg.num_heads,
        capacity_latents=41,
        frame_seq_length=FSL,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = FlowMatchScheduler()
    return Sampler(
        generator=generator,
        scheduler=scheduler,
        ladder=warped_ladder((1000, 750, 500, 250), scheduler),
        cache=cache,
        frame_seq_length=FSL,
        context_noise=16,
        rng=torch.Generator().manual_seed(11),
    )


def pair():
    model = tiny_generator()
    reference = sampler_for(CausalGeneratorAdapter(model, field_builder, input_dtype=None))
    fast = sampler_for(FastGenerator(model, field_builder, input_dtype=None, patch_linear=False))
    return reference, fast


def assert_caches_equal(a: KVCache, b: KVCache) -> None:
    assert a.end == b.end
    for ka, kb, va, vb in zip(a.keys, b.keys, a.values, b.values):
        assert torch.equal(ka[:, : a.end], kb[:, : a.end])
        assert torch.equal(va[:, : a.end], vb[:, : a.end])


def test_prefix_schedule_is_bit_identical():
    reference, fast = pair()
    cond = window_conditions(13, slot=False, seed=5)
    noise = torch.randn(
        1, 12, TINY["in_dim"], LAT_H, LAT_W, generator=torch.Generator().manual_seed(7)
    )
    sink = torch.randn(
        1, 1, TINY["in_dim"], LAT_H, LAT_W, generator=torch.Generator().manual_seed(8)
    )
    a = reference.rollout_prefix(noise, sink, cond)
    b = fast.rollout_prefix(noise, sink, cond)
    assert torch.equal(a, b)
    assert_caches_equal(reference.cache, fast.cache)


@pytest.mark.parametrize("slot", [True, False])
def test_reconstituted_blocks_are_bit_identical(slot):
    reference, fast = pair()
    frames = WINDOW if slot else WINDOW - 4
    for block in range(2):  # the cache is reset and refilled every block
        cond = window_conditions(frames, slot=slot, seed=20 + block)
        g = torch.Generator().manual_seed(30 + block)
        context = torch.randn(1, frames - 4, TINY["in_dim"], LAT_H, LAT_W, generator=g)
        noisy = torch.randn(1, 4, TINY["in_dim"], LAT_H, LAT_W, generator=g)
        a = reference.generate_block(context, noisy, cond)
        b = fast.generate_block(context, noisy, cond)
        assert torch.equal(a, b), f"block {block}: max |diff| {(a - b).abs().max().item()}"
        assert_caches_equal(reference.cache, fast.cache)


def test_fixed_prompt_work_is_done_once():
    _, fast = pair()
    cond = window_conditions(WINDOW, slot=True, seed=1)
    context = torch.randn(1, WINDOW - 4, TINY["in_dim"], LAT_H, LAT_W)
    fast.generate_block(context, torch.randn(1, 4, TINY["in_dim"], LAT_H, LAT_W), cond)
    key = fast.generator._prompt_key
    fast.generate_block(context, torch.randn(1, 4, TINY["in_dim"], LAT_H, LAT_W), cond)
    assert fast.generator._prompt_key == key and fast.generator.stats["calls"] == 2 * 9
    assert fast.generator.stats["prologue_reused"] == 2 * 3  # rungs 2-4 reuse rung 1's prologue


def test_patchify_linear_matches_conv3d():
    conv = torch.nn.Conv3d(8, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    x = torch.randn(1, 8, 4, 8, 12, dtype=torch.float64)
    conv = conv.double()
    torch.testing.assert_close(patchify_linear(conv, x), conv(x), rtol=0, atol=1e-12)


def test_attention_backends_build_on_any_device():
    """The kernels are chosen on the host: picking one needs no GPU."""
    for name in ("flash", "cudnn"):
        assert callable(attention_backend(name))
    with pytest.raises(ValueError):
        attention_backend("nope")
