"""A tiny generator of the paper's architecture, and random inputs for it."""

import torch

from worldcast.modeling.action import ActionConfig
from worldcast.modeling.wan22.model import GeneratorConfig, KVCache

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
TINY_ACTION = dict(hidden_dim=64, adaln_rank=8)
TINY_OBS_HIDDEN = 16


def randomize_(module: torch.nn.Module, seed: int, scale: float = 0.2) -> torch.nn.Module:
    """Overwrite every parameter with seeded noise (zero-initialised outputs would hide the paths
    under test)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * scale)
    return module


def tiny_config() -> GeneratorConfig:
    return GeneratorConfig(
        **TINY,
        action=ActionConfig(**TINY_ACTION),
        obs_signal_hidden=TINY_OBS_HIDDEN,
    )


def random_c2w(batch: int, frames: int, *, seed: int, spread: float = 300.0) -> torch.Tensor:
    """``[B, F, 4, 4]`` random rigid camera-to-world matrices (Source units)."""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(batch, frames, 3, 3, generator=g))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[..., None, :]
    q = q * torch.det(q)[..., None, None]  # det +1
    c2w = torch.eye(4).repeat(batch, frames, 1, 1)
    c2w[..., :3, :3] = q
    c2w[..., :3, 3] = torch.randn(batch, frames, 3, generator=g) * spread
    return c2w


def new_kv_cache(model, batch: int, capacity_frames: int, frame_seq_length: int):
    """A float32 :class:`KVCache` of ``capacity_frames`` latent frames for ``model``."""
    return KVCache.allocate(
        num_blocks=len(model.blocks),
        num_heads=model.num_heads,
        head_dim=model.dim // model.num_heads,
        capacity_latents=capacity_frames,
        frame_seq_length=frame_seq_length,
        batch_size=batch,
        dtype=torch.float32,
    )
