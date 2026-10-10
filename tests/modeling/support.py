"""A tiny generator of the paper's architecture, random conditions of a window for it, a tiny VAE
of the Wan2.2 VAE's layout and a random latent scale for it."""

from collections.abc import Mapping
from typing import Any

import torch

from worldcast.modeling.controls import ControlConfig
from worldcast.modeling.observer_signals import ObserverSignalConfig
from worldcast.modeling.state_injector import FIELD_CHANNELS
from worldcast.modeling.wan22.attention import sdpa_attention
from worldcast.modeling.wan22.model import GeneratorConfig, WorldCastGenerator
from worldcast.modeling.wan22.vae import VAEConfig, Wan22VAE

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
TINY_CONTROLS = dict(hidden=64, adaln_rank=8)
TINY_OBSERVER_SIGNALS_HIDDEN = 16
#: Latent grid of the tiny windows; 4 x 6 = 24 tokens per latent frame.
LATENT_H, LATENT_W = 8, 12
GRID_H, GRID_W = LATENT_H // 2, LATENT_W // 2
FRAME_TOKENS = GRID_H * GRID_W
#: Players of a tiny window's player state table.
PLAYERS = 4


def tiny_config(**overrides) -> GeneratorConfig:
    """The paper's generator at tiny dimensions; ``overrides`` replace fields."""
    fields = dict(
        **TINY,
        controls=ControlConfig(**TINY_CONTROLS),
        observer_signals=ObserverSignalConfig(hidden=TINY_OBSERVER_SIGNALS_HIDDEN),
    )
    return GeneratorConfig(**{**fields, **overrides})


def randomize_(module: torch.nn.Module, seed: int, scale: float = 0.2) -> torch.nn.Module:
    """Overwrite every parameter with seeded noise (zero-initialised outputs would hide the paths
    under test)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * scale)
    return module


def tiny_generator(seed: int = 3, **overrides) -> WorldCastGenerator:
    """A tiny generator with every parameter random, in eval mode, on the CPU's attention."""
    torch.manual_seed(seed)
    model = WorldCastGenerator(tiny_config(**overrides), attention=sdpa_attention)
    return randomize_(model, seed + 1).eval()


def make_tiny_vae() -> Wan22VAE:
    """A small random ``Wan22VAE`` with the Wan2.2 VAE's layout (48 latent channels, four frames
    per latent frame)."""
    torch.manual_seed(37)
    config = VAEConfig(dim=8, dec_dim=8, dim_mult=(1, 1, 2, 2), num_res_blocks=1)
    return Wan22VAE(config).eval()


def random_latent_scale(z_dim: int) -> list[torch.Tensor]:
    """A random ``[mean, 1 / std]`` of ``z_dim`` latent channels, the VAE's ``scale`` argument."""
    g = torch.Generator().manual_seed(33)
    mean = torch.randn(z_dim, generator=g)
    inv_std = 1.0 / (torch.rand(z_dim, generator=g) + 0.5)
    return [mean, inv_std]


def random_c2w(batch: int, frames: int, *, seed: int) -> torch.Tensor:
    """``[B, F, 4, 4]`` random rigid camera-to-world matrices, a few hundred u apart."""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(batch, frames, 3, 3, generator=g))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[..., None, :]
    q = q * torch.det(q)[..., None, None]  # det +1
    c2w = torch.eye(4).repeat(batch, frames, 1, 1)
    c2w[..., :3, :3] = q
    c2w[..., :3, 3] = torch.randn(batch, frames, 3, generator=g) * 300.0
    return c2w


def window_conditions(
    frames: int, *, batch: int = 1, anchor: int = 0, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Random generator conditions of a window of ``frames`` latent frames.

    The prompt, the controls over the ``1 + 4 (frames - 1)`` video frames, the observer signals,
    the cameras (the anchor at window position ``anchor``) and the table of :data:`PLAYERS`
    player states that :data:`field_builder` reads: the client (slot 0) at the origin looking
    along +x, the others in front of it, all alive.
    """
    players = PLAYERS
    g = torch.Generator().manual_seed(seed)
    video_frames = 1 + 4 * (frames - 1)
    table = torch.zeros(batch, frames, players, 6)
    table[..., 0] = torch.rand(batch, 1, players, generator=g) * 400 + 150
    table[..., 1] = (torch.rand(batch, 1, players, generator=g) - 0.5) * 300
    table[..., 0] += torch.arange(frames)[None, :, None] * 5.0
    table[:, :, 0, :3] = 0.0
    table[..., 3] = (torch.rand(batch, frames, players, generator=g) - 0.5) * 360
    table[..., 5] = 1.0
    controls = torch.zeros(batch, frames, players, 16, 14)
    controls[..., :11] = torch.randint(0, 2, (batch, frames, players, 16, 11), generator=g).float()
    controls[..., -3:-1] = (torch.rand(batch, frames, players, 16, 2, generator=g) - 0.5) * 0.2
    controls[..., -1] = 1.0
    visible = torch.randint(0, 2, (batch, frames, players), generator=g).float()
    visible[:, :, 1] = 1.0  # at least one player always written
    c2w = random_c2w(batch, frames, seed=seed + 1)
    return dict(
        prompt_embeds=torch.randn(batch, 5, TINY["text_dim"], generator=g),
        buttons=torch.randint(0, 2, (batch, video_frames, 11), generator=g).float(),
        view_deltas=torch.randn(batch, video_frames, 2, generator=g),
        weapon=torch.randint(0, 52, (batch, video_frames), generator=g),
        obs_flash_flag=torch.randint(0, 2, (batch, frames), generator=g),
        obs_flash_valid=torch.randint(0, 2, (batch, frames), generator=g),
        obs_scope_on=torch.randint(0, 2, (batch, frames), generator=g),
        obs_scope_level=torch.randint(0, 3, (batch, frames), generator=g),
        obs_scope_valid=torch.randint(0, 2, (batch, frames), generator=g),
        ray_frame_c2w=c2w,
        ray_frame_tans=torch.rand(batch, frames, 2, generator=g) + 0.5,
        ray_anchor_c2w=c2w[:, anchor],
        player_state_table=table,
        player_controls=controls,
        client_slot=torch.zeros(batch, dtype=torch.long),
        player_team_ids=torch.tensor([[i * 2 // players for i in range(players)]] * batch),
        player_alive=torch.ones(batch, frames, players),
        player_visible=visible,
        player_weapons=torch.randint(0, 52, (batch, frames, players), generator=g),
    )


class _StandInFieldBuilder:
    """A deterministic stand-in for the player state field ``[B, n, 23, 4, 6]`` that reads the
    player state table's frames of the call and the weapon embedding."""

    condition_keys = (
        "player_state_table",
        "player_controls",
        "client_slot",
        "player_team_ids",
        "player_alive",
        "player_visible",
        "player_weapons",
    )

    def __call__(
        self,
        conditions: Mapping[str, Any],
        weapon_embedding: torch.Tensor,
        frame_offset: int,
        num_frames: int,
    ) -> torch.Tensor:
        table = conditions["player_state_table"]
        frames = table[:, frame_offset : frame_offset + num_frames, :, :1]
        base = torch.tanh(frames.mean(dim=2, keepdim=True) / 100.0)[..., None]  # [B, n, 1, 1, 1]
        size = FIELD_CHANNELS * FRAME_TOKENS
        grid = torch.arange(size, dtype=torch.float32).view(1, 1, FIELD_CHANNELS, GRID_H, GRID_W)
        return torch.sin(grid / 7.0 + base) + weapon_embedding.sum() * 0.01


field_builder = _StandInFieldBuilder()
