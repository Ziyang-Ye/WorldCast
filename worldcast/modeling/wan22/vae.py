"""The Wan2.2-TI2V-5B video VAE and its streaming decoder.

Latents ``[B, 48, T, h, w]`` decode to pixels ``[B, 3, 1 + 4 (T-1), 16 h, 16 w]`` in ``[-1, 1]``
(384 x 672 for ``h, w = 24, 42``). The decoder is causal in time and keeps, per causal conv, the
last input frames it saw (:class:`CausalConvCache`); :class:`Wan22StreamingDecoder` keeps that
cache across chunks, so consecutive chunks concatenate to one :meth:`Wan22VAE.decode`. The
state-dict keys are those of ``Wan2.2_VAE.pth``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CACHE_T",
    "LATENT_CHANNELS",
    "LATENT_MEAN",
    "LATENT_STD",
    "VAE_CHECKPOINT_NAME",
    "VAEConfig",
    "WAN22_VAE",
    "CausalConvCache",
    "Wan22VAE",
    "Wan22StreamingDecoder",
    "count_causal_convs",
    "latent_mean_std",
    "latent_scale",
    "load_wan22_vae",
    "patchify",
    "unpatchify",
]

#: Input frames a causal conv keeps from the previous call.
CACHE_T = 2
#: Latent channels of the Wan2.2 VAE.
LATENT_CHANNELS = 48
#: File name inside the Wan2.2-TI2V-5B snapshot directory.
VAE_CHECKPOINT_NAME = "Wan2.2_VAE.pth"

#: Per-channel latent mean and std of the Wan2.2 VAE.
LATENT_MEAN: tuple[float, ...] = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)
LATENT_STD: tuple[float, ...] = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)


def latent_mean_std() -> tuple[torch.Tensor, torch.Tensor]:
    """``(mean, std)``, float32 ``[48]`` on the CPU, with ``std = 1 / (1 / LATENT_STD)`` as decoded.

    The round trip changes 6 of the 48 channels in float32; the decode scale comes out the same
    either way, but the round trip is kept rather than relied on.
    """
    mean = torch.tensor(LATENT_MEAN, dtype=torch.float32)
    inv_std = 1.0 / torch.tensor(LATENT_STD, dtype=torch.float32)
    return mean, inv_std.reciprocal()


def latent_scale(device: torch.device | str, dtype: torch.dtype) -> list[torch.Tensor]:
    """``[mean, 1/std]`` on ``device`` in ``dtype``, the ``scale`` argument of encode and decode.

    The reciprocal is taken after the cast, as decoded: in bf16, ``(1.0 / std).to(bf16)`` differs in
    18 of the 48 channels.
    """
    mean, std = latent_mean_std()
    # Attribution: the constants are Wan2.2's (Apache-2.0); the reciprocal after the cast follows
    # CausVid's decode_to_pixel (github.com/tianweiy/CausVid at fab2440f, MIT) via Self Forcing
    # (github.com/guandeh17/Self-Forcing, Apache-2.0).
    return [mean.to(device=device, dtype=dtype), 1.0 / std.to(device=device, dtype=dtype)]


# ================================================================================== tensor reshapes
def _frames_to_batch(x: torch.Tensor) -> torch.Tensor:
    """``rearrange(x, "b c t h w -> (b t) c h w")`` with einops' exact ops."""
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def _batch_to_frames(x: torch.Tensor, t: int) -> torch.Tensor:
    """``rearrange(x, "(b t) c h w -> b c t h w", t=t)`` with einops' exact ops (a view)."""
    bt, c, h, w = x.shape
    return x.reshape(bt // t, t, c, h, w).permute(0, 2, 1, 3, 4)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``"b c f (h q) (w r) -> b (c r q) f h w"``: ``[b, c, f, H, W] -> [b, c p p, f, h, w]``."""
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"patchify expects [b, c, f, H, W], got {tuple(x.shape)}")
    p = patch_size
    b, c, f, hh, ww = x.shape
    h, w = hh // p, ww // p
    return (
        x.reshape(b, c, f, h, p, w, p).permute(0, 1, 6, 4, 2, 3, 5).reshape(b, c * p * p, f, h, w)
    )


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``"b (c r q) f h w -> b c f (h q) (w r)"``: ``[b, c p p, f, h, w] -> [b, c, f, H, W]``."""
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"unpatchify expects [b, c p p, f, h, w], got {tuple(x.shape)}")
    p = patch_size
    b, cpp, f, h, w = x.shape
    c = cpp // (p * p)
    return (
        x.reshape(b, c, p, p, f, h, w).permute(0, 1, 4, 5, 3, 6, 2).reshape(b, c, f, h * p, w * p)
    )


# ======================================================================================== the cache
#: Slot value of a temporal-upsample conv after the first frame (which that conv skips).
_FIRST_FRAME_SEEN = "Rep"


class CausalConvCache:
    """Temporal state of every :class:`CausalConv3d` of an encoder or decoder, for one video.

    Every frame visits the convs in the same order; ``cursor`` is the next slot and is rewound
    before each frame. A slot holds the last ``CACHE_T`` input frames of its conv, ``None`` before
    the first frame, or the first-frame marker of a temporal-upsample conv.
    """

    def __init__(self, num_slots: int) -> None:
        self.slots: list[None | str | torch.Tensor] = [None] * int(num_slots)
        self.cursor = 0

    def rewind(self) -> None:
        self.cursor = 0

    def causal_conv(self, conv: "CausalConv3d", x: torch.Tensor) -> torch.Tensor:
        """``conv`` of ``x`` ``[b, c, t, h, w]`` after its slot's frames; ``x``'s tail is stored."""
        i = self.cursor
        prev = self.slots[i]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and prev is not None:
            # A one-frame input: keep the previous call's last frame as well.
            cache_x = torch.cat(
                [prev[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
            )
        x = conv(x, prev)
        self.slots[i] = cache_x
        self.cursor = i + 1
        return x


def count_causal_convs(module: nn.Module) -> int:
    """Number of :class:`CausalConv3d` in ``module`` (the cache size)."""
    return sum(1 for m in module.modules() if isinstance(m, CausalConv3d))


# =========================================================================================== layers
class CausalConv3d(nn.Conv3d):
    """3-D conv padded only on the past side in time (``2 * pad_t`` frames before, none after).

    ``cache_x`` (the previous frames of the same stream) replaces that many zero frames.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMSNorm(nn.Module):
    """``normalize(x, dim=1) * sqrt(C) * gamma`` over channels (channel-first)."""

    def __init__(self, dim: int, images: bool = True) -> None:
        super().__init__()
        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones((dim, *broadcastable_dims)))
        self.bias = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    """Nearest-exact upsampling computed in float32 (bf16 support), returned in the input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """Spatial x2 up- or downsampling, with a causal temporal x2 in the ``3d`` modes."""

    MODES = ("upsample2d", "upsample3d", "downsample2d", "downsample3d")

    def __init__(self, dim: int, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown resample mode {mode!r}")
        super().__init__()
        self.mode = mode
        if mode.startswith("upsample"):
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        else:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        if mode == "upsample3d":
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample3d":
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x: torch.Tensor, cache: CausalConvCache) -> torch.Tensor:
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            i = cache.cursor
            prev = cache.slots[i]
            if prev is None:
                # First frame of the stream: no temporal upsampling, remember that it was seen.
                cache.slots[i] = _FIRST_FRAME_SEEN
                cache.cursor = i + 1
            else:
                first = isinstance(prev, str)
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and not first:
                    cache_x = torch.cat(
                        [prev[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                if cache_x.shape[2] < 2 and first:
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2
                    )
                x = self.time_conv(x) if first else self.time_conv(x, prev)
                cache.slots[i] = cache_x
                cache.cursor = i + 1
                # [b, 2c, t] -> [b, c, 2t]: the two output halves interleave in time.
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = _frames_to_batch(x)
        x = self.resample(x)
        x = _batch_to_frames(x, t)

        if self.mode == "downsample3d":
            i = cache.cursor
            prev = cache.slots[i]
            if prev is None:
                cache.slots[i] = x.clone()
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([prev[:, :, -1:, :, :], x], 2))
                cache.slots[i] = cache_x
            cache.cursor = i + 1
        return x


class ResidualBlock(nn.Module):
    """``RMSNorm -> SiLU -> CausalConv3d``, twice, plus a (1x1x1 conv) shortcut."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Identity(),  # dropout in training; keeps the checkpoint's Sequential indices
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, cache: CausalConvCache) -> torch.Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            x = cache.causal_conv(layer, x) if isinstance(layer, CausalConv3d) else layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention per frame (no temporal mixing)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = _frames_to_batch(x)
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = _batch_to_frames(x, t)
        return x + identity


class AvgDown3D(nn.Module):
    """Shortcut of a down block: space/time-to-channel then group mean (parameter free)."""

    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        if (in_channels * self.factor) % out_channels:
            raise ValueError("in_channels * factor must be divisible by out_channels")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.view(B, C, T // ft, ft, H // fs, fs, W // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // ft, H // fs, W // fs)
        x = x.view(B, self.out_channels, self.group_size, T // ft, H // fs, W // fs)
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    """Shortcut of an up block: channel repeat then channel-to-space/time (parameter free)."""

    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        if (out_channels * self.factor) % in_channels:
            raise ValueError("out_channels * factor must be divisible by in_channels")
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        ft, fs = self.factor_t, self.factor_s
        x = x.view(x.size(0), self.out_channels, ft, fs, fs, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * ft, x.size(4) * fs, x.size(6) * fs)
        if first_chunk:
            # latent 0 is one pixel frame: drop the frames the temporal upsample invented
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class DownResidualBlock(nn.Module):
    """``mult`` residual blocks and an optional downsample, plus an ``AvgDown3D`` shortcut."""

    def __init__(
        self, in_dim: int, out_dim: int, mult: int, temperal_downsample: bool, down_flag: bool
    ) -> None:
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        downsamples: list[nn.Module] = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if down_flag:
            downsamples.append(
                Resample(out_dim, mode="downsample3d" if temperal_downsample else "downsample2d")
            )
        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x: torch.Tensor, cache: CausalConvCache) -> torch.Tensor:
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, cache)
        return x + self.avg_shortcut(x_copy)


class UpResidualBlock(nn.Module):
    """``mult`` residual blocks and an optional upsample, plus a ``DupUp3D`` shortcut."""

    def __init__(
        self, in_dim: int, out_dim: int, mult: int, temperal_upsample: bool, up_flag: bool
    ) -> None:
        super().__init__()
        if up_flag:
            self.avg_shortcut: DupUp3D | None = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None
        upsamples: list[nn.Module] = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if up_flag:
            upsamples.append(
                Resample(out_dim, mode="upsample3d" if temperal_upsample else "upsample2d")
            )
        self.upsamples = nn.Sequential(*upsamples)

    def forward(
        self, x: torch.Tensor, cache: CausalConvCache, first_chunk: bool = False
    ) -> torch.Tensor:
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, cache)
        if self.avg_shortcut is not None:
            return x_main + self.avg_shortcut(x, first_chunk)
        return x_main


class Encoder3d(nn.Module):
    """Patchified pixels ``[b, 12, t, H/2, W/2]`` -> mean | log-variance ``[b, 2 z, t', h, w]``."""

    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        temperal_downsample: Sequence[bool],
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [1] + list(dim_mult)]
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(
                DownResidualBlock(
                    in_dim, out_dim, num_res_blocks, t_down, down_flag=i != len(dim_mult) - 1
                )
            )
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim),
        )
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, cache: CausalConvCache) -> torch.Tensor:
        x = cache.causal_conv(self.conv1, x)
        for layer in self.downsamples:
            x = layer(x, cache)
        for layer in self.middle:
            x = layer(x, cache) if isinstance(layer, ResidualBlock) else layer(x)
        for layer in self.head:
            x = cache.causal_conv(layer, x) if isinstance(layer, CausalConv3d) else layer(x)
        return x


class Decoder3d(nn.Module):
    """One latent frame ``[b, z, 1, h, w]`` -> patchified pixels ``[b, 12, 1 or 4, 8 h, 8 w]``."""

    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        temperal_upsample: Sequence[bool],
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult)[::-1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0]),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0]),
        )
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                UpResidualBlock(
                    in_dim, out_dim, num_res_blocks + 1, t_up, up_flag=i != len(dim_mult) - 1
                )
            )
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 12, 3, padding=1)
        )

    def forward(
        self, x: torch.Tensor, cache: CausalConvCache, first_chunk: bool = False
    ) -> torch.Tensor:
        x = cache.causal_conv(self.conv1, x)
        for layer in self.middle:
            x = layer(x, cache) if isinstance(layer, ResidualBlock) else layer(x)
        for layer in self.upsamples:
            x = layer(x, cache, first_chunk)
        for layer in self.head:
            x = cache.causal_conv(layer, x) if isinstance(layer, CausalConv3d) else layer(x)
        return x


# ========================================================================================== the VAE
@dataclass(frozen=True)
class VAEConfig:
    """VAE dimensions; the defaults (:data:`WAN22_VAE`) are the Wan2.2-TI2V-5B VAE's."""

    dim: int = 160
    dec_dim: int = 256
    z_dim: int = LATENT_CHANNELS
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temperal_downsample: tuple[bool, ...] = (False, True, True)


WAN22_VAE = VAEConfig()


def _normalized_to_raw(z: torch.Tensor, scale: Sequence[torch.Tensor], z_dim: int) -> torch.Tensor:
    """Undo the latent normalisation: ``z / (1/std) + mean`` with ``scale = [mean, 1/std]``."""
    return z / scale[1].view(1, z_dim, 1, 1, 1) + scale[0].view(1, z_dim, 1, 1, 1)


class Wan22VAE(nn.Module):
    """The Wan2.2 VAE on normalised latents; ``scale`` is :func:`latent_scale`."""

    def __init__(self, config: VAEConfig = WAN22_VAE) -> None:
        super().__init__()
        self.config = config
        self.z_dim = config.z_dim
        self.encoder = Encoder3d(
            config.dim,
            config.z_dim * 2,
            config.dim_mult,
            config.num_res_blocks,
            config.temperal_downsample,
        )
        self.conv1 = CausalConv3d(config.z_dim * 2, config.z_dim * 2, 1)
        self.conv2 = CausalConv3d(config.z_dim, config.z_dim, 1)
        self.decoder = Decoder3d(
            config.dec_dim,
            config.z_dim,
            config.dim_mult,
            config.num_res_blocks,
            tuple(config.temperal_downsample)[::-1],
        )

    def encode(self, x: torch.Tensor, scale: Sequence[torch.Tensor]) -> torch.Tensor:
        """Pixels ``[b, 3, 1 + 4 k, H, W]`` in [-1, 1] -> normalised mean ``[b, z, 1 + k, h, w]``
        (``h, w = H / 16, W / 16``)."""
        cache = CausalConvCache(count_causal_convs(self.encoder))
        x = patchify(x, patch_size=2)
        chunks = [x[:, :, :1]] + [
            x[:, :, 1 + 4 * i : 5 + 4 * i] for i in range((x.shape[2] - 1) // 4)
        ]
        outs = []
        for chunk in chunks:  # frame 0 alone, then 4 frames per latent
            cache.rewind()
            outs.append(self.encoder(chunk, cache))
        mu, _log_var = self.conv1(torch.cat(outs, 2)).chunk(2, dim=1)
        return (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)

    def decode(self, z: torch.Tensor, scale: Sequence[torch.Tensor]) -> torch.Tensor:
        """Normalised latents ``[b, z, T, h, w]`` -> pixels ``[b, 3, 1 + 4 (T-1), 16 h, 16 w]``.

        One fresh cache over the whole sequence (:class:`Wan22StreamingDecoder` gives the same
        frames chunk by chunk); the pixels are not clamped.
        """
        cache = CausalConvCache(count_causal_convs(self.decoder))
        x = self.conv2(_normalized_to_raw(z, scale, self.z_dim))
        outs = []
        for i in range(x.shape[2]):
            cache.rewind()
            outs.append(self.decoder(x[:, :, i : i + 1], cache, first_chunk=(i == 0)))
        return unpatchify(torch.cat(outs, 2), patch_size=2)


class Wan22StreamingDecoder:
    """Chunked decoding of one video: consecutive chunks through :meth:`decode_chunk` concatenate to
    :meth:`Wan22VAE.decode` of the whole sequence. Call :meth:`reset` before a new video."""

    def __init__(self, model: Wan22VAE) -> None:
        self.model = model
        self.reset()

    def reset(self) -> None:
        """Start a new video: empty cache, next latent is latent 0."""
        self._cache = CausalConvCache(count_causal_convs(self.model.decoder))
        self._decoded_frames = 0

    @torch.no_grad()
    def decode_chunk(self, z_chunk: torch.Tensor, scale: Sequence[torch.Tensor]) -> torch.Tensor:
        """The next chunk of latents ``[B, z, T, h, w]`` -> pixels ``[B, 3, T', 16 h, 16 w]``.

        ``T' = 1 + 4 (T - 1)`` for the chunk holding latent 0, else ``4 T``; not clamped.
        """
        if z_chunk.ndim != 5:
            raise ValueError("z_chunk must have shape [B, z_dim, T, H, W]")
        if z_chunk.shape[2] < 1:
            raise ValueError("z_chunk must contain at least one latent frame")
        x = self.model.conv2(_normalized_to_raw(z_chunk, scale, self.model.z_dim))
        outs = []
        for i in range(x.shape[2]):
            self._cache.rewind()
            outs.append(
                self.model.decoder(
                    x[:, :, i : i + 1], self._cache, first_chunk=(self._decoded_frames == 0)
                )
            )
            self._decoded_frames += 1
        return unpatchify(torch.cat(outs, 2), patch_size=2)


def load_wan22_vae(
    checkpoint_path: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    config: VAEConfig = WAN22_VAE,
) -> Wan22VAE:
    """The VAE with ``Wan2.2_VAE.pth`` loaded strictly, in eval mode without grads."""
    with torch.device("meta"):
        model = Wan22VAE(config)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, assign=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=dtype)
