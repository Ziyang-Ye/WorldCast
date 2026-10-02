"""The decoder of TAEHV ``taew2_2``: a tiny causal decoder for Wan2.2-TI2V-5B latents (48 x h x w -> 3 x 16h x 16w).

A fast preview decoder for the client's latents (``taew2_2.pth``, 9.9M decoder parameters against the Wan2.2 VAE's
~0.7B). It reads the latents the DiT produces (normalised, as the client stores them) and returns frames in [0, 1].
Each latent frame gives four video frames; the first three outputs of a stream are dropped, so latent 0 gives one
frame, as the Wan2.2 VAE does. :class:`StreamingTinyDecoder` decodes a stream a chunk of latents at a time.

Attribution: the architecture and weights are TAEHV by Ollin Boer Bohan (github.com/madebyollin/taehv at 011dfc2,
MIT License, (c) 2025 Ollin Boer Bohan). Only the decoder is reproduced here, under its checkpoint layout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TinyDecoder", "StreamingTinyDecoder", "load_taew2_2"]

LATENT_CHANNELS = 48
PATCH = 2  # output pixel-shuffle factor
WIDTHS = (256, 128, 64, 64)
TIME_UPSCALE = (False, True, True)
FRAMES_TO_TRIM = 2 ** sum(TIME_UPSCALE) - 1


def _conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class _MemBlock(nn.Module):
    """A residual block that also sees the previous frame's input at its depth."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            _conv(n_out, n_out),
            nn.ReLU(inplace=True),
            _conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class _TGrow(nn.Module):
    """Temporal upsampling: each frame becomes ``stride`` frames."""

    def __init__(self, n_f: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


class TinyDecoder(nn.Module):
    """The ``decoder`` of TAEHV with the ``taew2_2`` settings (keys ``decoder.<i>.*``)."""

    def __init__(self) -> None:
        super().__init__()
        n = WIDTHS
        layers: list[nn.Module] = [_Clamp(), _conv(LATENT_CHANNELS, n[0]), nn.ReLU(inplace=True)]
        for i in range(3):
            layers += [_MemBlock(n[i], n[i]) for _ in range(3)]
            layers += [
                nn.Upsample(scale_factor=2),
                _TGrow(n[i], 2 if TIME_UPSCALE[i] else 1),
                _conv(n[i], n[i + 1], bias=False),
            ]
        layers += [nn.ReLU(inplace=True), _conv(n[3], 3 * PATCH**2)]
        self.decoder = nn.Sequential(*layers)

    def load_checkpoint(self, path: str) -> "TinyDecoder":
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if k.startswith("decoder.")}
        own = self.state_dict()
        for i, layer in enumerate(self.decoder):  # TGrow(stride 1) keeps the last-timestep channels
            key = f"decoder.{i}.conv.weight"
            if isinstance(layer, _TGrow) and state[key].shape[0] > own[key].shape[0]:
                state[key] = state[key][-own[key].shape[0] :]
        self.load_state_dict(state, strict=True)
        return self


class StreamingTinyDecoder:
    """Decode one latent stream chunk by chunk; the per-block memory is carried across chunks, so the frames do
    not depend on the chunking (up to kernel choice).

    ``decode(latents [n, 48, h, w])`` returns ``[t, 3, 16 h, 16 w]`` in [0, 1] in the decoder's dtype:
    ``4 n`` frames, minus the three dropped at the start of the stream.
    """

    def __init__(self, decoder: TinyDecoder) -> None:
        self.decoder = decoder
        self.reset()

    def reset(self) -> None:
        self._memory: list[torch.Tensor | None] = [None] * len(self.decoder.decoder)
        self._to_trim = FRAMES_TO_TRIM

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        param = self.decoder.decoder[1].weight
        x = latents.to(device=param.device, dtype=param.dtype)
        for i, layer in enumerate(self.decoder.decoder):
            if isinstance(layer, _MemBlock):
                prev = self._memory[i]
                prev = torch.zeros_like(x[:1]) if prev is None else prev
                past = torch.cat([prev, x[:-1]], 0)
                self._memory[i] = x[-1:].clone()
                x = layer(x, past)
            else:
                x = layer(x)
        frames = F.pixel_shuffle(x, PATCH).clamp_(0, 1)
        if self._to_trim:
            cut = min(self._to_trim, int(frames.shape[0]))
            frames, self._to_trim = frames[cut:], self._to_trim - cut
        return frames


def load_taew2_2(
    path: str, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float16
) -> TinyDecoder:
    """The ``taew2_2`` decoder in eval mode, no grads (upstream runs it in fp16)."""
    return (
        TinyDecoder()
        .load_checkpoint(path)
        .eval()
        .requires_grad_(False)
        .to(device=device, dtype=dtype)
    )
