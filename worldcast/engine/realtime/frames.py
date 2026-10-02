"""From latent frames to bytes on the wire: streaming decoders and per-frame encoders.

Decoders take a client's latents one chunk at a time (``[n, 48, 24, 42]``, any float dtype, as the client stores
them) and return uint8 frames ``[t, 3, 384, 672]`` on their device; the first chunk of a stream starts with latent
0, which gives one frame, every later latent four.

* :class:`WanFrameDecoder`: the Wan2.2 VAE with its causal conv cache carried across chunks (the paper decoder;
  its frames do not depend on the chunking except through the kernel choice of one 1x1x1 conv).
* :class:`TinyFrameDecoder`: the ``taew2_2`` tiny decoder (:mod:`worldcast.engine.realtime.taehv`).

Encoders turn uint8 frames into bytes: :class:`JpegEncoder` (libjpeg on the CPU, or nvJPEG on the GPU) and
:class:`H264Encoder` (libx264, ``ultrafast`` / ``zerolatency``: no B-frames, no lookahead, one packet per frame).
"""

from collections.abc import Sequence

import torch

from worldcast.modeling.wan22.vae import Wan22StreamingDecoder, Wan22VAE, latent_scale

from .taehv import StreamingTinyDecoder, TinyDecoder

__all__ = [
    "to_uint8",
    "WanFrameDecoder",
    "TinyFrameDecoder",
    "JpegEncoder",
    "H264Encoder",
    "make_decoder",
]


def to_uint8(pixels: torch.Tensor) -> torch.Tensor:
    """``[t, 3, H, W]`` in [0, 1] -> uint8, as the paper's mp4 frames (``round`` half to even on ``x * 255``)."""
    return (pixels.float().clamp(0, 1) * 255).round().to(torch.uint8)


class WanFrameDecoder:
    """The Wan2.2 VAE decoder, streamed (paper post-processing: clamp to [-1, 1], ``x / 2 + 1/2``, clamp)."""

    name = "wan"

    def __init__(self, vae: Wan22VAE) -> None:
        param = next(vae.parameters())
        self.device, self.dtype = param.device, param.dtype
        self.scale = latent_scale(self.device, self.dtype)
        self.stream = Wan22StreamingDecoder(vae)

    def reset(self) -> None:
        self.stream.reset()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(device=self.device, dtype=self.dtype)[None].permute(0, 2, 1, 3, 4)
        pixels = self.stream.decode_chunk(z, self.scale).float().clamp_(-1, 1)
        return to_uint8((pixels.permute(0, 2, 1, 3, 4)[0] * 0.5 + 0.5).clamp(0, 1))


class TinyFrameDecoder:
    """The ``taew2_2`` tiny decoder, streamed."""

    name = "taehv"

    def __init__(self, decoder: TinyDecoder) -> None:
        param = next(decoder.parameters())
        self.device, self.dtype = param.device, param.dtype
        self.stream = StreamingTinyDecoder(decoder)

    def reset(self) -> None:
        self.stream.reset()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return to_uint8(self.stream.decode(latents))


def make_decoder(name: str, *, wan_vae: Wan22VAE | None = None, tiny: TinyDecoder | None = None):
    """A fresh streaming decoder over loaded weights (``wan`` or ``taehv``)."""
    if name == "wan":
        if wan_vae is None:
            raise ValueError(
                "the wan decoder needs the Wan2.2 VAE (paths.wan22_root/Wan2.2_VAE.pth)"
            )
        return WanFrameDecoder(wan_vae)
    if name == "taehv":
        if tiny is None:
            raise ValueError("the taehv decoder needs the taew2_2 weights")
        return TinyFrameDecoder(tiny)
    raise ValueError(f"unknown decoder {name!r}")


class JpegEncoder:
    """JPEG per frame: ``device='cpu'`` with libjpeg (default), ``cuda`` with nvJPEG (torchvision).

    torchvision's nvJPEG encoder runs on its own stream and orders itself only against the stream that was current
    at its first use; frames decoded on another stream came out wrong on the H20. The CUDA path therefore waits for
    the frames before and for the whole device after encoding, which stalls the decode thread behind the next
    block's generation; use it only without ``decode_overlap``.
    """

    def __init__(self, quality: int = 90, device: str = "cpu") -> None:
        self.quality, self.device = int(quality), str(device)

    def encode(self, frames: torch.Tensor) -> list[bytes]:
        from torchvision.io import encode_jpeg

        if self.device == "cuda" and frames.is_cuda:
            torch.cuda.current_stream(frames.device).synchronize()
            data = encode_jpeg(list(frames), quality=self.quality)
            torch.cuda.synchronize(frames.device)
        else:
            data = [encode_jpeg(f, quality=self.quality) for f in frames.cpu()]
        return [d.cpu().numpy().tobytes() for d in data]


class H264Encoder:
    """H.264 with libx264 (PyAV), tuned for latency: ``ultrafast``, ``zerolatency``, every frame its own packet."""

    def __init__(
        self, width: int, height: int, fps: float = 16.0, bitrate: int = 4_000_000
    ) -> None:
        from fractions import Fraction

        import av

        self._av = av
        self.codec = av.CodecContext.create("libx264", "w")
        self.codec.width, self.codec.height = int(width), int(height)
        self.codec.pix_fmt = "yuv420p"
        self.codec.time_base = Fraction(1, int(round(fps)))
        self.codec.bit_rate = int(bitrate)
        self.codec.options = {"preset": "ultrafast", "tune": "zerolatency", "bframes": "0"}
        self._pts = 0

    def encode(self, frames: torch.Tensor) -> list[bytes]:
        out = []
        for f in frames.permute(0, 2, 3, 1).cpu().numpy():
            frame = self._av.VideoFrame.from_ndarray(f, format="rgb24").reformat(format="yuv420p")
            frame.pts = self._pts
            self._pts += 1
            out.append(b"".join(bytes(p) for p in self.codec.encode(frame)))
        return out


def encode_frames(encoder, frames: torch.Tensor) -> Sequence:
    """``encoder.encode(frames)``, or the uint8 HWC arrays when there is no encoder."""
    if encoder is None:
        return list(frames.permute(0, 2, 3, 1).cpu().numpy())
    return encoder.encode(frames)
