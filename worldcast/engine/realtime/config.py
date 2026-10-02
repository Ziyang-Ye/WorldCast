"""Serving options of :class:`worldcast.engine.realtime.engine.Engine`.

The defaults are the paper client: the release generator call, the Wan2.2 VAE, lock-step with the peers. The
options marked *exact* keep every latent bit-identical to it; the others change numbers and are opt-in.
docs/latency.md lists what each option costs and saves.
"""

from dataclasses import dataclass, replace

__all__ = ["RealtimeConfig", "GENERATORS", "ATTENTIONS", "DECODERS", "ENCODERS"]

#: ``eager``: the release generator call (paper path). ``fast``: the same kernels without the per-call host
#: syncs and repeated work (exact, :mod:`worldcast.engine.realtime.fast`).
GENERATORS = ("eager", "fast")
#: ``flash``: flash-attention 2/3 as deployed (exact). ``cudnn``: PyTorch SDPA on cuDNN's Hopper kernel.
#: ``fa3``: FlashAttention-3 (``flash_attn_interface``). The last two change the bits.
ATTENTIONS = ("flash", "cudnn", "fa3")
#: ``wan``: the Wan2.2 VAE, streamed one latent frame at a time (paper decoder). ``taehv``: the tiny
#: ``taew2_2`` decoder. ``none``: latents only.
DECODERS = ("wan", "taehv", "none")
#: Per-frame encoding for the network: ``jpeg``, ``h264`` or ``none`` (raw uint8 frames).
ENCODERS = ("jpeg", "h264", "none")


@dataclass(frozen=True)
class RealtimeConfig:
    """How an :class:`~worldcast.engine.realtime.engine.Engine` serves frames.

    Generation:
        generator: :data:`GENERATORS` (``fast`` is exact).
        cuda_graphs: replay the 30 DiT blocks of each recurring call shape from a CUDA graph (exact; needs
            ``generator="fast"`` and CUDA).
        attention: :data:`ATTENTIONS` (only ``flash`` is exact).
        compile: ``torch.compile`` the dense parts of each DiT block (not exact).
        early_prefill: write the next block's context into the KV cache as soon as the previous block is done,
            before its controls arrive; only the 4-step ladder waits for the controls (exact with GT tracks; with
            predicted states the retrieval and the ray anchor use the controls held at prefill time).
    Decoding and transport:
        decoder: :data:`DECODERS`. ``taehv_path``: the ``taew2_2.pth`` weights.
        decode_overlap: decode and encode on a worker thread and a second CUDA stream, overlapped with the next
            block's generation (exact for the latents).
        decode_device: device of the decoder (default: the generator's).
        encoder: :data:`ENCODERS`; ``jpeg_quality``; ``jpeg_device`` ``cpu`` (libjpeg, 1.7 ms a frame) or
            ``cuda`` (nvJPEG; not safe with ``decode_overlap``, see
            :class:`~worldcast.engine.realtime.frames.JpegEncoder`).
    Peers:
        lockstep: wait for every peer before each block (paper evaluation; deterministic). Off: use the
            messages that have arrived (lowest latency; the output then depends on timing).
        poll_s: lock-step poll period, s.
    Sub-block modes (experimental, not exact; docs/latency.md has the quality test):
        commit_latents: latent frames kept from each generation (4 = the paper's block). With fewer, the next
            generation starts right after them with newer controls, so the controls are taken every
            ``commit_latents / 4`` s, at the cost of ``4 / commit_latents`` generations per second.
        target_latents: latent frames denoised per generation (4 = the paper's block; 1 = a one-latent target).
    """

    generator: str = "eager"
    cuda_graphs: bool = False
    attention: str = "flash"
    compile: bool = False
    early_prefill: bool = False
    decoder: str = "wan"
    taehv_path: str | None = None
    decode_overlap: bool = False
    decode_device: str | None = None
    encoder: str = "none"
    jpeg_quality: int = 90
    jpeg_device: str = "cpu"
    lockstep: bool = True
    poll_s: float = 0.002
    commit_latents: int = 4
    target_latents: int = 4

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("generator", self.generator, GENERATORS),
            ("attention", self.attention, ATTENTIONS),
            ("decoder", self.decoder, DECODERS),
            ("encoder", self.encoder, ENCODERS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if self.cuda_graphs and self.generator != "fast":
            raise ValueError("cuda_graphs needs generator='fast' (host-side cache indices)")
        if self.compile and self.generator != "fast":
            raise ValueError("compile needs generator='fast'")
        if self.decoder == "taehv" and not self.taehv_path:
            raise ValueError("decoder='taehv' needs taehv_path (the taew2_2.pth weights)")
        if not 1 <= int(self.jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.jpeg_device not in ("cuda", "cpu"):
            raise ValueError("jpeg_device must be 'cuda' or 'cpu'")
        if float(self.poll_s) <= 0:
            raise ValueError("poll_s must be positive")
        if not 1 <= int(self.commit_latents) <= int(self.target_latents) <= 4:
            raise ValueError("need 1 <= commit_latents <= target_latents <= 4")
        if self.sub_block and self.lockstep:
            raise ValueError(
                "the sub-block modes publish whole blocks late: run them without lock-step"
            )

    @property
    def sub_block(self) -> bool:
        return int(self.commit_latents) < 4 or int(self.target_latents) < 4

    @property
    def exact(self) -> bool:
        """Whether the latents are bit-identical to the paper client."""
        return self.attention == "flash" and not self.compile and not self.sub_block

    @classmethod
    def paper(cls) -> "RealtimeConfig":
        """The paper client as it ran: release generator call, Wan2.2 VAE, lock-step."""
        return cls()

    @classmethod
    def low_latency(cls, taehv_path: str | None = None, **overrides) -> "RealtimeConfig":
        """The recommended demo settings (docs/latency.md): exact fast generator with CUDA graphs, early prefill,
        overlapped streaming decode (tiny VAE when ``taehv_path`` is given), JPEG frames, no lock-step.
        """
        base = cls(
            generator="fast",
            cuda_graphs=True,
            early_prefill=True,
            decode_overlap=True,
            decoder="taehv" if taehv_path else "wan",
            taehv_path=taehv_path,
            encoder="jpeg",
            lockstep=False,
        )
        return replace(base, **overrides)
