"""How a :class:`~worldcast.engine.inference.client.Client` is served: decoder, lockstep, speed.

The defaults keep every latent bit-identical to the paper's run; ``compile`` changes the last bits
of every step, not the image quality (docs/demo.md, "Latency"), as does the attention kernel
``fa3`` (``model.attention`` of the inference config).
:meth:`ServingOptions.low_latency` with :data:`LOW_LATENCY_ATTENTION` is the web demo's setting.
"""

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["DECODERS", "LOW_LATENCY_ATTENTION", "ServingOptions"]

#: ``model.attention`` of the web demo: FlashAttention-3 when it is installed.
LOW_LATENCY_ATTENTION = "fa3"
#: ``wan``: the Wan2.2 VAE streamed one latent frame at a time (the paper's decoder). ``taehv``:
#: the tiny ``taew2_2`` decoder. ``none``: no decode; each latent frame is one frame.
DECODERS = ("wan", "taehv", "none")
#: The devices ``decoder_device`` names: ``cpu``, ``cuda`` or ``cuda:<index>``.
_DEVICE = re.compile(r"cpu|cuda(:\d+)?")
#: The cards ``refresh_devices`` names: ``cuda:<index>``, or ``cpu`` (a stand-in for tests).
_CARD = re.compile(r"cpu|cuda:\d+")


@dataclass(frozen=True)
class ServingOptions:
    """How a client decodes and waits for the other clients, and what speeds it up.

    Attributes:
        cuda_graphs (bool): replay the DiT of each recurring call shape from a CUDA graph (exact).
        compile (bool): ``torch.compile`` the dense parts of each DiT block (not exact).
        decoder (str): one of :data:`DECODERS`.
        taehv_path (str | None): the ``taew2_2.pth`` weights of the ``taehv`` decoder.
        lockstep (bool): wait for every other client before each block (the paper's evaluation;
            the output is independent of timing). Off: use the messages that have arrived.
        decoder_device (str | None): the device the decoder runs on, e.g. ``cuda:1`` on a worker
            with two cards (``CUDA_VISIBLE_DEVICES=0,1``): its weights live there, each latent
            frame is copied there when it is decoded, and the demo's worker generates the next
            block while the frames of this one are decoded. ``None``: the client's device
            (``run.device``) and a block's frames decoded before the next block starts. The frames
            are the same either way (docs/demo.md, "The original VAE on a second card").
        decoder_channels_last (bool): keep the Wan2.2 VAE's decoding convolutions in the
            channels-last layout: a latent frame decodes faster, in other kernels, so the pixels
            are not the paper's decoder's byte for byte (``decoder="wan"`` only).
        refresh_devices (tuple[str, ...]): the cards a live client's refreshes run on
            (``live.refresh``), the client's own (``run.device``) first, e.g. ``("cuda:0",
            "cuda:1")`` on a worker with two cards: each other card holds a replica of the
            generator, which every block's context is prefilled on too, and a refresh runs on a
            card that is free, so that it never waits for the one before (docs/demo.md, "Refresh
            on more cards"); each card by its index, once, none the decoder's. Empty: the
            client's own card alone.
    """

    cuda_graphs: bool = True
    compile: bool = False
    decoder: str = "wan"
    taehv_path: str | None = None
    lockstep: bool = True
    decoder_device: str | None = None
    decoder_channels_last: bool = False
    refresh_devices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cards = tuple(map(str, self.refresh_devices))
        object.__setattr__(self, "refresh_devices", cards)
        for card in cards:
            if not _CARD.fullmatch(card):
                raise ValueError(f"refresh_devices are cuda:<index> (or cpu), got {card!r}")
        if len(cards) == 1:
            raise ValueError(
                "refresh_devices names the client's card and at least one more (empty: the"
                " client's alone)"
            )
        gpus = [card for card in cards if card != "cpu"]
        if len(set(gpus)) != len(gpus):
            raise ValueError(f"refresh_devices names a card twice: {', '.join(cards)}")
        if self.decoder_device is not None and str(self.decoder_device) in cards[1:]:
            raise ValueError(
                f"refresh_devices: {self.decoder_device} is the decoder's card (decoder_device),"
                " not a replica's"
            )
        if self.decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got {self.decoder!r}")
        if self.decoder == "taehv" and not self.taehv_path:
            raise ValueError("decoder='taehv' needs taehv_path (the taew2_2.pth weights)")
        if self.decoder_device is not None:
            if not _DEVICE.fullmatch(str(self.decoder_device)):
                raise ValueError(
                    f"decoder_device must be cpu, cuda or cuda:<index>, got {self.decoder_device!r}"
                )
            if self.decoder == "none":
                raise ValueError("decoder_device places a decoder: decoder='none' has none")
        if self.decoder_channels_last and self.decoder != "wan":
            raise ValueError("decoder_channels_last is the Wan2.2 VAE's: it needs decoder='wan'")

    @classmethod
    def low_latency(cls, taehv_path: str | None, **overrides: Any) -> "ServingOptions":
        """The web demo's setting: ``torch.compile``, the tiny decoder, no lockstep (docs/demo.md,
        "Latency", has what each costs).

        Args:
            taehv_path (str | None): the ``taew2_2.pth`` weights; ``None`` where ``overrides`` name
                another decoder.
            **overrides: fields to replace.
        """
        fast = dict(compile=True, decoder="taehv", taehv_path=taehv_path, lockstep=False)
        return cls(**{**fast, **overrides})
