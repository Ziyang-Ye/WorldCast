"""How a :class:`~worldcast.engine.inference.client.Client` is served: decoder, lockstep, speed.

The defaults keep every latent bit-identical to the paper's run; ``compile`` changes the last bits
of every step, not the image quality, as does the attention kernel ``fa3`` (``model.attention`` of
the inference config).
"""

from dataclasses import dataclass

__all__ = ["DECODERS", "ServingOptions"]

#: ``wan``: the Wan2.2 VAE streamed one latent frame at a time (the paper's decoder). ``none``: no
#: decode; each latent frame is one frame.
DECODERS = ("wan", "none")


@dataclass(frozen=True)
class ServingOptions:
    """How a client decodes and waits for the other clients, and what speeds it up.

    Attributes:
        cuda_graphs (bool): replay the DiT of each recurring call shape from a CUDA graph (exact).
        compile (bool): ``torch.compile`` the dense parts of each DiT block (not exact).
        decoder (str): one of :data:`DECODERS`.
        lockstep (bool): wait for every other client before each block (the paper's evaluation;
            the output is independent of timing). Off: no waiting; a client reads what the other
            clients have published so far.
    """

    cuda_graphs: bool = True
    compile: bool = False
    decoder: str = "wan"
    lockstep: bool = True

    def __post_init__(self) -> None:
        if self.decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got {self.decoder!r}")
