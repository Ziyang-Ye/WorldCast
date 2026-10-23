"""A live client's generator on more cards than its own, so that a refresh of the block on screen
need not wait for the one before (``ServingOptions.refresh_devices``; the web demo's, not the
paper's).

Each other card holds a :class:`GeneratorReplica`: the client's generator, depth head and prompt
embedding loaded again there, a KV cache of its own and a thread of its own. Every block's context
is prefilled on it as on the client's card, from the same window and the same context-write noise,
in parallel. A refresh is cut on the client's card (``LiveRollout.cut``: host work, one cut after
the other) and denoised on a card that is free (``LiveRollout.denoise_cut``), with the block's
entry noise and the re-noise drawn from the state the block's first denoising began with: the
inputs of the client's card, so the x0 of the client's card where the cards run the same kernels
(the worker's warm-up checks it, ``Client.replicas_equal``).
"""

import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch

from worldcast.config.inference import InferenceConfig
from worldcast.engine.generator import backbone_config, load_prompt_embeds
from worldcast.modeling.build import load_generator
from worldcast.modeling.depth_head import load_depth_predictor
from worldcast.modeling.wan22.attention import AttentionFn
from worldcast.player_state import field_builder
from worldcast.sampling.sampler import Sampler
from worldcast.sampling.schedulers import ContextWriteNoise
from worldcast.sampling.window import KV_CACHE_LATENTS

from .fast import FastGenerator
from .serving import ServingOptions

__all__ = ["GeneratorReplica", "card_device", "rng_state", "set_rng_state"]


def card_device(device: str | torch.device) -> torch.device:
    """``device`` with its index: ``cuda`` is the current CUDA device."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def rng_state(device: torch.device) -> torch.Tensor:
    """The state of the global RNG the denoising steps draw their re-noise from on ``device``."""
    return torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()


def set_rng_state(device: torch.device, state: torch.Tensor) -> None:
    """Set the global RNG of ``device`` to ``state`` (:func:`rng_state`)."""
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)


class GeneratorReplica:
    """The client's generator, depth head and prompt embedding on another card, with a KV cache
    and a thread of their own.

    Args:
        cfg (InferenceConfig): the client's config (``paths``: the weights).
        serving (ServingOptions): how the client is served (``cuda_graphs``, ``compile``).
        device (str | torch.device): the card.
        attention (AttentionFn): the client's attention kernel.
        dtype (torch.dtype): the generator's input dtype (bf16 on CUDA).
        context_noise (int): the context noise of the client's generator.

    Attributes:
        sampler (Sampler): the sampler of the replica's generator and KV cache; its re-noise
            draws from :attr:`rng`.
        rng (torch.Generator): the stream of the re-noise, set to the client card's state before a
            denoising.
        thread (ThreadPoolExecutor): the replica's thread: its prefills and denoisings, in order.
    """

    def __init__(
        self,
        cfg: InferenceConfig,
        serving: ServingOptions,
        device: str | torch.device,
        *,
        attention: AttentionFn,
        dtype: torch.dtype,
        context_noise: int,
    ) -> None:
        self.device = card_device(device)
        p, cuda = cfg.paths, self.device.type == "cuda"
        generator = load_generator(p.checkpoint, backbone_config(p.wan22_root), device=self.device)
        self.prompt_embeds = load_prompt_embeds(p.prompt_embedding, p.wan22_root, self.device)
        self.depth = load_depth_predictor(p.depth_head, p.depth_readout, device=self.device)
        fast = FastGenerator(
            generator,
            field_builder(),
            input_dtype=dtype,
            attention=attention,
            cuda_graphs=serving.cuda_graphs and cuda,
            compile=serving.compile,
        )
        self.rng = torch.Generator(device=self.device)
        cache = generator.allocate_kv_cache(KV_CACHE_LATENTS)
        self.sampler = Sampler.create(fast, cache, context_noise=context_noise, rng=self.rng)
        self._ident: int | None = None  # the replica's thread
        self.thread = ThreadPoolExecutor(1, f"worldcast-{self.device}", initializer=self._bind)
        self._prefill: Future | None = None

    def _bind(self) -> None:
        self._ident = threading.get_ident()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

    def conditions(self, conditions: Mapping[str, Any]) -> dict[str, Any]:
        """``conditions`` of the client's card on the replica's: the tensors copied, the prompt
        the replica's own (so that its cross-attention keys and values are computed once)."""
        moved = {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in conditions.items()
            if key != "prompt_embeds"
        }
        batch = int(conditions["prompt_embeds"].shape[0])
        moved["prompt_embeds"] = self.prompt_embeds.expand(batch, -1, -1)
        return moved

    def prefill(
        self,
        context: torch.Tensor,
        conditions: Mapping[str, Any],
        ranges: Sequence[tuple[int, int]],
        noise: ContextWriteNoise | None,
    ) -> None:
        """Prefill a window's context on the replica as the client's card does
        (:meth:`~worldcast.sampling.sampler.Sampler.prefill_context` after a reset of the cache), on
        the replica's thread; the inputs are copied here, on the caller's thread, right after the
        client's card made them. A prefill that failed raises in :meth:`prefilled`.

        Args:
            context (Tensor): ``[1, F, C, H, W]`` the context's latents, on the client's card.
            conditions (Mapping[str, Any]): the window's conditions, on the client's card.
            ranges (Sequence[tuple[int, int]]): the window's context writes.
            noise (ContextWriteNoise | None): their noise (it draws on the CPU, the same each
                time it is called).
        """
        latents = context.to(self.device, non_blocking=True)
        moved = self.conditions(conditions)

        def prefill() -> None:
            with torch.no_grad():
                self.sampler.cache.reset()
                self.sampler.prefill_context(
                    latents, moved, ranges=ranges, context_write_noise=noise
                )

        self._prefill = self.thread.submit(prefill)

    def prefilled(self) -> None:
        """Wait for the newest prefill; raise its error.

        Raises:
            RuntimeError: on the replica's own thread with the prefill still to come (behind the
                caller's job: a denoising is submitted to the replica after its block's prefill).
        """
        if self._prefill is None:
            return
        if threading.get_ident() == self._ident and not self._prefill.done():
            raise RuntimeError(
                f"a job on the replica of {self.device} waits for the prefill queued behind it"
            )
        self._prefill.result()
