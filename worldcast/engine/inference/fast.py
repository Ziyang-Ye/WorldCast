"""The generator call of the client: the generator's own pieces, without per-call costs.

:class:`FastGenerator` is a :class:`~worldcast.modeling.wan22.model.CausalGeneratorAdapter` whose
call runs the pieces of :meth:`WorldCastGenerator.forward` (``CallPlan.at``, ``prologue``,
``embed_timesteps``, ``run_dit_blocks_on_cache``) with the same kernels, and adds only:

* the prologue the denoising steps share (observer signals, ray embedding, controls, player state
  field), computed on the first denoising step of a block and reused by the others;
* the text embedding and every DiT block's cross-attention keys and values, once per prompt;
* the RoPE table of every call shape, kept across calls;
* on CUDA, the bf16 patch embedding as one GEMM (:func:`patchify_linear`), bit-equal to the conv;
* optionally, CUDA graphs of the DiT blocks, the field injection and the head of every call shape
  that recurs (the five context writes and the target of a block generated from its window);
* the choice of the attention kernel; only flash-attention 2 is exact, FlashAttention-3 and
  ``compile`` trade bits for speed.
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from worldcast.modeling.wan22.attention import AttentionFn
from worldcast.modeling.wan22.dit import DIT_BLOCK_PARTS, KVCache
from worldcast.modeling.wan22.model import (
    CallPlan,
    CallPrologue,
    CausalGeneratorAdapter,
    FieldBuilder,
    GeneratorConditions,
    WorldCastGenerator,
    call_inputs,
)
from worldcast.utils.precision import generator_autocast

__all__ = ["FastGenerator", "patchify_linear"]


def patchify_linear(conv: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    """``conv(x)`` for a Conv3d whose stride is its kernel, unpadded, as ``addmm`` per sample.

    A bf16 ``Conv3d`` on CUDA runs ``slow_conv_dilated3d``: a bias fill per output channel, an
    im2col, then one column-major GEMM ``output += columns^T weight^T`` with beta 1. This builds the
    same columns by a reshape and calls ``addmm(bias, weight, columns)``, which lowers to that GEMM:
    bit-equal on CUDA (the client reproduces the reference fingerprints through it,
    ``tools/verify_reference.py``), equal up to rounding on the CPU.

    Args:
        conv (nn.Conv3d): the patch embedding, kernel ``(kt, kh, kw)`` = stride, no padding.
        x (Tensor): ``[B, C, T, H, W]``, each of ``T, H, W`` a multiple of the kernel's.

    Returns:
        Tensor: ``[B, out_channels, T / kt, H / kh, W / kw]``.
    """
    kt, kh, kw = conv.kernel_size
    batch, channels, t, h, w = x.shape
    to, ho, wo = t // kt, h // kh, w // kw
    weight = conv.weight.contiguous().view(conv.out_channels, channels * kt * kh * kw)
    columns = (
        x.reshape(batch, channels, to, kt, ho, kh, wo, kw)
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .reshape(batch, channels * kt * kh * kw, to * ho * wo)
    )
    bias = conv.bias.view(-1, 1).expand(conv.out_channels, to * ho * wo).contiguous()
    return torch.stack(
        [
            torch.addmm(bias, weight, columns[n].contiguous()).view(-1, to, ho, wo)
            for n in range(batch)
        ]
    )


def _tensor_key(tensor: torch.Tensor) -> tuple:
    """Identity of a tensor (storage, view, version): while the tensor lives, no other takes its
    storage, so an equal key is the same values (an in-place write changes the version)."""
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor._version)


def _tensors_key(conditions: Mapping[str, Any]) -> tuple:
    """Identity of the tensors of a condition mapping."""
    return tuple(
        (name, *_tensor_key(v)) if torch.is_tensor(v) else (name, type(v).__name__, id(v))
        for name, v in sorted(conditions.items())
    )


@dataclass
class _Graph:
    graph: Any
    inputs: dict[str, torch.Tensor]
    output: torch.Tensor


class FastGenerator(CausalGeneratorAdapter):
    """The sampler's generator for one client (batch 1), bit-equal to the adapter it extends.

    It keeps what it computed for the tensors it was called with (the prompt, the conditions) and
    holds them, so the same tensors are never computed twice and never mistaken for others. The
    generator is frozen in place (``requires_grad_(False)``). With CUDA graphs it serves one KV
    cache: a graph replays on the buffers it was captured on.

    Args:
        generator (WorldCastGenerator): the loaded generator (bf16 on the GPU, or fp32 on the CPU).
        field_builder (FieldBuilder): builds the player state field.
        input_dtype (torch.dtype | None): the input cast (bf16, ``GENERATOR_DTYPE``, on the paper
            path; ``None`` for fp32 CPU runs).
        attention (AttentionFn | None): the kernel; ``None`` keeps the generator's.
        cuda_graphs (bool): capture recurring call shapes (CUDA only).
        compile (bool): ``torch.compile`` the three parts of each DiT block (not exact).
    """

    def __init__(
        self,
        generator: WorldCastGenerator,
        field_builder: FieldBuilder,
        *,
        input_dtype: torch.dtype | None = None,
        attention: AttentionFn | None = None,
        cuda_graphs: bool = False,
        compile: bool = False,
    ) -> None:
        super().__init__(generator.requires_grad_(False), field_builder, input_dtype=input_dtype)
        cuda = generator.patch_embedding.weight.is_cuda
        if cuda_graphs and not cuda:
            raise ValueError("cuda_graphs needs a CUDA generator")
        self.attention = attention or generator.attention
        self.cuda_graphs = bool(cuda_graphs)
        self._patch_embed = (
            functools.partial(patchify_linear, generator.patch_embedding) if cuda else None
        )
        self._parts = (
            tuple(torch.compile(part, dynamic=False) for part in DIT_BLOCK_PARTS)
            if compile
            else None
        )
        # the kept work, each with the tensors it was computed from (held, so that their key
        # stays theirs): the prompt's cross-attention keys and values, and one call prologue
        self._prompt: tuple[tuple, torch.Tensor] | None = None
        self._cross_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._rope: dict[tuple[int, ...], torch.Tensor] = {}
        self._prologue: dict[tuple, tuple[CallPrologue, list[torch.Tensor]]] = {}
        self._graphs: dict[tuple, _Graph] = {}
        self._graphed_cache: KVCache | None = None
        self._seen: set[tuple] = set()
        self._pool = None

    @torch.no_grad()
    def __call__(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        kv_cache: KVCache,
        frame_offset: int,
    ) -> torch.Tensor:
        """Flow ``[1, F, C, H, W]`` float32 of one call (:meth:`WorldCastGenerator.forward`)."""
        conditions_key = _tensors_key(conditions)  # the caller's tensors, before the cast
        refs = [v for v in conditions.values() if torch.is_tensor(v)]
        prompt = conditions["prompt_embeds"]
        g = self.generator
        cond, noisy, timestep = call_inputs(
            g, self.field_builder, self.input_dtype, conditions, noisy, timestep
        )
        with generator_autocast(noisy):
            tokens, grid = g.patchify(noisy, embed=self._patch_embed)
            if tokens.shape[0] != 1:
                raise ValueError("the fast path serves one client: batch 1")
            plan = CallPlan.at(grid, frame_offset, kv_cache)
            timestep_embedding, timestep_modulation = g.embed_timesteps(timestep, grid[0])
            prologue = self._prologue_of(conditions_key, refs, cond, plan, tokens.dtype)
            self._prepare_text(prompt, cond.prompt_embeds)
            inputs = dict(
                tokens=prologue.embed(tokens),
                timestep_embedding=timestep_embedding,
                timestep_modulation=timestep_modulation,
                control_embedding=prologue.control_embedding,
                field=prologue.field,
            )
            inputs = {name: value for name, value in inputs.items() if value is not None}
            if self.cuda_graphs:
                out = self._graphed(plan, inputs, kv_cache)
            else:
                out = self._run_blocks(kv_cache, plan, **inputs)
            kv_cache.end = plan.end
            return g.unpatchify(out, grid)

    def _run_blocks(
        self,
        cache: KVCache,
        plan: CallPlan,
        *,
        tokens: torch.Tensor,
        timestep_embedding: torch.Tensor,
        timestep_modulation: torch.Tensor,
        control_embedding: torch.Tensor,
        field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """:meth:`WorldCastGenerator.run_dit_blocks_on_cache` on the kept text, kernel and
        DiT-block parts: what a CUDA graph captures."""
        return self.generator.run_dit_blocks_on_cache(
            tokens,
            timestep_embedding=timestep_embedding,
            timestep_modulation=timestep_modulation,
            control_embedding=control_embedding,
            field=field,
            plan=plan,
            kv_cache=cache,
            rope=self._rope_table(plan),
            text=self._cross_kv,
            attention=self.attention,
            parts=self._parts,
        )

    # ---------------------------------------------------------------------------------- reused work
    def _prologue_of(
        self,
        conditions_key: tuple,
        refs: list[torch.Tensor],
        cond: GeneratorConditions,
        plan: CallPlan,
        dtype: torch.dtype,
    ) -> CallPrologue:
        key = (conditions_key, plan.frame_offset, plan.grid, dtype)
        if key not in self._prologue:
            prologue = self.generator.prologue(cond, plan.grid, plan.frame_offset, dtype)
            self._prologue = {key: (prologue, refs)}
        return self._prologue[key][0]

    def _prepare_text(self, prompt: torch.Tensor, prompt_embeds: torch.Tensor) -> None:
        """The text embedding and every DiT block's cross-attention keys and values, once per
        prompt.

        Args:
            prompt (Tensor): the caller's prompt tensor, which identifies the prompt.
            prompt_embeds (Tensor): the same after the input cast, which the generator embeds.
        """
        key = _tensor_key(prompt)
        if self._prompt is not None and self._prompt[0] == key:
            return
        text = self.generator.embed_text(prompt_embeds)
        cross_kv = [block.cross_attn.key_value(text) for block in self.generator.blocks]
        if self._graphs:  # captured graphs read these buffers: refill them in place
            for (k_old, v_old), (k, v) in zip(self._cross_kv, cross_kv):
                k_old.copy_(k)
                v_old.copy_(v)
        else:
            self._cross_kv = cross_kv
        self._prompt = (key, prompt)

    def _rope_table(self, plan: CallPlan) -> torch.Tensor:
        key = (plan.frame_offset, *plan.grid)
        if key not in self._rope:
            self._rope[key] = self.generator.rope_table(plan.grid, plan.frame_offset)
        return self._rope[key]

    # ---------------------------------------------------------------------------------- CUDA graphs
    def _graphed(
        self, plan: CallPlan, inputs: dict[str, torch.Tensor], cache: KVCache
    ) -> torch.Tensor:
        """Replay the call shape's graph; capture it on its second occurrence (it may never
        recur)."""
        key = (plan,) + tuple(
            (name, tuple(value.shape), value.dtype) for name, value in inputs.items()
        )
        if self._graphed_cache is not None and cache is not self._graphed_cache:
            raise ValueError("a generator with CUDA graphs serves the KV cache they captured")
        graph = self._graphs.get(key)
        if graph is None:
            if key not in self._seen:
                self._seen.add(key)
                return self._run_blocks(cache, plan, **inputs)
            graph = self._capture(key, plan, inputs, cache)
        for name, value in inputs.items():
            graph.inputs[name].copy_(value)
        graph.graph.replay()
        return graph.output

    def _capture(
        self, key: tuple, plan: CallPlan, inputs: dict[str, torch.Tensor], cache: KVCache
    ) -> _Graph:
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        buffers = {name: value.clone() for name, value in inputs.items()}
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
            output = self._run_blocks(cache, plan, **buffers)
        self._graphs[key] = _Graph(graph=graph, inputs=buffers, output=output)
        self._graphed_cache = cache
        return self._graphs[key]
