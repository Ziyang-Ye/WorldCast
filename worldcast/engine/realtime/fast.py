"""The generator call of the real-time engine: the generator's own pieces, without per-call costs.

:class:`FastGenerator` is a :class:`~worldcast.modeling.wan22.model.CausalGeneratorAdapter` whose
call runs the same pieces as :meth:`WorldCastGenerator.forward` (patch embedding, embeddings, the
three stages of every DiT block, field injection, head) with the same kernels, and adds only:

* the ladder-invariant prologue (observer signals, controls, ray code, player state field),
  computed on the first rung of a block and reused by the others;
* the text embedding and every block's cross-attention keys and values, once per prompt;
* the RoPE table of every call shape, kept across calls;
* on CUDA, the bf16 patch embedding as one GEMM (:func:`patchify_linear`), bit-equal to the conv;
* optionally, CUDA graphs of the blocks, the field injection and the head of every call shape that
  recurs (the five context writes and the target of a reconstituted block);
* the choice of the attention kernel; only ``flash`` is exact, ``cudnn``, ``fa3`` and ``compile``
  trade bits for speed (docs/latency.md).
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from worldcast.modeling.wan22.attention import (
    AttentionFn,
    cudnn_attention,
    fa3_attention,
    fa3_module,
    flash_attention,
)
from worldcast.modeling.wan22.model import (
    CausalDiTBlock,
    CausalGeneratorAdapter,
    FieldBuilder,
    GeneratorConditions,
    KVCache,
    WorldCastGenerator,
    causal_self_attention,
)
from worldcast.utils.precision import generator_autocast

__all__ = ["ATTENTION_BACKENDS", "FastGenerator", "attention_backend", "patchify_linear"]

#: Attention kernels of the fast path by name; only ``flash`` gives the paper's bits.
ATTENTION_BACKENDS: dict[str, AttentionFn] = {
    "flash": flash_attention,
    "cudnn": cudnn_attention,
    "fa3": fa3_attention,
}


def attention_backend(name: str) -> AttentionFn:
    """The kernel of :data:`ATTENTION_BACKENDS` called ``name``."""
    if name not in ATTENTION_BACKENDS:
        raise ValueError(f"unknown attention backend {name!r} ({' | '.join(ATTENTION_BACKENDS)})")
    if name == "fa3" and fa3_module() is None:
        raise RuntimeError("attention 'fa3' needs FlashAttention-3 (attention.FA3_MODULES)")
    return ATTENTION_BACKENDS[name]


def patchify_linear(conv: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    """``conv(x)`` for a Conv3d whose stride is its kernel, unpadded, as ``addmm`` per sample.

    A bf16 ``Conv3d`` on CUDA runs ``slow_conv_dilated3d``: a bias fill per output channel, an
    im2col, then one column-major GEMM ``output += columns^T weight^T`` with beta 1. This builds the
    same columns by a reshape and calls ``addmm(bias, weight, columns)``, which lowers to that GEMM:
    bit-equal on CUDA (checked by ``tools/bench_realtime.py``), equal up to rounding on the CPU.
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


def _tensors_key(conditions: Mapping[str, Any]) -> tuple:
    """Identity of the tensors of a condition mapping (storage, view, version)."""
    return tuple(
        (
            (name, v.data_ptr(), tuple(v.shape), v.stride(), v.dtype, v._version)
            if torch.is_tensor(v)
            else (name, type(v).__name__, id(v))
        )
        for name, v in sorted(conditions.items())
    )


@dataclass
class _Prologue:
    """What a block's rungs share: per-token observer signals and ray code, controls, field."""

    signals: torch.Tensor
    ray_code: torch.Tensor | None
    controls: torch.Tensor
    field: torch.Tensor
    #: the keyed tensors: while they live no other tensor takes their storage, so an equal key is
    #: the same tensors (an in-place write changes their version)
    refs: list[torch.Tensor]


@dataclass(frozen=True)
class _CallPlan:
    """Where one call sits in the window and the cache."""

    frame_offset: int
    frames: int
    start: int
    end: int
    rope: torch.Tensor


@dataclass
class _Graph:
    graph: Any
    inputs: dict[str, torch.Tensor]
    output: torch.Tensor


class FastGenerator(CausalGeneratorAdapter):
    """The sampler's generator for one client (batch 1), bit-equal to the adapter it extends.

    Args:
        generator (WorldCastGenerator): the loaded generator (bf16 on the GPU, or fp32 on the CPU).
        field_builder (FieldBuilder): builds the player state field.
        input_dtype (torch.dtype | None): the input cast (bf16 on the paper path; ``None`` for fp32
            CPU runs).
        attention (str | None): a name of :data:`ATTENTION_BACKENDS`; ``None`` keeps the
            generator's kernel.
        cuda_graphs (bool): capture recurring call shapes (CUDA only).
        patch_linear (bool | None): the patch embedding as one GEMM (default: on CUDA).
        compile (bool): ``torch.compile`` the three stages of each DiT block (not exact).
    """

    def __init__(
        self,
        generator: WorldCastGenerator,
        field_builder: FieldBuilder,
        *,
        input_dtype: torch.dtype | None = torch.bfloat16,
        attention: str | None = None,
        cuda_graphs: bool = False,
        patch_linear: bool | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__(generator.requires_grad_(False), field_builder, input_dtype=input_dtype)
        cuda = generator.patch_embedding.weight.is_cuda
        if cuda_graphs and not cuda:
            raise ValueError("cuda_graphs needs a CUDA generator")
        self.attention = generator.attention if attention is None else attention_backend(attention)
        self.cuda_graphs = bool(cuda_graphs)
        use_gemm = cuda if patch_linear is None else patch_linear
        self._patch_embed = (
            functools.partial(patchify_linear, generator.patch_embedding) if use_gemm else None
        )
        stages = (
            CausalDiTBlock.pre_attention,
            CausalDiTBlock.post_self_attention,
            CausalDiTBlock.post_cross_attention,
        )
        self._stages = tuple(torch.compile(s, dynamic=False) for s in stages) if compile else stages
        self._prompt_key: tuple | None = None
        self._cross_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._rope: dict[tuple[int, ...], torch.Tensor] = {}
        self._prologue: dict[tuple, _Prologue] = {}
        self._graphs: dict[tuple, _Graph] = {}
        self._seen: set[tuple] = set()
        self._pool = None
        self.stats = dict(calls=0, eager=0, captured=0, replayed=0, prologue_reused=0)

    @torch.no_grad()
    def __call__(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        kv_cache: KVCache,
        current_start: int,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        """Flow ``[1, F, C, H, W]`` float32 of one call (:meth:`WorldCastGenerator.forward`)."""
        self.stats["calls"] += 1
        conditions_key = _tensors_key(conditions)  # the caller's tensors, before the cast
        refs = [v for v in conditions.values() if torch.is_tensor(v)]
        noisy, timestep, cond = self.inputs(noisy, timestep, conditions)
        g = self.generator
        with generator_autocast(noisy):
            tokens, grid = g.patchify(noisy.permute(0, 2, 1, 3, 4), embed=self._patch_embed)
            if tokens.shape[0] != 1:
                raise ValueError("the fast path serves one client: batch 1")
            frames, height, width = grid
            frame_seq_length = height * width
            cache_start = current_start if cache_start is None else cache_start
            if current_start % frame_seq_length or cache_start % frame_seq_length:
                raise ValueError("current_start and cache_start must be latent-frame boundaries")
            frame_offset = current_start // frame_seq_length
            plan = _CallPlan(
                frame_offset=frame_offset,
                frames=frames,
                start=cache_start,
                end=kv_cache.span(cache_start, tokens.shape[1]),
                rope=self._rope_table(grid, frame_offset),
            )
            t_embedding, modulation = g.embed_timesteps(timestep)
            prologue = self._prologue_of(
                conditions_key, refs, cond, grid, frame_offset, tokens.dtype
            )
            tokens = tokens + prologue.signals
            if prologue.ray_code is not None:
                tokens = tokens + prologue.ray_code
            self._prepare_context(cond.prompt_embeds)

            inputs = dict(
                tokens=tokens,
                modulation=modulation,
                controls=prologue.controls,
                t_embedding=t_embedding,
                field=prologue.field,
            )
            if self.cuda_graphs:
                out = self._graphed(plan, grid, inputs, kv_cache)
            else:
                self.stats["eager"] += 1
                out = self._trunk(kv_cache, plan, **inputs)
            kv_cache.end = plan.end
            return g.unpatchify(out, grid).float().permute(0, 2, 1, 3, 4)

    def _trunk(
        self,
        cache: KVCache,
        plan: _CallPlan,
        *,
        tokens: torch.Tensor,
        modulation: torch.Tensor,
        controls: torch.Tensor,
        t_embedding: torch.Tensor,
        field: torch.Tensor,
    ) -> torch.Tensor:
        """The DiT blocks (as :meth:`CausalDiTBlock.forward`), the field injection and the head:
        the part a CUDA graph captures."""
        g = self.generator
        pre_attention, post_self_attention, post_cross_attention = self._stages
        for index, block in enumerate(g.blocks):
            qkv, block_modulation = pre_attention(block, tokens, modulation, controls)
            attended = causal_self_attention(
                *qkv,
                rope=plan.rope,
                cache=cache,
                block=index,
                start=plan.start,
                end=plan.end,
                attention=self.attention,
            )
            tokens, query = post_self_attention(block, tokens, attended, block_modulation)
            attended = self.attention(query, *self._cross_kv[index])
            tokens = post_cross_attention(block, tokens, attended, block_modulation)
            if index == g.state_injector.write_block:
                tokens = g.state_injector(
                    tokens, field, frame_offset=plan.frame_offset, num_frames=plan.frames
                )
        return g.head(tokens, t_embedding, controls)

    # ---------------------------------------------------------------------------------- reused work
    def _prologue_of(
        self,
        conditions_key: tuple,
        refs: list[torch.Tensor],
        cond: GeneratorConditions,
        grid: tuple[int, int, int],
        frame_offset: int,
        dtype: torch.dtype,
    ) -> _Prologue:
        key = (conditions_key, frame_offset, grid, dtype)
        if key in self._prologue:
            self.stats["prologue_reused"] += 1
            return self._prologue[key]
        g = self.generator
        frames, height, width = grid
        signals = g.embed_observer_signals(
            cond, frame_offset=frame_offset, num_frames=frames, dtype=dtype
        )
        ray_code = None
        if cond.rays is not None:
            ray_code = g.rays(
                cond.rays,
                frame_offset=frame_offset,
                num_frames=frames,
                grid=(height, width),
                dtype=dtype,
            )
        prologue = _Prologue(
            signals=signals.repeat_interleave(height * width, dim=1),
            ray_code=ray_code,
            controls=g.embed_controls(
                cond, frame_offset=frame_offset, num_frames=frames, dtype=dtype
            ),
            field=g.state_field(cond, frame_offset, frames),
            refs=refs,
        )
        self._prologue = {key: prologue}
        return prologue

    def _prepare_context(self, prompt_embeds: torch.Tensor) -> None:
        """The text embedding and every block's cross-attention keys and values, per prompt."""
        key = tuple((p.data_ptr(), tuple(p.shape), p.dtype, p._version) for p in prompt_embeds)
        if key == self._prompt_key:
            return
        context = self.generator.embed_text(prompt_embeds)
        cross_kv = [block.cross_attn.key_value(context) for block in self.generator.blocks]
        if self._graphs:  # captured graphs read these buffers: refill them in place
            for (k_old, v_old), (k, v) in zip(self._cross_kv, cross_kv):
                k_old.copy_(k)
                v_old.copy_(v)
        else:
            self._cross_kv = cross_kv
        self._prompt_key = key

    def _rope_table(self, grid: tuple[int, int, int], frame_offset: int) -> torch.Tensor:
        key = (frame_offset, *grid)
        if key not in self._rope:
            self._rope[key] = self.generator.rope_table(grid, frame_offset)
        return self._rope[key]

    # ---------------------------------------------------------------------------------- CUDA graphs
    def _graphed(
        self,
        plan: _CallPlan,
        grid: tuple[int, int, int],
        inputs: dict[str, torch.Tensor],
        cache: KVCache,
    ) -> torch.Tensor:
        """Replay the call shape's graph; capture it on its second occurrence (it may never
        recur)."""
        key = (plan.frame_offset, plan.frames, grid, plan.start, plan.end) + tuple(
            (name, tuple(value.shape), value.dtype) for name, value in inputs.items()
        )
        graph = self._graphs.get(key)
        if graph is None:
            if key not in self._seen:
                self._seen.add(key)
                self.stats["eager"] += 1
                return self._trunk(cache, plan, **inputs)
            graph = self._capture(key, plan, inputs, cache)
        for name, value in inputs.items():
            graph.inputs[name].copy_(value)
        graph.graph.replay()
        self.stats["replayed"] += 1
        return graph.output

    def _capture(
        self, key: tuple, plan: _CallPlan, inputs: dict[str, torch.Tensor], cache: KVCache
    ) -> _Graph:
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        buffers = {name: value.clone() for name, value in inputs.items()}
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
            output = self._trunk(cache, plan, **buffers)
        self._graphs[key] = _Graph(graph=graph, inputs=buffers, output=output)
        self.stats["captured"] += 1
        return self._graphs[key]
