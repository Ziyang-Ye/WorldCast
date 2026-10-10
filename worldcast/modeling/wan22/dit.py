"""The causal DiT block of the generator and what it is built from.

Wan2.2's DiT block (self-attention, text cross-attention, FFN) with per-frame AdaLN of the timestep
and of the controls (App. "Injection"), 3D RoPE, and the :class:`KVCache` that makes its
self-attention block-causal: a call's keys and values join the cache and its queries attend to the
whole cache. Tokens are frame-major: ``[B, F S, dim]`` for ``F`` latent frames of ``S = h w``
tokens.
"""

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from worldcast.utils.precision import fp32_island

from .attention import AttentionFn

__all__ = [
    "DIT_BLOCK_ADALN_VECTORS",
    "DIT_BLOCK_PARTS",
    "HEAD_ADALN_VECTORS",
    "ROPE_POSITIONS",
    "Attention",
    "CausalDiTBlock",
    "CausalHead",
    "DiTBlockParts",
    "KVCache",
    "LayerNorm",
    "RMSNorm",
    "adaln_adapter",
    "apply_rope",
    "causal_self_attention",
    "rope_frequencies",
    "rope_params",
    "rope_table",
    "sinusoidal_embedding_1d",
]


#: Positions of the rotary table along each axis (Wan2.2's): a window has at most this many latent
#: frames.
ROPE_POSITIONS = 1024
#: AdaLN vectors of a DiT block (shift, scale and gate of its self-attention branch and of its FFN
#: branch) and of the head (shift, scale).
DIT_BLOCK_ADALN_VECTORS = 6
HEAD_ADALN_VECTORS = 2


# ============================================================================= positional encodings
def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """``[N] -> [N, dim]`` float64 ``[cos | sin]`` timestep embedding."""
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len: int, dim: int) -> torch.Tensor:
    """``[max_seq_len, dim // 2]`` complex128 rotary table of one axis, built on the CPU."""
    freqs = torch.outer(
        torch.arange(max_seq_len, device="cpu"),
        1.0 / torch.pow(10000, torch.arange(0, dim, 2, device="cpu").to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _rope_axes(factors: int) -> list[int]:
    """How the ``head_dim // 2`` rotary factors of a token split over (frame, height, width)."""
    return [factors - 2 * (factors // 3), factors // 3, factors // 3]


def rope_frequencies(head_dim: int) -> torch.Tensor:
    """``[ROPE_POSITIONS, head_dim // 2]`` complex128: the rotary tables of the three axes
    (frame | height | width), which :func:`rope_table` reads."""
    axes = _rope_axes(head_dim // 2)
    return torch.cat([rope_params(ROPE_POSITIONS, 2 * factors) for factors in axes], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_table(
    freqs: torch.Tensor, grid: tuple[int, int, int], frame_offset: int = 0
) -> torch.Tensor:
    """The 3D RoPE factors (temporal | height | width) of every token of a call.

    Args:
        freqs (Tensor): ``[ROPE_POSITIONS, head_dim // 2]`` complex128 table
            (:func:`rope_frequencies`).
        grid (tuple[int, int, int]): token grid ``(F, h, w)`` of the call.
        frame_offset (int): window index of the call's first latent frame.

    Returns:
        Tensor: ``[F h w, 1, head_dim // 2]`` complex128.
    """
    # Attribution: Wan2.2's rope_apply (github.com/Wan-Video/Wan2.2, Apache-2.0, (c) 2024-2025 The
    # Alibaba Wan Team Authors); the temporal offset (freqs[0][offset:offset + frames])
    # follows CausVid (github.com/tianweiy/CausVid at fab2440f, MIT) via Self Forcing
    # (github.com/guandeh17/Self-Forcing, Apache-2.0).
    frames, height, width = grid
    temporal, vertical, horizontal = freqs.split(_rope_axes(freqs.shape[1]), dim=1)
    return torch.cat(
        [
            temporal[frame_offset : frame_offset + frames]
            .view(frames, 1, 1, -1)
            .expand(frames, height, width, -1),
            vertical[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            horizontal[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, 1, -1)


@torch.amp.autocast("cuda", enabled=False)
def apply_rope(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, L, heads, head_dim]`` by a :func:`rope_table` ``[L, 1, head_dim // 2]``.

    The product is taken in float64; the result is float32.
    """
    rotated = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(rotated * table).flatten(3).float()


# ============================================================================================ norms
class RMSNorm(nn.Module):
    """RMS norm computed in fp32, cast back to the input dtype, then scaled.

    Args:
        dim (int): width of the normed axis.
        eps (float): its epsilon.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., dim]`` -> the same shape and dtype."""
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32, cast back to the input dtype.

    Args:
        dim (int): width of the normed axis.
        eps (float): its epsilon.
        elementwise_affine (bool): learn a scale and a bias (the cross-attention's norm only).
    """

    def __init__(self, dim: int, eps: float, elementwise_affine: bool = False) -> None:
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., dim]`` -> the same shape and dtype."""
        return super().forward(x.float()).type_as(x)


# ======================================================================================== attention
class KVCache:
    """The self-attention keys and values of every DiT block, ``[B, capacity, heads, head_dim]``.

    Every DiT block holds the same token range ``[0, end)``. A call writes its keys and values
    at ``[start, start + n)`` and attends to ``[0, start + n)``, after which the generator moves
    ``end`` there; a write that reaches past ``end`` starts at ``end``, one inside rewrites in place
    (the denoising steps on their target frames). Nothing is evicted: the cache is allocated for
    every latent frame a rollout writes
    (:meth:`~worldcast.modeling.wan22.model.WorldCastGenerator.allocate_kv_cache`). The buffers
    have the dtype of the keys and values written; another dtype would round them.

    A plain class, neither a dataclass nor a container, so that input casts (FSDP's root cast,
    :func:`~worldcast.utils.precision.cast_floating_tensors`) hand on the caller's own cache.

    Args:
        keys (list[Tensor]): one ``[B, capacity, heads, head_dim]`` buffer per DiT block.
        values (list[Tensor]): likewise.
    """

    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor]) -> None:
        self.keys = keys
        self.values = values
        self.end = 0

    @classmethod
    def allocate(
        cls,
        *,
        num_dit_blocks: int,
        num_heads: int,
        head_dim: int,
        latent_frames: int,
        frame_tokens: int,
        dtype: torch.dtype,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> "KVCache":
        """Zeroed buffers for ``latent_frames`` latent frames of ``frame_tokens`` tokens each.

        Args:
            num_dit_blocks (int): DiT blocks of the generator.
            num_heads (int): attention heads.
            head_dim (int): width of a head.
            latent_frames (int): latent frames the cache holds.
            frame_tokens (int): tokens per latent frame.
            dtype (torch.dtype): dtype of the keys and values the generator writes.
            batch_size (int): rollouts in a batch.
            device (torch.device | str | None): where the buffers live.
        """
        shape = (batch_size, latent_frames * frame_tokens, num_heads, head_dim)
        return cls(
            [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_dit_blocks)],
            [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_dit_blocks)],
        )

    @property
    def capacity(self) -> int:
        return self.keys[0].shape[1]

    def reset(self) -> None:
        """Empty the cache; the buffers are kept."""
        self.end = 0

    def span(self, start: int, num_tokens: int) -> int:
        """The end of a write of ``num_tokens`` tokens at ``start``, checked against the cache."""
        end = start + num_tokens
        if start < 0:
            raise ValueError(f"a KV cache write starts at a token of the cache, not at {start}")
        if end > self.end and start != self.end:
            raise ValueError(
                f"a KV cache write past the end {self.end} must start there, not {start}"
            )
        if end > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} tokens; the write ends at {end}")
        return end

    def update(
        self, dit_block: int, key: torch.Tensor, value: torch.Tensor, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write a DiT block's keys and values at ``[start, end)``; return those of ``[0,
        end)``.

        Args:
            dit_block (int): index of the DiT block.
            key (Tensor): ``[B, end - start, heads, head_dim]`` keys of the call.
            value (Tensor): ``[B, end - start, heads, head_dim]`` values of the call.
            start (int): token index of the write.
            end (int): its end (:meth:`span`).
        """
        keys, values = self.keys[dit_block], self.values[dit_block]
        shape = (keys.shape[0], end - start, *keys.shape[2:])
        if tuple(key.shape) != shape or value.shape != key.shape:
            raise ValueError(
                f"the KV cache takes keys and values {list(shape)} ({keys.shape[0]} rollouts) at"
                f" [{start}, {end}), got {tuple(key.shape)} and {tuple(value.shape)}"
            )
        keys[:, start:end] = key
        values[:, start:end] = value
        return keys[:, :end], values[:, :end]


def causal_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rope: torch.Tensor,
    cache: KVCache,
    dit_block: int,
    start: int,
    end: int,
    attention: AttentionFn,
) -> torch.Tensor:
    """Block-causal self-attention of one call: the call's keys and values join the cache, its
    queries attend to the whole cache, so a range sees itself and the ranges written before it.

    Args:
        q (Tensor): ``[B, L, heads, head_dim]`` queries of the call's ``L`` tokens.
        k (Tensor): ``[B, L, heads, head_dim]`` keys.
        v (Tensor): ``[B, L, heads, head_dim]`` values.
        rope (Tensor): the call's :func:`rope_table`.
        cache (KVCache): written at ``[start, end)``.
        dit_block (int): index of the DiT block.
        start (int): token index of the write.
        end (int): ``start + L``, from :meth:`KVCache.span`.
        attention (AttentionFn): the kernel.

    Returns:
        Tensor: ``[B, L, heads, head_dim]``.
    """
    q = apply_rope(q, rope).type_as(v)
    k = apply_rope(k, rope).type_as(v)
    return attention(q, *cache.update(dit_block, k, v, start, end))


class Attention(nn.Module):
    """The ``q``, ``k``, ``v`` and ``o`` projections of one attention of a DiT block, with
    RMS-normed queries and keys.

    Args:
        dim (int): model width.
        num_heads (int): attention heads; ``dim // num_heads`` is the width of a head.
        eps (float): epsilon of the RMS norms.
    """

    def __init__(self, dim: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)

    def query(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, L, dim]`` -> ``[B, L, heads, head_dim]``."""
        return self._heads(self.norm_q(self.q(x)))

    def key_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, L, dim]`` -> keys and values ``[B, L, heads, head_dim]``."""
        return self._heads(self.norm_k(self.k(x))), self._heads(self.v(x))


# =========================================================================================== blocks
def adaln_adapter(dim: int, multiplier: int, rank: int) -> nn.Sequential:
    """The low-rank adapter of the control embedding on AdaLN vectors, zero-initialised on its
    output: ``SiLU -> Linear(dim, rank, no bias) -> SiLU -> Linear(rank, multiplier * dim)``.

    Args:
        dim (int): model width.
        multiplier (int): AdaLN vectors it adapts (:data:`DIT_BLOCK_ADALN_VECTORS`,
            :data:`HEAD_ADALN_VECTORS`).
        rank (int): its rank.
    """
    module = nn.Sequential(
        nn.SiLU(),
        nn.Linear(dim, rank, bias=False),
        nn.SiLU(),
        nn.Linear(rank, multiplier * dim, bias=True),
    )
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


def _by_frame(tokens: torch.Tensor, num_frames: int) -> torch.Tensor:
    """``[B, F S, D]`` -> ``[B, F, S, D]`` (a view): a frame's vector broadcasts over its tokens."""
    return tokens.unflatten(1, (num_frames, -1))


def _frame_modulation(
    table: torch.Tensor, timestep_modulation: torch.Tensor, control: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """``(timestep + table) + control``, split into the table's ``n`` vectors ``[B, F, 1, dim]``."""
    count, dim = table.shape[-2:]
    summed = (timestep_modulation + table) + control.unflatten(-1, (count, dim))
    return summed.chunk(count, dim=2)


#: ``(pre_attention, post_self_attention, post_cross_attention)`` of a DiT block, each called with
#: the DiT block first (:data:`DIT_BLOCK_PARTS`).
DiTBlockParts = tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]


class CausalDiTBlock(nn.Module):
    """One DiT block: self-attention, text cross-attention and FFN, with per-frame AdaLN of the
    timestep and the controls on the self-attention and FFN branches.

    The DiT block is three parts around its two attention kernels: :meth:`pre_attention`,
    :meth:`post_self_attention` and :meth:`post_cross_attention`. :meth:`forward` runs them with
    the caller's attention kernels (the client's fast path may compile them).
    :meth:`attach_control_adaln` must be called before :meth:`forward`.

    Args:
        dim (int): model width.
        ffn_dim (int): hidden width of the FFN.
        num_heads (int): attention heads.
        eps (float): epsilon of the norms (``GeneratorConfig.eps``).
    """

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = Attention(dim, num_heads, eps)
        self.norm3 = LayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = Attention(dim, num_heads, eps)
        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, DIT_BLOCK_ADALN_VECTORS, dim) / dim**0.5)

    def attach_control_adaln(self, rank: int) -> None:
        """Build ``control_adaln``, the zero-initialised adapter of the control embedding on the
        six AdaLN vectors. A method, not part of the constructor: the generator calls it after the
        backbone's initialisation, so that a fresh generator draws its weights as trained."""
        count, dim = self.modulation.shape[-2:]
        self.control_adaln = adaln_adapter(dim, count, rank)

    def pre_attention(
        self, x: torch.Tensor, timestep_modulation: torch.Tensor, control_embedding: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
        """The DiT block's AdaLN vectors and the self-attention's queries, keys and values.

        Args:
            x (Tensor): ``[B, F S, dim]`` tokens.
            timestep_modulation (Tensor): ``[B, F, 6, dim]`` float32.
            control_embedding (Tensor): ``[B, F, dim]`` control embedding.

        Returns:
            tuple: ``(q, k, v)``, each ``[B, F S, heads, head_dim]``, and the six AdaLN vectors
            ``[B, F, 1, dim]`` float32.
        """
        with fp32_island():
            modulation = _frame_modulation(
                self.modulation, timestep_modulation, self.control_adaln(control_embedding)
            )
        shift, scale = modulation[0], modulation[1]
        normed = _by_frame(self.norm1(x).float(), timestep_modulation.shape[1])
        h = (normed * (1 + scale) + shift).flatten(1, 2)
        return (self.self_attn.query(h), *self.self_attn.key_value(h)), modulation

    def post_self_attention(
        self, x: torch.Tensor, attended: torch.Tensor, modulation: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The gated self-attention residual, and the cross-attention's queries.

        Args:
            x (Tensor): ``[B, F S, dim]`` the DiT block's input tokens.
            attended (Tensor): ``[B, F S, heads, head_dim]`` self-attention output.
            modulation (tuple[Tensor, ...]): the AdaLN vectors of :meth:`pre_attention`.

        Returns:
            tuple[Tensor, Tensor]: tokens ``[B, F S, dim]`` and queries
            ``[B, F S, heads, head_dim]``.
        """
        gate = modulation[2]
        branch = _by_frame(self.self_attn.o(attended.flatten(2)), gate.shape[1])
        with fp32_island():
            x = x + (branch * gate).flatten(1, 2)
        return x, self.cross_attn.query(self.norm3(x))

    def post_cross_attention(
        self, x: torch.Tensor, attended: torch.Tensor, modulation: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """The cross-attention residual and the gated FFN residual.

        Args:
            x (Tensor): ``[B, F S, dim]`` tokens after :meth:`post_self_attention`.
            attended (Tensor): ``[B, F S, heads, head_dim]`` cross-attention output.
            modulation (tuple[Tensor, ...]): the AdaLN vectors of :meth:`pre_attention`.

        Returns:
            Tensor: ``[B, F S, dim]``.
        """
        x = x + self.cross_attn.o(attended.flatten(2))
        shift, scale, gate = modulation[3:]
        normed = _by_frame(self.norm2(x).float(), gate.shape[1])
        branch = _by_frame(self.ffn((normed * (1 + scale) + shift).flatten(1, 2)), gate.shape[1])
        with fp32_island():
            return x + (branch * gate).flatten(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        timestep_modulation: torch.Tensor,
        control_embedding: torch.Tensor,
        text: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        self_attention: AttentionFn,
        attention: AttentionFn,
        parts: DiTBlockParts | None = None,
    ) -> torch.Tensor:
        """One DiT block; every path enters here, so an FSDP-wrapped DiT block unshards.

        Args:
            x (Tensor): ``[B, F S, dim]`` tokens (``[context | noisy]`` under teacher forcing).
            timestep_modulation (Tensor): ``[B, F, 6, dim]`` float32 (of both copies).
            control_embedding (Tensor): ``[B, F, dim]`` control embedding (of both copies).
            text (Tensor | tuple[Tensor, Tensor]): ``[B, 512, dim]`` the embedded prompt, or the
                cross-attention's keys and values of it.
            self_attention (AttentionFn): ``fn(q, k, v)``, RoPE and the cache or the training
                mask included (:func:`causal_self_attention`).
            attention (AttentionFn): the kernel of the cross-attention.
            parts (DiTBlockParts | None): the three parts (default :data:`DIT_BLOCK_PARTS`).

        Returns:
            Tensor: ``[B, F S, dim]`` float32.
        """
        pre_attention, post_self_attention, post_cross_attention = parts or DIT_BLOCK_PARTS
        qkv, modulation = pre_attention(self, x, timestep_modulation, control_embedding)
        x, query = post_self_attention(self, x, self_attention(*qkv), modulation)
        key_value = text if isinstance(text, tuple) else self.cross_attn.key_value(text)
        return post_cross_attention(self, x, attention(query, *key_value), modulation)


#: The parts of :class:`CausalDiTBlock`.
DIT_BLOCK_PARTS: DiTBlockParts = (
    CausalDiTBlock.pre_attention,
    CausalDiTBlock.post_self_attention,
    CausalDiTBlock.post_cross_attention,
)


class CausalHead(nn.Module):
    """Output head: per-frame AdaLN of the timestep and the controls, then a linear map to the
    patches, in fp32. :meth:`attach_control_adaln` must be called before :meth:`forward`.

    Args:
        dim (int): model width.
        out_dim (int): latent channels.
        patch_size (tuple[int, int, int]): the patch of a token, ``(frames, height, width)``.
        eps (float): epsilon of the norm.
    """

    def __init__(
        self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, HEAD_ADALN_VECTORS, dim) / dim**0.5)

    def attach_control_adaln(self, rank: int) -> None:
        """Build ``control_adaln`` for the head's two AdaLN vectors
        (:meth:`CausalDiTBlock.attach_control_adaln`)."""
        count, dim = self.modulation.shape[-2:]
        self.control_adaln = adaln_adapter(dim, count, rank)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor, control_embedding: torch.Tensor
    ) -> torch.Tensor:
        """The patches of every token.

        Args:
            x (Tensor): ``[B, F S, dim]`` tokens after the last DiT block.
            timestep_embedding (Tensor): ``[B, F, dim]`` float32.
            control_embedding (Tensor): ``[B, F, dim]`` control embedding.

        Returns:
            Tensor: ``[B, F S, prod(patch_size) * out_dim]``.
        """
        num_frames = timestep_embedding.shape[1]
        with fp32_island():
            shift, scale = _frame_modulation(
                self.modulation,
                timestep_embedding.unsqueeze(2),
                self.control_adaln(control_embedding),
            )
            normed = _by_frame(self.norm(x), num_frames)
            return self.head(normed * (1 + scale) + shift).flatten(1, 2)
