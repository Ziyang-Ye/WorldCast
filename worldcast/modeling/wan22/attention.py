"""Attention kernels of the generator and the attention masks of training.

An :data:`AttentionFn` is ``fn(q, k, v)`` on ``[B, L, heads, head_dim]`` with no mask:
:func:`flash_attention` (flash-attention 2) is the kernel of the paper's runs and the only one that
gives their bits, :func:`sdpa_attention` a reference for the CPU, :func:`fa3_attention` a faster
kernel the client can serve with; :data:`ATTENTION_KERNELS` names them for the ``model.attention``
setting. A training mask is bidirectional (stages 1, 2, 2s) or block-causal teacher forcing over
``[context | noisy]`` (stage 3), see :class:`MaskLayout`.
"""

import functools
import importlib
import importlib.util
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from worldcast.data.latents import BLOCK
from worldcast.utils.precision import GENERATOR_DTYPE, HALF_DTYPES

try:  # flex attention (torch >= 2.5): the training kernel on CUDA
    from torch.nn.attention import flex_attention as _flex_module
except ImportError:  # pragma: no cover - torch 2.4: training uses the dense mask
    _flex_module = None

__all__ = [
    "ATTENTION_KERNELS",
    "FA2_MODULE",
    "FA3_MODULES",
    "FLEX_ATTENTION_AVAILABLE",
    "MASK_BLOCK_SIZE",
    "PAPER_ATTENTION",
    "AttentionFn",
    "MaskLayout",
    "TrainingMask",
    "attention_intervals",
    "attention_kernel",
    "causal_blocks",
    "dense_attention_mask",
    "fa3_attention",
    "fa3_available",
    "flash_attention",
    "flash_attention_backend",
    "flex_block_mask",
    "masked_attention",
    "masked_flex_attention",
    "masked_sdpa_attention",
    "sdpa_attention",
    "training_mask",
]

#: ``fn(q, k, v) -> out``: q ``[B, Lq, N, D]``, k and v ``[B, Lk, N, D]``, out ``[B, Lq, N, D]``.
AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

#: A training mask: a dense ``[L, L]`` bool tensor (True = may attend) or a flex ``BlockMask``
#: (``Any``: torch 2.4 has no such class to name).
TrainingMask = torch.Tensor | Any

#: Whether torch has flex attention (torch >= 2.5); without it training uses the dense mask.
FLEX_ATTENTION_AVAILABLE = _flex_module is not None
#: Flex attention's tile; both sides of a flex mask are padded to a multiple of it.
MASK_BLOCK_SIZE = 128

#: flash-attention 2, the paper's kernel (:func:`flash_attention`).
FA2_MODULE = "flash_attn"
#: Where :func:`fa3_attention` finds FlashAttention-3: a source install, or the Hugging Face kernels
#: hub's ``kernels-community/flash-attn3`` unpacked as ``flash_attn3_hub``.
FA3_MODULES = ("flash_attn_interface", "flash_attn3_hub")


# =================================================================================== full attention
@functools.cache
def _first_module(names: tuple[str, ...]) -> Any:
    """The first importable module of ``names``, or ``None``; imported on first use."""
    for name in names:
        if importlib.util.find_spec(name) is not None:
            return importlib.import_module(name)
    return None


def _kernel_module(names: tuple[str, ...]) -> Any:
    """The first importable module of ``names``; raises when none is installed."""
    module = _first_module(names)
    if module is None:
        raise RuntimeError(f"{' or '.join(names)} is not installed; use sdpa_attention")
    return module


def fa3_available() -> bool:
    """Whether FlashAttention-3 (a module of :data:`FA3_MODULES`) is installed."""
    return _first_module(FA3_MODULES) is not None


def flash_attention_backend() -> str:
    """The kernel of :func:`flash_attention`: ``"fa2"``, or ``"none"`` when flash-attn is missing.

    Record it with every reference run: other kernels give different bits.
    """
    return "fa2" if _first_module((FA2_MODULE,)) is not None else "none"


@functools.cache
def _cu_seqlens(batch: int, length: int, device: torch.device) -> torch.Tensor:
    """Sequence offsets ``[0, L, 2 L, ...]`` int32, kept on the device (no kernel per call)."""
    return torch.arange(0, (batch + 1) * length, length, dtype=torch.int32, device=device)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full attention with flash-attention 2's varlen kernel: no mask, no dropout, default scale.

    Args:
        q (Tensor): ``[B, Lq, N, D]`` on CUDA, ``D <= 256``; cast to bf16 unless fp16 or bf16.
        k (Tensor): ``[B, Lk, N, D]``, cast likewise.
        v (Tensor): ``[B, Lk, N, D]``, cast likewise.

    Returns:
        Tensor: ``[B, Lq, N, D]`` in the dtype of ``q``.
    """
    if q.device.type != "cuda" or q.size(-1) > 256:
        raise RuntimeError("flash_attention needs CUDA and head_dim <= 256; use sdpa_attention")
    flash_attn = _kernel_module((FA2_MODULE,))
    batch, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    q, k, v = (
        x.flatten(0, 1) if x.dtype in HALF_DTYPES else x.flatten(0, 1).to(GENERATOR_DTYPE)
        for x in (q, k, v)
    )
    x = flash_attn.flash_attn_varlen_func(
        q=q.to(v.dtype),
        k=k.to(v.dtype),
        v=v,
        cu_seqlens_q=_cu_seqlens(batch, lq, q.device),
        cu_seqlens_k=_cu_seqlens(batch, lk, q.device),
        max_seqlen_q=lq,
        max_seqlen_k=lk,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
    )
    return x.unflatten(0, (batch, lq)).type(out_dtype)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference full attention on ``scaled_dot_product_attention``, any device and float dtype.

    Args:
        q (Tensor): ``[B, Lq, N, D]``, cast to the dtype of ``v``.
        k (Tensor): ``[B, Lk, N, D]``, cast likewise.
        v (Tensor): ``[B, Lk, N, D]``.

    Returns:
        Tensor: ``[B, Lq, N, D]`` in the dtype of ``q``.
    """
    out = F.scaled_dot_product_attention(
        q.to(v.dtype).transpose(1, 2), k.to(v.dtype).transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2).contiguous().type(q.dtype)


def fa3_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """FlashAttention-3's ``flash_attn_func``, on CUDA; not bit-equal to :func:`flash_attention`.

    Args:
        q (Tensor): ``[B, Lq, N, D]`` fp16 or bf16, cast to the dtype of ``v``.
        k (Tensor): ``[B, Lk, N, D]``, cast likewise.
        v (Tensor): ``[B, Lk, N, D]`` fp16 or bf16.

    Returns:
        Tensor: ``[B, Lq, N, D]`` in the dtype of ``q``.
    """
    flash_attn = _kernel_module(FA3_MODULES)
    out = flash_attn.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, causal=False)
    return (out[0] if isinstance(out, tuple) else out).type(q.dtype)


#: The paper's kernel, by its name in :data:`ATTENTION_KERNELS`.
PAPER_ATTENTION = "flash"
#: The kernels by the name the ``model.attention`` setting gives them. Only the paper's gives its
#: bits; ``sdpa`` runs on any device.
ATTENTION_KERNELS: dict[str, AttentionFn] = {
    PAPER_ATTENTION: flash_attention,
    "fa3": fa3_attention,
    "sdpa": sdpa_attention,
}


def attention_kernel(name: str, device: torch.device | str) -> AttentionFn:
    """The kernel the ``model.attention`` setting ``name`` selects on ``device``: the kernel of
    :data:`ATTENTION_KERNELS` on CUDA, :func:`sdpa_attention` elsewhere (the flash kernels need
    CUDA)."""
    if name not in ATTENTION_KERNELS:
        raise ValueError(f"unknown attention kernel {name!r} ({' | '.join(ATTENTION_KERNELS)})")
    return ATTENTION_KERNELS[name] if torch.device(device).type == "cuda" else sdpa_attention


# =================================================================================== training masks
def causal_blocks(num_frames: int) -> list[tuple[int, int]]:
    """The causal blocks ``[(start, end), ...]`` of a window of ``num_frames`` latent frames, each
    the frames ``start .. end - 1``: the first frame alone, then blocks of four (the last one
    shorter when the frames do not fill it)."""
    if num_frames < 1:
        raise ValueError("a window has at least one latent frame")
    starts = range(1, num_frames, BLOCK)
    return [(0, 1)] + [(start, min(start + BLOCK, num_frames)) for start in starts]


@dataclass(frozen=True)
class MaskLayout:
    """Geometry of one training attention mask.

    Attributes:
        num_frames (int): latent frames ``F`` of the window.
        frame_tokens (int): tokens per latent frame (252).
        teacher_forcing (bool): the sequence is ``[context | noisy]`` in :func:`causal_blocks`
            (stage 3). A context token of block ``g`` attends to the context tokens of blocks ``<=
            g``; a noisy token to the context tokens of blocks ``< g`` and to the noisy tokens of
            block ``g``. Without it every token attends to the whole window (stages 1, 2 and 2s).
    """

    num_frames: int
    frame_tokens: int
    teacher_forcing: bool

    @property
    def total_length(self) -> int:
        return self.num_frames * self.frame_tokens * (2 if self.teacher_forcing else 1)

    @property
    def blocks(self) -> list[tuple[int, int]]:
        """The frames ``[(start, end), ...]`` that attend to each other."""
        return causal_blocks(self.num_frames) if self.teacher_forcing else [(0, self.num_frames)]


def attention_intervals(
    layout: MaskLayout, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The keys each query token attends to, as intervals.

    Query ``q`` attends to the keys in ``[starts[q], ends[q])`` (its own block; for a context
    token, the context up to its block), to the keys before ``context_ends[q]`` (for a noisy
    token, the context of the earlier blocks) and to itself.

    Returns:
        tuple[Tensor, Tensor, Tensor]: ``(starts, ends, context_ends)``, each
        ``[layout.total_length]`` long.
    """
    tokens = layout.frame_tokens
    video_length = layout.num_frames * tokens
    starts = torch.zeros(layout.total_length, dtype=torch.long, device=device)
    ends, context_ends = torch.zeros_like(starts), torch.zeros_like(starts)
    for start_frame, end_frame in layout.blocks:
        start, end = start_frame * tokens, end_frame * tokens
        ends[start:end] = end
        if layout.teacher_forcing:
            noisy = slice(video_length + start, video_length + end)
            starts[noisy] = video_length + start
            ends[noisy] = video_length + end
            context_ends[noisy] = start
    return starts, ends, context_ends


def _attends(
    query: torch.Tensor,
    key: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    context_ends: torch.Tensor,
) -> torch.Tensor:
    """Whether token ``query`` attends to token ``key``, given :func:`attention_intervals`: the
    predicate of both the dense and the flex mask."""
    in_block = (key >= starts[query]) & (key < ends[query])
    return in_block | (key < context_ends[query]) | (query == key)


def dense_attention_mask(layout: MaskLayout, device: torch.device | str = "cpu") -> torch.Tensor:
    """``[L, L]`` bool mask, True where query row ``q`` may attend to key column ``k``."""
    index = torch.arange(layout.total_length, device=device)
    return _attends(index[:, None], index[None, :], *attention_intervals(layout, device))


def flex_block_mask(layout: MaskLayout, device: torch.device | str = "cpu") -> Any:
    """The same mask as a flex ``BlockMask`` over the sequence padded to a multiple of 128.

    A padded row attends only to itself, so no row is empty; padded rows are sliced off after the
    attention.
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex attention needs torch >= 2.5")
    # Attribution: the padding to a multiple of 128, the q == k term and the eager create_block_mask
    # call follow CausVid's blockwise mask (github.com/tianweiy/CausVid at fab2440f, MIT) via Self
    # Forcing (github.com/guandeh17/Self-Forcing, Apache-2.0); create_block_mask is PyTorch's API.
    padded_length = math.ceil(layout.total_length / MASK_BLOCK_SIZE) * MASK_BLOCK_SIZE
    padding = padded_length - layout.total_length
    intervals = [F.pad(t, (0, padding)) for t in attention_intervals(layout, device)]

    def attention_mask(_batch, _head, query_index, key_index):
        return _attends(query_index, key_index, *intervals)

    return _flex_module.create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        device=device,
        _compile=False,
    )


@functools.cache
def _training_mask(layout: MaskLayout, device: torch.device) -> TrainingMask:
    if device.type == "cuda" and FLEX_ATTENTION_AVAILABLE:
        return flex_block_mask(layout, device)
    return dense_attention_mask(layout, device)


def training_mask(layout: MaskLayout, device: torch.device | str = "cpu") -> TrainingMask:
    """The training mask of ``layout`` on ``device``: a flex ``BlockMask`` on CUDA (torch >=
    2.5), the dense mask elsewhere. Built once and shared by every call: read it, never write
    it."""
    return _training_mask(layout, torch.device(device))


# ================================================================================= masked attention
def masked_sdpa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked attention on ``scaled_dot_product_attention``.

    Args:
        q (Tensor): ``[B, Lq, N, D]``.
        k (Tensor): ``[B, Lk, N, D]``.
        v (Tensor): ``[B, Lk, N, D]``.
        mask (Tensor): ``[Lq, Lk]`` bool, True where a query may attend to a key
            (:func:`dense_attention_mask`).

    Returns:
        Tensor: ``[B, Lq, N, D]``.
    """
    if tuple(mask.shape[-2:]) != (q.shape[1], k.shape[1]):
        raise ValueError(
            f"dense attention mask {tuple(mask.shape[-2:])} does not match Q={q.shape[1]}"
            f" KV={k.shape[1]}"
        )
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask[None, None]
    ).transpose(1, 2)


def _flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: Any) -> torch.Tensor:
    return _flex_module.flex_attention(q, k, v, block_mask=block_mask)


@functools.cache
def _compiled_flex() -> Callable:
    return torch.compile(_flex, dynamic=False, fullgraph=True, mode="default")


def masked_flex_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: Any, *, compiled: bool = True
) -> torch.Tensor:
    """Masked attention with flex attention and a ``BlockMask`` of :func:`flex_block_mask`.

    Args:
        q (Tensor): ``[B, Lq, N, D]``, padded to the mask's length inside.
        k (Tensor): ``[B, Lk, N, D]``, padded likewise.
        v (Tensor): ``[B, Lk, N, D]``, padded likewise.
        block_mask (BlockMask): the mask.
        compiled (bool): the compiled kernel; ``False`` runs eager flex attention, a slow
            reference that also runs on the CPU.

    Returns:
        Tensor: ``[B, Lq, N, D]``.
    """
    if not FLEX_ATTENTION_AVAILABLE or not isinstance(block_mask, _flex_module.BlockMask):
        raise TypeError(f"masked_flex_attention needs a flex BlockMask, got {type(block_mask)!r}")
    q_pad = int(block_mask.shape[-2]) - q.shape[1]
    kv_pad = int(block_mask.shape[-1]) - k.shape[1]
    if q_pad < 0 or kv_pad < 0:
        raise ValueError("attention mask is shorter than the token sequence")
    if q_pad:
        q = F.pad(q, (0, 0, 0, 0, 0, q_pad))
    if kv_pad:
        k = F.pad(k, (0, 0, 0, 0, 0, kv_pad))
        v = F.pad(v, (0, 0, 0, 0, 0, kv_pad))
    kernel = _compiled_flex() if compiled else _flex
    out = kernel(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask)
    return out.transpose(1, 2)[:, : q.shape[1] - q_pad]


def masked_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: TrainingMask
) -> torch.Tensor:
    """Masked attention: SDPA for a dense mask, flex attention for a ``BlockMask`` (compiled on
    CUDA, eager elsewhere).

    Args:
        q (Tensor): ``[B, Lq, N, D]``.
        k (Tensor): ``[B, Lk, N, D]``.
        v (Tensor): ``[B, Lk, N, D]``.
        mask (TrainingMask): the mask of :func:`training_mask`.

    Returns:
        Tensor: ``[B, Lq, N, D]``.
    """
    if isinstance(mask, torch.Tensor):
        return masked_sdpa_attention(q, k, v, mask)
    if FLEX_ATTENTION_AVAILABLE and isinstance(mask, _flex_module.BlockMask):
        return masked_flex_attention(q, k, v, mask, compiled=q.is_cuda)
    raise TypeError(f"unsupported attention mask type: {type(mask)!r}")
