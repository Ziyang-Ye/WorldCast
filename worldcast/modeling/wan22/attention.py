"""Attention kernels of the generator and the attention masks of training.

An :data:`AttentionFn` is ``fn(q, k, v)`` on ``[B, L, heads, head_dim]`` with no mask:
:func:`flash_attention` is the deployed kernel (CUDA), :func:`sdpa_attention` a reference for the
CPU, :func:`cudnn_attention` and :func:`fa3_attention` faster kernels of the real-time engine; only
the first gives the paper's bits. A training mask is bidirectional (stages 1, 2, 2s) or block-causal
teacher forcing over ``[clean | noisy]`` (stage 3), see :class:`MaskLayout`.
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

try:  # flex attention (torch >= 2.5): the training kernel on CUDA
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:  # pragma: no cover - old torch; the dense mask still works
    BlockMask = create_block_mask = _flex_attention = None
    FLEX_ATTENTION_AVAILABLE = False

try:  # FlashAttention-3 (Hopper), preferred when present
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    flash_attn_interface = None
    FLASH_ATTN_3_AVAILABLE = False

try:  # FlashAttention-2
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    flash_attn = None
    FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    "AttentionFn",
    "FA3_MODULES",
    "FLASH_ATTN_2_AVAILABLE",
    "FLASH_ATTN_3_AVAILABLE",
    "FLEX_ATTENTION_AVAILABLE",
    "MASK_BLOCK_SIZE",
    "MaskLayout",
    "TrainingMask",
    "attention_intervals",
    "build_attention_mask",
    "cudnn_attention",
    "dense_attention_mask",
    "fa3_attention",
    "fa3_module",
    "flash_attention",
    "flash_attention_backend",
    "flex_block_mask",
    "frame_groups",
    "masked_attention",
    "masked_flex_attention",
    "masked_sdpa_attention",
    "sdpa_attention",
]

#: ``fn(q, k, v) -> out``: q ``[B, Lq, N, D]``, k and v ``[B, Lk, N, D]``, out ``[B, Lq, N, D]``.
AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

#: A training mask: a dense ``[L, L]`` bool tensor (True = may attend) or a flex ``BlockMask``.
TrainingMask = torch.Tensor | Any

#: Flex attention's tile; both sides of a flex mask are padded to a multiple of it.
MASK_BLOCK_SIZE = 128

#: Where :func:`fa3_attention` finds FlashAttention-3: a source install, or the Hugging Face kernels
#: hub's ``kernels-community/flash-attn3`` unpacked as ``flash_attn3_hub``. The hub package has its
#: own name so that :func:`flash_attention` keeps flash-attention 2, the paper's kernel.
FA3_MODULES = ("flash_attn_interface", "flash_attn3_hub")

_HALF_DTYPES = (torch.float16, torch.bfloat16)


# =================================================================================== full attention
def flash_attention_backend() -> str:
    """The kernel of :func:`flash_attention`: ``"fa3"``, ``"fa2"`` or ``"none"``.

    Record it with every reference run: FA2 and FA3 give different bits.
    """
    if FLASH_ATTN_3_AVAILABLE:
        return "fa3"
    if FLASH_ATTN_2_AVAILABLE:
        return "fa2"
    return "none"


@functools.cache
def _cu_seqlens(batch: int, length: int, device: torch.device) -> torch.Tensor:
    """Sequence offsets ``[0, L, 2 L, ...]`` int32, kept on the device (no kernel per call)."""
    return torch.arange(0, (batch + 1) * length, length, dtype=torch.int32, device=device)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full attention with flash-attn's varlen kernel: no mask, no dropout, default scale.

    Args:
        q (Tensor): ``[B, Lq, N, D]`` on CUDA, ``D <= 256``; cast to bf16 unless fp16 or bf16.
        k (Tensor): ``[B, Lk, N, D]``, cast likewise.
        v (Tensor): ``[B, Lk, N, D]``, cast likewise.

    Returns:
        Tensor: ``[B, Lq, N, D]`` in the dtype of ``q``.
    """
    if q.device.type != "cuda" or q.size(-1) > 256:
        raise RuntimeError("flash_attention needs CUDA and head_dim <= 256; use sdpa_attention")
    if not (FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_2_AVAILABLE):
        raise RuntimeError("neither flash_attn_interface (FA3) nor flash_attn (FA2) is installed")
    batch, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    q, k, v = (
        x.flatten(0, 1) if x.dtype in _HALF_DTYPES else x.flatten(0, 1).to(torch.bfloat16)
        for x in (q, k, v)
    )
    common = dict(
        q=q.to(v.dtype),
        k=k.to(v.dtype),
        v=v,
        cu_seqlens_q=_cu_seqlens(batch, lq, q.device),
        cu_seqlens_k=_cu_seqlens(batch, lk, q.device),
        max_seqlen_q=lq,
        max_seqlen_k=lk,
        softmax_scale=None,
        causal=False,
        deterministic=False,
    )
    if FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(seqused_q=None, seqused_k=None, **common)[0]
    else:
        x = flash_attn.flash_attn_varlen_func(dropout_p=0.0, window_size=(-1, -1), **common)
    return x.unflatten(0, (batch, lq)).type(out_dtype)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference full attention on ``scaled_dot_product_attention``, any device and float dtype."""
    out = F.scaled_dot_product_attention(
        q.to(v.dtype).transpose(1, 2), k.to(v.dtype).transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2).contiguous().type(q.dtype)


def cudnn_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full attention on SDPA's cuDNN kernel (Hopper); not bit-equal to :func:`flash_attention`."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.to(v.dtype).transpose(1, 2), v.transpose(1, 2)
        )
    return out.transpose(1, 2).type(q.dtype)


def fa3_module() -> Any:
    """The first importable module of :data:`FA3_MODULES`, or ``None``."""
    for name in FA3_MODULES:
        if importlib.util.find_spec(name) is not None:
            return importlib.import_module(name)
    return None


def fa3_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """FlashAttention-3's ``flash_attn_func``; not bit-equal to :func:`flash_attention`."""
    out = fa3_module().flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, causal=False)
    return (out[0] if isinstance(out, tuple) else out).type(q.dtype)


# =================================================================================== training masks
@dataclass(frozen=True)
class MaskLayout:
    """Geometry of one training attention mask.

    Attributes:
        num_frames (int): latent frames ``F`` of the window.
        frame_seq_length (int): tokens per latent frame (252).
        num_frame_per_block (int): frames per causal block (the window for the bidirectional
            stages, 4 for stage 3).
        independent_first_frame (bool): frame 0 is a block of its own (stage 3: the sink).
        teacher_forcing (bool): the sequence is ``[clean | noisy]``. A clean token of block ``g``
            attends to the clean tokens of blocks ``<= g``; a noisy token to the clean tokens of
            blocks ``< g`` and to the noisy tokens of block ``g``.
    """

    num_frames: int
    frame_seq_length: int
    num_frame_per_block: int
    independent_first_frame: bool
    teacher_forcing: bool

    @property
    def total_length(self) -> int:
        return self.num_frames * self.frame_seq_length * (2 if self.teacher_forcing else 1)

    @classmethod
    def bidirectional(cls, num_frames: int, frame_seq_length: int) -> "MaskLayout":
        """Stages 1, 2 and 2s: one block spanning the window."""
        return cls(int(num_frames), int(frame_seq_length), int(num_frames), False, False)

    @classmethod
    def teacher_forcing_blocks(
        cls,
        num_frames: int,
        frame_seq_length: int,
        *,
        num_frame_per_block: int = 4,
        independent_first_frame: bool = True,
    ) -> "MaskLayout":
        """Stage 3: teacher forcing over blocks of ``num_frame_per_block`` after the sink."""
        return cls(
            int(num_frames),
            int(frame_seq_length),
            int(num_frame_per_block),
            bool(independent_first_frame),
            True,
        )


def frame_groups(
    num_frames: int, num_frame_per_block: int, independent_first_frame: bool
) -> list[tuple[int, int]]:
    """The causal blocks ``[(first_frame, end_frame), ...]`` of a window."""
    if num_frames < 1 or num_frame_per_block < 1:
        raise ValueError("a causal mask needs at least one frame and one frame per block")
    groups = [(0, 1)] if independent_first_frame else []
    start = len(groups)
    while start < num_frames:
        end = min(start + num_frame_per_block, num_frames)
        groups.append((start, end))
        start = end
    return groups


def attention_intervals(
    layout: MaskLayout, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Up to two key intervals per query token.

    Query ``q`` attends to the keys in ``[starts1[q], ends1[q])``, in ``[starts2[q], ends2[q])``
    and to itself.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: ``(starts1, ends1, starts2, ends2)``, each
        ``[layout.total_length]`` long.
    """
    tokens = layout.frame_seq_length
    video_length = layout.num_frames * tokens
    starts1 = torch.zeros(layout.total_length, dtype=torch.long, device=device)
    ends1, starts2, ends2 = (torch.zeros_like(starts1) for _ in range(3))
    groups = frame_groups(
        layout.num_frames, layout.num_frame_per_block, layout.independent_first_frame
    )
    for first_frame, end_frame in groups:
        start, end = first_frame * tokens, end_frame * tokens
        ends1[start:end] = end
        if layout.teacher_forcing:
            noisy = slice(video_length + start, video_length + end)
            starts1[noisy] = video_length + start
            ends1[noisy] = video_length + end
            ends2[noisy] = start
    return starts1, ends1, starts2, ends2


def dense_attention_mask(layout: MaskLayout, device: str | torch.device = "cpu") -> torch.Tensor:
    """``[L, L]`` bool mask, True where query row ``q`` may attend to key column ``k``."""
    starts1, ends1, starts2, ends2 = attention_intervals(layout, device)
    query = torch.arange(starts1.numel(), device=device)[:, None]
    key = torch.arange(starts1.numel(), device=device)[None, :]
    return (
        ((key >= starts1[query]) & (key < ends1[query]))
        | ((key >= starts2[query]) & (key < ends2[query]))
        | (query == key)
    )


def flex_block_mask(layout: MaskLayout, device: str | torch.device = "cuda") -> Any:
    """The same mask as a flex ``BlockMask`` over the sequence padded to a multiple of 128.

    A padded row attends only to itself, so no row is empty; padded rows are sliced off after the
    attention.
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex attention needs torch >= 2.5; use the dense mask")
    # Attribution: the padding to a multiple of 128, the q == k term and the eager create_block_mask
    # call follow CausVid's blockwise mask (github.com/tianweiy/CausVid at fab2440f, MIT) via Self
    # Forcing (github.com/guandeh17/Self-Forcing, Apache-2.0); create_block_mask is PyTorch's API.
    starts1, ends1, starts2, ends2 = attention_intervals(layout, device)
    total_length = starts1.numel()
    padded_length = math.ceil(total_length / MASK_BLOCK_SIZE) * MASK_BLOCK_SIZE
    padding = padded_length - total_length
    if padding:
        starts1, ends1, starts2, ends2 = (
            F.pad(t, (0, padding)) for t in (starts1, ends1, starts2, ends2)
        )

    def attention_mask(_batch, _head, query_index, key_index):
        in_first = (key_index >= starts1[query_index]) & (key_index < ends1[query_index])
        in_second = (key_index >= starts2[query_index]) & (key_index < ends2[query_index])
        return in_first | in_second | (query_index == key_index)

    return create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        device=device,
        _compile=False,
    )


def build_attention_mask(
    layout: MaskLayout, device: str | torch.device, *, backend: str = "auto"
) -> TrainingMask:
    """The training mask of ``layout``.

    Args:
        layout (MaskLayout): the mask geometry.
        device (str | torch.device): where the mask lives.
        backend (str): ``"flex"``, ``"dense"`` or ``"auto"`` (flex on CUDA, dense elsewhere).
    """
    device = torch.device(device)
    if backend == "auto":
        backend = "flex" if device.type == "cuda" else "dense"
    if backend == "flex":
        return flex_block_mask(layout, device)
    if backend == "dense":
        return dense_attention_mask(layout, device)
    raise ValueError(f"unknown mask backend {backend!r} (auto | flex | dense)")


# ================================================================================= masked attention
def masked_sdpa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked attention on ``scaled_dot_product_attention`` with a dense ``[Lq, Lk]`` bool mask."""
    if tuple(mask.shape[-2:]) != (q.shape[1], k.shape[1]):
        raise ValueError(
            f"dense attention mask {tuple(mask.shape[-2:])} does not match Q={q.shape[1]}"
            f" KV={k.shape[1]}"
        )
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask[None, None]
    ).transpose(1, 2)


def _flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: Any) -> torch.Tensor:
    return _flex_attention(q, k, v, block_mask=block_mask)


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
    if not FLEX_ATTENTION_AVAILABLE or not isinstance(block_mask, BlockMask):
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
    CUDA, eager elsewhere)."""
    if isinstance(mask, torch.Tensor):
        return masked_sdpa_attention(q, k, v, mask)
    if FLEX_ATTENTION_AVAILABLE and isinstance(mask, BlockMask):
        return masked_flex_attention(q, k, v, mask, compiled=q.is_cuda)
    raise TypeError(f"unsupported attention mask type: {type(mask)!r}")
