"""The WorldCast generator ``G_theta`` (Sec. 3, Eq. (1)): Wan2.2-TI2V-5B as a block-causal DiT.

:meth:`WorldCastGenerator.forward` runs one range of the window (sink, memory slot, recent or
target) through the :class:`KVCache`; :meth:`WorldCastGenerator.forward_train` runs the whole
window under an attention mask. The controls ``a_n`` modulate every DiT block, the ray code is
added before the first block (Eq. (3)) and the player state field after the second (Eq. (2)).
Precision: docs/inference.md, "Numerics that the paper's numbers depend on".
"""

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from worldcast.modeling.action import ActionConfig, ControlConditioner, adaln_adapter
from worldcast.modeling.obs_signal import OBS_SIGNAL_KEYS, ObserverSignalBranch
from worldcast.modeling.rays import RayConditions, RayEmbedding
from worldcast.modeling.state_injector import StateInjector, StateInjectorConfig
from worldcast.modeling.visibility_head import (
    VisibilityHead,
    VisibilityHeadConfig,
    VisibilityInputs,
)
from worldcast.utils.precision import cast_floating_tensors, generator_autocast

from .attention import (
    AttentionFn,
    MaskLayout,
    TrainingMask,
    build_attention_mask,
    flash_attention,
    masked_attention,
)

__all__ = [
    "Attention",
    "CausalDiTBlock",
    "CausalGeneratorAdapter",
    "CausalHead",
    "FieldBuilder",
    "GeneratorConditions",
    "GeneratorConfig",
    "KVCache",
    "StateFieldFn",
    "TrainingForward",
    "WorldCastGenerator",
    "apply_rope",
    "causal_self_attention",
    "coarsen_field",
    "rope_params",
    "rope_table",
    "sinusoidal_embedding_1d",
]

#: ``fn(frame_offset, num_frames) -> field``, called at the injection point inside the generator's
#: autocast. The field is ``[B, F_window, 23, h, w]`` or the call's rows ``[B, F, 23, h, w]``.
StateFieldFn = Callable[[int, int], torch.Tensor]

#: ``fn(conditions, weapon_embedding_weight, frame_offset, num_frames) -> field``, for example
#: ``worldcast.player_state.field.build_field``; the weight is the ``[52, 4]`` table.
FieldBuilder = Callable[[Mapping[str, Any], torch.Tensor, int, int], torch.Tensor]

#: Masks of :meth:`WorldCastGenerator.forward_train`: ``"full"`` (stages 1, 2, 2s) and
#: ``"teacher_forcing"`` (stage 3).
_TRAIN_MASKS = ("full", "teacher_forcing")


def _fp32_island():
    """An fp32 autocast region: on CUDA it computes the matmuls inside in fp32, as trained."""
    return torch.amp.autocast("cuda", dtype=torch.float32)


# ============================================================================= positional encodings
def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """``[N] -> [N, dim]`` float64 ``[cos | sin]`` timestep embedding."""
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """``[max_seq_len, dim // 2]`` complex128 rotary table, built on the CPU."""
    freqs = torch.outer(
        torch.arange(max_seq_len, device="cpu"),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, device="cpu").to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.amp.autocast("cuda", enabled=False)
def rope_table(
    freqs: torch.Tensor, grid: tuple[int, int, int], start_frame: int = 0
) -> torch.Tensor:
    """The 3D RoPE factors (temporal | height | width) of every token of a call.

    Args:
        freqs (Tensor): ``[1024, head_dim // 2]`` complex128 table (:func:`rope_params`).
        grid (tuple[int, int, int]): token grid ``(F, h, w)`` of the call.
        start_frame (int): window index of the call's first latent frame.

    Returns:
        Tensor: ``[F h w, 1, head_dim // 2]`` complex128.
    """
    # Attribution: Wan2.2's rope_apply (github.com/Wan-Video/Wan2.2, Apache-2.0, (c) 2024-2025 The
    # Alibaba Wan Team Authors); the temporal offset (freqs[0][start_frame:start_frame + frames])
    # follows CausVid (github.com/tianweiy/CausVid at fab2440f, MIT) via Self Forcing
    # (github.com/guandeh17/Self-Forcing, Apache-2.0).
    frames, height, width = grid
    c = freqs.shape[1]
    temporal, vertical, horizontal = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    return torch.cat(
        [
            temporal[start_frame : start_frame + frames]
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
    """RMS norm computed in fp32, cast back to the input dtype, then scaled."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32, cast back to the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


# ======================================================================================== attention
class KVCache:
    """The self-attention keys and values of every DiT block, ``[B, capacity, heads, head_dim]``.

    Every block holds the same token range ``[0, end)``. A call writes its keys and values at
    ``[start, start + n)`` and attends to ``[0, start + n)``, after which the cache ends there; a
    write that reaches past ``end`` starts at ``end``, one inside rewrites in place (the ladder on
    its target). Nothing is evicted: a rollout caches at most 25 latent frames.

    A plain class, neither a dataclass nor a container, so that input casts (FSDP's root cast,
    :func:`~worldcast.utils.precision.cast_floating_tensors`) hand on the caller's own cache.

    Args:
        keys (list[Tensor]): one ``[B, capacity, heads, head_dim]`` buffer per block.
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
        num_blocks: int,
        num_heads: int,
        head_dim: int,
        capacity_latents: int,
        frame_seq_length: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> "KVCache":
        """Zeroed buffers for ``capacity_latents`` latent frames of ``frame_seq_length`` tokens."""
        shape = (batch_size, capacity_latents * frame_seq_length, num_heads, head_dim)
        return cls(
            [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_blocks)],
            [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_blocks)],
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
        if end > self.end and start != self.end:
            raise ValueError(
                f"a KV cache write past the end {self.end} must start there, not {start}"
            )
        if end > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} tokens; the write ends at {end}")
        return end

    def update(
        self, block: int, key: torch.Tensor, value: torch.Tensor, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write a block's keys and values at ``[start, end)``; return those of ``[0, end)``."""
        self.keys[block][:, start:end] = key
        self.values[block][:, start:end] = value
        return self.keys[block][:, :end], self.values[block][:, :end]


def causal_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rope: torch.Tensor,
    cache: KVCache,
    block: int,
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
        block (int): index of the DiT block.
        start (int): token index of the write.
        end (int): ``start + L``, from :meth:`KVCache.span`.
        attention (AttentionFn): the kernel.

    Returns:
        Tensor: ``[B, L, heads, head_dim]``.
    """
    q = apply_rope(q, rope).type_as(v)
    k = apply_rope(k, rope).type_as(v)
    return attention(q, *cache.update(block, k, v, start, end))


def _masked_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rope: torch.Tensor,
    mask: TrainingMask,
    teacher_forcing: bool,
) -> torch.Tensor:
    """Self-attention over the whole window; under teacher forcing each copy is rotated at
    positions ``0 .. F-1``."""

    def rotate(t: torch.Tensor) -> torch.Tensor:
        if not teacher_forcing:
            return apply_rope(t, rope)
        return torch.cat([apply_rope(copy, rope) for copy in t.chunk(2, dim=1)], dim=1)

    return masked_attention(rotate(q).type_as(v), rotate(k).type_as(v), v, mask)


class Attention(nn.Module):
    """The ``q``, ``k``, ``v`` and ``o`` projections of one attention of a DiT block, with
    RMS-normed queries and keys."""

    def __init__(self, dim: int, num_heads: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)

    def query(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, L, dim]`` -> ``[B, L, heads, head_dim]``."""
        return self._heads(self.norm_q(self.q(x)))

    def key_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, L, dim]`` -> keys and values ``[B, L, heads, head_dim]``."""
        return self._heads(self.norm_k(self.k(x))), self._heads(self.v(x))


# =========================================================================================== blocks
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


class CausalDiTBlock(nn.Module):
    """One DiT block: self-attention, text cross-attention and FFN, with per-frame AdaLN of the
    timestep and the controls on the self-attention and FFN branches.

    The block is three stages around its two attention kernels: :meth:`pre_attention`,
    :meth:`post_self_attention` and :meth:`post_cross_attention`. :meth:`forward` runs them with
    the caller's self-attention; the real-time engine runs them with its own kernels.
    """

    def __init__(self, config: "GeneratorConfig") -> None:
        super().__init__()
        dim = config.dim
        self.norm1 = LayerNorm(dim, config.eps)
        self.self_attn = Attention(dim, config.num_heads, eps=config.eps)
        self.norm3 = LayerNorm(dim, config.eps, elementwise_affine=True)
        self.cross_attn = Attention(dim, config.num_heads, eps=config.eps)
        self.norm2 = LayerNorm(dim, config.eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # self.action_adaln is attached by WorldCastGenerator, after the backbone initialisation

    def pre_attention(
        self, x: torch.Tensor, e: torch.Tensor, action_emb: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
        """The block's AdaLN vectors and the self-attention's queries, keys and values.

        Args:
            x (Tensor): ``[B, F S, dim]`` tokens.
            e (Tensor): ``[B, F, 6, dim]`` float32 timestep modulation.
            action_emb (Tensor): ``[B, F, dim]`` control embedding.

        Returns:
            tuple: ``(q, k, v)``, each ``[B, F S, heads, head_dim]``, and the six AdaLN vectors
            ``[B, F, 1, dim]`` float32.
        """
        with _fp32_island():
            modulation = _frame_modulation(self.modulation, e, self.action_adaln(action_emb))
        shift, scale = modulation[0], modulation[1]
        normed = _by_frame(self.norm1(x).float(), e.shape[1])
        h = (normed * (1 + scale) + shift).flatten(1, 2)
        return (self.self_attn.query(h), *self.self_attn.key_value(h)), modulation

    def post_self_attention(
        self, x: torch.Tensor, attended: torch.Tensor, modulation: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The gated self-attention residual, and the cross-attention's queries.

        Args:
            x (Tensor): ``[B, F S, dim]`` the block's input tokens.
            attended (Tensor): ``[B, F S, heads, head_dim]`` self-attention output.
            modulation (tuple[Tensor, ...]): the AdaLN vectors of :meth:`pre_attention`.

        Returns:
            tuple[Tensor, Tensor]: tokens ``[B, F S, dim]`` and queries
            ``[B, F S, heads, head_dim]``.
        """
        gate = modulation[2]
        branch = _by_frame(self.self_attn.o(attended.flatten(2)), gate.shape[1])
        with _fp32_island():
            x = x + (branch * gate).flatten(1, 2)
        return x, self.cross_attn.query(self.norm3(x))

    def post_cross_attention(
        self, x: torch.Tensor, attended: torch.Tensor, modulation: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """The cross-attention residual and the gated FFN residual: ``[B, F S, dim]``."""
        x = x + self.cross_attn.o(attended.flatten(2))
        shift, scale, gate = modulation[3:]
        normed = _by_frame(self.norm2(x).float(), gate.shape[1])
        branch = _by_frame(self.ffn((normed * (1 + scale) + shift).flatten(1, 2)), gate.shape[1])
        with _fp32_island():
            return x + (branch * gate).flatten(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        e: torch.Tensor,
        action_emb: torch.Tensor,
        context: torch.Tensor,
        self_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        attention: AttentionFn,
    ) -> torch.Tensor:
        """One block; every path enters here, so an FSDP-wrapped block unshards.

        Args:
            x (Tensor): ``[B, F S, dim]`` tokens (``[clean | noisy]`` under teacher forcing).
            e (Tensor): ``[B, F, 6, dim]`` float32 timestep modulation (of both copies).
            action_emb (Tensor): ``[B, F, dim]`` control embedding (of both copies).
            context (Tensor): ``[B, 512, dim]`` embedded text.
            self_attention (Callable): ``fn(q, k, v)``, RoPE and the cache or the training mask
                included (:func:`causal_self_attention`).
            attention (AttentionFn): the kernel of the cross-attention.

        Returns:
            Tensor: ``[B, F S, dim]`` float32.
        """
        qkv, modulation = self.pre_attention(x, e, action_emb)
        x, query = self.post_self_attention(x, self_attention(*qkv), modulation)
        attended = attention(query, *self.cross_attn.key_value(context))
        return self.post_cross_attention(x, attended, modulation)


class CausalHead(nn.Module):
    """Output head: per-frame AdaLN of the timestep and the controls, then a linear map to the
    patches, in fp32."""

    def __init__(
        self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """``[B, F S, dim]`` tokens, ``[B, F, dim]`` timestep embedding and controls ->
        ``[B, F S, prod(patch) * out_dim]``."""
        num_frames = e.shape[1]
        with _fp32_island():
            shift, scale = _frame_modulation(
                self.modulation, e.unsqueeze(2), self.action_adaln(action_emb)
            )
            normed = _by_frame(self.norm(x), num_frames)
            return self.head(normed * (1 + scale) + shift).flatten(1, 2)


# =========================================================================================== config
@dataclass(frozen=True)
class GeneratorConfig:
    """Architecture of the generator; the defaults are the paper's.

    The backbone fields are those of the Wan2.2-TI2V-5B ``config.json``
    (:func:`worldcast.modeling.build.generator_config_from_snapshot`), except ``patch_size`` and
    ``text_dim``, which that file does not set.
    """

    in_dim: int = 48
    out_dim: int = 48
    dim: int = 3072
    ffn_dim: int = 14336
    freq_dim: int = 256
    text_dim: int = 4096
    text_len: int = 512
    num_heads: int = 24
    num_layers: int = 30
    patch_size: tuple[int, int, int] = (1, 2, 2)
    eps: float = 1e-6
    action: ActionConfig = field(default_factory=ActionConfig)
    state_injector: StateInjectorConfig = field(default_factory=StateInjectorConfig)
    ray_unit_u: float = 420.0
    obs_signal_hidden: int = 128
    #: Coarse-field ablation: max-pool the field by this factor and copy it back (paper: 1). The
    #: late-injection ablation is ``state_injector.write_block``.
    field_downsample: int = 1
    #: The visibility probe of stages 2s and 3 (training only; ``None`` at inference).
    visibility_head: VisibilityHeadConfig | None = None

    def __post_init__(self) -> None:
        if self.dim % self.num_heads or (self.dim // self.num_heads) % 2:
            raise ValueError("dim must split into an even head_dim")
        if not 0 <= self.state_injector.write_block < self.num_layers:
            raise ValueError("state_injector.write_block must index a DiT block")
        if int(self.field_downsample) < 1:
            raise ValueError("field_downsample must be a positive integer (1 = the paper's field)")
        head = self.visibility_head
        if head is not None and not self.state_injector.write_block <= head.block < self.num_layers:
            raise ValueError(
                f"visibility_head.block {head.block} must lie in"
                f" [{self.state_injector.write_block}, {self.num_layers})"
            )


# ======================================================================================= conditions
#: Condition-dict key -> :class:`GeneratorConditions` attribute.
_CONDITION_KEYS = {
    "prompt_embeds": "prompt_embeds",
    "button_condition": "buttons",
    "camera_condition": "camera",
    "weapon_condition": "weapon",
    **{key: key for key in OBS_SIGNAL_KEYS},
}
#: Condition-dict key -> :class:`~worldcast.modeling.rays.RayConditions` attribute.
_RAY_KEYS = {
    "state_wp_frame_c2w": "frame_c2w",
    "state_wp_frame_tans": "frame_tans",
    "state_wp_anchor_c2w": "anchor_c2w",
    "state_wp_memory_c2w": "memory_c2w",
    "state_wp_memory_frames": "memory_frames",
}
#: Inputs of the field builder: they reach the generator as ``state_field``.
_FIELD_INPUT_KEYS = (
    "peer_states",
    "peer_actions",
    "peer_observer_slot",
    "peer_team_ids",
    "peer_alive",
    "peer_visible",
    "peer_weapons",
)
#: Keys of a training batch that are not model inputs.
_NON_MODEL_KEYS = ("prompts", "unconditional_dict")


@dataclass
class GeneratorConditions:
    """What one generator call is conditioned on, besides the latents, timesteps and cache.

    The tensors span the window; the generator reads the call's frames. ``T = 1 + 4 (F_window - 1)``
    pixel frames.

    Attributes:
        prompt_embeds (Tensor | Sequence[Tensor]): ``[B, L <= 512, 4096]`` umT5 embedding of the
            fixed prompt, or ``B`` tensors ``[L_b, 4096]``.
        buttons (Tensor): ``[B, T, 11]`` the observer's buttons.
        camera (Tensor): ``[B, T, 2]`` camera deltas.
        weapon (Tensor): ``[B, T]`` long weapon ids.
        obs_flash_flag, obs_flash_valid, obs_scope_on, obs_scope_level, obs_scope_valid (Tensor):
            ``[B, F_window]`` observer signals.
        state_field (Tensor | StateFieldFn): the player state field ``[B, F_window, 23, 12, 21]``
            (or the call's rows), or a :data:`StateFieldFn`.
        rays (RayConditions | None): the window's cameras, or ``None`` for no ray code.
    """

    prompt_embeds: torch.Tensor | Sequence[torch.Tensor]
    buttons: torch.Tensor
    camera: torch.Tensor
    weapon: torch.Tensor
    obs_flash_flag: torch.Tensor
    obs_flash_valid: torch.Tensor
    obs_scope_on: torch.Tensor
    obs_scope_level: torch.Tensor
    obs_scope_valid: torch.Tensor
    state_field: torch.Tensor | StateFieldFn
    rays: RayConditions | None = None

    @classmethod
    def from_dict(
        cls, conditions: Mapping[str, Any], *, state_field: torch.Tensor | StateFieldFn
    ) -> "GeneratorConditions":
        """From a sampler or trainer condition dict.

        The field builder's ``peer_*`` inputs are skipped (the field arrives as ``state_field``);
        any other unknown key is an error, and the ray keys come all together or not at all.
        """
        known = (*_CONDITION_KEYS, *_RAY_KEYS, *_FIELD_INPUT_KEYS)
        unknown = sorted(key for key in conditions if key not in known)
        if unknown:
            raise ValueError(f"condition keys with no consumer in the generator: {unknown}")
        missing = sorted(key for key in _CONDITION_KEYS if conditions.get(key) is None)
        if missing:
            raise ValueError(f"conditions are missing {missing}")
        given = {key: conditions.get(key) is not None for key in _RAY_KEYS}
        if any(given.values()) and not all(given.values()):
            raise ValueError(f"ray conditions arrive together or not at all: {given}")
        rays = None
        if all(given.values()):
            rays = RayConditions(**{name: conditions[key] for key, name in _RAY_KEYS.items()})
        values = {name: conditions[key] for key, name in _CONDITION_KEYS.items()}
        return cls(state_field=state_field, rays=rays, **values)


def _conditions_with_field(
    conditions: Mapping[str, Any], generator: "WorldCastGenerator", field_builder: FieldBuilder
) -> GeneratorConditions:
    """``conditions`` with a field provider built on them and the generator's weapon embedding."""
    weapon_weight = generator.weapon_embedding.weight

    def state_field(frame_offset: int, num_frames: int) -> torch.Tensor:
        return field_builder(conditions, weapon_weight, frame_offset, num_frames)

    return GeneratorConditions.from_dict(conditions, state_field=state_field)


# ============================================================================= field ablation, init
def coarsen_field(field: torch.Tensor, factor: int) -> torch.Tensor:
    """The coarse-field ablation: max-pool ``factor x factor`` cells, copy each maximum back.

    ``ceil_mode`` keeps the last, narrower cell of the 21-token width. ``[B, F, C, h, w]`` -> the
    same shape and dtype. The copy-back is nearest interpolation: ``repeat_interleave`` gives the
    same field but sums the gradient in another order, which changes the training numbers.
    """
    factor = int(factor)
    if factor == 1:
        return field
    batch, frames, channels, grid_h, grid_w = field.shape
    flat = field.reshape(batch * frames, channels, grid_h, grid_w)
    pooled = F.max_pool2d(flat, kernel_size=factor, stride=factor, ceil_mode=True)
    upsampled = F.interpolate(pooled, size=(grid_h, grid_w), mode="nearest")
    return upsampled.reshape(batch, frames, channels, grid_h, grid_w)


#: The last four field channels (dying, corpse, live and corpse identity) start with zero stem
#: weights.
_ZERO_STEM_CHANNELS = 4


def _init_state_injector(dim: int, config: StateInjectorConfig) -> StateInjector:
    """A :class:`StateInjector` initialised as in the paper's training runs.

    Draws, in order, the weapon embedding, a stem over the first 19 channels (then widened with zero
    columns for the last four) and the output projection (then zeroed). A plain 23-channel stem
    would draw other numbers and shift every module built after it. Nothing is drawn on a ``meta``
    default device (checkpoint loading).
    """
    with torch.device("meta"):
        injector = StateInjector(dim, config)
    device = torch.get_default_device()
    if device.type == "meta":
        return injector
    injector.to_empty(device=device)
    with torch.no_grad():
        injector.weapon_embedding.reset_parameters()
        stem = nn.Conv2d(
            injector.field_channels - _ZERO_STEM_CHANNELS, injector.stem.out_channels, 3, padding=1
        )
        zero_columns = stem.weight.new_zeros(
            stem.weight.shape[0], _ZERO_STEM_CHANNELS, *stem.weight.shape[2:]
        )
        injector.stem.weight.copy_(torch.cat([stem.weight, zero_columns], dim=1))
        injector.stem.bias.copy_(stem.bias)
        injector.proj.reset_parameters()
        nn.init.zeros_(injector.proj.weight)
        nn.init.zeros_(injector.proj.bias)
    return injector


# ======================================================================================== generator
class WorldCastGenerator(nn.Module):
    """``G_theta``: the causal Wan2.2-TI2V-5B DiT with the WorldCast conditioning.

    :meth:`forward` runs one range of the window on the KV cache (inference, stage-4 rollout);
    :meth:`forward_train` runs the whole window under an attention mask (training stages 1-3). The
    pieces of a call (:meth:`patchify`, :meth:`embed_timesteps`, ...) are public for the real-time
    engine. Checkpoint keys: the Wan2.2 backbone's, plus ``blocks.N.action_adaln``,
    ``head.action_adaln``, ``action``, ``state_injector``, ``obs_signal``, ``rays`` and, with
    ``config.visibility_head``, ``visibility_head``.

    Args:
        config (GeneratorConfig): the architecture.
        attention (AttentionFn | None): the full-attention kernel (default :func:`flash_attention`).
    """

    def __init__(
        self, config: GeneratorConfig = GeneratorConfig(), *, attention: AttentionFn | None = None
    ) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        self.dim = dim
        self.num_heads = config.num_heads
        self.freq_dim = config.freq_dim
        self.text_len = config.text_len
        self.out_dim = config.out_dim
        self.patch_size = tuple(config.patch_size)

        self.patch_embedding = nn.Conv3d(
            config.in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([CausalDiTBlock(config) for _ in range(config.num_layers)])
        self.head = CausalHead(dim, config.out_dim, self.patch_size, config.eps)
        self._init_backbone_weights()

        # a plain complex128 attribute, never cast with the parameters
        head_dim = dim // config.num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, head_dim - 4 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        )

        # Each zero-initialised on its output, and built in this order so that a fresh
        # initialisation draws the same random numbers as the paper's training runs.
        self.action = ControlConditioner(dim, config.action)
        for block in self.blocks:
            block.action_adaln = adaln_adapter(dim, 6, config.action.adaln_rank)
        self.head.action_adaln = adaln_adapter(dim, 2, config.action.adaln_rank)
        self.state_injector = _init_state_injector(dim, config.state_injector)
        self.visibility_head: VisibilityHead | None = (
            VisibilityHead(dim, config.visibility_head)
            if config.visibility_head is not None
            else None
        )
        self.obs_signal = ObserverSignalBranch(dim, hidden=config.obs_signal_hidden)
        self.rays = RayEmbedding(dim, ray_unit_u=config.ray_unit_u)

        self._attention: AttentionFn = attention or flash_attention
        self.gradient_checkpointing = False
        self._train_masks: dict[tuple[Any, ...], TrainingMask] = {}

    def _init_backbone_weights(self) -> None:
        """Wan2.2's initialisation of the backbone (the conditioning initialises itself)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for module in (*self.text_embedding.modules(), *self.time_embedding.modules()):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)

    @property
    def attention(self) -> AttentionFn:
        return self._attention

    def set_attention(self, attention: AttentionFn) -> None:
        """Replace the full-attention kernel (for example ``attention.sdpa_attention`` on CPU)."""
        self._attention = attention

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Recompute each block in backward (:meth:`forward_train` only)."""
        self.gradient_checkpointing = bool(enabled)

    @property
    def weapon_embedding(self) -> nn.Embedding:
        return self.state_injector.weapon_embedding

    # ----------------------------------------------------------------------------- pieces of a call
    def patchify(
        self, x: torch.Tensor, embed: Callable[[torch.Tensor], torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Patch-embed latents, one sample at a time (a batched conv can round differently).

        Args:
            x (Tensor): ``[B, C, F, H, W]`` latents.
            embed (Callable | None): replaces the ``patch_embedding`` conv (the same map).

        Returns:
            tuple: tokens ``[B, F h w, dim]`` and the token grid ``(F, h, w)``.
        """
        embed = embed or self.patch_embedding
        patches = [embed(sample.unsqueeze(0)) for sample in x]
        tokens = torch.cat([p.flatten(2).transpose(1, 2) for p in patches])
        return tokens, tuple(patches[0].shape[2:])

    def unpatchify(self, tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """``[B, F h w, prod(patch) * C]`` -> ``[B, C, F, H, W]``."""
        p_t, p_h, p_w = self.patch_size
        frames, height, width = grid
        x = tokens.view(tokens.shape[0], frames, height, width, p_t, p_h, p_w, self.out_dim)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(tokens.shape[0], self.out_dim, frames * p_t, height * p_h, width * p_w)

    def rope_table(self, grid: tuple[int, int, int], start_frame: int) -> torch.Tensor:
        """The call's :func:`rope_table` on the generator's device."""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        return rope_table(self.freqs, grid, start_frame)

    def embed_timesteps(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, F]`` timesteps -> embedding ``[B, F, dim]`` and block modulation
        ``[B, F, 6, dim]``, both float32."""
        with _fp32_island():
            sinusoid = sinusoidal_embedding_1d(self.freq_dim, timestep.flatten())
            embedding = self.time_embedding(sinusoid.unflatten(0, timestep.shape).float())
            modulation = self.time_projection(embedding).unflatten(2, (6, self.dim))
        return embedding, modulation

    def embed_observer_signals(
        self, cond: GeneratorConditions, *, frame_offset: int, num_frames: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """``[B, num_frames, dim]`` observer-signal embedding of the call's frames.

        The whole window is embedded and the call's rows are cut out of it, as trained.
        """
        device = self.patch_embedding.weight.device
        signals = (
            cond.obs_flash_flag,
            cond.obs_flash_valid,
            cond.obs_scope_on,
            cond.obs_scope_level,
            cond.obs_scope_valid,
        )
        embedding = self.obs_signal(*(s.to(device=device) for s in signals))
        return embedding.to(device=device, dtype=dtype)[:, frame_offset : frame_offset + num_frames]

    def embed_controls(
        self, cond: GeneratorConditions, *, frame_offset: int, num_frames: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """``[B, num_frames, dim]`` embedding of the controls ``a_n`` of the call's frames."""
        device = self.patch_embedding.weight.device
        return self.action(
            cond.buttons.to(device=device, dtype=dtype),
            cond.camera.to(device=device, dtype=dtype),
            cond.weapon.to(device=device),
            latent_frames=num_frames,
            latent_start=frame_offset,
            dtype=dtype,
        )

    def embed_text(self, prompt_embeds: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """The prompt zero-padded to 512 tokens and embedded: ``[B, 512, dim]``."""
        padded = [
            torch.cat([p, p.new_zeros(self.text_len - p.size(0), p.size(1))]) for p in prompt_embeds
        ]
        return self.text_embedding(torch.stack(padded))

    def state_field(
        self, cond: GeneratorConditions, frame_offset: int, num_frames: int
    ) -> torch.Tensor:
        """The player state field of the call (Eq. (2)), coarse under the coarse-field ablation."""
        state_field = cond.state_field
        if callable(state_field):
            state_field = state_field(frame_offset, num_frames)
        state_field = coarsen_field(state_field, self.config.field_downsample)
        return state_field.to(device=self.patch_embedding.weight.device)

    # ----------------------------------------------------------------------------- KV-cache forward
    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        cond: GeneratorConditions,
        *,
        kv_cache: KVCache,
        current_start: int,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        """One generator call on one range of the window.

        Args:
            noisy (Tensor): ``[B, F, 48, 24, 42]`` the call's noisy latent frames.
            timestep (Tensor): ``[B, F]`` per-frame timesteps in ``[0, 1000]``.
            cond (GeneratorConditions): the window's conditions.
            kv_cache (KVCache): the call's keys and values are written at ``cache_start``.
            current_start (int): window token index of the call's first frame,
                ``window_index * 252``: the positions of RoPE and of the control history.
            cache_start (int | None): token index of the cache write (default ``current_start``).

        Returns:
            Tensor: flow prediction ``[B, F, 48, 24, 42]`` float32.
        """
        with generator_autocast(noisy):
            tokens, grid = self.patchify(noisy.permute(0, 2, 1, 3, 4))
            frames, height, width = grid
            frame_seq_length = height * width
            cache_start = current_start if cache_start is None else cache_start
            if current_start % frame_seq_length or cache_start % frame_seq_length:
                raise ValueError("current_start and cache_start must be latent-frame boundaries")
            frame_offset = current_start // frame_seq_length
            end = kv_cache.span(cache_start, tokens.shape[1])

            signals = self.embed_observer_signals(
                cond, frame_offset=frame_offset, num_frames=frames, dtype=tokens.dtype
            )
            tokens = tokens + signals.repeat_interleave(frame_seq_length, dim=1)
            t_embedding, modulation = self.embed_timesteps(timestep)
            controls = self.embed_controls(
                cond, frame_offset=frame_offset, num_frames=frames, dtype=tokens.dtype
            )
            if cond.rays is not None:
                tokens = tokens + self.rays(
                    cond.rays,
                    frame_offset=frame_offset,
                    num_frames=frames,
                    grid=(height, width),
                    dtype=tokens.dtype,
                )
            context = self.embed_text(cond.prompt_embeds)
            rope = self.rope_table(grid, frame_offset)

            # Attribution: handing each block its cache entry and the call's token offsets follows
            # CausVid's causal forward (github.com/tianweiy/CausVid at fab2440f, MIT) via Self
            # Forcing (github.com/guandeh17/Self-Forcing, Apache-2.0); the rest of this forward is
            # Wan2.2's WanModel.forward (Apache-2.0).
            for index, block in enumerate(self.blocks):
                self_attention = functools.partial(
                    causal_self_attention,
                    rope=rope,
                    cache=kv_cache,
                    block=index,
                    start=cache_start,
                    end=end,
                    attention=self._attention,
                )
                tokens = block(
                    tokens,
                    e=modulation,
                    action_emb=controls,
                    context=context,
                    self_attention=self_attention,
                    attention=self._attention,
                )
                if index == self.state_injector.write_block:
                    tokens = self.state_injector(
                        tokens,
                        self.state_field(cond, frame_offset, frames),
                        frame_offset=frame_offset,
                        num_frames=frames,
                    )
            kv_cache.end = end
            video = self.unpatchify(self.head(tokens, t_embedding, controls), grid).float()
            return video.permute(0, 2, 1, 3, 4)

    # ----------------------------------------------------------------------------- training forward
    def forward_train(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        cond: GeneratorConditions,
        *,
        mask: str = "full",
        clean_context: torch.Tensor | None = None,
        context_timestep: torch.Tensor | None = None,
        num_frame_per_block: int = 4,
        independent_first_frame: bool = True,
        mask_backend: str = "auto",
        visibility: VisibilityInputs | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The whole window in one forward under an attention mask (training stages 1-3).

        Under teacher forcing the observer signals and the field reach the noisy copy only; the
        controls and the ray code reach both copies.

        Args:
            noisy (Tensor): ``[B, F, 48, 24, 42]`` noisy latents of the window.
            timestep (Tensor): ``[B, F]`` per-frame timesteps in ``[0, 1000]``.
            cond (GeneratorConditions): the window's conditions; a field provider is called as
                ``state_field(0, F)``.
            mask (str): ``"full"`` (stages 1, 2, 2s: bidirectional) or ``"teacher_forcing"``
                (stage 3: ``[clean | noisy]``, block-causal, see ``attention.MaskLayout``).
            clean_context (Tensor | None): ``[B, F, 48, 24, 42]`` the context copy (teacher
                forcing only).
            context_timestep (Tensor | None): ``[B, F]`` timesteps of ``clean_context`` (zeros).
            num_frame_per_block (int): frames per block of the teacher-forcing mask (paper: 4).
            independent_first_frame (bool): the sink is a block of its own (paper: ``True``).
            mask_backend (str): ``"auto"`` (flex on CUDA, dense elsewhere), ``"flex"`` or
                ``"dense"``.
            visibility (VisibilityInputs | None): inputs of the visibility head, whose logits
                ``[B, F, P]`` float32 are then returned too.

        Returns:
            Tensor | tuple[Tensor, Tensor]: flow prediction ``[B, F, 48, 24, 42]`` float32 of the
            noisy copy, or ``(flow, logits)``.
        """
        if mask not in _TRAIN_MASKS:
            raise ValueError(f"unknown training mask {mask!r}; expected one of {_TRAIN_MASKS}")
        teacher_forcing = mask == "teacher_forcing"
        if teacher_forcing and clean_context is None:
            raise ValueError("the teacher-forcing mask needs clean_context")
        if not teacher_forcing and (clean_context is not None or context_timestep is not None):
            raise ValueError(
                "clean_context and context_timestep belong to the teacher-forcing mask only"
            )
        if teacher_forcing and clean_context.shape != noisy.shape:
            raise ValueError("clean_context must have the shape of noisy")
        if visibility is not None and self.visibility_head is None:
            raise ValueError(
                "visibility inputs were given but the generator has no visibility head"
            )
        with generator_autocast(noisy):
            video, logits = self._forward_train(
                noisy.permute(0, 2, 1, 3, 4),
                timestep,
                cond,
                clean_x=None if clean_context is None else clean_context.permute(0, 2, 1, 3, 4),
                context_timestep=context_timestep,
                num_frame_per_block=int(num_frame_per_block),
                independent_first_frame=bool(independent_first_frame),
                mask_backend=mask_backend,
                visibility=visibility,
            )
            flow = video.permute(0, 2, 1, 3, 4)
        return flow if visibility is None else (flow, logits)

    def train_attention_mask(
        self, layout: MaskLayout, device: torch.device, *, backend: str = "auto"
    ) -> TrainingMask:
        """The training mask of ``layout``, cached per geometry, device and backend."""
        device = torch.device(device)
        key = (layout, device.type, device.index, backend)
        if key not in self._train_masks:
            self._train_masks[key] = build_attention_mask(layout, device, backend=backend)
        return self._train_masks[key]

    def _forward_train(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: GeneratorConditions,
        *,
        clean_x: torch.Tensor | None,
        context_timestep: torch.Tensor | None,
        num_frame_per_block: int,
        independent_first_frame: bool,
        mask_backend: str,
        visibility: VisibilityInputs | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens, grid = self.patchify(x)
        frames, height, width = grid
        frame_seq_length = height * width

        # the observer signals are added before the clean copy exists, so it never receives them
        signals = self.embed_observer_signals(
            cond, frame_offset=0, num_frames=frames, dtype=tokens.dtype
        )
        tokens = tokens + signals.repeat_interleave(frame_seq_length, dim=1)
        t_embedding, modulation = self.embed_timesteps(t)
        controls = self.embed_controls(cond, frame_offset=0, num_frames=frames, dtype=tokens.dtype)

        teacher_forcing = clean_x is not None
        copies = 2 if teacher_forcing else 1
        if teacher_forcing:
            tokens = torch.cat([self.patchify(clean_x)[0], tokens], dim=1)
            if context_timestep is None:
                context_timestep = torch.zeros_like(t)
            modulation = torch.cat([self.embed_timesteps(context_timestep)[1], modulation], dim=1)
            layout = MaskLayout.teacher_forcing_blocks(
                frames,
                frame_seq_length,
                num_frame_per_block=num_frame_per_block,
                independent_first_frame=independent_first_frame,
            )
        else:
            layout = MaskLayout.bidirectional(frames, frame_seq_length)
        if cond.rays is not None:
            code = self.rays(
                cond.rays,
                frame_offset=0,
                num_frames=frames,
                grid=(height, width),
                dtype=tokens.dtype,
            )
            tokens = tokens + code.repeat(1, copies, 1)

        mask = self.train_attention_mask(layout, tokens.device, backend=mask_backend)
        block_kwargs = dict(
            e=modulation,
            action_emb=torch.cat([controls] * copies, dim=1) if teacher_forcing else controls,
            context=self.embed_text(cond.prompt_embeds),
            self_attention=functools.partial(
                _masked_self_attention,
                rope=self.rope_table(grid, 0),
                mask=mask,
                teacher_forcing=teacher_forcing,
            ),
            attention=self._attention,
        )
        visibility_head = self.visibility_head if visibility is not None else None
        logits = None
        for index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                tokens = torch.utils.checkpoint.checkpoint(
                    block, tokens, use_reentrant=False, **block_kwargs
                )
            else:
                tokens = block(tokens, **block_kwargs)
            if index == self.state_injector.write_block:
                tokens = self._inject_state_train(tokens, cond, frames, teacher_forcing)
            if visibility_head is not None and index == visibility_head.block:
                logits = visibility_head(
                    tokens,
                    visibility,
                    grid_h=height,
                    grid_w=width,
                    frame_offset=0,
                    video_frames=frames,
                    teacher_forcing=teacher_forcing,
                )

        if teacher_forcing:
            tokens = tokens[:, tokens.shape[1] // 2 :]
        video = self.unpatchify(self.head(tokens, t_embedding, controls), grid).float()
        return video, logits

    def _inject_state_train(
        self, tokens: torch.Tensor, cond: GeneratorConditions, frames: int, teacher_forcing: bool
    ) -> torch.Tensor:
        """Eq. (2) on the whole window; under teacher forcing on the noisy copy only."""
        state_field = self.state_field(cond, 0, frames)
        if not teacher_forcing:
            return self.state_injector(tokens, state_field, frame_offset=0, num_frames=frames)
        half = tokens.shape[1] // 2
        noisy = self.state_injector(
            tokens[:, half:], state_field, frame_offset=0, num_frames=frames
        )
        return torch.cat([tokens[:, :half], noisy], dim=1)


# ========================================================================================= wrappers
class TrainingForward(nn.Module):
    """The training forward as a module, for the trainer to wrap with FSDP (keys ``generator.``).

    FSDP unshards parameters and casts inputs in the hooks around ``forward``; calling
    ``forward_train`` on a wrapped generator would bypass them. Per call: the optional cast of every
    floating input (``input_dtype``; ``None`` when FSDP mixed precision casts, as in the paper's
    runs), the field built inside the forward (so the weapon embedding is unsharded and receives its
    gradient), then :meth:`WorldCastGenerator.forward_train`.

    Args:
        generator (WorldCastGenerator): the generator.
        field_builder (FieldBuilder): builds the player state field.
        input_dtype (torch.dtype | None): the input cast.
    """

    def __init__(
        self,
        generator: WorldCastGenerator,
        field_builder: FieldBuilder,
        *,
        input_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.field_builder = field_builder
        self.input_dtype = input_dtype

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """:meth:`WorldCastGenerator.forward_train` of a condition dict and keyword arguments."""
        model_conditions = {k: v for k, v in conditions.items() if k not in _NON_MODEL_KEYS}
        if self.input_dtype is not None:
            noisy, timestep, model_conditions, kwargs = cast_floating_tensors(
                (noisy, timestep, model_conditions, kwargs), self.input_dtype
            )
        cond = _conditions_with_field(model_conditions, self.generator, self.field_builder)
        return self.generator.forward_train(noisy, timestep, cond, **kwargs)


class CausalGeneratorAdapter:
    """The sampler's generator (:class:`worldcast.sampling.rollouts.CausalGenerator`) on top of
    :class:`WorldCastGenerator`.

    Per call: every floating input is cast to ``input_dtype`` (latents, timesteps and conditions;
    ``worldcast.utils.precision``), the player state field is built from the cast conditions at the
    injection point, and :meth:`WorldCastGenerator.forward` runs on the sampler's cache.

    Args:
        generator (WorldCastGenerator): the loaded generator.
        field_builder (FieldBuilder): builds the player state field, for example::

            lambda c, weapon_weight, frame_offset, num_frames: build_field(
                c["peer_states"], c["peer_actions"], c["peer_observer_slot"], c["peer_team_ids"],
                c["peer_alive"], c["peer_visible"], c["peer_weapons"], weapon_weight,
                frame_offset=frame_offset, video_frames=num_frames, config=field_config)

        input_dtype (torch.dtype | None): bf16 on the paper path; ``None`` (no cast) for fp32 CPU
            runs.
    """

    def __init__(
        self,
        generator: WorldCastGenerator,
        field_builder: FieldBuilder,
        *,
        input_dtype: torch.dtype | None = torch.bfloat16,
    ) -> None:
        self.generator = generator
        self.field_builder = field_builder
        self.input_dtype = input_dtype

    def inputs(
        self, noisy: torch.Tensor, timestep: torch.Tensor, conditions: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, GeneratorConditions]:
        """The call's inputs as the generator sees them: cast, with the field provider."""
        if self.input_dtype is not None:
            noisy, timestep, conditions = cast_floating_tensors(
                (noisy, timestep, dict(conditions)), self.input_dtype
            )
        return (
            noisy,
            timestep,
            _conditions_with_field(conditions, self.generator, self.field_builder),
        )

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
        """Flow ``[B, F, C, H, W]`` float32 of one call (:meth:`WorldCastGenerator.forward`)."""
        noisy, timestep, cond = self.inputs(noisy, timestep, conditions)
        return self.generator(
            noisy,
            timestep,
            cond,
            kv_cache=kv_cache,
            current_start=current_start,
            cache_start=cache_start,
        )
