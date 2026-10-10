"""The WorldCast generator ``G_theta`` (Sec. 3.1, Eq. (1)): Wan2.2-TI2V-5B as a block-causal DiT.

``V_n = G_theta(a_n, F_n, M_n, V_<n)``: the controls ``a_n`` modulate every DiT block (App.
"Injection"), the player state field ``F_n`` is added after the second DiT block (Eq. (2)), and the
memory frames ``M_n`` and the recent context ``V_<n`` are frames of the block's window, each with
its ray embedding (Eq. (3)). :meth:`WorldCastGenerator.forward` runs one range of the window
(first frame, memory frames, recent context or target frames) on the
:class:`~worldcast.modeling.wan22.dit.KVCache`; the training forward over the whole window is
:mod:`worldcast.modeling.wan22.training`. Precision: docs/inference.md, "Numerics".
"""

import functools
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.latents import FRAME_TOKENS, LATENT_CHANNELS
from worldcast.modeling.controls import CONTROL_KEYS, ControlConfig, ControlEmbedding
from worldcast.modeling.observer_signals import (
    ObserverSignalConfig,
    ObserverSignalEmbedding,
    ObserverSignals,
)
from worldcast.modeling.ray_embedding import RAY_CONDITION_KEYS, RayConditions, RayEmbedding
from worldcast.modeling.state_injector import StateInjector, StateInjectorConfig
from worldcast.modeling.visibility_probe import VisibilityProbe, VisibilityProbeConfig
from worldcast.utils.precision import cast_floating_tensors, fp32_island, generator_autocast

from .attention import AttentionFn, flash_attention
from .dit import (
    DIT_BLOCK_ADALN_VECTORS,
    CausalDiTBlock,
    CausalHead,
    DiTBlockParts,
    KVCache,
    causal_self_attention,
    rope_frequencies,
    rope_table,
    sinusoidal_embedding_1d,
)
from .text_encoder import TEXT_DIM, TEXT_LEN

__all__ = [
    "PATCH_SIZE",
    "WORLDCAST_MODULES",
    "CallPlan",
    "CallPrologue",
    "CausalGeneratorAdapter",
    "FieldBuilder",
    "GeneratorConditions",
    "GeneratorConfig",
    "PlayerStateFieldFn",
    "WorldCastGenerator",
    "call_inputs",
    "coarsen_field",
    "conditions_with_field",
    "worldcast_module",
]

#: The patches of a latent frame: one latent frame of 2 x 2 latent cells per token.
PATCH_SIZE = (1, 2, 2)
#: The modules WorldCast adds to the Wan2.2 backbone, by the root of their state-dict keys: the
#: conditioning and, in training, the visibility probe.
WORLDCAST_MODULES = (
    "controls",
    "state_injector",
    "observer_signals",
    "ray_embedding",
    "visibility_probe",
)

#: ``fn(frame_offset, num_frames) -> field``: the player state field ``[B, num_frames, 23, h, w]``
#: of the latent frames of a call, called inside the generator's autocast.
PlayerStateFieldFn = Callable[[int, int], torch.Tensor]


def worldcast_module(key: str) -> str | None:
    """The module of :data:`WORLDCAST_MODULES` a state-dict key of the generator belongs to
    (``None``: the backbone); the control adapters of the DiT blocks and of the head belong to
    ``controls``."""
    if ".control_adaln." in key:
        return "controls"
    root = key.split(".", 1)[0]
    return root if root in WORLDCAST_MODULES else None


class FieldBuilder(Protocol):
    """Builds the player state field of a generator call from the condition dict, for example
    ``worldcast.player_state.field.field_builder()``.

    Attributes:
        condition_keys (tuple[str, ...]): the entries of the condition dict the builder reads; the
            generator leaves them to it.
    """

    condition_keys: tuple[str, ...]

    def __call__(
        self,
        conditions: Mapping[str, Any],
        weapon_embedding: torch.Tensor,
        frame_offset: int,
        num_frames: int,
    ) -> torch.Tensor:
        """The field ``[B, num_frames, 23, h, w]`` of the latent frames ``[frame_offset,
        frame_offset + num_frames)`` of the window; ``weapon_embedding`` is the injector's ``[52,
        4]`` table."""
        ...


# =========================================================================================== config
@dataclass(frozen=True)
class GeneratorConfig:
    """Architecture of the generator; the defaults are the paper's.

    The backbone fields are those of the Wan2.2-TI2V-5B ``config.json``
    (:func:`worldcast.modeling.build.generator_config_from_snapshot`).
    """

    in_dim: int = LATENT_CHANNELS
    out_dim: int = LATENT_CHANNELS
    dim: int = 3072
    ffn_dim: int = 14336
    freq_dim: int = 256
    text_dim: int = TEXT_DIM
    text_len: int = TEXT_LEN
    num_heads: int = 24
    num_layers: int = 30
    eps: float = 1e-6
    controls: ControlConfig = field(default_factory=ControlConfig)
    #: The player state field's injector (``None``: a generator without the field, as in stage 1).
    state_injector: StateInjectorConfig | None = field(default_factory=StateInjectorConfig)
    #: The observer-signal embedding (``None``: a generator without it; the stages with scene
    #: state, 2s, 3 and 4, build it).
    observer_signals: ObserverSignalConfig | None = field(default_factory=ObserverSignalConfig)
    #: The ray embedding of every frame (built by the stages with scene state); it has no settings.
    ray_embedding: bool = True
    #: Coarse-field ablation: max-pool the field by this factor and copy it back (paper: 1). The
    #: late-injection ablation is ``state_injector.dit_block``.
    field_downsample: int = 1
    #: The visibility probe (training only: the stages with scene state build it, stages 2s and 3
    #: train it; ``None`` at inference).
    visibility_probe: VisibilityProbeConfig | None = None

    def __post_init__(self) -> None:
        if self.dim % self.num_heads or (self.dim // self.num_heads) % 2:
            raise ValueError("dim must split into an even head_dim")
        if self.freq_dim % 2:
            raise ValueError(f"freq_dim must be even ([cos | sin] halves), got {self.freq_dim}")
        injector = 0 if self.state_injector is None else self.state_injector.dit_block
        if not 0 <= injector < self.num_layers:
            raise ValueError("state_injector.dit_block must index a DiT block")
        if not isinstance(self.field_downsample, int) or self.field_downsample < 1:
            raise ValueError("field_downsample must be a positive integer (1 = the paper's field)")
        probe = self.visibility_probe
        if probe is not None and not injector <= probe.dit_block < self.num_layers:
            raise ValueError(
                f"visibility_probe.dit_block {probe.dit_block} must lie in"
                f" [{injector}, {self.num_layers})"
            )


# ======================================================================================= conditions
@dataclass(eq=False)
class GeneratorConditions:
    """What one generator call is conditioned on, besides the latents, timesteps and cache.

    The tensors span the window; the generator reads the call's frames. ``T = 1 + 4 (F_window - 1)``
    video frames. In a condition dict the prompt and the controls have the names of their
    attributes; the observer signals are the entries
    ``worldcast.data.labels.OBSERVER_SIGNAL_KEYS`` and the cameras the entries
    :data:`~worldcast.modeling.ray_embedding.RAY_CONDITION_KEYS`.

    Attributes:
        prompt_embeds (Tensor | Sequence[Tensor]): ``[B, L <= 512, 4096]`` umT5 embedding of the
            fixed prompt, or ``B`` tensors ``[L_b, 4096]``.
        buttons (Tensor): ``[B, T, 11]`` the client's buttons.
        view_deltas (Tensor): ``[B, T, 2]`` the client's pitch and yaw deltas.
        weapon (Tensor): ``[B, T]`` long, the client's held weapon.
        player_state_field (Tensor | PlayerStateFieldFn | None): the player state field of the
            window ``[B, F_window, 23, 12, 21]``, or a :data:`PlayerStateFieldFn` that gives the
            field of a call's frames; ``None`` for a generator without the state injector.
        observer_signals (ObserverSignals | None): the window's ``[B, F_window]`` observer
            signals; ``None`` for a generator without the observer-signal embedding.
        rays (RayConditions | None): the window's cameras, or ``None`` for no ray embedding.
    """

    prompt_embeds: torch.Tensor | Sequence[torch.Tensor]
    buttons: torch.Tensor
    view_deltas: torch.Tensor
    weapon: torch.Tensor
    player_state_field: torch.Tensor | PlayerStateFieldFn | None = None
    observer_signals: ObserverSignals | None = None
    rays: RayConditions | None = None

    @classmethod
    def from_conditions(
        cls,
        conditions: Mapping[str, Any],
        *,
        player_state_field: torch.Tensor | PlayerStateFieldFn | None = None,
        field_keys: Collection[str] = (),
    ) -> "GeneratorConditions":
        """From a sampler or trainer condition dict.

        Args:
            conditions (Mapping[str, Any]): the condition dict.
            player_state_field (Tensor | PlayerStateFieldFn | None): the field, which is not an
                entry of the dict.
            field_keys (Collection[str]): the entries a field builder reads
                (:attr:`FieldBuilder.condition_keys`); any other entry the generator does not
                read is an error.
        """
        required = ("prompt_embeds", *CONTROL_KEYS)
        known = (*required, *OBSERVER_SIGNAL_KEYS, *RAY_CONDITION_KEYS, *field_keys)
        unknown = sorted(key for key in conditions if key not in known)
        if unknown:
            raise ValueError(f"condition keys with no consumer in the generator: {unknown}")
        missing = sorted(key for key in required if conditions.get(key) is None)
        if missing:
            raise ValueError(f"conditions are missing {missing}")
        return cls(
            **{key: conditions[key] for key in required},
            player_state_field=player_state_field,
            observer_signals=ObserverSignals.from_conditions(conditions),
            rays=RayConditions.from_conditions(conditions),
        )

    def call_field(self, frame_offset: int, num_frames: int) -> torch.Tensor:
        """The player state field ``[B, num_frames, 23, h, w]`` of a call's latent frames
        ``[frame_offset, frame_offset + num_frames)``: the field function's, or the window's
        field cut to them."""
        if callable(self.player_state_field):
            return self.player_state_field(frame_offset, num_frames)
        return self.player_state_field[:, frame_offset : frame_offset + num_frames]


def conditions_with_field(
    generator: "WorldCastGenerator",
    field_builder: FieldBuilder | None,
    conditions: Mapping[str, Any],
) -> GeneratorConditions:
    """A condition dict as :class:`GeneratorConditions`, with the :data:`PlayerStateFieldFn`
    of its player state field.

    Args:
        generator (WorldCastGenerator): the generator the conditions are for; with a state
            injector, the field is built from the dict and the injector's weapon embedding when
            the generator asks for it.
        field_builder (FieldBuilder | None): builds the field; ``None`` for a generator without
            the state injector.
        conditions (Mapping[str, Any]): the condition dict.
    """
    if field_builder is None:
        if generator.state_injector is not None:
            raise ValueError("a generator with a state injector needs a field builder")
        return GeneratorConditions.from_conditions(conditions)

    def player_state_field(frame_offset: int, num_frames: int) -> torch.Tensor:
        weapon_embedding = generator.state_injector.weapon_embedding.weight
        return field_builder(conditions, weapon_embedding, frame_offset, num_frames)

    return GeneratorConditions.from_conditions(
        conditions,
        player_state_field=None if generator.state_injector is None else player_state_field,
        field_keys=field_builder.condition_keys,
    )


def call_inputs(
    generator: "WorldCastGenerator",
    field_builder: FieldBuilder | None,
    input_dtype: torch.dtype | None,
    conditions: Mapping[str, Any],
    *inputs: Any,
) -> tuple[Any, ...]:
    """The inputs of a generator call as the generator sees them.

    Args:
        generator (WorldCastGenerator): the generator.
        field_builder (FieldBuilder | None): builds its player state field
            (:func:`conditions_with_field`).
        input_dtype (torch.dtype | None): every floating tensor of the conditions and of
            ``inputs`` is cast to it (bf16 on the paper path; ``worldcast.utils.precision``), and
            the field is built from the cast conditions; ``None``: no cast.
        conditions (Mapping[str, Any]): the condition dict.
        *inputs (Any): the call's other inputs: tensors, or containers of them.

    Returns:
        tuple: the :class:`GeneratorConditions`, then ``inputs``.
    """
    if input_dtype is not None:
        conditions, *inputs = cast_floating_tensors((dict(conditions), *inputs), input_dtype)
    return conditions_with_field(generator, field_builder, conditions), *inputs


# ========================================================================================= one call
@dataclass(frozen=True)
class CallPlan:
    """Where one generator call sits in the window and in the KV cache.

    Attributes:
        grid (tuple[int, int, int]): token grid ``(F, h, w)`` of the call.
        frame_offset (int): window index of the call's first latent frame.
        start (int): token index of the cache write, ``frame_offset h w``.
        end (int): end of the write (:meth:`KVCache.span`).
    """

    grid: tuple[int, int, int]
    frame_offset: int
    start: int
    end: int

    @classmethod
    def at(cls, grid: tuple[int, int, int], frame_offset: int, kv_cache: KVCache) -> "CallPlan":
        """The place of a call, checked against the cache.

        Args:
            grid (tuple[int, int, int]): token grid ``(F, h, w)`` of the call.
            frame_offset (int): window index of the call's first latent frame.
            kv_cache (KVCache): the cache the call writes; a write past its end must start there
                (:meth:`KVCache.span`).
        """
        frames, height, width = grid
        start = frame_offset * height * width
        end = kv_cache.span(start, frames * height * width)
        return cls(grid=tuple(grid), frame_offset=frame_offset, start=start, end=end)


@dataclass(eq=False)
class CallPrologue:
    """What one call computes from its conditions before the DiT blocks; the denoising steps of a
    block share it.

    Attributes:
        signal_embedding (Tensor | None): ``[B, F h w, dim]`` observer-signal embedding of every
            token.
        ray_embedding (Tensor | None): ``[B, F h w, dim]`` ray embedding of every token (Eq. (3)).
        control_embedding (Tensor): ``[B, F, dim]`` control embedding (Eq. (1)).
        field (Tensor | None): ``[B, F, 23, h, w]`` the player state field of the call's frames
            (Eq. (2)); ``None`` without the injector.
    """

    signal_embedding: torch.Tensor | None
    ray_embedding: torch.Tensor | None
    control_embedding: torch.Tensor
    field: torch.Tensor | None

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens`` plus the observer-signal embedding, then plus the ray embedding."""
        if self.signal_embedding is not None:
            tokens = tokens + self.signal_embedding
        if self.ray_embedding is not None:
            tokens = tokens + self.ray_embedding
        return tokens


# =================================================================================== field ablation
def coarsen_field(field: torch.Tensor, factor: int) -> torch.Tensor:
    """The coarse-field ablation: max-pool ``factor x factor`` cells, copy each maximum back.

    ``ceil_mode`` keeps the last, narrower cell of the 21-token width. ``[B, F, C, h, w]`` -> the
    same shape and dtype. The copy-back is nearest interpolation: ``repeat_interleave`` gives the
    same field but sums the gradient in another order, which changes the training numbers.
    """
    if factor == 1:
        return field
    batch, frames, channels, grid_h, grid_w = field.shape
    flat = field.reshape(batch * frames, channels, grid_h, grid_w)
    pooled = F.max_pool2d(flat, kernel_size=factor, stride=factor, ceil_mode=True)
    upsampled = F.interpolate(pooled, size=(grid_h, grid_w), mode="nearest")
    return upsampled.reshape(batch, frames, channels, grid_h, grid_w)


# ======================================================================================== generator
def _init_backbone_weights(generator: "WorldCastGenerator") -> None:
    """Wan2.2's initialisation of the backbone (the conditioning initialises itself)."""
    for module in generator.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    nn.init.xavier_uniform_(generator.patch_embedding.weight.flatten(1))
    embeddings = (*generator.text_embedding.modules(), *generator.time_embedding.modules())
    for module in embeddings:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
    nn.init.zeros_(generator.head.head.weight)


def _check_latents(x: torch.Tensor, patch_weight: torch.Tensor) -> None:
    """The latents a patch embedding of weight ``[dim, C, 1, 2, 2]`` takes: ``[B, F, C, H, W]``
    with ``H`` and ``W`` that its patches tile, and off CUDA, where nothing casts the inputs, in
    the dtype of the parameters."""
    in_dim, _, patch_h, patch_w = patch_weight.shape[1:]
    channels = x.shape[2] if x.ndim == 5 else None
    if channels != in_dim or x.shape[3] % patch_h or x.shape[4] % patch_w:
        raise ValueError(
            f"the latents must be [B, F, {in_dim}, H, W] with H and W divisible by {patch_h} and"
            f" {patch_w}, got {tuple(x.shape)}"
        )
    if not x.is_cuda and x.dtype != patch_weight.dtype:
        raise ValueError(
            f"off CUDA nothing casts the inputs: the latents must be {patch_weight.dtype} like"
            f" the parameters, got {x.dtype}"
        )


def _padded_prompts(
    prompt_embeds: torch.Tensor | Sequence[torch.Tensor], text_len: int, text_dim: int
) -> torch.Tensor:
    """``B`` prompts ``[L_b <= text_len, text_dim]`` -> ``[B, text_len, text_dim]``, each
    zero-padded."""
    shapes = [tuple(prompt.shape) for prompt in prompt_embeds]
    if any(len(shape) != 2 or shape[0] > text_len or shape[1] != text_dim for shape in shapes):
        raise ValueError(f"every prompt must be [L <= {text_len}, {text_dim}], got {shapes}")
    return torch.stack(
        [
            torch.cat([prompt, prompt.new_zeros(text_len - len(prompt), text_dim)])
            for prompt in prompt_embeds
        ]
    )


def _check_conditions(
    generator: "WorldCastGenerator", cond: GeneratorConditions, last_frame: int
) -> None:
    """A condition is an error where the generator has no module for it, and the observer signals
    and the field where it has the module and they are missing; what spans the window must reach
    the call's last frame, window index ``last_frame``, and every condition holds the samples of
    the controls."""
    if not isinstance(cond, GeneratorConditions):
        raise TypeError(
            "the generator takes GeneratorConditions (GeneratorConditions.from_conditions of a"
            f" condition dict), got {type(cond).__name__}"
        )
    signals, field = cond.observer_signals, cond.player_state_field
    if generator.observer_signals is None and signals is not None:
        raise ValueError("the conditions carry observer signals: the generator embeds none")
    if generator.ray_embedding is None and cond.rays is not None:
        raise ValueError("the conditions carry cameras: the generator has no ray embedding")
    if generator.state_injector is None and field is not None:
        raise ValueError("the conditions carry a field: the generator has no state injector")
    if generator.observer_signals is not None and signals is None:
        raise ValueError(
            f"the generator embeds the observer signals: the conditions need {OBSERVER_SIGNAL_KEYS}"
        )
    if generator.state_injector is not None and field is None:
        raise ValueError("the generator has a state injector: the conditions have no field")
    window = {} if signals is None else {"the observer signals": signals.flash_flag}
    if torch.is_tensor(field):
        window["the field"] = field
    for name, value in window.items():
        if value.ndim < 2 or value.shape[1] <= last_frame:
            raise ValueError(
                f"{name} must span the window, [B, F_window, ...] up to latent frame"
                f" {last_frame} of the call, got {tuple(value.shape)}"
            )
    # a condition of one sample would broadcast over the others
    samples = {**window, "prompt_embeds": cond.prompt_embeds}
    if cond.rays is not None:  # their span is the ray embedding's check
        samples["the cameras"] = cond.rays.frame_c2w
    batch_size = len(cond.buttons)
    other = {name: len(value) for name, value in samples.items() if len(value) != batch_size}
    if other:
        raise ValueError(f"the controls hold {batch_size} samples; other counts: {other}")


class WorldCastGenerator(nn.Module):
    """``G_theta``: the causal Wan2.2-TI2V-5B DiT with the WorldCast conditioning.

    :meth:`forward` runs one range of the window on the KV cache (inference, the stage-4 rollout):
    :meth:`patchify`, :meth:`CallPlan.at`, :meth:`prologue`, :meth:`embed_timesteps`,
    :meth:`run_dit_blocks_on_cache` and :meth:`unpatchify`, which the client's fast path calls too.
    The training forward runs the same DiT-block loop, :meth:`run_dit_blocks`, under an attention
    mask.
    Checkpoint keys: the Wan2.2 backbone's, plus those of the modules of
    :data:`WORLDCAST_MODULES` the config builds.

    Args:
        config (GeneratorConfig): the architecture.
        attention (AttentionFn | None): the full-attention kernel (default :func:`flash_attention`,
            which needs CUDA).

    Attributes:
        attention (AttentionFn): the full-attention kernel; assign another to replace it.
        gradient_checkpointing (bool): recompute each DiT block in backward (the training forward
            only); off by default.
    """

    def __init__(
        self, config: GeneratorConfig = GeneratorConfig(), *, attention: AttentionFn | None = None
    ) -> None:
        super().__init__()
        if attention is not None and not callable(attention):
            raise TypeError(
                "attention must be a kernel fn(q, k, v), for example attention_kernel(name,"
                f" device), got {attention!r}"
            )
        self.config = config
        dim = config.dim
        self.dim = dim
        self.num_heads = config.num_heads

        self.patch_embedding = nn.Conv3d(
            config.in_dim, dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * DIT_BLOCK_ADALN_VECTORS)
        )
        self.blocks = nn.ModuleList(
            CausalDiTBlock(dim, config.ffn_dim, config.num_heads, config.eps)
            for _ in range(config.num_layers)
        )
        self.head = CausalHead(dim, config.out_dim, PATCH_SIZE, config.eps)
        _init_backbone_weights(self)

        # a plain complex128 attribute, never cast with the parameters
        self.freqs = rope_frequencies(dim // config.num_heads)

        # The conditioning starts at zero (the control adapters, the injector's ``proj``, the
        # observer signals' ``out`` and the ray MLP's last layer are zero-initialised); it is built
        # in this order so that a fresh initialisation draws the same random numbers as the
        # paper's training runs.
        self.controls = ControlEmbedding(dim, config.controls)
        for block in self.blocks:
            block.attach_control_adaln(config.controls.adaln_rank)
        self.head.attach_control_adaln(config.controls.adaln_rank)
        self.state_injector: StateInjector | None = None
        if config.state_injector is not None:
            self.state_injector = StateInjector(dim, config.state_injector)
        self.visibility_probe: VisibilityProbe | None = None
        if config.visibility_probe is not None:
            self.visibility_probe = VisibilityProbe(dim, config.visibility_probe)
        self.observer_signals: ObserverSignalEmbedding | None = None
        if config.observer_signals is not None:
            self.observer_signals = ObserverSignalEmbedding(dim, config.observer_signals)
        self.ray_embedding: RayEmbedding | None = None
        if config.ray_embedding:
            self.ray_embedding = RayEmbedding(dim)

        self.attention: AttentionFn = attention or flash_attention
        self.gradient_checkpointing = False

    def allocate_kv_cache(
        self,
        latent_frames: int,
        *,
        frame_tokens: int = FRAME_TOKENS,
        batch_size: int = 1,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> KVCache:
        """A zeroed KV cache of the generator's self-attention.

        Args:
            latent_frames (int): latent frames the cache holds: every frame a rollout writes.
            frame_tokens (int): tokens per latent frame (252 for the paper's latents).
            batch_size (int): rollouts in a batch.
            dtype (torch.dtype | None): dtype of the keys and values; default: the parameters'.
                A generator whose parameters stay float32 under an autocast (FSDP mixed
                precision) passes the autocast dtype.
            device (torch.device | str | None): default: the parameters'.
        """
        parameter = self.patch_embedding.weight
        return KVCache.allocate(
            num_dit_blocks=len(self.blocks),
            num_heads=self.num_heads,
            head_dim=self.dim // self.num_heads,
            latent_frames=latent_frames,
            frame_tokens=frame_tokens,
            dtype=parameter.dtype if dtype is None else dtype,
            batch_size=batch_size,
            device=parameter.device if device is None else device,
        )

    # ----------------------------------------------------------------------------- pieces of a call
    def patchify(
        self, x: torch.Tensor, embed: Callable[[torch.Tensor], torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Patch-embed latents, one sample at a time (a batched conv can round differently).

        Args:
            x (Tensor): ``[B, F, C, H, W]`` latents, ``H`` and ``W`` even.
            embed (Callable | None): replaces the ``patch_embedding`` conv on ``[1, C, F, H, W]``
                (the same map).

        Returns:
            tuple: tokens ``[B, F h w, dim]`` and the token grid ``(F, h, w)``.
        """
        _check_latents(x, self.patch_embedding.weight)
        embed = embed or self.patch_embedding
        patches = [embed(sample.unsqueeze(0)) for sample in x.permute(0, 2, 1, 3, 4)]
        tokens = torch.cat([p.flatten(2).transpose(1, 2) for p in patches])
        return tokens, tuple(patches[0].shape[2:])

    def unpatchify(self, tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """The head's output ``[B, F h w, prod(patch) * C]`` -> ``[B, F, C, H, W]`` float32."""
        p_t, p_h, p_w = PATCH_SIZE
        frames, height, width = grid
        channels = self.config.out_dim
        x = tokens.view(tokens.shape[0], frames, height, width, p_t, p_h, p_w, channels)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(tokens.shape[0], channels, frames * p_t, height * p_h, width * p_w)
        return x.float().permute(0, 2, 1, 3, 4)

    def rope_table(self, grid: tuple[int, int, int], frame_offset: int) -> torch.Tensor:
        """The call's :func:`rope_table` on the generator's device."""
        self.freqs = self.freqs.to(self.patch_embedding.weight.device)
        return rope_table(self.freqs, grid, frame_offset)

    def embed_timesteps(
        self, timestep: torch.Tensor, num_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The timesteps ``[B, num_frames]`` of a call's latent frames -> their embedding
        ``[B, num_frames, dim]`` and DiT-block modulation ``[B, num_frames, 6, dim]``, both
        float32."""
        if timestep.ndim != 2 or timestep.shape[1] != num_frames:
            raise ValueError(
                f"timestep must be [B, {num_frames}], one per latent frame, got"
                f" {tuple(timestep.shape)}"
            )
        with fp32_island():
            sinusoid = sinusoidal_embedding_1d(self.config.freq_dim, timestep.flatten())
            embedding = self.time_embedding(sinusoid.unflatten(0, timestep.shape).float())
            modulation = self.time_projection(embedding).unflatten(2, (-1, self.dim))
        return embedding, modulation

    def embed_text(self, prompt_embeds: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """The prompt zero-padded to 512 tokens and embedded: ``[B, 512, dim]``."""
        padded = _padded_prompts(prompt_embeds, self.config.text_len, self.config.text_dim)
        return self.text_embedding(padded)

    def prologue(
        self,
        cond: GeneratorConditions,
        grid: tuple[int, int, int],
        frame_offset: int,
        dtype: torch.dtype,
    ) -> CallPrologue:
        """What a call computes from its conditions before the DiT blocks.

        A condition without its module, and a module without its condition, are errors; only the
        cameras may be absent for a generator with a ray embedding, which then adds none.

        Args:
            cond (GeneratorConditions): the window's conditions.
            grid (tuple[int, int, int]): token grid ``(F, h, w)`` of the call.
            frame_offset (int): window index of the call's first latent frame.
            dtype (torch.dtype): dtype of the call's tokens.

        Returns:
            CallPrologue: the observer-signal, ray and control embeddings and the player state
            field (coarse under the coarse-field ablation) of the call's ``F`` latent frames.
        """
        frames, height, width = grid
        _check_conditions(self, cond, frame_offset + frames - 1)
        device = self.patch_embedding.weight.device
        call = dict(frame_offset=frame_offset, num_frames=frames)
        signal_embedding = ray_embedding = call_field = None
        if self.observer_signals is not None:
            # the whole window is embedded and the call's frames are cut out of it, as trained
            window = self.observer_signals(cond.observer_signals.to(device))
            window = window.to(device=device, dtype=dtype)[:, frame_offset : frame_offset + frames]
            signal_embedding = window.repeat_interleave(height * width, dim=1)
        if cond.rays is not None:
            ray_embedding = self.ray_embedding(cond.rays, grid=(height, width), dtype=dtype, **call)
        control_embedding = self.controls(
            cond.buttons.to(device=device, dtype=dtype),
            cond.view_deltas.to(device=device, dtype=dtype),
            cond.weapon.to(device=device),
            **call,
        ).to(dtype)
        if self.state_injector is not None:
            call_field = cond.call_field(frame_offset, frames)
            call_field = coarsen_field(call_field, self.config.field_downsample).to(device=device)
        return CallPrologue(
            signal_embedding=signal_embedding,
            ray_embedding=ray_embedding,
            control_embedding=control_embedding,
            field=call_field,
        )

    # ------------------------------------------------------------------------------- the DiT blocks
    def run_dit_blocks(
        self,
        tokens: torch.Tensor,
        *,
        timestep_modulation: torch.Tensor,
        control_embedding: torch.Tensor,
        text: torch.Tensor | Sequence[tuple[torch.Tensor, torch.Tensor]],
        self_attention: Callable[[int], AttentionFn],
        inject: Callable[[torch.Tensor], torch.Tensor] | None,
        probe: Callable[[torch.Tensor], None] | None = None,
        attention: AttentionFn | None = None,
        parts: DiTBlockParts | None = None,
        checkpoint: bool = False,
    ) -> torch.Tensor:
        """The DiT blocks of one call: the one loop of the KV-cache forward and of the training
        forward. After the state injector's DiT block ``inject`` adds the player state field (Eq.
        (2)); after the visibility probe's, ``probe`` reads the tokens.

        Args:
            tokens (Tensor): ``[B, L, dim]`` embedded tokens (:meth:`CallPrologue.embed`).
            timestep_modulation (Tensor): ``[B, F, 6, dim]`` float32 (:meth:`embed_timesteps`).
            control_embedding (Tensor): ``[B, F, dim]`` control embedding.
            text (Tensor | Sequence[tuple[Tensor, Tensor]]): ``[B, 512, dim]`` the embedded
                prompt (:meth:`embed_text`), or every DiT block's cross-attention keys and values
                of it.
            self_attention (Callable[[int], AttentionFn]): the self-attention ``fn(q, k, v)`` of
                the DiT block of an index (on the cache, or under the training mask).
            inject (Callable[[Tensor], Tensor] | None): ``tokens -> tokens`` with the field
                added; ``None`` for a generator without the state injector.
            probe (Callable[[Tensor], None] | None): reads the tokens for the visibility probe.
            attention (AttentionFn | None): the kernel of the cross-attention (default the
                generator's).
            parts (DiTBlockParts | None): the DiT blocks' parts (:meth:`CausalDiTBlock.forward`).
            checkpoint (bool): recompute each DiT block in backward.

        Returns:
            Tensor: ``[B, L, dim]``, the tokens after the last DiT block.
        """
        attention = attention or self.attention
        if not len(tokens) == len(timestep_modulation) == len(control_embedding):
            raise ValueError(
                f"the latents hold {len(tokens)} samples, the timesteps"
                f" {len(timestep_modulation)} and the controls {len(control_embedding)}"
            )
        # Attribution: handing each DiT block its cache entry and the call's token offsets follows
        # CausVid's causal forward (github.com/tianweiy/CausVid at fab2440f, MIT) via Self Forcing
        # (github.com/guandeh17/Self-Forcing, Apache-2.0); the rest is Wan2.2's WanModel.forward
        # (Apache-2.0).
        for index, block in enumerate(self.blocks):
            block_kwargs = dict(
                timestep_modulation=timestep_modulation,
                control_embedding=control_embedding,
                text=text if torch.is_tensor(text) else text[index],
                self_attention=self_attention(index),
                attention=attention,
                parts=parts,
            )
            if checkpoint:
                tokens = torch.utils.checkpoint.checkpoint(
                    block, tokens, use_reentrant=False, **block_kwargs
                )
            else:
                tokens = block(tokens, **block_kwargs)
            if inject is not None and index == self.state_injector.dit_block:
                tokens = inject(tokens)
            if probe is not None and index == self.visibility_probe.dit_block:
                probe(tokens)
        return tokens

    def run_dit_blocks_on_cache(
        self,
        tokens: torch.Tensor,
        *,
        timestep_embedding: torch.Tensor,
        timestep_modulation: torch.Tensor,
        control_embedding: torch.Tensor,
        field: torch.Tensor | None,
        plan: CallPlan,
        kv_cache: KVCache,
        rope: torch.Tensor,
        text: torch.Tensor | Sequence[tuple[torch.Tensor, torch.Tensor]],
        attention: AttentionFn | None = None,
        parts: DiTBlockParts | None = None,
    ) -> torch.Tensor:
        """The DiT blocks of one call on the KV cache, the player state field added after the
        state injector's DiT block (Eq. (2)), then the head.

        The caller then moves the cache's end, ``kv_cache.end = plan.end``, as :meth:`forward`
        does: a captured CUDA graph of this method replays its kernels only.

        Args:
            tokens (Tensor): ``[B, F h w, dim]`` embedded tokens (:meth:`CallPrologue.embed`).
            timestep_embedding (Tensor): ``[B, F, dim]`` float32 (:meth:`embed_timesteps`).
            timestep_modulation (Tensor): ``[B, F, 6, dim]`` float32.
            control_embedding (Tensor): ``[B, F, dim]`` control embedding.
            field (Tensor | None): the player state field of the call (:class:`CallPrologue`).
            plan (CallPlan): the call's place; the cache is written at ``[start, end)``.
            kv_cache (KVCache): the KV cache.
            rope (Tensor): the call's :meth:`rope_table`.
            text (Tensor | Sequence[tuple[Tensor, Tensor]]): ``[B, 512, dim]`` the embedded
                prompt, or every DiT block's cross-attention keys and values of it.
            attention (AttentionFn | None): the kernel (default the generator's).
            parts (DiTBlockParts | None): the DiT blocks' parts (:meth:`CausalDiTBlock.forward`).

        Returns:
            Tensor: ``[B, F h w, prod(patch) * out_dim]``, the head's output (:meth:`unpatchify`).
        """
        attention = attention or self.attention

        def self_attention(index: int) -> AttentionFn:
            return functools.partial(
                causal_self_attention,
                rope=rope,
                cache=kv_cache,
                dit_block=index,
                start=plan.start,
                end=plan.end,
                attention=attention,
            )

        inject = None
        if self.state_injector is not None:
            inject = functools.partial(self.state_injector, field=field)
        tokens = self.run_dit_blocks(
            tokens,
            timestep_modulation=timestep_modulation,
            control_embedding=control_embedding,
            text=text,
            self_attention=self_attention,
            inject=inject,
            attention=attention,
            parts=parts,
        )
        return self.head(tokens, timestep_embedding, control_embedding)

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        cond: GeneratorConditions,
        *,
        kv_cache: KVCache,
        frame_offset: int,
    ) -> torch.Tensor:
        """One generator call on one range of the window.

        Args:
            noisy (Tensor): ``[B, F, 48, 24, 42]`` the call's noisy latent frames.
            timestep (Tensor): ``[B, F]`` per-frame timesteps in ``[0, 1000]``.
            cond (GeneratorConditions): the window's conditions.
            kv_cache (KVCache): the call's keys and values are written at its frames' tokens;
                the call then moves ``kv_cache.end`` to the end of its write.
            frame_offset (int): window index of the call's first latent frame: the place of the
                cache write and the position of RoPE and of the control history.

        Returns:
            Tensor: flow prediction ``[B, F, 48, 24, 42]`` float32.
        """
        with generator_autocast(noisy):
            tokens, grid = self.patchify(noisy)
            plan = CallPlan.at(grid, frame_offset, kv_cache)
            prologue = self.prologue(cond, grid, frame_offset, tokens.dtype)
            timestep_embedding, timestep_modulation = self.embed_timesteps(timestep, grid[0])
            out = self.run_dit_blocks_on_cache(
                prologue.embed(tokens),
                timestep_embedding=timestep_embedding,
                timestep_modulation=timestep_modulation,
                control_embedding=prologue.control_embedding,
                field=prologue.field,
                plan=plan,
                kv_cache=kv_cache,
                rope=self.rope_table(grid, frame_offset),
                text=self.embed_text(cond.prompt_embeds),
            )
            kv_cache.end = plan.end
            return self.unpatchify(out, grid)


# ========================================================================== the sampler's generator
class CausalGeneratorAdapter:
    """The sampler's generator (:class:`worldcast.sampling.sampler.CausalGenerator`) on top of
    :class:`WorldCastGenerator`.

    Per call: the inputs are cast and the conditions given their field function
    (:func:`call_inputs`), and :meth:`WorldCastGenerator.forward` runs on the sampler's cache.

    Args:
        generator (WorldCastGenerator): the loaded generator.
        field_builder (FieldBuilder | None): builds the player state field, for example
            ``worldcast.player_state.field.field_builder()``; ``None`` for a generator without the
            state injector.
        input_dtype (torch.dtype | None): the dtype every floating input is cast to: bf16
            (``GENERATOR_DTYPE``) on the paper path; ``None`` (no cast) for a float32 generator on
            the CPU.
    """

    def __init__(
        self,
        generator: WorldCastGenerator,
        field_builder: FieldBuilder | None = None,
        *,
        input_dtype: torch.dtype | None = None,
    ) -> None:
        self.generator = generator
        self.field_builder = field_builder
        self.input_dtype = input_dtype

    def __call__(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        kv_cache: KVCache,
        frame_offset: int,
    ) -> torch.Tensor:
        """Flow ``[B, F, C, H, W]`` float32 of one call (:meth:`WorldCastGenerator.forward`)."""
        cond, noisy, timestep = call_inputs(
            self.generator, self.field_builder, self.input_dtype, conditions, noisy, timestep
        )
        return self.generator(noisy, timestep, cond, kv_cache=kv_cache, frame_offset=frame_offset)
