"""The generator's training forward: the whole window under an attention mask.

Stages 1, 2 and 2s attend bidirectionally. Stage 3 converts the generator to block-causal attention
through teacher forcing: the sequence is ``[context | noisy]``, and a noisy token of a block attends
to the context tokens of the earlier blocks and to the noisy tokens of its own
(:class:`~worldcast.modeling.wan22.attention.MaskLayout`). Stage 4 runs the same forward for its
teacher, its critic and the replay of its rollout.
"""

import functools
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from worldcast.modeling.state_injector import StateInjector
from worldcast.modeling.visibility_probe import VisibilityProbeInputs
from worldcast.utils.precision import generator_autocast

from .attention import MaskLayout, TrainingMask, masked_attention, training_mask
from .dit import apply_rope
from .model import FieldBuilder, GeneratorConditions, WorldCastGenerator, call_inputs

__all__ = ["TrainingForward", "forward_train"]


def _copies(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(context, noisy)`` copies of teacher forcing's ``[context | noisy]`` tokens: two
    slices, as trained (their gradients are summed as two terms)."""
    half = tokens.shape[1] // 2
    return tokens[:, :half], tokens[:, half:]


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


def _inject_field(
    tokens: torch.Tensor,
    *,
    injector: StateInjector,
    field: torch.Tensor,
    teacher_forcing: bool,
    field_on_context: bool,
) -> torch.Tensor:
    """Eq. (2) on the whole window; under teacher forcing on the noisy copy only, or on both
    copies with ``field_on_context``."""
    if not teacher_forcing:
        return injector(tokens, field)
    if field_on_context:
        return injector(tokens, field, copies=2)
    context, noisy = _copies(tokens)
    return torch.cat([context, injector(noisy, field)], dim=1)


def _run_dit_blocks_under_mask(
    generator: WorldCastGenerator,
    tokens: torch.Tensor,
    grid: tuple[int, int, int],
    *,
    timestep_modulation: torch.Tensor,
    control_embedding: torch.Tensor,
    text: torch.Tensor,
    field: torch.Tensor | None,
    visibility: VisibilityProbeInputs | None,
    teacher_forcing: bool,
    field_on_context: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The DiT blocks on the whole window (:meth:`WorldCastGenerator.run_dit_blocks`) with the
    masked self-attention, the field injection of the window and the visibility probe, which reads
    the noisy copy; returns the tokens and the probe's logits (``None`` without ``visibility``)."""
    frames, height, width = grid
    self_attention = functools.partial(
        _masked_self_attention,
        rope=generator.rope_table(grid, 0),
        mask=training_mask(MaskLayout(frames, height * width, teacher_forcing), tokens.device),
        teacher_forcing=teacher_forcing,
    )
    inject = None
    if generator.state_injector is not None:
        inject = functools.partial(
            _inject_field,
            injector=generator.state_injector,
            field=field,
            teacher_forcing=teacher_forcing,
            field_on_context=field_on_context,
        )
    logits = []

    def probe(hidden: torch.Tensor) -> None:
        if teacher_forcing:
            _, hidden = _copies(hidden)
        logits.append(generator.visibility_probe(hidden, visibility))

    tokens = generator.run_dit_blocks(
        tokens,
        timestep_modulation=timestep_modulation,
        control_embedding=control_embedding,
        text=text,
        self_attention=lambda index: self_attention,
        inject=inject,
        probe=None if visibility is None else probe,
        checkpoint=generator.gradient_checkpointing and torch.is_grad_enabled(),
    )
    return tokens, logits[0] if logits else None


def forward_train(
    generator: WorldCastGenerator,
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    cond: GeneratorConditions,
    *,
    context_latents: torch.Tensor | None = None,
    context_timestep: torch.Tensor | None = None,
    visibility: VisibilityProbeInputs | None = None,
    field_on_context: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """The generator on the whole window in one forward under an attention mask.

    Without ``context_latents`` the attention is bidirectional (stages 1, 2, 2s). With them it is
    teacher forcing (stage 3): ``[context | noisy]``, block-causal over the first frame and blocks
    of four; the observer signals reach the noisy copy only, the field the noisy copy only unless
    ``field_on_context``, the controls and the ray embedding both copies.

    Args:
        generator (WorldCastGenerator): the generator.
        noisy (Tensor): ``[B, F, 48, 24, 42]`` noisy latents of the window.
        timestep (Tensor): ``[B, F]`` per-frame timesteps in ``[0, 1000]``.
        cond (GeneratorConditions): the window's conditions; a field function is called as
            ``player_state_field(0, F)``.
        context_latents (Tensor | None): ``[B, F, 48, 24, 42]`` the context copy of teacher
            forcing: the window's latents, clean or, with scene state, noised per block to the
            context noise.
        context_timestep (Tensor | None): ``[B, F]`` timesteps of ``context_latents`` (``None``:
            0, a clean copy; with scene state the context noise, in [16, 32)).
        visibility (VisibilityProbeInputs | None): inputs of the visibility probe, whose logits
            ``[B, F, P]`` are then returned too.
        field_on_context (bool): teacher forcing: the field is added to the context copy as well
            (stages 3 and 4 without scene state, as trained).

    Returns:
        Tensor | tuple[Tensor, Tensor]: flow prediction ``[B, F, 48, 24, 42]`` float32 of the noisy
        copy, or ``(flow, logits)``.
    """
    if context_latents is not None and context_latents.shape != noisy.shape:
        raise ValueError("context_latents must have the shape of noisy")
    if context_latents is None and (context_timestep is not None or field_on_context):
        raise ValueError("context_timestep and field_on_context need context_latents")
    if visibility is not None and generator.visibility_probe is None:
        raise ValueError("visibility inputs were given but the generator has no visibility probe")
    teacher_forcing = context_latents is not None
    with generator_autocast(noisy):
        tokens, grid = generator.patchify(noisy)
        prologue = generator.prologue(cond, grid, 0, tokens.dtype)
        timestep_embedding, timestep_modulation = generator.embed_timesteps(timestep, grid[0])
        tokens, control_embedding = prologue.embed(tokens), prologue.control_embedding
        if teacher_forcing:
            if context_timestep is None:
                context_timestep = torch.zeros_like(timestep)
            context_tokens = generator.patchify(context_latents)[0]
            # the context copy carries the ray embedding and the controls, not the observer signals
            if prologue.ray_embedding is not None:
                context_tokens = context_tokens + prologue.ray_embedding
            tokens = torch.cat([context_tokens, tokens], dim=1)
            context_modulation = generator.embed_timesteps(context_timestep, grid[0])[1]
            timestep_modulation = torch.cat([context_modulation, timestep_modulation], dim=1)
            control_embedding = torch.cat([control_embedding] * 2, dim=1)
        tokens, logits = _run_dit_blocks_under_mask(
            generator,
            tokens,
            grid,
            timestep_modulation=timestep_modulation,
            control_embedding=control_embedding,
            text=generator.embed_text(cond.prompt_embeds),
            field=prologue.field,
            visibility=visibility,
            teacher_forcing=teacher_forcing,
            field_on_context=field_on_context,
        )
        if teacher_forcing:
            _, tokens = _copies(tokens)
        out = generator.head(tokens, timestep_embedding, prologue.control_embedding)
        flow = generator.unpatchify(out, grid)
    return flow if visibility is None else (flow, logits)


class TrainingForward(nn.Module):
    """:func:`forward_train` as a module, for the trainer to wrap with FSDP (keys ``generator.``).

    FSDP unshards parameters and casts inputs in the hooks around ``forward``. Per call: the
    inputs as the generator sees them (:func:`~worldcast.modeling.wan22.model.call_inputs`), the
    field built inside the forward (so the weapon embedding is unsharded and receives its
    gradient), then :func:`forward_train`.

    Args:
        generator (WorldCastGenerator): the generator.
        field_builder (FieldBuilder | None): builds the player state field; ``None`` for a
            generator without the state injector.
        input_dtype (torch.dtype | None): the dtype every floating input is cast to; ``None`` (no
            cast) when FSDP mixed precision casts, as in the paper's runs.
    """

    def __init__(
        self,
        generator: WorldCastGenerator,
        field_builder: FieldBuilder | None = None,
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
        """:func:`forward_train` of a condition dict.

        Args:
            noisy (Tensor): ``[B, F, 48, 24, 42]`` noisy latents of the window.
            timestep (Tensor): ``[B, F]`` per-frame timesteps.
            conditions (Mapping[str, Any]): the window's condition dict.
            **kwargs (Any): the keyword arguments of :func:`forward_train`.
        """
        cond, noisy, timestep, kwargs = call_inputs(
            self.generator,
            self.field_builder,
            self.input_dtype,
            conditions,
            noisy,
            timestep,
            kwargs,
        )
        return forward_train(self.generator, noisy, timestep, cond, **kwargs)
