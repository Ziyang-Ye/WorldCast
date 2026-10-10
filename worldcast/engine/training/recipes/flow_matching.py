"""The flow-matching step that stages 1-3 share (Eq. (5)).

One optimizer step, over ``optim.grad_accum_steps`` micro-batches::

    window = the ordinary window, or with scene state on 80 % of the steps the memory window:
             first frame | memory frames 4 | recent context | target frames 4
    x_t, t, v = sample_flow_matching(window latents)    # the first frame pinned at t = 0
    v_theta = generator(x_t, t, prompt, controls, player state field, rays, observer signals)
    loss = mean(W(t) m_frame c_k alpha_k / mean(alpha) (v_theta - v)^2)   (+ the probe's BCE)
    loss.backward()
then the clip, AdamW and the EMA. A recipe says how the window is attended:
:class:`~worldcast.engine.training.recipes.bidirectional.BidirectionalTrainer` (stages 1-2) or
:class:`~worldcast.engine.training.recipes.teacher_forcing.TeacherForcingTrainer` (stage 3).
"""

import time
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from worldcast.config.training import TrainConfig
from worldcast.data.latents import BLOCK, TOKEN_GRID
from worldcast.distributed.fsdp import (
    clip_grad_norm,
    full_state_dict,
    group_all_reduce,
    no_sync,
    sharded_parameters,
)
from worldcast.distributed.process_group import DistInfo
from worldcast.engine.checkpoint.training import EMA_FILE, MODEL_FILE, release_state
from worldcast.engine.generator import WindowInputs, window_inputs
from worldcast.engine.optim import MAX_GRAD_NORM, clip_per_group_, group_grad_norms
from worldcast.modeling.build import EMA_KEY, ONLINE_KEY
from worldcast.modeling.visibility_probe import VisibilityProbeInputs, visibility_bce
from worldcast.modeling.wan22.model import WorldCastGenerator
from worldcast.modeling.wan22.training import TrainingForward
from worldcast.modeling.wan22.vae import Wan22VAE, latent_scale
from worldcast.player_state import (
    PlayerStateFieldConfig,
    field_builder,
    latent_visibility,
    live_eligibility,
    project_view,
    splat,
)
from worldcast.sampling.conditions import control_conditions

from ..losses import (
    FlowMatchingSample,
    compose_weight,
    flow_matching_loss,
    foreground_weight_map,
    memory_weight,
    sample_flow_matching,
)
from ..trainer import STEP_TIMES, BatchStream, Trainer
from ..window import TrainingWindow, memory_windows, training_window

__all__ = [
    "FlowMatchingTrainer",
    "GeneratorForward",
    "TrainerParts",
    "encode_raw_frames",
    "visibility_inputs",
]


def visibility_inputs(conditions: Mapping[str, Any]) -> VisibilityProbeInputs:
    """The visibility probe's inputs: every player projected through the client's camera, its
    footprint splatted on the token grid without the visibility gate (in front, alive, not the
    client), its depth, relative yaw and whether it is in front, over the whole window."""
    grid_h, grid_w = TOKEN_GRID
    view = project_view(
        conditions["player_state_table"],
        conditions["player_controls"],
        conditions["client_slot"],
        grid_h=grid_h,
        grid_w=grid_w,
    )
    eligible = live_eligibility(
        view.in_front,
        alive=conditions["player_alive"],
        visible=torch.ones_like(conditions["player_alive"]),
        client_slot=conditions["client_slot"],
    ).eligible
    footprint = splat(view.uv, view.radius, eligible, grid_h=grid_h, grid_w=grid_w)
    return VisibilityProbeInputs(
        footprint=footprint,
        depth=view.depth,
        relative_yaw=view.relative_yaw,
        in_front=view.in_front,
    )


class GeneratorForward(TrainingForward):
    """The training forward the trainer wraps with FSDP, which also builds the visibility probe's
    inputs from the same (root-cast) conditions as the field.

    ``forward(noisy, timestep, conditions, *, visibility_probe=False, **kwargs)`` returns the flow,
    or ``(flow, logits)`` with ``visibility_probe``.

    Args:
        generator (WorldCastGenerator): the generator.
        field (PlayerStateFieldConfig): its player state field.
        visibility_gate (bool): the field's visibility gate.
    """

    def __init__(
        self,
        generator: WorldCastGenerator,
        field: PlayerStateFieldConfig,
        *,
        visibility_gate: bool = True,
    ) -> None:
        super().__init__(generator, field_builder(field, visibility_gate=visibility_gate))

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, Any],
        *,
        visibility_probe: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The flow of the noisy window; with ``visibility_probe`` also the probe's logits."""
        if visibility_probe:
            kwargs["visibility"] = visibility_inputs(conditions)
        return super().forward(noisy, timestep, conditions, **kwargs)


@dataclass
class TrainerParts:
    """What a flow-matching trainer runs on;
    :func:`~worldcast.engine.training.build.build_trainer_parts` builds it from a config.

    Attributes:
        generator (nn.Module): :class:`GeneratorForward`, FSDP-wrapped.
        optimizer (torch.optim.Optimizer): AdamW.
        data (BatchStream): this rank's batches.
        prompt_embeds (Tensor): ``[1, L, 4096]``.
        info (DistInfo): this process.
        vae (Wan22VAE | None): the VAE of stage 1.
    """

    generator: nn.Module
    optimizer: torch.optim.Optimizer
    data: BatchStream
    prompt_embeds: torch.Tensor
    info: DistInfo
    vae: Wan22VAE | None = None


class FlowMatchingTrainer(Trainer):
    """The flow-matching step of stages 1-3 (module docstring).

    A recipe sets :attr:`bidirectional` and gives :meth:`forward_kwargs`.
    """

    #: Every token attends to every token of the window (stages 1-2); else block-causally, by
    #: teacher forcing (stage 3).
    bidirectional: bool

    def __init__(self, cfg: TrainConfig, parts: TrainerParts) -> None:
        if (cfg.stage.recipe == "bidirectional") != self.bidirectional:
            raise ValueError(
                f"stage {cfg.run.stage} trains by {cfg.stage.recipe}, not with"
                f" {type(self).__name__}"
            )
        super().__init__(cfg, parts.info, data=parts.data, prompt_embeds=parts.prompt_embeds)
        self.generator, self.optimizer, self.vae = parts.generator, parts.optimizer, parts.vae
        self.memory = memory_windows(cfg)
        self.visibility_probe = cfg.stage.scene_state

    @property
    def ema_model(self) -> nn.Module:
        return self.generator

    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """The generator's AdamW."""
        return {"generator": self.optimizer}

    def checkpoint_models(self) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """``model.pt`` (the online weights) and, once it exists, ``model_ema.pt`` (the EMA)."""
        online = release_state(full_state_dict(self.generator))
        models = {MODEL_FILE: {ONLINE_KEY: online}}
        if self.ema is not None:
            models[EMA_FILE] = {EMA_KEY: release_state(self.ema.full_state_dict(self.generator))}
        return models

    def train_step(self) -> dict[str, Any]:
        """One optimizer step over ``optim.grad_accum_steps`` micro-batches; this rank's metrics
        (the loss terms as float32 means over the micro-batches)."""
        started = self.start_step()
        times = dict.fromkeys(STEP_TIMES, 0.0)
        self.generator.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum = self.cfg.optim.grad_accum_steps

        def micro_batch(batch: dict[str, Any], index: int) -> dict[str, torch.Tensor]:
            with no_sync(self.generator, index + 1 < accum):
                return self.forward_backward(batch, accum=accum)

        sums: dict[str, torch.Tensor] = {}
        for terms in self.accumulate(micro_batch, times):
            for key, value in terms.items():
                scaled = value.detach().float() * (1.0 / accum)
                sums[key] = scaled if key not in sums else sums[key] + scaled
        clip_started = time.perf_counter()
        grad_norm, group_norms = self.clip_gradients()
        self.optimizer.step()
        times["optimizer_time_sec"] += self.seconds_since(clip_started)
        self.step += 1
        self.update_ema()
        return {
            **{key: float(value) for key, value in sums.items()},
            **{f"grad_norm_{name}": value for name, value in group_norms.items()},
            "grad_norm": float(grad_norm),
            **{f"lr_{g['name']}": float(g["lr"]) for g in self.optimizer.param_groups},
            **times,
            **self.resources(started),
        }

    def forward_backward(self, batch: dict[str, Any], *, accum: int = 1) -> dict[str, torch.Tensor]:
        """One micro-batch: its window, the flow-matching sample, the forward, the weights, the loss
        and the backward of ``loss / accum``. Returns the detached loss terms."""
        stage = self.cfg.stage
        window = training_window(
            dict(batch),
            step=self.step,
            seed=self.cfg.run.seed,
            memory=self.memory,
            bidirectional=self.bidirectional,
        )
        inputs = self.inputs(window)
        clean = inputs.latents
        sample = sample_flow_matching(
            clean,
            self.scheduler,
            n_pin=window.pinned_frames,
            frames_per_timestep=stage.latent_frames if self.bidirectional else BLOCK,
            first_frame_alone=not self.bidirectional,
            window_timestep=not stage.player_state_field,
            timestep_dtype=self.dtype,
            context_noise=stage.scene_state and not self.bidirectional,
        )
        out = self.generator(
            sample.noisy,
            sample.timestep,
            inputs.conditions,
            visibility_probe=self.visibility_probe,
            **self.forward_kwargs(sample),
        )
        flow, logits = out if self.visibility_probe else (out, None)
        grid = (int(clean.shape[3]), int(clean.shape[4]))
        c_k = alpha_k = None
        if window.memory_mask is not None:
            c_k = memory_weight(window.memory_mask, int(clean.shape[1]), grid=grid).to(self.device)
        if stage.player_state_field and self.cfg.loss.foreground_weight:
            alpha_k = foreground_weight_map(
                inputs.player_state, inputs.visible, window.batch, grid=grid
            )
        weight = compose_weight(
            sample.timestep,
            self.scheduler,
            n_pin=sample.n_pin,
            frame_loss_mask=None if window.loss_mask is None else window.loss_mask.to(self.device),
            c_k=c_k,
            alpha_k=alpha_k,
        )
        loss = flow_loss = flow_matching_loss(flow, sample.target, weight)
        terms = {"flow_loss": flow_loss.detach()}
        if logits is not None:
            visible, valid = latent_visibility(
                window.batch["client_visibility"].to(self.device),
                window.batch["client_visibility_valid"].to(self.device),
            )
            # the probe reads a detached hidden state: its loss trains the probe only
            bce = visibility_bce(logits, visible, valid)
            loss = loss + bce
            terms["visibility_loss"] = bce.detach()
        (loss / accum).backward()
        terms.update(loss=loss.detach(), memory_window=torch.tensor(float(window.has_memory)))
        return terms

    @abstractmethod
    def forward_kwargs(self, sample: FlowMatchingSample) -> dict[str, Any]:
        """What the recipe's forward takes besides the noisy window."""

    def inputs(self, window: TrainingWindow) -> WindowInputs:
        """The window's clean latents ``[B, F, 48, 24, 42]`` and the generator's conditions (stage
        1: the frames encoded with the VAE, the prompt and the controls)."""
        if not self.cfg.stage.player_state_field:
            return self.raw_video_inputs(window.batch)
        scene_state = self.cfg.stage.scene_state
        return window_inputs(
            window.batch,
            prompt_embeds=self.prompt_embeds,
            device=self.device,
            dtype=self.dtype,
            observer_signals=scene_state,
            rays=window.ray_conditions(self.device) if scene_state else None,
        )

    def raw_video_inputs(self, batch: dict[str, Any]) -> WindowInputs:
        """Stage 1: the frames encoded with the VAE; the prompt and the controls."""
        clean = encode_raw_frames(self.vae, batch["frames"], device=self.device, dtype=self.dtype)
        conditions = {
            "prompt_embeds": self.prompt_embeds.expand(int(clean.shape[0]), -1, -1),
            **control_conditions(batch, device=self.device, dtype=self.dtype),
        }
        return WindowInputs(clean, conditions)

    def clip_gradients(self) -> tuple[float, dict[str, float]]:
        """The global clip (``optim.clip: global``) or each group's own; returns the global norm
        and each group's (``{}`` for the global clip), before the clip."""
        if self.cfg.optim.clip == "global":
            return float(clip_grad_norm(self.generator, MAX_GRAD_NORM)), {}
        norms = group_grad_norms(
            self.optimizer.param_groups,
            sharded=sharded_parameters(self.generator),
            all_reduce=group_all_reduce(self.generator),
            device=self.device,
        )
        return clip_per_group_(self.optimizer.param_groups, norms, MAX_GRAD_NORM), norms

    def update_ema(self) -> None:
        """The EMA starts at the first step ``>= ema.start_step`` and is updated in that same call
        and after every later step; none when ``ema.decay`` is 0."""
        if self.cfg.ema.decay > 0.0:
            if self.ema is None and self.step >= self.cfg.ema.start_step:
                self.start_ema()
            if self.ema is not None:
                self.ema.update(self.generator)


def encode_raw_frames(
    vae: Wan22VAE, frames: torch.Tensor, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """The VAE latents ``[B, F, 48, H / 16, W / 16]`` (``dtype``) of a batch of frames ``[B, 3, T,
    H, W]``, encoded one sample at a time without gradient."""
    pixels = frames.to(device=device, dtype=dtype, non_blocking=True)
    scale = latent_scale(pixels.device, pixels.dtype)
    with torch.no_grad():
        latents = [vae.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixels]
    return torch.stack(latents, dim=0).permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype)
