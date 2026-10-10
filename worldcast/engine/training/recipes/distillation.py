"""Stage 4: distribution matching distillation of the stage-3 model into the four-step generator.

Three models, each wrapped by FSDP: the **generator** (the stage-3 model), the frozen **teacher**
and the trained **critic**. The teacher and the critic are the two score models: both the stage-2
model, scoring the whole 41-latent window bidirectionally, with the stage-2 model's inputs. One
step::

    if step % GENERATOR_EVERY == 0:     # the generator
        per micro-batch: generator_loss, backward;  clip, AdamW; EMA update once it exists
    per micro-batch: critic_loss, backward;         clip, AdamW    # every step
    the EMA starts at the end of step ``ema.start_step``, without an update

The critic is updated on every step.

**Rollout** (:meth:`DistillationTrainer.rollout`), self-forcing, without gradient on the KV cache:
the recorded first frame is written at t = 0; each of the 10 blocks of 4 latent frames starts
from its noise and runs the denoising steps down to one exit step drawn for all blocks (rank 0's
draw), and is written to the cache at the context noise (16 with scene state, clean without). Then
one teacher-forcing **replay** of the generator recomputes every block's exit prediction, with the
gradient on generator steps, the rollout (at the context noise) as its clean context.

**Generator loss**: the rollout x diffused to a score timestep (uniform over the schedule, shifted,
clamped to [20, 980]) with the first frame kept clean; ``g = (critic - teacher) / mean|x -
teacher|`` of the two x0 estimates; ``0.5 MSE(x, sg[x - g])`` in float64 over frames 1-40. **Critic
loss**: the critic's flow-matching loss on a fresh rollout, frames 1-40. Windows are ordinary; with
scene state they are drawn from the windows that have memory frames, and pass no camera to the ray
embedding.
"""

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch import nn

from worldcast.config.training import TrainConfig
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.latents import BLOCK
from worldcast.distributed.fsdp import clip_grad_norm, full_state_dict, live_module
from worldcast.distributed.process_group import DistInfo
from worldcast.engine.checkpoint.training import CRITIC_FILE, CRITIC_KEY, MODEL_FILE, release_state
from worldcast.engine.generator import window_inputs
from worldcast.engine.optim import MAX_GRAD_NORM
from worldcast.modeling.build import EMA_KEY, ONLINE_KEY
from worldcast.modeling.wan22.dit import KVCache
from worldcast.modeling.wan22.model import CausalGeneratorAdapter, WorldCastGenerator
from worldcast.modeling.wan22.training import TrainingForward
from worldcast.player_state import PlayerStateFieldConfig, field_builder
from worldcast.sampling.sampler import CONTEXT_NOISE, DENOISING_STEPS, Sampler
from worldcast.sampling.schedulers import (
    FlowMatchScheduler,
    draw_noise,
    flow_to_x0,
    noise_context,
    renoise_frames,
    run_denoising_steps,
)

from .. import losses
from ..trainer import STEP_TIMES, BatchStream, Trainer

__all__ = [
    "CRITIC_LR",
    "GENERATOR_EVERY",
    "DiffusedRollout",
    "DistillationParts",
    "DistillationTrainer",
    "RolloutForward",
    "RolloutInputs",
    "RolloutSampler",
    "ScoreForward",
]

#: The generator is updated on every this-many-th step, the critic on every step.
GENERATOR_EVERY = 5
#: The critic's learning rate (the generator's is ``optim.lr``).
CRITIC_LR = 4e-7


class RolloutForward(TrainingForward):
    """The generator as FSDP wraps it: one call on the KV cache (``kv_cache`` given) or the
    teacher-forcing replay (``context_latents`` given). Returns ``(flow, x0)``, x0 of the inputs as
    the generator received them.

    Args:
        generator (WorldCastGenerator): the generator.
        field (PlayerStateFieldConfig): its player state field.
        scheduler (FlowMatchScheduler): the table.
        field_on_context (bool): the replay adds the field to the clean context copy too (the
            stage without scene state, as trained).
        visibility_gate (bool): the field's visibility gate.
    """

    def __init__(
        self,
        generator: WorldCastGenerator,
        field: PlayerStateFieldConfig,
        scheduler: FlowMatchScheduler,
        *,
        field_on_context: bool = False,
        visibility_gate: bool = True,
    ) -> None:
        super().__init__(generator, field_builder(field, visibility_gate=visibility_gate))
        self.scheduler = scheduler
        self.field_on_context = field_on_context

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        conditions: dict[str, Any],
        *,
        kv_cache: KVCache | None = None,
        frame_offset: int = 0,
        context_latents: torch.Tensor | None = None,
        context_timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One call on the KV cache, or the replay on the clean context copy."""
        if kv_cache is not None:
            # built per call: FSDP's size wrap may replace the generator attribute by a unit
            causal = CausalGeneratorAdapter(self.generator, self.field_builder)
            flow = causal(noisy, timestep, conditions, kv_cache=kv_cache, frame_offset=frame_offset)
        else:
            flow = super().forward(
                noisy,
                timestep,
                conditions,
                context_latents=context_latents,
                context_timestep=context_timestep,
                field_on_context=self.field_on_context,
            )
        return flow, flow_to_x0(flow, noisy, timestep, self.scheduler)


class ScoreForward(TrainingForward):
    """A score model, the teacher or the critic: the stage-2 model on the whole 41-latent window,
    bidirectional. Returns ``(flow, x0)``.

    Args:
        generator (WorldCastGenerator): the stage-2 model.
        field (PlayerStateFieldConfig): its player state field.
        scheduler (FlowMatchScheduler): the table.
        visibility_gate (bool): the field's visibility gate.
    """

    def __init__(
        self,
        generator: WorldCastGenerator,
        field: PlayerStateFieldConfig,
        scheduler: FlowMatchScheduler,
        *,
        visibility_gate: bool = True,
    ) -> None:
        super().__init__(generator, field_builder(field, visibility_gate=visibility_gate))
        self.scheduler = scheduler

    def forward(
        self, noisy: torch.Tensor, timestep: torch.Tensor, conditions: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The whole window at once: ``(flow, x0)``."""
        flow = super().forward(noisy, timestep, conditions)
        return flow, flow_to_x0(flow, noisy, timestep, self.scheduler)


class RolloutSampler(Sampler):
    """The block-causal sampler on the FSDP-wrapped :class:`RolloutForward`: FSDP's root casts the
    inputs, and the forward itself returns x0 of the inputs as it received them."""

    def predict_x0(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditions: dict[str, Any],
        *,
        frame_offset: int,
    ) -> torch.Tensor:
        """x0 of the wrapped forward's ``(flow, x0)``, uncast here."""
        return self.predict_flow(x, timestep, conditions, frame_offset=frame_offset)[1]


class DiffusedRollout(NamedTuple):
    """The rollout x diffused for the score models (:meth:`DistillationTrainer.diffuse`).

    Attributes:
        timestep (Tensor): ``[B, F]`` the score timestep, the same for every frame.
        noise (Tensor): ``[B, F, C, H, W]`` eps.
        noisy (Tensor): ``[B, F, C, H, W]`` x_t, the first frame clean.
        pinned_timestep (Tensor): ``[B, F]`` the timestep with the first frame at 0.
    """

    timestep: torch.Tensor
    noise: torch.Tensor
    noisy: torch.Tensor
    pinned_timestep: torch.Tensor


class RolloutInputs(NamedTuple):
    """A window as the distillation step reads it (:meth:`DistillationTrainer.inputs`).

    Attributes:
        first_frame (Tensor): ``[B, 1, C, H, W]`` the window's recorded first frame.
        conditions (dict[str, Any]): the generator's conditions.
        score_conditions (dict[str, Any]): the score models' (the stage-2 model's: without the
            observer signals).
    """

    first_frame: torch.Tensor
    conditions: dict[str, Any]
    score_conditions: dict[str, Any]


@dataclass
class DistillationParts:
    """What a distillation run trains;
    :func:`~worldcast.engine.training.build.build_distillation_parts` builds it from a config.

    Attributes:
        generator (nn.Module): :class:`RolloutForward`, FSDP-wrapped.
        teacher (nn.Module): :class:`ScoreForward` of the frozen teacher, FSDP-wrapped.
        critic (nn.Module): :class:`ScoreForward` of the critic, FSDP-wrapped.
        generator_optimizer (torch.optim.Optimizer): AdamW of the generator.
        critic_optimizer (torch.optim.Optimizer): AdamW of the critic.
        data (BatchStream): this rank's batches.
        prompt_embeds (Tensor): ``[1, L, 4096]``.
        info (DistInfo): this process.
    """

    generator: nn.Module
    teacher: nn.Module
    critic: nn.Module
    generator_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    data: BatchStream
    prompt_embeds: torch.Tensor
    info: DistInfo


class DistillationTrainer(Trainer):
    """Stage 4 (module docstring)."""

    def __init__(self, cfg: TrainConfig, parts: DistillationParts) -> None:
        super().__init__(cfg, parts.info, data=parts.data, prompt_embeds=parts.prompt_embeds)
        self.generator, self.teacher, self.critic = parts.generator, parts.teacher, parts.critic
        self.generator_optimizer = parts.generator_optimizer
        self.critic_optimizer = parts.critic_optimizer
        self.context_noise = CONTEXT_NOISE if cfg.stage.scene_state else 0
        self.draws: dict[str, list] = {}

    @property
    def ema_model(self) -> nn.Module:
        return self.generator

    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """The generator's AdamW and the critic's."""
        return {"generator": self.generator_optimizer, "critic": self.critic_optimizer}

    def checkpoint_models(self) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """``model.pt`` (the generator and its EMA) and ``critic.pt``."""
        model = {ONLINE_KEY: release_state(full_state_dict(self.generator))}
        if self.ema is not None:
            model[EMA_KEY] = release_state(self.ema.full_state_dict(self.generator))
        critic = {CRITIC_KEY: release_state(full_state_dict(self.critic))}
        return {MODEL_FILE: model, CRITIC_FILE: critic}

    def train_step(self) -> dict[str, Any]:
        """A generator update on every :data:`GENERATOR_EVERY`-th step, a critic update on every
        step; this rank's metrics. ``generator_loss`` and ``critic_loss`` are the losses of the
        update's last micro-batch, the draws those of all its micro-batches."""
        started = self.start_step()
        times = dict.fromkeys(STEP_TIMES, 0.0)
        self.draws = {}
        train_generator = self.step % GENERATOR_EVERY == 0
        metrics: dict[str, Any] = {}
        if train_generator:
            update = self.generator, self.generator_optimizer, self.generator_loss
            metrics.update(self._update("generator", *update, times))
            if self.ema is not None:
                self.ema.update(self.generator)
        update = self.critic, self.critic_optimizer, self.critic_loss
        metrics.update(self._update("critic", *update, times))
        self.step += 1
        if self.ema is None and self.cfg.ema.decay > 0 and self.step >= self.cfg.ema.start_step:
            self.start_ema()
        return {
            **metrics,
            "trained_generator": train_generator,
            **self.draws,
            "lr_generator": float(self.generator_optimizer.param_groups[0]["lr"]),
            "lr_critic": float(self.critic_optimizer.param_groups[0]["lr"]),
            **times,
            **self.resources(started),
        }

    def _update(
        self,
        name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_of: Callable[[dict[str, Any]], torch.Tensor],
        times: dict[str, float],
    ) -> dict[str, float]:
        """One optimizer step of ``model`` on ``loss_of`` over the accumulated micro-batches: its
        gradient norm before the clip and the loss of the last micro-batch."""
        accum = self.cfg.optim.grad_accum_steps
        optimizer.zero_grad(set_to_none=True)

        def micro_batch(batch: dict[str, Any], index: int) -> torch.Tensor:
            loss = loss_of(batch)
            (loss * (1.0 / accum)).backward()
            return loss

        loss = self.accumulate(micro_batch, times)[-1]
        clip_started = time.perf_counter()
        metrics = {
            f"{name}_grad_norm": float(clip_grad_norm(model, MAX_GRAD_NORM)),
            f"{name}_loss": float(loss.detach()),
        }
        optimizer.step()
        # Free the gradients before the model runs again (no effect on any number): FSDP refuses
        # a forward over stale unsharded gradients on one rank.
        optimizer.zero_grad(set_to_none=True)
        times["optimizer_time_sec"] += self.seconds_since(clip_started)
        return metrics

    def generator_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """The generator's distribution matching loss on one micro-batch, 0-dim float64."""
        first_frame, conditions, score_conditions = self.inputs(batch, "generator")
        x = self.rollout(first_frame, conditions, grad=True).to(self.dtype)
        mask = losses.distribution_matching_mask(x)
        with torch.no_grad():
            diffused = self.diffuse(x, "generator")
            t, noisy = diffused.pinned_timestep, diffused.noisy
            _, critic_x0 = self.critic(noisy, t, score_conditions)
            _, teacher_x0 = self.teacher(noisy, t, score_conditions)
            gradient = losses.distribution_matching_gradient(critic_x0, teacher_x0, x)
        return losses.distribution_matching_loss(x, gradient, mask)

    def critic_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """The critic's flow-matching loss on a rollout made without gradient, 0-dim float32."""
        first_frame, conditions, score_conditions = self.inputs(batch, "critic")
        with torch.no_grad():
            x = self.rollout(first_frame, conditions, grad=False).to(self.dtype)
        diffused = self.diffuse(x, "critic")
        _, critic_x0 = self.critic(diffused.noisy, diffused.pinned_timestep, score_conditions)
        return losses.critic_loss(
            x, critic_x0, diffused.noise, diffused.noisy, diffused.timestep, self.scheduler
        )

    def diffuse(self, x: torch.Tensor, phase: str) -> DiffusedRollout:
        """The rollout diffused for the score models, ``x_t = (1 - sigma_t) x + sigma_t eps``: the
        score timestep is drawn first, then eps (x's dtype); the first frame stays clean."""
        batch_size, frames = int(x.shape[0]), int(x.shape[1])
        t = losses.sample_score_timestep(batch_size, frames, device=x.device)
        eps = draw_noise(x)
        x_t = renoise_frames(x, t, self.scheduler, noise=eps.flatten(0, 1))
        self._record(phase, "score_timestep", t[:, 0].tolist())
        noisy, pinned_t, _ = losses.pin_clean_frames(x_t, t, x[:, :1])
        return DiffusedRollout(t, eps, noisy, pinned_t)

    def inputs(self, batch: dict[str, Any], phase: str) -> RolloutInputs:
        """The window's first frame, the generator's conditions and the score models'."""
        self._record(phase, "dataset_index", [m["dataset_index"] for m in batch["metadata"]])
        window = window_inputs(
            batch,
            prompt_embeds=self.prompt_embeds,
            device=self.device,
            dtype=self.dtype,
            observer_signals=self.cfg.stage.scene_state,
        )
        conditions = window.conditions
        scored = {k: v for k, v in conditions.items() if k not in OBSERVER_SIGNAL_KEYS}
        return RolloutInputs(window.latents[:, :1], conditions, scored)

    def _record(self, phase: str, name: str, values: list) -> None:
        self.draws.setdefault(f"{phase}_{name}", []).extend(values)

    def exit_step(self, num_blocks: int) -> list[int]:
        """The denoising step every block exits at: ``randint(0, 4, (num_blocks,))`` on every rank,
        rank 0's first value for all blocks."""
        flags = torch.randint(0, len(DENOISING_STEPS), (num_blocks,), device=self.device)
        if self.info.initialized:
            dist.broadcast(flags, src=0)
        return [int(flags[0])] * num_blocks

    def rollout(
        self, first_frame: torch.Tensor, conditions: dict[str, Any], *, grad: bool
    ) -> torch.Tensor:
        """The self-forcing rollout of one window and its replay (module docstring); returns x0
        ``[B, F, C, H, W]``.

        RNG, in order: the entry noise (``[B, F - 1, C, H, W]`` in the compute dtype), the exit
        step, then per block one draw per denoising step before the exit and, with context noise,
        one for its cache write; with context noise, one for the replay's context.

        Args:
            first_frame (Tensor): ``[B, 1, C, H, W]`` the window's first frame.
            conditions (dict[str, Any]): the generator's conditions (:meth:`inputs`).
            grad (bool): the replay records the gradient.
        """
        frames = self.cfg.stage.latent_frames
        batch_size, latent_shape = int(first_frame.shape[0]), first_frame.shape[2:]
        entry_noise = torch.randn(
            [batch_size, frames - 1, *latent_shape], device=self.device, dtype=self.dtype
        )
        # FSDP's mixed precision keeps the parameters in float32: the cache takes the compute dtype
        cache = live_module(self.generator).generator.allocate_kv_cache(
            frames, batch_size=batch_size, dtype=self.dtype, device=self.device
        )
        sampler = RolloutSampler.create(self.generator, cache, context_noise=self.context_noise)
        video = torch.zeros(
            [batch_size, frames, *latent_shape], device=self.device, dtype=self.dtype
        )
        video[:, :1] = first_frame
        replay_inputs = [video[:, :1]]
        replay_timesteps = [torch.zeros([batch_size, 1], device=self.device)]
        with torch.no_grad():
            sampler.predict_x0(first_frame, replay_timesteps[0], conditions, frame_offset=0)
            exits = self.exit_step((frames - 1) // BLOCK)
            self._record("generator" if grad else "critic", "exit_step", exits[:1])
            for index, exit_step in enumerate(exits):
                first = 1 + index * BLOCK
                leave = run_denoising_steps(
                    functools.partial(
                        sampler.predict_x0, conditions=conditions, frame_offset=first
                    ),
                    entry_noise[:, first - 1 : first - 1 + BLOCK],
                    sampler.denoising_timesteps,
                    sampler.scheduler,
                    exit_step=exit_step,
                )
                replay_inputs.append(leave.input)
                replay_timesteps.append(leave.timestep.float())
                video[:, first : first + BLOCK] = leave.x0
                sampler.write_context(leave.x0.detach(), conditions, frame_offset=first)
        context, context_t = noise_context(
            video.detach(), sampler.scheduler, context_noise=self.context_noise
        )
        with torch.set_grad_enabled(grad):
            _, x0 = self.generator(
                torch.cat([x.detach() for x in replay_inputs], dim=1),
                torch.cat(replay_timesteps, dim=1),
                conditions,
                context_latents=context,
                context_timestep=context_t,
            )
        return x0
