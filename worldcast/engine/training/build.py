"""Build a stage's run from its config: the generator, the data and the trainer of its recipe.

A run is built in an order that fixes its random draws: the base seed, then the generator (a
module the stage adds starts fresh from that seed), FSDP, the optimizer, the prompt and the data,
then the seed ``base + rank`` the steps draw from.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from worldcast.config.training import TrainConfig
from worldcast.data.map_mesh import MeshLibrary
from worldcast.data.memory_selection import MemoryFrameSource
from worldcast.data.recordings import MediaIndex
from worldcast.data.stream import ResumableDataStream
from worldcast.data.training import (
    RAW_VIDEO_MAX_TICK_GAP_SECONDS,
    BucketWindows,
    RawVideoWindows,
    read_bucket_index,
    read_media_exclusion,
)
from worldcast.data.window import DataPaths, WindowSpec
from worldcast.distributed.fsdp import fsdp_wrap, live_module
from worldcast.distributed.process_group import DistInfo, use_deterministic_algorithms
from worldcast.engine.checkpoint.training import (
    CRITIC_FILE,
    CRITIC_KEY,
    MODEL_FILE,
    load_wan22_backbone,
)
from worldcast.engine.generator import load_prompt_embeds, load_vae
from worldcast.engine.optim import build_optimizer
from worldcast.engine.stage import built_state, field_config, generator_config
from worldcast.modeling.build import ONLINE_KEY, read_generator
from worldcast.modeling.wan22.attention import AttentionFn, attention_kernel
from worldcast.modeling.wan22.model import (
    WORLDCAST_MODULES,
    GeneratorConfig,
    WorldCastGenerator,
    worldcast_module,
)
from worldcast.modeling.wan22.vae import Wan22VAE
from worldcast.sampling.schedulers import FlowMatchScheduler
from worldcast.utils.precision import enable_tf32
from worldcast.utils.seed import set_seed
from worldcast.utils.weights import check_state_dict, module_from_state

from .recipes.bidirectional import BidirectionalTrainer
from .recipes.distillation import (
    CRITIC_LR,
    DistillationParts,
    DistillationTrainer,
    RolloutForward,
    ScoreForward,
)
from .recipes.flow_matching import GeneratorForward, TrainerParts
from .recipes.teacher_forcing import TeacherForcingTrainer
from .trainer import Trainer
from .window import memory_frame_config

__all__ = [
    "build_data_stream",
    "build_dataset",
    "build_distillation_parts",
    "build_score_model",
    "build_trainer",
    "build_trainer_parts",
    "build_training_generator",
    "initial_generator_state",
    "start_run",
    "wrap",
]

log = logging.getLogger(__name__)


def build_trainer(cfg: TrainConfig, info: DistInfo, *, resume: Path | None = None) -> Trainer:
    """The trainer of ``cfg``'s stage.

    Args:
        cfg (TrainConfig): the run's config; ``cfg.stage.recipe`` picks the trainer.
        info (DistInfo): this rank of the process group.
        resume (Path | None): a complete checkpoint to continue from (weights, optimizer, EMA,
            data position and RNG states); ``None``: start from ``checkpoint.init``.
    """
    recipe = cfg.stage.recipe
    if recipe == "distillation":
        trainer = DistillationTrainer(cfg, build_distillation_parts(cfg, info, resume=resume))
    else:
        flow_matching = BidirectionalTrainer if recipe == "bidirectional" else TeacherForcingTrainer
        trainer = flow_matching(cfg, build_trainer_parts(cfg, info, resume=resume))
    if resume is not None:
        trainer.resume(resume)
    return trainer


# ====================================================================================== the process
def start_run(cfg: TrainConfig) -> None:
    """Set up the process before anything is built: TF32 as trained, deterministic algorithms in
    the distillation stage (as trained), and the base seed."""
    if cfg.stage.recipe == "distillation":
        use_deterministic_algorithms()
    enable_tf32()
    set_seed(cfg.run.seed)


def wrap(module: nn.Module, cfg: TrainConfig, info: DistInfo, *, unit: str) -> nn.Module:
    """``module`` under FSDP (``unit``: ``size`` or ``root``), sharded as ``fsdp.sharding``.

    A process without a process group (started without torchrun) trains the module as it is, on
    its device: fp32 parameters under autocast, not FSDP's bf16 parameters of the paper's runs.
    """
    if not info.initialized:
        return module.to(info.device)
    return fsdp_wrap(module, sharding=cfg.fsdp.sharding, wrap=unit, device=info.device)


# ==================================================================================== the generator
def build_training_generator(
    config: GeneratorConfig,
    state: Mapping[str, torch.Tensor],
    *,
    new_modules: tuple[str, ...] = (),
    attention: AttentionFn | None = None,
) -> tuple[WorldCastGenerator, list[str]]:
    """The generator with ``state`` loaded, in training mode with gradient checkpointing.

    Every key must be the generator's. Only the modules in ``new_modules`` may be missing, each as
    a whole; the generator is then built on the CPU from the current RNG, so that they start as
    trained. Otherwise the state is assigned, in float32, without a random draw
    (:func:`~worldcast.utils.weights.module_from_state`).

    Args:
        config (GeneratorConfig): the architecture.
        state (Mapping[str, Tensor]): generator keys (:func:`read_generator`).
        new_modules (tuple[str, ...]): names of
            :data:`~worldcast.modeling.wan22.model.WORLDCAST_MODULES`.
        attention (AttentionFn | None): the kernel (flash-attention by default).

    Returns:
        tuple[WorldCastGenerator, list[str]]: the generator and the modules that started fresh.
    """
    unknown = sorted(set(new_modules) - set(WORLDCAST_MODULES))
    if unknown:
        raise ValueError(f"new modules {unknown} are not among {WORLDCAST_MODULES}")

    def build() -> WorldCastGenerator:
        return WorldCastGenerator(config, attention=attention)

    with torch.device("meta"):
        expected = build().state_dict()
    missing = check_state_dict(expected, state)
    lacking = {worldcast_module(key) for key in missing}
    fresh = [module for module in new_modules if module in lacking]
    unexplained = sorted(key for key in missing if worldcast_module(key) not in fresh)
    if unexplained:
        raise KeyError(f"the checkpoint lacks generator keys: {unexplained[:8]}")
    partial = sorted(set(fresh) & {worldcast_module(key) for key in state})
    if partial:
        raise KeyError(f"the checkpoint holds the new modules {partial} in part")
    if fresh:
        generator = build()
        with torch.no_grad():
            tensors = {**dict(generator.named_parameters()), **dict(generator.named_buffers())}
            for key, value in state.items():
                tensors[key].copy_(value.to(dtype=tensors[key].dtype))
    else:
        float32 = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
        generator = module_from_state(build, float32)
    generator.gradient_checkpointing = True
    return generator.train(), fresh


def initial_generator_state(cfg: TrainConfig, resume: Path | None) -> dict[str, torch.Tensor]:
    """The generator weights the run starts from: the resumed checkpoint's online weights, else
    ``checkpoint.init`` (``wan22``: the backbone of the Wan2.2 snapshot)."""
    if resume is not None:
        return read_generator(resume / MODEL_FILE, ONLINE_KEY)
    if not cfg.checkpoint.init:
        raise ValueError(f"stage {cfg.run.stage} starts from a checkpoint: set checkpoint.init")
    if cfg.checkpoint.init == "wan22":
        if not cfg.model.wan22_root:
            raise ValueError("checkpoint.init wan22 reads the backbone of model.wan22_root")
        return load_wan22_backbone(cfg.model.wan22_root)
    return read_generator(cfg.checkpoint.init)


def build_score_model(
    cfg: TrainConfig,
    state: Mapping[str, torch.Tensor],
    *,
    trainable: bool,
    attention: AttentionFn | None = None,
) -> WorldCastGenerator:
    """A score model of the distillation, the teacher or the critic: the stage-2 model with
    ``state``, whose modules of scene state (which the stage-2 model does not have) are left out.

    Args:
        cfg (TrainConfig): the distillation run.
        state (Mapping[str, Tensor]): generator keys (:func:`read_generator`).
        trainable (bool): the critic trains; the teacher is frozen.
        attention (AttentionFn | None): the kernel (flash-attention by default).
    """
    config = generator_config(cfg, score_model=True)
    model, _ = build_training_generator(config, built_state(state, config), attention=attention)
    return model.requires_grad_(trainable)


# ========================================================================================= the data
def _require(cfg: TrainConfig, *names: str) -> None:
    missing = [n for n in names if not getattr(cfg.data, n)]
    if missing:
        raise ValueError(f"stage {cfg.run.stage} needs data.{', data.'.join(missing)}")


def build_dataset(cfg: TrainConfig) -> Dataset:
    """The stage's windows: raw video in stage 1, the bucket index over the latent cache from
    stage 2 on, with memory frames in the stages with scene state."""
    stage, d = cfg.stage, cfg.data
    if not stage.player_state_field:
        _require(cfg, "train_manifest")
        spec = WindowSpec(
            stage.latent_frames,
            max_tick_gap_seconds=RAW_VIDEO_MAX_TICK_GAP_SECONDS,
            camera_encoding=stage.camera_encoding,
        )
        return RawVideoWindows(d.train_manifest, spec=spec, dataset_root=d.dataset_root)
    _require(cfg, "bucket_dir")
    paths = DataPaths.from_config(d, "data")
    exclusion = frozenset()
    if d.exclude_media_manifest:
        exclusion = read_media_exclusion(d.exclude_media_manifest)
    spec = WindowSpec(stage.latent_frames, camera_encoding=stage.camera_encoding)
    memory_frames = None
    if stage.scene_state:
        _require(cfg, "collision_meshes")
        memory_frames = MemoryFrameSource(
            dataset_root=paths.dataset_root,
            latent_cache_root=paths.latent_cache_root,
            observer_signal_label_root=paths.observer_signal_label_root,
            meshes=MeshLibrary(d.collision_meshes),
            spec=spec,
            config=memory_frame_config(cfg),
        )
    return BucketWindows(
        read_bucket_index(d.bucket_dir, exclusion, d.bucket_weights),
        media_index=MediaIndex.load(paths.media_index),
        paths=paths,
        spec=spec,
        memory_frames=memory_frames,
    )


def build_data_stream(
    cfg: TrainConfig, info: DistInfo, dataset: Dataset | None = None
) -> ResumableDataStream:
    """This rank's resumable batches (the loader seeded ``seed + rank + 100003``)."""
    return ResumableDataStream(
        build_dataset(cfg) if dataset is None else dataset,
        rank=info.rank,
        world_size=info.world_size,
        seed=cfg.run.seed,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=info.device.type == "cuda",
    )


def _run_inputs(
    cfg: TrainConfig, info: DistInfo, dataset: Dataset | None
) -> tuple[torch.Tensor, ResumableDataStream, Wan22VAE | None]:
    """The last steps of a run's construction, in their order: the prompt embedding, this rank's
    data stream, the VAE of stage 1, then the seed ``base + rank`` the steps draw from."""
    prompt = load_prompt_embeds(cfg.data.prompt_embedding, cfg.model.wan22_root, info.device)
    data = build_data_stream(cfg, info, dataset)
    vae = None if cfg.stage.player_state_field else load_vae(cfg.model.wan22_root, info.device)
    set_seed(cfg.run.seed + info.rank)
    return prompt, data, vae


# ======================================================================================== the parts
def build_trainer_parts(
    cfg: TrainConfig,
    info: DistInfo,
    *,
    resume: Path | None = None,
    dataset: Dataset | None = None,
) -> TrainerParts:
    """A flow-matching run (module docstring).

    Args:
        cfg (TrainConfig): the run.
        info (DistInfo): this process.
        resume (Path | None): the checkpoint the run continues from.
        dataset (Dataset | None): the windows (default: the stage's, :func:`build_dataset`).
    """
    stage = cfg.stage
    start_run(cfg)
    generator, fresh = build_training_generator(
        generator_config(cfg),
        initial_generator_state(cfg, resume),
        new_modules=() if resume else stage.new_modules,
        attention=attention_kernel(cfg.model.attention, info.device),
    )
    if info.is_main and fresh:
        log.info("fresh modules: %s", fresh)
    forward = GeneratorForward(
        generator,
        field_config(cfg, bidirectional=stage.recipe == "bidirectional"),
        visibility_gate=cfg.model.visibility_gate,
    )
    # the memory frames differ across ranks, so the stages with scene state keep one FSDP unit
    wrapped = wrap(forward, cfg, info, unit="root" if stage.scene_state else "size")
    optimizer = build_optimizer(live_module(wrapped).named_parameters(), cfg.optim.lr)
    prompt, data, vae = _run_inputs(cfg, info, dataset)
    return TrainerParts(wrapped, optimizer, data, prompt, info, vae)


def build_distillation_parts(
    cfg: TrainConfig,
    info: DistInfo,
    *,
    resume: Path | None = None,
    dataset: Dataset | None = None,
) -> DistillationParts:
    """A distillation run: the base seed, the generator (``checkpoint.init``), the teacher
    (``distillation.teacher``) and the critic (``distillation.critic``, on resume ``critic.pt``),
    each wrapped by FSDP in turn, the two AdamW, the prompt, the data, then the seed ``base +
    rank``.

    Args:
        cfg (TrainConfig): the run.
        info (DistInfo): this process.
        resume (Path | None): the checkpoint the run continues from.
        dataset (Dataset | None): the windows (default: the stage's, :func:`build_dataset`).
    """
    if not cfg.distillation.teacher or not (cfg.distillation.critic or resume):
        raise ValueError("stage 4 needs distillation.teacher and distillation.critic")
    start_run(cfg)
    attention = attention_kernel(cfg.model.attention, info.device)
    scheduler = FlowMatchScheduler()
    gate = cfg.model.visibility_gate
    scene_state = cfg.stage.scene_state
    generator, _ = build_training_generator(
        generator_config(cfg), initial_generator_state(cfg, resume), attention=attention
    )
    rollout = RolloutForward(
        generator,
        field_config(cfg, bidirectional=False),
        scheduler,
        field_on_context=not scene_state,
        visibility_gate=gate,
    )
    wrapped = wrap(rollout, cfg, info, unit="root" if scene_state else "size")
    field = field_config(cfg, bidirectional=True)

    def score_model(state: Mapping[str, torch.Tensor], *, trainable: bool) -> nn.Module:
        model = build_score_model(cfg, state, trainable=trainable, attention=attention)
        forward = ScoreForward(model, field, scheduler, visibility_gate=gate)
        return wrap(forward, cfg, info, unit="size")

    teacher = score_model(read_generator(cfg.distillation.teacher), trainable=False)
    if resume is None:
        critic_state = read_generator(cfg.distillation.critic)
    else:
        critic_state = read_generator(resume / CRITIC_FILE, CRITIC_KEY)
    critic = score_model(critic_state, trainable=True)
    generator_optimizer = build_optimizer(
        live_module(wrapped).named_parameters(), cfg.optim.lr, module_lrs=False, foreach=True
    )
    critic_optimizer = build_optimizer(
        live_module(critic).named_parameters(), CRITIC_LR, module_lrs=False, foreach=True
    )
    prompt, data, _ = _run_inputs(cfg, info, dataset)
    return DistillationParts(
        generator=wrapped,
        teacher=teacher,
        critic=critic,
        generator_optimizer=generator_optimizer,
        critic_optimizer=critic_optimizer,
        data=data,
        prompt_embeds=prompt,
        info=info,
    )
