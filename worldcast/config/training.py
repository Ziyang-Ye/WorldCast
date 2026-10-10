"""The training config: what a user sets for a run of one of the paper's training stages.

A run names its stage (``run.stage``). What the paper fixes for a stage is its :class:`Stage` in
:data:`STAGES`; the rest is a constant of the module that uses it: the learning rates per module and
AdamW (:mod:`worldcast.engine.optim`), the loss weights and the score timesteps
(:mod:`worldcast.engine.training.losses`), the sampling of the windows
(:mod:`worldcast.data.training`) and the memory frames (:mod:`worldcast.data.memory_selection`).
The settings below default to the paper run of ``run.stage`` (:data:`STAGE_SETTINGS`), wherever
the stage is named (a file or ``--set``), so ``configs/train/stage<stage>.yaml`` only names its
stage. Nothing here imports torch.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .loader import Paths, build, check_at_least, check_choice, load, merge, set_dotted

__all__ = [
    "BUCKET_FILES",
    "STAGE_SETTINGS",
    "STAGES",
    "WINDOW_LATENT_FRAMES",
    "CheckpointConfig",
    "DataConfig",
    "DistillationConfig",
    "EMAConfig",
    "FSDPConfig",
    "LossConfig",
    "OptimConfig",
    "Stage",
    "TrainConfig",
    "TrainModelConfig",
    "TrainRunConfig",
    "ValidationConfig",
    "load_train_config",
    "paper_config",
]

#: Latent frames of a 10 s window: the training windows from stage ``1_long`` on, and the windows
#: validation scores.
WINDOW_LATENT_FRAMES = 41
#: The bucket files of the training split, one weight each in ``data.bucket_weights``
#: (:data:`worldcast.data.training.BUCKET_NAMES`).
BUCKET_FILES = 5


@dataclass(frozen=True)
class Stage:
    """What the paper fixes for one training stage (Sec. 3.5, App. "Training stages and scaling").

    Attributes:
        recipe (str): ``bidirectional`` (stages 1 and 2), ``teacher_forcing`` (stage 3) or
            ``distillation`` (stage 4).
        camera_encoding (str): the camera encoding of the controls the stage trained with:
            ``noclip`` with scene state, ``clip`` without.
        latent_frames (int): latent frames of a training window.
        player_state_field (bool): the player state field; stage 1, without it, tunes Wan2.2 on
            raw video.
        scene_state (bool): memory frames, with the ray embedding, the observer-signal embedding and
            the visibility probe; context noise in stages 3 and 4.
        new_modules (tuple[str, ...]): the modules the stage adds; they start fresh and may be
            missing from the checkpoint the stage starts from.
    """

    recipe: str
    camera_encoding: str
    latent_frames: int = WINDOW_LATENT_FRAMES
    player_state_field: bool = True
    scene_state: bool = False
    new_modules: tuple[str, ...] = ()


#: The paper's stages: 1 and 1_long (stage 1 on five-second, then on ten-second windows), 2 (+ the
#: player state field), 2s (+ scene state), 3 (teacher forcing) and 4 (distillation), the last two
#: with and without scene state.
STAGES = {
    "1": Stage(
        "bidirectional",
        "clip",
        latent_frames=21,
        player_state_field=False,
        new_modules=("controls",),
    ),
    "1_long": Stage("bidirectional", "clip", player_state_field=False),
    "2": Stage("bidirectional", "clip", new_modules=("state_injector",)),
    "2s": Stage(
        "bidirectional",
        "noclip",
        scene_state=True,
        new_modules=("ray_embedding", "observer_signals", "visibility_probe"),
    ),
    "3": Stage("teacher_forcing", "noclip", scene_state=True),
    "3_noscene": Stage("teacher_forcing", "clip"),
    "4": Stage("distillation", "noclip", scene_state=True),
    "4_noscene": Stage("distillation", "clip"),
}


@dataclass(frozen=True)
class TrainRunConfig:
    """The run.

    Attributes:
        stage (str): one of :data:`STAGES`; it has no default.
        name (str): a label, printed and stored with the checkpoints.
        seed (int): the base seed.
        max_steps (int): optimizer steps.
        output_dir (str | None): checkpoints and ``metrics.jsonl``.
        log_interval (int): print the metrics every this many steps (0: never).
    """

    stage: str = ""
    name: str = ""
    seed: int = 20260829
    max_steps: int = 25000
    output_dir: str | None = None
    log_interval: int = 10

    def __post_init__(self) -> None:
        check_choice("run.stage", self.stage, tuple(STAGES))
        check_at_least("run.seed", self.seed, 0)
        check_at_least("run.max_steps", self.max_steps, 0)
        check_at_least("run.log_interval", self.log_interval, 0)


@dataclass(frozen=True)
class TrainModelConfig:
    """The generator.

    Attributes:
        wan22_root (str | None): the Wan2.2-TI2V-5B snapshot: ``config.json``, the backbone (stage 1
            starts from it), the VAE (stage 1 encodes video) and umT5 (without a prompt embedding).
        attention (str): the attention kernel, a name of
            :data:`worldcast.modeling.wan22.attention.ATTENTION_KERNELS`: ``flash`` (the paper's)
            or ``sdpa`` (runs on the CPU; not bit-equal on the GPU).
        state_injector_block (int): the DiT block (counted from 0) after which the state
            injector adds the player state field: 1, the second block; 22 in the ablation "Trained
            with late injection" (Table 5).
        field_downsample (int): 1; 2 in the ablation "Trained with a coarse field".
        visibility_gate (bool): the field's GT visibility gate and confidence; off in the ablation
            "Trained without visibility".
        dims (dict[str, int]): the generator's dimensions that differ from the paper's, by the
            dotted name of the field in :class:`worldcast.modeling.wan22.model.GeneratorConfig`
            (``{"dim": 32, "num_heads": 2, "controls.hidden": 64}``: a small model for tests);
            empty for the paper's.
    """

    wan22_root: str | None = None
    attention: str = "flash"
    state_injector_block: int = 1
    field_downsample: int = 1
    visibility_gate: bool = True
    dims: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_at_least("model.state_injector_block", self.state_injector_block, 0)
        check_at_least("model.field_downsample", self.field_downsample, 1)


@dataclass(frozen=True)
class DataConfig:
    """The data (docs/data.md).

    Attributes:
        bucket_dir (str | None): ``train_q*.jsonl``, the windows of stages 2-4.
        media_index (str | None): every recording.
        dataset_root (str | None): the recordings (tick tables, videos).
        latent_cache_root (str | None): the cached latents of every window.
        visibility_label_root (str | None): the GT visibility labels.
        observer_signal_label_root (str | None): the flash and scope labels.
        exclude_media_manifest (str | None): the held-out media, left out of training.
        train_manifest (str | None): the raw-video manifest of stage 1.
        bucket_weights (tuple[float, ...] | None): the sampling weight of each bucket file, in the
            order of :data:`worldcast.data.training.BUCKET_NAMES`; unset, the paper's
            (:data:`worldcast.data.training.BUCKET_WEIGHTS`).
        collision_meshes (dict[str, str]): ``{map_name: path}``, for the memory frames.
        prompt_embedding (str | None): the umT5 embedding of the fixed prompt; unset, umT5 encodes
            it.
        batch_size (int): windows per GPU per micro-batch.
        num_workers (int): loader workers per GPU.
    """

    bucket_dir: str | None = None
    media_index: str | None = None
    dataset_root: str | None = None
    latent_cache_root: str | None = None
    visibility_label_root: str | None = None
    observer_signal_label_root: str | None = None
    exclude_media_manifest: str | None = None
    train_manifest: str | None = None
    bucket_weights: tuple[float, ...] | None = None
    collision_meshes: dict[str, str] = field(default_factory=dict)
    prompt_embedding: str | None = None
    batch_size: int = 1
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.bucket_weights is not None:
            if len(self.bucket_weights) != BUCKET_FILES:
                raise ValueError(
                    f"data.bucket_weights must hold one weight for each of the {BUCKET_FILES}"
                    f" bucket files, got {len(self.bucket_weights)}"
                )
            if not all(math.isfinite(w) and w > 0.0 for w in self.bucket_weights):
                raise ValueError(
                    f"data.bucket_weights must be numbers > 0, got {list(self.bucket_weights)!r}"
                )
        check_at_least("data.batch_size", self.batch_size, 1)
        check_at_least("data.num_workers", self.num_workers, 0)


@dataclass(frozen=True)
class LossConfig:
    """The flow-matching loss.

    Attributes:
        foreground_weight (bool): alpha_k from stage 2 on; off in the ablation "Trained without
            foreground weight" (Table 5).
    """

    foreground_weight: bool = True


@dataclass(frozen=True)
class OptimConfig:
    """The optimizer.

    Attributes:
        lr (float): learning rate of the backbone and the controls (the distillation stage's of
            every module).
        grad_accum_steps (int): micro-batches per optimizer step.
        clip (str): ``per_group`` (each module's gradient norm on its own) or ``global``, as each
            paper run clipped; the distillation stage clips each of its models globally and takes
            ``global`` only.
    """

    lr: float = 2.8e-5
    grad_accum_steps: int = 1
    clip: str = "per_group"

    def __post_init__(self) -> None:
        check_at_least("optim.lr", self.lr, 0.0)
        check_at_least("optim.grad_accum_steps", self.grad_accum_steps, 1)
        check_choice("optim.clip", self.clip, ("per_group", "global"))


@dataclass(frozen=True)
class FSDPConfig:
    """FSDP.

    Attributes:
        sharding (str): ``full`` or ``hybrid_full`` (shard within a node, replicate across nodes).
    """

    sharding: str = "hybrid_full"

    def __post_init__(self) -> None:
        check_choice("fsdp.sharding", self.sharding, ("full", "hybrid_full"))


@dataclass(frozen=True)
class EMAConfig:
    """The EMA of the generator.

    Attributes:
        decay (float): 0 for none.
        start_step (int): the EMA starts at this step.
    """

    decay: float = 0.999
    start_step: int = 300

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError(f"ema.decay must lie in [0, 1), got {self.decay!r}")
        check_at_least("ema.start_step", self.start_step, 0)


@dataclass(frozen=True)
class CheckpointConfig:
    """Initialisation and saving.

    Attributes:
        init (str | None): the checkpoint the stage starts from (docs/training.md), ``wan22`` for
            the backbone of ``model.wan22_root``.
        interval (int): save every this many steps, and at the end (0: at the end only).
        keep (int): complete checkpoints kept (0: all).
        keep_shards (int): complete checkpoints that keep their resume files (0: all).
    """

    init: str | None = None
    interval: int = 500
    keep: int = 0
    keep_shards: int = 0

    def __post_init__(self) -> None:
        check_at_least("checkpoint.interval", self.interval, 0)
        check_at_least("checkpoint.keep", self.keep, 0)
        check_at_least("checkpoint.keep_shards", self.keep_shards, 0)


@dataclass(frozen=True)
class DistillationConfig:
    """The distillation stage: the frozen teacher and the critic's initialisation, both the stage-2
    model at step 25,000 in the paper.

    Attributes:
        teacher (str | None): the teacher's checkpoint.
        critic (str | None): the critic's checkpoint.
    """

    teacher: str | None = None
    critic: str | None = None


@dataclass(frozen=True)
class ValidationConfig:
    """In-training validation (docs/training.md, "Evaluation"): the eval64 windows scored with the
    EMA weights, an ``event: validation`` row of ``metrics.jsonl``.

    Attributes:
        index (str | None): ``eval64_index.jsonl``.
        index_sha256 (str | None): the sha256 the index must have; unset, the paper's index
            (:data:`worldcast.engine.evaluation.protocols.UNIPC`).
        interval (int): validate every this many steps; 0 (the default) for none, 1000 in the
            paper runs.
    """

    index: str | None = None
    index_sha256: str | None = None
    interval: int = 0

    def __post_init__(self) -> None:
        if self.index_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.index_sha256):
            raise ValueError(
                "validation.index_sha256 must be a sha256, 64 lowercase hex digits, got"
                f" {self.index_sha256!r}"
            )
        check_at_least("validation.interval", self.interval, 0)


@dataclass(frozen=True)
class TrainConfig:
    """One training run (:func:`load_train_config`, :func:`paper_config`)."""

    run: TrainRunConfig = field(default_factory=TrainRunConfig)
    model: TrainModelConfig = field(default_factory=TrainModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @property
    def stage(self) -> Stage:
        return STAGES[self.run.stage]

    def __post_init__(self) -> None:
        stage = self.stage
        if stage.recipe == "distillation" and self.optim.clip != "global":
            raise ValueError(
                f"optim.clip must be 'global' in stage {self.run.stage}, got {self.optim.clip!r}:"
                " the distillation clips each of its models globally"
            )
        if self.validation.interval:
            if stage.recipe == "distillation" or stage.latent_frames != WINDOW_LATENT_FRAMES:
                raise ValueError(
                    f"validation scores {WINDOW_LATENT_FRAMES}-latent windows in 20 steps: stages"
                    " 1_long to 3"
                )
            if not self.validation.index:
                raise ValueError("validation.interval needs validation.index")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        """The config of nested mappings over the settings of their ``run.stage``
        (:data:`STAGE_SETTINGS`), validated; unknown keys are errors."""
        stage = (data.get("run") or {}).get("stage")
        return build(cls, merge(STAGE_SETTINGS.get(str(stage), {}), data))


#: The settings of each paper run that differ from the defaults above.
STAGE_SETTINGS: dict[str, dict[str, Any]] = {
    "1": {
        "run": {"name": "worldcast-stage1", "seed": 20260714, "max_steps": 34000},
        "data": {"batch_size": 4},
        "optim": {"lr": 1e-5, "clip": "global"},
        "ema": {"decay": 0.0, "start_step": 0},
        "checkpoint": {"init": "wan22", "interval": 1000},
    },
    "1_long": {
        "run": {"name": "worldcast-stage1-long", "seed": 20260714, "max_steps": 6000},
        "data": {"batch_size": 2},
        "optim": {"lr": 1e-5, "clip": "global"},
        "ema": {"decay": 0.0, "start_step": 0},
        "checkpoint": {"interval": 1000},
    },
    "2": {"run": {"name": "worldcast-stage2"}, "optim": {"clip": "global"}},
    "2s": {
        "run": {"name": "worldcast-stage2s", "seed": 20260902, "max_steps": 5000},
        "data": {"num_workers": 14},
        "optim": {"grad_accum_steps": 2},
        "fsdp": {"sharding": "full"},
        "checkpoint": {"interval": 125},
    },
    "3": {
        "run": {"name": "worldcast-stage3", "seed": 20260902, "max_steps": 5000},
        "data": {"num_workers": 14},
        "fsdp": {"sharding": "full"},
        "checkpoint": {"interval": 125},
    },
    "3_noscene": {
        "run": {"name": "worldcast-stage3-noscene", "seed": 20260901, "max_steps": 5000},
        "optim": {"grad_accum_steps": 2},
    },
    "4": {
        "run": {"name": "worldcast-stage4", "seed": 20260912, "max_steps": 600, "log_interval": 5},
        "data": {"num_workers": 14},
        "optim": {"lr": 2e-6, "clip": "global"},
        "fsdp": {"sharding": "full"},
        "ema": {"decay": 0.99, "start_step": 200},
        "checkpoint": {"interval": 100},
    },
    "4_noscene": {
        "run": {
            "name": "worldcast-stage4-noscene",
            "seed": 20260912,
            "max_steps": 600,
            "log_interval": 5,
        },
        "optim": {"lr": 2e-6, "grad_accum_steps": 2, "clip": "global"},
        "ema": {"decay": 0.99, "start_step": 200},
        "checkpoint": {"interval": 100},
    },
}


def paper_config(stage: str, overrides: Mapping[str, Any] | None = None) -> TrainConfig:
    """The paper run's config of ``stage`` without paths, with dotted ``overrides``
    (``{"run.seed": 1}``) over it."""
    return TrainConfig.from_dict(set_dotted({"run": {"stage": stage}}, overrides or {}))


def load_train_config(paths: Paths, overrides: Mapping[str, Any] | None = None) -> TrainConfig:
    """YAML files merged in order (``inherit: <path>`` pulls in a base file first), then the
    dotted ``overrides`` (``{"run.seed": 1}``), over the settings of the resulting ``run.stage``;
    validated."""
    return load(paths, overrides, TrainConfig.from_dict)
