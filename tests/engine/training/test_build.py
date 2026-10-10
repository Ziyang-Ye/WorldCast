"""Building a stage's run: the trainer of its recipe, the generator from a checkpoint, the score
models, and a run of one process without a process group."""

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from tests.engine.training.support import (
    RUN_DIMS,
    ItemDataset,
    write_init_checkpoint,
    write_prompt_embedding,
)
from worldcast import distributed as D
from worldcast.config.training import paper_config
from worldcast.engine.checkpoint.training import find_latest_checkpoint
from worldcast.engine.stage import generator_config
from worldcast.engine.training import build
from worldcast.engine.training.recipes.bidirectional import BidirectionalTrainer
from worldcast.modeling.build import read_generator
from worldcast.modeling.wan22.model import WorldCastGenerator


class Recorded:
    """Stands in for a trainer: keeps what it was built from and resumed with."""

    def __init__(self, cfg, parts) -> None:
        self.cfg, self.parts, self.resumed = cfg, parts, None

    def resume(self, directory) -> None:
        self.resumed = directory


@pytest.mark.parametrize(
    "stage, trainer, parts",
    [
        ("1", "BidirectionalTrainer", "build_trainer_parts"),
        ("2s", "BidirectionalTrainer", "build_trainer_parts"),
        ("3", "TeacherForcingTrainer", "build_trainer_parts"),
        ("3_noscene", "TeacherForcingTrainer", "build_trainer_parts"),
        ("4", "DistillationTrainer", "build_distillation_parts"),
    ],
)
def test_the_stage_recipe_picks_the_trainer_and_its_parts(stage, trainer, parts, monkeypatch):
    for name in ("BidirectionalTrainer", "TeacherForcingTrainer", "DistillationTrainer"):
        monkeypatch.setattr(build, name, type(name, (Recorded,), {}))
    for name in ("build_trainer_parts", "build_distillation_parts"):
        monkeypatch.setattr(build, name, lambda cfg, info, resume, name=name: (name, resume))
    cfg, info = paper_config(stage), D.DistInfo()
    fresh = build.build_trainer(cfg, info)
    assert type(fresh).__name__ == trainer and fresh.parts == (parts, None)
    assert fresh.cfg is cfg and fresh.resumed is None
    resumed = build.build_trainer(cfg, info, resume=Path("checkpoint_model_000100"))
    assert resumed.parts == (parts, resumed.resumed) == (parts, Path("checkpoint_model_000100"))


def test_the_bucket_weights_of_the_config_reach_the_bucket_index(monkeypatch, tmp_path):
    class Stop(Exception):
        pass

    def read_bucket_index(bucket_dir, exclusion, bucket_weights):
        raise Stop(bucket_dir, exclusion, bucket_weights)

    monkeypatch.setattr(build, "read_bucket_index", read_bucket_index)
    names = ("bucket_dir", "dataset_root", "media_index", "latent_cache_root")
    names += ("visibility_label_root", "observer_signal_label_root")
    paths = {f"data.{name}": str(tmp_path / name) for name in names}
    for weights in (None, [1.0, 2.0, 3.0, 4.0, 5.0]):
        cfg = paper_config("2", {**paths, "data.bucket_weights": weights})
        with pytest.raises(Stop) as stopped:
            build.build_dataset(cfg)
        assert stopped.value.args == (
            str(tmp_path / "bucket_dir"),
            frozenset(),
            None if weights is None else tuple(weights),
        )


# ------------------------------------------------------------------------------------ the generator
def _state(stage: str) -> tuple[dict[str, torch.Tensor], object]:
    config = generator_config(paper_config(stage, {"model.dims": RUN_DIMS}))
    with torch.device("meta"):
        keys = WorldCastGenerator(config).state_dict()
    return {key: torch.zeros(value.shape) for key, value in keys.items()}, config


def test_only_the_new_modules_of_a_stage_may_be_missing_and_each_as_a_whole():
    state, config = _state("2s")
    scene = ("observer_signals.", "ray_embedding.", "visibility_probe.")
    stage2 = {key: value for key, value in state.items() if not key.startswith(scene)}
    new = ("observer_signals", "ray_embedding", "visibility_probe")
    generator, fresh = build.build_training_generator(config, stage2, new_modules=new)
    assert fresh == list(new) and generator.training
    assert bool((generator.blocks[0].self_attn.q.weight == 0).all())  # loaded, not drawn
    # a complete checkpoint starts nothing fresh
    assert build.build_training_generator(config, state, new_modules=new)[1] == []
    with pytest.raises(KeyError, match="the checkpoint lacks generator keys"):
        build.build_training_generator(config, stage2, new_modules=("ray_embedding",))
    partial = {key: value for key, value in state.items() if key != "ray_embedding.mlp.0.weight"}
    with pytest.raises(KeyError, match=r"holds the new modules \['ray_embedding'\] in part"):
        build.build_training_generator(config, partial, new_modules=new)
    with pytest.raises(ValueError, match=r"new modules \['depth_head'\] are not among"):
        build.build_training_generator(config, state, new_modules=("depth_head",))


def test_a_score_model_is_the_stage_2_model(tmp_path):
    cfg = paper_config("4", {"model.dims": RUN_DIMS})
    stage2s = paper_config("2s", {"model.dims": RUN_DIMS})
    checkpoint = write_init_checkpoint(tmp_path / "stage2s.pt", stage2s)
    state = read_generator(checkpoint)
    assert any(key.startswith("ray_embedding.") for key in state)
    critic = build.build_score_model(cfg, state, trainable=True)
    assert critic.ray_embedding is None and critic.observer_signals is None
    assert critic.visibility_probe is None and critic.state_injector is not None
    assert all(p.requires_grad for p in critic.parameters())
    teacher = build.build_score_model(cfg, state, trainable=False)
    assert not any(p.requires_grad for p in teacher.parameters())
    assert all(torch.equal(v, state[k]) for k, v in teacher.state_dict().items())


def test_a_run_starts_from_a_checkpoint(tmp_path):
    cfg = paper_config("2", {"model.dims": RUN_DIMS})
    with pytest.raises(ValueError, match="stage 2 starts from a checkpoint: set checkpoint.init"):
        build.initial_generator_state(cfg, None)
    wan22 = dataclasses.replace(cfg, checkpoint=dataclasses.replace(cfg.checkpoint, init="wan22"))
    with pytest.raises(ValueError, match="reads the backbone of model.wan22_root"):
        build.initial_generator_state(wan22, None)


# ---------------------------------------------------------------------- one process, no torchrun
def _stage2(tmp_path: Path, **overrides):
    init = write_init_checkpoint(
        tmp_path / "stage1.pt",
        paper_config("2", {"model.dims": RUN_DIMS}),
        drop=("state_injector.",),
    )
    settings = {
        "model.dims": RUN_DIMS,
        "model.attention": "sdpa",
        "run.output_dir": str(tmp_path / "run"),
        "run.max_steps": 2,
        "checkpoint.init": str(init),
        "checkpoint.interval": 0,
        "data.prompt_embedding": str(write_prompt_embedding(tmp_path / "prompt.safetensors")),
        "data.num_workers": 0,
        **overrides,
    }
    return paper_config("2", settings)


def test_a_process_without_a_process_group_trains_unwrapped(tmp_path):
    info = D.DistInfo()
    assert not info.initialized  # this module runs outside the tests' gloo group
    cfg = _stage2(tmp_path)
    parts = build.build_trainer_parts(cfg, info, dataset=ItemDataset(2))
    assert not D.is_fsdp(parts.generator)
    trainer = BidirectionalTrainer(cfg, parts)
    trainer.fit()
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").open()]
    assert [row["step"] for row in rows] == [1, 2] and all(math.isfinite(r["loss"]) for r in rows)
    directory = find_latest_checkpoint(cfg.run.output_dir)
    assert directory == tmp_path / "run" / "checkpoint_model_000002"
    parts = build.build_trainer_parts(cfg, info, resume=directory, dataset=ItemDataset(2))
    resumed = BidirectionalTrainer(cfg, parts)
    resumed.resume(directory)
    assert resumed.step == 2
    live, saved = trainer.generator.state_dict(), resumed.generator.state_dict()
    assert all(torch.equal(live[key], saved[key]) for key in live)
