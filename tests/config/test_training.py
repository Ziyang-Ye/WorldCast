"""The training configs: the stage table, the stage files that name a stage, the ablation overlays,
the overrides and the refusals."""

from pathlib import Path

import pytest
import yaml

from worldcast.config import STAGES, TrainConfig, config_to_dict, load_train_config, paper_config
from worldcast.config.loader import parse_overrides

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "train"
STAGE_FILES = {stage: f"stage{stage}.yaml" for stage in STAGES}
#: ablation file -> the one mechanism key it changes (besides the run's name and length and the
#: 32 x 2 topology)
ABLATIONS = {
    "no_visibility": ("model", "visibility_gate", False),
    "no_foreground_weight": ("loss", "foreground_weight", False),
    "late_injection": ("model", "state_injector_block", 22),
    "coarse_field": ("model", "field_downsample", 2),
}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("collision_meshes", "dims"):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_the_stage_table():
    assert list(STAGES) == ["1", "1_long", "2", "2s", "3", "3_noscene", "4", "4_noscene"]
    assert {name: stage.recipe for name, stage in STAGES.items()} == {
        "1": "bidirectional",
        "1_long": "bidirectional",
        "2": "bidirectional",
        "2s": "bidirectional",
        "3": "teacher_forcing",
        "3_noscene": "teacher_forcing",
        "4": "distillation",
        "4_noscene": "distillation",
    }
    assert [name for name, stage in STAGES.items() if stage.scene_state] == ["2s", "3", "4"]
    assert [name for name, stage in STAGES.items() if not stage.player_state_field] == [
        "1",
        "1_long",
    ]
    assert {stage.latent_frames for stage in STAGES.values()} == {21, 41}
    assert STAGES["1"].latent_frames == 21


def test_every_stage_has_its_file():
    assert sorted(p.name for p in CONFIGS.glob("stage*.yaml")) == sorted(STAGE_FILES.values())


@pytest.mark.parametrize("stage", STAGES)
def test_a_stage_file_names_its_stage_and_sets_nothing(stage):
    """The stage table is the one source of a stage's settings."""
    path = CONFIGS / STAGE_FILES[stage]
    assert yaml.safe_load(path.read_text()) == {"run": {"stage": stage}}
    assert load_train_config(path) == paper_config(stage)


def test_the_stage_settings_overlay_the_defaults():
    stage2, stage4 = paper_config("2"), paper_config("4")
    assert (stage2.run.max_steps, stage2.optim.lr, stage2.optim.clip) == (25000, 2.8e-5, "global")
    assert (stage2.run.seed, stage2.ema.decay, stage2.fsdp.sharding) == (
        20260829,
        0.999,
        "hybrid_full",
    )
    assert (stage4.run.max_steps, stage4.optim.lr, stage4.ema.decay) == (600, 2e-6, 0.99)
    assert (stage4.run.seed, stage4.ema.start_step, stage4.fsdp.sharding) == (20260912, 200, "full")
    assert stage4.stage.recipe == "distillation" and stage4.stage.scene_state
    assert not paper_config("4_noscene").stage.scene_state
    # a file's own keys win over the stage's settings
    assert TrainConfig.from_dict({"run": {"stage": "4", "max_steps": 10}}).run.max_steps == 10


def test_the_settings_follow_run_stage_wherever_it_is_set():
    stage2 = CONFIGS / "stage2.yaml"
    assert load_train_config(stage2, {"run.stage": "2s"}) == paper_config("2s")
    assert load_train_config(stage2, parse_overrides(["run.stage=4"])) == paper_config("4")
    assert paper_config("2", {"run.stage": "1"}) == paper_config("1")
    # the other overrides still win over the new stage's settings
    cfg = load_train_config(stage2, {"run.stage": "4", "optim.lr": 1e-4})
    assert (cfg.run.stage, cfg.optim.lr, cfg.run.max_steps, cfg.ema.decay) == ("4", 1e-4, 600, 0.99)


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_ablation_overlays_change_one_mechanism_key(name):
    cfg = load_train_config(CONFIGS / "ablations" / f"{name}.yaml")
    section, key, value = ABLATIONS[name]
    assert getattr(getattr(cfg, section), key) == value
    got, want = _flat(config_to_dict(cfg)), _flat(config_to_dict(paper_config("2")))
    changed = {k for k in got if got[k] != want[k]}
    # the ablation runs clipped each group on its own, the stage-2 run the global norm
    expected = {"run.name", "run.max_steps", "optim.grad_accum_steps", "optim.clip"}
    assert changed == expected | {f"{section}.{key}"}
    assert (cfg.run.max_steps, cfg.optim.grad_accum_steps, cfg.optim.clip) == (6000, 2, "per_group")


def test_an_override_sets_one_key_of_a_mapping_valued_setting():
    cfg = paper_config(
        "3",
        {
            "optim.lr": 1e-5,
            "data.bucket_dir": "/data/buckets",
            "data.collision_meshes.de_dust2": "/meshes/de_dust2.glb",
            "model.dims.dim": 384,
        },
    )
    assert (cfg.optim.lr, cfg.data.bucket_dir) == (1e-5, "/data/buckets")
    assert cfg.data.collision_meshes == {"de_dust2": "/meshes/de_dust2.glb"}
    assert cfg.model.dims == {"dim": 384}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"optim.learning_rate": 1e-4}, "unknown config keys in optim"),
        ({"run.stage": "5"}, "run.stage must be one of"),
        ({"optim.clip": "none"}, "optim.clip must be one of"),
        ({"fsdp.sharding": "none"}, "fsdp.sharding must be one of"),
        ({"run.max_steps": -5}, "run.max_steps must be >= 0"),
        ({"run.log_interval": -1}, "run.log_interval must be >= 0"),
        ({"optim.lr": -1.0}, "optim.lr must be >= 0"),
        ({"optim.grad_accum_steps": 0}, "optim.grad_accum_steps must be >= 1"),
        ({"data.batch_size": 0}, "data.batch_size must be >= 1"),
        ({"data.num_workers": -3}, "data.num_workers must be >= 0"),
        ({"ema.decay": 1.0}, "ema.decay must lie in"),
        ({"ema.start_step": -1}, "ema.start_step must be >= 0"),
        ({"checkpoint.interval": -1}, "checkpoint.interval must be >= 0"),
        ({"checkpoint.keep": -1}, "checkpoint.keep must be >= 0"),
        ({"model.field_downsample": 0}, "model.field_downsample must be >= 1"),
        ({"validation.interval": 1000}, "validation.interval needs validation.index"),
    ],
)
def test_invalid_configs_are_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        paper_config("2", overrides)


def test_the_distillation_takes_the_global_clip_only():
    """Stage 4 clips each of its models globally: another ``optim.clip`` would be ignored, so it
    is refused."""
    for stage in ("4", "4_noscene"):
        assert paper_config(stage).optim.clip == "global"
        message = f"optim.clip must be 'global' in stage {stage}, got 'per_group'"
        with pytest.raises(ValueError, match=message):
            paper_config(stage, {"optim.clip": "per_group"})
    # the other stages clip either way
    assert [paper_config("3", {"optim.clip": c}).optim.clip for c in ("global", "per_group")] == [
        "global",
        "per_group",
    ]


def test_validation_scores_ten_second_windows_in_twenty_steps():
    validated = {"validation.interval": 1000, "validation.index": "eval64_index.jsonl"}
    assert paper_config("3", validated).validation.interval == 1000
    for stage in ("1", "4"):
        with pytest.raises(ValueError, match="validation scores 41-latent windows"):
            paper_config(stage, validated)


def test_the_bucket_weights_and_the_validation_digest_default_to_the_papers():
    from worldcast.config.training import BUCKET_FILES
    from worldcast.data.training import BUCKET_NAMES, BUCKET_WEIGHTS
    from worldcast.engine.evaluation.protocols import UNIPC

    assert BUCKET_FILES == len(BUCKET_NAMES) == len(BUCKET_WEIGHTS)
    for stage in STAGES:
        cfg = paper_config(stage)
        assert cfg.data.bucket_weights is None and cfg.validation.index_sha256 is None
    assert UNIPC.index_sha256 == "a15580a409d02b580e7220073e1ff5f037724c49e749b86905e335fe80030c49"


def test_the_bucket_weights_and_the_validation_digest_are_settings():
    digest = "0123456789abcdef" * 4
    cfg = paper_config(
        "3",
        parse_overrides(
            ["data.bucket_weights=[1, 0.5, 2.25, 1, 1e-3]", f"validation.index_sha256={digest}"]
        ),
    )
    assert cfg.data.bucket_weights == (1.0, 0.5, 2.25, 1.0, 0.001)
    assert all(isinstance(w, float) for w in cfg.data.bucket_weights)
    assert cfg.validation.index_sha256 == digest
    assert TrainConfig.from_dict(config_to_dict(cfg)) == cfg


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"data.bucket_weights": [1] * 4}, "hold one weight for each of the 5 bucket files, got 4"),
        ({"data.bucket_weights": [1] * 6}, "hold one weight for each of the 5 bucket files, got 6"),
        ({"data.bucket_weights": []}, "hold one weight for each of the 5 bucket files, got 0"),
        (
            {"data.bucket_weights": [1, 1, 0, 1, 1]},
            r"data.bucket_weights must be numbers > 0, got \[1.0, 1.0, 0.0",
        ),
        ({"data.bucket_weights": [1, -0.5, 1, 1, 1]}, "data.bucket_weights must be numbers > 0"),
        (
            {"data.bucket_weights": [1, 1, float("nan"), 1, 1]},
            "data.bucket_weights must be numbers > 0",
        ),
        (
            {"data.bucket_weights": [1, 1, float("inf"), 1, 1]},
            "data.bucket_weights must be numbers > 0",
        ),
        ({"data.bucket_weights": [1, 1, "heavy", 1, 1]}, "data.bucket_weights must be a number"),
        ({"data.bucket_weights": [1, 1, True, 1, 1]}, "data.bucket_weights must be a number"),
        ({"data.bucket_weights": 1.0}, "data.bucket_weights must be a list"),
        ({"validation.index_sha256": "a15580a4"}, "validation.index_sha256 must be a sha256"),
        ({"validation.index_sha256": "A" * 64}, "validation.index_sha256 must be a sha256"),
        ({"validation.index_sha256": "g" * 64}, "validation.index_sha256 must be a sha256"),
        ({"validation.index_sha256": "a" * 65}, "validation.index_sha256 must be a sha256"),
        ({"validation.index_sha256": 7}, "validation.index_sha256 must be a sha256"),
    ],
)
def test_wrong_bucket_weights_and_digests_are_refused_when_the_config_is_built(overrides, message):
    with pytest.raises(ValueError, match=message):
        paper_config("2", overrides)
