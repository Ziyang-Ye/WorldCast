"""The inference config: the defaults, the path files and the refusals."""

from pathlib import Path

import pytest

from worldcast.config import InferenceConfig, config_to_dict, load_config
from worldcast.config.inference import MEMORY_BOUND, PLAYER_STATE_SOURCES, PathsConfig
from worldcast.data.latents import BLOCK

REPO = Path(__file__).resolve().parents[2]


def test_the_defaults_are_the_papers_settings():
    cfg = InferenceConfig()
    assert (cfg.run.seed, cfg.run.latent_frames, cfg.run.device) == (20260917, 441, "cuda")
    assert (cfg.model.attention, cfg.player_state.source) == ("flash", "gt")
    assert cfg.scene_state.bound == MEMORY_BOUND == 64
    assert InferenceConfig.from_dict(config_to_dict(cfg)) == cfg


def test_the_player_states_come_from_the_recording_or_the_closed_loop():
    assert PLAYER_STATE_SOURCES == ("gt", "predicted")
    for source in PLAYER_STATE_SOURCES:
        cfg = InferenceConfig().with_overrides({"player_state.source": source})
        assert cfg.player_state.source == source
    with pytest.raises(ValueError, match="player_state.source"):
        InferenceConfig().with_overrides({"player_state.source": "unknown"})


def test_overrides(tmp_path):
    (tmp_path / "paths.yaml").write_text("paths: {out_dir: runs/a}\nrun: {seed: 3}\n")
    cfg = load_config(tmp_path / "paths.yaml", {"run.seed": 5, "model.attention": "sdpa"})
    assert (cfg.paths.out_dir, cfg.run.seed, cfg.model.attention) == ("runs/a", 5, "sdpa")
    again = cfg.with_overrides({"run.max_blocks": 2})
    assert (again.run.max_blocks, again.run.seed, again.paths.out_dir) == (2, 5, "runs/a")
    assert cfg.run.max_blocks == 0  # the config itself is unchanged


def test_the_example_path_files_hold_only_config_keys():
    examples = REPO / "examples"
    for paths in [examples / "data_paths.yaml", *sorted(examples.glob("data/*/config.yaml"))]:
        assert load_config(paths).paths.observer_signal_label_root, paths


def test_an_unset_path_is_named():
    paths = PathsConfig(checkpoint="generator.safetensors")
    paths.require("checkpoint")
    with pytest.raises(ValueError, match="not set: paths.depth_head, paths.out_dir"):
        paths.require("checkpoint", "depth_head", "out_dir")


def test_a_run_is_the_first_frame_and_whole_blocks():
    """The config spells the block's four latent frames itself (it imports no module above it)."""
    for k in (1, 6, 110):
        run = InferenceConfig.from_dict({"run": {"latent_frames": 1 + BLOCK * k}}).run
        assert run.latent_frames == 1 + BLOCK * k
    for bad in (1, BLOCK, 2 + BLOCK, 2 + BLOCK * 110):
        with pytest.raises(ValueError, match=r"must be 1 \+ 4k with k >= 1"):
            InferenceConfig.from_dict({"run": {"latent_frames": bad}})


@pytest.mark.parametrize("overrides", [{"run.unknown": 1}, {"nosection.x": 1}])
def test_unknown_override_refused(overrides):
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config((), overrides)
    with pytest.raises(ValueError, match="unknown config keys"):
        InferenceConfig().with_overrides(overrides)


@pytest.mark.parametrize(
    "data",
    [
        {"run": {"seed": "1"}},
        {"run": {"seed": True}},
        {"run": {"latent_frames": 442}},
        {"run": {"latent_frames": 1}},
        {"run": {"index_row": -1}},
        {"run": {"max_blocks": -1}},
        {"player_state": {"source": "oracle"}},
        {"world_state": {"wait_s": 0}},
        {"scene_state": {"bound": 0}},
        {"window": {"recent": 8}},  # the paper's window is not a setting
        {"extra": {}},
    ],
)
def test_invalid_configs_refused(data):
    with pytest.raises(ValueError):
        InferenceConfig.from_dict(data)
