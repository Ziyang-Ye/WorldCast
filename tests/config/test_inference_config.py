"""The inference config: the YAML, the defaults, the derived window and the refusals."""

from pathlib import Path

import pytest

from worldcast.config.inference import (
    InferenceConfig,
    config_from_dict,
    config_to_dict,
    load_config,
    with_overrides,
)

REPO = Path(__file__).resolve().parents[2]


def test_yaml_equals_defaults():
    cfg = load_config(REPO / "configs" / "infer" / "worldcast_4step.yaml")
    assert cfg == InferenceConfig()
    assert config_from_dict(config_to_dict(cfg)) == cfg


def test_derived_window_numbers():
    w = InferenceConfig().window
    assert (
        w.first_target,
        w.plain_prefix_latents,
        w.window_latents,
        w.window_latents_without_memory,
        w.max_cached_latents,
    ) == (25, 24, 21, 17, 25)
    assert InferenceConfig().model.frame_seq_length == 252


@pytest.mark.parametrize(
    "override",
    [
        {"run.unknown": 1},
        {"nosection.x": 1},
    ],
)
def test_unknown_override_refused(override):
    with pytest.raises(KeyError):
        with_overrides(InferenceConfig(), override)


@pytest.mark.parametrize(
    "data, exc",
    [
        ({"run": {"seed": "1"}}, ValueError),
        ({"run": {"seed": True}}, ValueError),
        ({"run": {"latents": 442}}, ValueError),
        ({"window": {"recent": 8, "min_target_latent": 4}}, ValueError),
        ({"window": {"kv_cache_latents": 24}}, ValueError),
        ({"memory": {"k": 2}}, NotImplementedError),
        ({"memory": {"start_latent": 29}}, ValueError),
        ({"sampler": {"context_noise": 0}}, ValueError),
        ({"sampler": {"model_input_dtype": "float16"}}, ValueError),
        ({"data": {"camera_encoding": "clip"}}, NotImplementedError),
        ({"sampler": {"framewise_condition_axes": {"not_an_input": 1}}}, ValueError),
        ({"extra": {}}, ValueError),
    ],
)
def test_invalid_configs_refused(data, exc):
    with pytest.raises(exc):
        config_from_dict(data)
