"""A training stage's generator config, the weights it takes of a checkpoint and its player state
field."""

import pytest
import torch

from tests.engine.training.support import TINY_DIMS as DIMS
from worldcast.config.training import STAGES, paper_config
from worldcast.data.latents import BLOCK
from worldcast.engine.stage import built_state, field_config, generator_config
from worldcast.modeling.wan22.model import WORLDCAST_MODULES, WorldCastGenerator
from worldcast.player_state import PlayerStateFieldConfig


def test_the_stage_decides_the_modules_and_the_ablation_switches():
    stage1 = generator_config(paper_config("1"))
    assert stage1.state_injector is None and not stage1.ray_embedding
    assert stage1.visibility_probe is None and not stage1.observer_signals

    stage2 = generator_config(paper_config("2"))
    assert stage2.state_injector.dit_block == 1 and stage2.field_downsample == 1
    assert not stage2.ray_embedding and stage2.visibility_probe is None

    stage3 = generator_config(paper_config("3"))
    assert stage3.ray_embedding and stage3.observer_signals
    assert stage3.visibility_probe is not None
    score_model = generator_config(paper_config("4"), score_model=True)
    assert score_model == stage2  # the distillation's teacher and critic are the stage-2 model

    ablated = paper_config("2", {"model.state_injector_block": 22, "model.field_downsample": 2})
    late = generator_config(ablated)
    assert late.state_injector.dit_block == 22 and late.field_downsample == 2


def test_model_dims_override_nested_dimensions():
    dims = {"dim": 32, "num_heads": 2, "controls.hidden": 64}
    cfg = paper_config("3", {"model.dims": dims})
    config = generator_config(cfg)
    assert config.dim == 32 and config.controls.hidden == 64
    # a section the stage does not build is left alone
    stage1 = paper_config("1", {"model.dims": {"visibility_probe.hidden": 16}})
    assert generator_config(stage1).visibility_probe is None


def test_the_stages_add_the_generators_conditioning_modules():
    added = {module for stage in STAGES.values() for module in stage.new_modules}
    assert added == set(WORLDCAST_MODULES)


def test_a_checkpoint_loses_the_optional_modules_its_generator_does_not_build():
    stage2s, stage2 = (generator_config(paper_config(s, {"model.dims": DIMS})) for s in ("2s", "2"))
    with torch.device("meta"):
        full = dict.fromkeys(WorldCastGenerator(stage2s).state_dict(), torch.zeros(1))
        expected = set(WorldCastGenerator(stage2).state_dict())
    assert set(built_state(full, stage2)) == expected
    assert set(built_state(full, stage2s)) == set(full)
    # only those modules are dropped: any other key stays, for the loader to refuse
    other = {**full, "depth_head.weight": torch.zeros(1)}
    assert "depth_head.weight" in built_state(other, stage2)
    stage1 = generator_config(paper_config("1", {"model.dims": DIMS}))
    assert any(key.startswith("state_injector.") for key in built_state(full, stage1))


@pytest.mark.parametrize("stage", ["2", "3"])
def test_the_fields_confidence_restarts_at_each_block_of_the_attention(stage):
    cfg = paper_config(stage)
    whole = field_config(cfg, bidirectional=True)
    assert (whole.frames_per_block, whole.first_frame_alone) == (41, False)
    causal = field_config(cfg, bidirectional=False)
    assert (causal.frames_per_block, causal.first_frame_alone) == (BLOCK, True)
    assert causal == PlayerStateFieldConfig()
    ungated = paper_config(stage, {"model.visibility_gate": False})
    assert field_config(ungated, bidirectional=False).confidence_floor == 1.0
