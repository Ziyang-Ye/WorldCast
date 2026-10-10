"""Stage 3 on a tiny model (CPU, FSDP on a gloo group): teacher forcing with and without scene
state."""

import math

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
from worldcast.data.latents import BLOCK
from worldcast.engine.training.build import build_trainer_parts
from worldcast.engine.training.losses import sample_flow_matching
from worldcast.engine.training.recipes.teacher_forcing import TeacherForcingTrainer


def _trainer(tmp_path, stage: str) -> TeacherForcingTrainer:
    init, prompt = tmp_path / "init.pt", tmp_path / "prompt.safetensors"
    write_init_checkpoint(init, paper_config(stage, {"model.dims": RUN_DIMS}))
    write_prompt_embedding(prompt)
    cfg = paper_config(
        stage,
        {
            "model.dims": RUN_DIMS,
            "model.attention": "sdpa",
            "run.max_steps": 1,
            "checkpoint.init": str(init),
            "data.prompt_embedding": str(prompt),
            "data.num_workers": 0,
            "optim.grad_accum_steps": 1,
        },
    )
    target = 37 if cfg.stage.scene_state else None
    parts = build_trainer_parts(cfg, D.DistInfo(), dataset=ItemDataset(4, target_start=target))
    return TeacherForcingTrainer(cfg, parts)


@pytest.mark.parametrize("stage", ["3", "3_noscene"])
def test_a_teacher_forcing_step(tmp_path, gloo, stage):
    trainer = _trainer(tmp_path, stage)
    scene_state = trainer.cfg.stage.scene_state
    live = D.live_module(trainer.generator)
    before = {n: p.detach().clone() for n, p in live.named_parameters()}
    metrics = trainer.train_step()
    assert trainer.step == 1 and math.isfinite(metrics["loss"]) and metrics["grad_norm"] > 0
    assert ("visibility_loss" in metrics) == scene_state
    # each parameter group is clipped on its own
    groups = {"backbone", "state_injector"}
    if scene_state:
        groups |= {"visibility_probe", "observer_signals", "ray_embedding"}
    assert {k.removeprefix("grad_norm_") for k in metrics if k.startswith("grad_norm_")} == groups
    assert [n for n, p in live.named_parameters() if (p != before[n]).any()]

    # one timestep per block under block-causal attention; the context copy is noised with scene
    # state, and without it the field is added to the clean copy as well
    clean = torch.randn(1, 1 + 2 * BLOCK, 48, 4, 4)
    sample = sample_flow_matching(
        clean,
        trainer.scheduler,
        n_pin=1,
        frames_per_timestep=BLOCK,
        first_frame_alone=True,
        timestep_dtype=torch.float32,
        context_noise=scene_state,
    )
    assert sample.timestep[0, 0] == 0 and len(sample.timestep[0, 1:].unique()) <= 2
    kwargs = trainer.forward_kwargs(sample)
    assert set(kwargs) == {"context_latents", "context_timestep", "field_on_context"}
    assert kwargs["field_on_context"] == (not scene_state)
    if scene_state:
        context_timestep = kwargs["context_timestep"]
        assert bool(((context_timestep >= 16) & (context_timestep < 32)).all())
        assert not torch.equal(kwargs["context_latents"], clean)
    else:
        assert kwargs["context_timestep"] is None and torch.equal(kwargs["context_latents"], clean)
