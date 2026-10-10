"""Stage 4 (distribution matching distillation) on a tiny model (CPU, FSDP on a gloo group).

Six steps of each stage-4 config run once, on the paper's window layout (41 latents of 48 x 24 x 42)
with a 2-layer model built from the config: the generator from a stage-3 checkpoint, the teacher and
the critic from two different stage-2 checkpoints (the paper starts both from one, which makes the
first distribution matching gradient exactly 0 and the first update invisible), the prompt file and
the resumable data stream. The tests read what the six steps did: the update schedule, the EMA, the
draws of each rollout and the checkpoint. A run interrupted and resumed continues exactly.
"""

from dataclasses import dataclass
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
from worldcast.config.training import TrainConfig, paper_config
from worldcast.engine.training.build import build_distillation_parts
from worldcast.engine.training.recipes import distillation
from worldcast.engine.training.recipes.distillation import DistillationTrainer

STEPS = 6


@pytest.fixture(autouse=True)
def restore_deterministic_algorithms():
    """The stage-4 build switches deterministic algorithms on for the process (as the stage was
    trained, :func:`~worldcast.engine.training.build.start_run`); the rest of the test session
    keeps its own setting."""
    enabled = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled)


def _config(tmp: Path, stage: str, output: str = "run", **overrides) -> TrainConfig:
    """The stage's config on tiny dimensions, with its three checkpoints and its prompt file."""
    dims = {"model.dims": RUN_DIMS}
    paths = {
        "checkpoint.init": tmp / "stage3_ema.pt",
        "distillation.teacher": tmp / "stage2_ema.pt",
        "distillation.critic": tmp / "critic_init.pt",
    }
    if not paths["checkpoint.init"].exists():
        write_init_checkpoint(paths["checkpoint.init"], paper_config(stage, dims), seed=4)
        write_init_checkpoint(paths["distillation.teacher"], paper_config("2", dims), seed=5)
        write_init_checkpoint(paths["distillation.critic"], paper_config("2", dims), seed=6)
        write_prompt_embedding(tmp / "prompt.safetensors")
    settings = {
        **dims,
        **{key: str(path) for key, path in paths.items()},
        "model.attention": "sdpa",
        "run.output_dir": str(tmp / output),
        "data.prompt_embedding": str(tmp / "prompt.safetensors"),
        "data.num_workers": 0,
        "ema.start_step": 2,
        "optim.grad_accum_steps": 1,
        **overrides,
    }
    return paper_config(stage, settings)


def _trainer(cfg: TrainConfig, resume: Path | None = None) -> DistillationTrainer:
    windows = ItemDataset(4, target_start=29 if cfg.stage.scene_state else None)
    parts = build_distillation_parts(cfg, D.DistInfo(), resume=resume, dataset=windows)
    trainer = DistillationTrainer(cfg, parts)
    if resume is not None:
        trainer.resume(resume)
    return trainer


def _params(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in D.live_module(module).named_parameters()}


def _moved(before: dict[str, torch.Tensor], module: torch.nn.Module) -> list[str]:
    after = _params(module)
    return sorted(name for name in before if not torch.equal(before[name], after[name]))


@dataclass
class Run:
    """Six steps of a stage-4 run, as seen from outside.

    Attributes:
        trainer (DistillationTrainer): the trainer after the steps.
        deterministic (bool): deterministic algorithms were on once the run was built.
        metrics (list[dict]): each step's metrics.
        generator_moved (list[list[str]]): the generator parameters each step changed.
        critic_moved (list[list[str]]): the critic parameters each step changed.
        teacher_moved (list[str]): the teacher parameters the run changed.
        live (list[dict]): the generator's parameters after each step.
        ema (list[dict | None]): the EMA after each step.
        checkpoint (Path): the checkpoint saved after the last step.
    """

    trainer: DistillationTrainer
    deterministic: bool
    metrics: list[dict]
    generator_moved: list[list[str]]
    critic_moved: list[list[str]]
    teacher_moved: list[str]
    live: list[dict[str, torch.Tensor]]
    ema: list[dict[str, torch.Tensor] | None]
    checkpoint: Path


@pytest.fixture(scope="module", params=["4", "4_noscene"])
def run(request, tmp_path_factory, gloo) -> Run:
    enabled = torch.are_deterministic_algorithms_enabled()
    try:
        tmp = tmp_path_factory.mktemp("distillation")
        trainer = _trainer(_config(tmp, request.param, **{"run.max_steps": STEPS}))
        out = Run(
            trainer, torch.are_deterministic_algorithms_enabled(), [], [], [], [], [], [], tmp
        )
        teacher = _params(trainer.teacher)
        for _ in range(STEPS):
            generator, critic = _params(trainer.generator), _params(trainer.critic)
            out.metrics.append(trainer.train_step())
            out.generator_moved.append(_moved(generator, trainer.generator))
            out.critic_moved.append(_moved(critic, trainer.critic))
            out.live.append(_params(trainer.generator))
            ema = trainer.ema
            out.ema.append(None if ema is None else {k: v.clone() for k, v in ema.shadow.items()})
        out.teacher_moved = _moved(teacher, trainer.teacher)
        out.checkpoint = trainer.save()
        return out
    finally:
        torch.use_deterministic_algorithms(enabled)


def test_the_run_is_built_as_the_stage_was_trained(run):
    assert run.deterministic
    trainer, lr = run.trainer, run.trainer.cfg.optim.lr
    # the multi-tensor AdamW, every module of the generator at one learning rate
    assert {(g["lr"], g["foreach"]) for g in trainer.generator_optimizer.param_groups} == {
        (lr, True)
    }
    assert {(g["lr"], g["foreach"]) for g in trainer.critic_optimizer.param_groups} == {
        (4e-7, True)
    }
    assert trainer.context_noise == (16 if trainer.cfg.stage.scene_state else 0)


def test_the_generator_is_updated_every_fifth_step_and_the_critic_on_every_step(run):
    generator_steps = [True, False, False, False, False, True]
    assert [metrics["trained_generator"] for metrics in run.metrics] == generator_steps
    assert [bool(moved) for moved in run.generator_moved] == generator_steps
    assert all(run.critic_moved) and not run.teacher_moved
    for metrics in run.metrics:
        assert {"critic_loss", "critic_grad_norm", "lr_generator", "lr_critic"} <= set(metrics)
        updated = {"generator_loss", "generator_grad_norm"} <= set(metrics)
        assert updated == metrics["trained_generator"]
        values = [v for v in metrics.values() if isinstance(v, float)]
        assert all(torch.isfinite(torch.tensor(values)))
    # every module of the generator moves but the probe (no loss of its own in this stage) and the
    # ray embedding (the windows are ordinary: no camera reaches it)
    modules = {name.split(".")[1] for name in run.generator_moved[0]}
    assert not modules & {"visibility_probe", "ray_embedding"} and "blocks" in modules


def test_the_ema_starts_without_an_update_and_follows_the_generator_updates(run):
    # ema.start_step = 2: it starts at the end of the second step, as a copy of the weights
    assert [ema is None for ema in run.ema] == [True, False, False, False, False, False]
    started = run.ema[1]
    assert all(torch.equal(value, run.live[1][name].float()) for name, value in started.items())
    # the critic's steps leave it alone; the next generator step moves it by 1 % to the weights
    for later in run.ema[2:5]:
        assert all(torch.equal(later[name], value) for name, value in started.items())
    for name, value in started.items():
        expected = value.mul(0.99).add(run.live[5][name].float(), alpha=1.0 - 0.99)
        assert torch.equal(run.ema[5][name], expected)
    assert any(not torch.equal(run.ema[5][name], value) for name, value in started.items())


def test_each_rollout_draws_its_window_its_exit_step_and_its_score_timestep(run):
    for metrics in run.metrics:
        assert ("generator_exit_step" in metrics) == metrics["trained_generator"]
        for phase in ("generator", "critic") if metrics["trained_generator"] else ("critic",):
            (exit_step,) = metrics[f"{phase}_exit_step"]
            (score_timestep,) = metrics[f"{phase}_score_timestep"]
            (window,) = metrics[f"{phase}_dataset_index"]
            assert 0 <= exit_step < 4 and 20 <= score_timestep <= 980 and 0 <= window < 4
    # one exit step for all ten blocks of a rollout
    exits = run.trainer.exit_step(10)
    assert len(exits) == 10 and len(set(exits)) == 1 and 0 <= exits[0] < 4


def test_a_checkpoint_holds_the_generator_its_ema_and_the_critic(run):
    assert run.checkpoint.name == "checkpoint_model_000006"
    files = ["checkpoint.ready.json", "critic.pt", "model.pt", "rank_00000.pt"]
    assert sorted(path.name for path in run.checkpoint.iterdir()) == files
    model = torch.load(run.checkpoint / "model.pt", weights_only=True)
    critic = torch.load(run.checkpoint / "critic.pt", weights_only=True)
    assert {"generator", "generator_ema", "step"} <= set(model) and "critic" in critic
    assert all(key.startswith("model.") for key in model["generator_ema"])
    assert not any(key.startswith("model.visibility_probe") for key in critic["critic"])
    live = {n.removeprefix("generator."): p for n, p in run.live[-1].items()}
    assert all(torch.equal(model["generator"]["model." + n], p) for n, p in live.items())


def test_a_resumed_run_continues_exactly(tmp_path, gloo, monkeypatch):
    """Two steps, a checkpoint, a rebuild from it and a third step (a generator step, with the EMA
    started at the end of the second) give the weights, the EMA, the optimizer states, the data
    position and the RNG state of three uninterrupted steps."""
    monkeypatch.setattr(distillation, "GENERATOR_EVERY", 2)
    stage = "4"

    def state(trainer: DistillationTrainer) -> dict[str, dict[str, torch.Tensor]]:
        """The run's tensors by part: the weights, the EMA, the AdamW moments and the RNG."""
        tensors = {
            "generator": _params(trainer.generator),
            "critic": _params(trainer.critic),
            "ema": dict(trainer.ema.shadow),
            "rng": {"torch": torch.get_rng_state()},
        }
        for name, optimizer in trainer.optimizers().items():
            slots = optimizer.state_dict()["state"]
            tensors[f"{name} optimizer"] = {
                f"{index}.{key}": torch.as_tensor(value)
                for index, slot in slots.items()
                for key, value in slot.items()
            }
        return tensors

    continuous = _trainer(_config(tmp_path, stage, "continuous"))
    metrics = [continuous.train_step() for _ in range(3)]
    assert [m["trained_generator"] for m in metrics] == [True, False, True]
    want = state(continuous)
    interrupted = _trainer(_config(tmp_path, stage, "interrupted"))
    for _ in range(2):
        interrupted.train_step()
    directory = interrupted.save()
    del interrupted

    resumed = _trainer(_config(tmp_path, stage, "interrupted"), resume=directory)
    assert resumed.step == 2 and resumed.ema is not None
    last = resumed.train_step()
    for key in ("generator_loss", "critic_loss", "generator_exit_step", "critic_score_timestep"):
        assert last[key] == metrics[2][key], key
    got = state(resumed)
    assert want.keys() == got.keys()
    for part in want:
        assert want[part].keys() == got[part].keys() and len(want[part]) > 0, part
        assert all(torch.equal(want[part][name], got[part][name]) for name in want[part]), part
    assert continuous.data.state_dict()["sampler"] == resumed.data.state_dict()["sampler"]
