"""The step loop every stage shares, and the checkpoints and resume of the flow-matching trainer
built from its config (CPU, FSDP on a gloo group).

* A run assembled by ``build_trainer_parts`` (base seed -> generator from the previous stage's
  checkpoint, fresh modules drawn -> FSDP -> optimizer -> prompt file -> resumable data stream ->
  per-rank seed) trains two steps;
* the same run interrupted after one step, saved, rebuilt from the checkpoint and resumed must reach
  the same weights, AdamW state, EMA, data position and RNG state bit for bit (exact resume, same
  topology);
* the saved ``model_ema.pt`` / ``model.pt`` load into the inference client's ``load_generator``
  (release key layout).
* a stage without ``checkpoint.init`` and without a resume refuses to start (no random 5B
  generator);
* ``checkpoint.init`` reads a safetensors file of the generator's keys and the trainer's torch
  checkpoints alike, and refuses other key names;
* each stage builds only the modules it trains, and ``metrics.jsonl`` records every step;
* a recipe written from :class:`Trainer`'s docstring trains, and one without a hook is refused.
"""

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.engine.training.support import (
    RUN_DIMS,
    FixedBatches,
    ItemDataset,
    write_init_checkpoint,
    write_prompt_embedding,
)
from worldcast import distributed as D
from worldcast.config.training import STAGES, paper_config
from worldcast.engine.checkpoint.training import (
    find_latest_checkpoint,
    prune_checkpoints,
    prune_shards,
)
from worldcast.engine.optim import GROUPS
from worldcast.engine.stage import generator_config
from worldcast.engine.training.build import (
    build_trainer_parts,
    build_training_generator,
    initial_generator_state,
)
from worldcast.engine.training.recipes.bidirectional import BidirectionalTrainer
from worldcast.engine.training.trainer import STEP_TIMES, Trainer
from worldcast.modeling.build import load_generator, read_generator
from worldcast.modeling.wan22.attention import sdpa_attention
from worldcast.modeling.wan22.model import WorldCastGenerator


def _cfg(tmp_path, output: str):
    init = tmp_path / "stage2_ema.pt"
    prompt = tmp_path / "prompt.safetensors"
    base = paper_config("2s", {"model.dims": RUN_DIMS})
    if not init.exists():
        write_init_checkpoint(
            init, base, drop=("ray_embedding.", "visibility_probe.", "observer_signals.")
        )
        write_prompt_embedding(prompt)
    return paper_config(
        "2s",
        {
            "model.dims": RUN_DIMS,
            "model.attention": "sdpa",
            "run.output_dir": str(tmp_path / output),
            "run.max_steps": 2,
            "checkpoint.init": str(init),
            "checkpoint.interval": 1,
            "data.prompt_embedding": str(prompt),
            "data.num_workers": 0,
            "ema.start_step": 0,
            "optim.grad_accum_steps": 2,
        },
    )


def _trainer(cfg, resume=None):
    info = D.DistInfo()
    parts = build_trainer_parts(cfg, info, resume=resume, dataset=ItemDataset(6, target_start=37))
    trainer = BidirectionalTrainer(cfg, parts)
    if resume is not None:
        trainer.resume(resume)
    return trainer


def _snapshot(trainer):
    live = D.live_module(trainer.generator)
    params = {n: p.detach().clone() for n, p in live.named_parameters()}
    state = {
        n: {k: v.clone() for k, v in trainer.optimizer.state[p].items()}
        for n, p in live.named_parameters()
        if p in trainer.optimizer.state
    }
    ema = {k: v.clone() for k, v in trainer.ema.shadow.items()}
    return params, state, ema, torch.get_rng_state(), trainer.data.state_dict()["sampler"]


def test_exact_resume_and_inference_load(tmp_path, gloo):
    cfg = _cfg(tmp_path, "continuous")
    continuous = _trainer(cfg)
    m1 = continuous.train_step()
    m2 = continuous.train_step()
    reference = _snapshot(continuous)
    assert continuous.step == 2 and continuous.ema is not None

    cfg_b = _cfg(tmp_path, "interrupted")
    first = _trainer(cfg_b)
    assert first.train_step()["loss"] == m1["loss"]
    directory = first.save()
    assert (directory / "checkpoint.ready.json").is_file()
    ready = json.loads((directory / "checkpoint.ready.json").read_text())
    assert ready["step"] == 1 and ready["world_size"] == 1 and ready["stage"] == "2s"
    assert find_latest_checkpoint(cfg_b.run.output_dir) == directory
    del first

    resumed = _trainer(cfg_b, resume=directory)
    assert resumed.step == 1
    assert resumed.train_step()["loss"] == m2["loss"]
    params, state, ema, rng, data = _snapshot(resumed)
    ref_params, ref_state, ref_ema, ref_rng, ref_data = reference
    assert params.keys() == ref_params.keys()
    assert all(torch.equal(params[k], ref_params[k]) for k in params)
    assert state.keys() == ref_state.keys()
    assert all(torch.equal(state[k][s], ref_state[k][s]) for k in state for s in state[k])
    assert all(torch.equal(ema[k], ref_ema[k]) for k in ema)
    assert torch.equal(rng, ref_rng) and data == ref_data

    # the release key layout loads into the inference client's loader (the visibility probe is
    # dropped there)
    inference_cfg = dataclasses.replace(generator_config(cfg), visibility_probe=None)
    for name in ("model_ema.pt", "model.pt"):
        payload = torch.load(str(directory / name), map_location="cpu", weights_only=True)
        if name == "model.pt":  # the online weights go to the same loader under the EMA key
            torch.save({"generator_ema": payload["generator"]}, str(tmp_path / "online.pt"))
            path = tmp_path / "online.pt"
        else:
            path = directory / name
        generator = load_generator(path, inference_cfg, device="cpu", attention=sdpa_attention)
        source = payload["generator_ema" if name == "model_ema.pt" else "generator"]
        state = generator.state_dict()
        assert set(state) == {
            k[len("model.") :] for k in source if not k.startswith("model.visibility_probe.")
        }
        assert all(torch.equal(state[k], source["model." + k].to(torch.bfloat16)) for k in state)


def test_init_new_modules_are_fresh_and_others_loaded(tmp_path, gloo):
    cfg = _cfg(tmp_path, "init")
    trainer = _trainer(cfg)
    init = read_generator(cfg.checkpoint.init)
    generator = D.live_module(trainer.generator).generator
    for name, value in generator.state_dict().items():
        if name.startswith(("ray_embedding.", "visibility_probe.", "observer_signals.")):
            assert name not in init
        else:
            assert torch.equal(value, init[name]), name
    # the zero-initialised outputs of the fresh modules
    assert not bool(generator.ray_embedding.mlp[-1].weight.any()) and not bool(
        generator.observer_signals.out.weight.any()
    )


def test_checkpoint_keep_prunes_the_oldest_complete_checkpoints(tmp_path, gloo):
    """``checkpoint.keep``: after each save only the newest ``keep`` complete checkpoints remain (0
    keeps all); an incomplete directory (no ready marker) is never deleted, and resume finds the
    newest one.
    """
    cfg = _cfg(tmp_path, "keep")
    cfg = dataclasses.replace(
        cfg,
        run=dataclasses.replace(cfg.run, max_steps=3),
        checkpoint=dataclasses.replace(cfg.checkpoint, keep=2),
    )
    out = Path(cfg.run.output_dir)
    (out / "checkpoint_model_000000").mkdir(parents=True)  # an interrupted save: left alone
    _trainer(cfg).fit()
    assert sorted(p.name for p in out.glob("checkpoint_model_*")) == [
        "checkpoint_model_000000",
        "checkpoint_model_000002",
        "checkpoint_model_000003",
    ]
    assert find_latest_checkpoint(out) == out / "checkpoint_model_000003"
    assert prune_checkpoints(out, 0) == [] and prune_checkpoints(out, 5) == []
    assert prune_checkpoints(out, 1) == [out / "checkpoint_model_000002"]


def test_checkpoint_keep_shards_prunes_the_resume_files_of_older_checkpoints(tmp_path, gloo):
    """``checkpoint.keep_shards``: every checkpoint keeps its weights, only the newest
    ``keep_shards`` keep their ``rank_*.pt`` files, and resume finds the newest checkpoint that
    still has them."""
    cfg = _cfg(tmp_path, "keep_shards")
    cfg = dataclasses.replace(
        cfg,
        run=dataclasses.replace(cfg.run, max_steps=3),
        checkpoint=dataclasses.replace(cfg.checkpoint, keep_shards=1),
    )
    out = Path(cfg.run.output_dir)
    _trainer(cfg).fit()
    for step in (1, 2, 3):
        directory = out / f"checkpoint_model_{step:06d}"
        assert (directory / "model.pt").is_file() and (directory / "model_ema.pt").is_file()
        assert (directory / "rank_00000.pt").is_file() == (step == 3)
    assert find_latest_checkpoint(out) == out / "checkpoint_model_000003"
    (out / "checkpoint_model_000003" / "rank_00000.pt").unlink()
    assert find_latest_checkpoint(out) is None
    assert prune_shards(out, 0) == [] and prune_shards(out, 1) == []


def test_a_stage_without_checkpoint_init_refuses_to_start(tmp_path, gloo):
    """Stage 2 with ``checkpoint.init`` unset (as the shipped config) and no resume raises before
    any model is
    built: it never trains a randomly initialised generator."""
    cfg = paper_config(
        "2",
        {
            "model.dims": RUN_DIMS,
            "data.prompt_embedding": str(write_prompt_embedding(tmp_path / "prompt.safetensors")),
            "data.num_workers": 0,
        },
    )
    assert cfg.checkpoint.init is None
    with pytest.raises(ValueError, match="set checkpoint.init"):
        build_trainer_parts(cfg, D.DistInfo(), dataset=ItemDataset(2))


def test_checkpoint_init_reads_safetensors_and_training_checkpoints(tmp_path):
    """A safetensors file (the generator's keys) and the trainer's torch checkpoints (the same keys
    under the ``model.`` root) give the same generator state; other key names are refused."""
    from safetensors.torch import save_file

    cfg = paper_config("2s", {"model.dims": RUN_DIMS})
    release = write_init_checkpoint(tmp_path / "release.pt", cfg)
    state = read_generator(release)
    assert any(k.startswith("visibility_probe.") for k in state)  # kept for training
    safetensors = tmp_path / "generator.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(safetensors))
    cfg3 = paper_config("3", {"model.dims": RUN_DIMS})
    for path in (safetensors, release):
        cfg_init = dataclasses.replace(
            cfg3, checkpoint=dataclasses.replace(cfg3.checkpoint, init=str(path))
        )
        got = initial_generator_state(cfg_init, None)
        assert got.keys() == state.keys() and all(torch.equal(got[k], state[k]) for k in got)
    with torch.device("meta"):
        expected = WorldCastGenerator(generator_config(cfg3)).state_dict()
    assert state.keys() == expected.keys()  # stage 3 starts from it with nothing fresh
    renamed = {k.replace("controls.", "inputs."): v for k, v in state.items()}
    with pytest.raises(KeyError, match="does not have"):
        build_training_generator(generator_config(cfg3), renamed)
    with pytest.raises(KeyError, match="outside the 'model.' root"):
        torch.save({"generator_ema": state}, str(tmp_path / "rootless.pt"))
        read_generator(tmp_path / "rootless.pt")


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_each_stage_builds_only_the_modules_it_trains(stage):
    """Stages without scene state build no observer-signal embedding, ray embedding or visibility
    probe; stage 1 has no state injector; the distillation's score models are the stage-2
    model."""
    cfg = paper_config(stage, {"model.dims": RUN_DIMS})
    scene = {"observer_signals", "ray_embedding", "visibility_probe"}
    modules = {"controls", "state_injector"} if cfg.stage.player_state_field else {"controls"}
    if cfg.stage.scene_state:
        modules |= scene
    with torch.device("meta"):
        generator = WorldCastGenerator(generator_config(cfg))
        score_model = WorldCastGenerator(generator_config(cfg, score_model=True))
    built = {name for name, _ in generator.named_children()} & (scene | modules)
    assert built == modules
    assert not {name for name, _ in score_model.named_children()} & scene


def test_metrics_rows_record_every_step(tmp_path, gloo):
    cfg = _cfg(tmp_path, "metrics")
    cfg = dataclasses.replace(cfg, checkpoint=dataclasses.replace(cfg.checkpoint, interval=0))
    _trainer(cfg).fit()
    rows = [json.loads(line) for line in (Path(cfg.run.output_dir) / "metrics.jsonl").open()]
    assert [r["step"] for r in rows] == [1, 2]
    times = {"data_wait_sec", "forward_backward_time_sec", "optimizer_time_sec", "step_time_sec"}
    losses = {"loss", "flow_loss", "visibility_loss", "grad_norm", "memory_window"}
    lrs = {f"lr_{group}" for group in GROUPS}
    for row in rows:
        assert {"timestamp", *times, *losses, *lrs} <= set(row)
        assert {f"grad_norm_{group}" for group in GROUPS} <= set(row)
        assert row["lr_backbone"] == cfg.optim.lr and row["lr_state_injector"] == 1.4e-3
        assert all(row[k] >= 0.0 for k in times)


# ------------------------------------------------------------------------------- the step loop
class LinearTrainer(Trainer):
    """A recipe written from :class:`Trainer`'s docstring: a linear layer regressed onto zero."""

    def __init__(self, cfg) -> None:
        batches = [{"x": torch.full((1, 2), float(i))} for i in range(8)]
        super().__init__(
            cfg, D.DistInfo(), data=FixedBatches(batches), prompt_embeds=torch.zeros(1, 1, 4)
        )
        torch.manual_seed(0)
        self.model = nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train_step(self) -> dict:
        times = dict.fromkeys(STEP_TIMES, 0.0)
        self.optimizer.zero_grad()

        def micro_batch(batch: dict, index: int) -> float:
            self.model(batch["x"]).square().mean().backward()
            return float(batch["x"][0, 0])

        seen = self.accumulate(micro_batch, times)
        self.optimizer.step()
        self.step += 1
        loss = float(self.model.weight.detach().abs().sum())
        return {"loss": loss, "seen": seen, "lr_model": 0.01, **times}

    def checkpoint_models(self) -> dict:
        return {"model.pt": {"generator": dict(self.model.state_dict())}}

    def optimizers(self) -> dict:
        return {"model": self.optimizer}

    @property
    def ema_model(self) -> nn.Module:
        return self.model


def _loop_config(tmp_path, **overrides):
    settings = {
        "run.output_dir": str(tmp_path / "run"),
        "run.max_steps": 3,
        "run.log_interval": 2,
        "checkpoint.interval": 2,
        "optim.grad_accum_steps": 2,
        **overrides,
    }
    return paper_config("2", settings)


def test_a_recipe_written_from_the_trainers_docstring_trains(tmp_path, caplog):
    trainer = LinearTrainer(_loop_config(tmp_path))
    with caplog.at_level("INFO", logger="worldcast.engine.training.trainer"):
        trainer.fit()
    assert trainer.step == 3
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").open()]
    assert [list(row)[:2] for row in rows] == [["step", "timestamp"]] * 3
    # two micro-batches a step, in the stream's order; a list and an lr are rank 0's own values
    assert [row["seen"] for row in rows] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert all(row["lr_model"] == 0.01 and row["data_wait_sec"] >= 0.0 for row in rows)
    # a checkpoint every ``checkpoint.interval`` steps and at the end; a line every log interval
    saved = sorted(path.name for path in (tmp_path / "run").glob("checkpoint_model_*"))
    assert saved == ["checkpoint_model_000002", "checkpoint_model_000003"]
    assert [record.getMessage().split()[0] for record in caplog.records] == ["step=2"]

    resumed = LinearTrainer(_loop_config(tmp_path))
    resumed.resume(tmp_path / "run" / "checkpoint_model_000002")
    assert resumed.step == 2 and resumed.data.index == 4
    other_seed = LinearTrainer(_loop_config(tmp_path, **{"run.seed": 7}))
    with pytest.raises(RuntimeError, match="run.seed differs"):
        other_seed.resume(tmp_path / "run" / "checkpoint_model_000002")


def test_without_an_output_directory_nothing_is_written(tmp_path):
    trainer = LinearTrainer(_loop_config(tmp_path, **{"run.output_dir": None}))
    trainer.fit()
    assert trainer.step == 3 and list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="run.output_dir is not set"):
        trainer.save()


def test_a_recipe_without_a_hook_is_refused_at_construction(tmp_path):
    class Incomplete(LinearTrainer):
        train_step = Trainer.train_step

    with pytest.raises(TypeError, match="abstract method"):
        Incomplete(_loop_config(tmp_path))
