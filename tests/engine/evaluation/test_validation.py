"""The in-training validation on a tiny stage-2 run under FSDP (CPU, a gloo group): the EMA weights
are scored, the trainer's own weights restored, one ``event: validation`` row written; a run that
validates trains as one that does not."""

import json
from pathlib import Path

import pytest
import torch

from tests.engine.evaluation.support import stage_config, tiny_evaluator
from tests.engine.training.support import ItemDataset, write_init_checkpoint
from worldcast import distributed as D
from worldcast.engine.evaluation.protocols import UNIPC
from worldcast.engine.evaluation.validation import Validation, gather_generator
from worldcast.engine.training.build import build_trainer_parts
from worldcast.engine.training.recipes.bidirectional import BidirectionalTrainer


def _trainer(tmp_path, output: str, **overrides) -> BidirectionalTrainer:
    init = tmp_path / "stage1.pt"
    if not init.exists():
        write_init_checkpoint(init, stage_config(tmp_path, "2"), drop=("state_injector.",))
    cfg = stage_config(
        tmp_path,
        "2",
        **{
            "checkpoint.init": str(init),
            "data.num_workers": 0,
            "ema.start_step": 0,
            "run.output_dir": str(tmp_path / output),
            **overrides,
        },
    )
    parts = build_trainer_parts(cfg, D.DistInfo(), dataset=ItemDataset(2))
    return BidirectionalTrainer(cfg, parts)


def test_validation_scores_the_ema_weights(tmp_path, index_rows, gloo):
    trainer = _trainer(tmp_path, "run")
    trainer.train_step()
    trainer.train_step()
    params = dict(D.live_module(trainer.generator).named_parameters())
    live = {name: p.detach().clone() for name, p in params.items()}
    assert any(not torch.equal(live[n], trainer.ema.shadow[n]) for n in live)

    copy = gather_generator(
        trainer.generator, trainer.ema, device=torch.device("cpu"), dtype=torch.float32
    )
    shadow = {n.replace("_fsdp_wrapped_module.", ""): v for n, v in trainer.ema.shadow.items()}
    for name, value in copy.state_dict().items():
        assert torch.equal(value, shadow["generator." + name].reshape(value.shape))
    assert all(torch.equal(params[n], live[n]) for n in live)

    validation = Validation(trainer.cfg, trainer.info)
    validation.evaluator = tiny_evaluator(trainer.cfg, UNIPC, index_rows, count=1)
    row = validation(trainer.generator, step=trainer.step, ema=trainer.ema)
    assert all(torch.equal(params[n], live[n]) for n in live)  # the live weights are restored
    assert (row["event"], row["step"], row["weight_source"]) == ("validation", 2, "ema")
    assert (row["sampler"], row["sample_count"]) == ("unipc", 1)
    assert validation.evaluator.run(copy)["psnr"] == row["psnr"]
    online = validation(trainer.generator, step=trainer.step)
    assert online["weight_source"] == "live" and online["psnr"] != row["psnr"]


def test_a_run_that_validates_trains_as_one_that_does_not(tmp_path, index_rows, gloo):
    plain = _trainer(tmp_path, "plain", **{"run.max_steps": 2, "checkpoint.interval": 0})
    assert plain.validation is None
    plain.fit()
    rng = torch.get_rng_state()
    validating = _trainer(
        tmp_path,
        "validating",
        **{
            "run.max_steps": 2,
            "checkpoint.interval": 0,
            "validation.index": "index.jsonl",
            "validation.interval": 1,
        },
    )
    validating.validation.evaluator = tiny_evaluator(validating.cfg, UNIPC, index_rows, count=1)
    validating.fit()

    def rows(trainer):
        path = Path(trainer.cfg.run.output_dir) / "metrics.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    train = [r for r in rows(validating) if "event" not in r]
    scored = [r for r in rows(validating) if r.get("event") == "validation"]
    assert [r["step"] for r in scored] == [1, 2] and all("timestamp" in r for r in scored)
    assert [r["loss"] for r in train] == [r["loss"] for r in rows(plain)]
    a = dict(D.live_module(plain.generator).named_parameters())
    b = dict(D.live_module(validating.generator).named_parameters())
    assert all(torch.equal(a[n], b[n]) for n in a)
    assert torch.equal(torch.get_rng_state(), rng)


def test_validation_settings_are_checked(tmp_path):
    with pytest.raises(ValueError, match="needs validation.index"):
        stage_config(tmp_path, "2", **{"validation.interval": 1000})
    with pytest.raises(ValueError, match="stages 1_long to 3"):
        stage_config(tmp_path, "4", **{"validation.interval": 1000, "validation.index": "x"})
