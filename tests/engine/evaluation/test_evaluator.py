"""The evaluator end to end on a tiny generator with synthetic windows (CPU), and the loading of the
generator it scores."""

import hashlib
import json
import math

import pytest
import torch

from tests.engine.evaluation.support import PixelStub, stage_config, tiny_evaluator
from tests.engine.training.support import write_init_checkpoint
from worldcast.distributed.process_group import DistInfo
from worldcast.engine.evaluation import evaluator as evaluator_module
from worldcast.engine.evaluation.evaluator import Evaluator, load_eval_generator
from worldcast.engine.evaluation.protocols import FOUR_STEP, UNIPC, select_windows
from worldcast.modeling.build import ONLINE_KEY


def _generator(tmp_path, cfg):
    init = tmp_path / f"init{cfg.run.stage}.pt"
    write_init_checkpoint(init, cfg)
    return load_eval_generator(cfg, init, device="cpu")


@pytest.mark.parametrize(
    ("stage", "protocol"),
    [
        ("1_long", UNIPC),
        ("2", UNIPC),
        ("2s", UNIPC),
        ("3_noscene", UNIPC),
        ("3", UNIPC),
        ("3", FOUR_STEP),
        ("4_noscene", FOUR_STEP),
        ("4", FOUR_STEP),
    ],
    ids=lambda value: value if isinstance(value, str) else value.sampler,
)
def test_evaluator_end_to_end(tmp_path, index_rows, stage, protocol):
    cfg = stage_config(tmp_path, stage)
    evaluator = tiny_evaluator(cfg, protocol, index_rows)
    generator = _generator(tmp_path, cfg)
    rng = torch.get_rng_state()
    row = evaluator.run(generator)
    assert torch.equal(torch.get_rng_state(), rng)  # a window draws from its own generator only
    assert row["window_set"] == "worldcast-maps4-eval64-v1" and row["sampler"] == protocol.sampler
    assert row["denoising_steps"] == protocol.denoising_steps and row["sample_count"] == 2
    assert [s["index"] for s in row["samples"]] == [0, 1]
    for key in ("psnr", "ssim", "lpips", "pixel_mse", "latent_mse"):
        assert math.isfinite(row[key])
        assert row[key] == pytest.approx(sum(s[key] for s in row["samples"]) / 2)
    assert sum(m["sample_count"] for m in row["by_map"].values()) == 2
    assert evaluator.score(generator, evaluator.windows[1]) == row["samples"][1]


def test_each_rank_scores_its_share_of_the_windows(tmp_path, index_rows):
    cfg = stage_config(tmp_path, "2")
    evaluator = tiny_evaluator(cfg, UNIPC, index_rows, count=3)
    generator = _generator(tmp_path, cfg)
    first = evaluator.run(generator, info=DistInfo(rank=0, world_size=2))
    assert [s["index"] for s in first["samples"]] == [0, 2]
    assert evaluator.run(generator, info=DistInfo(rank=1, world_size=2)) is None


def test_a_stage_is_scored_with_a_sampler_it_can_run(tmp_path, index_rows):
    with pytest.raises(
        ValueError, match="stage 2 is bidirectional: score it with the sampler unipc"
    ):
        tiny_evaluator(stage_config(tmp_path, "2"), FOUR_STEP, index_rows)
    with pytest.raises(ValueError, match="score it with the sampler four_step"):
        tiny_evaluator(stage_config(tmp_path, "4_noscene"), UNIPC, index_rows)
    with pytest.raises(ValueError, match="windows of 41 latent frames; stage 1 trains on 21"):
        tiny_evaluator(stage_config(tmp_path, "1"), UNIPC, index_rows)


def test_the_evaluator_reads_the_index_the_config_names_by_its_digest(
    tmp_path, index_rows, monkeypatch
):
    rows = [dict(row, media_id=f"{row['media_id']}-{k}") for k in range(3) for row in index_rows]
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows[:64]))
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    monkeypatch.setattr(evaluator_module, "_window_dataset", lambda cfg, windows: None)
    monkeypatch.setattr(evaluator_module, "load_vae", lambda root, device: PixelStub())

    with pytest.raises(ValueError, match="is not the index of worldcast-maps4-eval64-v1"):
        Evaluator.from_config(stage_config(tmp_path, "2"), UNIPC, index=index, device="cpu")
    cfg = stage_config(tmp_path, "2", **{"validation.index_sha256": digest})
    evaluator = Evaluator.from_config(cfg, UNIPC, index=index, device="cpu", count=3)
    windows = select_windows(rows[:64], UNIPC)
    assert evaluator.windows == windows[:3]
    assert evaluator.protocol.index_sha256 == digest
    assert evaluator.protocol.selection_sha256 != UNIPC.selection_sha256
    assert UNIPC.index_sha256 != digest  # the protocol of the paper is not touched


def test_the_loader_must_serve_the_window_asked_for(tmp_path, index_rows):
    evaluator = tiny_evaluator(stage_config(tmp_path, "2"), UNIPC, index_rows)
    served = evaluator.load_item
    evaluator.load_item = lambda window: served(evaluator.windows[1])
    with pytest.raises(RuntimeError, match="the loader served"):
        evaluator.inputs(evaluator.windows[0])


def test_load_eval_generator_builds_the_stage_modules_only(tmp_path):
    """A checkpoint with the modules of scene state and the probe loads into a stage-2 generator
    without them; a checkpoint that lacks a module of the stage is refused."""
    cfg = stage_config(tmp_path, "2")
    full = write_init_checkpoint(tmp_path / "full.pt", stage_config(tmp_path, "2s"))
    generator = load_eval_generator(cfg, full, device="cpu")
    assert generator.observer_signals is None and generator.ray_embedding is None
    assert generator.visibility_probe is None and generator.state_injector is not None
    assert not any(p.requires_grad for p in generator.parameters())
    with pytest.raises(KeyError, match="has no 'generator' entry"):
        load_eval_generator(cfg, full, weights=ONLINE_KEY, device="cpu")
    broken = tmp_path / "broken.pt"
    write_init_checkpoint(broken, cfg, drop=("blocks.1.",))
    with pytest.raises(KeyError, match="missing"):
        load_eval_generator(cfg, broken, device="cpu")
