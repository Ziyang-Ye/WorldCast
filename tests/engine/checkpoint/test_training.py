"""Training checkpoints: the directory a run saves, when it is complete, what is pruned and what a
run resumes from."""

import json

import pytest
import torch

from worldcast.distributed.process_group import DistInfo, capture_rng_state
from worldcast.engine.checkpoint import training as C


def _save(output_dir, step: int, **keep):
    return C.save_checkpoint(
        output_dir,
        step=step,
        info=DistInfo(),
        metadata={"step": step, "stage": "2"},
        models={"model.pt": {"generator": {"model.w": torch.full((2,), float(step))}}},
        rank_state={"optimizer": step},
        **keep,
    )


def test_a_checkpoint_is_complete_once_its_marker_is_written(tmp_path):
    directory = _save(tmp_path, 100)
    assert directory == C.checkpoint_dir(tmp_path, 100) == tmp_path / "checkpoint_model_000100"
    files = ["checkpoint.ready.json", "model.pt", "rank_00000.pt"]
    assert sorted(p.name for p in directory.iterdir()) == files
    ready = json.loads((directory / "checkpoint.ready.json").read_text())
    assert (ready["step"], ready["stage"], ready["world_size"]) == (100, "2", 1)
    assert set(ready["files"]) == {"model.pt", "rank_00000.pt"}
    model = torch.load(directory / "model.pt", weights_only=True)
    assert model["step"] == 100 and model["generator"]["model.w"].tolist() == [100.0, 100.0]
    assert C.is_resumable(directory) and C.find_latest_checkpoint(tmp_path) == directory


def test_a_rank_resumes_from_its_own_file_on_the_same_world_size(tmp_path):
    rng = capture_rng_state(torch.device("cpu"))
    directory = C.save_checkpoint(
        tmp_path,
        step=3,
        info=DistInfo(),
        metadata={"step": 3},
        models={},
        rank_state={"optimizer": 7, "rng": rng},
    )
    state = C.load_rank_state(directory, DistInfo())
    assert (state["rank"], state["step"], state["world_size"], state["optimizer"]) == (0, 3, 1, 7)
    # the python and numpy RNG states come back as they were saved
    assert state["rng"]["python"] == rng["python"] and isinstance(rng["python"], tuple)
    assert state["rng"]["numpy"][0] == "MT19937"
    with pytest.raises(RuntimeError, match="ran on 1 ranks, this run on 2"):
        C.load_rank_state(directory, DistInfo(world_size=2))


def test_an_interrupted_save_is_never_resumed(tmp_path):
    first = _save(tmp_path, 100)
    later = C.checkpoint_dir(tmp_path, 200)
    later.mkdir()
    (later / "rank_00000.pt").write_bytes(b"")  # no marker: the save did not finish
    assert not C.is_resumable(later) and C.find_latest_checkpoint(tmp_path) == first
    assert C.find_latest_checkpoint(tmp_path / "absent") is None


def test_pruning_keeps_the_newest_checkpoints_and_their_resume_files(tmp_path):
    for step in (100, 200, 300):
        _save(tmp_path, step, keep=2, keep_shards=1)
    kept = sorted(p.name for p in tmp_path.iterdir())
    assert kept == ["checkpoint_model_000200", "checkpoint_model_000300"]
    older, newest = (tmp_path / name for name in kept)
    # the older one keeps its weights without its resume files
    assert (older / "model.pt").is_file() and not C.is_resumable(older)
    assert C.find_latest_checkpoint(tmp_path) == newest
    assert C.prune_checkpoints(tmp_path, 0) == [] and C.prune_shards(tmp_path, 0) == []
    assert C.prune_shards(tmp_path, 1) == []  # nothing left to prune
    assert C.prune_checkpoints(tmp_path, 1) == [older]


def test_a_trainers_state_is_saved_under_the_generators_key_names():
    state = {"generator.blocks.0.weight": torch.zeros(1)}
    assert list(C.generator_state(state)) == ["blocks.0.weight"]
    assert list(C.release_state(state)) == ["model.blocks.0.weight"]


def test_the_wan22_backbone_is_read_from_every_shard(tmp_path):
    from safetensors.torch import save_file

    save_file({"blocks.0.weight": torch.ones(1)}, tmp_path / "a.safetensors")
    save_file({"head.weight": torch.zeros(2)}, tmp_path / "b.safetensors")
    index = {"weight_map": {"blocks.0.weight": "a.safetensors", "head.weight": "b.safetensors"}}
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))
    state = C.load_wan22_backbone(tmp_path)
    assert sorted(state) == ["blocks.0.weight", "head.weight"]
    assert state["head.weight"].tolist() == [0.0, 0.0]
