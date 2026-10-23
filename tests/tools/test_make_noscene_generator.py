"""``tools/make_noscene_generator.py``: a checkpoint of the generator trained without scene state
that carries the unused modules of scene state and the visibility probe becomes its client weight
file (bf16, the generator's keys, the metadata of the stage) and a config that points a client at
it, which runs it; a training checkpoint too; a checkpoint with a weight of anything else, or
without one of the generator's, is refused and nothing is written."""

import json

import numpy as np
import pytest
import torch
import yaml

import tests.engine.inference.support as sw
from tests.modeling.support import randomize_
from tests.tools.support import load_tool, write_config
from worldcast.config.inference import load_config
from worldcast.engine.inference.client import run_client
from worldcast.engine.inference.loading import ClientModels
from worldcast.engine.inference.serving import ServingOptions
from worldcast.modeling.build import NOSCENE_FILE
from worldcast.modeling.visibility_probe import VisibilityProbeConfig
from worldcast.modeling.wan22.model import GeneratorConfig, WorldCastGenerator


def checkpoint() -> dict[str, torch.Tensor]:
    """A tiny generator of stage 4_noscene's dimensions as its checkpoint holds it: fp32, with the
    ray and observer-signal embeddings and the visibility probe, which the stage does not run."""
    config = GeneratorConfig(**sw.TINY, visibility_probe=VisibilityProbeConfig(dit_block=2))
    return randomize_(WorldCastGenerator(config), 21).state_dict()


def client_keys() -> set[str]:
    config = GeneratorConfig(**sw.TINY, observer_signals=None, ray_embedding=False)
    with torch.device("meta"):
        return set(WorldCastGenerator(config).state_dict())


@pytest.fixture
def made(tmp_path):
    """The tiny weights beside the generator (its snapshot's config.json, the depth head, the
    prompt) and the checkpoint, as a ``.safetensors`` file with the metadata of a release file."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    weights = sw.make_weights(tmp_path / "weights")
    state = checkpoint()
    source = tmp_path / "noscene_fp32.safetensors"
    save_file(state, str(source), metadata={"format": "pt", "stage": "4_noscene", "step": "600"})
    return dict(weights=weights, state=state, source=source)


def make(made, out_dir, capsys, source=None):
    tool = load_tool("make_noscene_generator")
    arguments = ["--checkpoint", str(source or made["source"]), "--out-dir", str(out_dir)]
    status = tool.main([*arguments, "--wan22-root", made["weights"]["wan22_root"]])
    return status, capsys.readouterr()


def test_the_client_weight_file_holds_the_generators_weights_in_bf16_and_reports_what_it_left_out(
    made, tmp_path, capsys
):
    from safetensors import safe_open

    status, printed = make(made, tmp_path / "out", capsys)
    assert status == 0
    out = tmp_path / "out" / "worldcast" / NOSCENE_FILE
    with safe_open(str(out), framework="pt") as handle:
        assert handle.metadata() == {"format": "pt", "stage": "4_noscene", "step": "600"}
        assert set(handle.keys()) == client_keys()
        for key in handle.keys():
            saved = handle.get_tensor(key)
            assert saved.dtype == torch.bfloat16
            assert torch.equal(saved, made["state"][key].to(torch.bfloat16)), key
    report = json.loads(printed.out)
    assert report["checkpoint_stage"] == "4_noscene" and report["step"] == "600"
    assert report["out"] == str(out) and report["tensors"] == len(client_keys())
    assert report["dtype"] == "torch.bfloat16"
    left_out = report["left_out"]
    assert sorted(left_out) == ["observer_signals", "ray_embedding", "visibility_probe"]
    for module, row in left_out.items():
        keys = [k for k in made["state"] if k.split(".")[0] == module]
        assert row["tensors"] == len(keys)
        assert row["parameters"] == sum(made["state"][k].numel() for k in keys)
        assert row["max_abs"] == max(float(made["state"][k].abs().max()) for k in keys)
    settings = yaml.safe_load((tmp_path / "out" / "noscene.yaml").read_text())
    assert settings == {
        "paths": {"checkpoint": str(out.resolve())},
        "model": {"scene_state": False},
    }


def test_a_client_runs_the_client_weight_file_with_its_config(made, tmp_path, monkeypatch, capsys):
    """The offline client of a synthetic round, ``weights/noscene.yaml`` given after the other
    configs: it loads the generator without scene state and generates the first six blocks and a
    window block."""
    assert make(made, tmp_path / "weights", capsys)[0] == 0
    world = sw.make_world(tmp_path / "world", clients=(0,))
    sw.patch_ticks(monkeypatch, world["tables"])
    base = sw.config(world, made["weights"], tmp_path / "run", max_blocks=1)
    written = write_config(base, tmp_path / "config.yaml")
    overrides = {
        "run.index_row": 0,
        "paths.world_state_dir": str(tmp_path / "world_state"),
        "paths.out_dir": str(tmp_path / "run"),
    }
    cfg = load_config([written, tmp_path / "weights" / "noscene.yaml"], overrides)
    assert not cfg.model.scene_state
    generator = ClientModels.load(cfg, ServingOptions(decoder="none")).generator
    assert generator.ray_embedding is None and generator.observer_signals is None
    summary = run_client(cfg)
    assert [block["window_frames"] for block in summary["blocks"]] == [29]
    latents = np.load(tmp_path / "run" / "latents.npy")
    assert latents.shape == (29, 48, 24, 42) and np.isfinite(latents).all()


def test_a_training_checkpoint_of_the_stage_is_made_the_same(made, tmp_path, capsys):
    from safetensors.torch import load_file

    source = tmp_path / "model_ema.pt"
    entry = {"model." + key: value for key, value in made["state"].items()}
    torch.save({"step": 600, "stage": "4_noscene", "generator_ema": entry}, source)
    assert make(made, tmp_path / "a", capsys)[0] == 0
    status, printed = make(made, tmp_path / "b", capsys, source=source)
    assert status == 0 and json.loads(printed.out)["step"] == "600"
    a = load_file(str(tmp_path / "a" / "worldcast" / NOSCENE_FILE))
    b = load_file(str(tmp_path / "b" / "worldcast" / NOSCENE_FILE))
    assert set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in a)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda state: state.update(extra=torch.zeros(3)), "keys the module does not have"),
        (
            lambda state: state.update({"state_injector.extra": torch.zeros(3)}),
            "keys the module does not have",
        ),
        (lambda state: state.pop("state_injector.proj.weight"), "lack keys of the generator"),
        (
            lambda state: state.update({"head.head.bias": torch.zeros(5)}),
            "shapes differ",
        ),
    ],
    ids=["backbone", "module-it-runs", "missing", "shape"],
)
def test_a_checkpoint_of_another_generator_is_refused(made, tmp_path, capsys, change, message):
    from safetensors.torch import save_file

    state = dict(made["state"])
    change(state)
    source = tmp_path / "other.safetensors"
    save_file(state, str(source))
    status, printed = make(made, tmp_path / "out", capsys, source=source)
    assert status == 1 and message in printed.err and "stage 4_noscene" in printed.err
    assert not (tmp_path / "out").exists()
