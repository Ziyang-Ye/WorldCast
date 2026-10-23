"""``tools/replay_live.py`` on the CPU: a recorded client of the synthetic round played through the
live path, in evaluation form and with the settings of the demo's worker, and what the tool asks
of its config."""

import json

import numpy as np
import pytest
import yaml

import tests.engine.inference.support as sw
from tests.tools.support import load_tool, synthetic_round, write_config


@pytest.fixture
def live_round(tmp_path, monkeypatch):
    """The synthetic round (a session of 29 latent frames) with a tiny state model, on open
    ground, and its config as a YAML file."""
    world, cfg = synthetic_round(tmp_path, monkeypatch, max_blocks=1)
    sw.patch_extent(monkeypatch)
    cfg = cfg.with_overrides(sw.write_state_model(tmp_path / "state_model.pt"))
    return write_config(cfg, tmp_path / "config.yaml"), tmp_path


def test_a_client_that_stands_and_turns_is_replayed_as_the_evaluation_runs_it(
    tmp_path, monkeypatch, capsys
):
    """The documented pair of commands on a recording a live client can play to the bit
    (``solo_world``: it stands and turns, nobody is visible, nothing flashes), 37 latent frames
    on a live round of 29, which slides: ``tools/run_client.py`` and ``tools/replay_live.py``
    write the same ``latents.npy``, the live client reading its own position with the state
    model."""
    world = sw.solo_world(tmp_path / "world", steady=True)
    sw.patch_ticks(monkeypatch, world["tables"])
    sw.patch_extent(monkeypatch)
    weights = sw.make_weights(tmp_path / "weights")
    cfg = sw.config(world, weights, tmp_path / "run", max_blocks=3)
    config = write_config(
        cfg.with_overrides(sw.write_state_model(tmp_path / "state_model.pt")), tmp_path / "c.yaml"
    )
    evaluation, replay = tmp_path / "evaluation", tmp_path / "replay"
    arguments = ["--config", config, "--index-row", "0"]
    load_tool("run_client").main(
        [
            *arguments,
            "--world-state-dir",
            str(tmp_path / "world_state"),
            "--out-dir",
            str(evaluation),
        ]
    )
    assert load_tool("replay_live").main([*arguments, "--out-dir", str(replay)]) == 0
    latents = np.load(replay / "latents.npy")
    assert latents.shape == (37, 48, 24, 42) and latents.dtype == np.float32
    assert np.array_equal(latents, np.load(evaluation / "latents.npy"))
    report = json.loads((replay / "replay.json").read_text())
    assert report["media_id"] == sw.media_id(0) and report["demo"] is None
    assert report["config"]["player_state"]["source"] == "live"
    assert report["serving"]["decoder"] == "none" and not report["serving"]["lockstep"]
    # the evaluation's settings, live: the config as given but for the source and the kept round
    assert set(report["differs_from_config"]) == {"player_state.source", "run.latent_frames"}
    assert "settings that differ from the config: player_state.source" in capsys.readouterr().out


def test_the_demos_settings_are_the_workers(live_round):
    config, tmp = live_round
    demo = tmp / "demo.yaml"
    demo.write_text(
        yaml.safe_dump(
            dict(
                scene_state=dict(bound=32),
                client=dict(
                    kind="worldcast",
                    worldcast=dict(
                        overrides={
                            "player_state.source": "live",
                            "live.idle_context": "freeze",
                            "live.idle_with_peers": True,
                        },
                        taehv_path="taew2_2.pth",
                        serving=dict(compile=False),
                    ),
                ),
            )
        )
    )
    out = tmp / "demo"
    arguments = ["--config", config, "--index-row", "0", "--out-dir", str(out), "--demo", str(demo)]
    assert load_tool("replay_live").main(arguments) == 0
    report = json.loads((out / "replay.json").read_text())
    assert report["demo"] == str(demo) and report["serving"]["decoder"] == "none"
    assert {key: pair[1] for key, pair in report["differs_from_config"].items()} == {
        "live.idle_context": "freeze",
        "live.idle_with_peers": True,
        "model.attention": "fa3",
        "player_state.source": "live",
        "run.latent_frames": 29,
        "scene_state.bound": 32,
    }


def test_a_replay_names_what_its_config_lacks(live_round, capsys):
    config, tmp = live_round
    tool = load_tool("replay_live")
    out = ["--out-dir", str(tmp / "out")]
    for arguments, message in (
        (["--index-row", "0"], "not set: paths.out_dir"),
        (out, "run.index_row is not set"),
        ([*out, "--index-row", "0", "--set", "player_state.source=gt"], "plays the live path"),
        ([*out, "--index-row", "0", "--set", "paths.state_model=null"], "paths.state_model"),
        ([*out, "--index-row", "0", "--demo", "nowhere.yaml"], "nowhere.yaml"),
    ):
        with pytest.raises(SystemExit) as refused:
            tool.main(["--config", config, *arguments])
        assert refused.value.code == 2 and message in capsys.readouterr().err
