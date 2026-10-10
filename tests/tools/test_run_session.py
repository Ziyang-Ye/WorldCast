"""``tools/run_session.py`` on the CPU: a session whose client runs in this process, the length
``--round-length`` gives each round, and the GPUs the clients take, checked before one starts."""

import contextlib
import json

import numpy as np
import pytest
import torch
import yaml

import tests.engine.inference.support as sw
from tests.tools.support import load_tool
from worldcast.engine.inference import session


def in_process(monkeypatch) -> list[list[str]]:
    """Run the clients of a session in this process; returns the commands as they start."""
    client, commands = load_tool("run_client"), []

    class InProcess:
        """Stands in for ``subprocess.Popen``: the client's ``main`` in this process."""

        def __init__(self, command, env, stdout, stderr):
            commands.append(command)
            with contextlib.redirect_stdout(stdout):
                self.code = client.main(command[2:])

        def wait(self) -> int:
            return self.code

    monkeypatch.setattr(session.subprocess, "Popen", InProcess)
    return commands


def test_a_session_runs_its_clients(synthetic, tmp_path, monkeypatch, capsys):
    config, _, media_id, latents = synthetic
    commands = in_process(monkeypatch)
    out = tmp_path / "session"
    tool = load_tool("run_session")
    assert tool.main(["--config", config, "--all-rounds", "--out-dir", str(out)]) == 0
    (round_dir,) = out.iterdir()
    assert round_dir.name == f"{sw.MATCH}-{sw.MAP}-r{sw.ROUND:02d}-s000000"
    assert np.array_equal(np.load(round_dir / media_id / "latents.npy"), latents)
    assert "--device" not in commands[0]  # run.device of the config decides
    assert commands[0][commands[0].index("--index-row") + 1] == "0"
    capsys.readouterr()
    for arguments, message in (
        (["--round-of", "5", "--out-dir", str(out)], "no such row in the 1-row index"),
        (["--all-rounds", "--out-dir", str(out)], "world_state is not empty"),  # not fresh
        (["--all-rounds", "--out-dir", str(out), "--set", "run.seed=-1"], "run.seed must be"),
        # no abbreviations: examples/run.sh reads --gpus and --device as spelled
        (["--all-rounds", "--out-dir", str(out), "--gpu", "4,5"], "unrecognized arguments: --gpu"),
    ):
        with pytest.raises(SystemExit) as refused:
            tool.main(["--config", config, *arguments])
        assert refused.value.code == 2 and message in capsys.readouterr().err


def test_round_length_sets_the_latent_frames_of_each_round(synthetic, tmp_path, monkeypatch):
    config, cfg, _, _ = synthetic
    record = json.loads(open(cfg.paths.round_index).read())
    index = tmp_path / "round_index.jsonl"
    index.write_text(json.dumps(dict(record, round_seconds=87)) + "\n")
    monkeypatch.setattr(session.subprocess, "Popen", Recorded)
    Recorded.commands = []
    arguments = ["--config", config, "--set", f"paths.round_index={index}", "--index-row", "0"]
    tool = load_tool("run_session")
    assert tool.main([*arguments, "--round-length", "--out-dir", str(tmp_path / "a")]) == 1
    # 87 s: eight whole ten-second steps, a block per second after the first frame
    assert Recorded.commands[0][-8:-6] == ["--set", "run.latent_frames=321"]
    index.write_text(json.dumps(record) + "\n")  # no round_seconds
    with pytest.raises(SystemExit):
        tool.main([*arguments, "--round-length", "--out-dir", str(tmp_path / "b")])


class Recorded:
    """Stands in for ``subprocess.Popen``: records the command and fails."""

    commands: list[list[str]] = []

    def __init__(self, command, env, stdout, stderr):
        Recorded.commands.append(command)

    def wait(self) -> int:
        return 1


class Placed:
    """Stands in for ``subprocess.Popen``: records the GPU of each client, which succeeds."""

    gpus: list[str] = []

    def __init__(self, command, env, stdout, stderr):
        Placed.gpus.append(env["CUDA_VISIBLE_DEVICES"])

    def wait(self) -> int:
        return 0


def test_the_clients_take_the_visible_gpus_and_no_other(tmp_path, monkeypatch, capsys):
    """Three clients of the config's device, ``cuda``, on a machine of two visible GPUs of 95.6
    GiB (an H20's), then of 24 GiB."""
    world = sw.make_world(tmp_path / "world", clients=(0, 3, 7))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {"round_index": world["round_index"]}}))
    monkeypatch.setattr(session.subprocess, "Popen", Placed)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    Placed.gpus = []
    tool = load_tool("run_session")
    monkeypatch.setattr(tool, "gpu_memory", lambda: {"4": 95.6, "6": 95.6})
    arguments = ["--config", str(config), "--all-rounds"]
    assert tool.main([*arguments, "--out-dir", str(tmp_path / "session")]) == 0
    assert Placed.gpus == ["4", "6", "4"]
    assert capsys.readouterr().out.splitlines()[1] == (
        "  GPUs 4,6,4: 3 clients on 2 visible GPUs in turn, about 15 GiB each (--gpus places them)"
    )
    assert tool.main([*arguments, "--gpus", "4, 6,6", "--out-dir", str(tmp_path / "listed")]) == 0
    assert Placed.gpus[3:] == ["4", "6", "6"] and "GPUs" not in capsys.readouterr().out
    # refused before a client starts
    for gpus, count, memory, message in (
        (["--gpus", "4,6,5"], 2, 95.6, "--gpus: no GPU 5 among the visible GPUs 4,6"),
        (["--gpus", "0,1,2"], 2, 95.6, "--gpus: no GPU 0,1,2 among the visible GPUs 4,6"),
        (
            ["--gpus", "4,6"],
            2,
            95.6,
            "--gpus: 2 GPU ids for the 3 clients of rows [0, 1, 2]: an id per client (an id can"
            " repeat)",
        ),
        ([], 0, 95.6, "no GPU is visible (--device cpu runs the clients on the CPU)"),
        (
            [],
            2,
            24.0,
            "error: GPU 4 has 24 GiB for 2 clients: a client takes about 15 GiB, place them on"
            " more GPUs (--gpus, CUDA_VISIBLE_DEVICES)\n",
        ),
        (["--gpus", "6,6,4"], 2, 24.0, "error: GPU 6 has 24 GiB for 2 clients: a client"),
    ):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
        monkeypatch.setattr(tool, "gpu_memory", lambda: {"4": memory, "6": memory})
        with pytest.raises(SystemExit) as refused:
            tool.main([*arguments, *gpus, "--out-dir", str(tmp_path / "refused")])
        assert refused.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "refused").exists() and len(Placed.gpus) == 6
    # without a visible GPU the CPU's clients take none, and no GPU's memory is asked for
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(tool, "gpu_memory", None)
    assert tool.main([*arguments, "--device", "cpu", "--out-dir", str(tmp_path / "cpu")]) == 0
    assert Placed.gpus[6:] == ["", "", ""] and "GPUs" not in capsys.readouterr().out
