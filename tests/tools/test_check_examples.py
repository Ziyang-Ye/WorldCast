"""``tools/check_examples.py`` on the CPU: the inputs of a case, read from its config."""

from dataclasses import replace

import pytest

import tests.engine.inference.support as sw
from tests.engine.inference.support import offline_run
from tests.tools.support import load_tool, write_config


def test_a_case_is_checked_from_its_config(tmp_path, monkeypatch, capsys):
    offline = offline_run(tmp_path / "offline")
    sw.patch_ticks(monkeypatch, offline["world"]["tables"])
    tool = load_tool("check_examples")
    monkeypatch.setattr(tool, "EXAMPLES_MANIFEST", offline["table"])  # the run's own fingerprints
    cfg = offline["cfg"]
    longer = write_config(cfg, tmp_path / "longer.yaml")  # asks for 441 latent frames
    case = replace(cfg, run=replace(cfg.run, latent_frames=29))
    config = write_config(case, tmp_path / "config.yaml")
    assert tool.main([config]) == 0
    (line,) = capsys.readouterr().out.splitlines()[1:]  # one line per client
    assert line.startswith(f"  {offline['media_id']}: 29 latents, 10 players recorded")
    assert line.endswith("; the first frame is the reference run's")
    assert tool.main([config, longer]) == 1
    assert "  FAILED: " in capsys.readouterr().out
    with pytest.raises(SystemExit) as refused:
        tool.main([str(tmp_path / "nowhere" / "config.yaml")])
    assert refused.value.code == 2 and "nowhere" in capsys.readouterr().err


def test_a_missing_dependency_is_a_usage_error(tmp_path, monkeypatch, capsys):
    tool = load_tool("check_examples")

    def check_inputs(cfg, runs):
        raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")

    monkeypatch.setattr(tool, "check_inputs", check_inputs)
    config = tmp_path / "config.yaml"
    config.write_text("run: {latent_frames: 29}\n")
    with pytest.raises(SystemExit) as refused:
        tool.main([str(config)])
    error = capsys.readouterr().err
    assert refused.value.code == 2 and "usage:" in error
    assert (
        "No module named 'pyarrow': install the package with its dependencies, pip install -e ."
        in error
    )
