"""``tools/train.py`` on the CPU: the resolved config it prints, and what it asks for before it
joins a process group."""

from pathlib import Path

import pytest
import yaml

from tests.tools.support import load_tool

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "train"
STAGE4 = str(CONFIGS / "stage4.yaml")
PATHS = str(REPO / "examples" / "train_paths.yaml")


def test_the_resolved_config_is_printed(capsys):
    command = ["--config", STAGE4, "--config", PATHS, "--set", "run.output_dir=runs/stage4"]
    assert load_tool("train").main([*command, "--print-config"]) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert (printed["run"]["stage"], printed["run"]["max_steps"]) == ("4", 600)
    assert printed["run"]["output_dir"] == "runs/stage4"
    assert printed["data"]["bucket_dir"] == "data/buckets"  # of the path template


def test_an_ablation_and_its_overrides_are_resolved(capsys):
    command = ["--config", str(CONFIGS / "ablations" / "late_injection.yaml")]
    command += ["--set", "optim.lr=1e-5", "--set", "run.output_dir=runs/x", "--print-config"]
    assert load_tool("train").main(command) == 0
    cfg = yaml.safe_load(capsys.readouterr().out)
    values = (cfg["run"]["stage"], cfg["model"]["state_injector_block"], cfg["optim"]["lr"])
    assert values == ("2", 22, 1e-5) and cfg["run"]["output_dir"] == "runs/x"


def test_a_run_needs_its_stage_its_output_directory_and_a_complete_checkpoint(tmp_path, capsys):
    tool = load_tool("train")
    out = f"run.output_dir={tmp_path}"
    for arguments, message in (
        ([], "run.stage"),  # no stage file
        (["--config", STAGE4, "--set", "optim.rate=1"], "unknown config key"),
        (["--config", STAGE4], "set run.output_dir"),
        (
            ["--config", STAGE4, "--set", out, "--resume", str(tmp_path)],
            "not a complete checkpoint",
        ),
    ):
        with pytest.raises(SystemExit) as refused:
            tool.main(arguments)
        assert refused.value.code == 2 and message in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []  # nothing started
