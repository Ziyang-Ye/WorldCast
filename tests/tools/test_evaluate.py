"""``tools/evaluate.py``: the sampler a stage is scored with, and what it asks of the config."""

from pathlib import Path

import pytest

from tests.tools.support import load_tool

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "train"


class _Chosen(Exception):
    """Carries the protocol the tool chose out of its ``main``."""


@pytest.mark.parametrize(
    "stage, arguments, sampler, steps",
    [
        ("stage2", [], "unipc", 20),
        ("stage3", [], "unipc", 20),
        ("stage3", ["--sampler", "four_step"], "four_step", 4),
        ("stage4", [], "four_step", 4),
    ],
)
def test_the_stage_gives_the_sampler(stage, arguments, sampler, steps, monkeypatch, tmp_path):
    tool = load_tool("evaluate")

    def from_config(cfg, protocol, *, index, device, count):
        raise _Chosen(protocol, index, count)

    monkeypatch.setattr(tool.Evaluator, "from_config", from_config)
    command = ["--config", str(CONFIGS / f"{stage}.yaml"), "--set", "validation.index=index.jsonl"]
    command += ["--checkpoint", "model.safetensors", "--out", str(tmp_path / "scores.json")]
    with pytest.raises(_Chosen) as chosen:
        tool.main([*command, "--windows", "8", *arguments])
    protocol, index, count = chosen.value.args
    assert (protocol.sampler, protocol.denoising_steps) == (sampler, steps)
    assert (index, count) == ("index.jsonl", 8)


def test_the_samplers_and_the_validation_index(capsys):
    tool = load_tool("evaluate")
    assert sorted(tool.PROTOCOLS) == ["four_step", "unipc"]
    command = ["--config", str(CONFIGS / "stage2.yaml"), "--checkpoint", "m", "--out", "o"]
    for arguments, message in (
        (command, "set validation.index"),
        ([*command, "--set", "optim.lr=fast"], "optim.lr must be a number"),
    ):
        with pytest.raises(SystemExit) as stopped:
            tool.main(arguments)
        assert stopped.value.code == 2 and message in capsys.readouterr().err
