"""tools/prepare_data.py: every step is a subcommand with its own options."""

import re

import pytest

from tests.tools.support import load_tool

STEPS = list(load_tool("prepare_data").STEPS)


@pytest.mark.parametrize("step", STEPS)
def test_every_step_has_help(step, capsys):
    with pytest.raises(SystemExit) as stop:
        load_tool("prepare_data").main([step, "--help"])
    assert stop.value.code == 0
    assert step in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, said",
    [
        ("media-index --dataset-root {tmp}/a --out {tmp}/x.jsonl", "OpenCS2 index does not exist"),
        ("manifest --dataset-root {tmp}/a --media-index {tmp}/a --out {tmp}/x.jsonl", "/a"),
        ("observer-labels --out-dir {tmp}/x", "not set: data.media_index"),
        ("latent-cache", "not set: model.wan22_root"),
        ("buckets weights --bucket-dir {tmp}/a", "bucket file does not exist"),
        ("visibility --media-index {tmp}/a --dataset-root {tmp}/a --out-dir {tmp}/x", "/a"),
        ("meshes --glb {tmp}/a.glb --map de_dust2 --out-dir {tmp}/x", "a.glb: no such file"),
    ],
)
def test_a_step_that_cannot_run_ends_with_its_usage_error(argv, said, tmp_path, capsys):
    """Every step ends alike: the usage of the step (of the subcommand, for ``buckets``),
    ``prepare_data.py <step>: error: <message>`` and exit status 2, and it writes nothing."""
    argv = argv.format(tmp=tmp_path).split()
    with pytest.raises(SystemExit) as stop:
        load_tool("prepare_data").main(argv)
    assert stop.value.code == 2
    step = " ".join(argv[:2] if argv[0] == "buckets" else argv[:1])
    usage, *_, error = capsys.readouterr().err.splitlines()
    assert re.match(rf"usage: \S+ {step} \[-h\]", usage), usage
    assert re.match(rf"\S+ {step}: error: .*{re.escape(said)}", error), error
    assert list(tmp_path.iterdir()) == []
