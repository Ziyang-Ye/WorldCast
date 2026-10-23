"""Every tool: it has its test module (``test_examples_scripts`` tests the examples' scripts),
``--help`` shows its docstring, and without arguments it asks for them and does nothing."""

from pathlib import Path

import pytest

from tests.tools.support import TOOLS, load_tool

NAMES = sorted(path.stem for path in TOOLS.glob("*.py"))


def test_every_tool_has_its_test_module():
    tested = {path.stem.removeprefix("test_") for path in Path(__file__).parent.glob("test_*.py")}
    assert tested == {*NAMES, "help", "examples_scripts"}
    assert NAMES == [
        "bench_live",
        "bench_realtime",
        "build_demo_library",
        "check_examples",
        "decode",
        "download_eval_data",
        "download_examples",
        "download_weights",
        "evaluate",
        "evaluate_paper",
        "make_noscene_generator",
        "make_overviews",
        "make_prompt_embedding",
        "make_showcase",
        "prepare_data",
        "replay_live",
        "run_client",
        "run_session",
        "serve_demo",
        "setup_demo",
        "train",
        "verify_reference",
    ]


@pytest.mark.parametrize("name", NAMES)
def test_help_shows_the_docstring(name, capsys):
    tool = load_tool(name)
    with pytest.raises(SystemExit) as stopped:
        tool.main(["--help"])
    assert stopped.value.code == 0
    summary = tool.__doc__.strip().splitlines()[0]
    assert summary in capsys.readouterr().out


@pytest.mark.parametrize("name", NAMES)
def test_without_arguments_a_tool_asks_for_them(name, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        load_tool(name).main([])
    assert stopped.value.code == 2 and "usage:" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []  # nothing written
