"""``tools/setup_demo.py``: the checks it makes and the command it prints."""

import json

import pytest

from tests.tools.support import load_tool
from worldcast.demo.library import synthetic_library


@pytest.fixture
def tool():
    return load_tool("setup_demo")


def test_a_library_is_summed_up_by_map_with_its_overviews(tool, tmp_path):
    library = synthetic_library()
    library.root = tmp_path
    path = library.save()
    assert tool.check_library(path) == (
        f"{len(library.round_starts)} round starts (arena {len(library.round_starts)});"
        " no overview images (plain panels)"
    )
    arena = {"origin": [-1000, 1000], "scale": 2.0, "size": [1000, 1000], "image": "o/arena.png"}
    (tmp_path / "overviews.json").write_text(json.dumps({"overviews": {"arena": arena}}))
    assert tool.check_library(path).endswith("overviews for arena")


def test_an_overview_the_coordinator_would_refuse_ends_the_tool_with_a_message(
    tool, tmp_path, capsys
):
    library = synthetic_library()
    library.root = tmp_path
    path = library.save()
    (tmp_path / "overviews.json").write_text('{"overviews": {"arena": {"scale": 2.0}}}')
    with pytest.raises(SystemExit):
        tool.main(["--library", str(path), "--check-only"])
    assert "does not load" in capsys.readouterr().err


def test_a_library_that_does_not_load_ends_the_tool_with_a_message(tool, tmp_path, capsys):
    (tmp_path / "library.json").write_text('{"round_starts": []}')
    with pytest.raises(SystemExit):
        tool.main(["--library", str(tmp_path / "library.json"), "--check-only"])
    assert "does not load" in capsys.readouterr().err


def test_the_command_names_the_cards_and_the_library(tool):
    command = tool.serve_command("lib/library.json", "0,1")
    assert "--role all --gpus 0,1 --config configs/demo/worldcast.yaml" in command
    assert "--set library=lib/library.json" in command
