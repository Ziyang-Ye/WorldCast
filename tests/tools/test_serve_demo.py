"""``tools/serve_demo.py --print-config``: the config a role runs with, as a file that loads
back."""

from pathlib import Path

import pytest
import yaml

from tests.tools.support import load_tool
from worldcast.config.loader import config_to_dict
from worldcast.demo.config import load_config

WORLDCAST = Path(__file__).resolve().parents[2] / "configs" / "demo" / "worldcast.yaml"


def test_print_config_prints_the_resolved_settings(tmp_path, capsys):
    tool = load_tool("serve_demo")
    arguments = ["--role", "worker", "--config", str(WORLDCAST), "--set", "play.fps=16"]
    assert tool.main([*arguments, "--print-config"]) == 0
    printed = capsys.readouterr().out
    resolved = load_config([WORLDCAST], {"play.fps": 16})
    assert yaml.safe_load(printed) == config_to_dict(resolved)
    (tmp_path / "resolved.yaml").write_text(printed)
    assert load_config([tmp_path / "resolved.yaml"]) == resolved


def test_the_local_and_all_roles_print_the_route_they_serve(capsys):
    tool = load_tool("serve_demo")
    assert tool.main(["--role", "coordinator", "--print-config"]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["coordinator"]["frame_route"] == "direct"
    for role in ("local", "all"):
        assert tool.main(["--role", role, "--print-config"]) == 0
        assert yaml.safe_load(capsys.readouterr().out)["coordinator"]["frame_route"] == "proxy"


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--role", "local", "--config", str(WORLDCAST)], "one --role worker per GPU"),
        (["--role", "local", "--set", "run.seed=1"], "unknown config keys in <root>: ['run']"),
        (["--role", "worker", "--set", "client.kind=gpu"], "client.kind must be one of"),
    ],
)
def test_a_refused_config_is_a_usage_error(arguments, message, capsys):
    with pytest.raises(SystemExit) as refused:
        load_tool("serve_demo").main([*arguments, "--print-config"])
    assert refused.value.code == 2 and message in capsys.readouterr().err
