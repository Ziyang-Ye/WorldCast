"""``tools/download_weights.py`` with the Hub stubbed out: a file the repository does not hold ends
the tool with its name, and a Hub that cannot be reached with a usage message."""

from pathlib import Path

import pytest
import yaml

from tests.tools.support import load_tool
from worldcast import hub


def stub_hub(monkeypatch, missing: str = ""):
    """The Hub reached, holding every release file except ``missing``."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import EntryNotFoundError

    def hf_hub_download(repo, name, local_dir=None, local_files_only=False):
        if name == missing:
            raise EntryNotFoundError(f"404: {name}")
        return str(Path(local_dir) / name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(hub, "_hub_reachable", lambda: True)


def test_the_paths_file_is_written_and_printed(tmp_path, monkeypatch, capsys):
    stub_hub(monkeypatch)
    out = tmp_path / "weights"
    assert load_tool("download_weights").main(["--out-dir", str(out), "--skip-wan22"]) == 0
    paths = yaml.safe_load((out / "paths.yaml").read_text())["paths"]
    assert paths["checkpoint"] == str(out.resolve() / "worldcast/worldcast_4step_bf16.safetensors")
    assert f"{out.resolve() / 'paths.yaml'}:" in capsys.readouterr().out


def test_a_missing_file_is_named(tmp_path, monkeypatch, capsys):
    stub_hub(monkeypatch, missing=hub.WEIGHTS["depth_head"])
    with pytest.raises(SystemExit) as stopped:
        load_tool("download_weights").main(["--out-dir", str(tmp_path / "weights"), "--skip-wan22"])
    message = "depth_head.safetensors is not in ZiyangYe/WorldCast on the Hugging Face Hub"
    assert stopped.value.code == 2 and message in capsys.readouterr().err


def test_a_hub_that_cannot_be_reached_is_a_usage_error(tmp_path, monkeypatch, capsys):
    """Without the Hub: the files not in --out-dir are named."""
    monkeypatch.setattr(hub, "_hub_reachable", lambda: False)
    tool = load_tool("download_weights")
    out = tmp_path / "weights"
    release = out.resolve() / "worldcast"
    with pytest.raises(SystemExit) as stopped:
        tool.main(["--out-dir", str(out), "--skip-wan22"])
    error = capsys.readouterr().err
    assert stopped.value.code == 2 and error.startswith("usage:")
    missing = ", ".join(str(release / name) for name in hub.WEIGHTS.values())
    assert error.endswith(
        f"error: the Hugging Face Hub cannot be reached (offline?): no copy of {missing}\n"
    )
