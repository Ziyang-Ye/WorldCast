"""``tools/download_weights.py`` with the Hub stubbed out: a file the repository does not hold ends
the tool with its name, after the paths file is written, and a Hub that cannot be reached with a
usage message; the metrics' weights it is asked for."""

from pathlib import Path

import pytest
import yaml

from tests.tools.support import load_tool
from worldcast import hub


def test_a_missing_training_checkpoint_is_named(tmp_path, monkeypatch, capsys):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import EntryNotFoundError

    def hf_hub_download(repo, name, local_dir=None, local_files_only=False):
        if name in hub.TRAINING_CHECKPOINTS:
            raise EntryNotFoundError(f"404: {name}")
        return str(Path(local_dir) / name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(hub, "_hub_reachable", lambda: True)
    monkeypatch.setattr(hub, "TRAINING_CHECKPOINTS", ("stage.safetensors",))
    tool = load_tool("download_weights")
    out = tmp_path / "weights"
    arguments = ["--out-dir", str(out), "--skip-wan22", "--skip-taehv"]
    assert tool.main(arguments) == 0
    paths = yaml.safe_load((out / "paths.yaml").read_text())["paths"]
    assert paths["checkpoint"] == str(out.resolve() / "worldcast/worldcast_4step_bf16.safetensors")
    assert f"{out.resolve() / 'paths.yaml'}:" in capsys.readouterr().out
    (out / "paths.yaml").unlink()
    with pytest.raises(SystemExit) as stopped:
        tool.main([*arguments, "--with-training-checkpoints"])
    message = "stage.safetensors is not in ZiyangYe/WorldCast on the Hugging Face Hub"
    assert stopped.value.code == 2 and message in capsys.readouterr().err
    assert (out / "paths.yaml").is_file()  # written before the training checkpoints


def test_a_hub_that_cannot_be_reached_is_a_usage_error(tmp_path, monkeypatch, capsys):
    """Without the Hub: the files not in --out-dir are named; with them in place, a tiny decoder
    that cannot be fetched is named after the paths file is written."""
    monkeypatch.setattr(hub, "_hub_reachable", lambda: False)
    tool = load_tool("download_weights")
    out = tmp_path / "weights"
    release = out.resolve() / "worldcast"
    with pytest.raises(SystemExit) as stopped:
        tool.main(["--out-dir", str(out), "--skip-wan22", "--skip-taehv"])
    error = capsys.readouterr().err
    assert stopped.value.code == 2 and error.startswith("usage:")
    missing = ", ".join(str(release / name) for name in hub.WEIGHTS.values())
    assert error.endswith(
        f"error: the Hugging Face Hub cannot be reached (offline?): no copy of {missing}\n"
    )
    release.mkdir(parents=True)
    for name in hub.WEIGHTS.values():
        (release / name).write_bytes(b"in place")

    def refused(part):
        raise ConnectionError(f"no copy of {part.with_name(hub.TAEHV_FILE)}")

    monkeypatch.setattr(hub, "_download_taehv", refused)
    with pytest.raises(SystemExit) as stopped:
        tool.main(["--out-dir", str(out), "--skip-wan22"])
    assert stopped.value.code == 2
    assert capsys.readouterr().err.endswith(f"error: no copy of {out.resolve()}/taew2_2.pth\n")
    assert (out / "paths.yaml").is_file()


def test_the_weights_of_the_metrics_it_is_asked_for(tmp_path, monkeypatch, capsys):
    tool = load_tool("download_weights")
    asked = []

    def download_weights(out_dir, tables_dir, **options):
        asked.append(options["metrics"])
        (Path(out_dir) / "paths.yaml").parent.mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "paths.yaml").write_text("paths: {}\n")
        return Path(out_dir) / "paths.yaml"

    monkeypatch.setattr(tool, "download_weights", download_weights)
    out = str(tmp_path / "weights")
    assert tool.main(["--out-dir", out]) == 0
    assert tool.main(["--out-dir", out, "--with-metrics"]) == 0
    assert tool.main(["--out-dir", out, "--with-metrics", "i3d,loftr"]) == 0
    assert asked == [[], list(hub.METRIC_WEIGHTS), ["i3d", "loftr"]]
    with pytest.raises(SystemExit) as stopped:
        tool.main(["--out-dir", out, "--with-metrics", "i3d,vbench"])
    assert stopped.value.code == 2
    assert "name some of detector, gim_dkm, loftr, i3d" in capsys.readouterr().err
