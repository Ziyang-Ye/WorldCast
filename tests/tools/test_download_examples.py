"""``tools/download_examples.py`` on the CPU: a case copied from a local source into ``--out-dir``,
with its config; a Hub that cannot be reached, and a file that is not there, end the tool with a
usage message."""

import hashlib
import json

import pytest
import yaml

from tests.tools.support import load_tool
from worldcast import hub

#: The one file of the case ``a``, as the repository names it.
NAME = "examples/data/a/first_latents/p00.npz"


@pytest.fixture
def tool(tmp_path, monkeypatch):
    """The tool, in a repository of one case ``a`` (its manifest names one file)."""
    case = dict(map="de_nuke", match_id=7, round=3, clients=["p00"], latent_frames=81)
    case["files"] = {NAME: {"bytes": 6, "sha256": hashlib.sha256(b"latent").hexdigest()}}
    manifest = tmp_path / "repo" / "examples" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"cases": {"a": case}}))
    tool = load_tool("download_examples")
    monkeypatch.setattr(tool, "ROOT", tmp_path / "repo")
    return tool


def test_a_case_is_downloaded_into_out_dir_with_its_config(tool, tmp_path, capsys):
    """From a local copy of the Hub's repository."""
    (tmp_path / "hub" / NAME).parent.mkdir(parents=True)
    (tmp_path / "hub" / NAME).write_bytes(b"latent")
    out = tmp_path / "data disk"
    assert tool.main(["--out-dir", str(out), "--source", str(tmp_path / "hub")]) == 0
    assert (out / "data/a/first_latents/p00.npz").read_bytes() == b"latent"
    config = yaml.safe_load((out / "data/a/config.yaml").read_text())
    assert config["run"] == {"latent_frames": 81}
    assert config["paths"]["latent_cache_root"] == str(out.resolve() / "data/a/first_latents")
    index = tmp_path.resolve() / "repo/examples/data/a/round_index.jsonl"
    assert config["paths"]["round_index"] == str(index)
    assert "a: 1 files, 0.0 MB (1 fetched)" in capsys.readouterr().out
    with pytest.raises(SystemExit) as refused:
        tool.main(["--out-dir", str(out), "b"])
    assert refused.value.code == 2 and "unknown case(s) ['b']; known: a" in capsys.readouterr().err


def test_what_cannot_be_fetched_is_a_usage_error(tool, tmp_path, monkeypatch, capsys):
    """The Hub (stubbed out) holds no file; without a connection the library finds no copy."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    def hf_hub_download(repo, name, local_dir=None, local_files_only=False):
        if local_files_only:
            raise LocalEntryNotFoundError(f"no local copy of {name}")
        raise EntryNotFoundError(f"404: {name}")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    out = str(tmp_path / "examples")
    for arguments, reachable, message in (
        (
            [],
            False,
            "error: the Hugging Face Hub cannot be reached (offline?): no copy of"
            f" {tmp_path / 'examples/data/a/first_latents/p00.npz'}; pass --source <a local copy"
            " of the repository>\n",
        ),
        ([], True, f"error: {NAME} is not in ZiyangYe/WorldCast on the Hugging Face Hub\n"),
        (
            ["--source", str(tmp_path / "copy")],
            True,
            f"error: [Errno 2] No such file or directory: '{tmp_path / 'copy' / NAME}'\n",
        ),
    ):
        monkeypatch.setattr(hub, "_hub_reachable", lambda: reachable)
        with pytest.raises(SystemExit) as refused:
            tool.main(["--out-dir", out, *arguments])
        error = capsys.readouterr().err
        assert refused.value.code == 2 and error.startswith("usage:") and error.endswith(message)
    assert not (tmp_path / "examples" / "data" / "a" / "config.yaml").exists()
