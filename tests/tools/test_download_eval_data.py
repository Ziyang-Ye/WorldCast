"""``tools/download_eval_data.py`` and the evaluation data of ``worldcast.hub`` on the CPU: a set
copied from a local copy of the dataset with exactly its files and its config; a file fetched once;
a set listed for another index, a file with another digest and an unknown set refused."""

import hashlib
import json

import pytest
import yaml

from tests.tools.support import load_tool
from worldcast import hub
from worldcast.config.inference import load_config

#: Two rounds of two players each; the set ``rounds`` reads one client of the first round, the
#: set ``windows`` one view of the second.
ROWS = [
    {"match_id": 7, "map_name": "de_nuke", "round": 3, "media_id": f"7-de_nuke-r03-p0{slot}"}
    for slot in (0, 1)
] + [
    {"match_id": 8, "map_name": "de_mirage", "round": 5, "media_id": f"8-de_mirage-r05-p0{slot}"}
    for slot in (0, 1)
]
for row in ROWS:
    row["ticks_path"] = (
        f"rounds/match_id={row['match_id']}/map_name={row['map_name']}"
        f"/round={row['round']:02d}/player={row['media_id'][-2:]}/ticks.parquet"
    )
#: A tick table of the first round whose player has no media row.
UNLISTED_TICKS = "rounds/match_id=7/map_name=de_nuke/round=03/player=06/ticks.parquet"
SETS = {
    "rounds": [["7-de_nuke-r03-p00", "win_000000"]],
    "windows": [["8-de_mirage-r05-p01", "win_000320"]],
}


def build(tmp_path, monkeypatch, *, index_digest=None):
    """A repository root with the sets' indices and a local copy of the dataset; returns both."""
    root, source = tmp_path / "repo", tmp_path / "hub"
    files, sets = {}, {}
    for name, windows in SETS.items():
        index = root / hub.EVAL_SETS[name]
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(f"{name}\n")
        digest = index_digest or hashlib.sha256(index.read_bytes()).hexdigest()
        sets[name] = {"index": hub.EVAL_SETS[name], "index_sha256": digest, "windows": windows}

    def put(path: str, data: bytes) -> None:
        (source / path).parent.mkdir(parents=True, exist_ok=True)
        (source / path).write_bytes(data)
        files[path] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    put("eval/media_index.jsonl", "".join(json.dumps(row) + "\n" for row in ROWS).encode())
    for row in ROWS:
        put(f"eval/opencs2/{row['ticks_path']}", row["media_id"].encode())
        for folder in ("latents", "vislabels", "obslabels/flashlabels", "obslabels/scopelabels"):
            put(f"eval/{folder}/{row['media_id']}.npz", f"{folder} {row['media_id']}".encode())
    put(f"eval/opencs2/{UNLISTED_TICKS}", b"unlisted")
    manifest = json.dumps({"about": "", "files": files, "sets": sets}).encode()
    (source / hub.EVAL_DATA_MANIFEST).write_bytes(manifest)
    monkeypatch.setattr(hub, "EVAL_DATA_MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest())
    return root, source


def fetched_files(out) -> set[str]:
    return {str(path.relative_to(out)) for path in (out / "eval").rglob("*") if path.is_file()}


def test_a_set_brings_its_media_and_every_player_of_their_rounds(tmp_path, monkeypatch):
    root, source = build(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert hub.download_eval_data("rounds", out, root, source) == 1 + 1 + 4 + 3
    assert fetched_files(out) == {
        "eval/manifest.json",
        "eval/media_index.jsonl",
        "eval/latents/7-de_nuke-r03-p00.npz",
        "eval/vislabels/7-de_nuke-r03-p00.npz",
        "eval/obslabels/flashlabels/7-de_nuke-r03-p00.npz",
        "eval/obslabels/scopelabels/7-de_nuke-r03-p00.npz",
        f"eval/opencs2/{ROWS[0]['ticks_path']}",
        f"eval/opencs2/{ROWS[1]['ticks_path']}",
        f"eval/opencs2/{UNLISTED_TICKS}",
    }
    assert hub.download_eval_data("rounds", out, root, source) == 0


def test_the_configs_name_the_paths_as_the_tools_read_them(tmp_path, monkeypatch):
    root, source = build(tmp_path, monkeypatch)
    out = tmp_path / "data disk"
    hub.download_eval_data("rounds", out, root, source)
    config = load_config([hub.write_eval_data_config("rounds", out, root)])
    assert config.paths.round_index == str((root / hub.EVAL_SETS["rounds"]).resolve())
    assert config.paths.latent_cache_root == str(out.resolve() / "eval" / "latents")
    validation = yaml.safe_load(hub.write_eval_data_config("validation", out, root).read_text())
    assert validation["validation"] == {
        "index": str((root / hub.EVAL_SETS["validation"]).resolve())
    }
    assert validation["data"]["dataset_root"] == str(out.resolve() / "eval" / "opencs2")
    assert (
        "round_index"
        not in yaml.safe_load(hub.write_eval_data_config("windows", out, root).read_text())["paths"]
    )


def test_a_set_listed_for_another_index_is_refused(tmp_path, monkeypatch):
    root, source = build(tmp_path, monkeypatch, index_digest="0" * 64)
    with pytest.raises(ValueError, match="another configs/eval/round_index.jsonl"):
        hub.download_eval_data("rounds", tmp_path / "out", root, source)


def test_a_file_with_another_digest_is_refused(tmp_path, monkeypatch):
    root, source = build(tmp_path, monkeypatch)
    (source / "eval/latents/7-de_nuke-r03-p00.npz").write_bytes(b"other")
    with pytest.raises(RuntimeError, match="7-de_nuke-r03-p00.npz: sha256"):
        hub.download_eval_data("rounds", tmp_path / "out", root, source)
    assert not (tmp_path / "out/eval/latents/7-de_nuke-r03-p00.npz").exists()


def test_the_tool_downloads_sets_and_refuses_an_unknown_one(tmp_path, monkeypatch, capsys):
    root, source = build(tmp_path, monkeypatch)
    tool = load_tool("download_eval_data")
    monkeypatch.setattr(tool, "ROOT", root)
    out = tmp_path / "out"
    assert tool.main(["--out-dir", str(out), "--source", str(source), "rounds", "windows"]) == 0
    printed = capsys.readouterr().out
    assert "rounds: 9 files fetched" in printed and "windows: 6 files fetched" in printed
    assert (out / "eval" / "windows.yaml").is_file()
    with pytest.raises(SystemExit):
        tool.main(["--out-dir", str(out), "--source", str(source), "table4"])
    assert "invalid choice: 'table4'" in capsys.readouterr().err
    with pytest.raises(ValueError, match="unknown evaluation set 'table4'"):
        hub.download_eval_data("table4", out, root, source)
