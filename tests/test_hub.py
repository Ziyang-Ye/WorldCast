"""The release's files: a file is fetched once and checked by its sha256; the weights and what the
paths file names; the example cases and their configs."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from worldcast import hub
from worldcast.config.inference import load_config

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / hub.EXAMPLES_MANIFEST


def test_the_downloads_load_no_torch():
    code = "import sys, worldcast.hub; print('torch' in sys.modules)"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert run.stdout.strip() == "False"


def test_a_file_is_fetched_once_and_checked(tmp_path):
    target, fetched = tmp_path / "weights" / "file.bin", []

    def fetch(part: Path) -> None:
        fetched.append(part.name)
        part.write_bytes(b"release")

    digest = hashlib.sha256(b"release").hexdigest()
    assert hub.fetch_verified(target, digest, fetch) and target.read_bytes() == b"release"
    assert not hub.fetch_verified(target, digest, fetch) and fetched == ["file.bin.part"]
    # another file under that name is replaced; a download with another digest leaves nothing
    target.write_bytes(b"stale")
    assert hub.fetch_verified(target, digest, fetch) and target.read_bytes() == b"release"
    with pytest.raises(RuntimeError, match="sha256"):
        hub.fetch_verified(tmp_path / "other.bin", "0" * 64, fetch)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["weights"]


def test_the_example_cases_of_the_manifest():
    assert hub.EXAMPLES_MANIFEST == "examples/manifest.json"
    cases = hub.example_cases(MANIFEST)
    assert list(hub.example_cases(MANIFEST, ["nuke_r07", "dust2_r09"])) == ["dust2_r09", "nuke_r07"]
    assert len(cases) == 6 and len(cases["mirage_r16"]["clients"]) == 3
    assert cases["mirage_r16"]["latent_frames"] == 121
    with pytest.raises(ValueError, match="unknown case"):
        hub.example_cases(MANIFEST, ["mirage_r17"])


def test_an_example_case_is_copied_from_a_local_source(tmp_path):
    files = {"examples/data/a/vislabels/p00.npz": b"labels", "examples/expected/a.mp4": b"video"}
    for name, content in files.items():
        (tmp_path / "source" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "source" / name).write_bytes(content)
    case = {
        "files": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in files.items()
        }
    }
    out = tmp_path / "out"
    assert hub.download_example(case, out, tmp_path / "source") == 2
    assert hub.download_example(case, out, tmp_path / "nowhere") == 0  # in place: nothing is read
    # the files land as the repository's examples/ holds them, without that directory
    assert (out / "data/a/vislabels/p00.npz").read_bytes() == b"labels"
    assert (out / "expected/a.mp4").read_bytes() == b"video"


def test_the_config_of_a_downloaded_case_names_its_paths_and_its_length(tmp_path):
    case = hub.example_cases(MANIFEST, ["mirage_r16"])["mirage_r16"]
    written = hub.write_example_config("mirage_r16", case, tmp_path, MANIFEST)
    data, indices = tmp_path.resolve() / "data" / "mirage_r16", REPO / "examples/data/mirage_r16"
    assert written == data / "config.yaml"
    assert yaml.safe_load(written.read_text()) == {
        "paths": {
            "round_index": str(indices / "round_index.jsonl"),
            "media_index": str(indices / "media_index.jsonl"),
            "dataset_root": str(data / "opencs2"),
            "latent_cache_root": str(data / "first_latents"),
            "visibility_label_root": str(data / "vislabels"),
            "observer_signal_label_root": str(data / "obslabels"),
        },
        "run": {"latent_frames": 121},
    }
    cfg = load_config(written)  # the tracked indices are there: the config loads as it is
    assert Path(cfg.paths.round_index).is_file() and Path(cfg.paths.media_index).is_file()


@pytest.fixture
def stub_hub(monkeypatch):
    """The Hugging Face Hub stubbed out, and reached; records what is fetched. The WorldCast
    repository holds the files named in ``held``. Told not to reach the Hub, a download takes the
    copy in its local directory, as the Hub's library does (:func:`test_the_hubs_library_offline`
    pins that)."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    fetched, held = [], {*hub.WEIGHTS.values(), hub.REPOSITORY_CONFIG}

    def hf_hub_download(repo, name, local_dir=None, local_files_only=False):
        if local_files_only:
            if not (Path(local_dir) / name).is_file():
                raise LocalEntryNotFoundError(f"no local copy of {name}")
        elif name not in held:
            raise EntryNotFoundError(f"404: {name}")
        else:
            fetched.append((repo, name))
        return str(Path(local_dir) / name)

    def snapshot_download(repo, revision, allow_patterns, local_dir):
        fetched.append((repo, revision, tuple(allow_patterns)))
        return local_dir

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(hub, "_hub_reachable", lambda: True)
    return fetched, held


def test_the_paths_file_names_the_release_files(tmp_path, stub_hub):
    fetched, _ = stub_hub
    tables = REPO / "configs" / "state_model"
    out = tmp_path / "weights"
    paths_file = hub.download_weights(out, tables)
    paths = yaml.safe_load(paths_file.read_text())["paths"]
    release = out.resolve() / "worldcast"
    assert paths == {
        "checkpoint": str(release / "worldcast_4step_bf16.safetensors"),
        "state_model": str(release / "state_model.safetensors"),
        "depth_head": str(release / "depth_head.safetensors"),
        "depth_readout": str(release / "depth_readout.safetensors"),
        "prompt_embedding": str(release / "fixed_prompt_umt5xxl_bf16.safetensors"),
        "state_model_cells": str(tables / "cells.json"),
        "physics_prior": str(tables / "physics_prior.json"),
        "wan22_root": str(out.resolve() / "Wan2.2-TI2V-5B"),
    }
    assert all(Path(paths[key]).is_file() for key in hub.CLOSED_LOOP_TABLES)
    names = [entry[1] for entry in fetched if entry[0] == hub.REPOSITORY]
    assert names == [*hub.WEIGHTS.values(), hub.REPOSITORY_CONFIG]  # and its description
    # the Wan2.2 snapshot at the pinned revision: its config, VAE and tokenizer, not the encoder
    snapshot = ("config.json", "Wan2.2_VAE.pth", "google/umt5-xxl/*")
    revision = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
    assert ("Wan-AI/Wan2.2-TI2V-5B", revision, snapshot) in fetched


def test_a_file_the_repository_lacks_is_named(tmp_path, stub_hub, monkeypatch):
    monkeypatch.setattr(hub, "WEIGHTS", {**hub.WEIGHTS, "extra": "extra.safetensors"})
    tables, out = REPO / "configs" / "state_model", tmp_path / "weights"
    message = "extra.safetensors is not in ZiyangYe/WorldCast on the Hugging Face Hub"
    with pytest.raises(FileNotFoundError, match=message):
        hub.download_weights(out, tables, wan22=False)
    assert not (out / "paths.yaml").exists()


def offline_message(*missing) -> str:
    return f"the Hugging Face Hub cannot be reached (offline?): no copy of {', '.join(missing)}"


def test_without_the_hub_the_copies_in_place_are_taken(tmp_path, stub_hub, monkeypatch):
    """Every file not in place is named; the Wan2.2 snapshot is its config, VAE and tokenizer
    (and umT5 when asked), not a folder."""
    fetched, _ = stub_hub
    monkeypatch.setattr(hub, "_hub_reachable", lambda: False)
    tables, out = REPO / "configs" / "state_model", tmp_path / "weights"
    release, snapshot = out.resolve() / "worldcast", out.resolve() / "Wan2.2-TI2V-5B"
    wan22 = [
        f"{snapshot}/config.json",
        f"{snapshot}/Wan2.2_VAE.pth",
        f"{snapshot}/google/umt5-xxl/*",
    ]
    with pytest.raises(ConnectionError) as offline:
        hub.download_weights(out, tables)
    assert str(offline.value) == offline_message(
        *[f"{release}/{name}" for name in hub.WEIGHTS.values()], *wan22
    )
    for name in hub.WEIGHTS.values():
        (release / name).parent.mkdir(parents=True, exist_ok=True)
        (release / name).write_bytes(b"in place")
    (snapshot / "google" / "umt5-xxl").mkdir(parents=True)  # a folder without its files
    (snapshot / "config.json").write_text("{}")
    with pytest.raises(ConnectionError) as offline:
        hub.download_weights(out, tables)
    assert str(offline.value) == offline_message(*wan22[1:])
    (snapshot / "Wan2.2_VAE.pth").write_bytes(b"in place")
    (snapshot / "google" / "umt5-xxl" / "tokenizer.json").write_text("{}")
    with pytest.raises(ConnectionError) as offline:
        hub.download_weights(out, tables, t5=True)
    assert str(offline.value) == offline_message(f"{snapshot}/models_t5_umt5-xxl-enc-bf16.pth")
    paths = yaml.safe_load(hub.download_weights(out, tables).read_text())["paths"]
    assert paths["checkpoint"] == f"{release}/worldcast_4step_bf16.safetensors"
    assert paths["wan22_root"] == str(snapshot)
    assert fetched == []  # nothing asked of the Hub


def test_the_hubs_library_offline(tmp_path, monkeypatch):
    """The Hub's library itself, never asked to reach the Hub: a file in the local directory, or
    in the Hub's cache, is taken; a file in neither is named, the case's file in place."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(hub, "_hub_reachable", lambda: False)
    local = tmp_path / "weights" / "worldcast"
    (local / "stage.safetensors").parent.mkdir(parents=True)
    (local / "stage.safetensors").write_bytes(b"in place")
    assert hub._download_release_file("stage.safetensors", local) == str(
        local / "stage.safetensors"
    )
    with pytest.raises(ConnectionError) as offline:
        hub._download_release_file("other.safetensors", local)
    assert str(offline.value) == offline_message(str(local / "other.safetensors"))
    name, content = "examples/data/a/vislabels/p00.npz", b"labels"
    case = {"files": {name: {"bytes": 6, "sha256": hashlib.sha256(content).hexdigest()}}}
    with pytest.raises(ConnectionError) as offline:
        hub.download_example(case, tmp_path / "out")
    assert str(offline.value) == offline_message(str(tmp_path / "out/data/a/vislabels/p00.npz"))
    repository = tmp_path / "cache" / f"models--{hub.REPOSITORY.replace('/', '--')}"
    commit = "0" * 40
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text(commit)
    (repository / "snapshots" / commit / name).parent.mkdir(parents=True)
    (repository / "snapshots" / commit / name).write_bytes(content)
    assert hub.download_example(case, tmp_path / "out") == 1
    assert (tmp_path / "out/data/a/vislabels/p00.npz").read_bytes() == content


class ConnectError(Exception):
    """An error of httpx's transport, which huggingface_hub 1 and later raise without a
    connection."""


def test_whether_the_hub_answers_is_asked_once_with_one_request(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import HfHubHTTPError

    asked = []

    def answer(error):
        def repo_info(self, repo, timeout):
            asked.append((repo, timeout))
            if error is not None:
                raise error

        return repo_info

    for error, answers in (
        (None, True),
        (HfHubHTTPError("401 Client Error"), True),  # an answer, if not the one hoped for
        # the transport errors of huggingface_hub 0 are OSErrors
        (ConnectionError("Name or service not known"), False),
        (TimeoutError("Connection to huggingface.co timed out"), False),
        (ConnectError("[Errno 8] nodename nor servname provided"), False),
    ):
        monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", answer(error))
        hub._hub_reachable.cache_clear()
        assert hub._hub_reachable() is answers and hub._hub_reachable() is answers
    hub._hub_reachable.cache_clear()
    assert asked == [("ZiyangYe/WorldCast", 10.0)] * 5
