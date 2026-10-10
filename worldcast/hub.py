"""The release's files: the weights, the pinned third-party files and the example cases.

The weights come from the WorldCast repository on the Hugging Face Hub (docs/inference.md,
"Weights"), the Wan2.2 VAE, tokenizer and text encoder from ``Wan-AI/Wan2.2-TI2V-5B`` at a pinned
revision. The example cases are listed, with each file's sha256, in ``examples/manifest.json``. A
file whose digest the release pins is fetched once and checked.
"""

import json
import shutil
from collections.abc import Callable, Sequence
from functools import cache, partial
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from worldcast.utils.files import sha256_file

__all__ = [
    "CLOSED_LOOP_TABLES",
    "EXAMPLES_MANIFEST",
    "REPOSITORY",
    "REPOSITORY_CONFIG",
    "WAN22_REPO",
    "WAN22_REVISION",
    "WEIGHTS",
    "download_example",
    "download_weights",
    "example_cases",
    "fetch_verified",
    "write_example_config",
]

#: The WorldCast repository on the Hugging Face Hub: the weights and the example cases.
REPOSITORY = "ZiyangYe/WorldCast"
#: The release files a client reads, by ``paths`` key.
WEIGHTS = {
    "checkpoint": "worldcast_4step_bf16.safetensors",  # the generator's EMA weights, bf16
    "state_model": "state_model.safetensors",  # predicted player states (Sec. 3.4)
    "depth_head": "depth_head.safetensors",
    "depth_readout": "depth_readout.safetensors",
    "prompt_embedding": "fixed_prompt_umt5xxl_bf16.safetensors",  # [1, 512, 4096] bf16
}
#: The repository's description: its weight files by ``paths`` key, the base model and its revision.
#: No client reads it. The Hub counts a download of a repository by the request of this file, so a
#: download of the weights fetches it with them.
REPOSITORY_CONFIG = "config.json"
#: The tables of the closed loop, shipped with the code in ``configs/state_model/``, by ``paths``
#: key: the state model's cell table and the speed table of the extrapolation from the controls.
CLOSED_LOOP_TABLES = {
    "state_model_cells": "cells.json",
    "physics_prior": "physics_prior.json",
}
#: The Hugging Face repository of Wan2.2-TI2V-5B: its ``config.json``, VAE, tokenizer and umT5.
WAN22_REPO = "Wan-AI/Wan2.2-TI2V-5B"
#: The revision of :data:`WAN22_REPO` the paper's models were trained and run with.
WAN22_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
#: The list of the example cases (their files with size and sha256, their reference runs'
#: fingerprints), relative to the repository root; the cases' round and media indices are beside
#: it, in ``data/<case>/``.
EXAMPLES_MANIFEST = "examples/manifest.json"
#: The directory of the WorldCast repository that holds the example cases' files.
_EXAMPLES_DIR = "examples"
#: How long the check that the Hugging Face Hub answers waits for it, s.
_HUB_TIMEOUT_S = 10.0
#: What a download says when the Hub cannot be reached and no copy of a file is there.
_HUB_UNREACHABLE = "the Hugging Face Hub cannot be reached (offline?)"


def fetch_verified(target: str | Path, sha256: str, fetch: Callable[[Path], None]) -> bool:
    """Make ``target`` the file with the hex digest ``sha256``.

    Args:
        target (str | Path): where the file belongs; kept if it is there with that digest.
        sha256 (str): the file's hex sha256.
        fetch (Callable[[Path], None]): writes the file to the path it is given.

    Returns:
        bool: whether the file was fetched.

    Raises:
        RuntimeError: the fetched file has another digest (nothing is left at ``target``).
    """
    target = Path(target)
    if target.is_file() and sha256_file(target) == sha256:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    fetch(part)
    digest = sha256_file(part)
    if digest != sha256:
        part.unlink()
        raise RuntimeError(f"{target.name}: sha256 {digest}, expected {sha256}")
    part.replace(target)
    return True


@cache
def _hub_reachable() -> bool:
    """Whether the Hugging Face Hub answers, asked once with one request: a download retries five
    times, over a minute, before it gives up on a Hub it cannot reach."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    try:
        HfApi().repo_info(REPOSITORY, timeout=_HUB_TIMEOUT_S)
    except HfHubHTTPError:  # an answer
        return True
    except Exception:  # none: an error of the transport, requests' or httpx's by the version
        return False
    return True


def _not_in_place(missing: Sequence[str | Path]) -> ConnectionError:
    """The error of a download without the Hub: what is not in place."""
    return ConnectionError(f"{_HUB_UNREACHABLE}: no copy of {', '.join(map(str, missing))}")


def _download_release_file(name: str, local_dir: str | Path | None = None) -> str:
    """Fetch a file of the WorldCast repository (into ``local_dir``, else the Hub's cache), or
    take the copy already there when the Hub cannot be reached; its path.

    Raises:
        ConnectionError: the Hub cannot be reached and no copy is there (it names the file in
            ``local_dir``, else the file of the repository).
        FileNotFoundError: the repository holds no file of that name.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    offline = not _hub_reachable()
    local_dir = None if local_dir is None else str(local_dir)
    try:
        return hf_hub_download(REPOSITORY, name, local_dir=local_dir, local_files_only=offline)
    except LocalEntryNotFoundError:  # a subclass of EntryNotFoundError: the Hub was not reached
        raise _not_in_place([Path(local_dir) / name if local_dir else name]) from None
    except EntryNotFoundError:
        raise FileNotFoundError(f"{name} is not in {REPOSITORY} on the Hugging Face Hub") from None


def download_weights(
    out_dir: str | Path,
    tables_dir: str | Path,
    *,
    wan22: bool = True,
    t5: bool = False,
) -> Path:
    """Download the release weights and write the ``paths`` entries that name them.

    When the Hugging Face Hub cannot be reached, the files already in ``out_dir`` are taken as they
    are, if every one is there.

    Args:
        out_dir (str | Path): receives ``worldcast/`` (the release files and the repository's
            ``config.json``), ``Wan2.2-TI2V-5B/``, and ``paths.yaml``.
        tables_dir (str | Path): ``configs/state_model/``, the closed loop's tables.
        wan22 (bool): the Wan2.2 snapshot's ``config.json`` (the backbone's dimensions), VAE and
            umT5 tokenizer.
        t5 (bool): also its umT5-XXL encoder (11 GB), only to re-make the prompt embedding.

    Returns:
        Path: ``<out_dir>/paths.yaml``, to pass as a ``--config``.

    Raises:
        ConnectionError: the Hugging Face Hub cannot be reached and files are not in ``out_dir``;
            it names them.
        FileNotFoundError: the WorldCast repository holds no file of one of the names.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    release, snapshot = out / "worldcast", out / "Wan2.2-TI2V-5B"
    patterns = []
    if wan22:
        from worldcast.modeling.wan22.text_encoder import T5_CHECKPOINT_NAME, TOKENIZER_SUBDIR
        from worldcast.modeling.wan22.vae import VAE_CHECKPOINT_NAME

        patterns = ["config.json", VAE_CHECKPOINT_NAME, f"{Path(TOKENIZER_SUBDIR).as_posix()}/*"]
        patterns += [T5_CHECKPOINT_NAME] if t5 else []
    offline = not _hub_reachable()
    if offline:  # the copies in place, every one
        missing = [release / name for name in WEIGHTS.values() if not (release / name).is_file()]
        missing += [
            snapshot / pattern
            for pattern in patterns
            if not any(path.is_file() for path in snapshot.glob(pattern))
        ]
        if missing:
            raise _not_in_place(missing)
        paths = {key: str(release / name) for key, name in WEIGHTS.items()}
    else:
        paths = {key: _download_release_file(name, release) for key, name in WEIGHTS.items()}
        _download_release_file(REPOSITORY_CONFIG, release)
    tables = Path(tables_dir).resolve()
    paths.update({key: str(tables / name) for key, name in CLOSED_LOOP_TABLES.items()})
    if wan22 and offline:
        paths["wan22_root"] = str(snapshot)
    elif wan22:
        from huggingface_hub import snapshot_download

        paths["wan22_root"] = snapshot_download(
            WAN22_REPO, revision=WAN22_REVISION, allow_patterns=patterns, local_dir=str(snapshot)
        )
    header = "# written by tools/download_weights.py\n"
    (out / "paths.yaml").write_text(header + yaml.safe_dump({"paths": paths}, sort_keys=False))
    return out / "paths.yaml"


def example_cases(manifest: str | Path, names: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """The cases of the manifest (:data:`EXAMPLES_MANIFEST`) named ``names``, every case when empty.

    Raises:
        ValueError: the manifest has no case of one of the names.
    """
    cases = json.loads(Path(manifest).read_text())["cases"]
    unknown = [name for name in names if name not in cases]
    if unknown:
        raise ValueError(f"unknown case(s) {unknown}; known: {', '.join(cases)}")
    return {name: case for name, case in cases.items() if not names or name in names}


def _copy_release_file(name: str, source: str | Path | None, part: Path) -> None:
    """Copy a file of the WorldCast repository from ``source``, else from the Hugging Face Hub."""
    if source is not None:
        shutil.copyfile(Path(source) / name, part)
        return
    try:
        shutil.copyfile(_download_release_file(name), part)
    except ConnectionError:  # the copy that is missing is the case's file
        raise _not_in_place([part.with_name(PurePosixPath(name).name)]) from None


def download_example(
    case: dict[str, Any], out_dir: str | Path, source: str | Path | None = None
) -> int:
    """Put the files of an example case under ``out_dir``; returns how many were fetched.

    Args:
        case (dict[str, Any]): a case of :func:`example_cases`.
        out_dir (str | Path): receives ``data/<case>/`` (the case's tick tables, first latents and
            labels) and ``expected/<case>.mp4``, as the repository's ``examples/`` holds them.
        source (str | Path | None): a local copy of :data:`REPOSITORY` to copy from; ``None``: the
            Hugging Face Hub, or the copy in its cache when it cannot be reached.

    Raises:
        ConnectionError: the Hub cannot be reached and a file is neither in place nor in its cache
            (it names the file in place).
        FileNotFoundError: the repository, or ``source``, holds no file of that name.
    """
    fetched = 0
    for name, file in case["files"].items():
        target = Path(out_dir) / PurePosixPath(name).relative_to(_EXAMPLES_DIR)
        fetched += fetch_verified(target, file["sha256"], partial(_copy_release_file, name, source))
    return fetched


def write_example_config(
    name: str, case: dict[str, Any], out_dir: str | Path, manifest: str | Path
) -> Path:
    """Write the config of a downloaded example case: its data paths and its length.

    Args:
        name (str): the case's name.
        case (dict[str, Any]): the case of :func:`example_cases`.
        out_dir (str | Path): where :func:`download_example` put the case.
        manifest (str | Path): the manifest; the case's round and media index are beside it, in
            ``data/<name>/``.

    Returns:
        Path: ``<out_dir>/data/<name>/config.yaml``, to pass as a ``--config``.
    """
    data = Path(out_dir).resolve() / "data" / name
    indices = Path(manifest).resolve().parent / "data" / name
    paths = {
        "round_index": indices / "round_index.jsonl",
        "media_index": indices / "media_index.jsonl",
        "dataset_root": data / "opencs2",
        "latent_cache_root": data / "first_latents",
        "visibility_label_root": data / "vislabels",
        "observer_signal_label_root": data / "obslabels",
    }
    config = {
        "paths": {key: str(path) for key, path in paths.items()},
        "run": {"latent_frames": case["latent_frames"]},
    }
    header = (
        f"# {name}: {case['map']}, match {case['match_id']} round {case['round']},"
        f" {len(case['clients'])} clients; written by tools/download_examples.py\n"
    )
    data.mkdir(parents=True, exist_ok=True)
    (data / "config.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False))
    return data / "config.yaml"
