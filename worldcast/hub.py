"""The release's files: the weights, the pinned third-party files, the example cases and the
evaluation data.

The weights come from the WorldCast repository on the Hugging Face Hub (docs/inference.md,
"Weights"), the Wan2.2 VAE, tokenizer and text encoder from ``Wan-AI/Wan2.2-TI2V-5B`` at a pinned
revision, the demo's tiny decoder from ``madebyollin/taehv`` at a pinned commit, and the weights of
the paper's metrics (:data:`METRIC_WEIGHTS`) from where each is published. The example cases are
listed, with each file's sha256, in ``examples/manifest.json``; the evaluation data of the paper's
evaluation sets is a dataset on the Hub (:data:`DATA_REPOSITORY`), listed in its own manifest. A
file whose digest the release pins is fetched once and checked.
"""

import json
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from functools import cache, partial
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

import yaml

from worldcast.utils.files import sha256_file

__all__ = [
    "CLOSED_LOOP_TABLES",
    "DATA_REPOSITORY",
    "DATA_REVISION",
    "EVAL_DATA_MANIFEST",
    "EVAL_DATA_MANIFEST_SHA256",
    "EVAL_SETS",
    "EXAMPLES_MANIFEST",
    "METRICS_DIR",
    "METRIC_WEIGHTS",
    "REPOSITORY",
    "REPOSITORY_CONFIG",
    "TAEHV_FILE",
    "TAEHV_REVISION",
    "TAEHV_SHA256",
    "TAEHV_URL",
    "TRAINING_CHECKPOINTS",
    "WAN22_REPO",
    "WAN22_REVISION",
    "WEIGHTS",
    "MetricWeight",
    "check_metric_weight",
    "download_eval_data",
    "download_example",
    "download_metric_weights",
    "download_weights",
    "example_cases",
    "fetch_verified",
    "write_eval_data_config",
    "write_example_config",
]

#: The WorldCast repository on the Hugging Face Hub: the weights and the example cases.
REPOSITORY = "ZiyangYe/WorldCast"
#: The repository's description: its weight files by ``paths`` key, the base model and its revision.
#: No client reads it. The Hub counts a download of a repository by the request of this file, so a
#: download of the weights fetches it with them.
REPOSITORY_CONFIG = "config.json"
#: The release files a client reads, by ``paths`` key.
WEIGHTS = {
    "checkpoint": "worldcast_4step_bf16.safetensors",  # the generator's EMA weights, bf16
    "state_model": "state_model.safetensors",  # predicted player states (Sec. 3.4)
    "depth_head": "depth_head.safetensors",
    "depth_readout": "depth_readout.safetensors",
    "prompt_embedding": "fixed_prompt_umt5xxl_bf16.safetensors",  # [1, 512, 4096] bf16
}
#: The tables of the closed loop, shipped with the code in ``configs/state_model/``, by ``paths``
#: key: the state model's cell table and the speed table of the extrapolation from the controls.
CLOSED_LOOP_TABLES = {
    "state_model_cells": "cells.json",
    "physics_prior": "physics_prior.json",
}
#: The training checkpoints (EMA weights, fp32, the generator's names): stages 2, 2s and 3.
TRAINING_CHECKPOINTS = (
    "worldcast_stage2_bidirectional_fp32.safetensors",
    "worldcast_stage2s_bidirectional_fp32.safetensors",
    "worldcast_stage3_ar_fp32.safetensors",
)
#: The Hugging Face repository of Wan2.2-TI2V-5B: its ``config.json``, VAE, tokenizer and umT5.
WAN22_REPO = "Wan-AI/Wan2.2-TI2V-5B"
#: The revision of :data:`WAN22_REPO` the paper's models were trained and run with.
WAN22_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
#: The tiny Wan2.2 decoder of the demo, github.com/madebyollin/taehv (MIT, (c) 2025 Ollin Boer
#: Bohan), 22,884,021 bytes.
TAEHV_FILE = "taew2_2.pth"
#: The commit of the TAEHV repository :data:`TAEHV_FILE` is taken from.
TAEHV_REVISION = "011dfc2112197741c540e0bdd5b7b67bcc930771"
#: Where :data:`TAEHV_FILE` is fetched from.
TAEHV_URL = f"https://raw.githubusercontent.com/madebyollin/taehv/{TAEHV_REVISION}/{TAEHV_FILE}"
#: The hex sha256 of :data:`TAEHV_FILE` at that commit.
TAEHV_SHA256 = "d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a"


class MetricWeight(NamedTuple):
    """A weights file of the paper's metrics (``tools/evaluate_paper.py``), from where it is
    published.

    Attributes:
        source (str): the Hugging Face repository ``<owner>/<name>``, or the file's URL.
        revision (str | None): the repository's revision; ``None`` for a URL.
        file (str): the file in the repository; for a URL, the name it is saved under.
        sha256 (str): the file's hex sha256.
        repo_type (str): the kind of repository, ``model`` or ``space``.
    """

    source: str
    revision: str | None
    file: str
    sha256: str
    repo_type: str = "model"


#: The weights of the paper's metrics by ``paths`` key: the detector from the WorldCast repository,
#: the others from where their authors publish them.
METRIC_WEIGHTS = {
    # the fine-tuned YOLO11x of Rendered and Unmatched, AGPL-3.0 as ultralytics, 114,445,273 bytes
    "detector": MetricWeight(
        REPOSITORY,
        "7f3f1c84efd22bbde2e28f2b354189f3ccf34a53",
        "detector/yolo11x_cs2_player.pt",
        "ac2bba62f3a1192c4d465409204014f3168c1b0bd90fafae6a9d34d708197d94",
    ),
    # GIM's dense matcher (Mat. Pix.) as its authors' Space serves it, 843,735,230 bytes
    "gim_dkm": MetricWeight(
        "xuelunshen/gim-online",
        "ab4a4eb1dbd5db062b6fc8d9912f3abff4f62cdf",
        "hloc/matchers/networks/dkm/gim_dkm_100h.ckpt",
        "bbe056bcf1b401e8dbc0ea983e69ed168eb0a330dcee35f3d9cf53048bf69ab8",
        "space",
    ),
    # LoFTR outdoor (Inliers, RotErr) where kornia fetches it, 46,341,978 bytes
    "loftr": MetricWeight(
        "https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt",
        None,
        "loftr_outdoor.ckpt",
        "21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec",
    ),
    # the Kinetics-400 I3D of FVD as TorchScript, 51,235,320 bytes
    "i3d": MetricWeight(
        "flateon/FVD-I3D-torchscript",
        "1c2d61711c52571ad617a1063e6fa691b212e184",
        "i3d_torchscript.pt",
        "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2",
    ),
}
#: The directory of the download that receives the weights of the paper's metrics.
METRICS_DIR = "metrics"
#: The list of the example cases (their files with size and sha256, their reference runs'
#: fingerprints), relative to the repository root; the cases' round and media indices are beside
#: it, in ``data/<case>/``.
EXAMPLES_MANIFEST = "examples/manifest.json"
#: The directory of the WorldCast repository that holds the example cases' files.
_EXAMPLES_DIR = "examples"
#: The dataset of the paper's evaluation data on the Hugging Face Hub: the latents, labels and tick
#: tables of its evaluation sets, derived from OpenCS2 (CC BY 4.0).
DATA_REPOSITORY = "ZiyangYe/WorldCast-data"
#: The revision of :data:`DATA_REPOSITORY` the release reads.
DATA_REVISION = "2a6ff8729b84d528175b2d1a10563666b460b778"
#: The dataset's list of its files (size and sha256) and of each set's windows.
EVAL_DATA_MANIFEST = "eval/manifest.json"
#: The hex sha256 of :data:`EVAL_DATA_MANIFEST`.
EVAL_DATA_MANIFEST_SHA256 = "a79b8ad915159b028dd6c719a78aad7de60a629be934b4e3f3b933708e764a77"
#: The evaluation sets by name: each one's index, relative to the repository root.
EVAL_SETS = {
    "validation": "configs/data/validation_index.jsonl",
    "windows": "configs/eval/window_views.jsonl",
    "rounds": "configs/eval/round_index.jsonl",
}
#: The media index of the evaluation data, in :data:`DATA_REPOSITORY`.
_EVAL_MEDIA_INDEX = "eval/media_index.jsonl"
#: How long the check that the Hugging Face Hub answers waits for it, s.
_HUB_TIMEOUT_S = 10.0
#: What a download says when the Hub cannot be reached and no copy of a file is there.
_HUB_UNREACHABLE = "the Hugging Face Hub cannot be reached (offline?)"
#: How long the fetch of a file by its URL waits for an answer, s.
_URL_TIMEOUT_S = 60


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


def _download_url(url: str, part: Path, what: str) -> None:
    """Fetch ``url`` into ``part``; when it cannot be reached, the error names the file in place
    (``part`` without its suffix) and ``what`` it is."""
    try:
        with urllib.request.urlopen(url, timeout=_URL_TIMEOUT_S) as response:
            with open(part, "wb") as file:
                shutil.copyfileobj(response, file)
    except urllib.error.HTTPError:  # an answer, not a failed connection
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError):  # none, or a cut one
        target = part.with_name(part.name.removesuffix(".part"))
        raise ConnectionError(
            f"{url} cannot be reached (offline?): no copy of {target} ({what})"
        ) from None


def _download_taehv(part: Path) -> None:
    _download_url(TAEHV_URL, part, "the demo's tiny decoder; --skip-taehv goes without it")


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


def _fetch_hub_file(repo: str, revision: str, name: str, repo_type: str, part: Path) -> None:
    """Write the file ``name`` of a Hugging Face repository at ``revision`` to ``part``.

    Raises:
        ConnectionError: the Hub cannot be reached (it names the file in place).
        FileNotFoundError: the repository, its revision or the file is not on the Hub (or not
            readable with the user's token).
    """
    if not _hub_reachable():
        raise _not_in_place([part.with_name(part.name.removesuffix(".part"))])
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    with tempfile.TemporaryDirectory(dir=part.parent) as scratch:
        try:
            path = hf_hub_download(
                repo, name, revision=revision, repo_type=repo_type, local_dir=scratch
            )
        except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
            raise FileNotFoundError(
                f"{name} is not in {repo} at {revision} on the Hugging Face Hub"
            ) from None
        shutil.move(path, part)


def _fetch_metric_weight(weight: MetricWeight, part: Path) -> None:
    """Write a weights file of :data:`METRIC_WEIGHTS` to ``part``.

    Raises:
        ConnectionError: its source cannot be reached (it names the file in place).
        FileNotFoundError: the repository, its revision or the file is not on the Hub (or not
            readable with the user's token).
    """
    if weight.revision is None:
        _download_url(weight.source, part, "a weights file of the paper's metrics")
        return
    _fetch_hub_file(weight.source, weight.revision, weight.file, weight.repo_type, part)


def _check_metric_keys(keys: Sequence[str]) -> None:
    unknown = [key for key in keys if key not in METRIC_WEIGHTS]
    if unknown:
        raise ValueError(f"unknown metric weights {unknown}; known: {', '.join(METRIC_WEIGHTS)}")


def check_metric_weight(key: str, path: str | Path) -> None:
    """Check that the file at ``path`` is the weights file ``key`` of :data:`METRIC_WEIGHTS`.

    Args:
        key (str): the weights, by ``paths`` key.
        path (str | Path): the file.

    Raises:
        ValueError: its sha256 is another.
    """
    expected = METRIC_WEIGHTS[key].sha256
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{path}: sha256 {observed} is not {key} ({expected})")


def download_metric_weights(
    out_dir: str | Path, keys: Sequence[str] = tuple(METRIC_WEIGHTS)
) -> dict[str, str]:
    """Fetch weights of the paper's metrics, each checked by its sha256 (:data:`METRIC_WEIGHTS`).

    Args:
        out_dir (str | Path): receives the files, by the last part of their name; a file already
            there with its digest is kept.
        keys (Sequence[str]): the weights, by ``paths`` key.

    Returns:
        dict[str, str]: the files' paths by ``paths`` key.

    Raises:
        ValueError: a key :data:`METRIC_WEIGHTS` does not have.
        ConnectionError: a source cannot be reached and the file is not in place; it names it.
        FileNotFoundError: a repository, its revision or a file is not on the Hub.
        RuntimeError: a fetched file has another digest.
    """
    _check_metric_keys(keys)
    out = Path(out_dir).resolve()
    paths = {}
    for key in keys:
        weight = METRIC_WEIGHTS[key]
        target = out / PurePosixPath(weight.file).name
        fetch_verified(target, weight.sha256, partial(_fetch_metric_weight, weight))
        paths[key] = str(target)
    return paths


def _write_paths(out: Path, paths: dict[str, str]) -> None:
    header = "# written by tools/download_weights.py\n"
    (out / "paths.yaml").write_text(header + yaml.safe_dump({"paths": paths}, sort_keys=False))


def download_weights(
    out_dir: str | Path,
    tables_dir: str | Path,
    *,
    wan22: bool = True,
    t5: bool = False,
    taehv: bool = True,
    training_checkpoints: bool = False,
    metrics: Sequence[str] = (),
) -> Path:
    """Download the release weights and write the ``paths`` entries that name them.

    When the Hugging Face Hub cannot be reached, the files already in ``out_dir`` are taken as they
    are, if every one is there.

    Args:
        out_dir (str | Path): receives ``worldcast/`` (the release files and the repository's
            ``config.json``), ``Wan2.2-TI2V-5B/``, ``paths.yaml``, ``taew2_2.pth`` and
            ``metrics/``.
        tables_dir (str | Path): ``configs/state_model/``, the closed loop's tables.
        wan22 (bool): the Wan2.2 snapshot's ``config.json`` (the backbone's dimensions), VAE and
            umT5 tokenizer.
        t5 (bool): also its umT5-XXL encoder (11 GB), only to re-make the prompt embedding.
        taehv (bool): the demo's tiny decoder, after ``paths.yaml`` is written (no ``paths`` key
            names it).
        training_checkpoints (bool): also the checkpoints of stages 2, 2s and 3 (fp32, 20 GB each),
            after ``paths.yaml`` is written (no ``paths`` key names them).
        metrics (Sequence[str]): also these weights of the paper's metrics, by ``paths`` key
            (:func:`download_metric_weights`), last; ``paths.yaml`` names each once it is in place.

    Returns:
        Path: ``<out_dir>/paths.yaml``, to pass as a ``--config``.

    Raises:
        ConnectionError: the Hugging Face Hub (or, for the tiny decoder or LoFTR, their server)
            cannot be reached and files are not in ``out_dir``; it names them.
        FileNotFoundError: the WorldCast repository (or a repository of a metric's weights) holds no
            file of one of the names.
    """
    _check_metric_keys(metrics)
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
    _write_paths(out, paths)
    if taehv:
        fetch_verified(out / TAEHV_FILE, TAEHV_SHA256, _download_taehv)
    if training_checkpoints:
        for name in TRAINING_CHECKPOINTS:
            _download_release_file(name, release)
    for key in metrics:
        paths.update(download_metric_weights(out / METRICS_DIR, [key]))
        _write_paths(out, paths)
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


def _fetch_data_file(name: str, source: str | Path | None, part: Path) -> None:
    """Write a file of :data:`DATA_REPOSITORY` to ``part``: from ``source``, a local copy of the
    dataset, else from the Hub."""
    if source is not None:
        shutil.copyfile(Path(source) / name, part)
        return
    _fetch_hub_file(DATA_REPOSITORY, DATA_REVISION, name, "dataset", part)


def _check_eval_set(name: str) -> None:
    if name not in EVAL_SETS:
        raise ValueError(f"unknown evaluation set {name!r}; known: {', '.join(EVAL_SETS)}")


def download_eval_data(
    name: str, out_dir: str | Path, root: str | Path, source: str | Path | None = None
) -> int:
    """Put the evaluation data of one set under ``out_dir``; returns how many files were fetched.

    The set's windows are listed in the dataset's manifest (:data:`EVAL_DATA_MANIFEST`, checked by
    its sha256), which also gives each file's sha256. A set needs the latents and labels of its
    clients or views and the tick tables of every player of their rounds; a file already in place
    with its digest is kept.

    Args:
        name (str): the set, a key of :data:`EVAL_SETS`.
        out_dir (str | Path): receives ``eval/`` as the dataset holds it.
        root (str | Path): the repository root, which holds the set's index.
        source (str | Path | None): a local copy of :data:`DATA_REPOSITORY` to copy from;
            ``None``: the Hugging Face Hub.

    Raises:
        ValueError: an unknown set, or the manifest lists the set for another index than the
            repository's.
        ConnectionError: the Hub cannot be reached and a file is not in place (it names it).
        FileNotFoundError: the dataset, or ``source``, holds no file of that name.
        RuntimeError: a fetched file has another digest.
    """
    _check_eval_set(name)
    out = Path(out_dir)
    fetched = 0

    def fetch(path: str, sha256: str) -> None:
        nonlocal fetched
        fetched += fetch_verified(out / path, sha256, partial(_fetch_data_file, path, source))

    fetch(EVAL_DATA_MANIFEST, EVAL_DATA_MANIFEST_SHA256)
    manifest = json.loads((out / EVAL_DATA_MANIFEST).read_text())
    files, listed = manifest["files"], manifest["sets"][name]
    if listed["index_sha256"] != sha256_file(Path(root) / EVAL_SETS[name]):
        raise ValueError(
            f"the evaluation data lists the set {name!r} for another {EVAL_SETS[name]} than this"
            " repository's"
        )
    fetch(_EVAL_MEDIA_INDEX, files[_EVAL_MEDIA_INDEX]["sha256"])
    rows = [json.loads(line) for line in (out / _EVAL_MEDIA_INDEX).read_text().splitlines()]
    media = {media_id for media_id, _ in listed["windows"]}
    paths = [
        f"eval/{folder}/{media_id}.npz"
        for media_id in sorted(media)
        for folder in ("latents", "vislabels", "obslabels/flashlabels", "obslabels/scopelabels")
    ]
    # every tick table of a round, also of a player the media index has no row for: the entity
    # criterion boxes every player of the round
    rounds = tuple(
        {
            f"eval/opencs2/{PurePosixPath(r['ticks_path']).parent.parent}/"
            for r in rows
            if r["media_id"] in media
        }
    )
    paths += sorted(path for path in files if path.startswith(rounds))
    for path in paths:
        fetch(path, files[path]["sha256"])
    return fetched


def write_eval_data_config(name: str, out_dir: str | Path, root: str | Path) -> Path:
    """Write the config of a downloaded evaluation set: its data paths, as the tools that read the
    set name them (``data`` and ``validation.index`` of a training config for ``validation``,
    ``paths`` of an inference config for the others, with the round index for ``rounds``).

    Args:
        name (str): the set, a key of :data:`EVAL_SETS`.
        out_dir (str | Path): where :func:`download_eval_data` put it.
        root (str | Path): the repository root, which holds the set's index.

    Returns:
        Path: ``<out_dir>/eval/<name>.yaml``, to pass as a ``--config``.
    """
    _check_eval_set(name)
    data = Path(out_dir).resolve() / "eval"
    paths = {
        "media_index": str(data / "media_index.jsonl"),
        "dataset_root": str(data / "opencs2"),
        "latent_cache_root": str(data / "latents"),
        "visibility_label_root": str(data / "vislabels"),
        "observer_signal_label_root": str(data / "obslabels"),
    }
    index = str(Path(root).resolve() / EVAL_SETS[name])
    if name == "validation":
        config = {"data": paths, "validation": {"index": index}}
    elif name == "rounds":
        config = {"paths": {"round_index": index, **paths}}
    else:
        config = {"paths": paths}
    header = (
        f"# the evaluation set {name} ({EVAL_SETS[name]}); written by tools/download_eval_data.py\n"
    )
    data.mkdir(parents=True, exist_ok=True)
    (data / f"{name}.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False))
    return data / f"{name}.yaml"
