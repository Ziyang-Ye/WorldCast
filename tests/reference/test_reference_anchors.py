"""Reference anchors: the release client against the paper run's own fingerprints (GPU, the real weights and data).

The nine surviving Table-3 WorldCast cells (d57-d59, d75-d77, d90-d92: three rounds x three clients) recorded the
first 32 hex digits of the sha256 of four float32 tensors (``memory_deploy.tensor_fingerprint``):

* ``noise``: the CPU entry-noise draw ``randn((1, latents - 1, 48, 24, 42), Generator.manual_seed(20260917))``;
* ``initial``: the sink, latent 0 of the cached first window, cast to bf16;
* ``prefix``: latents 0-24 of the client (the plain prefix; generator and sampler, no peers);
* ``final``: all latents of the client (``latents.npy``; the whole three-client round).

Skipped unless ``WORLDCAST_REFERENCE_CONFIG`` names a YAML (merged over ``configs/infer/worldcast_4step.yaml``)
whose ``paths`` point at the release weights and the paper's data: ``round_index`` must be the paper's 96-row
``wholeround_index.jsonl`` (cell dNN is its row NN). Optional: ``WORLDCAST_REFERENCE_GPUS`` (default ``0,1,2``),
``WORLDCAST_REFERENCE_ROUNDS`` (e.g. ``d90``; default all three), ``WORLDCAST_REFERENCE_OUT`` (default: a temp dir).

Where bit-equality is expected: ``initial`` anywhere; ``noise`` on x86-64 with torch 2.9.1 (the CPU normal kernel is
platform dependent; arm64 / torch 2.8 gives another draw); ``prefix`` and ``final`` on an NVIDIA H20 with torch
2.9.1+cu128 and flash-attn 2.8.3 (FA2), the paper's stack, where all 27 fingerprints were reproduced. Elsewhere
compare the decoded videos with a PSNR threshold instead.

Run::

    WORLDCAST_REFERENCE_CONFIG=ref.yaml python -m pytest tests/reference -q
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SEED = 20260917
#: cell -> (index row, media id, latents requested = latents generated, fingerprints)
CELLS = {
    "d57": (
        57,
        "2392812-de_dust2-r09-p05",
        441,
        dict(
            noise="575641875628552dffd0ca1e395ea1ba",
            initial="6fbdb2df99cb178bb1126454262b194b",
            prefix="9efddaced5fb523b2f68b8e8c181f191",
            final="3ded190845885cd9a24696d0d8c898cb",
        ),
    ),
    "d58": (
        58,
        "2392812-de_dust2-r09-p08",
        441,
        dict(
            noise="575641875628552dffd0ca1e395ea1ba",
            initial="47cd0d0b3c762d09ac2f85eea93f91be",
            prefix="6257fffd31cae0711d0dcf8f8c1703e8",
            final="9c61593771fc69dc085013a466d57e59",
        ),
    ),
    "d59": (
        59,
        "2392812-de_dust2-r09-p09",
        441,
        dict(
            noise="575641875628552dffd0ca1e395ea1ba",
            initial="b5a2e2cc971d565a1e74d98a08610552",
            prefix="cbca2ed3e2e610810b7ee8015c85dced",
            final="df88be446719b004174774041d1bdf55",
        ),
    ),
    "d75": (
        75,
        "2392796-de_mirage-r07-p05",
        281,
        dict(
            noise="cd407421bebbbcdb807053aaf5f1af5e",
            initial="f7e3a37e69042a419321cf21b4527951",
            prefix="c9695dac2e42d95a60e37c5a28f9fcc5",
            final="3ab3057683bca28fd7753607e8e53a8c",
        ),
    ),
    "d76": (
        76,
        "2392796-de_mirage-r07-p06",
        281,
        dict(
            noise="cd407421bebbbcdb807053aaf5f1af5e",
            initial="5b710e5f81651752a44c93376de3ef7c",
            prefix="36c3dfe9fd974c57b3be0a6c8f8bb502",
            final="bddd2af25bd4993cb9448ad10b7f7c32",
        ),
    ),
    "d77": (
        77,
        "2392796-de_mirage-r07-p09",
        281,
        dict(
            noise="cd407421bebbbcdb807053aaf5f1af5e",
            initial="e7ca844c1cc5a367005a7c7e2867bd42",
            prefix="4597cffeed86a09c8ea08c5603492f85",
            final="b5981222d0b195b18e3597c0f0d4ee1a",
        ),
    ),
    "d90": (
        90,
        "2393226-de_mirage-r16-p00",
        241,
        dict(
            noise="754ce76614ffb77c0d715363e0d216b1",
            initial="268fc85969ea3ceb1b1813376e534a10",
            prefix="3e9a8ea6ca49eb655f1139079be1569c",
            final="41e1927eb9bcb78006ac2f778498d99e",
        ),
    ),
    "d91": (
        91,
        "2393226-de_mirage-r16-p01",
        241,
        dict(
            noise="754ce76614ffb77c0d715363e0d216b1",
            initial="dd6ae99974a9eb4d1de0838db4848729",
            prefix="af0fe59f49bf7ec77a6642126a293da8",
            final="c92cdd2d77fc6a65accc629992a8ad5e",
        ),
    ),
    "d92": (
        92,
        "2393226-de_mirage-r16-p02",
        241,
        dict(
            noise="754ce76614ffb77c0d715363e0d216b1",
            initial="c40968128a8663b230fbf5a4b4070c60",
            prefix="d5f350b0ad3384063ac0922e792f8a65",
            final="3e438e0e1ecdd869e810ff098543c1a3",
        ),
    ),
}
ROUNDS = {"d57": ("d57", "d58", "d59"), "d75": ("d75", "d76", "d77"), "d90": ("d90", "d91", "d92")}
PLAIN_PREFIX = 25


def fingerprint(array) -> str:
    """``memory_deploy.tensor_fingerprint``: sha256 of the contiguous float32 bytes, first 32 hex digits."""
    a = (
        array.detach().float().cpu().numpy()
        if hasattr(array, "detach")
        else np.asarray(array, dtype=np.float32)
    )
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:32]


@pytest.fixture(scope="module")
def reference_config():
    path = os.environ.get("WORLDCAST_REFERENCE_CONFIG")
    if not path:
        pytest.skip(
            "set WORLDCAST_REFERENCE_CONFIG to a YAML with the release weights and the paper's data"
        )
    from worldcast.config.loader import DEFAULT_CONFIG, load_cli_config

    return path, load_cli_config([str(DEFAULT_CONFIG), path], {})


def _rounds():
    wanted = os.environ.get("WORLDCAST_REFERENCE_ROUNDS")
    return [r for r in ROUNDS if not wanted or r in wanted.split(",")]


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_entry_noise(reference_config, cell):
    _, _, latents, want = CELLS[cell]
    g = torch.Generator(device="cpu").manual_seed(SEED)
    noise = torch.randn((1, latents - 1, 48, 24, 42), generator=g)
    assert (
        fingerprint(noise) == want["noise"]
    ), "the CPU normal kernel differs (expected on x86-64, torch 2.9.1)"


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_initial_latent(reference_config, cell):
    from worldcast.data.index import load_round_index_row
    from worldcast.data.latents import load_first_latent

    _, cfg = reference_config
    row_index, media_id, _, want = CELLS[cell]
    row = load_round_index_row(cfg.paths.round_index, row_index)
    assert row.media_id == media_id, "paths.round_index is not the paper's wholeround_index.jsonl"
    sink = load_first_latent(cfg.paths.latent_cache_root, media_id, row.start_frame)[None].to(
        torch.bfloat16
    )
    assert fingerprint(sink) == want["initial"]


@pytest.fixture(scope="module")
def round_outputs(reference_config, tmp_path_factory):
    """Run each requested round once (``tools/run_session.py``, one GPU per client); ``{cell: latents.npy}``."""
    path, cfg = reference_config
    if torch.cuda.device_count() < 3:
        pytest.skip("a round needs three GPUs (one per client, lock-step)")
    out = Path(os.environ.get("WORLDCAST_REFERENCE_OUT") or tmp_path_factory.mktemp("reference"))
    gpus = os.environ.get("WORLDCAST_REFERENCE_GPUS", "0,1,2")
    found = {}
    for name in _rounds():
        cells = ROUNDS[name]
        rows = [str(CELLS[c][0]) for c in cells]
        latents = CELLS[cells[0]][2]
        session = out / name
        cmd = [
            sys.executable,
            str(REPO / "tools" / "run_session.py"),
            "--config",
            str(REPO / "configs" / "infer" / "worldcast_4step.yaml"),
            "--config",
            path,
            "--set",
            f"run.latents={latents}",
            "--set",
            f"run.seed={SEED}",
            "--rows",
            *rows,
            "--gpus",
            gpus,
            "--out-dir",
            str(session),
        ]
        subprocess.run(cmd, check=True)
        for cell in cells:
            found[cell] = next(session.glob(f"*/{CELLS[cell][1]}/latents.npy"))
    return found


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_plain_prefix_and_final_latents(round_outputs, cell):
    if cell not in round_outputs:
        pytest.skip("round not requested (WORLDCAST_REFERENCE_ROUNDS)")
    _, _, latents, want = CELLS[cell]
    lat = np.load(round_outputs[cell])
    assert lat.dtype == np.float32 and lat.shape == (latents, 48, 24, 42)
    assert (
        fingerprint(lat[:PLAIN_PREFIX]) == want["prefix"]
    ), "plain prefix (generator + sampler) differs"
    assert fingerprint(lat) == want["final"], "final latents (the three-client round) differ"
