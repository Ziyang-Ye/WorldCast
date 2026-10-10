"""The client against the fingerprints of the paper's Table-3 runs (GPU, the paper's data).

``examples/table3_fingerprints.json`` holds what the paper's runs recorded for three rounds of three
clients (its ``about``): the entry noise, the first frame, latents 0-24 and all latents.

Skipped unless ``WORLDCAST_REFERENCE_CONFIG`` names a YAML whose ``paths`` point at the release
weights and the paper's data (``round_index`` is the paper's 96-row index). Optional:
``WORLDCAST_REFERENCE_GPUS`` (default ``0,1,2``), ``WORLDCAST_REFERENCE_ROUNDS`` (round names of
the table, comma-separated; default all), ``WORLDCAST_REFERENCE_OUT`` (default: a temp dir).

Where bit equality is expected: the first frame anywhere; the entry noise on x86-64 with torch 2.9.1
(the CPU normal kernel is platform dependent); latents 0-24 and all latents on an NVIDIA H20 with
torch 2.9.1+cu128 and flash-attn 2.8.3 (FA2), the stack of the reference runs. Run::

    WORLDCAST_REFERENCE_CONFIG=ref.yaml python -m pytest tests/reference -q
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from worldcast.config.inference import load_config
from worldcast.data import load_first_latent, load_round_index_row
from worldcast.engine.inference.reference import (
    ReferenceRun,
    entry_noise_fingerprint,
    first_frame_fingerprint,
    verify_latents,
)

REPO = Path(__file__).resolve().parents[2]
TABLE = json.loads((REPO / "examples" / "table3_fingerprints.json").read_text())
#: media id -> (index row, latents requested = latents generated, fingerprints)
CLIENTS = {
    media_id: (row, case["latent_frames"], recorded)
    for case in TABLE["cases"].values()
    for row, (media_id, recorded) in zip(case["rows"], case["fingerprints"].items())
}


@pytest.fixture(scope="module")
def reference_config():
    path = os.environ.get("WORLDCAST_REFERENCE_CONFIG")
    if not path:
        pytest.skip(
            "set WORLDCAST_REFERENCE_CONFIG to a YAML with the release weights and the paper's data"
        )
    return path, load_config(path)


@pytest.mark.parametrize("media_id", sorted(CLIENTS))
def test_entry_noise(reference_config, media_id):
    _, latents, recorded = CLIENTS[media_id]
    assert (
        entry_noise_fingerprint(TABLE["seed"], latents) == recorded["entry_noise"]
    ), "the CPU normal kernel differs (expected on x86-64, torch 2.9.1)"


@pytest.mark.parametrize("media_id", sorted(CLIENTS))
def test_first_frame(reference_config, media_id):
    _, cfg = reference_config
    index_row, _, recorded = CLIENTS[media_id]
    row = load_round_index_row(cfg.paths.round_index, index_row)
    assert row.media_id == media_id, "paths.round_index is not the paper's 96-row index"
    first_latent = load_first_latent(cfg.paths.latent_cache_root, media_id, row.start_frame)
    assert first_frame_fingerprint(first_latent) == recorded["first_frame"]


@pytest.fixture(scope="module")
def round_outputs(reference_config, tmp_path_factory):
    """Run each requested round once (``tools/run_session.py``); ``{media_id: latents.npy}``."""
    path, _ = reference_config
    if torch.cuda.device_count() < 3:
        pytest.skip("a round needs three GPUs (one per client, lockstep)")
    out = Path(os.environ.get("WORLDCAST_REFERENCE_OUT") or tmp_path_factory.mktemp("reference"))
    gpus = os.environ.get("WORLDCAST_REFERENCE_GPUS", "0,1,2")
    wanted = os.environ.get("WORLDCAST_REFERENCE_ROUNDS")
    found = {}
    for name, case in TABLE["cases"].items():
        if wanted and name not in wanted.split(","):
            continue
        session = out / name
        cmd = [sys.executable, str(REPO / "tools" / "run_session.py"), "--config", path]
        cmd += ["--set", f"run.latent_frames={case['latent_frames']}"]
        cmd += ["--set", f"run.seed={TABLE['seed']}"]
        cmd += ["--index-row", *map(str, case["rows"]), "--gpus", gpus, "--out-dir", str(session)]
        subprocess.run(cmd, check=True)
        for media_id in case["fingerprints"]:
            found[media_id] = next(session.glob(f"*/{media_id}/latents.npy"))
    return found


@pytest.mark.parametrize("media_id", sorted(CLIENTS))
def test_latents(round_outputs, media_id):
    if media_id not in round_outputs:
        pytest.skip("round not requested (WORLDCAST_REFERENCE_ROUNDS)")
    _, latents, recorded = CLIENTS[media_id]
    first, whole = verify_latents(round_outputs[media_id], [ReferenceRun(latents, recorded)])
    assert first.ok, "latents 0-24 (generator, sampler) differ"
    assert whole.reference and whole.ok, "all latents (the three-client round) differ"
