#!/usr/bin/env python3
"""Download the release weights and the parts of the Wan2.2-TI2V-5B snapshot a client needs.

From the WorldCast weights repository (config ``weights.hf_repo_id``, docs/checkpoints.md): the
4-step generator (bf16), the depth head and its read-out, and the fixed prompt's umT5 embedding, all
``.safetensors``. From ``Wan-AI/Wan2.2-TI2V-5B`` at the pinned revision: ``config.json`` (backbone
dimensions), ``Wan2.2_VAE.pth`` (decode) and the umT5 tokenizer; with ``--with-t5`` also the 11 GB
umT5-XXL encoder (only needed to re-make the prompt embedding). The Wan2.2 backbone weights are not
downloaded: the WorldCast checkpoint replaces every one of them. From ``madebyollin/taehv`` (MIT)
at a pinned commit: ``taew2_2.pth``, the tiny decoder of the low-latency engine and the web demo,
checked by sha256 (``--skip-taehv`` leaves it out).

``--with-training-checkpoints`` also fetches the stage-3 (block-causal) and stage-2s (bidirectional)
models, fp32, 20 GB each: training starting points, not used by inference.

Writes ``<out-dir>/paths.yaml`` with the ``paths`` entries to pass as a second ``--config``.

Example::

    python tools/download_weights.py --out-dir weights
    python tools/run_client.py --config configs/infer/worldcast_4step.yaml \\
        --config weights/paths.yaml ...
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run from a checkout without installing the package
    sys.path.insert(0, str(REPO))

from worldcast.config.loader import add_config_args, load_cli_config, parse_overrides  # noqa: E402

WAN22_FILES = ["config.json", "Wan2.2_VAE.pth", "google/umt5-xxl/*"]
WAN22_T5 = "models_t5_umt5-xxl-enc-bf16.pth"
#: Training checkpoints in the weights repository (EMA weights, fp32, generator key names).
TRAINING_CHECKPOINTS = (
    "worldcast_stage3_ar_fp32.safetensors",
    "worldcast_stage2s_bidirectional_fp32.safetensors",
)
#: Tiny Wan2.2 VAE decoder, github.com/madebyollin/taehv (MIT, (c) 2025 Ollin Boer Bohan).
TAEHV_FILE = "taew2_2.pth"
TAEHV_REVISION = "011dfc2112197741c540e0bdd5b7b67bcc930771"
TAEHV_URL = f"https://raw.githubusercontent.com/madebyollin/taehv/{TAEHV_REVISION}/{TAEHV_FILE}"
#: 22,884,021 bytes.
TAEHV_SHA256 = "d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_taehv(out: Path) -> Path:
    """``<out>/taew2_2.pth`` at the pinned commit (kept if already there with the right sha256)."""
    path = out / TAEHV_FILE
    if path.is_file() and _sha256(path) == TAEHV_SHA256:
        return path
    partial = path.with_suffix(".part")
    with urllib.request.urlopen(TAEHV_URL, timeout=60) as response, open(partial, "wb") as f:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            f.write(chunk)
    got = _sha256(partial)
    if got != TAEHV_SHA256:
        partial.unlink()
        raise RuntimeError(f"{TAEHV_URL}: sha256 {got}, expected {TAEHV_SHA256}")
    partial.replace(path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--out-dir", required=True, help="download directory")
    parser.add_argument("--with-t5", action="store_true", help="also the umT5-XXL encoder (11 GB)")
    parser.add_argument(
        "--skip-wan22", action="store_true", help="only the WorldCast release files"
    )
    parser.add_argument(
        "--skip-taehv", action="store_true", help="not the tiny decoder (taew2_2.pth)"
    )
    parser.add_argument(
        "--with-training-checkpoints",
        action="store_true",
        help="also the stage-3 and stage-2s training checkpoints (fp32, 20 GB each)",
    )
    args = parser.parse_args(argv)
    cfg = load_cli_config(args.config, parse_overrides(args.overrides))
    w = cfg.weights

    from huggingface_hub import hf_hub_download, snapshot_download

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in (
        ("checkpoint", w.checkpoint_file),
        ("depth_head", w.depth_head_file),
        ("depth_readout", w.depth_readout_file),
        ("prompt_embedding", w.prompt_embedding_file),
    ):
        paths[key] = hf_hub_download(
            w.hf_repo_id, name, revision=w.hf_revision, local_dir=str(out / "worldcast")
        )
        print(f"{key}: {paths[key]}")
    if args.with_training_checkpoints:
        for name in TRAINING_CHECKPOINTS:
            path = hf_hub_download(
                w.hf_repo_id, name, revision=w.hf_revision, local_dir=str(out / "worldcast")
            )
            print(f"training checkpoint: {path}")
    if not args.skip_wan22:
        patterns = WAN22_FILES + ([WAN22_T5] if args.with_t5 else [])
        root = snapshot_download(
            w.wan22_repo_id,
            revision=w.wan22_revision,
            allow_patterns=patterns,
            local_dir=str(out / "Wan2.2-TI2V-5B"),
        )
        paths["wan22_root"] = str(root)
        print(f"wan22_root: {root} ({', '.join(patterns)})")
    if not args.skip_taehv:
        path = download_taehv(out)
        print(f"taehv: {path} (madebyollin/taehv@{TAEHV_REVISION[:12]}, sha256 ok)")
    lines = ["# written by tools/download_weights.py", "paths:" if paths else "paths: {}"]
    lines += [f"  {key}: {value}" for key, value in paths.items()]
    (out / "paths.yaml").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'paths.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
