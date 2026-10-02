#!/usr/bin/env python3
"""Check the example cases on the CPU, without weights; after a GPU run, compare its latents with the reference run's.

Without ``--run-dir``: loads every client's inputs the way a client does (round and media index, the tick tables of
all ten players, visibility and observer-signal labels, the first latent), checks that the round is one lock-step
group whose clients all cover the case length, and that each first latent is the reference run's. With
``--run-dir``: compares each client's ``latents.npy`` with the fingerprints recorded in the reference runs of the cases
(``examples/manifest.json``; bit-equal only on the paper's stack, docs/inference.md).

Examples::

    python tools/check_examples.py                                          # every downloaded case
    python tools/check_examples.py dust2_r09
    python tools/check_examples.py dust2_r09 --run-dir runs/examples/dust2_r09
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run from a checkout without installing the package
    sys.path.insert(0, str(REPO))

from worldcast.config.loader import DEFAULT_CONFIG, load_cli_config  # noqa: E402

EXAMPLES = REPO / "examples"
PLAIN_PREFIX = 25  # latents 0-24: the plain prefix (no peers yet)


def fingerprint(array) -> str:
    """sha256 of the contiguous float32 bytes, first 32 hex digits (the fingerprint recorded in the reference runs)."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()[:32]


def check_inputs(case: str, recorded: dict) -> None:
    import torch
    from run_session import groups_of

    from worldcast.data.index import read_round_index
    from worldcast.engine.inference.client import Client

    cfg = load_cli_config([str(DEFAULT_CONFIG), str(EXAMPLES / "data" / case / "config.yaml")], {})
    rows = read_round_index(cfg.paths.round_index)
    if groups_of(cfg.paths.round_index) != [list(range(len(rows)))]:
        raise SystemExit(f"{case}: the round index is not one lock-step group")
    client = Client(cfg)
    for row in rows:
        window = client.load_window(row)
        n = window.spec.latent_frames
        if n != cfg.run.latents:
            raise SystemExit(
                f"{case}: {row.media_id} covers {n} latents, the case runs {cfg.run.latents}"
            )
        sink = fingerprint(window.first_latent[None].to(torch.bfloat16).float().numpy())
        if sink != recorded[row.media_id]["initial"]:
            raise SystemExit(
                f"{case}: the first latent of {row.media_id} is not the reference run's"
            )
        visible = window.item.observer_visibility.numpy()
        others = np.delete(visible, window.observer.player_slot, axis=0)
        print(
            f"  {row.media_id}: {n} latents, {len(window.round_slots)} players recorded, "
            f"another player in view in {others.any(axis=0).mean():.0%} of the frames"
        )


def client_latents(run_dir: Path) -> dict:
    """``{media_id: latents.npy}`` of every client under ``run_dir``, at any depth: the media id is the one in the
    ``client.json`` next to the latents (written by every client), else the directory name. Covers both
    ``examples/run.sh`` (``<round>/<media_id>/``) and clients started by hand with any ``--out-dir``.
    """
    found = {}
    for path in sorted(run_dir.rglob("latents.npy")):
        meta = path.with_name("client.json")
        media_id = json.loads(meta.read_text())["media_id"] if meta.is_file() else path.parent.name
        if media_id in found:
            raise SystemExit(
                f"two runs of {media_id} under {run_dir}: {found[media_id]} and {path}"
            )
        found[media_id] = path
    return found


def check_run(case: str, run_dir: Path, recorded: dict) -> bool:
    ok = True
    found = client_latents(run_dir)
    for media_id, want in recorded.items():
        if media_id not in found:
            print(f"  {media_id}: no latents.npy under {run_dir}")
            ok = False
            continue
        lat = np.load(found[media_id])
        got = {"prefix": fingerprint(lat[:PLAIN_PREFIX]), "final": fingerprint(lat)}
        parts = []
        for key in ("prefix", "final"):
            if want[key] is None:
                parts.append(f"{key} not recorded")
            else:
                same = got[key] == want[key]
                ok &= same
                parts.append(f"{key} {'equal' if same else 'differs'}")
        print(f"  {media_id}: {lat.shape[0]} latents, " + ", ".join(parts))
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "cases", nargs="*", help="case names (default: every case whose data is downloaded)"
    )
    parser.add_argument(
        "--run-dir", help="output directory of examples/run.sh (one case): compare its latents"
    )
    args = parser.parse_args(argv)

    cases = json.loads((EXAMPLES / "manifest.json").read_text())["cases"]
    unknown = [c for c in args.cases if c not in cases]
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; known: {', '.join(cases)}")
    if args.run_dir:
        if len(args.cases) != 1:
            raise SystemExit("--run-dir compares one case: name it")
        case = args.cases[0]
        print(f"{case}: {args.run_dir} against the reference run's fingerprints")
        return 0 if check_run(case, Path(args.run_dir), cases[case]["fingerprints"]) else 1

    todo = args.cases or [c for c in cases if (EXAMPLES / "data" / c / "first_latents").is_dir()]
    if not todo:
        raise SystemExit("no example data found: python tools/download_examples.py")
    for case in todo:
        missing = [
            name
            for name in cases[case]["files"]
            if "/data/" in name and not (REPO / name).is_file()
        ]
        if missing:
            raise SystemExit(
                f"{case}: {len(missing)} data files missing: python tools/download_examples.py"
                f" {case}"
            )
        print(
            f"{case}: {cases[case]['map']}, match {cases[case]['match_id']} round"
            f" {cases[case]['round']}"
        )
        check_inputs(case, cases[case]["fingerprints"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
