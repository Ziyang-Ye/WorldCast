#!/usr/bin/env python3
"""Compare a client's latents with the fingerprints of the reference runs.

The reference runs are those of the example cases (``examples/manifest.json``) and the paper's
Table-3 runs that recorded fingerprints (``examples/table3_fingerprints.json``), looked up by media
id (``worldcast.engine.inference.reference``). Bit equality needs the stack of the reference runs
(docs/inference.md, "Numerics").

``client``: run one client through latent 24 (generator and sampler; no scene state, no other
client) and compare its first frame, its latents 0-24 and, where the reference run recorded it (the
Table-3 runs), its entry noise. One GPU. Also writes ``latents_0_24.npy``, and the
fingerprint of every block for bisecting::

    python tools/verify_reference.py client --config weights/paths.yaml \\
        --config examples/data/mirage_r16/config.yaml --index-row 0 --out-dir runs/verify/client

``check``: compare the ``latents.npy`` of every client with a reference run under finished sessions
(``tools/run_session.py``, ``examples/run.sh``)::

    python tools/verify_reference.py check runs/examples/mirage_r16

Both print one line per fingerprint and exit 1 on any difference. ``client`` writes
``<out-dir>/report.json``.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from worldcast.config.inference import load_config
from worldcast.config.loader import add_config_args, parse_overrides, usage_errors
from worldcast.data import load_round_index_row
from worldcast.engine.inference.reference import read_reference_runs, verify_client, verify_latents
from worldcast.hub import EXAMPLES_MANIFEST

ROOT = Path(__file__).resolve().parents[1]
#: The fingerprints of the reference runs: the example cases', the paper's Table-3 runs'.
TABLES = (ROOT / EXAMPLES_MANIFEST, ROOT / "examples" / "table3_fingerprints.json")


def verify_one_client(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """The ``client`` mode."""
    with usage_errors(parser):
        cfg = load_config(args.config, parse_overrides(args.overrides))
        cfg.paths.require("round_index")
        row = load_round_index_row(cfg.paths.round_index, args.index_row)
    runs = read_reference_runs(*TABLES).get(row.media_id)
    if not runs:
        parser.error(f"{row.media_id} (row {args.index_row}) has no reference run")
    print(f"{row.media_id} (row {args.index_row})", flush=True)
    result = verify_client(cfg, row, runs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "latents_0_24.npy", result.latents)
    print(f"  environment: {json.dumps(result.environment)}", flush=True)
    print(f"  latents 1-{len(result.latents) - 1}: {result.seconds:.1f} s", flush=True)
    for check in result.checks:
        print(f"  {check}", flush=True)
    for name, value in result.blocks.items():
        print(f"         latents {name:7s} {value}", flush=True)
    match = all(check.ok for check in result.checks)
    report = dict(
        media_id=row.media_id,
        index_row=args.index_row,
        config=list(args.config),
        environment=result.environment,
        fingerprints={check.name: check.value for check in result.checks},
        reference={check.name: check.reference for check in result.checks},
        blocks=result.blocks,
        seconds=round(result.seconds, 2),
        match=match,
    )
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return 0 if match else 1


def check_sessions(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """The ``check`` mode."""
    runs = read_reference_runs(*TABLES)
    clients = []
    for path in sorted(p for root in args.dirs for p in Path(root).rglob("latents.npy")):
        meta = path.with_name("client.json")
        media_id = json.loads(meta.read_text())["media_id"] if meta.is_file() else None
        if media_id in runs:
            checks = verify_latents(path, runs[media_id])
            print(f"{media_id}: {path}", flush=True)
            for check in checks:
                print(f"  {check}", flush=True)
            clients.append(checks)
    if not clients:
        parser.error(f"no latents.npy of a client with a reference run under {args.dirs}")
    matching = sum(all(check.ok for check in checks) for checks in clients)
    partial = sum(any(check.reference is None for check in checks) for checks in clients)
    note = f"; {partial} on latents 0-24 only (their reference run had another length)"
    print(
        f"{matching}/{len(clients)} clients match their reference run bit for bit"
        + (note if partial else ""),
        flush=True,
    )
    return 0 if matching == len(clients) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``client`` or the ``check`` mode; returns 1 on any difference, else 0."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    client = sub.add_parser("client", help="one client through latent 24")
    add_config_args(client)
    client.add_argument(
        "--index-row", type=int, required=True, help="the client's row of the round index"
    )
    client.add_argument(
        "--out-dir", required=True, help="receives latents_0_24.npy and report.json"
    )
    check = sub.add_parser("check", help="the latents.npy of finished sessions")
    check.add_argument("dirs", nargs="+", help="session output directories, searched recursively")
    args = parser.parse_args(argv)
    mode = verify_one_client if args.mode == "client" else check_sessions
    return mode(args, parser)


if __name__ == "__main__":
    sys.exit(main())
