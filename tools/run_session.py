#!/usr/bin/env python3
"""Run the clients of one round (or of every round) together: one process and one GPU per client, one shared pool.

Each client is ``tools/run_client.py`` in its own process (never threads: the scene state's numpy reprojections
are not bit-reproducible when several clients share a process; docs/inference.md). The clients advance in
lock-step through a fresh ``LocalDirPool`` directory. A client that fails marks itself failed in the pool and its
peers stop with an error; the session then exits non-zero.

Examples::

    # the round of index row 59 (rows count from 0; its group_media: rows 57, 58, 59), on GPUs 0-2
    python tools/run_session.py --config configs/infer/worldcast_4step.yaml --config weights/paths.yaml \\
        --config data/paths.yaml --group-of 59 --gpus 0,1,2 --out-dir runs/dust2-r09

    # every round of the index, one after the other (the paper's whole-round setting: 32 rounds x 3 clients,
    # each round as long as the paper ran it)
    python tools/run_session.py ... --all-groups --round-length --gpus 0,1,2 --out-dir runs/wholeround

Outputs: ``<out-dir>/<round>/pool/`` (the pool), ``<out-dir>/<round>/<media_id>/{latents.npy,client.json,client.log}``,
where ``<round>`` is ``<match_id>-<map_name>-r<round>-s<start_frame>``. ``--round-length`` reads ``round_seconds`` from
the round index (docs/data.md).
"""

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run from a checkout without installing the package
    sys.path.insert(0, str(REPO))

from worldcast.config.loader import add_config_args, load_cli_config, parse_overrides  # noqa: E402

#: default environment of every client process (the paper runner's values; none changes a number we know of).
#: A variable already set in the caller's environment wins.
CLIENT_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "PYTHONUNBUFFERED": "1",
}


def groups_of(index_path: str) -> list[list[int]]:
    """Row numbers of every round group in the index (rows sharing ``group_media`` and ``start_frame``), in order."""
    from worldcast.data.index import read_round_index

    rows = read_round_index(index_path)
    seen: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        key = (tuple(sorted(row.group_media or (row.media_id,))), int(row.start_frame))
        seen.setdefault(key, []).append(i)
    for (members, _), idx in seen.items():
        if sorted(rows[i].media_id for i in idx) != list(members):
            raise SystemExit(f"index rows {idx} do not cover their group {list(members)}")
    return list(seen.values())


def round_latents(index_path: str, row: int, max_seconds: int) -> int:
    """The paper runner's length rule: ``N = 1 + 4 * floor10(min(round_seconds, max_seconds))`` (``round_seconds``
    of the raw index row; d59: 110 s -> 441, d75: 74 s -> 281, d90: 67 s -> 241). The entry noise is drawn for this
    requested length, so it matters for bit-exact reproduction even where the coverage clip gives the same N.
    """
    import json

    lines = [line for line in Path(index_path).read_text().splitlines() if line.strip()]
    seconds = int(json.loads(lines[row])["round_seconds"])
    return 1 + 4 * (min(seconds, int(max_seconds)) // 10 * 10)


def run_group(
    rows: Sequence[int],
    *,
    configs,
    sets: Sequence[str],
    gpus: Sequence[str],
    out_dir: Path,
    attention: str,
    device: str = "cuda",
) -> int:
    """Start one client per row on its GPU; wait for all; return the number of failed clients."""
    from worldcast.data.index import load_round_index_row

    cfg = load_cli_config(configs, parse_overrides(sets))
    if len(gpus) < len(rows):
        raise SystemExit(
            f"{len(rows)} clients need {len(rows)} GPUs (lock-step runs them at once), got"
            f" {list(gpus)}"
        )
    pool = out_dir / "pool"
    if pool.exists() and any(pool.iterdir()):
        raise SystemExit(f"{pool} is not empty: every session needs a fresh pool directory")
    pool.mkdir(parents=True, exist_ok=True)
    procs = []
    for row, gpu in zip(rows, gpus):
        media_id = load_round_index_row(cfg.paths.round_index, row).media_id
        client_dir = out_dir / media_id
        client_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(REPO / "tools" / "run_client.py")]
        for c in configs or []:
            cmd += ["--config", str(c)]
        for item in sets:
            cmd += ["--set", item]
        cmd += [
            "--index-row",
            str(row),
            "--live-pool-dir",
            str(pool),
            "--out-dir",
            str(client_dir),
            "--device",
            device,
            "--attention",
            attention,
        ]
        env = {**CLIENT_ENV, **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        log = open(client_dir / "client.log", "w")
        procs.append(
            (media_id, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        )
        print(f"  started {media_id} (row {row}) on GPU {gpu}", flush=True)
    failed = 0
    for media_id, proc, log in procs:
        code = proc.wait()
        log.close()
        print(
            f"  {media_id}: {'ok' if code == 0 else f'FAILED (exit {code}), see its client.log'}",
            flush=True,
        )
        failed += int(code != 0)
    return failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--rows", type=int, nargs="+", help="index rows to run together (one round)")
    which.add_argument(
        "--group-of", type=int, metavar="ROW", help="the round of this index row (its group_media)"
    )
    which.add_argument(
        "--all-groups", action="store_true", help="every round of the index, one after the other"
    )
    parser.add_argument(
        "--gpus", default=None, help="comma-separated GPU ids, one per client (default 0,1,...)"
    )
    parser.add_argument(
        "--out-dir", required=True, help="session output directory (one subdirectory per round)"
    )
    parser.add_argument(
        "--attention",
        choices=("flash", "sdpa"),
        default="flash",
        help=(
            "attention kernel of every client: flash (the paper's; needs flash-attn) or sdpa "
            "(not bit-equal)"
        ),
    )
    parser.add_argument(
        "--device", default="cuda", help="torch device of every client (cpu only for tests)"
    )
    parser.add_argument(
        "--round-length",
        action="store_true",
        help="set run.latents per round from the index's round_seconds, as the paper's runner did",
    )
    parser.add_argument(
        "--max-seconds", type=int, default=120, help="cap of --round-length (paper: 120)"
    )
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.overrides)
    cfg = load_cli_config(args.config, overrides)
    cfg.paths.require("round_index")
    groups = groups_of(cfg.paths.round_index)
    if args.rows:
        todo = [list(args.rows)]
    elif args.group_of is not None:
        todo = [g for g in groups if args.group_of in g]
    else:
        todo = groups
    out = Path(args.out_dir)
    failed = 0
    for rows in todo:
        gpus = args.gpus.split(",") if args.gpus else [str(i) for i in range(len(rows))]
        from worldcast.data.index import load_round_index_row

        first = load_round_index_row(cfg.paths.round_index, rows[0])
        name = f"{first.match_id}-{first.map_name}-r{first.round:02d}-s{first.start_frame:06d}"
        sets = list(args.overrides)
        if args.round_length:
            lengths = {round_latents(cfg.paths.round_index, r, args.max_seconds) for r in rows}
            if len(lengths) != 1:
                raise SystemExit(f"rows {rows} disagree on round_seconds")
            sets.append(f"run.latents={lengths.pop()}")
        print(
            f"round {name}: rows {rows}" + (f", {sets[-1]}" if args.round_length else ""),
            flush=True,
        )
        t0 = time.time()
        failed += run_group(
            rows,
            configs=args.config,
            sets=sets,
            gpus=gpus,
            out_dir=out / name,
            attention=args.attention,
            device=args.device,
        )
        print(f"round {name}: {time.time() - t0:.0f} s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
