#!/usr/bin/env python3
"""Run one WorldCast client: one row of the round index, one GPU, in lock-step with the round's other clients.

The clients of a round find each other through ``paths.live_pool_dir`` (use ``tools/run_session.py`` to start all
of them). Writes ``<out_dir>/latents.npy`` ([N, 48, 24, 42] float32) and ``<out_dir>/client.json``.

Example::

    python tools/run_client.py --config configs/infer/worldcast_4step.yaml --config weights/paths.yaml \\
        --config data/paths.yaml --index-row 59 --live-pool-dir runs/dust2-r09/pool --out-dir runs/dust2-r09/p09

Index rows count from 0. Without flash-attention pass ``--attention sdpa`` (runs, but is not bit-equal to the
paper's kernel). Data paths: ``examples/data_paths.yaml`` and ``docs/data.md``; weights:
``tools/download_weights.py``.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run from a checkout without installing the package
    sys.path.insert(0, str(REPO))

from worldcast.config.loader import add_config_args, load_cli_config, parse_overrides  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument(
        "--index-row",
        type=int,
        help="row of the round index this client renders, counted from 0 (run.index_row)",
    )
    parser.add_argument(
        "--live-pool-dir", help="pool directory shared by the round's clients (paths.live_pool_dir)"
    )
    parser.add_argument("--out-dir", help="where latents.npy is written (paths.out_dir)")
    parser.add_argument("--device", help="torch device, e.g. cuda or cuda:1 (run.device)")
    parser.add_argument(
        "--attention",
        choices=("flash", "sdpa"),
        default="flash",
        help="attention kernel: flash (the paper's; needs flash-attn) or sdpa (not bit-equal)",
    )
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.overrides)
    for key, value in (
        ("run.index_row", args.index_row),
        ("paths.live_pool_dir", args.live_pool_dir),
        ("paths.out_dir", args.out_dir),
        ("run.device", args.device),
    ):
        if value is not None:
            overrides[key] = value
    cfg = load_cli_config(args.config, overrides)

    from worldcast.engine.inference.client import Client
    from worldcast.modeling.wan22.attention import sdpa_attention

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    result = Client(cfg, attention=sdpa_attention if args.attention == "sdpa" else None).run()
    reads = [b for b in result.blocks if b["entry"] is not None]
    print(
        f"{result.media_id}: {result.num_latents} latents -> {result.latents_path} "
        f"({len(result.blocks)} reconstituted blocks, {len(reads)} with a memory entry, "
        f"lock-step wait {result.wait.seconds_total:.0f} s, total {time.time() - t0:.0f} s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
