#!/usr/bin/env python3
"""Run one WorldCast client: a row of the round index, on one GPU, in lockstep with its round.

The clients of a round exchange their messages through the shared world state in
``paths.world_state_dir`` (``tools/run_session.py`` starts them all). Writes
``<out_dir>/latents.npy`` ([N, 48, 24, 42] float32) and ``<out_dir>/client.json``. Index rows
count from 0.

Example, the first of the five clients of an example case (start the others, rows 1-4, on the same
world state)::

    python tools/run_client.py --config weights/paths.yaml \\
        --config examples/data/dust2_r09/config.yaml --index-row 0 \\
        --world-state-dir runs/dust2_r09/world_state --out-dir runs/dust2_r09/p05

Without flash-attention pass ``--set model.attention=sdpa`` (runs, but is not bit-equal to the
paper's kernel).
"""

import argparse
import logging
import sys
import time
from collections.abc import Sequence

from worldcast.config.inference import load_config
from worldcast.config.loader import add_config_args, parse_overrides, usage_errors
from worldcast.engine.inference.client import run_client


def main(argv: Sequence[str] | None = None) -> int:
    """Run the client of ``--index-row``; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--index-row", type=int, help="the client's row of the round index")
    parser.add_argument("--world-state-dir", help="the shared world state of the round's clients")
    parser.add_argument("--out-dir", help="where latents.npy and client.json are written")
    parser.add_argument("--device", help="torch device, e.g. cuda or cuda:1")
    args = parser.parse_args(argv)

    with usage_errors(parser):
        overrides = parse_overrides(args.overrides)
        for key, value in (
            ("run.index_row", args.index_row),
            ("paths.world_state_dir", args.world_state_dir),
            ("paths.out_dir", args.out_dir),
            ("run.device", args.device),
        ):
            if value is not None:
                overrides[key] = value
        cfg = load_config(args.config, overrides)
        cfg.paths.require("world_state_dir", "out_dir")
        if cfg.run.index_row is None:
            raise ValueError("run.index_row is not set: pass --index-row")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    result = run_client(cfg)
    reads = sum(b["entry"] is not None for b in result["blocks"])
    print(
        f"{result['media_id']}: {result['latent_frames']} latents -> {cfg.paths.out_dir} "
        f"({len(result['blocks'])} blocks read the scene state, {reads} retrieved a memory entry, "
        f"lockstep wait {result['wait']['seconds_total']:.0f} s, total {time.time() - t0:.0f} s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
