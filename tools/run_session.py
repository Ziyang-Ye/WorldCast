#!/usr/bin/env python3
"""Run the clients of a round (or of every round) together: a process per client, each on a GPU.

Each client is ``tools/run_client.py`` in its own process (``worldcast.engine.inference.session``).
Without ``--gpus`` the clients take the visible GPUs in turn, so a GPU runs two clients (or more)
when there are fewer GPUs than clients, with the same latents. ``--gpus`` names a visible GPU per
client. Before a client starts, a GPU that is not visible, or without the memory for its clients
(about 15 GiB each, by ``nvidia-smi``), ends the tool. A client that fails marks itself failed in
the shared world state, the other clients stop with an error and the session exits non-zero. The
clients get the environment of the reference runs (``CLIENT_ENV``) where the caller's sets no
other value.

Examples::

    # an example case: its five clients on GPUs 0-4
    python tools/run_session.py --config weights/paths.yaml \\
        --config examples/data/dust2_r09/config.yaml --all-rounds --gpus 0,1,2,3,4 \\
        --out-dir runs/examples/dust2_r09

    # every round of the index, each as long as it was recorded (docs/reproduce.md)
    python tools/run_session.py --config weights/paths.yaml --config data/paths.yaml \\
        --all-rounds --round-length --gpus 0,1,2 --out-dir runs/table3

Outputs ``<out-dir>/<round>/world_state/`` and
``<out-dir>/<round>/<media_id>/{latents.npy,client.json,client.log}``, where ``<round>`` is
``<match_id>-<map_name>-r<round>-s<start_frame>``.
"""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from worldcast.config.inference import load_config
from worldcast.config.loader import add_config_args, parse_overrides, usage_errors
from worldcast.data import read_round_index
from worldcast.engine.inference.session import (
    CLIENT_MEMORY_GIB,
    MAX_ROUND_SECONDS,
    check_gpu_memory,
    gpu_memory,
    place_clients,
    round_latents,
    round_name,
    rounds_of,
    run_round,
    visible_gpus,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chosen rounds one after the other; returns 1 if a client failed, else 0."""
    parser = argparse.ArgumentParser(  # examples/run.sh reads --gpus and --device as spelled
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    add_config_args(parser)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--index-row", type=int, nargs="+", help="index rows to run together")
    which.add_argument("--round-of", type=int, metavar="INDEX_ROW", help="the round of this row")
    which.add_argument("--all-rounds", action="store_true", help="every round of the index")
    parser.add_argument(
        "--gpus", help="comma-separated GPU ids, one per client (default: the visible GPUs in turn)"
    )
    parser.add_argument("--out-dir", required=True, help="one subdirectory per round")
    parser.add_argument("--device", help="torch device of every client (default: run.device)")
    parser.add_argument(
        "--round-length",
        action="store_true",
        help="set run.latent_frames per round from the index's round_seconds, as Table 3 ran",
    )
    parser.add_argument(
        "--max-seconds", type=int, default=MAX_ROUND_SECONDS, help="cap of --round-length"
    )
    args = parser.parse_args(argv)

    with usage_errors(parser):
        cfg = load_config(args.config, parse_overrides(args.overrides))
        cfg.paths.require("round_index")
        rows = read_round_index(cfg.paths.round_index)
        rounds = rounds_of(rows)
    if args.index_row:
        rounds = [list(args.index_row)]
    elif args.round_of is not None:
        rounds = [index for index in rounds if args.round_of in index]
    if not rounds or any(not 0 <= i < len(rows) for index in rounds for i in index):
        parser.error(f"no such row in the {len(rows)}-row index {cfg.paths.round_index}")
    # the GPU of each client of each round, checked before a client loads its 10 GB
    on_gpus, visible = (args.device or cfg.run.device).startswith("cuda"), visible_gpus()
    if on_gpus and not visible:
        parser.error("no GPU is visible (--device cpu runs the clients on the CPU)")
    listed = [gpu.strip() for gpu in args.gpus.split(",")] if args.gpus else []
    unknown = ",".join(gpu for gpu in listed if gpu not in visible)
    if on_gpus and unknown:
        parser.error(f"--gpus: no GPU {unknown} among the visible GPUs {','.join(visible)}")
    # on the CPU without a visible GPU, none: as examples/run.sh decodes them
    placed = [listed or place_clients(len(index), visible or [""]) for index in rounds]
    for index, gpus in zip(rounds, placed):
        if len(gpus) < len(index):
            parser.error(
                f"--gpus: {len(gpus)} GPU ids for the {len(index)} clients of rows {index}: an id"
                " per client (an id can repeat)"
            )
    if on_gpus:
        memory = gpu_memory()
        with usage_errors(parser):
            for index, gpus in zip(rounds, placed):
                check_gpu_memory(gpus[: len(index)], memory)

    command = [sys.executable, str(Path(__file__).with_name("run_client.py"))]
    command += [a for path in args.config for a in ("--config", str(path))]
    command += [a for item in args.overrides for a in ("--set", item)]
    command += ["--device", args.device] if args.device else []
    failures = 0
    for index, gpus in zip(rounds, placed):
        name, length = round_name(rows[index[0]]), []
        if args.round_length:
            seconds = [rows[i].round_seconds for i in index]
            if None in seconds:
                parser.error(f"--round-length: rows {index} need round_seconds")
            lengths = {round_latents(s, args.max_seconds) for s in seconds}
            if len(lengths) != 1:
                parser.error(f"rows {index} disagree on round_seconds")
            length = ["--set", f"run.latent_frames={lengths.pop()}"]
        print(f"round {name}: rows {index} {' '.join(length[1:])}", flush=True)
        if on_gpus and not listed:
            print(
                f"  GPUs {','.join(gpus)}: {len(index)} clients on {len(visible)} visible"
                f" GPU{'s' if len(visible) > 1 else ''} in turn, about {CLIENT_MEMORY_GIB} GiB each"
                " (--gpus places them)",
                flush=True,
            )
        started = time.time()
        with usage_errors(parser):  # a world state that is not fresh
            failed = run_round(
                command + length, index, rows, gpus=gpus, out_dir=Path(args.out_dir) / name
            )
        for media_id in failed:
            print(f"  {media_id} FAILED: see its client.log", flush=True)
        print(f"round {name}: {time.time() - started:.0f} s", flush=True)
        failures += len(failed)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
