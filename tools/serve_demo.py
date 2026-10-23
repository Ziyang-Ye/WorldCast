#!/usr/bin/env python3
"""Serve WorldCast Live, the networked multiplayer demo (docs/demo.md).

Roles: ``all`` (a coordinator and one worker process per ``--gpus`` card, on this machine),
``coordinator`` (one, CPU: lobby, rooms, relay; serves the web page), ``worker`` (one per GPU: one
player's client), ``local`` (a coordinator and ``--workers`` workers in this process, for the mock
client on a laptop). ``--config`` files merge over the defaults
(``worldcast/demo/config.py``), ``--set`` overrides one key, ``--print-config`` prints the config
the role runs with.

Examples::

    # one machine, a worker per GPU: open http://localhost:8100
    python tools/serve_demo.py --role all --gpus 0,1,2 --config configs/demo/worldcast.yaml \\
        --set library=demo-library/library.json

    # on a laptop, no GPU: the mock client, two players, open http://localhost:8100
    python tools/serve_demo.py --role local --workers 2

    # across machines: the coordinator on a CPU host, one worker per GPU
    python tools/serve_demo.py --role coordinator --set library=/data/demo-library/library.json
    CUDA_VISIBLE_DEVICES=0 python tools/serve_demo.py --role worker \\
        --config configs/demo/worldcast.yaml --set library=/data/demo-library/library.json \\
        --set worker.coordinator_url=ws://lobby:8100 --set worker.port=8101 \\
        --set worker.advertise_url=ws://gpu-1:8101

    # the same worker with the original Wan2.2 VAE decoding on a second card: two GPUs for it
    # (cuda:0 generates, cuda:1 decodes), the display at a rate the decoder sustains
    CUDA_VISIBLE_DEVICES=0,1 python tools/serve_demo.py --role worker \\
        --config configs/demo/worldcast.yaml --config configs/demo/worldcast_wan_vae.yaml \\
        --set play.fps=11 --set library=/data/demo-library/library.json \\
        --set worker.coordinator_url=ws://lobby:8100 --set worker.port=8101 \\
        --set worker.advertise_url=ws://gpu-1:8101
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from worldcast.config.loader import add_config_args, config_to_dict, parse_overrides, usage_errors
from worldcast.demo.config import load_config


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the role until interrupted, or print its config; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--role", choices=("all", "coordinator", "worker", "local"), required=True)
    add_config_args(parser)
    parser.add_argument("--workers", type=int, default=2, help="--role local: workers to start")
    parser.add_argument(
        "--gpus", default="0", help="--role all: the cards, one worker each: 0,1,2 (0+1: two cards)"
    )
    parser.add_argument("--print-config", action="store_true", help="print the config and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s {args.role:<11} %(message)s", datefmt="%H:%M:%S"
    )
    with usage_errors(parser):
        cfg = load_config(args.config, parse_overrides(args.overrides))
        if args.role == "local":
            from worldcast.demo import local

            cfg = local.local_config(cfg, args.workers)
        if args.role == "all":
            from worldcast.demo import launch

            cfg = launch.all_config(cfg)
            gpus = launch.parse_gpus(args.gpus)
    if args.print_config:
        print(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
    elif args.role == "coordinator":
        from worldcast.demo.coordinator import run

        run(cfg)
    elif args.role == "worker":
        from worldcast.demo.worker import run

        run(cfg)
    elif args.role == "all":
        launch.run(Path(__file__).resolve(), cfg, gpus, args.config, args.overrides)
    else:
        local.run(cfg, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
