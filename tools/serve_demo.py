#!/usr/bin/env python3
"""Serve WorldCast Live, the networked multiplayer demo (docs/demo.md).

Roles: ``coordinator`` (one, CPU: lobby, rooms, relay; serves the web page), ``worker`` (one per GPU: one player's
engine), ``local`` (a coordinator and ``--workers`` workers on this machine, for trying it out).

Examples::

    # on a laptop, no GPU: the mock engine, two players, open http://localhost:8100
    python tools/serve_demo.py --role local --workers 2

    # across machines: the coordinator on a CPU host, one worker per GPU
    python tools/serve_demo.py --role coordinator --set library=/data/demo-library/library.json
    CUDA_VISIBLE_DEVICES=0 python tools/serve_demo.py --role worker --set engine.kind=worldcast \\
        --set worker.coordinator_url=ws://lobby-host:8100 --set worker.port=8101 \\
        --set worker.advertise_url=ws://gpu-host-1:8101 --set library=/data/demo-library/library.json
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run from a checkout without installing the package
    sys.path.insert(0, str(REPO))


def run_local(args) -> int:
    """A coordinator and ``args.workers`` workers as child processes; Ctrl-C stops them all."""
    from demo.config import load_config

    cfg = load_config(args.config, args.overrides)
    base = [sys.executable, str(Path(__file__).resolve())]
    for path in args.config or []:
        base += ["--config", path]
    for item in args.overrides:
        base += ["--set", item]
    coordinator = f"ws://127.0.0.1:{cfg.coordinator.port}"
    commands = [base + ["--role", "coordinator"]]
    for i in range(args.workers):
        port = cfg.worker.port + i
        commands.append(
            base
            + [
                "--role",
                "worker",
                "--set",
                f"worker.port={port}",
                "--set",
                f"worker.coordinator_url={coordinator}",
                "--set",
                f"worker.worker_id=local-{i + 1}",
                "--set",
                f"worker.advertise_url=ws://127.0.0.1:{port}",
            ]
        )
    procs = []
    for command in commands:
        procs.append(subprocess.Popen(command, env={**os.environ, "PYTHONUNBUFFERED": "1"}))
        time.sleep(0.3)
    print(
        f"WorldCast Live: http://localhost:{cfg.coordinator.port}  ({args.workers} worker(s),"
        f" engine {cfg.engine.kind}; Ctrl-C stops)",
        flush=True,
    )
    stopped = False
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        stopped = True
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
    return 0 if stopped else max(abs(p.returncode or 0) for p in procs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--role", choices=("coordinator", "worker", "local"), required=True)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="YAML",
        help="config file; repeat to merge in order (default: configs/demo/demo.yaml)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="override one dotted key, e.g. --set worker.port=8102",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="--role local: how many workers to start"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s {args.role:<11} %(message)s", datefmt="%H:%M:%S"
    )
    if args.role == "local":
        return run_local(args)
    from demo.config import load_config

    cfg = load_config(args.config, args.overrides)
    if args.role == "coordinator":
        from demo.coordinator import run
    else:
        from demo.worker import run
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
