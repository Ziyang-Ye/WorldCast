"""The whole demo on one machine with a GPU worker per card: a coordinator in this process and a
worker process for each GPU (``tools/serve_demo.py --role all --gpus 0,1,2``).

A WorldCast client needs a process of its own, so each worker is ``--role worker`` pinned to its
card with ``CUDA_VISIBLE_DEVICES``; browsers reach the workers through the coordinator's port, so
one address is all a player needs.
"""

import logging
import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config import DemoConfig
from .coordinator import run as run_coordinator
from .local import LOOPBACK

__all__ = ["all_config", "parse_gpus", "run", "worker_commands"]

log = logging.getLogger(__name__)
#: Seconds a worker gets to shut down before it is killed.
STOP_S = 10.0


def all_config(config: DemoConfig) -> DemoConfig:
    """``config`` as ``--role all`` serves it: browsers reach the workers through the
    coordinator's port (``coordinator.frame_route: proxy``)."""
    return replace(config, coordinator=replace(config.coordinator, frame_route="proxy"))


def parse_gpus(text: str) -> list[str]:
    """The cards of ``--gpus``: comma-separated ids; each id is one card, or several joined by
    ``+`` for a worker that spans cards (``0,1+2`` is two workers, the second on cards 1 and 2).

    Raises:
        ValueError: no card, an empty id or the same card twice.
    """
    gpus = [gpu.strip() for gpu in text.split(",")]
    cards = [card for gpu in gpus for card in gpu.split("+")]
    if not text.strip() or any(not card.strip() for card in cards):
        raise ValueError(f"--gpus takes card ids like 0,1,2 (or 0+1 for two cards), not {text!r}")
    if len(set(cards)) != len(cards):
        raise ValueError(f"--gpus names a card twice: {text!r}")
    return [gpu.replace("+", ",") for gpu in gpus]


def worker_commands(
    script: Path,
    config: DemoConfig,
    gpus: Sequence[str],
    configs: Sequence[str],
    overrides: Sequence[str],
) -> list[tuple[list[str], dict[str, str]]]:
    """The command and environment of each worker process, one per entry of ``gpus``.

    Worker ``i`` listens on the loopback port ``worker.port + i``, reaches the coordinator on the
    loopback and is reached by the coordinator there.
    """
    commands = []
    for i, gpu in enumerate(gpus):
        port = config.worker.port + i
        command = [sys.executable, str(script), "--role", "worker"]
        for path in configs:
            command += ["--config", str(path)]
        for override in [
            *overrides,
            f"worker.port={port}",
            f"worker.host={LOOPBACK}",
            f"worker.coordinator_url=ws://{LOOPBACK}:{config.coordinator.port}",
            f"worker.advertise_url=ws://{LOOPBACK}:{port}",
            f"worker.worker_id=gpu-{gpu.replace(',', '-')}",
        ]:
            command += ["--set", override]
        commands.append((command, {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}))
    return commands


def run(
    script: Path,
    config: DemoConfig,
    gpus: Sequence[str],
    configs: Sequence[str],
    overrides: Sequence[str],
) -> None:
    """Serve the coordinator here and a worker process per entry of ``gpus`` until interrupted;
    the workers end with it. ``config`` is as :func:`all_config` returns it."""
    processes = [
        subprocess.Popen(command, env=env)
        for command, env in worker_commands(script, config, gpus, configs, overrides)
    ]
    log.info(
        "WorldCast Live: http://localhost:%d (%d workers on GPU %s)",
        config.coordinator.port,
        len(processes),
        " ".join(gpus),
    )
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        run_coordinator(config)
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(STOP_S)
            except subprocess.TimeoutExpired:
                process.kill()


def _interrupt(*_: object) -> None:
    """A SIGTERM stops the coordinator as Ctrl-C does, and with it the workers."""
    raise KeyboardInterrupt
