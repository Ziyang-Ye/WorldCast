"""``--role all``: a coordinator and a worker process per card, started by one command."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

from worldcast.demo import launch
from worldcast.demo.config import load_config

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "serve_demo.py"


def test_gpus_are_cards_or_cards_joined_for_one_worker():
    assert launch.parse_gpus("0,1,2") == ["0", "1", "2"]
    assert launch.parse_gpus("0, 1+2") == ["0", "1,2"]
    for bad in ("", "0,,1", "0,0", "0+0"):
        with pytest.raises(ValueError, match="--gpus"):
            launch.parse_gpus(bad)


def test_each_worker_gets_its_card_and_a_loopback_port_of_its_own():
    config = load_config((), {"coordinator.port": 9100, "worker.port": 9200})
    (first, env_first), (second, env_second) = launch.worker_commands(
        TOOL, config, ["0", "1,2"], ["a.yaml"], ["library=/lib/library.json"]
    )
    assert (env_first["CUDA_VISIBLE_DEVICES"], env_second["CUDA_VISIBLE_DEVICES"]) == ("0", "1,2")
    assert first[2:4] == ["--role", "worker"] and first[4:6] == ["--config", "a.yaml"]
    assert "worker.port=9200" in first and "worker.port=9201" in second
    assert "worker.coordinator_url=ws://127.0.0.1:9100" in second
    assert "worker.advertise_url=ws://127.0.0.1:9201" in second
    assert "library=/lib/library.json" in first and "worker.worker_id=gpu-1-2" in second


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_one_command_serves_a_worker_per_card_and_stops_them_with_it():
    port, worker_port = free_port(), free_port()
    server = subprocess.Popen(
        [
            sys.executable, str(TOOL), "--role", "all", "--gpus", "0,1",
            "--set", f"coordinator.port={port}", "--set", "coordinator.host=127.0.0.1",
            "--set", f"worker.port={worker_port}",
        ],  # fmt: skip
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    try:
        deadline = time.time() + 60
        total = 0
        while time.time() < deadline and total < 2:
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/lobby", timeout=2) as answer:
                    total = json.load(answer)["workers"]["total"]
            except OSError:
                time.sleep(0.5)
        assert total == 2
    finally:
        server.send_signal(signal.SIGINT)
        server.wait(30)
    with socket.socket() as sock:
        sock.settimeout(1)
        assert sock.connect_ex(("127.0.0.1", worker_port)) != 0
