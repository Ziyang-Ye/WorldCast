"""The local role: every worker in this process, so one WorldCast client at most; with mock
clients it loads no torch."""

import subprocess
import sys
from pathlib import Path

import pytest

from worldcast.demo import local
from worldcast.demo.config import load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "demo"


def test_two_worldcast_clients_are_refused_before_any_model_loads():
    config = load_config([CONFIGS / "worldcast.yaml"])
    with pytest.raises(ValueError, match="one --role worker per GPU"):
        local.local_config(config, workers=2)


def test_the_laptop_demo_loads_no_torch():
    """The coordinator, a worker and the mock client, as ``--role local`` imports them."""
    code = "import sys, worldcast.demo.local, worldcast.demo.mock; print('torch' in sys.modules)"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert run.stdout.strip() == "False"


def test_browsers_reach_local_workers_through_the_coordinator():
    config = load_config()
    assert config.coordinator.frame_route == "direct"
    assert local.local_config(config, workers=2).coordinator.frame_route == "proxy"
