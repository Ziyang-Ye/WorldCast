"""The demo's inference is the evaluation's but for the settings live play names: its resolved
inference config and serving options against those of ``tools/run_client.py``, key by key."""

from dataclasses import asdict, replace
from pathlib import Path

from worldcast.config.inference import InferenceConfig, load_config
from worldcast.config.loader import differences
from worldcast.demo.config import load_config as load_demo_config
from worldcast.demo.worldcast_client import inference_config
from worldcast.engine.inference.live_rollout import LIVE_ROUND_LATENTS
from worldcast.engine.inference.serving import LOW_LATENCY_ATTENTION, ServingOptions

WORKER = Path(__file__).resolve().parents[2] / "configs" / "demo" / "worldcast.yaml"
#: Where the inference config of a worker differs from the evaluation's (its defaults, which the
#: paper's runs set; the paths come from the same files), the evaluation's value and the worker's,
#: and why.
CONFIG = {
    # the source of the player states: the player's keys (the paper's are the recording's)
    "player_state.source": ("gt", "live"),
    # the round a client keeps, which slides (the evaluation's is the recording's length)
    "run.latent_frames": (441, LIVE_ROUND_LATENTS),
    # latency: a block must be ready as its second of frames runs out
    "model.attention": ("flash", LOW_LATENCY_ATTENTION),
    "scene_state.bound": (64, 32),
    # a player who stands still keeps the context of its first blocks (a live client's setting)
    "live.idle_context": ("none", "freeze"),
    "live.idle_with_peers": (False, True),
    # a wall the depth head sees stops the player, whose keys are cleared there (a live client's)
    "live.wall_keys": ("held", "guard"),
    "live.dead_zone_u": (0.0, 4.0),
    "live.hop": ("step", "spread"),
}
#: The same for how the client is served: the evaluation's decode is a separate step with the
#: Wan2.2 VAE, and its clients wait for each other.
SERVING = {
    "compile": (False, True),
    "decoder": ("none", "taehv"),
    "taehv_path": (None, "weights/taew2_2.pth"),
    "lockstep": (True, False),
}


def test_the_workers_inference_config_is_the_evaluations_but_for_the_named_settings():
    demo = load_demo_config([WORKER])
    worker = replace(demo.client.worldcast, configs=[])
    assert differences(load_config(), inference_config(worker, demo.scene_state)) == CONFIG


def test_the_workers_serving_is_the_evaluations_but_for_the_named_options():
    worker = load_demo_config([WORKER]).client.worldcast
    evaluation = ServingOptions(decoder="none")  # tools/run_client.py
    assert differences(asdict(evaluation), asdict(worker.serving_options())) == SERVING


def test_what_the_demo_sets_is_what_the_inference_config_names_and_nothing_else():
    """No key of the demo's overrides is made up: each is a setting of the inference config (a
    misspelt one would not load), and the YAML sets no more than the live source, the idle context
    and the wall guard, which the table above names."""
    overrides = load_demo_config([WORKER]).client.worldcast.overrides
    assert set(overrides) == {
        "player_state.source",
        "live.idle_context",
        "live.idle_with_peers",
        "live.wall_keys",
        "live.dead_zone_u",
        "live.hop",
    }
    assert differences(InferenceConfig(), InferenceConfig().with_overrides(overrides)).keys() == {
        key for key in CONFIG if key in overrides
    }
