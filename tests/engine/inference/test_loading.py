"""What a client loads: the generator it can run, and the length of its window."""

import pytest
import torch

import tests.engine.inference.support as sw
from worldcast.config.inference import InferenceConfig
from worldcast.engine.inference.loading import ClientModels, client_latents
from worldcast.engine.inference.serving import ServingOptions
from worldcast.modeling.build import CHECKPOINT_ROOT, EMA_KEY
from worldcast.modeling.wan22.model import worldcast_module


def test_a_generator_without_scene_state_is_refused_by_name(synthetic, tmp_path):
    weights = dict(synthetic["weights"])
    state = torch.load(weights["checkpoint"])[EMA_KEY]
    scene_state = ("observer_signals", "ray_embedding")
    kept = {
        key: value
        for key, value in state.items()
        if worldcast_module(key.removeprefix(CHECKPOINT_ROOT)) not in scene_state
    }
    weights["checkpoint"] = str(tmp_path / "without_scene_state.pt")
    torch.save({EMA_KEY: kept}, weights["checkpoint"])
    cfg = sw.config(synthetic["world"], weights, tmp_path / "out", max_blocks=1)
    with pytest.raises(ValueError, match="without observer_signals, ray_embedding: a client"):
        ClientModels.load(cfg, ServingOptions(decoder="none"))


#: (video frames the recording covers, run.max_blocks) -> latent frames N, or None: too short.
LATENTS = {
    (5000, 0): 441,  # run.latent_frames
    (1761, 0): 441,
    (1757, 0): 437,  # the coverage, rounded down to 1 + 4 k
    (120, 0): 29,
    (117, 0): 29,
    (116, 0): 29,
    (113, 0): 29,
    (112, 0): None,  # shorter than the first six blocks plus one
    (0, 0): None,
    (5000, 1): 29,  # latent frames 0-24 and one block
    (5000, 3): 37,
    (120, 3): 29,
}


@pytest.mark.parametrize(("covered", "max_blocks"), LATENTS)
def test_window_length(covered, max_blocks):
    """``client_latents`` clips to the coverage and to ``run.max_blocks``, then rounds down to
    ``1 + 4 k``."""
    cfg = InferenceConfig().with_overrides({"run.max_blocks": max_blocks})
    want = LATENTS[covered, max_blocks]
    if want is None:
        with pytest.raises(ValueError):
            client_latents(cfg, covered)
    else:
        assert client_latents(cfg, covered) == want
