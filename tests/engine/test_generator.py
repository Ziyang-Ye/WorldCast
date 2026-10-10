"""The generator's config, prompt embedding, VAE, cameras and window inputs."""

import json

import numpy as np
import pytest
import torch

from tests.engine.training.support import make_item, write_prompt_embedding
from worldcast.data.camera import window_cameras
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.window import collate_windows
from worldcast.engine.generator import (
    backbone_config,
    client_cameras,
    load_prompt_embeds,
    load_vae,
    window_conditions,
    window_inputs,
)
from worldcast.modeling.controls import CONTROL_KEYS
from worldcast.modeling.ray_embedding import RayConditions
from worldcast.modeling.wan22.model import GeneratorConfig


def test_the_backbone_dimensions_come_from_the_snapshot(tmp_path):
    assert backbone_config(None) == GeneratorConfig()
    assert backbone_config(tmp_path) == GeneratorConfig()  # a root without config.json
    snapshot = {"dim": 64, "num_heads": 4, "num_layers": 3}
    (tmp_path / "config.json").write_text(json.dumps(snapshot))
    config = backbone_config(tmp_path)
    assert (config.dim, config.num_heads, config.num_layers) == (64, 4, 3)
    assert config.controls == GeneratorConfig().controls


def test_the_prompt_embedding_is_read_from_its_file(tmp_path):
    path = write_prompt_embedding(tmp_path / "prompt.safetensors")
    embeds = load_prompt_embeds(path, None, "cpu")
    assert embeds.shape == (1, 512, 24)
    with pytest.raises(ValueError, match="prompt embedding"):
        load_prompt_embeds(None, None, "cpu")


def test_the_vae_is_that_of_the_wan22_root(tmp_path):
    with pytest.raises(ValueError, match="set the Wan2.2 root: the VAE is its Wan2.2_VAE.pth"):
        load_vae(None, "cpu")
    with pytest.raises(FileNotFoundError, match="Wan2.2_VAE.pth"):
        load_vae(tmp_path, "cpu")


def test_the_clients_cameras_are_a_batchs_camera_entries():
    item = make_item(3)
    slot = int(item["client_slot"])
    signals = {key: np.asarray(item[key]) for key in OBSERVER_SIGNAL_KEYS}
    states, weapons = item["player_states"][slot], item["player_weapon_ids"][slot]
    cameras = client_cameras(states, signals, weapons, 41)
    assert list(cameras) == ["window_c2w", "window_tans"]
    assert cameras["window_c2w"].shape == (1, 41, 4, 4) and cameras["window_tans"].shape == (
        1,
        41,
        2,
    )
    c2w, tans = window_cameras(states, signals, weapons, 41)
    assert torch.equal(cameras["window_c2w"][0], c2w)
    assert np.array_equal(cameras["window_tans"][0].numpy(), tans)
    # the camera sits at the client's position at each latent frame (video frames 0, 4, ..)
    assert torch.allclose(cameras["window_c2w"][0, :, :2, 3], states[::4, :2])
    rays = RayConditions.contiguous(cameras["window_c2w"], cameras["window_tans"])
    assert torch.equal(rays.conditions("cpu")["ray_anchor_c2w"], cameras["window_c2w"][:, 0])


def test_window_inputs_assemble_the_conditions():
    batch = collate_windows([make_item(0), make_item(1)])
    prompt = torch.randn(1, 7, 24)
    inputs = window_inputs(batch, prompt_embeds=prompt, device="cpu", dtype=torch.float32)
    assert inputs.latents.shape == (2, 41, 48, 24, 42)
    assert inputs.conditions["prompt_embeds"].shape == (2, 7, 24)
    assert inputs.visible.shape == (2, 41, 10) and inputs.visible.dtype == torch.bool
    assert torch.equal(inputs.conditions["player_visible"], inputs.visible.float())
    assert torch.equal(inputs.conditions["player_state_table"], inputs.player_state.table())
    assert set(CONTROL_KEYS) | set(OBSERVER_SIGNAL_KEYS) <= set(inputs.conditions)
    with pytest.raises(ValueError, match="latents must be one client's"):
        window_inputs(
            dict(batch, latents=batch["latents"][0]),
            prompt_embeds=prompt,
            device="cpu",
            dtype=torch.float32,
        )

    rays = {"window_c2w": torch.eye(4).expand(2, 41, 4, 4)}
    plain = window_inputs(
        batch,
        prompt_embeds=prompt,
        device="cpu",
        dtype=torch.float32,
        observer_signals=False,
        rays=rays,
    )
    assert not set(OBSERVER_SIGNAL_KEYS) & set(plain.conditions)
    assert plain.conditions["window_c2w"] is rays["window_c2w"]
    # the conditions alone: the batch's latents are never read
    full = window_inputs(batch, prompt_embeds=prompt, device="cpu", dtype=torch.float32, rays=rays)
    alone = window_conditions(
        {key: value for key, value in batch.items() if key != "latents"},
        prompt_embeds=prompt,
        device="cpu",
        dtype=torch.float32,
        rays=rays,
    )
    assert alone.keys() == full.conditions.keys()
    assert all(torch.equal(alone[key], value) for key, value in full.conditions.items())
