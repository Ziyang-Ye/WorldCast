"""The latent cache."""

import numpy as np
import pytest
import torch

from worldcast.config.training import WINDOW_LATENT_FRAMES
from worldcast.data.latent_cache import (
    cached_window_starts,
    load_block_latents,
    load_first_latent,
    load_window_latents,
)
from worldcast.data.latents import window_key

MEDIA = "2393033-de_nuke-r15-p00"


@pytest.fixture(scope="module")
def latent_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("latents")
    rng = np.random.default_rng(42)
    windows = {
        window_key(s): (rng.standard_normal((1, WINDOW_LATENT_FRAMES, 48, 24, 42)) * 1.5).astype(
            np.float16
        )
        for s in (0, 80)
    }
    np.savez(root / f"{MEDIA}.npz", **windows)
    np.savez(root / "bad-shape.npz", win_000000=np.zeros((1, 21, 48, 24, 42), np.float16))
    return root


def test_cached_windows_are_read_as_float32(latent_cache):
    assert cached_window_starts(latent_cache, MEDIA).tolist() == [0, 80]
    assert cached_window_starts(latent_cache, "absent-media").tolist() == []
    window = load_window_latents(latent_cache, MEDIA, 80)
    assert window.shape == (WINDOW_LATENT_FRAMES, 48, 24, 42) and window.dtype == torch.float32
    with np.load(latent_cache / f"{MEDIA}.npz") as archive:
        assert torch.equal(window, torch.from_numpy(archive["win_000080"][0]).float())
    assert torch.equal(load_block_latents(latent_cache, MEDIA, 80, 5), window[5:9])
    assert torch.equal(load_first_latent(latent_cache, MEDIA, 80), window[:1])
    with pytest.raises(RuntimeError, match="latent frames 39..42 are not all in the cached window"):
        load_block_latents(latent_cache, MEDIA, 80, 39)


def test_first_latent_errors(latent_cache):
    with pytest.raises(FileNotFoundError):
        load_first_latent(latent_cache, "absent-media", 0)
    with pytest.raises(KeyError):
        load_first_latent(latent_cache, MEDIA, 16)
    with pytest.raises(ValueError):
        load_first_latent(latent_cache, "bad-shape", 0)
