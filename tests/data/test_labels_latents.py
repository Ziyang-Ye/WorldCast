"""The cached first latent and the visibility labels."""

import numpy as np
import pytest

from worldcast.data.latents import CACHE_LATENT_FRAMES, load_first_latent, window_key
from worldcast.data.visibility_labels import VisibilityLabels


def test_visibility_root_checks(tmp_path):
    with pytest.raises(FileNotFoundError):
        VisibilityLabels(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        VisibilityLabels(tmp_path)  # a directory without npz files


@pytest.fixture(scope="module")
def latent_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("latents")
    rng = np.random.default_rng(42)
    windows = {
        window_key(s): (rng.standard_normal((1, CACHE_LATENT_FRAMES, 48, 24, 42)) * 1.5).astype(
            np.float16
        )
        for s in (0, 80)
    }
    np.savez(root / "2393033-de_nuke-r15-p00.npz", **windows)
    np.savez(root / "bad-shape.npz", win_000000=np.zeros((1, 21, 48, 24, 42), np.float16))
    return root


def test_first_latent_errors(latent_cache):
    with pytest.raises(FileNotFoundError):
        load_first_latent(latent_cache, "absent-media", 0)
    with pytest.raises(KeyError):
        load_first_latent(latent_cache, "2393033-de_nuke-r15-p00", 16)
    with pytest.raises(ValueError):
        load_first_latent(latent_cache, "bad-shape", 0)
