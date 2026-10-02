"""Fixtures shared by the tests."""

import pytest
import torch

PAPER_TEMPORAL = (False, True, True)  # Wan2_2_VAE's temperal_downsample


@pytest.fixture()
def new_tiny_vae():
    """A small random new ``Wan22VAE`` (paper temporal layout), for tests that need no old code."""
    from worldcast.modeling.wan22 import vae as V

    torch.manual_seed(37)
    config = V.VAEConfig(
        dim=8,
        dec_dim=8,
        z_dim=48,
        dim_mult=(1, 1, 2, 2),
        num_res_blocks=1,
        temperal_downsample=PAPER_TEMPORAL,
    )
    return V.Wan22VAE(config).eval()
