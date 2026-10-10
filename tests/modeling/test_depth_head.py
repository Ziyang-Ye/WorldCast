"""The depth head and its read-out: sizes, the temporal context of a frame, loading."""

import math

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tests.modeling.support import randomize_
from worldcast.data.latents import LATENT_GRID
from worldcast.modeling.depth_head import DepthHead, DepthReadout, load_depth_predictor


def _tiny_head(seed: int = 1) -> DepthHead:
    return randomize_(DepthHead(width=16, blocks=2, half_blocks=1), seed).eval()


def test_paper_architecture():
    """A 44M-parameter head (width 384, ten blocks and five at half resolution) and a 0.21M
    read-out on the latent grid."""
    with torch.device("meta"):
        head, readout = DepthHead(), DepthReadout()
    assert (len(head.blocks), len(head.dblocks), head.inp.out_channels) == (10, 5, 384)
    assert head.inp.in_channels == 3 * 48 and head.out.out_channels == 48
    assert sum(p.numel() for p in head.parameters()) == 44_199_600
    assert sum(p.numel() for p in readout.parameters()) == 208_132


def test_a_frame_sees_its_two_neighbours():
    """Each latent frame is read with the frames before and after it, the ends replicated: a
    change of frame 2 reaches frames 1-3, and a single frame is its own context."""
    head = _tiny_head()
    x = torch.randn(1, 5, 48, 6, 9, generator=torch.Generator().manual_seed(2))  # odd width
    changed = x.clone()
    changed[:, 2] += 1.0
    with torch.no_grad():
        out, moved = head(x), head(changed)
        single = head(x[:, :1])
        thrice = head(x[:, :1].expand(1, 3, 48, 6, 9))
    assert out.shape == x.shape
    same = [bool(torch.equal(out[:, f], moved[:, f])) for f in range(5)]
    assert same == [True, False, False, False, True]
    torch.testing.assert_close(single[:, 0], thrice[:, 1], rtol=0, atol=1e-5)


def test_the_head_names_latents_of_another_shape():
    head = _tiny_head()
    for shape in ((1, 3, 16, 6, 8), (3, 48, 6, 8)):  # other channels, no batch axis
        with pytest.raises(ValueError, match=r"reads latents \[B, F, 48, h, w\]"):
            head(torch.zeros(shape))


def test_the_readout_predicts_log_depth_around_200_units():
    readout = DepthReadout(width=16)
    torch.nn.init.zeros_(readout.net[-1].weight)
    torch.nn.init.zeros_(readout.net[-1].bias)
    out = readout(torch.randn(2, 48, 6, 8))
    assert out.shape == (2, 4, 6, 8)
    torch.testing.assert_close(out, torch.full_like(out, math.log(200.0)))


def test_the_readout_reads_the_heads_output():
    """The read-out takes any leading axes: the head's ``[B, F, 48, h, w]`` as it is."""
    head, readout = _tiny_head(), randomize_(DepthReadout(width=8), 3)
    x = torch.randn(2, 3, 48, 6, 8, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        features = head(x)
        depth = readout(features)
        assert depth.shape == (2, 3, 4, 6, 8)
        assert torch.equal(depth[1], readout(features[1]))


@pytest.mark.parametrize("suffix", [".pt", ".safetensors"])
def test_predictor_loads_at_the_files_size_without_a_random_draw(tmp_path, suffix):
    head, readout = _tiny_head(), randomize_(DepthReadout(width=8), 3)
    paths = [tmp_path / f"head{suffix}", tmp_path / f"readout{suffix}"]
    for module, path in zip((head, readout), paths):
        save = torch.save if suffix == ".pt" else save_file
        save(module.state_dict(), str(path))
    torch.manual_seed(0)
    before = torch.get_rng_state()
    predictor = load_depth_predictor(*paths)
    assert torch.equal(torch.get_rng_state(), before)
    assert (len(predictor.head.blocks), len(predictor.head.dblocks)) == (2, 1)
    assert predictor.readout.net[0].out_channels == 8
    assert not predictor.head.training and not any(
        p.requires_grad for p in predictor.head.parameters()
    )

    latents = torch.randn(3, 48, *LATENT_GRID, generator=torch.Generator().manual_seed(4))
    depth = predictor.log_depth(latents.to(torch.bfloat16))
    with torch.no_grad():
        want = readout(head(latents.to(torch.bfloat16).float()[None])[0])
    assert depth.shape == (3, 4, *LATENT_GRID) and depth.dtype == np.float32
    assert np.array_equal(depth, want.numpy())
    assert np.array_equal(predictor.log_depth(latents.to(torch.bfloat16).float().numpy()), depth)
    with pytest.raises(ValueError, match=r"\[F, 48, 24, 42\]"):
        predictor.log_depth(latents[None])
