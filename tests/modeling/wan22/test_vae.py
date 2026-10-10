"""The Wan2.2 VAE: its latent statistics, the shapes of encode and decode, the streaming decoder."""

import functools

import pytest
import torch

from tests.modeling.support import random_latent_scale
from worldcast.modeling.wan22 import vae as V


def _latents(frames: int, seed: int) -> torch.Tensor:
    return torch.randn(1, 48, frames, 4, 6, generator=torch.Generator().manual_seed(seed))


def test_latent_statistics_and_the_decode_scale():
    """``latent_scale`` is ``[mean, 1 / std]`` with the reciprocal taken after the cast: in bf16
    that differs from the cast reciprocal in 18 of the 48 channels."""
    mean, std = V.latent_mean_std()
    assert mean.shape == std.shape == (48,) and mean.dtype == std.dtype == torch.float32
    assert mean[0].item() == pytest.approx(-0.2289) and std[0].item() == pytest.approx(0.4765)
    assert int((std != torch.tensor(V.LATENT_STD)).sum()) == 6  # std is 1 / (1 / LATENT_STD)
    bf16 = torch.bfloat16
    scale = V.latent_scale("cpu", bf16)
    assert torch.equal(scale[0], mean.to(bf16)) and torch.equal(scale[1], 1.0 / std.to(bf16))
    assert int((scale[1] != (1.0 / std).to(bf16)).sum()) == 18


def test_patchify_stacks_each_2x2_patch_on_the_channels():
    x = torch.arange(24, dtype=torch.float32).view(1, 1, 1, 4, 6)
    patches = V.patchify(x, 2)
    assert patches.shape == (1, 4, 1, 2, 3)
    # the patch at row 1, column 2: its left column top to bottom, then its right column
    assert patches[0, :, 0, 1, 2].tolist() == [16.0, 22.0, 17.0, 23.0]
    assert torch.equal(V.unpatchify(patches, 2), x)
    with pytest.raises(ValueError, match="patchify expects"):
        V.patchify(x[0], 2)


def test_encode_and_decode_are_causal_in_time(tiny_vae):
    """``1 + 4 k`` frames at 16x the latent grid <-> ``1 + k`` latent frames; a latent frame
    depends on the frames up to its own, a decoded frame on the latents up to its own."""
    scale = random_latent_scale(48)
    g = torch.Generator().manual_seed(1)
    video = torch.rand(1, 3, 9, 32, 48, generator=g) * 2 - 1
    later = video.clone()
    later[:, :, 5:] += 0.5  # the frames of latent 2
    with torch.no_grad():
        z, z_later = tiny_vae.encode(video, scale), tiny_vae.encode(later, scale)
        z_changed = z.clone()
        z_changed[:, :, 2] += 0.5
        pixels, pixels_later = tiny_vae.decode(z, scale), tiny_vae.decode(z_changed, scale)
    assert z.shape == (1, 48, 3, 2, 3) and pixels.shape == video.shape
    assert torch.equal(z_later[:, :, :2], z[:, :, :2]) and not torch.equal(
        z_later[:, :, 2], z[:, :, 2]
    )
    assert torch.equal(pixels_later[:, :, :5], pixels[:, :, :5])
    assert not torch.equal(pixels_later[:, :, 5:], pixels[:, :, 5:])


def test_encode_names_a_frame_count_or_size_it_cannot_take(tiny_vae):
    with pytest.raises(ValueError, match="1 \\+ 4 k frames, got 12"):
        tiny_vae.encode(torch.zeros(1, 3, 12, 32, 48))
    with pytest.raises(ValueError, match="divisible by 16, got \\(30, 48\\)"):
        tiny_vae.encode(torch.zeros(1, 3, 9, 30, 48))
    for shape in ((1, 9, 3, 32, 48), (3, 9, 32, 48)):  # frame-major, no batch axis
        with pytest.raises(ValueError, match=r"encodes pixels \[b, 3, T, H, W\]"):
            tiny_vae.encode(torch.zeros(shape))


def test_the_default_scale_is_for_the_48_latent_channels():
    vae = V.Wan22VAE(
        V.VAEConfig(dim=8, dec_dim=8, z_dim=4, dim_mult=(1, 1, 2, 2), num_res_blocks=1)
    )
    z = torch.zeros(1, 4, 1, 2, 3)
    with pytest.raises(ValueError, match="pass one for z_dim = 4"):
        vae.decode(z)
    scale = [torch.zeros(4), torch.ones(4)]
    assert vae.decode(z, scale).shape == (1, 3, 1, 32, 48)


def test_the_scale_defaults_to_the_latent_statistics_of_the_input(tiny_vae):
    z = _latents(2, seed=44)
    video = torch.rand(1, 3, 5, 32, 48, generator=torch.Generator().manual_seed(2)) * 2 - 1
    scale = V.latent_scale(z.device, z.dtype)
    with torch.no_grad():
        assert torch.equal(tiny_vae.decode(z), tiny_vae.decode(z, scale))
        assert torch.equal(tiny_vae.encode(video), tiny_vae.encode(video, scale))
        chunks = [V.Wan22StreamingDecoder(tiny_vae).decode_chunk(z, s) for s in (None, scale)]
    assert torch.equal(*chunks)
    with pytest.raises(ValueError, match="1 \\+ 4 k frames, got 6"):
        tiny_vae.encode(torch.zeros(1, 3, 6, 32, 48))


def test_streaming_concatenates_to_the_one_call_decode(tiny_vae):
    z, scale = _latents(6, seed=45), random_latent_scale(48)
    full = tiny_vae.decode(z, scale)
    decoder = V.Wan22StreamingDecoder(tiny_vae)
    chunks = [decoder.decode_chunk(z[:, :, a:b], scale) for a, b in ((0, 1), (1, 3), (3, 6))]
    assert [chunk.shape[2] for chunk in chunks] == [1, 8, 12]
    assert full.shape == (1, 3, 21, 64, 96) and not full.requires_grad
    assert torch.equal(torch.cat(chunks, 2), full)


def test_streaming_reset_restarts_the_stream(tiny_vae):
    z, scale = _latents(3, seed=47), random_latent_scale(48)
    decoder = V.Wan22StreamingDecoder(tiny_vae)
    first = decoder.decode_chunk(z, scale)
    continued = decoder.decode_chunk(z, scale)
    decoder.reset()
    assert continued.shape[2] == 12 and first.shape[2] == 9  # latent 0 is one frame
    assert torch.equal(decoder.decode_chunk(z, scale), first)


def test_streaming_decoder_rejects_bad_inputs(tiny_vae):
    decoder, scale = V.Wan22StreamingDecoder(tiny_vae), random_latent_scale(48)
    for shape in ((48, 3, 4, 6), (1, 48, 0, 4, 6), (1, 3, 48, 4, 6)):  # the last: frame-major
        with pytest.raises(ValueError, match=r"decodes latents \[B, 48, F >= 1, h, w\]"):
            decoder.decode_chunk(torch.zeros(shape), scale)


def test_load_assigns_the_checkpoint_without_a_random_draw(tiny_vae, tmp_path, monkeypatch):
    """The loader builds the Wan2.2 VAE; here its class is built at the tiny size."""
    path = tmp_path / V.VAE_CHECKPOINT_NAME
    torch.save(tiny_vae.state_dict(), path)
    torch.manual_seed(0)
    before = torch.get_rng_state()
    monkeypatch.setattr(V, "Wan22VAE", functools.partial(V.Wan22VAE, tiny_vae.config))
    loaded = V.load_wan22_vae(path)
    assert torch.equal(torch.get_rng_state(), before)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    state, want = loaded.state_dict(), tiny_vae.state_dict()
    assert state.keys() == want.keys() and all(torch.equal(state[k], want[k]) for k in want)
