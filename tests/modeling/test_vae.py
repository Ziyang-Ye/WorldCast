"""The streaming Wan2.2 VAE decoder."""

import pytest
import torch

from worldcast.modeling.wan22 import vae as V


def _random_scale(z_dim: int, dtype=torch.float32):
    g = torch.Generator().manual_seed(33)
    mean = torch.randn(z_dim, generator=g)
    inv_std = 1.0 / (torch.rand(z_dim, generator=g) + 0.5)
    return [mean.to(dtype), inv_std.to(dtype)]


def test_streaming_concatenates_to_the_one_call_decode(new_tiny_vae):
    new = new_tiny_vae
    z = torch.randn(1, 48, 6, 4, 6, generator=torch.Generator().manual_seed(45))
    scale = _random_scale(48)
    with torch.no_grad():
        full = new.decode(z, scale)
    dec = V.Wan22StreamingDecoder(new)
    chunks = [dec.decode_chunk(z[:, :, a:b], scale) for a, b in ((0, 1), (1, 3), (3, 6))]
    assert [c.shape[2] for c in chunks] == [1, 8, 12]
    assert torch.equal(torch.cat(chunks, 2), full)


def test_streaming_reset_restarts_the_stream(new_tiny_vae):
    new = new_tiny_vae
    z = torch.randn(1, 48, 3, 4, 6, generator=torch.Generator().manual_seed(47))
    scale = _random_scale(48)
    dec = V.Wan22StreamingDecoder(new)
    first = dec.decode_chunk(z, scale)
    dec.reset()
    second = dec.decode_chunk(z, scale)
    assert torch.equal(first, second)


def test_streaming_decoder_rejects_bad_inputs(new_tiny_vae):
    new = new_tiny_vae
    dec = V.Wan22StreamingDecoder(new)
    with pytest.raises(ValueError, match="shape"):
        dec.decode_chunk(torch.zeros(48, 3, 4, 6), _random_scale(48))
    with pytest.raises(ValueError, match="at least one"):
        dec.decode_chunk(torch.zeros(1, 48, 0, 4, 6), _random_scale(48))
