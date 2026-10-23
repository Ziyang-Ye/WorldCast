"""The tiny decoder of the client's latents."""

import torch

from worldcast.engine.inference.taehv import (
    FRAMES_TO_TRIM,
    StreamingTinyDecoder,
    TinyDecoder,
    load_taew2_2,
)


def _decoder(seed: int = 0) -> TinyDecoder:
    torch.manual_seed(seed)
    return TinyDecoder().eval()


def test_each_latent_frame_gives_four_video_frames_after_the_first():
    stream = StreamingTinyDecoder(_decoder())
    latents = torch.randn(5, 48, 3, 4, generator=torch.Generator().manual_seed(1))
    frames = stream.decode(latents)
    assert FRAMES_TO_TRIM == 3
    assert frames.shape == (4 * 5 - 3, 3, 48, 64)  # latent 0 gives one frame
    assert float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0
    assert stream.decode(latents[:1]).shape[0] == 4  # the stream goes on: nothing more to drop


def test_the_frames_do_not_depend_on_the_chunking():
    decoder = _decoder()
    latents = torch.randn(5, 48, 3, 4, generator=torch.Generator().manual_seed(2))
    whole = StreamingTinyDecoder(decoder).decode(latents)
    stream = StreamingTinyDecoder(decoder)
    chunks = torch.cat([stream.decode(latents[:1]), stream.decode(latents[1:])])
    assert torch.allclose(chunks, whole, atol=1e-5)
    stream.reset()
    assert torch.allclose(stream.decode(latents), whole, atol=1e-5)


def test_a_stream_goes_on_from_a_snapshot_as_it_did_after_it():
    """A block decoded latent by latent (as the client decodes) equals its first latents and,
    after other latents were decoded from the same point, the rest resumed from the snapshot taken
    there (a refreshed block switches at a latent so), bit for bit; a snapshot is not changed by
    the decoding after it."""
    decoder = _decoder()
    generator = torch.Generator().manual_seed(4)
    first, block, other = (torch.randn(n, 48, 3, 4, generator=generator) for n in (1, 4, 4))

    def latent_by_latent(stream: StreamingTinyDecoder, latents: torch.Tensor) -> list:
        return [stream.decode(latents[i : i + 1]) for i in range(len(latents))]

    whole = StreamingTinyDecoder(decoder)
    whole.decode(first)
    reference = latent_by_latent(whole, block)
    for k in range(4):
        stream = StreamingTinyDecoder(decoder)
        stream.decode(first)
        head = latent_by_latent(stream, block[:k])
        state = stream.snapshot()
        kept = [None if m is None else m.clone() for m in state[0]]
        latent_by_latent(stream, other[k:])  # another version of the block's later latents
        assert all(m is None or torch.equal(m, c) for m, c in zip(state[0], kept))
        stream.restore(state)
        resumed = head + latent_by_latent(stream, block[k:])
        assert all(torch.equal(a, b) for a, b in zip(resumed, reference))
    fresh = StreamingTinyDecoder(decoder)
    start = fresh.snapshot()
    fresh.decode(first)
    fresh.restore(start)  # the stream's own start: its first frames are dropped again
    assert fresh.decode(first).shape[0] == 1


def test_the_checkpoint_loads_strictly(tmp_path):
    source = _decoder(3)
    state = dict(source.state_dict(), **{"encoder.0.weight": torch.zeros(1)})
    # upstream keeps two frames' worth of channels in the first temporal layer, which grows none
    first = next(i for i, layer in enumerate(source.decoder) if type(layer).__name__ == "_TGrow")
    key = f"decoder.{first}.conv.weight"
    state[key] = torch.cat([torch.zeros_like(state[key]), state[key]])
    torch.save(state, tmp_path / "taew2_2.pth")
    loaded = load_taew2_2(tmp_path / "taew2_2.pth", device="cpu")
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in loaded.state_dict().items())
