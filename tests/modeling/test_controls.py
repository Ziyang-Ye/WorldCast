"""The control embedding: the 20-frame history of a latent frame."""

import pytest
import torch

from worldcast.modeling.controls import ControlConfig, ControlEmbedding


def _embedding(dim: int = 16) -> ControlEmbedding:
    torch.manual_seed(0)
    return ControlEmbedding(dim, ControlConfig(hidden=32)).eval()


def test_paper_architecture():
    """11 buttons, the pitch and yaw deltas and a 32-d embedding of 52 weapons over 20 video
    frames: 900 numbers."""
    embedding = ControlEmbedding(3072)
    assert embedding.weapon_embedding.weight.shape == (52, 32)
    assert embedding.net[0].normalized_shape == (900,)
    assert (embedding.net[1].in_features, embedding.net[3].out_features) == (900, 3072)


def test_a_latent_frame_reads_the_twenty_video_frames_that_end_at_it():
    """Latent frame ``j`` of the window reads the video frames ``4 j - 19 .. 4 j``, clamped at
    video frame 0."""
    embedding = _embedding()
    index = torch.arange(33, dtype=torch.float32).view(1, 33, 1)  # the frame index as the feature
    history = embedding.group_history(index, frame_offset=7, num_frames=2)
    assert history.shape == (1, 2, 20)
    assert history[0, 0].tolist() == [float(t) for t in range(9, 29)]  # latent frame 7 ends at 28
    assert history[0, 1].tolist() == [float(t) for t in range(13, 33)]
    first = embedding.group_history(index, frame_offset=0, num_frames=2)
    assert first[0, 0].tolist() == [0.0] * 20  # latent frame 0 has video frame 0 alone
    assert first[0, 1].tolist() == [0.0] * 16 + [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="do not reach latent frame 8"):
        embedding.group_history(index[:, :32], frame_offset=7, num_frames=2)


def test_embedding_of_the_calls_frames_is_that_of_the_window():
    embedding = _embedding()
    g = torch.Generator().manual_seed(1)
    buttons = torch.randint(0, 2, (2, 33, 11), generator=g).float()
    view_deltas = torch.randn(2, 33, 2, generator=g)
    weapon = torch.randint(0, 52, (2, 33), generator=g)
    controls = (buttons, view_deltas, weapon)
    with torch.no_grad():
        window = embedding(*controls, frame_offset=0, num_frames=9)
        call = embedding(*controls, frame_offset=5, num_frames=4)
    assert window.shape == (2, 9, 16) and window.dtype == torch.float32
    torch.testing.assert_close(call, window[:, 5:9], rtol=0, atol=1e-6)


def test_the_embedding_names_a_wrong_input():
    embedding = _embedding()
    buttons, view_deltas = torch.zeros(1, 5, 11), torch.zeros(1, 5, 2)
    weapon = torch.zeros(1, 5, dtype=torch.long)
    at = dict(frame_offset=0, num_frames=1)
    with pytest.raises(ValueError, match="out-of-range"):
        embedding(buttons, view_deltas, weapon + 52, **at)
    with pytest.raises(
        ValueError, match=r"buttons must be floating \[B, T, 11\] .* got torch.bool"
    ):
        embedding(buttons.bool(), view_deltas, weapon, **at)
    with pytest.raises(
        ValueError, match=r"weapon must hold integer ids \[B, T\], got torch.float32"
    ):
        embedding(buttons, view_deltas, weapon.float(), **at)
    # the three controls cover the same video frames
    with pytest.raises(ValueError, match=r"buttons must be floating \[B, T, 11\] over the T = 5"):
        embedding(buttons[:, 1:], view_deltas, weapon, **at)
    with pytest.raises(ValueError, match=r"view_deltas must be floating \[B, T, 2\]"):
        embedding(buttons, torch.zeros(1, 5, 3), weapon, **at)
    with pytest.raises(ValueError, match=r"weapon must hold integer ids \[B, T\], got .* \(5,\)"):
        embedding(buttons, view_deltas, weapon[0], **at)
