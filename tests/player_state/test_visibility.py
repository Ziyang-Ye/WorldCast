"""The visibility gate: the batch's labels folded to latent frames, the gate and its confidence."""

import pytest
import torch

from worldcast.player_state.visibility import (
    fold_video_frames,
    latent_visibility,
    live_eligibility,
    pad_framewise_to_window,
    visibility_confidence,
    visible_latent_frames,
)


def test_a_latent_frame_is_visible_if_any_of_its_video_frames_is():
    visibility = torch.zeros(1, 1, 9, dtype=torch.bool)
    visibility[0, 0, 3] = True  # one video frame of latent frame 1
    valid = torch.ones(1, 1, 9, dtype=torch.bool)
    valid[0, 0, 6] = False  # one video frame of latent frame 2
    visible, known = latent_visibility(visibility, valid)
    assert visible[0, :, 0].tolist() == [False, True, False]
    assert known[0, :, 0].tolist() == [True, True, False]  # known in all of its video frames
    # the labels of the loader are float32 in {0, 1}
    assert torch.equal(latent_visibility(visibility.float(), valid)[0], visible)
    # the video frames 5 .. 12 of the latent frames 2 and 3
    labels = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.bool)[:, None]
    assert fold_video_frames(labels, 2, 2)[:, 0].tolist() == [True, False]
    assert fold_video_frames(labels[:5], 0, 2)[:, 0].tolist() == [False, True]  # 1 + 4 frames


def test_the_gate_reads_the_batchs_labels():
    visibility = torch.zeros(1, 2, 9)
    visibility[0, 1, [0, 7]] = 1.0  # player 1: the first frame, and a video frame of latent frame 2
    valid = torch.ones(1, 2, 9, dtype=torch.bool)
    valid[0, 0, 2] = False  # unknown reads as not visible
    batch = {"client_visibility": visibility, "client_visibility_valid": valid}
    visible = visible_latent_frames(batch)
    assert visible.dtype == torch.bool
    assert visible[0].tolist() == [[False, True], [False, False], [False, True]]
    valid[0, 1, 7] = False
    with pytest.raises(
        ValueError, match=r"visible where client_visibility_valid is False \(1 of 18\)"
    ):
        visible_latent_frames(batch)
    with pytest.raises(ValueError, match="no visibility labels"):
        visible_latent_frames({"client_visibility": visibility})


def test_the_gate_drops_the_client_the_dead_and_the_unseen():
    in_front = torch.ones(1, 5, 4, dtype=torch.bool)
    in_front[0, :, 3] = False
    alive = torch.ones(1, 4, 4)
    alive[0, :, 1] = 0.0
    visible = torch.ones(1, 4, 4)
    visible[0, 0, 2] = 0.0
    gate = live_eligibility(
        in_front, alive=alive, visible=visible, client_slot=torch.tensor([0]), frame_offset=1
    )
    assert gate.is_client[0, 0].tolist() == [True, False, False, False]
    assert not bool(gate.eligible[0, 0].any())  # frame 0 is outside the call: padded with zeros
    assert gate.eligible[0, 1].tolist() == [False, False, False, False]
    assert gate.eligible[0, 2].tolist() == [False, False, True, False]
    padded = pad_framewise_to_window(alive, window=in_front, frame_offset=1)
    assert torch.equal(gate.alive_window, padded) and torch.equal(padded[:, 1:], alive)
    assert not bool(padded[:, 0].any())


def test_the_confidence_is_a_causal_average_that_restarts_at_block_starts():
    visible = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).view(1, 7, 1)
    e = visibility_confidence(visible)[0, :, 0].tolist()
    # blocks: frame 0 | frames 1-4 | frames 5-6; e = 0.3 + 0.7 ema, alpha 0.5
    ema = [1.0, 0.0, 0.5, 0.75, 0.375, 0.0, 0.5]
    assert e == pytest.approx([0.3 + 0.7 * v for v in ema])
    one_block = visibility_confidence(visible, frames_per_block=7, first_frame_alone=False)
    assert one_block[0, 1, 0].item() == pytest.approx(0.3 + 0.7 * 0.5)
    assert bool((visibility_confidence(visible, floor=1.0) == 1.0).all())
