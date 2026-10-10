"""Memory frames: a block of a player's window and what a block's window reads of it."""

import numpy as np
import pytest
import torch

from tests.data.support import media_record, tick_table, unscoped_signals, write_observer_labels
from worldcast.data.camera import half_angle_tangents
from worldcast.data.memory_frames import (
    MemoryBlock,
    MemoryFrames,
    batch_keys,
    block_memory_frames,
    memory_frame_inputs,
    recorded_block,
)
from worldcast.data.window import PlayerFrames, WindowSpec

SPEC = WindowSpec(latent_frames=9)


def test_a_block_spans_its_four_latent_frames():
    block = MemoryBlock.at("m0", 3, window_start=16, f0=5, c2w=None)
    assert (block.t_first, block.t_last) == (16 + 8 * 5, 16 + 8 * 8)


def test_a_recorded_block_is_keyed_at_the_cameras_of_its_latent_frames():
    table = tick_table()
    block = recorded_block(0, media_record(table), table, 0, 1, SPEC)
    assert (block.media_id, block.slot, block.window_start, block.f0) == ("m0", 0, 0, 1)
    # latent frame f is read at video frame 4 f, 0.25 f seconds into the window
    seconds = 0.25 * torch.arange(1, 5)
    assert torch.allclose(block.c2w[:, 0, 3], 250.0 * seconds)
    assert torch.allclose(block.c2w[:, 2, 3], torch.full((4,), 64.0))  # the eye height
    yaw = torch.deg2rad(30.0 * seconds)
    forward = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros(4)], dim=-1)
    assert torch.allclose(block.c2w[:, :3, 2], forward, atol=1e-6)


def test_a_block_the_recording_does_not_serve_is_none():
    table = tick_table(dies_at=1.6)
    media = media_record(table)
    assert recorded_block(0, media, table, 0, 1, SPEC) is not None
    assert recorded_block(0, media, table, 0, 5, SPEC) is None  # dead from latent frame 7 on
    short = tick_table(seconds=1.5)
    assert recorded_block(0, media_record(short), short, 0, 5, SPEC) is None  # ticks end before


def test_a_window_reads_the_blocks_16_video_frames_and_4_latent_signals():
    table = tick_table()
    frames = PlayerFrames.from_ticks(table, media_record(table), 0, SPEC)
    signals = unscoped_signals(SPEC.latent_frames)
    signals["obs_flash_flag"][5:9] = [1, 0, 1, 1]
    inputs = memory_frame_inputs(frames, signals, 5)
    assert inputs["states"].shape == (16, 6) and inputs["control_substeps"].shape[:2] == (16, 4)
    assert torch.equal(inputs["states"], torch.from_numpy(frames.states[17:33]))
    assert inputs["obs_flash_flag"].tolist() == [1, 0, 1, 1]
    unscoped = torch.tensor([half_angle_tangents()] * 4)
    assert torch.allclose(inputs["tans"], unscoped) and inputs["tans"].dtype == torch.float32
    assert np.isclose(float(inputs["tans"][0, 0]), 4.0 / 3.0, atol=1e-3)


def test_the_memory_frames_of_a_block_under_their_batch_keys():
    table = tick_table()
    frames = PlayerFrames.from_ticks(table, media_record(table), 0, SPEC)
    block = recorded_block(0, media_record(table), table, 0, 5, SPEC)
    latents = np.arange(4 * 48 * 24 * 42, dtype=np.float16).reshape(4, 48, 24, 42)
    tensors = block_memory_frames(block, latents, frames, unscoped_signals(SPEC.latent_frames))
    assert tensors["latents"].dtype == torch.float32 and tensors["latents"][0, 0, 0, 1] == 1.0
    assert torch.equal(tensors["c2w"], block.c2w) and tensors["states"].shape == (16, 6)
    assert sorted(batch_keys(tensors)) == sorted("memory_frames_" + name for name in tensors)
    with pytest.raises(RuntimeError, match="memory frames are 4 latent frames"):
        block_memory_frames(block, latents[:3], frames, unscoped_signals(SPEC.latent_frames))


def test_a_clients_memory_frames_read_the_blocks_own_player(tmp_path):
    client, other = tick_table(), tick_table(y=300.0)
    media = {0: media_record(client, 0, "m0"), 4: media_record(other, 4, "m4")}
    scoped = np.zeros(128)
    scoped[40] = 1  # the other player is scoped at source frame 40, the last of its latent frame 5
    write_observer_labels(tmp_path, "m4", 128, scoped_vis=scoped, level=scoped)
    source = MemoryFrames(
        client_media="m0",
        client_frames=PlayerFrames.from_ticks(client, media[0], 0, SPEC),
        client_signals=unscoped_signals(SPEC.latent_frames),
        round_slots=media,
        tick_tables={0: client, 4: other},
        spec=SPEC,
        observer_signal_label_root=tmp_path,
    )
    latents = torch.zeros(4, 48, 24, 42)
    own = source(recorded_block(0, media[0], client, 0, 5, SPEC), latents)
    theirs = source(recorded_block(4, media[4], other, 0, 5, SPEC), latents)
    assert own["states"][:, 1].tolist() == [0.0] * 16 and own["obs_scope_on"].tolist() == [0] * 4
    assert theirs["states"][:, 1].tolist() == [300.0] * 16
    assert theirs["obs_scope_on"].tolist() == [1, 0, 0, 0]
    assert theirs["c2w"][:, 1, 3].tolist() == [300.0] * 4
