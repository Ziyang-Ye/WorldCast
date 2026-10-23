"""One client's rollout, block by block, on the CPU with a tiny random model."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tests.engine.inference.support as sw
from worldcast.data.latents import BLOCK, FIRST_TARGET
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference.fast import FastGenerator
from worldcast.engine.inference.field_rows import FieldRows
from worldcast.engine.inference.loading import ClientModels, load_window
from worldcast.engine.inference.messages import MessagePart, MessageWorldState
from worldcast.engine.inference.rollout import BlockRead, Rollout
from worldcast.engine.inference.serving import ServingOptions
from worldcast.modeling.wan22.attention import sdpa_attention
from worldcast.player_state import continuous_row_state, field_builder
from worldcast.sampling.sampler import Sampler
from worldcast.sampling.window import KV_CACHE_LATENTS

OWN, RECORDED, HIDDEN = FieldRows.OWN, FieldRows.RECORDED, FieldRows.HIDDEN


def _config(synthetic, world=None, out=None):
    return sw.config(
        world or synthetic["world"],
        synthetic["weights"],
        out or synthetic["tmp"] / "rollout",
        max_blocks=sw.MAX_BLOCKS,
    )


def _rollout(cfg, world_state) -> Rollout:
    models = ClientModels.load(cfg, ServingOptions(decoder="none"))
    generator = FastGenerator(
        models.generator,
        field_builder(),
        input_dtype=torch.float32,
        attention=sdpa_attention,
        cuda_graphs=False,
        compile=False,
    )
    cache = models.generator.allocate_kv_cache(KV_CACHE_LATENTS)
    row = load_round_index_row(cfg.paths.round_index, 0)
    return Rollout(
        cfg,
        models,
        Sampler.create(generator, cache),
        load_window(cfg, row),
        world_state,
        lockstep=False,
        dtype=torch.float32,
    )


def _messages() -> MessageWorldState:
    return MessageWorldState(client=sw.media_id(0), poll_s=0.01)


def test_a_rollout_block_by_block_gives_the_clients_latents(synthetic, reference):
    sent: list[MessagePart] = []
    world_state = MessageWorldState(client=sw.media_id(0), poll_s=0.01, outbox=sent.append)
    rollout = _rollout(_config(synthetic), world_state)
    assert rollout.latent_frames == 1 + BLOCK * (6 + sw.MAX_BLOCKS)
    rollout.open()
    reads = []
    for f0 in range(1, rollout.latent_frames, BLOCK):
        x0 = rollout.denoise(f0, lambda: None)
        assert x0.shape == (1, BLOCK, 48, 24, 42)
        reads.append(rollout.commit(f0, x0))
        assert rollout.prepared is None  # a commit takes the prepared block
        rollout.prepare_next(f0)
        # the next block is prepared before its controls iff it reads the scene state
        reads_scene = FIRST_TARGET <= f0 + BLOCK < rollout.latent_frames
        assert (rollout.prepared is not None) == reads_scene
    assert np.array_equal(rollout.output[0].float().numpy(), reference)
    assert np.array_equal(rollout.store.numpy(), reference)

    # the first six blocks read nothing; the blocks after them read the client's own entries
    assert reads[:6] == [None] * 6 and len(reads) == 6 + sw.MAX_BLOCKS
    first: BlockRead = reads[6]
    assert first.t_target == rollout.source_frame(FIRST_TARGET) == 8 * FIRST_TARGET
    assert first.admitted == 0 and first.candidates == 6
    assert first.window_frames == (17 if first.entry is None else 21)
    assert first.entry is None or first.entry[0] == sw.media_id(0)
    assert first.missing >= first.coverage >= 0
    # every block went out as a memory entry; the blocks that read the scene state sent a step
    # record first
    kinds = [part.kind for part in sent]
    assert kinds.count("scene") == 6 + sw.MAX_BLOCKS
    assert kinds.count("step") == sw.MAX_BLOCKS


def test_the_methods_of_a_rollout_are_called_in_order(synthetic, monkeypatch):
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    rollout = _rollout(_config(synthetic), _messages())
    with pytest.raises(RuntimeError, match="the rollout is not open: call open\\(\\) first"):
        rollout.denoise(1, lambda: None)
    rollout.open()
    with pytest.raises(RuntimeError, match="block 25 was not prepared: call prepare_next\\(21\\)"):
        rollout.denoise(FIRST_TARGET, lambda: None)


# ------------------------------------------------------------------------ the rows the field read
def _run_reading_field_rows(rollout: Rollout, monkeypatch) -> tuple[dict, dict, dict]:
    """Run ``rollout`` block by block. Returns, by latent frame: the rows read as soon as its block
    is committed (latent frame 0: once the rollout is open); the rows of the next block, read then,
    before it is prepared; and the player-state table the generator was given ``[P, 6 | 13]``."""
    given = []
    denoise = rollout.sampler.denoise

    def spy(noisy, conditions, **kwargs):
        given.append(conditions["player_state_table"][0].double().numpy())
        return denoise(noisy, conditions, **kwargs)

    monkeypatch.setattr(rollout.sampler, "denoise", spy)
    rollout.open()
    read, early, field = {0: rollout.field_rows(0)}, {}, {}
    for f0 in range(1, rollout.latent_frames, BLOCK):
        rollout.commit(f0, rollout.denoise(f0, lambda: None))
        frames = range(f0, f0 + BLOCK)
        read.update((f, rollout.field_rows(f)) for f in frames)
        # the table of a window ends with its four target frames; the round's has every latent frame
        table = given[-1]
        field.update(zip(frames, table[-BLOCK:] if f0 >= FIRST_TARGET else table[f0 : f0 + BLOCK]))
        following = range(f0 + BLOCK, min(f0 + 2 * BLOCK, rollout.latent_frames))
        early.update((f, rollout.field_rows(f)) for f in following)
        rollout.prepare_next(f0)
    return read, early, field


def _assert_the_field_read_the_rows(read, field):
    """The rows are those of the table the generator was given: x, y, z and alive; from latent
    frame 25 on, where the table carries the continuous columns (6, 7), yaw and pitch too."""
    assert sorted(field) == list(range(1, 33))
    for latent, table in field.items():
        rows = read[latent].states
        assert np.array_equal(rows[:, [0, 1, 2, 5]], table[:, [0, 1, 2, 5]])
        if latent >= FIRST_TARGET:
            assert table.shape == (10, 13)
            assert np.array_equal(rows[:, [3, 4]], table[:, [6, 7]])
        else:
            assert table.shape == (10, 6)


def test_the_field_rows_under_gt_are_the_recorded_rows(synthetic, reference, monkeypatch):
    rollout = _rollout(_config(synthetic), _messages())
    read, early, field = _run_reading_field_rows(rollout, monkeypatch)
    # a rollout that reads the rows after every block gives the latents of one that does not
    assert np.array_equal(rollout.output[0].float().numpy(), reference)
    _assert_the_field_read_the_rows(read, field)
    # gt writes no row: a row read before its block is the row read after
    assert all(np.array_equal(rows.states, read[f].states) for f, rows in early.items())

    assert sorted(read) == list(range(33))
    recorded = np.stack([rollout.recording.player_frames[slot].states for slot in range(10)])
    angles = continuous_row_state(rollout.round_batch)[0, :, :, :2].double().numpy()
    for latent, rows in read.items():
        assert rows.states.shape == (10, 6) and rows.states.dtype == np.float64
        xyz_alive = recorded[:, 4 * latent][:, [0, 1, 2, 5]]
        assert np.array_equal(rows.states[:, [0, 1, 2, 5]], xyz_alive)
        assert np.array_equal(rows.states[:, 3:5], angles[latent])
        assert rows.source.dtype == np.int8
        assert rows.source.tolist() == [OWN] + [RECORDED] * 9
        assert rows.drawn is None and rows.status == "gt"
    # the first frame's angles are the recorded ones
    first = read[0].states
    assert np.allclose(first[:, 3], (recorded[:, 0, 3] + 180.0) % 360.0 - 180.0, atol=1e-4)
    assert np.allclose(first[:, 4], recorded[:, 0, 4], atol=1e-4)


def test_the_field_rows_under_predicted_are_the_rewritten_table(synthetic, monkeypatch, tmp_path):
    world = sw.make_world(tmp_path / "world", clients=(0,), seconds=11.0)  # 10 s state windows
    sw.patch_ticks(monkeypatch, world["tables"])
    closed = {"player_state.source": "predicted", **sw.write_state_model(tmp_path / "state.pt")}
    cfg = _config(synthetic, world, tmp_path / "out").with_overrides(closed)
    rollout = _rollout(cfg, _messages())
    read, early, field = _run_reading_field_rows(rollout, monkeypatch)
    _assert_the_field_read_the_rows(read, field)

    # the rows read when their block was committed are the table's at the end: written once
    table = rollout.closed.table.states[0].double().numpy()
    recorded = rollout.recorded_states[0].double().numpy()
    angles = continuous_row_state(rollout.round_batch)[0, :, :, :2].double().numpy()
    for latent, rows in read.items():
        frame = 4 * latent
        assert np.array_equal(rows.states[:, :3], table[:, frame, :3])
        assert np.array_equal(rows.states[:, 3:5], angles[latent])
        assert np.array_equal(rows.states[:, 5], recorded[:, frame, 5])
        # a lone client: only its own slot has a position track
        assert np.array_equal(rows.states[1:, :3], recorded[1:, frame, :3])
        assert rows.source.tolist() == [OWN] + [RECORDED] * 9
        assert rows.drawn is None and rows.status == "predicted"
    own = np.stack([read[f].states[0, :3] for f in range(1, 33)])
    assert not np.allclose(own, recorded[0, 4:133:4, :3])  # the client's predicted positions
    # until its block's positions are read, a row holds the position of the latent frame before
    # the block: only the rows of a committed block are final
    assert sorted(early) == list(range(5, 33))
    for latent, rows in early.items():
        before = read[latent - 1 - (latent - 1) % BLOCK].states[0, :3]
        assert np.array_equal(rows.states[0, :3], before)
        assert not np.array_equal(rows.states[0, :3], read[latent].states[0, :3])


def test_a_slot_without_media_is_hidden(synthetic, monkeypatch, tmp_path):
    world = sw.make_world(tmp_path / "world", clients=(0,))
    index = Path(world["media_index"])
    rows = [line for line in index.read_text().splitlines() if json.loads(line)["player_slot"] != 7]
    index.write_text("".join(row + "\n" for row in rows))
    sw.patch_ticks(monkeypatch, world["tables"])
    rollout = _rollout(_config(synthetic, world, tmp_path / "out"), _messages())

    first = rollout.field_rows(0)
    assert first.source.tolist() == [OWN] + [RECORDED] * 6 + [HIDDEN] + [RECORDED] * 2
    assert first.states[7].tolist() == [0.0] * 6
    for latent in (-1, 33):
        with pytest.raises(ValueError, match=f"latent frame {latent} of a rollout of 33 latent"):
            rollout.field_rows(latent)
