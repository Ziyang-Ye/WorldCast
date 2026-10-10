"""One client's rollout, block by block, on the CPU with a tiny random model."""

import numpy as np
import pytest
import torch

import tests.engine.inference.support as sw
from worldcast.data.latents import BLOCK, FIRST_TARGET
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference.fast import FastGenerator
from worldcast.engine.inference.loading import ClientModels, load_window
from worldcast.engine.inference.rollout import BlockRead, Rollout
from worldcast.engine.inference.serving import ServingOptions
from worldcast.modeling.wan22.attention import sdpa_attention
from worldcast.player_state import field_builder
from worldcast.sampling.sampler import Sampler
from worldcast.sampling.window import KV_CACHE_LATENTS


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


def _world_state(tmp_path):
    return sw.directory_world_state(tmp_path / "world_state", sw.media_id(0), poll_s=0.01)


def test_a_rollout_block_by_block_gives_the_clients_latents(synthetic, reference, tmp_path):
    world_state = _world_state(tmp_path)
    sent: list[tuple[str, tuple]] = []
    for name in ("publish_block", "publish_step"):

        def publish(*args, _publish=getattr(world_state, name), _name=name, **kwargs):
            sent.append((_name, args))
            return _publish(*args, **kwargs)

        setattr(world_state, name, publish)
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
    # every block went out as a memory entry; the blocks that read the scene state published a
    # step record first
    kinds = [name for name, _ in sent]
    assert kinds.count("publish_block") == 6 + sw.MAX_BLOCKS
    assert kinds.count("publish_step") == sw.MAX_BLOCKS


def test_the_methods_of_a_rollout_are_called_in_order(synthetic, monkeypatch, tmp_path):
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    rollout = _rollout(_config(synthetic), _world_state(tmp_path))
    with pytest.raises(RuntimeError, match="the rollout is not open: call open\\(\\) first"):
        rollout.denoise(1, lambda: None)
    rollout.open()
    with pytest.raises(RuntimeError, match="block 25 was not prepared: call prepare_next\\(21\\)"):
        rollout.denoise(FIRST_TARGET, lambda: None)
