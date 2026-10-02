"""The web demo's GPU adapter (``demo.worldcast_engine.WorldCastEngine``, its ``EngineProtocol``) over the engine, on
the CPU with the tiny random model: start, steps with the controls as a callable taken right before each ladder,
frames with the player's state, the peer messages, and an empty step at the end of the recording."""

import numpy as np
import pytest
import torch
import yaml

import tests.engine.realtime.synthetic_world as sw

demo_engine = pytest.importorskip("demo.worldcast_engine")
from demo.config import WorldCastEngineConfig  # noqa: E402
from demo.engine import BlockActions  # noqa: E402
from demo.library import Library, RoundStart, Seat  # noqa: E402
from worldcast.config.inference import config_to_dict  # noqa: E402


def block_actions(block: int, first: int, frames: int = 16) -> BlockActions:
    zeros = np.zeros
    return BlockActions(
        block=block,
        first_frame=first,
        times=np.arange(frames, dtype=np.float64) / 16 + first / 16,
        buttons=zeros((frames, 11), np.float32),
        turn=zeros((frames, 2), np.float32),
        camera=zeros((frames, 2), np.float32),
        weapon=np.full(frames, 2, np.int64),
        substeps=zeros((frames, 4, 13), np.float32),
        substep_valid=np.ones((frames, 4), bool),
        input_seq=np.full(frames, -1, np.int64),
    )


def test_demo_adapter_runs_the_engine(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    world = sw.make_world(tmp_path / "world", clients=(0,))
    sw.patch_ticks(monkeypatch, world["tables"])
    weights = sw.make_weights(tmp_path / "weights")
    cfg = sw.config(world, weights, tmp_path / "out", max_blocks=1)
    path = tmp_path / "inference.yaml"
    path.write_text(yaml.safe_dump(config_to_dict(cfg)))
    config = WorldCastEngineConfig(
        configs=[str(path)],
        realtime=dict(
            generator="fast",
            early_prefill=True,
            decode_overlap=True,
            decoder="none",
            lockstep=False,
        ),
        own_state="gt",
    )
    seat = Seat(seat=0, media_id=sw.media_id(0), team="T", spawn=(0.0, 0.0, 0.0, 0.0, 0.0))
    start = RoundStart(
        id="r",
        map=sw.MAP,
        label="synthetic",
        start_frame=0,
        seats=(seat,),
        match_id=sw.MATCH,
        round=sw.ROUND,
    )
    engine = demo_engine.WorldCastEngine(config, Library(rounds={"r": start}))
    state = engine.start(start, seat, peers=[])
    assert state.seat == 0
    frames, steps, taken = [], 0, []
    while True:

        def controls(block=steps, first=len(frames)):
            taken.append(block)
            return block_actions(block, first)

        got = list(engine.step(controls))
        if not got:
            break
        frames.extend(got)
        steps += 1
    assert steps == 7 and taken == list(range(7))  # 6 plain blocks + 1 reconstituted
    assert [f.index for f in frames] == list(range(len(frames))) and frames[0].state is not None
    kinds = [m.kind for m in engine.take_messages()]
    assert kinds.count("scene") == 7 and kinds.count("state") == 7 and kinds.count("step") == 1
