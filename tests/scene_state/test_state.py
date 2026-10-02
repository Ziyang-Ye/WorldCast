"""The scene state of one client."""

import pytest
import torch

import tests.scene_state.scene_world as sw
from worldcast.scene_state.state import SceneState

STRIDE, BLOCK, B, N, RECENT = 8, 4, 3, 45, 12


CLIENTS = ("m0", "m1", "m2")


def _block(k, f0, trajs, **kw):
    return dict(
        media_id=CLIENTS[k],
        slot=k,
        window_start=0,
        f0=f0,
        orig_first=STRIDE * f0,
        orig_last=STRIDE * (f0 + BLOCK - 1),
        c2w=torch.as_tensor(trajs[k][f0 : f0 + BLOCK], dtype=torch.float32),
        **kw,
    )


def test_the_own_block_must_be_ingested_before_the_peers():
    trajs = {0: sw.trajectory(0, N)}
    scene = SceneState(
        client="m0",
        tans=[sw.TAN],
        depth_fn=sw.StubDepth(trajs).depth_grid,
        peer_latents=lambda b, t: None,
    )
    blocks = [_block(0, 1, trajs, kind="generated")]
    with pytest.raises(RuntimeError, match="ingest_peers"):
        scene.ingest_peers(blocks, t_target=STRIDE * 5)
    out = torch.zeros((N,) + sw.LAT_SHAPE)
    out[1:5] = sw.gen_block(0, 1, None)
    assert scene.ingest_own(blocks, t_target=STRIDE * 5, own_latents=out) == 1
    assert scene.ingest_peers(blocks, t_target=STRIDE * 5) == 0
    with pytest.raises(ValueError, match="recent latents"):
        scene.retrieve(
            query_c2w=trajs[0][5:9], recent_c2w=trajs[0][1:5], recent_latents=None, t_target=40
        )
