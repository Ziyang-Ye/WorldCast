"""Scene state shared through the pool in lock-step: a failing client and a missing peer."""

import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest
import torch

import tests.scene_state.scene_world as sw
from worldcast.engine.inference.pool import (
    LocalDirPool,
    PeerBlocks,
    PeerFailedError,
    PeerTimeoutError,
)
from worldcast.scene_state.state import SceneState

STRIDE, BLOCK, B, N, RECENT = 8, 4, 5, 45, 12
PREFIX = (1, 5, 9)
CLIENTS = ("m0", "m1", "m2")
SLOTS = {"m0": 0, "m1": 1, "m2": 2, "m3": 3}  # m3 is in the pool scope and never publishes
DEAD = ("m2", 5)  # a published block every reader's candidate builder refuses
WAIT_S, POLL_S = 120.0, 0.002
TRAJS = {k: sw.trajectory(k, N) for k in range(3)}


def _c2w(k, a, b):
    return torch.as_tensor(TRAJS[k][a:b], dtype=torch.float32)


def _own(k, s):
    return dict(
        media_id=CLIENTS[k],
        slot=k,
        window_start=0,
        f0=s,
        orig_first=STRIDE * s,
        orig_last=STRIDE * (s + BLOCK - 1),
        c2w=_c2w(k, s, s + BLOCK),
    )


def _cand_at(slot, media, ws, f0):
    if (str(media), int(f0)) == DEAD:
        return None, "dead"
    return (
        dict(
            media_id=str(media),
            slot=int(slot),
            window_start=int(ws),
            f0=int(f0),
            orig_first=int(ws) + STRIDE * int(f0),
            orig_last=int(ws) + STRIDE * (int(f0) + BLOCK - 1),
            c2w=_c2w(int(slot), int(f0), int(f0) + BLOCK),
            own_window=False,
        ),
        "",
    )


def _bank(bank):
    return [(c, [(e.uid, e.seq) for e in es]) for c, es in bank.entries.items()]


def _jitter(rng):
    time.sleep(rng.uniform(0.0, 0.004))


# ------------------------------------------------------------------------------------------ clients
def _client_new(root, k, seed, *, fail_at=None, wait_s=WAIT_S):
    m = CLIENTS[k]
    rng = random.Random(seed)
    pool = LocalDirPool(root, client=m, stride=STRIDE, poll_s=POLL_S, cell=f"cell-{m}")
    blocks = PeerBlocks(
        ego_media=m,
        ego_slot=k,
        sources={p: s for p, s in SLOTS.items() if p != m},
        candidate_at=_cand_at,
    )
    scene = SceneState(
        client=m,
        tans=[sw.TAN],
        depth_fn=sw.StubDepth(TRAJS).depth_grid,
        peer_latents=pool.peer_latents,
        bound=B,
    )
    peers = [p for p in CLIENTS if p != m]
    out = torch.zeros((N,) + sw.LAT_SHAPE)
    out[0] = sw.base_latent(k, 0)
    rows = []
    try:
        for s in PREFIX:
            out[s : s + BLOCK] = sw.gen_block(k, s, None)
            pool.publish_block(
                window_start=0,
                f0=s,
                latents=out[s : s + BLOCK],
                orig_first=STRIDE * s,
                orig_last=STRIDE * (s + BLOCK - 1),
                extra={"mode": "plain"},
            )
            blocks.add_own(_own(k, s))
        for s in range(PREFIX[-1] + BLOCK, N - BLOCK + 1, BLOCK):
            T = STRIDE * s
            if s == fail_at:
                raise RuntimeError(f"{m}: injected failure at block {s}")
            _jitter(rng)
            scene.ingest_own(blocks.blocks, t_target=T, own_latents=out)
            wd, res = scene.drain_own_step()
            pool.publish_step(t_target=T, withdrawn=wd, resident=res)
            _jitter(rng)
            n0 = len(blocks.blocks)
            blocks.admit(pool, t_target=T, peers=peers, max_wait_s=wait_s)
            admitted = [(b["media_id"], b["orig_first"]) for b in blocks.blocks[n0:]]
            scene.ingest_peers(blocks.blocks, t_target=T)
            scene.follow_withdrawals(pool, peers=blocks.peers, t_target=T, skipped=blocks.skipped)
            r = scene.retrieve(
                query_c2w=_c2w(k, s, s + BLOCK),
                recent_c2w=_c2w(k, s - RECENT, s),
                recent_latents=out[s - RECENT : s],
                t_target=T,
            )
            slot = None
            if r.entry is not None:
                assert r.entry.block in blocks.eligible(t_target=T)
                b = blocks.blocks[r.entry.block]
                f0 = int(b["f0"])
                slot = (
                    out[f0 : f0 + BLOCK]
                    if b["media_id"] == m
                    else pool.fetch_block(
                        media_id=b["media_id"], window_start=b["window_start"], f0=f0, t_target=T
                    )
                )
            out[s : s + BLOCK] = sw.gen_block(k, s, slot)
            _jitter(rng)
            pool.publish_block(
                window_start=0,
                f0=s,
                latents=out[s : s + BLOCK],
                orig_first=STRIDE * s,
                orig_last=STRIDE * (s + BLOCK - 1),
                extra={"mode": "reconstituted"},
            )
            blocks.add_own(_own(k, s))
            rows.append(
                dict(
                    s=s,
                    step=(wd, res),
                    admitted=admitted,
                    read=None if r.entry is None else r.entry.uid,
                    scores=list(r.scores.items()),
                    score=r.score,
                    n_hole=r.n_hole,
                    n_candidates=r.n_candidates,
                    follow=asdict(scene.follow_stats),
                    bank=_bank(scene.bank),
                )
            )
    except BaseException:
        pool.mark_done(status="failed", note="test client raised")
        raise
    pool.mark_done(status="ok")
    return dict(rows=rows, out=out, skipped=sorted(blocks.skipped), waits=pool.wait_stats)


# ---------------------------------------------------- the new failure policy, inside a live session
# (threads are fine here: only the outcome is checked, not bits)
def test_a_client_that_fails_stops_its_peers(tmp_path):
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [
            ex.submit(_client_new, tmp_path, k, k, fail_at=(21 if k == 2 else None))
            for k in range(3)
        ]
        with pytest.raises(RuntimeError, match="injected failure"):
            futs[2].result(timeout=300)
        for f in futs[:2]:
            with pytest.raises(PeerFailedError, match="m2"):
                f.result(timeout=300)
    done = (tmp_path / "m2" / "DONE.json").read_text()
    assert '"status": "failed"' in done


def test_a_peer_that_never_arrives_times_out(tmp_path):
    """m2 never starts: the first client to time out raises (and marks itself failed); the other
    stops on either its own timeout or that failure.  Nobody continues without m2."""
    errors = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_client_new, tmp_path, k, k, wait_s=0.3) for k in range(2)]
        for f in futs:
            with pytest.raises((PeerTimeoutError, PeerFailedError)) as info:
                f.result(timeout=300)
            errors.append(info.value)
    assert any(isinstance(e, PeerTimeoutError) and "m2" in str(e) for e in errors)
