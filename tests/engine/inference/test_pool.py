"""The pool's admission of peer blocks."""

import torch

from worldcast.engine.inference import pool as wp

STRIDE, BLOCK = 8, 4


def _new_store(root, media, **kw):
    return wp.LocalDirPool(root, client=media, stride=STRIDE, cell=f"cell-{media}", **kw)


# ------------------------------------------------------------------------------------------- admission
def _cand_at(dead=()):
    def f(slot, media, ws, f0):
        if (str(media), int(f0)) in dead:
            return None, "dead"
        c2w = torch.eye(4).repeat(BLOCK, 1, 1)
        c2w[:, 0, 3] = float(f0) + 100.0 * slot
        return (
            dict(
                media_id=str(media),
                slot=int(slot),
                window_start=int(ws),
                f0=int(f0),
                orig_first=int(ws) + STRIDE * int(f0),
                orig_last=int(ws) + STRIDE * (int(f0) + BLOCK - 1),
                c2w=c2w,
                own_window=False,
            ),
            "",
        )

    return f


def test_admission_waits_for_the_block_and_the_step_record(tmp_path):
    pool = _new_store(tmp_path, "R")
    calls = {}

    def fake_wait(peers, **kw):
        calls.update(kw, peers=list(peers))
        return 0.0

    pool.wait_for_peers = fake_wait
    pb = wp.PeerBlocks(ego_media="R", ego_slot=0, sources={"W": 1}, candidate_at=_cand_at())
    pb.admit(pool, t_target=200, peers=("W",), max_wait_s=900.0)
    assert calls == dict(peers=["W"], need_orig_last=192, need_step=200, max_wait_s=900.0)
