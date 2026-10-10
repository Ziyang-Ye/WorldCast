"""The shared world state on a directory: its files, what a reader accepts of them, and several
clients in lockstep on it."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from tests.engine.inference.support import block_key, block_latents, directory_world_state
from worldcast.data.latents import BLOCK, SOURCE_FRAMES_PER_LATENT
from worldcast.engine.inference import world_state as ws
from worldcast.utils.fingerprints import sha256_float32


def test_a_block_is_a_record_and_its_latents(tmp_path):
    owner = directory_world_state(tmp_path, "A")
    owner.publish_block(window_start=16, f0=5, latents=block_latents(0))
    window = tmp_path / "A" / "win_000016"
    assert sorted(path.name for path in window.iterdir()) == ["blk_000005.json", "blk_000005.npy"]
    assert json.loads((window / "blk_000005.json").read_text()) == dict(
        media_id="A",
        window_start=16,
        f0=5,
        t_first=56,
        t_last=80,
        latents_sha256=sha256_float32(block_latents(0)),
    )
    stored = np.load(window / "blk_000005.npy")
    assert stored.dtype == np.float32 and np.array_equal(stored, block_latents(0))


def test_one_key_holds_one_block(tmp_path):
    owner = directory_world_state(tmp_path, "A")
    owner.publish_block(window_start=0, f0=1, latents=block_latents(0))
    owner.publish_block(window_start=0, f0=1, latents=block_latents(0))  # the same bytes: a no-op
    with pytest.raises(ws.WorldStateError, match="already holds a DIFFERENT block"):
        owner.publish_block(window_start=0, f0=1, latents=block_latents(1))
    assert len(owner.published_blocks("A")) == 1


def test_a_block_without_its_record_is_not_published(tmp_path):
    reader = directory_world_state(tmp_path, "B")
    window = tmp_path / "A" / "win_000000"
    window.mkdir(parents=True)
    np.save(window / "blk_000001.npy", block_latents(0))
    np.save(window / ".pub-partial.npy", block_latents(0))  # a file still being written
    reader.refresh()
    assert reader.published_blocks("A") == []


def test_a_read_is_checked_against_the_owners_sha(tmp_path):
    owner, reader = directory_world_state(tmp_path, "A"), directory_world_state(tmp_path, "B")
    owner.publish_block(window_start=0, f0=1, latents=block_latents(0))
    reader.refresh()
    path = tmp_path / "A" / "win_000000" / "blk_000001.npy"
    np.save(path, block_latents(1))
    with pytest.raises(ws.WorldStateError, match="sha256 of the latents read"):
        reader.block_latents(block_key("A", 0, 1), 33)
    np.save(path, block_latents(0)[:3])
    with pytest.raises(ws.WorldStateError, match=r"latents are \(3, 2, 3, 5\)"):
        reader.block_latents(block_key("A", 0, 1), 33)


def test_one_client_publishes_one_step_record_per_block(tmp_path):
    owner = directory_world_state(tmp_path, "A")
    owner.publish_step(t_target=40, withdrawn=[16, 8], resident=[24])
    owner.publish_step(t_target=40, withdrawn=[8, 16], resident=[24])  # the same record: a no-op
    with pytest.raises(ws.WorldStateError, match="already holds a DIFFERENT step record"):
        owner.publish_step(t_target=40, withdrawn=[], resident=[24])
    assert json.loads((tmp_path / "A" / "steps" / "step_000040.json").read_text()) == dict(
        media_id="A", t_target=40, withdrawn=[8, 16], resident=[24]
    )


def test_the_position_and_the_done_files(tmp_path):
    owner = directory_world_state(tmp_path, "A")
    owner.publish_position(64, [4, 5], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]]))
    assert json.loads((tmp_path / "A" / "state" / "state_000064.json").read_text()) == dict(
        latent_frames=[4, 5], xyz=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]]
    )
    owner.mark_done(status=ws.DONE_FAILED, note="out of memory")
    assert json.loads((tmp_path / "A" / "DONE.json").read_text()) == dict(
        media_id="A", status="failed", note="out of memory"
    )
    assert sorted(path.name for path in (tmp_path / "A").iterdir()) == ["DONE.json", "state"]


# ------------------------------------------------------------------ several clients in lockstep
def _run_blocks(
    root: Path,
    client: str,
    others: list[str],
    *,
    fail_at: int | None = None,
    max_wait_s: float = 120.0,
) -> None:
    """A client's blocks 1, 5, .. 21 in lockstep: per block its step record, the wait for the other
    clients, then the block. Like ``run_client`` it marks itself failed when it raises."""
    world_state = directory_world_state(root, client, poll_s=0.002)
    try:
        world_state.publish_block(window_start=0, f0=1, latents=block_latents(1))
        for f0 in range(1 + BLOCK, 25, BLOCK):
            if f0 == fail_at:
                raise RuntimeError(f"{client}: injected failure at block {f0}")
            t_target = SOURCE_FRAMES_PER_LATENT * f0
            world_state.publish_step(t_target=t_target, withdrawn=[], resident=[])
            world_state.wait_for_others(others, t_target=t_target, max_wait_s=max_wait_s)
            world_state.publish_block(window_start=0, f0=f0, latents=block_latents(f0))
    except BaseException:
        world_state.mark_done(status=ws.DONE_FAILED, note="test client raised")
        raise
    world_state.mark_done(status=ws.DONE_OK)


def _others(client: str) -> list[str]:
    return [other for other in "ABC" if other != client]


def test_a_client_that_fails_stops_the_others(tmp_path):
    with ThreadPoolExecutor(max_workers=3) as pool:
        runs = {
            client: pool.submit(
                _run_blocks,
                tmp_path,
                client,
                _others(client),
                fail_at=13 if client == "C" else None,
            )
            for client in "ABC"
        }
        with pytest.raises(RuntimeError, match="injected failure"):
            runs["C"].result(timeout=300)
        for client in "AB":
            with pytest.raises(ws.ClientFailedError, match="C"):
                runs[client].result(timeout=300)
    reader = directory_world_state(tmp_path, "R")
    assert [reader.done_status(client) for client in "ABC"] == [ws.DONE_FAILED] * 3
    # C's blocks before the failure stay published: the last one ends at latent frame 12
    assert reader.progress("C") == SOURCE_FRAMES_PER_LATENT * 12


def test_a_client_that_never_arrives_times_out(tmp_path):
    """C never starts: the first client to time out raises and marks itself failed; the other stops
    on its own timeout or on that failure. Neither continues without C."""
    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = [
            pool.submit(_run_blocks, tmp_path, client, _others(client), max_wait_s=0.3)
            for client in "AB"
        ]
        for run in runs:
            with pytest.raises((ws.LockstepTimeoutError, ws.ClientFailedError)) as info:
                run.result(timeout=300)
            errors.append(info.value)
    assert any(isinstance(e, ws.LockstepTimeoutError) and "C" in str(e) for e in errors)
    # no block after the first: it ends at latent frame 4
    assert directory_world_state(tmp_path, "R").progress("A") == SOURCE_FRAMES_PER_LATENT * 4
