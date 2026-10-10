"""The shared world state on a directory: publishing, the rules of a read, the step records, the
positions and the lockstep waits."""

from pathlib import Path

import numpy as np
import pytest

from tests.engine.inference.support import block_key, block_latents, directory_world_state
from worldcast.engine.inference import world_state as ws
from worldcast.engine.inference.world_state import PublishedBlock, StepRecord, WorldState
from worldcast.utils.fingerprints import sha256_float32


@pytest.fixture
def clients(tmp_path):
    """``clients(media_id)``: the world state as that client sees it, shared with the clients made
    before it."""

    def client(media_id: str) -> WorldState:
        return directory_world_state(tmp_path, media_id, poll_s=0.5)

    return client


class Clock:
    """Stands in for ``time`` in the lockstep wait: a sleep advances it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(ws, "time", clock)
    return clock


def test_the_poll_period_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="poll_s must be positive"):
        directory_world_state(tmp_path, "A", poll_s=0.0)


# ------------------------------------------------------------------------------------- publishing
def test_a_published_block_carries_its_span_and_its_sha(clients):
    owner = clients("A")
    owner.publish_block(window_start=16, f0=5, latents=block_latents(0))
    # latent frames 5-8 of a rollout from source frame 16, 8 source frames per latent frame
    expected = PublishedBlock("A", 16, 5, 56, 80, sha256_float32(block_latents(0)))
    assert owner.published_blocks("A") == [expected]
    assert owner.progress("A") == 80 and owner.progress("B") is None
    with pytest.raises(ValueError, match="4 latents"):
        owner.publish_block(window_start=16, f0=9, latents=block_latents(0)[:3])


def test_a_published_block_is_a_copy_and_so_is_a_read(clients):
    owner, reader = clients("A"), clients("B")
    latents = block_latents(0)
    owner.publish_block(window_start=0, f0=1, latents=latents)
    latents += 1.0
    reader.refresh()
    first = reader.block_latents(block_key("A", 0, 1), 33)
    assert first.dtype == np.float32 and np.array_equal(first, block_latents(0))
    first += 1.0
    assert np.array_equal(reader.block_latents(block_key("A", 0, 1), 33), block_latents(0))


def test_a_block_is_served_at_its_key_after_its_causal_cut(clients):
    owner, reader = clients("A"), clients("B")
    owner.publish_block(window_start=0, f0=1, latents=block_latents(0))
    reader.refresh()
    # the block's last latent frame (4) is at source frame 32
    with pytest.raises(ws.WorldStateError, match="ends at 32, not before the reader's causal cut"):
        reader.block_latents(block_key("A", 0, 1), 32)
    assert np.array_equal(reader.block_latents(block_key("A", 0, 1), 33), block_latents(0))
    with pytest.raises(ws.WorldStateError, match="no block was published at that key"):
        reader.block_latents(block_key("A", 0, 2), 33)
    with pytest.raises(ws.WorldStateError, match="no block was published at that key"):
        reader.block_latents(block_key("A", 8, 0), 33)  # the same source frames, another key


def test_step_records_and_the_done_status(clients):
    owner, reader = clients("A"), clients("B")
    owner.publish_step(t_target=40, withdrawn=[16, 8], resident=[24])
    reader.refresh()
    assert reader.has_step("A", 40) and not reader.has_step("A", 48)
    assert reader.read_steps("A", upto=40) == [StepRecord("A", 40, (8, 16), (24,))]
    assert reader.read_steps("A", upto=39) == []
    assert reader.done_status("A") is None and not reader.is_done("A")
    owner.mark_done(status=ws.DONE_FAILED, note="boom")
    reader.refresh()
    assert reader.done_status("A") == "failed" and reader.is_done("A")
    with pytest.raises(ValueError, match="status must be 'ok' or 'failed'"):
        owner.mark_done(status="maybe")


def test_positions_round_trip_exactly(clients):
    owner, reader = clients("A"), clients("B")
    xyz = np.random.default_rng(0).normal(0, 1e3, (4, 3))
    owner.publish_position(64, [4, 5, 6, 7], xyz)
    latent_frames, read = reader.read_position("A", 64)
    assert latent_frames == [4, 5, 6, 7] and read.dtype == np.float64 and np.array_equal(read, xyz)
    assert reader.read_position("A", 96) is None and reader.read_position("C", 64) is None


# ------------------------------------------------------------------------------------ the lockstep
def test_the_lockstep_waits_for_the_blocks_and_the_step_record(clients, clock):
    reader, owner = clients("B"), clients("A")
    wait = dict(t_target=40, max_wait_s=2.0)
    with pytest.raises(ws.LockstepTimeoutError, match=r"after 2.0 s, \['A'\] have not published"):
        reader.wait_for_others(["A"], **wait)
    assert clock.slept == [0.5] * 4  # polled every poll_s until max_wait_s
    owner.publish_block(window_start=0, f0=1, latents=block_latents(0))  # ends at 32 = 40 - 8
    with pytest.raises(ws.LockstepTimeoutError, match="the step record of block 40"):
        reader.wait_for_others(["A"], **wait)
    owner.publish_step(t_target=40, withdrawn=[], resident=[8])
    assert reader.wait_for_others(["A"], **wait) is None
    assert reader.wait_for_others([], **wait) is None  # no lockstep: nothing to wait for
    assert (reader.wait_stats.n_waits, reader.wait_stats.seconds_total) == (1, 0.0)


def test_a_wait_counts_its_seconds(clients, clock):
    reader, owner = clients("B"), clients("A")
    polls = clock.sleep

    def sleep(seconds: float) -> None:
        polls(seconds)
        if clock.now == 1.0:  # the other client publishes while this one waits
            owner.publish_position(40, [1], np.zeros((1, 3)))

    clock.sleep = sleep
    reader.wait_for_positions(["A"], t_target=40, max_wait_s=5.0)
    assert (reader.wait_stats.n_waits, reader.wait_stats.seconds_total) == (1, 1.0)


@pytest.mark.parametrize("wait", ["wait_for_others", "wait_for_positions"])
def test_both_waits_skip_a_finished_client_and_fail_on_a_failed_one(clients, clock, wait):
    reader = clients("B")
    clients("A").mark_done(status=ws.DONE_OK)
    getattr(reader, wait)(["A"], t_target=40, max_wait_s=1.0)  # finished: not waited for
    clients("C").mark_done(status=ws.DONE_FAILED, note="out of memory")
    with pytest.raises(ws.ClientFailedError, match=r"\['C'\] marked themselves failed"):
        getattr(reader, wait)(["A", "C"], t_target=40, max_wait_s=1.0)
    assert clock.slept == [] and reader.wait_stats.n_waits == 1


def test_the_world_state_loads_without_torch():
    """The shared world state is numpy only: a reader of it needs no model stack."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import worldcast.engine.inference.world_state\n"
        "import worldcast.engine.inference.directory\n"
        "assert 'torch' not in sys.modules\n"
    )
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
