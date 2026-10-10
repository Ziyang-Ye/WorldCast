"""The memory bank (Sec. 3.3; App. "Scene state in detail"): publish and withdraw, the reprojection
cache, and retrieval."""

import numpy as np
import pytest

import tests.scene_state.support as sw
from worldcast.scene_state import MemoryBank, MemoryEntry
from worldcast.scene_state.bank import missing_coverage, pick_best
from worldcast.scene_state.geometry import NPIX

B = 6
CLIENT = "A"
#: A camera at the origin that looks along +z, four times: a block that does not move.
STILL = np.stack([np.eye(4)] * 4)
#: Pixels of a column of the depth grid over a block's four cameras.
COLUMN = 4 * 24


def wall(owner: str, block: int, columns: range, *, depth: float = 300.0, t_first: int = 8):
    """An entry of a still block that saw a wall ``depth`` u ahead in ``columns`` of the grid and
    nothing elsewhere."""
    grid = np.full((4, 24, 42), 4096.0)
    grid[:, :, columns.start : columns.stop] = depth
    return MemoryEntry(
        owner=owner,
        block=block,
        t_first=t_first,
        t_last=t_first + 24,
        c2w=STILL,
        tans=[sw.TAN],
        depth=grid,
    )


def recent_wall(columns: range) -> np.ndarray:
    """The points of a recent context that saw the wall in ``columns``."""
    return wall("recent", 0, columns).points


# ------------------------------------------------------------------------------ publish, withdraw
def test_an_entry_is_a_blocks_cameras_and_the_points_its_depth_shows():
    entry = wall("B", 3, range(10, 12))
    assert entry.uid == ("B", 3) and (entry.n_views, entry.n_hit) == (4, 2 * COLUMN)
    assert entry.points.shape == (2 * COLUMN, 3) and entry.points[:, 2].tolist() == [300.0] * 192
    assert entry.c2w.dtype == entry.depth.dtype == np.float64 and entry.tans.shape == (4, 2)


def test_a_client_keeps_at_most_b_entries_of_its_own_and_every_entry_of_the_others():
    bank = MemoryBank(client=CLIENT, bound=2)
    first, twin = wall(CLIENT, 0, range(0, 21)), wall(CLIENT, 1, range(0, 21), t_first=40)
    apart = wall(CLIENT, 2, range(30, 42), t_first=72)
    assert bank.publish(first).withdrawn == [] and bank.publish(twin).withdrawn == []
    assert (first.seq, twin.seq) == (0, 1)
    assert bank.covered_share(first) == bank.covered_share(twin) == 1.0  # they show one wall
    # the third own entry is one too many: the best covered entry goes, the older on a tie
    result = bank.publish(apart)
    assert result.published and result.withdrawn == [first]
    assert bank.resident(CLIENT) == [twin, apart]
    assert bank.covered_share(twin) == bank.covered_share(apart) == 0.0
    # the other clients' entries are not bounded here: their owners withdraw them
    theirs = [wall("B", b, range(0, 5), t_first=8 + 32 * b) for b in range(4)]
    assert all(bank.publish(entry).withdrawn == [] for entry in theirs)
    assert bank.resident("B") == theirs and len(bank) == 6
    assert bank.withdraw(("B", 1)) is True and bank.withdraw(("B", 1)) is False
    assert [entry.block for entry in bank.resident("B")] == [0, 2, 3]
    with pytest.raises(ValueError, match="own entries are withdrawn by its bound"):
        bank.withdraw((CLIENT, 1))
    with pytest.raises(ValueError, match=r"entry \('B', 2\) offered twice"):
        bank.publish(wall("B", 2, range(0, 5)))
    assert bank.publish(wall("B", 1, range(0, 5))).published  # withdrawn before: not held


def test_a_block_without_a_surface_is_not_stored():
    bank = MemoryBank(client=CLIENT)
    assert bank.bound == 64
    result = bank.publish(wall(CLIENT, 0, range(0, 0)))
    assert (result.published, result.withdrawn) == (False, []) and len(bank) == 0


def test_the_reprojections_are_cached_in_float32():
    """As in the paper's runs: a depth 2e-6 u beyond the 24 u tolerance is within it once cached."""
    bank = MemoryBank(client=CLIENT)
    near = wall(CLIENT, 0, range(21, 22), depth=100.0)
    far = wall(CLIENT, 1, range(21, 22), depth=124.000002, t_first=40)
    bank.publish(near)
    bank.publish(far)
    assert bank.covered_share(near) == 1.0  # float32(124.000002) is 124.0
    assert bank.covered_share(far) == 0.0  # 100 is 24.000002 from the 124.000002 it stores


def _entries(owner: str, k: int, n: int, seed: int) -> list[MemoryEntry]:
    """``n`` consecutive blocks of a player that walks through the room of the support module."""
    rng = np.random.default_rng(seed)
    traj = sw.trajectory(k, 4 * n + 1)
    out = []
    for b in range(n):
        cams = traj[1 + 4 * b : 5 + 4 * b]
        depth = np.stack([sw.room_depth(c) for c in cams])
        depth = depth * (1.0 + 0.03 * rng.standard_normal(depth.shape))
        t0 = 32 * b + 8
        out.append(
            MemoryEntry(
                owner=owner,
                block=b,
                t_first=t0,
                t_last=t0 + 24,
                c2w=cams,
                tans=[sw.TAN],
                depth=depth,
            )
        )
    return out


@pytest.fixture
def reprojections(monkeypatch):
    """Counts every reprojection (an entry's points projected into another entry's cameras)."""
    calls = [0]
    zbuffers = MemoryEntry.zbuffers

    def counted(self, points):
        calls[0] += 1
        return zbuffers(self, points)

    monkeypatch.setattr(MemoryEntry, "zbuffers", counted)
    return calls


def test_publishing_costs_at_most_2b_reprojections_and_withdrawal_none(reprojections):
    bank = MemoryBank(client=CLIENT, bound=B)
    own, others = _entries(CLIENT, 0, 3 * B, 0), _entries("P", 1, 3 * B, 1)
    n_withdrawn = 0
    for mine, theirs in zip(own, others):
        held = len(bank.resident(CLIENT))
        before = reprojections[0]
        n_withdrawn += len(bank.publish(mine).withdrawn)  # the withdrawal included
        assert reprojections[0] - before == 2 * held <= 2 * B
        before = reprojections[0]
        bank.publish(theirs)
        assert bank.withdraw(theirs.uid) and reprojections[0] == before
    assert n_withdrawn == 2 * B and len(bank.resident(CLIENT)) == B


def _fresh_share(entry: MemoryEntry, others: list[MemoryEntry]) -> float:
    """The share of ``entry``'s surface pixels that ``others`` cover, counted without a cache."""
    nearest = np.full(entry.n_views * NPIX, np.inf)
    for other in others:
        z = entry.zbuffers(other.points).reshape(-1).astype(np.float32)
        nearest = np.minimum(nearest, z.astype(np.float64))
    return float(entry.covered_by(nearest.reshape(entry.n_views, NPIX)).sum()) / entry.n_hit


def test_cached_coverage_equals_a_fresh_count():
    """After every publish and withdrawal, each own entry's cached share equals the share counted
    from scratch, and the withdrawn entry is the best covered (the older on a tie)."""
    bank = MemoryBank(client=CLIENT, bound=B)
    n_withdrawn = 0
    for entry in _entries(CLIENT, 2, 3 * B, 2):
        candidates = bank.resident(CLIENT) + [entry]
        shares = [_fresh_share(e, [f for f in candidates if f is not e]) for e in candidates]
        result = bank.publish(entry)
        n_withdrawn += len(result.withdrawn)
        if len(candidates) > B:
            best = max(range(len(candidates)), key=lambda i: (shares[i], -candidates[i].seq))
            assert result.withdrawn == [candidates[best]]
        held = bank.resident(CLIENT)
        for e in held:
            assert bank.covered_share(e) == _fresh_share(e, [f for f in held if f is not e])
    assert n_withdrawn == 2 * B


# -------------------------------------------------------------------------------------- retrieval
def _retrieve(bank: MemoryBank, recent: range = range(0, 0), t_target: int = 1000):
    return bank.retrieve(
        next_c2w=STILL, next_tans=[sw.TAN], recent_points=recent_wall(recent), t_target=t_target
    )


def test_retrieval_reads_the_entry_that_covers_the_most_missing_pixels():
    bank = MemoryBank(client=CLIENT)
    left = wall(CLIENT, 0, range(0, 21))
    strip = wall("B", 0, range(0, 10), t_first=40)
    right = wall("C", 0, range(30, 42), t_first=72)
    for entry in (left, strip, right):
        bank.publish(entry)
    # nothing recent: every pixel of the next block's four cameras is missing
    result = _retrieve(bank)
    assert result.entry is left
    assert (result.coverage, result.n_missing, result.n_candidates) == (21 * COLUMN, 4 * NPIX, 3)
    coverage, n_missing = missing_coverage(STILL, [sw.TAN], np.zeros((0, 3)), [left, strip, right])
    assert coverage.tolist() == [21 * COLUMN, 10 * COLUMN, 12 * COLUMN] and n_missing == 4 * NPIX
    # the recent context saw the columns 0 .. 14: only what it lacks counts
    result = _retrieve(bank, recent=range(0, 15))
    assert result.entry is right
    assert (result.coverage, result.n_missing) == (12 * COLUMN, 27 * COLUMN)
    coverage, _ = missing_coverage(STILL, [sw.TAN], recent_wall(range(0, 15)), [left, strip, right])
    assert coverage.tolist() == [6 * COLUMN, 0, 12 * COLUMN]


def test_retrieval_reads_nothing_when_no_entry_covers_a_missing_pixel():
    bank = MemoryBank(client=CLIENT)
    assert _retrieve(bank).entry is None and _retrieve(bank).n_candidates == 0
    bank.publish(wall("B", 0, range(0, 10)))
    seen = _retrieve(bank, recent=range(0, 12))
    assert seen.entry is None and (seen.coverage, seen.n_missing) == (0, 30 * COLUMN)
    # an entry that ends at the next block's time or later is no candidate
    assert _retrieve(bank, t_target=32).entry is None
    assert _retrieve(bank, t_target=33).entry is not None


def test_an_entry_behind_another_entrys_surface_does_not_cover():
    bank = MemoryBank(client=CLIENT)
    near = wall("B", 0, range(0, 26), depth=300.0)
    behind = wall("C", 0, range(0, 42), depth=1000.0, t_first=40)
    bank.publish(near)
    bank.publish(behind)
    # the nearest surface is the reference: the far wall counts only where the near one is absent
    coverage, _ = missing_coverage(STILL, [sw.TAN], np.zeros((0, 3)), [near, behind])
    assert coverage.tolist() == [26 * COLUMN, 16 * COLUMN]
    assert _retrieve(bank).entry is near
    # within the tolerance of the reference (the larger of 24 u and 5 %) both cover
    close = wall("D", 0, range(0, 26), depth=315.0, t_first=72)
    beyond = wall("E", 0, range(0, 26), depth=324.5, t_first=72)
    coverage, _ = missing_coverage(STILL, [sw.TAN], np.zeros((0, 3)), [near, close, beyond])
    assert coverage.tolist() == [26 * COLUMN, 26 * COLUMN, 0]


def test_a_tie_goes_to_the_entry_that_ended_later_then_to_the_one_published_later():
    bank = MemoryBank(client=CLIENT)
    early = wall("B", 0, range(0, 10), t_first=8)
    late = wall("C", 0, range(0, 10), t_first=40)
    twin = wall("D", 0, range(0, 10), t_first=40)
    for entry in (late, early, twin):
        bank.publish(entry)
    assert _retrieve(bank).entry is twin  # as late as ``late``, published after it
    bank.withdraw(twin.uid)
    assert _retrieve(bank).entry is late
    coverage = np.array([5, 5, 2])
    assert pick_best([late, early, twin], coverage) is late
    assert pick_best([late, early, twin], np.array([0, 0, 0])) is None
    assert pick_best([], np.zeros(0, np.int64)) is None
