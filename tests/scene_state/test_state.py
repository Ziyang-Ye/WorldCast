"""One client's scene state (Sec. 3.3): its own blocks published, the other clients' blocks copied
and their withdrawals followed, and retrieval."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from worldcast.data.memory_frames import MemoryBlock
from worldcast.scene_state import CopyMismatchError, SceneState
from worldcast.scene_state.geometry import NPIX
from worldcast.scene_state.state import FollowStats

TAN = (1.3333, 0.75)
#: A camera at the origin that looks along +z, four times: a block that does not move.
STILL = torch.eye(4).repeat(4, 1, 1)
#: Pixels of a column of the depth grid over a block's four cameras.
COLUMN = 4 * 24


class WallDepth:
    """Stands in for the depth head. A latent frame is ``(first column, stop column)``: it shows a
    wall 300 u ahead in these columns of the depth grid and no surface elsewhere."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, latents) -> np.ndarray:
        latents = np.asarray(latents)
        self.calls.append(len(latents))
        log_depth = np.full((len(latents), 4, 24, 42), np.log(4096.0), np.float32)
        for frame, (first, stop) in enumerate(latents.astype(int)):
            log_depth[frame, :, :, first:stop] = np.log(300.0)
        return log_depth


def showing(*columns: tuple[int, int]) -> np.ndarray:
    """Latents ``[4 n, 2]`` of ``n`` blocks, each showing the wall in its ``(first, stop)``
    columns."""
    return np.repeat(np.asarray(columns, np.float32), 4, 0)


def block(owner: str, f0: int) -> MemoryBlock:
    """``owner``'s still block of the latent frames ``f0 .. f0 + 3`` (source frames ``8 f0 .. 8 f0 +
    24``)."""
    return MemoryBlock.at(owner, "AB".index(owner), 0, f0, STILL)


def scene_state(depth: WallDepth, *, bound: int = 64, published=None) -> SceneState:
    """Client A's scene state; the other clients published ``{(media_id, f0): latents}``."""
    published = {} if published is None else published
    return SceneState(
        client="A",
        tans=TAN,
        depth_fn=depth,
        block_latents=lambda block, t_target: published[block.media_id, block.f0],
        bound=bound,
    )


@dataclass
class Record:
    t_target: int
    withdrawn: tuple[int, ...]
    resident: tuple[int, ...]


class Steps:
    """The other clients' step records, in memory."""

    def __init__(self, **records: list[Record]) -> None:
        self.records = records
        self.reads: list[tuple[str, int]] = []

    def read_steps(self, media_id: str, *, upto: int) -> list[Record]:
        self.reads.append((media_id, upto))
        return [record for record in self.records.get(media_id, []) if record.t_target <= upto]


# ------------------------------------------------------------------------------------ own blocks
def test_the_field_of_view_is_stored_in_float32():
    """As in the paper's runs."""
    scene = scene_state(WallDepth())
    assert scene.tans.dtype == np.float32 and scene.tans.shape == (1, 2)
    assert scene.tans.astype(np.float64).tolist() == [[1.3332999944686890, 0.75]]
    assert scene.bank.client == "A" and scene.bank.bound == 64


def test_a_client_publishes_its_blocks_that_ended_before_the_next_block():
    depth = WallDepth()
    scene = scene_state(depth)
    blocks = [block("A", 1), block("A", 5)]  # source frames 8 .. 32 and 40 .. 64
    latents = np.concatenate([np.zeros((1, 2), np.float32), showing((0, 21), (30, 42))])
    assert scene.publish_own(blocks, t_target=32, own_latents=latents) == 0
    assert scene.publish_own(blocks, t_target=64, own_latents=latents) == 1  # 32 < 64, not 64
    assert scene.publish_own(blocks, t_target=65, own_latents=latents) == 1
    assert scene.publish_own(blocks, t_target=65, own_latents=latents) == 0  # nothing twice
    assert depth.calls == [4, 4]  # one depth call per block, on its own four latent frames
    first, second = scene.bank.resident("A")
    # an entry's id is the block's position in the list; its depth is that of its latent frames
    assert (first.uid, first.t_first, first.t_last, first.n_hit) == (("A", 0), 8, 32, 21 * COLUMN)
    assert (second.uid, second.t_first, second.t_last, second.n_hit) == (("A", 1), 40, 64, 1152)
    np.testing.assert_allclose(first.depth.reshape(4, 24, 42)[:, :, :21], 300.0, rtol=1e-6)
    assert first.c2w.dtype == np.float64 and (first.tans == scene.tans.astype(np.float64)).all()


def test_the_withdrawals_beyond_the_bound_go_into_the_step_record_once():
    scene = scene_state(WallDepth(), bound=2)
    blocks = [block("A", 1), block("A", 5), block("A", 9), block("A", 13)]
    latents = np.concatenate(
        [np.zeros((1, 2), np.float32), showing((0, 21), (0, 21), (30, 42), (0, 0))]
    )
    assert scene.publish_own(blocks, t_target=72, own_latents=latents) == 2
    assert scene.drain_withdrawals() == ([], [8, 40])
    # a third own entry is one too many: of the two that show one wall, the older is withdrawn
    assert scene.publish_own(blocks, t_target=104, own_latents=latents) == 1
    assert scene.drain_withdrawals() == ([8], [40, 72])
    assert scene.drain_withdrawals() == ([], [40, 72])
    # a block without a surface pixel counts as published and is not stored
    assert scene.publish_own(blocks, t_target=136, own_latents=latents) == 1
    assert scene.drain_withdrawals() == ([], [40, 72])
    assert [entry.block for entry in scene.bank.resident("A")] == [1, 2]
    # an own entry leaves through the bound and follows no record: nothing of it is kept
    assert scene._uid_at == {} and scene._empty == set()


# ------------------------------------------------------------------------ the other clients' blocks
def test_another_clients_block_is_copied_with_the_latents_that_client_published():
    depth = WallDepth()
    asked = []
    published = {("B", 1): showing((0, 10)), ("B", 5): showing((10, 20)), ("B", 9): showing((5, 6))}

    def block_latents(block, t_target):
        asked.append((block.media_id, block.f0, t_target))
        return published[block.media_id, block.f0]

    scene = SceneState(client="A", tans=TAN, depth_fn=depth, block_latents=block_latents, bound=1)
    blocks = [block("A", 1), block("B", 1), block("B", 5), block("B", 9)]
    with pytest.raises(RuntimeError, match=r"own block 0 \(t_last 32\) reached publish_others"):
        scene.publish_others(blocks, t_target=72)
    own = np.concatenate([np.zeros((1, 2), np.float32), showing((0, 21))])
    assert scene.publish_own(blocks, t_target=72, own_latents=own) == 1
    assert scene.publish_others(blocks, t_target=72) == 2  # B's block 9 ends at 96
    assert asked == [("B", 1, 72), ("B", 5, 72)] and depth.calls == [4, 4, 4]
    held = scene.bank.resident("B")
    assert [(e.uid, e.t_first, e.n_hit) for e in held] == [
        (("B", 1), 8, 10 * COLUMN),
        (("B", 2), 40, 10 * COLUMN),
    ]
    # the bound holds for the client's own entries; another client's leave when it withdraws them
    assert scene.publish_others(blocks, t_target=104) == 1
    assert len(scene.bank.resident("B")) == 3 and scene.drain_withdrawals() == ([], [8])


def followed(*f0s: int, empty: tuple[int, ...] = ()) -> SceneState:
    """Client A with its copies of B's blocks ``f0s``; those in ``empty`` show no surface."""
    published = {("B", f0): showing((0, 0) if f0 in empty else (0, 10)) for f0 in f0s}
    scene = scene_state(WallDepth(), published=published)
    blocks = [block("B", f0) for f0 in f0s]
    assert scene.publish_others(blocks, t_target=1000) == len(f0s)
    return scene


def held(scene: SceneState) -> list[int]:
    return [entry.t_first for entry in scene.bank.resident("B")]


def test_a_client_follows_the_other_clients_withdrawals():
    scene = followed(1, 5, 9)
    steps = Steps(
        B=[
            Record(104, withdrawn=(), resident=(8, 40, 72)),
            Record(136, withdrawn=(40,), resident=(8, 72)),
            Record(168, withdrawn=(8,), resident=(72,)),
        ]
    )
    scene.follow_withdrawals(steps, others=["C", "B"], t_target=104)
    assert held(scene) == [8, 40, 72] and steps.reads == [("B", 104), ("C", 104)]
    assert scene.follow_stats == FollowStats(steps_applied=1, copy_checks=1, copy_equal=1)
    scene.follow_withdrawals(steps, others=["B"], t_target=136)
    assert held(scene) == [8, 72]
    scene.follow_withdrawals(steps, others=["B"], t_target=136)  # a record is applied once
    assert held(scene) == [8, 72]
    assert scene.follow_stats == FollowStats(
        steps_applied=2, withdrawals_applied=1, copy_checks=3, copy_equal=3
    )
    # the records up to the block's time are applied; a time without a record is not checked
    scene.follow_withdrawals(steps, others=["B"], t_target=200)
    assert held(scene) == [72]
    assert scene.follow_stats == FollowStats(
        steps_applied=3, withdrawals_applied=2, copy_checks=3, copy_equal=3
    )


def test_what_is_kept_of_another_clients_entries_goes_with_their_withdrawal():
    scene = followed(1, 5, 9, 13, empty=(13,))
    assert set(scene._uid_at) == {("B", 8), ("B", 40), ("B", 72)} and scene._empty == {("B", 104)}
    steps = Steps(
        B=[
            Record(136, withdrawn=(40, 104), resident=(8, 72)),
            Record(168, withdrawn=(8,), resident=(72,)),
        ]
    )
    scene.follow_withdrawals(steps, others=["B"], t_target=168)
    assert set(scene._uid_at) == {("B", 72)} and scene._empty == set()
    assert scene.follow_stats.withdrawals_applied == 2
    assert scene.follow_stats.withdrawals_not_held_unpublished == 1


def test_the_records_applied_are_remembered_only_while_the_world_state_still_has_them():
    scene = followed(1)
    steps = Steps(
        B=[Record(104, withdrawn=(), resident=(8,)), Record(136, withdrawn=(), resident=(8,))]
    )
    scene.follow_withdrawals(steps, others=["B"], t_target=136)
    assert scene._followed == {"B": {104, 136}}
    steps.records["B"] = steps.records["B"][1:]  # the world state has kept the newest record only
    scene.follow_withdrawals(steps, others=["B"], t_target=136)
    assert scene._followed == {"B": {136}} and scene.follow_stats.steps_applied == 2


def test_the_copy_is_checked_against_the_other_clients_record():
    steps = Steps(B=[Record(104, withdrawn=(), resident=(8, 72))])
    scene = followed(1, 5, 9)
    with pytest.raises(CopyMismatchError) as error:
        scene.follow_withdrawals(steps, others=["B"], t_target=104)
    assert str(error.value) == (
        "t_target 104: this client's copy of 'B' holds 3 entries and that client's record lists 2;"
        " only here [40], only there []"
    )
    with pytest.raises(CopyMismatchError, match=r"only here \[\], only there \[200\]"):
        followed(1, 5, 9).follow_withdrawals(
            Steps(B=[Record(104, withdrawn=(), resident=(8, 40, 72, 200))]),
            others=["B"],
            t_target=104,
        )


def test_blocks_never_admitted_or_without_a_surface_are_no_mismatch():
    # B lists the block at 200, which this client never admitted, then withdraws it and a block
    # this client never held
    steps = Steps(
        B=[
            Record(104, withdrawn=(), resident=(8, 40, 72, 200)),
            Record(136, withdrawn=(200, 300), resident=(8, 40, 72)),
        ]
    )
    scene = followed(1, 5, 9)
    scene.follow_withdrawals(steps, others=["B"], t_target=104, skipped=[("B", 200), ("C", 8)])
    scene.follow_withdrawals(steps, others=["B"], t_target=136, skipped=[("B", 200)])
    assert held(scene) == [8, 40, 72]
    assert scene.follow_stats == FollowStats(
        steps_applied=2,
        withdrawals_not_held=1,
        withdrawals_not_held_skipped=1,
        copy_checks=2,
        copy_equal=2,
    )
    # B's block at 40 has no surface pixel here: B's record may list it or not
    counted = []
    for resident in ((8, 40, 72), (8, 72)):
        scene = followed(1, 5, 9, empty=(5,))
        steps = Steps(B=[Record(104, withdrawn=(), resident=resident)])
        scene.follow_withdrawals(steps, others=["B"], t_target=104)
        assert held(scene) == [8, 72]
        counted.append((scene.follow_stats.copy_equal, scene.follow_stats.copy_equal_modulo_empty))
    assert counted == [(0, 1), (1, 0)]
    # and its withdrawal by B is nothing to apply
    scene = followed(1, 5, 9, empty=(5,))
    steps = Steps(B=[Record(104, withdrawn=(40,), resident=(8, 72))])
    scene.follow_withdrawals(steps, others=["B"], t_target=104)
    assert held(scene) == [8, 72]
    assert scene.follow_stats == FollowStats(
        steps_applied=1, withdrawals_not_held_unpublished=1, copy_checks=1, copy_equal=1
    )


# -------------------------------------------------------------------------------------- retrieval
def test_retrieval_reads_the_entry_of_any_client_that_covers_the_most_missing_pixels():
    depth = WallDepth()
    scene = scene_state(depth, published={("B", 1): showing((30, 42))})
    blocks = [block("A", 1), block("B", 1)]
    own = np.concatenate([np.zeros((1, 2), np.float32), showing((0, 21))])
    scene.publish_own(blocks, t_target=72, own_latents=own)
    scene.publish_others(blocks, t_target=72)
    depth.calls.clear()
    # the twelve recent latent frames saw the columns 0 .. 14, read in one depth call
    result = scene.retrieve(
        next_c2w=STILL,
        recent_c2w=STILL.repeat(3, 1, 1),
        recent_latents=showing((0, 15), (0, 15), (0, 15)),
        t_target=72,
    )
    assert depth.calls == [12]
    assert result.entry.uid == ("B", 1) and result.entry is scene.bank.resident("B")[0]
    assert (result.coverage, result.n_missing, result.n_candidates) == (12 * COLUMN, 27 * COLUMN, 2)
    # without a recent context every pixel is missing: the client's own, larger entry is read
    none = torch.zeros(0, 4, 4)
    result = scene.retrieve(next_c2w=STILL, recent_c2w=none, recent_latents=None, t_target=72)
    assert depth.calls == [12] and result.entry.uid == ("A", 0)
    assert (result.coverage, result.n_missing, result.n_candidates) == (21 * COLUMN, 4 * NPIX, 2)
    # an entry that has not ended before the block is no candidate
    result = scene.retrieve(next_c2w=STILL, recent_c2w=none, recent_latents=None, t_target=32)
    assert result.entry is None and (result.coverage, result.n_candidates) == (0, 0)
    with pytest.raises(
        ValueError, match=r"one recent latent frame per recent camera \(12\), got 8"
    ):
        scene.retrieve(
            next_c2w=STILL,
            recent_c2w=STILL.repeat(3, 1, 1),
            recent_latents=showing((0, 15), (0, 15)),
            t_target=72,
        )
