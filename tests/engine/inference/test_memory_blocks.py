"""The blocks a client knows: its own, and the other clients' blocks it admits."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.engine.inference.support import block_latents, directory_world_state
from tests.player_state.support import PRIOR, flat_depth, standing_round
from worldcast.data.latents import BLOCK, FIRST_TARGET, block_span, source_frame
from worldcast.data.memory_frames import MemoryBlock
from worldcast.engine.inference.memory_blocks import LiveMemoryBlocks, MemoryBlocks
from worldcast.engine.inference.messages import MessagePart, MessageWorldState
from worldcast.engine.inference.world_state import LockstepTimeoutError, PublishedBlock
from worldcast.modeling.state_model import DISPLACEMENT_UNIT_U
from worldcast.player_state import ClosedLoop

#: The other clients of the round and their player slots.
OWNERS = {"A": 1, "C": 2}
#: The closed-loop round: the owner A and the readers B and C, by player slot.
ROUND_MEDIA = {0: "A", 1: "B", 2: "C"}
#: Where they stand at the round start (feet, u).
START_XYZ = [[0.0, 0.0, 0.0], [0.0, 800.0, 0.0], [0.0, -800.0, 0.0]]
#: The source frame of the closed-loop round's latent frame 0.
START_FRAME = 16
#: Latent frames of the closed-loop round: the first six blocks, then blocks 25 and 29.
LATENT_FRAMES = 33
#: How far every client of the closed-loop round walks along +x per latent frame, u.
WALK_U = 50.0


def block_at(slot: int, window_start: int, f0: int) -> MemoryBlock | None:
    """Every player's block, except the dead player 2's."""
    if slot == 2:
        return None
    c2w = torch.eye(4).repeat(BLOCK, 1, 1)
    c2w[:, 0, 3] = float(f0) + 100.0 * slot
    owner = {slot: media_id for media_id, slot in OWNERS.items()}[slot]
    return MemoryBlock.at(owner, slot, window_start, f0, c2w)


def test_admission_waits_for_the_other_clients(tmp_path):
    reader = directory_world_state(tmp_path, "B")
    blocks = MemoryBlocks(client="B", owners=OWNERS, block_at=block_at)
    with pytest.raises(LockstepTimeoutError, match=r"\['A'\] have not published"):
        blocks.admit(reader, t_target=40, others=("A",), max_wait_s=0.0)
    assert blocks.admit(reader, t_target=40, others=(), max_wait_s=0.0) == 0  # no lockstep


def test_a_block_is_admitted_once_its_causal_cut_passed(tmp_path):
    owner, reader = directory_world_state(tmp_path, "A"), directory_world_state(tmp_path, "B")
    blocks = MemoryBlocks(client="B", owners=OWNERS, block_at=block_at)
    owner.publish_block(window_start=0, f0=1, latents=block_latents(0))  # ends at source frame 32
    assert blocks.admit(reader, t_target=32, others=(), max_wait_s=0.0) == 0
    assert blocks.admit(reader, t_target=40, others=(), max_wait_s=0.0) == 1
    assert blocks.admit(reader, t_target=48, others=(), max_wait_s=0.0) == 0  # admitted once
    assert [(b.media_id, b.f0, b.t_last) for b in blocks.blocks] == [("A", 1, 32)]


def test_blocks_are_admitted_in_slot_order(tmp_path):
    reader = directory_world_state(tmp_path, "B")
    owners = {"D": 3, "A": 1}

    def at(slot: int, window_start: int, f0: int) -> MemoryBlock:
        owner = {slot: media_id for media_id, slot in owners.items()}[slot]
        return MemoryBlock.at(owner, slot, window_start, f0, torch.eye(4).repeat(BLOCK, 1, 1))

    for owner, f0 in (("D", 1), ("A", 5), ("A", 1)):
        published = directory_world_state(tmp_path, owner)
        published.publish_block(window_start=0, f0=f0, latents=block_latents(f0))
    blocks = MemoryBlocks(client="B", owners=owners, block_at=at)
    assert blocks.admit(reader, t_target=80, others=(), max_wait_s=0.0) == 3
    assert [(b.media_id, b.f0) for b in blocks.blocks] == [("A", 1), ("A", 5), ("D", 1)]


def test_a_published_block_without_a_reader_block_is_skipped(tmp_path):
    reader = directory_world_state(tmp_path, "B")
    dead = directory_world_state(tmp_path, "C")
    dead.publish_block(window_start=0, f0=1, latents=block_latents(0))  # ends at source frame 32
    blocks = MemoryBlocks(client="B", owners=OWNERS, block_at=block_at)
    # skipped when first seen, before its causal cut has passed
    assert blocks.admit(reader, t_target=32, others=(), max_wait_s=0.0) == 0
    assert blocks.skipped == {("C", 8)}
    assert blocks.admit(reader, t_target=80, others=(), max_wait_s=0.0) == 0
    assert blocks.skipped == {("C", 8)} and blocks.blocks == []


def test_own_blocks_are_the_clients():
    blocks = MemoryBlocks(client="B", owners=OWNERS, block_at=block_at)
    own = MemoryBlock.at("B", 0, 0, 1, torch.eye(4).repeat(BLOCK, 1, 1))
    blocks.add_own(own)
    assert blocks.blocks == [own]
    with pytest.raises(ValueError, match="add_own got a block of 'A', not 'B'"):
        blocks.add_own(block_at(1, 0, 1))
    with pytest.raises(ValueError, match="they must not contain this client"):
        MemoryBlocks(client="A", owners=OWNERS, block_at=block_at)


class WalkingStateModel:
    """Stands in for the state model: every client is :data:`WALK_U` further along +x at each
    latent frame, by its place and its motion estimate alike, so Eq. (4) puts it there."""

    def read(self, latents, latent_frames):
        f = np.asarray(latent_frames, np.float64)[:, None]
        place = WALK_U * f * np.array([1.0, 0.0, 0.0])
        motion = np.tile(np.float32([WALK_U / DISPLACEMENT_UNIT_U, 0.0, 0.0]), (len(f), 1))
        return place, motion


class ClosedLoopClient:
    """One client of the closed-loop round on the shared world state ``root`` that follows the
    clients ``others``: its closed loop, the blocks of ``others`` it knows, keyed where their owners
    generated them (the block loop's ``block_at``), and its exchange before each block."""

    def __init__(self, root: Path, slot: int, others: tuple[str, ...]) -> None:
        batch = standing_round(START_XYZ, latents=LATENT_FRAMES, client_slot=slot)
        recorded = batch["player_states"]
        self.media_id = ROUND_MEDIA[slot]
        self.world_state = directory_world_state(root, self.media_id)
        self.closed = ClosedLoop.build(
            batch,
            reader=WalkingStateModel(),
            client=self.media_id,
            client_slot=slot,
            round_media=ROUND_MEDIA,
            others=others,
            latent_frames=LATENT_FRAMES,
            start_frame=START_FRAME,
            prior=PRIOR,
            depth_fn=flat_depth(1100.0),
            world_state=self.world_state,
        )

        def block_at(slot: int, window_start: int, f0: int) -> MemoryBlock:
            block = MemoryBlock.at(ROUND_MEDIA[slot], slot, window_start, f0, None)
            return self.closed.as_generated(block, recorded)

        owners = {m: p for p, m in ROUND_MEDIA.items() if m in others}
        self.blocks = MemoryBlocks(client=self.media_id, owners=owners, block_at=block_at)

    def exchange(self, s: int) -> int:
        """The exchange before block ``s``, in the block loop's order: publish the own positions,
        admit (from the first window on), take the other clients' positions. Returns the number
        of blocks admitted."""
        self.closed.publish_position(s, torch.zeros(LATENT_FRAMES, 1, 1, 1))
        admitted = 0
        if s >= FIRST_TARGET:
            t = source_frame(START_FRAME, s)
            admitted = self.blocks.admit(self.world_state, t_target=t, others=(), max_wait_s=0.0)
        self.closed.read_positions(s)
        return admitted


def test_a_block_seen_before_its_causal_cut_is_keyed_at_admission(tmp_path):
    # B and C follow A alone: without lockstep no client reads a position before it is published
    owner = ClosedLoopClient(tmp_path, 0, others=())
    early, late = (ClosedLoopClient(tmp_path, slot, others=("A",)) for slot in (1, 2))
    for s in (1, 5, 9, 13, 17, 21):
        for c in (owner, early, late):
            c.exchange(s)
    owner.exchange(25)
    assert late.exchange(25) == 0
    owner.world_state.publish_block(window_start=START_FRAME, f0=25, latents=block_latents(25))
    # A runs ahead of B: B sees A's block 25 before its table holds the block's frames
    assert early.exchange(25) == 0
    owner.exchange(29)
    assert early.exchange(29) == late.exchange(29) == 1
    keyed_early, keyed_late = early.blocks.blocks[0], late.blocks.blocks[0]
    assert (keyed_early.media_id, keyed_early.f0) == ("A", 25)
    assert torch.equal(keyed_early.c2w, keyed_late.c2w)
    # A's position at latent frame 24, held over block 25 (no control held), not the 1000 u of
    # latent frame 20 that the table held before it read A's positions for block 25
    assert keyed_early.c2w[:, 0, 3].tolist() == [1200.0, 1200.0, 1200.0, 1200.0]


# ----------------------------------------------------------------------- a session without an end
#: The newest blocks of each client a live reader's world state keeps.
RETAIN = 4
#: The source frame of the peer's round start.
PEER_START = 64


def peer_block(world_state: MessageWorldState, k: int) -> None:
    """The peer's ``k``-th block as it arrives: its ``live`` part, then its ``scene`` part."""
    f0 = 1 + BLOCK * k
    t_first, t_last = block_span(PEER_START, f0)
    times = (np.arange(16) + 16 * k + 1) / 16.0
    part = MessagePart(
        "peer",
        "live",
        f0=f0,
        times=times,
        rows=np.zeros((16, 5), np.float32),
        buttons=np.zeros(16, np.uint16),
        turn=np.zeros((16, 2), np.float32),
        weapon=np.zeros(16, np.uint8),
    )
    block = PublishedBlock("peer", PEER_START, f0, t_first, t_last, "0" * 64)
    world_state.receive(part)
    world_state.receive(MessagePart("peer", "scene", block=block, latents=block_latents(k)))
    world_state.refresh()


def live_reader() -> tuple[MessageWorldState, LiveMemoryBlocks]:
    """A reader of the peer, whose blocks it knows by the ``live`` part of the peer's world state
    (the demo's :meth:`~worldcast.player_state.live.LiveLoop.block_at`)."""
    world_state = MessageWorldState(client="me", poll_s=0.01, retain=RETAIN)

    def block_at(slot: int, window_start: int, f0: int) -> MemoryBlock | None:
        if world_state.live_part("peer", f0) is None:
            return None
        return MemoryBlock.at("peer", slot, window_start, f0, torch.eye(4).repeat(BLOCK, 1, 1))

    return world_state, LiveMemoryBlocks(client="me", owners={"peer": 1}, block_at=block_at)


def t_of(k: int) -> int:
    """The causal cut after the peer's ``k``-th block: the source frame of its next block."""
    return block_span(PEER_START, 1 + BLOCK * (k + 1))[0]


def test_a_live_reader_keeps_each_admitted_blocks_live_part_for_its_entry():
    world_state, known = live_reader()
    for k in range(3):
        peer_block(world_state, k)
    assert known.admit(world_state, t_target=t_of(1), others=(), max_wait_s=0.0) == 2
    assert [b.f0 for b in known.blocks] == [1, 5]
    assert sorted(known.parts) == [("peer", 1), ("peer", 5)]
    assert known.parts[("peer", 5)] is world_state.live_part("peer", 5)


def test_a_block_whose_part_left_before_it_was_admitted_is_expired_not_left_pending():
    """A peer ``RETAIN`` blocks ahead of its reader: the world state keeps its newest blocks, so the
    oldest ones it has seen but not admitted are gone before their causal cut passes. Without
    ``keep`` their keys stay pending for ever."""
    world_state, known = live_reader()
    for k in range(RETAIN):
        peer_block(world_state, k)
    known.admit(world_state, t_target=t_of(-1), others=(), max_wait_s=0.0)
    assert len(known._pending) == RETAIN and not known.blocks
    for k in range(RETAIN, 2 * RETAIN + 2):  # the peer goes on: the oldest parts leave
        peer_block(world_state, k)
    known.keep(world_state, set(), 0)
    held = {("peer", block.t_first) for block in world_state.published_blocks("peer")}
    assert len(held) == RETAIN and known._pending <= held and known._seen <= held
    assert known.expired == RETAIN  # the four it had seen, all older than the world state's four
    assert known.admit(world_state, t_target=t_of(2 * RETAIN + 1), others=(), max_wait_s=0.0) == 4
    assert [b.f0 for b in known.blocks] == [
        1 + BLOCK * k for k in range(RETAIN + 2, 2 * RETAIN + 2)
    ]
    assert known.vanished == 0 and all(block is not None for block in known.blocks)


def test_a_block_whose_part_is_gone_at_its_admission_is_dropped_and_counted():
    world_state, known = live_reader()
    for k in range(2):
        peer_block(world_state, k)
    known.admit(world_state, t_target=t_of(-1), others=(), max_wait_s=0.0)
    del world_state._entry_live[("peer", 1)], world_state._live[("peer", 1)]
    assert known.admit(world_state, t_target=t_of(1), others=(), max_wait_s=0.0) == 1
    assert (known.vanished, [b.f0 for b in known.blocks]) == (1, [5])
    assert sorted(known.parts) == [("peer", 5)]


def test_a_block_the_bank_no_longer_holds_is_forgotten_with_its_latents_and_part():
    """Below the scene state's watermark a block that no entry refers to is ``None`` in the list
    (the entries refer to blocks by index), with its latents and ``live`` part; a block the bank
    holds, and one not yet published, are kept."""
    world_state, known = live_reader()
    for k in range(4):
        peer_block(world_state, k)
    known.admit(world_state, t_target=t_of(3), others=(), max_wait_s=0.0)
    for block in known.blocks:
        known.latents[("peer", block.f0)] = torch.zeros(BLOCK, 1)
    kept = dict(known.parts)
    known.keep(world_state, {1}, 3)  # blocks 0-2 published, the bank holds the entry of block 1
    assert [None if b is None else b.f0 for b in known.blocks] == [None, 5, None, 13]
    assert sorted(known.parts) == [("peer", 5), ("peer", 13)] == sorted(known.latents)
    assert known.parts[("peer", 5)] is kept[("peer", 5)]
    known.keep(world_state, set(), 4)  # nothing held any more, every block published
    assert known.blocks == [None] * 4 and not known.parts and not known.latents
