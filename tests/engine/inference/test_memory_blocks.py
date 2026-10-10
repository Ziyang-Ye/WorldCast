"""The blocks a client knows: its own, and the other clients' blocks it admits."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.engine.inference.support import block_latents, directory_world_state
from tests.player_state.support import PRIOR, flat_depth, standing_round
from worldcast.data.latents import BLOCK, FIRST_TARGET, source_frame
from worldcast.data.memory_frames import MemoryBlock
from worldcast.engine.inference.memory_blocks import MemoryBlocks
from worldcast.engine.inference.world_state import LockstepTimeoutError
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
