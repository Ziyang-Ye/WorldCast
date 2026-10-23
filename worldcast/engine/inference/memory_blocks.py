"""The blocks a client may retrieve its memory frames from: its own and the other clients'."""

from collections.abc import Callable, Mapping, Sequence

import torch

from worldcast.data.memory_frames import MemoryBlock
from worldcast.player_state.live import LivePart

from .world_state import WorldState

__all__ = ["BlockAtFn", "LiveMemoryBlocks", "MemoryBlocks"]

#: ``(slot, window_start, f0) -> block | None``: the block of the player at ``slot`` at a published
#: key, with the cameras the reader keys it at
#: (:func:`worldcast.data.memory_frames.recorded_block`), or ``None`` when the recording does not
#: cover the block or the player is dead in it. :meth:`MemoryBlocks.admit` asks it when it first
#: sees a published block, to skip a ``None`` at once, and again when it admits the block; whether
#: it answers ``None`` must not change between the two.
BlockAtFn = Callable[[int, int, int], MemoryBlock | None]


class MemoryBlocks:
    """The finished blocks one client knows: its own and the other clients' blocks it admitted.

    Append-only; a block's index in :attr:`blocks` is how the scene state refers to it
    (:attr:`worldcast.scene_state.bank.MemoryEntry.block`). The order breaks retrieval ties: the
    first six blocks, then per block the other clients' blocks admitted at that step in ``(slot,
    media, t_first)`` order, then the client's own new block.

    Args:
        client (str): this client's media id.
        owners (Mapping[str, int]): ``{media_id: slot}`` of the other players whose blocks may be
            admitted.
        block_at (BlockAtFn): the reader's block for a published key.
    """

    def __init__(self, *, client: str, owners: Mapping[str, int], block_at: BlockAtFn) -> None:
        self.client = str(client)
        self.owners = {str(m): int(s) for m, s in owners.items()}
        if self.client in self.owners:
            raise ValueError("owners are the other clients: they must not contain this client")
        self.block_at = block_at
        self.blocks: list[MemoryBlock] = []
        #: ``(media, t_first)`` of published blocks never admitted (no block at the key).
        self.skipped: set[tuple[str, int]] = set()
        self._pending: set[tuple[str, int]] = set()
        self._seen: set[tuple[str, int]] = set()

    def add_own(self, block: MemoryBlock) -> None:
        """Append this client's generated block."""
        if block.media_id != self.client:
            raise ValueError(f"add_own got a block of {block.media_id!r}, not {self.client!r}")
        self.blocks.append(block)

    def admit(
        self, world_state: WorldState, *, t_target: int, others: Sequence[str], max_wait_s: float
    ) -> int:
        """Wait for ``others`` (lockstep), then admit the published blocks that ended before
        ``t_target``.

        A block is admitted at its owner's key (:meth:`WorldState.wait_for_others` has the rule).
        The reader's block for it (``block_at``) is made at the step that admits it, not when the
        block is first seen: in the closed loop its cameras come from the reader's player-state
        table, which holds the block's frames only from the step before, and an owner that runs
        ahead publishes a block one step before its causal cut has passed.

        Args:
            world_state (WorldState): the shared world state.
            t_target (int): the source frame of the block about to be generated.
            others (Sequence[str]): the clients to wait for; none without lockstep.
            max_wait_s (float): the longest lockstep wait, s.

        Returns:
            int: the number of blocks admitted.
        """
        world_state.wait_for_others(others, t_target=t_target, max_wait_s=max_wait_s)
        world_state.refresh()
        n = 0
        for media, slot in sorted(self.owners.items(), key=lambda owner: (owner[1], owner[0])):
            for published in world_state.published_blocks(media):
                key = (media, published.t_first)
                if key not in self._seen:
                    self._seen.add(key)
                    if self.block_at(slot, published.window_start, published.f0) is None:
                        self.skipped.add(key)
                        continue
                    self._pending.add(key)
                if key not in self._pending or published.t_last >= t_target:
                    continue  # skipped, admitted, or its causal cut has not passed yet
                self._pending.discard(key)
                self.blocks.append(self.block_at(slot, published.window_start, published.f0))
                n += 1
        return n


class LiveMemoryBlocks(MemoryBlocks):
    """The known blocks of a client that plays for as long as it likes: bounded.

    A block that the memory bank holds no entry of any more, and that the scene state has
    published (below its watermark), is replaced by ``None`` in :attr:`blocks` at every
    :meth:`keep` (the scene state and the bank refer to blocks by index, so the list keeps its
    length), with its latents and its ``live`` part: a block keeps those for as long as its entry
    can be retrieved, and the message world state does not keep them that long. The keys of the
    blocks seen and pending are pruned to the blocks the world state still holds.

    Attributes:
        latents (dict[tuple[str, int], torch.Tensor]): ``[4, 48, 24, 42]`` the latents of the
            blocks, by ``(media_id, f0)``.
        parts (dict[tuple[str, int], LivePart]): their ``live`` parts.
        vanished (int): blocks whose ``live`` part was gone when they were to be admitted: dropped.
        expired (int): pending blocks whose ``scene`` part left the world state before their
            causal cut passed: dropped, they cannot be admitted any more.
    """

    def __init__(self, *, client: str, owners: Mapping[str, int], block_at: BlockAtFn) -> None:
        super().__init__(client=client, owners=owners, block_at=block_at)
        self.latents: dict[tuple[str, int], torch.Tensor] = {}
        self.parts: dict[tuple[str, int], LivePart] = {}
        self.vanished = self.expired = 0
        self._published: set[int] = set()  # the indices below the watermark that hold a block
        self._published_to = 0  # the watermark of the newest keep

    def admit(
        self, world_state: WorldState, *, t_target: int, others: Sequence[str], max_wait_s: float
    ) -> int:
        """:meth:`MemoryBlocks.admit`, keeping each admitted block's ``live`` part; a block whose
        part is gone is dropped and counted in :attr:`vanished` (the blocks added by this call are
        referred to by nothing yet, so the indices stay)."""
        before = len(self.blocks)
        admitted = super().admit(
            world_state, t_target=t_target, others=others, max_wait_s=max_wait_s
        )
        added = [block for block in self.blocks[before:] if block is not None]
        self.vanished += admitted - len(added)
        self.blocks[before:] = added
        for block in added:
            self.parts[(block.media_id, block.f0)] = world_state.live_part(block.media_id, block.f0)
        return len(added)

    def keep(self, world_state: WorldState, resident: set[int], watermark: int) -> None:
        """Forget what nothing can retrieve any more.

        Args:
            world_state (WorldState): the message world state, for the blocks it still holds.
            resident (set[int]): the indices of the blocks the bank holds entries of.
            watermark (int): the scene state's first block not yet published
                (:attr:`~worldcast.scene_state.SceneState.watermark`).
        """
        self._published.update(range(self._published_to, watermark))
        self._published_to = watermark
        for index in self._published - resident:
            block = self.blocks[index]
            for held in (self.latents, self.parts):
                held.pop((block.media_id, block.f0), None)
            self.blocks[index] = None
        self._published &= resident
        held = {(m, p.t_first) for m in self.owners for p in world_state.published_blocks(m)}
        self._seen &= held
        expired = self._pending - held
        self._pending -= expired
        self.expired += len(expired)
