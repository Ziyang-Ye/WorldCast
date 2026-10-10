"""The blocks a client may retrieve its memory frames from: its own and the other clients'."""

from collections.abc import Callable, Mapping, Sequence

from worldcast.data.memory_frames import MemoryBlock

from .world_state import WorldState

__all__ = ["BlockAtFn", "MemoryBlocks"]

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
