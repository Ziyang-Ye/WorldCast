"""A client's scene state (Sec. 3.3): blocks published into its memory bank, the other clients'
withdrawals followed, and retrieval.

A client's :class:`~worldcast.scene_state.bank.MemoryBank` holds its own entries (at most ``B``) and
a copy of every other client's, which changes only by following that client's published
withdrawals. docs/inference.md, "The block loop", gives the order of the calls per block.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from worldcast.config.inference import MEMORY_BOUND
from worldcast.data.camera import as_float64
from worldcast.data.latents import BLOCK
from worldcast.data.memory_frames import MemoryBlock
from worldcast.modeling.depth_head import DepthFn

from .bank import MemoryBank, MemoryEntry, RetrieveResult, Uid
from .geometry import axial_from_log_depth, view_points

__all__ = [
    "BlockLatentsFn",
    "CopyMismatchError",
    "FollowStats",
    "SceneState",
    "Step",
    "StepSource",
]

#: ``(block, t_target) -> [4, 48, 24, 42]`` float32, the latents another client published for an
#: admitted block (``worldcast.engine.inference.world_state.WorldState.block_latents``).
BlockLatentsFn = Callable[[MemoryBlock, int], np.ndarray]


class Step(Protocol):
    """A client's step record, published before it generates a block: its withdrawals and its
    entries.

    Attributes:
        t_target (int): source-frame time of the block.
        withdrawn (Sequence[int]): ``t_first`` of the own entries withdrawn since its last record.
        resident (Sequence[int]): ``t_first`` of the own entries it holds.
    """

    t_target: int
    withdrawn: Sequence[int]
    resident: Sequence[int]


class StepSource(Protocol):
    """Where a client reads the other clients' step records
    (``worldcast.engine.inference.world_state.WorldState``)."""

    def read_steps(self, media_id: str, *, upto: int) -> Sequence[Step]:
        """Every step record of ``media_id`` with ``t_target <= upto``, oldest first."""
        ...


class CopyMismatchError(RuntimeError):
    """A client's copy of another client's entries differs from the entries that client holds."""


@dataclass
class FollowStats:
    """What :meth:`SceneState.follow_withdrawals` did over a session (``follow`` of
    ``client.json``).

    Attributes:
        steps_applied (int): step records of the other clients applied.
        withdrawals_applied (int): withdrawals that removed an entry from this client's copy.
        withdrawals_not_held (int): withdrawals of an entry the copy did not hold.
        withdrawals_not_held_skipped (int): of those, the blocks this client never admitted.
        withdrawals_not_held_unpublished (int): of those, the blocks without a surface pixel here,
            which the bank never stored.
        copy_checks (int): comparisons of a copy with the entries its owner's record lists.
        copy_equal (int): the copy held exactly the listed entries.
        copy_equal_modulo_empty (int): it held them but for the blocks without a surface pixel
            here.
    """

    steps_applied: int = 0
    withdrawals_applied: int = 0
    withdrawals_not_held: int = 0
    withdrawals_not_held_skipped: int = 0
    withdrawals_not_held_unpublished: int = 0
    copy_checks: int = 0
    copy_equal: int = 0
    copy_equal_modulo_empty: int = 0


def _cameras(c2w: torch.Tensor | np.ndarray) -> np.ndarray:
    return as_float64(c2w).reshape(-1, 4, 4)


class SceneState:
    """One client's scene state.

    Run one client per process: numpy reprojections under a multithreaded BLAS are not
    bit-reproducible across threads of one process.

    The blocks are :class:`~worldcast.data.memory_frames.MemoryBlock` records in a list that only
    grows; a block's position in it is its id in the bank (:attr:`MemoryEntry.block`).

    Args:
        client (str): this client's media id (the owner of the entries it publishes).
        tans (tuple[float, float]): ``(tan_h, tan_v)``, the field of view of every camera of
            retrieval (the client's unscoped one, stored float32 as in the paper's runs).
        depth_fn (DepthFn): the depth head.
        block_latents (BlockLatentsFn): the latents of another client's admitted block.
        bound (int): ``B``, own entries kept (paper: 64).
    """

    def __init__(
        self,
        *,
        client: str,
        tans: tuple[float, float],
        depth_fn: DepthFn,
        block_latents: BlockLatentsFn,
        bound: int = MEMORY_BOUND,
    ) -> None:
        self.client = str(client)
        self.tans = np.asarray(tans, dtype=np.float32).reshape(1, 2)
        self.depth_fn = depth_fn
        self.block_latents = block_latents
        self.bank = MemoryBank(client=self.client, bound=int(bound))
        #: Positions in the block list of the blocks published, from the watermark on: the first
        #: block that is not published; every block before it is.
        self._published: set[int] = set()
        self.watermark = 0
        #: ``(owner, t_first)`` of the other clients' entries held, and of their blocks without a
        #: surface pixel, until their owner withdraws them (an own entry leaves through the bound).
        self._uid_at: dict[tuple[str, int], Uid] = {}
        self._empty: set[tuple[str, int]] = set()
        #: ``t_first`` of the own entries withdrawn since the last step record.
        self._withdrawn: list[int] = []
        #: Per other client, the ``t_target`` of its step records already applied.
        self._followed: dict[str, set[int]] = {}
        #: What following the other clients' withdrawals did so far.
        self.follow_stats = FollowStats()

    def _axial(self, latents: torch.Tensor | np.ndarray) -> np.ndarray:
        return axial_from_log_depth(self.depth_fn(latents))

    def _unpublished(
        self, blocks: Sequence[MemoryBlock], t_target: int
    ) -> Iterator[tuple[int, MemoryBlock]]:
        while self.watermark in self._published:
            self._published.remove(self.watermark)
            self.watermark += 1
        for index in range(self.watermark, len(blocks)):
            if index not in self._published and blocks[index].t_last < t_target:
                yield index, blocks[index]

    def _publish(self, index: int, block: MemoryBlock, latents: torch.Tensor | np.ndarray) -> None:
        self._published.add(index)
        entry = MemoryEntry(
            owner=block.media_id,
            block=index,
            t_first=block.t_first,
            t_last=block.t_last,
            c2w=_cameras(block.c2w),
            tans=self.tans,
            depth=self._axial(latents),
        )
        result = self.bank.publish(entry)
        if entry.owner != self.client:
            if result.published:
                self._uid_at[(entry.owner, entry.t_first)] = entry.uid
            else:
                self._empty.add((entry.owner, entry.t_first))
        self._withdrawn.extend(e.t_first for e in result.withdrawn)

    def publish_own(
        self,
        blocks: Sequence[MemoryBlock],
        *,
        t_target: int,
        own_latents: torch.Tensor | np.ndarray,
    ) -> int:
        """Publish this client's own blocks that ended before ``t_target`` into its bank.

        An entry's depth is read off its own four latent frames, one depth call per block.

        Args:
            blocks (Sequence[MemoryBlock]): the client's block list.
            t_target (int): source-frame time of the block about to be generated.
            own_latents (torch.Tensor | np.ndarray): the client's latents ``[N, 48, 24, 42]``.

        Returns:
            int: blocks published (those without a surface pixel included).
        """
        count = 0
        for index, block in self._unpublished(blocks, int(t_target)):
            if block.media_id == self.client:
                self._publish(index, block, own_latents[block.f0 : block.f0 + BLOCK])
                count += 1
        return count

    def publish_others(self, blocks: Sequence[MemoryBlock], *, t_target: int) -> int:
        """Copy the other clients' admitted blocks that ended before ``t_target`` into the bank.

        Call after :meth:`publish_own` and the admission of this block.

        Returns:
            int: blocks copied (those without a surface pixel included).

        Raises:
            RuntimeError: an own block that :meth:`publish_own` has not published.
        """
        count = 0
        for index, block in self._unpublished(blocks, int(t_target)):
            if block.media_id == self.client:
                raise RuntimeError(
                    f"own block {index} (t_last {block.t_last}) reached publish_others: the own"
                    " block is published before the step record, and the other clients' after the"
                    " wait"
                )
            self._publish(index, block, self.block_latents(block, int(t_target)))
            count += 1
        return count

    def drain_withdrawals(self) -> tuple[list[int], list[int]]:
        """This client's step record ``(withdrawn, resident)``: the sorted ``t_first`` of the own
        entries withdrawn since the last call, and of the own entries held now."""
        withdrawn = sorted(self._withdrawn)
        self._withdrawn = []
        return withdrawn, sorted(e.t_first for e in self.bank.resident(self.client))

    def follow_withdrawals(
        self,
        steps: StepSource,
        *,
        others: Iterable[str],
        t_target: int,
        skipped: Iterable[tuple[str, int]] = (),
    ) -> None:
        """Apply every other client's withdrawals up to ``t_target`` and check the copy of its
        entries against its record.

        Args:
            steps (StepSource): the shared world state.
            others (Iterable[str]): every other client whose blocks this client may hold.
            t_target (int): this block's time; records up to it are applied, the one at it is
                checked.
            skipped (Iterable[tuple[str, int]]): ``(client, t_first)`` of published blocks this
                client never admitted.

        Raises:
            CopyMismatchError: the copy of a client's entries differs from the entries its record
                at ``t_target`` lists (but for the blocks never admitted, and possibly those
                without a surface pixel here).
        """
        t_target = int(t_target)
        stats = self.follow_stats
        skipped = {(str(owner), int(t_first)) for owner, t_first in skipped}
        for other in sorted(str(client) for client in others):
            followed = self._followed.setdefault(other, set())
            records = steps.read_steps(other, upto=t_target)
            for record in records:
                if int(record.t_target) in followed:
                    continue
                for t_first in record.withdrawn:
                    block = (other, int(t_first))
                    uid = self._uid_at.pop(block, None)
                    if uid is not None and self.bank.withdraw(uid):
                        stats.withdrawals_applied += 1
                    elif block in skipped:
                        stats.withdrawals_not_held_skipped += 1
                    elif block in self._empty:
                        self._empty.remove(block)
                        stats.withdrawals_not_held_unpublished += 1
                    else:
                        stats.withdrawals_not_held += 1
                followed.add(int(record.t_target))
                stats.steps_applied += 1
            followed.intersection_update(int(record.t_target) for record in records)
            current = [record for record in records if int(record.t_target) == t_target]
            if not current:
                continue
            theirs = {int(t_first) for t_first in current[0].resident}
            theirs -= {t_first for owner, t_first in skipped if owner == other}
            empty = {t_first for owner, t_first in self._empty if owner == other}
            held = {e.t_first for e in self.bank.resident(other)}
            stats.copy_checks += 1
            if held == theirs:
                stats.copy_equal += 1
            elif held == theirs - empty:
                stats.copy_equal_modulo_empty += 1
            else:
                raise CopyMismatchError(
                    f"t_target {t_target}: this client's copy of {other!r} holds {len(held)}"
                    f" entries and that client's record lists {len(theirs)}; only here"
                    f" {sorted(held - theirs)[:8]}, only there {sorted(theirs - held)[:8]}"
                )

    def retrieve(
        self,
        *,
        next_c2w: torch.Tensor | np.ndarray,
        recent_c2w: torch.Tensor | np.ndarray,
        recent_latents: torch.Tensor | np.ndarray,
        t_target: int,
    ) -> RetrieveResult:
        """Retrieve at most one memory entry for the block at ``t_target``.

        The recent surface is the depth of all recent latent frames read in one call: the depth head
        reads each latent frame with its neighbours inside a call, so call boundaries are part of
        the numbers.

        Args:
            next_c2w (torch.Tensor | np.ndarray): ``[4, 4, 4]`` the block's cameras (the recorded
                ones in Table 3).
            recent_c2w (torch.Tensor | np.ndarray): ``[R, 4, 4]`` cameras of the recent latent
                frames (R = 12 from latent frame 25 on).
            recent_latents (torch.Tensor | np.ndarray): ``[R, 48, 24, 42]`` the recent latent
                frames.
            t_target (int): source-frame time of the block; only entries with ``t_last < t_target``
                are candidates, the client's own included.

        Returns:
            RetrieveResult: ``entry`` None means no memory frames.
        """
        recent = _cameras(recent_c2w)
        recent_points = np.zeros((0, 3))
        if len(recent):
            if len(recent_latents) != len(recent):
                raise ValueError(
                    f"retrieval needs one recent latent frame per recent camera ({len(recent)}),"
                    f" got {len(recent_latents)}"
                )
            points, hit = view_points(recent, self.tans, self._axial(recent_latents))
            recent_points = points[hit]
        return self.bank.retrieve(
            next_c2w=_cameras(next_c2w),
            next_tans=self.tans,
            recent_points=recent_points,
            t_target=t_target,
        )
