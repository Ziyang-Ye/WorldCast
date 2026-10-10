"""The shared world state: what the clients of a round publish and read once per block.

Every client, once per block, publishes one message and reads the latest message of every other
client (App. B): the block it finished, as a memory entry whose fp32 latents
a reader fetches to recompute the depth; the memory entries it withdrew (its step record); and, in
the closed loop, its estimated position. :class:`WorldState` is the shared world state as one
client sees it, with the lockstep wait and the rules of a read. Its storage is a directory shared
by the clients (:class:`~worldcast.engine.inference.directory.DirectoryWorldState`, the paper's
evaluation setup).
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from worldcast.data.latents import BLOCK, SOURCE_FRAMES_PER_LATENT, block_span
from worldcast.utils.fingerprints import float32_array, sha256_float32

__all__ = [
    "DONE_FAILED",
    "DONE_OK",
    "BlockKey",
    "ClientFailedError",
    "LockstepTimeoutError",
    "PublishedBlock",
    "StepRecord",
    "WaitStats",
    "WorldState",
    "WorldStateError",
]

#: The statuses of a client that publishes nothing more.
DONE_OK, DONE_FAILED = "ok", "failed"


class WorldStateError(RuntimeError):
    """A request that the protocol forbids or the data cannot satisfy."""


class LockstepTimeoutError(WorldStateError):
    """Another client did not reach the lockstep point within ``max_wait_s``."""


class ClientFailedError(WorldStateError):
    """Another client marked itself failed."""


class BlockKey(Protocol):
    """A block by its owner and its place in the owner's rollout: :class:`PublishedBlock`, or the
    reader's :class:`~worldcast.data.memory_frames.MemoryBlock` of it."""

    media_id: str
    window_start: int
    f0: int


@dataclass(frozen=True)
class PublishedBlock:
    """A block its owner published as a memory entry.

    Attributes:
        media_id (str): the owner, the client that generated the block.
        window_start (int): the source frame the owner's rollout starts at.
        f0 (int): the block's first latent frame in that rollout.
        t_first (int): the source frame of its first latent frame
            (:func:`~worldcast.data.latents.block_span`).
        t_last (int): the source frame of its last latent frame.
        latents_sha256 (str): sha256 of its float32 latents.
    """

    media_id: str
    window_start: int
    f0: int
    t_first: int
    t_last: int
    latents_sha256: str


@dataclass(frozen=True)
class StepRecord:
    """An owner's record of its block at source frame ``t_target``: the ``t_first`` of the own
    memory entries it withdrew at that block's write and of those it holds afterwards."""

    media_id: str
    t_target: int
    withdrawn: tuple[int, ...]
    resident: tuple[int, ...]


@dataclass
class WaitStats:
    """The lockstep waits of one client: how many, and their seconds in all."""

    n_waits: int = 0
    seconds_total: float = 0.0


class WorldState(ABC):
    """The shared world state as one client sees it: it publishes its own records and reads the
    other clients'.

    A subclass implements the storage (the abstract methods); the lockstep wait and the rules of a
    read (a block is served at its owner's key, after its causal cut, checked against its sha256)
    are shared.

    Args:
        client (str): this client's media id; it publishes under this name only.
        poll_s (float): the lockstep poll period, s.
    """

    def __init__(self, *, client: str, poll_s: float) -> None:
        if float(poll_s) <= 0:
            raise ValueError(f"poll_s must be positive, got {poll_s!r}")
        self.client, self.poll_s = str(client), float(poll_s)
        self.wait_stats = WaitStats()

    # -------------------------------------------------------------------------------- publishing
    def publish_block(self, *, window_start: int, f0: int, latents: ArrayLike) -> None:
        """Publish this client's finished block at its key ``(window_start, f0)``, atomically.

        Args:
            window_start (int): the source frame the client's rollout starts at.
            f0 (int): the block's first latent frame in the rollout.
            latents (ArrayLike): ``[4, C, H, W]`` clean latents, a tensor or an array; a float32
                copy is published.
        """
        a = np.array(float32_array(latents))
        if a.ndim != 4 or a.shape[0] != BLOCK:
            raise ValueError(f"a published block is {BLOCK} latents [4, C, H, W], got {a.shape}")
        t_first, t_last = block_span(window_start, f0)
        block = PublishedBlock(
            self.client, int(window_start), int(f0), t_first, t_last, sha256_float32(a)
        )
        self._store_block(block, a)

    def publish_step(
        self, *, t_target: int, withdrawn: Sequence[int], resident: Sequence[int]
    ) -> None:
        """Publish this client's step record for its block at ``t_target``.

        Args:
            t_target (int): the source frame of the block's first latent frame.
            withdrawn (Sequence[int]): ``t_first`` of the own memory entries withdrawn.
            resident (Sequence[int]): ``t_first`` of the own memory entries held afterwards.
        """
        withdrawn, resident = (tuple(sorted(int(v) for v in x)) for x in (withdrawn, resident))
        self._store_step(StepRecord(self.client, int(t_target), withdrawn, resident))

    def publish_position(
        self, t_target: int, latent_frames: Sequence[int], xyz: np.ndarray
    ) -> None:
        """Publish this client's estimated positions for its block at ``t_target`` (closed loop).

        Args:
            t_target (int): the source frame of the block's first latent frame.
            latent_frames (Sequence[int]): the latent frames the positions are estimated at.
            xyz (np.ndarray): ``[n, 3]`` positions, u.
        """
        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        self._store_position(int(t_target), [int(k) for k in latent_frames], xyz)

    def mark_done(self, *, status: str = DONE_OK, note: str = "") -> None:
        """Declare that this client publishes nothing more.

        Args:
            status (str): :data:`DONE_OK` or :data:`DONE_FAILED`.
            note (str): why it failed.
        """
        if status not in (DONE_OK, DONE_FAILED):
            raise ValueError(f"status must be {DONE_OK!r} or {DONE_FAILED!r}, got {status!r}")
        self._store_done(status, str(note))

    @abstractmethod
    def _store_block(self, block: PublishedBlock, latents: np.ndarray) -> None:
        """Store this client's block with its float32 latents."""

    @abstractmethod
    def _store_step(self, record: StepRecord) -> None:
        """Store this client's step record."""

    @abstractmethod
    def _store_position(self, t_target: int, latent_frames: list[int], xyz: np.ndarray) -> None:
        """Store this client's positions ``xyz`` ``[n, 3]`` float64 for block ``t_target``."""

    @abstractmethod
    def _store_done(self, status: str, note: str) -> None:
        """Store this client's done status."""

    # ----------------------------------------------------------------------------------- reading
    @abstractmethod
    def refresh(self) -> None:
        """Take in what the other clients have published since the last call."""

    @abstractmethod
    def published_blocks(self, media_id: str) -> list[PublishedBlock]:
        """Every block ``media_id`` published as of the last :meth:`refresh`, by ``t_first``."""

    @abstractmethod
    def read_steps(self, media_id: str, *, upto: int) -> list[StepRecord]:
        """Every step record of ``media_id`` with ``t_target <= upto``, oldest first."""

    @abstractmethod
    def has_step(self, media_id: str, t_target: int) -> bool:
        """Whether ``media_id`` has published its step record for ``t_target``."""

    @abstractmethod
    def read_position(self, media_id: str, t_target: int) -> tuple[list[int], np.ndarray] | None:
        """``(latent_frames, xyz [n, 3] float64)`` that ``media_id`` published for block
        ``t_target``, or None while it has not."""

    @abstractmethod
    def done_status(self, media_id: str) -> str | None:
        """``None`` while ``media_id`` is running, else its done status."""

    @abstractmethod
    def _read_block(self, block: PublishedBlock) -> np.ndarray:
        """The stored latents of ``block``, ``[4, C, H, W]``."""

    def progress(self, media_id: str) -> int | None:
        """The largest ``t_last`` ``media_id`` published as of the last refresh, or None."""
        return max((b.t_last for b in self.published_blocks(media_id)), default=None)

    def is_done(self, media_id: str) -> bool:
        """Whether ``media_id`` publishes nothing more."""
        return self.done_status(media_id) is not None

    def resolve(self, block: BlockKey, *, t_target: int) -> PublishedBlock:
        """The block its owner published at the key of ``block``, if it ended in time.

        Args:
            block (BlockKey): the owner and the key ``(window_start, f0)``.
            t_target (int): the source frame of the reader's block.

        Raises:
            WorldStateError: no block was published at that key (a splice of two blocks included),
                or the block does not end before ``t_target`` (the causal cut).
        """
        key = (int(block.window_start), int(block.f0))
        t_first, _ = block_span(*key)
        published = self.published_blocks(block.media_id)
        found = next((b for b in published if b.t_first == t_first), None)
        if found is None or (found.window_start, found.f0) != key:
            raise WorldStateError(
                f"{block.media_id} @ win {key[0]} f0 {key[1]}: no block was published at that"
                " key; a served block is exactly one published block at its owner's own key"
            )
        if found.t_last >= t_target:
            raise WorldStateError(
                f"{block.media_id} @ {t_first}: the block ends at {found.t_last}, not before the"
                f" reader's causal cut {t_target}"
            )
        return found

    def block_latents(self, block: BlockKey, t_target: int) -> np.ndarray:
        """The latents another client published for ``block``: a float32 ``[4, C, H, W]`` copy,
        checked against the sha256 its owner stamped; :meth:`resolve` has the rules."""
        published = self.resolve(block, t_target=int(t_target))
        latents = np.array(self._read_block(published), dtype=np.float32)
        if latents.ndim != 4 or latents.shape[0] != BLOCK:
            raise WorldStateError(
                f"{published.media_id} @ {published.t_first}: latents are {latents.shape}, not"
                f" [{BLOCK}, C, H, W]"
            )
        read = sha256_float32(latents)
        if read != published.latents_sha256:
            raise WorldStateError(
                f"{published.media_id} @ {published.t_first}: sha256 of the latents read is"
                f" {read[:16]}, the owner stamped {published.latents_sha256[:16]}"
                " (truncated, overwritten or corrupted)"
            )
        return latents

    # ---------------------------------------------------------------------------------- lockstep
    def _wait(
        self,
        others: Sequence[str],
        lacking: Callable[[str], bool],
        *,
        what: str,
        max_wait_s: float,
    ) -> None:
        """Poll until no running client of ``others`` is ``lacking`` what it must have published.

        Raises:
            ClientFailedError: another client marked itself failed.
            LockstepTimeoutError: another client is still lacking after ``max_wait_s``.
        """
        others = [str(m) for m in others]
        if not others:
            return
        started = time.monotonic()
        while True:
            self.refresh()
            failed = [m for m in others if self.done_status(m) == DONE_FAILED]
            if failed:
                raise ClientFailedError(
                    f"{self.client}: {failed} marked themselves failed while this client waited"
                    f" for {what}"
                )
            late = [m for m in others if not self.is_done(m) and lacking(m)]
            waited = time.monotonic() - started
            if not late:
                break
            if waited >= max_wait_s:
                raise LockstepTimeoutError(
                    f"{self.client}: after {max_wait_s} s, {late} have not published {what}"
                )
            time.sleep(self.poll_s)
        self.wait_stats.n_waits += 1
        self.wait_stats.seconds_total += waited

    def wait_for_others(self, others: Sequence[str], *, t_target: int, max_wait_s: float) -> None:
        """Lockstep: wait until every other client published its blocks that ended before
        ``t_target`` and its step record for ``t_target``.

        An owner publishes that record before it waits and waits only for earlier blocks, so the
        protocol cannot deadlock, and every step admits exactly each other client's blocks with
        ``t_last < t_target``. A client done ``ok`` is not waited for.

        Args:
            others (Sequence[str]): the media ids of the other clients.
            t_target (int): the source frame of the block about to be generated.
            max_wait_s (float): the longest wait, s.

        Raises:
            ClientFailedError: another client marked itself failed.
            LockstepTimeoutError: another client is still short after ``max_wait_s``.
        """
        need = int(t_target) - SOURCE_FRAMES_PER_LATENT

        def lacking(media_id: str) -> bool:
            progress = self.progress(media_id)
            return progress is None or progress < need or not self.has_step(media_id, t_target)

        what = f"up to t_last {need} and the step record of block {t_target}"
        self._wait(others, lacking, what=what, max_wait_s=max_wait_s)

    def wait_for_positions(
        self, others: Sequence[str], *, t_target: int, max_wait_s: float
    ) -> None:
        """Lockstep of the closed loop's first six blocks, which have no step records: wait until
        every other client published its position for block ``t_target``.

        Args and errors: :meth:`wait_for_others`.
        """

        def lacking(media_id: str) -> bool:
            return self.read_position(media_id, t_target) is None

        what = f"a position for block {t_target}"
        self._wait(others, lacking, what=what, max_wait_s=max_wait_s)
