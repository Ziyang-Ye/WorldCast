"""The memory bank (Sec. 3.3): memory entries, publish / withdraw, and retrieval.

A memory entry is one generated block: its four cameras and the depth read off its latents,
back-projected to world points. Every entry has one owner, the client that generated it. A client
keeps at most ``B`` entries of its own, withdrawing the most redundant beyond that; its copies of
the other clients' entries leave only when their owners withdraw them. Retrieval reads the one entry
that covers the most missing pixels of the next block (App. "Scene state in detail").
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from worldcast.config.inference import MEMORY_BOUND

from .geometry import NPIX, covered, per_camera_tans, tolerance, view_points, zbuffer

__all__ = [
    "MemoryBank",
    "MemoryEntry",
    "PublishResult",
    "RetrieveResult",
    "Uid",
    "missing_coverage",
    "pick_best",
]

#: An entry's identity: its owner and the caller's id of the block, unique per owner.
Uid = tuple[str, int]


@dataclass
class MemoryEntry:
    """One generated block as a memory entry.

    Attributes:
        owner (str): the client that generated the block (its media id).
        block (int): the caller's id of the block, unique per owner.
        t_first (int): source-frame time of the block's first latent frame.
        t_last (int): source-frame time of its last latent frame.
        c2w (np.ndarray): ``[4, 4, 4]`` cameras (stored float64).
        tans (np.ndarray): ``[4, 2]`` or ``[1, 2]`` field-of-view tangents (stored float64 ``[4,
            2]``).
        depth (np.ndarray): ``[4, 24, 42]`` axial depth, u (stored float64 ``[4, NPIX]``).
        seq (int): publication order within the bank, set by :meth:`MemoryBank.publish` (smaller is
            older).
        points (np.ndarray): ``[n_hit, 3]`` float64 world points of the surface pixels (derived).
        hit (np.ndarray): ``[4, NPIX]`` bool surface mask (derived).
    """

    owner: str
    block: int
    t_first: int
    t_last: int
    c2w: np.ndarray
    tans: np.ndarray
    depth: np.ndarray
    seq: int = -1
    points: np.ndarray = field(init=False, repr=False)
    hit: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.c2w = np.asarray(self.c2w, dtype=np.float64).reshape(-1, 4, 4)
        self.tans = per_camera_tans(self.tans, len(self.c2w))
        self.depth = np.asarray(self.depth, dtype=np.float64).reshape(len(self.c2w), NPIX)
        points, self.hit = view_points(self.c2w, self.tans, self.depth)
        self.points = points[self.hit]

    @property
    def uid(self) -> Uid:
        return (str(self.owner), int(self.block))

    @property
    def n_views(self) -> int:
        return int(len(self.c2w))

    @property
    def n_hit(self) -> int:
        return int(self.hit.sum())

    def zbuffers(self, points: np.ndarray) -> np.ndarray:
        """``points`` ``[n, 3]`` projected into this entry's cameras: float64 ``[n_views, NPIX]``
        axial depth, ``inf`` where no point lands."""
        return np.stack([zbuffer(points, self.c2w[m], self.tans[m]) for m in range(self.n_views)])

    def covered_by(self, zbuf: np.ndarray) -> np.ndarray:
        """``[n_views, NPIX]`` bool: this entry's surface pixels the z-buffers ``zbuf`` cover."""
        return covered(zbuf, self.depth, self.hit)


@dataclass
class PublishResult:
    """Outcome of :meth:`MemoryBank.publish`.

    Attributes:
        published (bool): False only for a block without any surface pixel (not stored).
        withdrawn (list[MemoryEntry]): the entries the client withdrew to get back to ``B`` (empty
            for another client's entry).
    """

    published: bool
    withdrawn: list[MemoryEntry]


@dataclass
class RetrieveResult:
    """Outcome of :meth:`MemoryBank.retrieve`.

    Attributes:
        entry (MemoryEntry | None): the retrieved entry, or None (the block has no memory frames).
        coverage (int): the missing pixels the best candidate covers (0 without a candidate).
        n_missing (int): missing pixels of the next block over its four cameras.
        n_candidates (int): entries scored.
    """

    entry: MemoryEntry | None
    coverage: int
    n_missing: int
    n_candidates: int


def missing_coverage(
    next_c2w: ArrayLike,
    next_tans: ArrayLike,
    recent_points: ArrayLike,
    candidates: Sequence[MemoryEntry],
) -> tuple[np.ndarray, int]:
    """How many missing pixels of the next block each candidate covers.

    A pixel of the next block's cameras is missing when no ``recent_points`` reach it. Every
    candidate's points are projected into the same cameras; the nearest of them at a pixel is the
    reference surface, and an entry covers a missing pixel when one of its points lies within
    :func:`~worldcast.scene_state.geometry.tolerance` of it.

    Args:
        next_c2w (ArrayLike): ``[4, 4, 4]`` the next block's cameras (it has no picture yet).
        next_tans (ArrayLike): ``[4, 2]`` or ``[1, 2]``.
        recent_points (ArrayLike): ``[n, 3]`` world points of the client's recent latent frames, u.
        candidates (Sequence[MemoryEntry]): the entries that may be retrieved.

    Returns:
        tuple[np.ndarray, int]: ``[len(candidates)]`` int64 covered missing pixels, and the number
        of missing pixels over the four cameras.
    """
    cams = np.asarray(next_c2w, dtype=np.float64).reshape(-1, 4, 4)
    tans = per_camera_tans(next_tans, len(cams))
    recent = np.asarray(recent_points, dtype=np.float64).reshape(-1, 3)
    n = len(candidates)
    if n:
        points = np.concatenate([e.points for e in candidates], 0)
        group = np.concatenate(
            [np.full(len(e.points), i, np.int64) for i, e in enumerate(candidates)]
        )
    coverage = np.zeros(n, np.int64)
    n_missing = 0
    for m in range(len(cams)):
        missing = ~np.isfinite(zbuffer(recent, cams[m], tans[m]))
        n_missing += int(missing.sum())
        if n:
            z = zbuffer(points, cams[m], tans[m], group=group, n_groups=n)
            reference = z.min(0)
            with np.errstate(invalid="ignore"):
                front = np.isfinite(z) & (z <= reference[None] + tolerance(reference)[None])
            coverage += (front & missing[None]).sum(1)
    return coverage, n_missing


def pick_best(candidates: Sequence[MemoryEntry], coverage: np.ndarray) -> MemoryEntry | None:
    """The retrieved entry: the most covered missing pixels, ties to the latest ``(t_last, seq)``.

    Args:
        candidates (Sequence[MemoryEntry]): the candidates.
        coverage (np.ndarray): ``[len(candidates)]`` int covered missing pixels.

    Returns:
        MemoryEntry | None: None without a candidate or when the best coverage is 0.
    """
    if len(candidates) and int(coverage.max()) > 0:
        return candidates[
            max(
                range(len(candidates)),
                key=lambda k: (int(coverage[k]), candidates[k].t_last, candidates[k].seq),
            )
        ]
    return None


@dataclass
class _Coverage:
    """What the client's other entries cover of one of its entries (:class:`MemoryBank`).

    Attributes:
        slot (int): the entry's row in the other own entries' ``z``.
        z (np.ndarray): ``[B + 1, n_views * NPIX]`` float32, per slot the z-buffer of that own
            entry's points in this entry's cameras (``inf`` where none lands, and for free slots).
        nearest (np.ndarray): ``[n_views * NPIX]`` float64, the minimum of ``z`` over the slots.
        n_covered (int): this entry's surface pixels that ``nearest`` covers.
    """

    slot: int
    z: np.ndarray
    nearest: np.ndarray
    n_covered: int


class MemoryBank:
    """One client's copy of the scene state: its own entries and the other clients'.

    The client's own entries obey the bound ``B``; another client's entries leave only through
    :meth:`withdraw`. The reprojections between the client's own entries are cached (App. "Scene
    state in detail"): publishing an own entry projects its points into the cameras of each other
    own entry and theirs into its cameras, at most ``2B`` reprojections, and updates every own
    entry's coverage by the others; withdrawal reads the coverages and requires none.

    Args:
        client (str): the client that holds the bank (its media id).
        bound (int): ``B``, the most entries of its own the client keeps (paper: 64).
    """

    def __init__(self, *, client: str, bound: int = MEMORY_BOUND) -> None:
        self.client = str(client)
        self.bound = int(bound)
        self.entries: dict[str, list[MemoryEntry]] = {}
        self._coverage: dict[Uid, _Coverage] = {}
        self._free_slots = list(range(self.bound + 1))
        self._seq = 0

    def __len__(self) -> int:
        return sum(len(held) for held in self.entries.values())

    def resident(self, owner: str | None = None) -> list[MemoryEntry]:
        """The entries the bank holds of ``owner`` in publication order, or of every client
        (None)."""
        if owner is not None:
            return list(self.entries.get(str(owner), []))
        return [e for held in self.entries.values() for e in held]

    def covered_share(self, entry: MemoryEntry) -> float:
        """Share of an own entry's surface pixels that the client's other entries cover (cached)."""
        return float(self._coverage[entry.uid].n_covered) / entry.n_hit

    def publish(self, entry: MemoryEntry) -> PublishResult:
        """Add an entry (the client's own, or its copy of another client's) and set its ``seq``.

        An entry without a surface pixel is not stored. An own entry beyond the bound ``B`` makes
        the client withdraw the entry its other entries cover best.

        Raises:
            ValueError: the bank already holds an entry of this uid.
        """
        held = self.entries.setdefault(str(entry.owner), [])
        if any(e.uid == entry.uid for e in held):
            raise ValueError(f"entry {entry.uid} offered twice")
        if entry.n_hit == 0:
            return PublishResult(False, [])
        entry.seq = self._seq
        self._seq += 1
        own = str(entry.owner) == self.client
        if own:
            self._cover(entry, held)
        held.append(entry)
        return PublishResult(True, self._withdraw_own() if own else [])

    @staticmethod
    def _reproject(source: MemoryEntry, target: MemoryEntry) -> np.ndarray:
        # z cached in float32 and compared in float64, as in the paper's runs: withdrawal follows
        # the cast
        return target.zbuffers(source.points).reshape(-1).astype(np.float32)

    @staticmethod
    def _n_covered(entry: MemoryEntry, nearest: np.ndarray) -> int:
        return int(entry.covered_by(nearest.reshape(entry.n_views, NPIX)).sum())

    def _cover(self, entry: MemoryEntry, others: Sequence[MemoryEntry]) -> None:
        """Reproject a new own entry into the other own entries and theirs into it (``2
        len(others)`` reprojections), and update every coverage."""
        slot = self._free_slots.pop()
        z = np.full((self.bound + 1, entry.n_views * NPIX), np.inf, np.float32)
        for other in others:
            theirs = self._coverage[other.uid]
            theirs.z[slot] = self._reproject(entry, other)
            np.minimum(theirs.nearest, theirs.z[slot], out=theirs.nearest)
            theirs.n_covered = self._n_covered(other, theirs.nearest)
            z[theirs.slot] = self._reproject(other, entry)
        nearest = z.min(0).astype(np.float64)
        self._coverage[entry.uid] = _Coverage(slot, z, nearest, self._n_covered(entry, nearest))

    def _uncover(self, gone: MemoryEntry) -> None:
        """Take a withdrawn own entry out of the other own entries' coverage (no reprojection)."""
        slot = self._coverage.pop(gone.uid).slot
        for other in self.entries[self.client]:
            theirs = self._coverage[other.uid]
            z = theirs.z[slot]
            # only where the withdrawn entry was the nearest can the minimum change
            pixels = np.flatnonzero((z == theirs.nearest) & np.isfinite(z))
            z[:] = np.inf
            if pixels.size:
                theirs.nearest[pixels] = theirs.z[:, pixels].min(0)
                theirs.n_covered = self._n_covered(other, theirs.nearest)
        self._free_slots.append(slot)

    def _withdraw_own(self) -> list[MemoryEntry]:
        """While the client holds more than ``B`` own entries, withdraw the one its other entries
        cover best (largest share of its surface pixels), the older on a tie."""
        withdrawn: list[MemoryEntry] = []
        own = self.entries.get(self.client, [])
        while len(own) > self.bound:
            shares = [self.covered_share(e) for e in own]
            gone = own.pop(max(range(len(own)), key=lambda i: (shares[i], -own[i].seq)))
            self._uncover(gone)
            withdrawn.append(gone)
        return withdrawn

    def withdraw(self, uid: Uid) -> bool:
        """Remove another client's entry because its owner withdrew it; True if it was held.

        Raises:
            ValueError: ``uid`` is one of the client's own entries, which leave only through the
                bound.
        """
        owner = str(uid[0])
        if owner == self.client:
            raise ValueError(
                f"withdraw({uid}): the client's own entries are withdrawn by its bound, not by a"
                " record"
            )
        held = self.entries.get(owner, [])
        for i, e in enumerate(held):
            if e.uid == tuple(uid):
                held.pop(i)
                return True
        return False

    def retrieve(
        self, *, next_c2w: ArrayLike, next_tans: ArrayLike, recent_points: ArrayLike, t_target: int
    ) -> RetrieveResult:
        """Retrieval (at most one entry): the entry that covers the most missing pixels of the next
        block (:func:`missing_coverage`, :func:`pick_best`).

        Args:
            next_c2w (ArrayLike): ``[4, 4, 4]`` the next block's cameras.
            next_tans (ArrayLike): ``[4, 2]`` or ``[1, 2]``.
            recent_points (ArrayLike): ``[n, 3]`` world points of the client's recent latent frames,
                u.
            t_target (int): source-frame time of the next block; the candidates are the held
                entries, of any owner, that ended before it (``t_last < t_target``).

        Returns:
            RetrieveResult: the retrieved entry and its coverage.
        """
        candidates = [e for e in self.resident() if e.t_last < int(t_target)]
        coverage, n_missing = missing_coverage(next_c2w, next_tans, recent_points, candidates)
        return RetrieveResult(
            entry=pick_best(candidates, coverage),
            coverage=int(coverage.max()) if candidates else 0,
            n_missing=n_missing,
            n_candidates=len(candidates),
        )
