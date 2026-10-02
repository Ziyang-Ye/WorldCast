"""The memory bank (Sec. 3.3): memory entries, publish / evict / withdraw, and retrieval.

A memory entry is one generated block: its four cameras and the depth read off its latents,
back-projected to world points. A client keeps at most ``B`` entries of its own; copies of other
writers' entries leave only when those writers withdraw them. Retrieval reads the one entry that
fills the most holes of the next block (App. "Scene state in detail").
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .geometry import NPIX, explained, margin, view_points, zbuffer

__all__ = [
    "BOUND",
    "Uid",
    "MemoryEntry",
    "PublishResult",
    "RetrieveResult",
    "MemoryBank",
    "pick_best",
]

#: ``B``: the most entries of its own a client keeps (paper: 64).
BOUND: int = 64

#: An entry's identity in one bank: (writer client, index of its block in the reader's block list).
Uid = tuple[str, int]


@dataclass
class MemoryEntry:
    """One generated block as a memory entry.

    Attributes:
        client (str): the writer (its media id).
        block (int): index of the block in the reading client's block list
            (:class:`worldcast.engine.inference.pool.PeerBlocks`).
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

    client: str
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
        tans = np.asarray(self.tans, dtype=np.float64).reshape(-1, 2)
        self.tans = np.repeat(tans, len(self.c2w), 0) if len(tans) == 1 else tans
        self.depth = np.asarray(self.depth, dtype=np.float64).reshape(len(self.c2w), NPIX)
        points, self.hit = view_points(self.c2w, self.tans, self.depth)
        self.points = points[self.hit]

    @property
    def uid(self) -> Uid:
        return (str(self.client), int(self.block))

    @property
    def n_views(self) -> int:
        return int(len(self.c2w))

    @property
    def n_hit(self) -> int:
        return int(self.hit.sum())

    def zbuf_of(self, points) -> np.ndarray:
        """``points`` ``[n, 3]`` splatted into this entry's cameras: float64 ``[n_views, NPIX]``."""
        return np.stack([zbuffer(points, self.c2w[m], self.tans[m]) for m in range(self.n_views)])

    def explained_by(self, zbuf) -> np.ndarray:
        """``[n_views, NPIX]`` bool: this entry's surface pixels the z-buffer ``zbuf`` covers."""
        return explained(zbuf, self.depth, self.hit)


@dataclass
class PublishResult:
    """Outcome of :meth:`MemoryBank.publish`.

    Attributes:
        published (bool): False only for a block without any surface pixel (not stored).
        n_hit (int): the block's surface pixels over its four cameras.
        evicted (list[Uid]): entries the owner withdrew to get back to ``B`` (always empty for
            another writer's entry).
    """

    published: bool
    n_hit: int
    evicted: list[Uid]


@dataclass
class RetrieveResult:
    """Outcome of :meth:`MemoryBank.retrieve`.

    Attributes:
        entry (MemoryEntry | None): the retrieved entry, or None (the block has no memory slot).
        score (int): the best candidate's covered holes (0 without a candidate).
        n_hole (int): holes of the next block over its four cameras.
        n_pixels (int): pixels of the next block over its four cameras (``4 * NPIX``).
        n_candidates (int): entries scored.
        scores (dict[Uid, int]): covered holes per candidate, in candidate order.
    """

    entry: MemoryEntry | None
    score: int
    n_hole: int
    n_pixels: int
    n_candidates: int
    scores: dict[Uid, int]


def pick_best(cands: Sequence[MemoryEntry], score: np.ndarray) -> MemoryEntry | None:
    """The fill rule's pick: the most covered holes, ties to the latest ``(t_last, seq, block)``.

    Args:
        cands (Sequence[MemoryEntry]): the candidates.
        score (np.ndarray): ``[len(cands)]`` int covered holes.

    Returns:
        MemoryEntry | None: None without a candidate or when the best score is 0.
    """
    if len(cands) and int(score.max()) > 0:
        return cands[
            max(
                range(len(cands)),
                key=lambda k: (int(score[k]), cands[k].t_last, cands[k].seq, cands[k].block),
            )
        ]
    return None


class MemoryBank:
    """One client's view of the scene state: its own entries and copies of every other writer's.

    The owner's entries obey the bound ``B``; another writer's entries leave only through
    :meth:`withdraw`. Reprojections between the owner's entries are cached sparsely (flat pixel
    indices int32 and z float32 of a source's points splatted into a destination's cameras), so a
    publish adds at most ``2B`` of them.

    Args:
        owner (str): the client's media id.
        bound (int): ``B``.
    """

    def __init__(self, *, owner: str, bound: int = BOUND) -> None:
        self.owner = str(owner)
        self.bound = int(bound)
        self.entries: dict[str, list[MemoryEntry]] = {}
        self._pair: dict[tuple[Uid, Uid], tuple[np.ndarray, np.ndarray]] = {}
        self._seq = 0
        self.n_published = 0
        self.n_evicted = 0
        self.n_withdrawn = 0

    def __len__(self) -> int:
        return sum(len(v) for v in self.entries.values())

    def resident(self, client: str | None = None) -> list[MemoryEntry]:
        """Resident entries of ``client`` in publication order, or of every client (None)."""
        if client is not None:
            return list(self.entries.get(str(client), []))
        return [e for v in self.entries.values() for e in v]

    def per_client(self) -> dict[str, int]:
        """Resident entries per writer."""
        return {c: len(v) for c, v in self.entries.items()}

    def _sparse(self, src: MemoryEntry, dst: MemoryEntry) -> tuple[np.ndarray, np.ndarray]:
        key = (src.uid, dst.uid)
        got = self._pair.get(key)
        if got is None:
            # z cached in float32 and compared in float64, as deployed: eviction depends on the cast
            zb = dst.zbuf_of(src.points).reshape(-1)
            idx = np.flatnonzero(np.isfinite(zb))
            got = (idx.astype(np.int32), zb[idx].astype(np.float32))
            self._pair[key] = got
        return got

    def _zbuf_from(self, srcs: Sequence[MemoryEntry], dst: MemoryEntry) -> np.ndarray:
        zb = np.full(dst.n_views * NPIX, np.inf)
        for s in srcs:
            idx, z = self._sparse(s, dst)
            if idx.size:
                np.minimum.at(zb, idx, z.astype(np.float64))
        return zb.reshape(dst.n_views, NPIX)

    def covered_share(self, dst: MemoryEntry, srcs: Sequence[MemoryEntry]) -> float:
        """Share of ``dst``'s surface pixels the points of ``srcs`` cover (0.0 when it has none)."""
        n = dst.n_hit
        if n == 0:
            return 0.0
        return float(dst.explained_by(self._zbuf_from(srcs, dst)).sum()) / n

    def _drop_pairs(self, uid: Uid) -> None:
        for k in [k for k in self._pair if uid in k]:
            del self._pair[k]

    def publish(self, entry: MemoryEntry) -> PublishResult:
        """Add an entry (own or another writer's copy) and set its ``seq``; the owner's obey ``B``.

        An entry with no surface pixel is not stored. Raises ``ValueError`` for a uid offered
        before.
        """
        held = self.entries.setdefault(str(entry.client), [])
        if any(e.uid == entry.uid for e in held):
            raise ValueError(f"entry {entry.uid} offered twice")
        n_hit = entry.n_hit
        if n_hit == 0:
            return PublishResult(False, 0, [])
        entry.seq = self._seq
        self._seq += 1
        held.append(entry)
        self.n_published += 1
        evicted = self._evict_own() if str(entry.client) == self.owner else []
        return PublishResult(True, n_hit, evicted)

    def _evict_own(self) -> list[Uid]:
        """While the owner holds more than ``B`` entries, withdraw the one its other entries cover
        best (largest share of its surface pixels), the older on a tie."""
        out: list[Uid] = []
        own = self.entries.get(self.owner, [])
        while len(own) > self.bound:
            shares = [self.covered_share(e, [f for f in own if f is not e]) for e in own]
            gone = own.pop(max(range(len(own)), key=lambda i: (shares[i], -own[i].seq)))
            self._drop_pairs(gone.uid)
            self.n_evicted += 1
            out.append(gone.uid)
        return out

    def withdraw(self, uid: Uid) -> bool:
        """Remove another writer's entry because that writer withdrew it; True if it was held.

        Raises ``ValueError`` for the owner's entries, which leave only through the bound.
        """
        client = str(uid[0])
        if client == self.owner:
            raise ValueError(
                f"withdraw({uid}): the owner's entries are withdrawn by its own bound, not by a"
                " record"
            )
        held = self.entries.get(client, [])
        for i, e in enumerate(held):
            if e.uid == tuple(uid):
                self._drop_pairs(held.pop(i).uid)
                self.n_withdrawn += 1
                return True
        return False

    def retrieve(
        self, *, next_c2w, next_tans, recent_points, candidates: Sequence[MemoryEntry]
    ) -> RetrieveResult:
        """The fill rule (k = 1): the candidate that covers the most holes of the next block.

        A pixel of the next block's cameras is a hole when no ``recent_points`` reach it. Every
        candidate's points are splatted into the same cameras; the nearest surface at a pixel is the
        reference, and an entry covers a hole when one of its points lies within
        :func:`~worldcast.scene_state.geometry.margin` of it.

        Args:
            next_c2w (np.ndarray): ``[4, 4, 4]`` the next block's cameras (it has no picture yet).
            next_tans (np.ndarray): ``[4, 2]`` or ``[1, 2]``.
            recent_points (np.ndarray): ``[n, 3]`` world points of the reader's recent latent
                frames, u.
            candidates (Sequence[MemoryEntry]): the entries that may be retrieved (the caller's
                causal cut); nothing else is splatted.

        Returns:
            RetrieveResult: the pick and its scores.
        """
        cams = np.asarray(next_c2w, dtype=np.float64).reshape(-1, 4, 4)
        tans = np.asarray(next_tans, dtype=np.float64).reshape(-1, 2)
        if len(tans) == 1:
            tans = np.repeat(tans, len(cams), 0)
        recent = np.asarray(recent_points, dtype=np.float64).reshape(-1, 3)
        cands = list(candidates)
        n_e = len(cands)
        if n_e:
            points = np.concatenate([e.points for e in cands], 0)
            point_owner = np.concatenate(
                [np.full(len(e.points), i, np.int64) for i, e in enumerate(cands)]
            )
        score = np.zeros(n_e, np.int64)
        n_hole = 0
        for m in range(len(cams)):
            hole = ~np.isfinite(zbuffer(recent, cams[m], tans[m]))
            n_hole += int(hole.sum())
            if n_e:
                ze = zbuffer(points, cams[m], tans[m], owner=point_owner, n_owner=n_e)
                zp = ze.min(0)  # the reference surface
                with np.errstate(invalid="ignore"):
                    front = np.isfinite(ze) & (ze <= zp[None] + margin(zp)[None])
                score += (front & hole[None]).sum(1)
        scores = {e.uid: int(s) for e, s in zip(cands, score)}
        return RetrieveResult(
            entry=pick_best(cands, score),
            score=int(score.max()) if n_e else 0,
            n_hole=int(n_hole),
            n_pixels=int(len(cams) * NPIX),
            n_candidates=n_e,
            scores=scores,
        )
