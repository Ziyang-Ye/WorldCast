"""A client's scene state (Sec. 3.3): own and peer blocks in, withdrawals followed, retrieval.

A client's :class:`~worldcast.scene_state.bank.MemoryBank` holds its own entries (at most ``B``) and
a copy of every peer's, which changes only by following that peer's published withdrawals.
docs/inference.md, "The block loop", gives the order of the calls per block.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .bank import BOUND, MemoryBank, MemoryEntry, RetrieveResult, Uid
from .geometry import axial_from_log_grid, view_points

__all__ = [
    "BLOCK",
    "DepthFn",
    "PeerLatentsFn",
    "StepSource",
    "FollowStats",
    "CopyMismatchError",
    "SceneState",
]

#: Latent frames per block (one memory entry).
BLOCK: int = 4

#: Latent frames ``[T, 48, 24, 42]`` (torch or numpy, float32) -> log axial depth ``[T, 4, 24, 42]``
#: (numpy float32; 4 = the latent frame's pixel frames): the depth head and its read-out
#: (``worldcast.modeling.depth_head``).
DepthFn = Callable[[Any], np.ndarray]

#: ``(block, t_target) -> [4, 48, 24, 42]`` float32, the latents a peer published for an admitted
#: block (``worldcast.engine.inference.pool.PoolBackend.peer_latents``).
PeerLatentsFn = Callable[[Mapping[str, Any], int], Any]


class StepSource(Protocol):
    """Where a client reads its peers' step records (``engine.inference.pool.PoolBackend``)."""

    def read_steps(self, media_id: str, *, upto: int) -> Sequence[Any]:
        """Every step record of ``media_id`` with ``t_target <= upto``, oldest first; each has
        ``t_target``, ``withdrawn`` and ``resident`` (published ``orig_first`` values)."""
        ...


class CopyMismatchError(RuntimeError):
    """A reader's copy of a writer's entries differs from the writer's resident list."""


@dataclass
class FollowStats:
    """Counters of :meth:`SceneState.follow_withdrawals` (written to ``client.json``)."""

    steps_applied: int = 0
    withdrawals_applied: int = 0
    withdrawals_not_held: int = 0
    withdrawals_not_held_skipped: int = 0
    withdrawals_not_held_unpublished: int = 0
    copy_checks: int = 0
    copy_equal: int = 0
    copy_equal_modulo_empty: int = 0


def _poses(c2w) -> np.ndarray:
    """``[N, 4, 4]`` float64 camera-to-world from a tensor or an array."""
    c2w = c2w.detach().cpu().numpy() if hasattr(c2w, "detach") else c2w
    return np.asarray(c2w, dtype=np.float64).reshape(-1, 4, 4)


class SceneState:
    """One client's scene state.

    Run one client per process: numpy reprojections under a multithreaded BLAS are not
    bit-reproducible across threads of one process.

    A block is a mapping with at least ``media_id``, ``f0`` (its first latent frame in its writer's
    rollout), ``orig_first``, ``orig_last`` (source-frame times of its first and last latent frame)
    and ``c2w`` ``[4, 4, 4]``; the list it lives in
    (:attr:`worldcast.engine.inference.pool.PeerBlocks.blocks`) is append-only, and a block's
    position in it is its address (:attr:`MemoryEntry.block`).

    Args:
        client (str): this client's media id (the writer its blocks are published under).
        tans (np.ndarray): ``[[tan_h, tan_v]]``, the field of view of every camera of retrieval (the
            client's unscoped one, stored float32 as deployed), or ``[n >= 4, 2]``.
        depth_fn (DepthFn): the depth head.
        peer_latents (PeerLatentsFn): the latents of an admitted peer block.
        bound (int): ``B``, own entries kept (paper: 64).
    """

    def __init__(
        self,
        *,
        client: str,
        tans,
        depth_fn: DepthFn,
        peer_latents: PeerLatentsFn,
        bound: int = BOUND,
    ) -> None:
        self.client = str(client)
        self.tans = np.asarray(tans, dtype=np.float32).reshape(-1, 2)
        self.depth_fn = depth_fn
        self.peer_latents = peer_latents
        self.bank = MemoryBank(owner=self.client, bound=int(bound))
        self._written: set[int] = set()
        self._uid_pub: dict[Uid, int] = {}  # entry uid -> orig_first of the published block
        self._pub_uid: dict[tuple[str, int], Uid] = {}  # (client, orig_first) -> entry uid
        self._unpublished: set[tuple[str, int]] = set()  # (client, orig_first) offered, not stored
        self._own_withdrawn_step: list[int] = []
        self._follow_applied: dict[str, set[int]] = {}
        self.follow_stats = FollowStats()

    def _tans_for(self, n: int) -> np.ndarray:
        if self.tans.shape[0] >= n:
            return self.tans[:n]
        return np.repeat(self.tans[:1], n, axis=0)

    def _axial(self, latents) -> np.ndarray:
        return axial_from_log_grid(self.depth_fn(latents))

    def ingest_own(self, blocks: Sequence[Mapping[str, Any]], *, t_target: int, own_latents) -> int:
        """Publish this client's own blocks that ended before ``t_target`` into the bank.

        An entry's depth is read off its own four latent frames, one depth call per block.

        Args:
            blocks (Sequence[Mapping[str, Any]]): the client's block list.
            t_target (int): source-frame time of the block about to be generated.
            own_latents (torch.Tensor | np.ndarray): the client's latent store ``[N, 48, 24, 42]``.

        Returns:
            int: blocks ingested.
        """
        return self._ingest(blocks, t_target=int(t_target), own_latents=own_latents, own=True)

    def ingest_peers(self, blocks: Sequence[Mapping[str, Any]], *, t_target: int) -> int:
        """Copy the peers' admitted blocks that ended before ``t_target`` into the bank.

        Call after :meth:`ingest_own` and the admission of this block; an own block met here raises
        ``RuntimeError``. Returns the number of blocks ingested.
        """
        return self._ingest(blocks, t_target=int(t_target), own_latents=None, own=False)

    def _ingest(self, blocks, *, t_target: int, own_latents, own: bool) -> int:
        n_new = 0
        for i, block in enumerate(blocks):
            if i in self._written or int(block["orig_last"]) >= t_target:
                continue
            is_own = str(block["media_id"]) == self.client
            if own and not is_own:
                continue
            if not own and is_own:
                raise RuntimeError(
                    f"own block {i} (orig_last {int(block['orig_last'])}) reached ingest_peers: the"
                    " own block is ingested before the step record, and the peers' after the wait"
                )
            self._written.add(i)
            if is_own:
                f0 = int(block["f0"])
                latents = own_latents[f0 : f0 + BLOCK]
            else:
                latents = self.peer_latents(block, t_target)
            c2w = _poses(block["c2w"])
            entry = MemoryEntry(
                client=str(block["media_id"]),
                block=int(i),
                t_first=int(block["orig_first"]),
                t_last=int(block["orig_last"]),
                c2w=c2w,
                tans=self._tans_for(len(c2w)),
                depth=self._axial(latents),
            )
            result = self.bank.publish(entry)
            pub = int(block["orig_first"])
            if result.published:
                self._uid_pub[entry.uid] = pub
                self._pub_uid[(entry.client, pub)] = entry.uid
            else:
                self._unpublished.add((entry.client, pub))
            if is_own:
                self._own_withdrawn_step.extend(self._uid_pub[u] for u in result.evicted)
            n_new += 1
        return n_new

    def drain_own_step(self) -> tuple[list[int], list[int]]:
        """This client's step record ``(withdrawn, resident)``.

        Each is the sorted ``orig_first`` of the blocks withdrawn since the last drain / held now.
        """
        wd = sorted(int(v) for v in self._own_withdrawn_step)
        self._own_withdrawn_step = []
        res = sorted(int(self._uid_pub[e.uid]) for e in self.bank.resident(self.client))
        return wd, res

    def follow_withdrawals(
        self,
        steps: StepSource,
        *,
        peers: Iterable[str],
        t_target: int,
        skipped: Iterable[tuple[str, int]] = (),
    ) -> None:
        """Apply every peer's withdrawals up to ``t_target`` and check the copy against its record.

        Args:
            steps (StepSource): the pool.
            peers (Iterable[str]): every writer whose blocks this client may hold.
            t_target (int): this block's time; records up to it are applied, the one at it is
                checked.
            skipped (Iterable[tuple[str, int]]): ``(client, orig_first)`` of published blocks this
                client never admitted.

        Raises:
            CopyMismatchError: a copy differs from its writer's resident list at ``t_target``.
        """
        st = self.follow_stats
        skipped = {(str(m), int(o)) for m, o in skipped}
        for p in sorted(str(x) for x in peers):
            done = self._follow_applied.setdefault(p, set())
            recs = steps.read_steps(p, upto=int(t_target))
            for r in recs:
                t = int(r.t_target)
                if t in done:
                    continue
                for o in r.withdrawn:
                    uid = self._pub_uid.get((p, int(o)))
                    if uid is not None and self.bank.withdraw(uid):
                        st.withdrawals_applied += 1
                    elif (p, int(o)) in skipped:
                        st.withdrawals_not_held_skipped += 1
                    elif (p, int(o)) in self._unpublished:
                        st.withdrawals_not_held_unpublished += 1
                    else:
                        st.withdrawals_not_held += 1
                done.add(t)
                st.steps_applied += 1
            now = [r for r in recs if int(r.t_target) == int(t_target)]
            if not now:
                continue
            want = {int(o) for o in now[0].resident} - {o for (m, o) in skipped if m == p}
            empty = {o for (m, o) in self._unpublished if m == p}
            have = {int(self._uid_pub[e.uid]) for e in self.bank.resident(p)}
            st.copy_checks += 1
            if have == want:
                st.copy_equal += 1
            elif have == want - empty:
                st.copy_equal_modulo_empty += 1
            else:
                raise CopyMismatchError(
                    f"t_target {int(t_target)}: this reader's copy of {p!r} holds {len(have)}"
                    f" entries and the writer's pool {len(want)}; only here"
                    f" {sorted(have - want)[:8]}, only there {sorted(want - have)[:8]}"
                )

    def retrieve(self, *, query_c2w, recent_c2w, recent_latents, t_target: int) -> RetrieveResult:
        """Retrieve at most one memory entry for the block at ``t_target`` (fill rule, k = 1).

        The recent surface is the depth of all recent latent frames read in one call: the depth head
        reads each latent frame with its neighbours inside a call, so call boundaries are part of
        the numbers.

        Args:
            query_c2w (np.ndarray): ``[4, 4, 4]`` the block's cameras (recorded in Table 3).
            recent_c2w (np.ndarray): ``[R, 4, 4]`` cameras of the recent latent frames (R = 12 from
                latent 25 on).
            recent_latents (torch.Tensor): ``[R, 48, 24, 42]`` the recent latent frames.
            t_target (int): source-frame time of the block; only entries with ``t_last < t_target``
                are candidates, the client's own included.

        Returns:
            RetrieveResult: ``entry`` None means no memory slot.
        """
        q, r = _poses(query_c2w), _poses(recent_c2w)
        if len(r):
            if recent_latents is None or int(recent_latents.shape[0]) != len(r):
                raise ValueError(
                    "retrieval needs the reader's recent latents, one per recent camera "
                    f"({len(r)}); got "
                    f"{None if recent_latents is None else int(recent_latents.shape[0])}"
                )
            points, hit = view_points(r, self._tans_for(len(r)), self._axial(recent_latents))
            recent_pts = points[hit]
        else:
            recent_pts = np.zeros((0, 3))
        cands = [e for e in self.bank.resident() if e.t_last < int(t_target)]
        return self.bank.retrieve(
            next_c2w=q, next_tans=self._tans_for(len(q)), recent_points=recent_pts, candidates=cands
        )

    def entry_block(self, result: RetrieveResult) -> int | None:
        """The block-list index whose four latent frames fill the memory slot, or None."""
        return None if result.entry is None else int(result.entry.block)
