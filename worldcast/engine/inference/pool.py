"""The peer pool: clients exchange finished blocks and step records through it and advance in lock-step.

:class:`PoolBackend` is the pool as one client sees it, :class:`LocalDirPool` implements it on a shared directory (the
paper's evaluation setup) and :class:`PeerBlocks` is a client's ordered list of its own and admitted peer blocks
(Paper App. "Distributed deployment"). Readers fetch the fp32 latents of every admitted peer block and recompute its
depth; depth is never sent.
"""

import hashlib
import json
import os
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "BLOCK",
    "TMP_PREFIX",
    "WAIT_POLICIES",
    "DONE_OK",
    "DONE_FAILED",
    "block_sha",
    "latent_sha",
    "PoolError",
    "PeerTimeoutError",
    "PeerFailedError",
    "PublishedBlock",
    "StepRecord",
    "WaitStats",
    "PoolBackend",
    "LocalDirPool",
    "CandidateFn",
    "PeerBlocks",
]

#: Latent frames per published block.
BLOCK: int = 4
#: Temp-file prefix; matches neither ``blk_*.npy`` nor an exact path, so a partial file is invisible.
TMP_PREFIX: str = ".pub-"
#: What a lock-step timeout or a failed peer does: ``fatal`` raises, ``degrade`` stops waiting for that peer.
WAIT_POLICIES: tuple[str, ...] = ("fatal", "degrade")
#: ``DONE.json`` statuses.
DONE_OK: str = "ok"
DONE_FAILED: str = "failed"


def _float32(x) -> np.ndarray:
    """Contiguous float32 numpy copy or view of a tensor or array."""
    a = (
        x.detach().cpu().float().numpy()
        if hasattr(x, "detach")
        else np.asarray(x, dtype=np.float32)
    )
    return np.ascontiguousarray(a, dtype=np.float32)


def block_sha(x) -> str:
    """sha256 hex of the float32 bytes of a block ``[4, C, H, W]`` (tensor or array)."""
    return hashlib.sha256(_float32(x).tobytes()).hexdigest()


def latent_sha(x) -> str:
    """sha256 hex of the float32 bytes of one latent ``[C, H, W]`` (tensor or array)."""
    return hashlib.sha256(_float32(x).tobytes()).hexdigest()


class PoolError(RuntimeError):
    """A pool request that the protocol forbids or the data cannot satisfy."""


class PeerTimeoutError(PoolError):
    """A peer did not reach the lock-step point within ``max_wait_s``."""


class PeerFailedError(PoolError):
    """A peer marked itself failed (``DONE.json`` with ``status='failed'``)."""


@dataclass(frozen=True)
class PublishedBlock:
    """The sidecar fields of one published block.

    ``window_start``, ``f0``: the writer's address (its rollout start and the block's first latent). ``orig_first``,
    ``orig_last``: original-frame times of its first and last latent, ``window_start + stride * f0`` and
    ``window_start + stride * (f0 + 3)``.
    """

    media_id: str
    window_start: int
    f0: int
    orig_first: int
    orig_last: int
    n_latents: int
    latents_sha256: str


@dataclass(frozen=True)
class StepRecord:
    """A writer's record of its block at ``t_target``: the sorted ``orig_first`` of the own entries it withdrew at that
    block's write, and of the own entries resident afterwards."""

    media_id: str
    t_target: int
    withdrawn: tuple[int, ...]
    resident: tuple[int, ...]


@dataclass
class WaitStats:
    """Lock-step bookkeeping of one client (all zero or empty on a run without timeouts)."""

    n_waits: int = 0
    n_timeouts: int = 0
    seconds_total: float = 0.0
    seconds_max: float = 0.0
    last_seconds: float = 0.0
    lagging: set[str] = field(default_factory=set)


class PoolBackend(ABC):
    """The pool as one client sees it: writer of its own blocks and step records, reader of its peers'.

    A backend implements the storage primitives (the abstract methods); the lock-step wait, the address and causal
    rules and the verified reads are shared.

    Args:
        client: this client's media id; it publishes under this name only.
        stride: original frames per latent (8: skip_frame 2 x 4 pixel frames per latent).
        poll_s: lock-step poll period, s (paper run: 2.0).
        on_timeout: one of :data:`WAIT_POLICIES`.
    """

    def __init__(
        self, *, client: str, stride: int, poll_s: float = 2.0, on_timeout: str = "fatal"
    ) -> None:
        if str(on_timeout) not in WAIT_POLICIES:
            raise ValueError(f"on_timeout must be one of {WAIT_POLICIES}, got {on_timeout!r}")
        if int(stride) <= 0:
            raise ValueError(f"stride must be positive, got {stride!r}")
        if float(poll_s) <= 0:
            raise ValueError(f"poll_s must be positive, got {poll_s!r}")
        self.client = str(client)
        self.stride = int(stride)
        self.poll_s = float(poll_s)
        self.on_timeout = str(on_timeout)
        self.wait_stats = WaitStats()

    @abstractmethod
    def publish_block(
        self,
        *,
        window_start: int,
        f0: int,
        latents,
        orig_first: int,
        orig_last: int,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Publish this client's finished block ``[4, C, H, W]`` (float32 on the wire), atomically."""

    @abstractmethod
    def publish_step(
        self, *, t_target: int, withdrawn: Sequence[int], resident: Sequence[int]
    ) -> None:
        """Publish this client's step record for its block at ``t_target``."""

    @abstractmethod
    def mark_done(self, *, status: str = DONE_OK, note: str = "") -> None:
        """Declare that this client will publish nothing more (``ok`` or ``failed``)."""

    @abstractmethod
    def refresh(self) -> int:
        """Poll the pool once; returns the number of blocks seen for the first time."""

    @abstractmethod
    def published_blocks(self, media_id: str) -> list[PublishedBlock]:
        """Every block ``media_id`` has published as of the last :meth:`refresh`, by ``orig_first``."""

    @abstractmethod
    def read_steps(self, media_id: str, *, upto: int) -> list[StepRecord]:
        """Every step record of ``media_id`` with ``t_target <= upto``, oldest first."""

    @abstractmethod
    def has_step(self, media_id: str, t_target: int) -> bool:
        """Whether ``media_id`` has published its step record for ``t_target``."""

    @abstractmethod
    def done_status(self, media_id: str) -> str | None:
        """``None`` while ``media_id`` is running, else its DONE status (``ok`` / ``failed``)."""

    @abstractmethod
    def _read_block(self, block: PublishedBlock) -> np.ndarray:
        """The published latents of ``block``, float32 ``[n_latents, C, H, W]``, verified against its sha."""

    def progress(self, media_id: str) -> int | None:
        """The largest ``orig_last`` ``media_id`` has published (as of the last refresh), or None."""
        return max((int(b.orig_last) for b in self.published_blocks(media_id)), default=None)

    def is_done(self, media_id: str) -> bool:
        return self.done_status(media_id) is not None

    def wait_for_peers(
        self,
        peers: Sequence[str],
        *,
        need_orig_last: int,
        need_step: int,
        max_wait_s: float,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> float:
        """Wait until every peer has published up to ``need_orig_last`` and its step record ``need_step``.

        Before block ``t_target`` a client waits for ``need_orig_last = t_target - stride`` (every peer block that
        ended before it) and ``need_step = t_target``. A writer publishes that record before it waits and waits only
        for earlier blocks, so the protocol cannot deadlock, and without a timeout every step admits exactly each
        peer's blocks with ``orig_last < t_target``. A peer done ``ok`` is not waited for. On a timeout or a failed
        peer, ``fatal`` raises :class:`PeerTimeoutError` / :class:`PeerFailedError`; ``degrade`` adds the short peers
        to ``wait_stats.lagging`` and never waits for them again (the output then depends on timing).

        ``clock`` and ``sleep`` default to ``time.monotonic`` and ``time.sleep``. Returns the seconds waited.
        """
        clock = clock or time.monotonic
        sleep = sleep or time.sleep
        st = self.wait_stats
        st.last_seconds = 0.0
        waiting = [str(p) for p in peers if str(p) not in st.lagging]
        if not waiting:
            return 0.0
        if float(max_wait_s) <= 0:
            raise ValueError(
                f"lock-step needs a positive wait bound, got max_wait_s={max_wait_s!r}"
            )
        t0 = clock()
        while True:
            self.refresh()
            failed = [p for p in waiting if self.done_status(p) == DONE_FAILED]
            if failed and self.on_timeout == "fatal":
                raise PeerFailedError(
                    f"{self.client}: peer(s) {failed} marked themselves failed while this client "
                    f"waited for orig_last >= {int(need_orig_last)} / step {int(need_step)}"
                )
            short = [
                p
                for p in waiting
                if not self.is_done(p)
                and (
                    self.progress(p) is None
                    or self.progress(p) < int(need_orig_last)
                    or not self.has_step(p, int(need_step))
                )
            ]
            if not short:
                break
            if clock() - t0 >= float(max_wait_s):
                if self.on_timeout == "fatal":
                    raise PeerTimeoutError(
                        f"{self.client}: after {float(max_wait_s)} s, peer(s) {short} have not"
                        f" published up to orig_last {int(need_orig_last)} and step"
                        f" {int(need_step)} (progress {[self.progress(p) for p in short]})"
                    )
                st.lagging.update(short)
                st.n_timeouts += 1
                break
            sleep(self.poll_s)
        waited = float(clock() - t0)
        st.n_waits += 1
        st.seconds_total += waited
        st.seconds_max = max(st.seconds_max, waited)
        st.last_seconds = waited
        return waited

    def resolve(
        self, *, media_id: str, window_start: int, f0: int, t_target: int
    ) -> PublishedBlock:
        """The block published at exactly the writer's address ``(window_start, f0)``, if it ended before ``t_target``.

        Raises :class:`PoolError` when no block was published at that address (a splice of two blocks included) or
        when the block does not end before ``t_target`` (the causal cut).
        """
        orig_first = int(window_start) + self.stride * int(f0)
        b = next(
            (b for b in self.published_blocks(str(media_id)) if int(b.orig_first) == orig_first),
            None,
        )
        if b is None or (int(b.window_start), int(b.f0)) != (int(window_start), int(f0)):
            raise PoolError(
                f"{media_id} @ win {int(window_start)} f0 {int(f0)}: no block was published at that"
                " address; a served block is exactly one published block at its writer's own"
                " address"
            )
        if int(b.orig_last) >= int(t_target):
            raise PoolError(
                f"{media_id} @ orig {orig_first}: the block ends at {int(b.orig_last)}, not before"
                f" the reader's causal cut {int(t_target)}"
            )
        return b

    def covers(self, *, media_id: str, window_start: int, f0: int, t_target: int) -> bool:
        """Whether :meth:`resolve` would serve this block now."""
        try:
            self.resolve(media_id=media_id, window_start=window_start, f0=f0, t_target=t_target)
        except PoolError:
            return False
        return True

    def peek_block(self, *, media_id: str, window_start: int, f0: int, t_target: int) -> np.ndarray:
        """A peer block's latents, float32 ``[4, C, H, W]`` numpy, sha-verified."""
        return np.array(
            self._read_block(
                self.resolve(media_id=media_id, window_start=window_start, f0=f0, t_target=t_target)
            ),
            dtype=np.float32,
            copy=True,
        )

    def fetch_block(
        self, *, media_id: str, window_start: int, f0: int, t_target: int
    ) -> torch.Tensor:
        """A peer block's latents for the memory slot, float32 ``[4, C, H, W]`` CPU tensor, sha-verified."""
        return torch.from_numpy(
            self.peek_block(media_id=media_id, window_start=window_start, f0=f0, t_target=t_target)
        )

    def peer_latents(self, block: Mapping[str, Any], t_target: int) -> np.ndarray:
        """:data:`worldcast.scene_state.state.PeerLatentsFn` over this pool: the latents of an admitted peer block."""
        return self.peek_block(
            media_id=str(block["media_id"]),
            window_start=int(block["window_start"]),
            f0=int(block["f0"]),
            t_target=int(t_target),
        )


class LocalDirPool(PoolBackend):
    """The pool on one directory shared by the clients of a session (use a fresh directory per session)::

        <root>/<media>/win_<window_start:06d>/blk_<f0:06d>.json   sidecar, written first
        <root>/<media>/win_<window_start:06d>/blk_<f0:06d>.npy    the block's 4 latents, float32, written last
        <root>/<media>/steps/step_<t_target:06d>.json             step record (withdrawn / resident orig_first)
        <root>/<media>/DONE.json                                  the writer publishes nothing more (ok | failed)

    Every file goes to a ``.pub-*`` temp file in its directory, is fsynced and renamed into place, so a reader sees
    nothing or a whole file. Republishing different bytes at an address raises. ``cell`` labels the sidecars and
    records and plays no part in any decision. Other arguments: :class:`PoolBackend`.
    """

    def __init__(
        self,
        root,
        *,
        client: str,
        stride: int,
        poll_s: float = 2.0,
        on_timeout: str = "fatal",
        cell: str = "",
    ) -> None:
        super().__init__(client=client, stride=stride, poll_s=poll_s, on_timeout=on_timeout)
        self.root = Path(root)
        self.cell = str(cell)
        self.root.mkdir(parents=True, exist_ok=True)
        # media -> {orig_first: (block, npy path, json path)}, as of the last refresh (plus this client's writes)
        self._index: dict[str, dict[int, tuple[PublishedBlock, Path, Path]]] = {}
        self._seen: set[str] = set()
        self._step_cache: dict[str, dict[int, StepRecord]] = {}
        self.n_published_blocks = 0
        self.n_publish_idempotent = 0
        self.refresh()

    def _media_dir(self, media_id: str) -> Path:
        return self.root / str(media_id)

    def _paths(self, media_id: str, window_start: int, f0: int) -> tuple[Path, Path]:
        d = self._media_dir(media_id) / f"win_{int(window_start):06d}"
        return d / f"blk_{int(f0):06d}.npy", d / f"blk_{int(f0):06d}.json"

    def _step_path(self, media_id: str, t_target: int) -> Path:
        return self._media_dir(media_id) / "steps" / f"step_{int(t_target):06d}.json"

    def _done_path(self, media_id: str) -> Path:
        return self._media_dir(media_id) / "DONE.json"

    @staticmethod
    def _atomic_write(dest: Path, write: Callable[[Any], Any]) -> None:
        """Write via a ``.pub-`` temp file in the destination directory, fsync, then ``os.replace``."""
        fd, tmp = tempfile.mkstemp(prefix=TMP_PREFIX, suffix=dest.suffix, dir=str(dest.parent))
        try:
            with os.fdopen(fd, "wb") as fh:
                write(fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, str(dest))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _block_of(meta: Mapping[str, Any]) -> PublishedBlock:
        return PublishedBlock(
            media_id=str(meta["media_id"]),
            window_start=int(meta["window_start"]),
            f0=int(meta["f0"]),
            orig_first=int(meta["orig_first"]),
            orig_last=int(meta["orig_last"]),
            n_latents=int(meta["n_latents"]),
            latents_sha256=str(meta["latents_sha256"]),
        )

    def publish_block(
        self,
        *,
        window_start: int,
        f0: int,
        latents,
        orig_first: int,
        orig_last: int,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Publish this client's finished block: sidecar first, float32 latents last, each atomically.

        ``latents``: ``[4, C, H, W]`` clean latents (tensor or array). ``orig_first`` / ``orig_last`` must follow the
        stride map. ``extra``: extra sidecar keys (the client stamps ``{"mode": "plain" | "reconstituted"}``).
        Republishing identical bytes is a no-op.
        """
        a = _float32(latents)
        if a.ndim != 4 or int(a.shape[0]) != BLOCK:
            raise ValueError(
                f"a published block is {BLOCK} latents [4, C, H, W], got {tuple(a.shape)}"
            )
        if int(orig_first) != int(window_start) + self.stride * int(f0):
            raise ValueError(
                f"orig_first {int(orig_first)} disagrees with window_start {int(window_start)} + "
                f"stride {self.stride} * f0 {int(f0)}"
            )
        if int(orig_last) != int(window_start) + self.stride * (int(f0) + BLOCK - 1):
            raise ValueError(f"orig_last {int(orig_last)} disagrees with the stride map")
        sha = block_sha(a)
        npy, js = self._paths(self.client, window_start, f0)
        npy.parent.mkdir(parents=True, exist_ok=True)
        if npy.exists() and js.exists():
            held = str(json.loads(js.read_text()).get("latents_sha256"))
            if held == sha:
                self.n_publish_idempotent += 1
                return
            raise PoolError(
                f"{npy} already holds a DIFFERENT block (sha {held[:16]} vs {sha[:16]}): one"
                " address, one block; use a fresh pool directory per session"
            )
        # producer, tree_sha and checkpoint_sha keep the sidecar schema of the paper run's pool.
        meta = dict(
            media_id=self.client,
            window_start=int(window_start),
            f0=int(f0),
            orig_first=int(orig_first),
            orig_last=int(orig_last),
            n_latents=BLOCK,
            latent_shape=[int(v) for v in a.shape[1:]],
            dtype="float32",
            latents_sha256=sha,
            per_latent_sha256=[latent_sha(a[j]) for j in range(BLOCK)],
            cell=self.cell,
            producer=self.client,
            tree_sha="",
            checkpoint_sha="",
        )
        if extra:
            meta.update(extra)
        meta_bytes = json.dumps(meta, sort_keys=True).encode()
        self._atomic_write(js, lambda fh: fh.write(meta_bytes))
        self._atomic_write(npy, lambda fh: np.save(fh, a))
        # a client knows its own writes without polling for them
        self._seen.add(str(npy))
        self._index.setdefault(self.client, {})[int(orig_first)] = (self._block_of(meta), npy, js)
        self.n_published_blocks += 1

    def publish_step(
        self, *, t_target: int, withdrawn: Sequence[int], resident: Sequence[int]
    ) -> None:
        """Publish this client's step record for ``t_target``; the identical record again is a no-op."""
        rec = dict(
            media_id=self.client,
            t_target=int(t_target),
            withdrawn=sorted(int(v) for v in withdrawn),
            resident=sorted(int(v) for v in resident),
            cell=self.cell,
        )
        body = json.dumps(rec, sort_keys=True).encode()
        path = self._step_path(self.client, t_target)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() == body:
                return
            raise PoolError(
                f"{path} already holds a DIFFERENT step record: one writer, one record per block"
            )
        self._atomic_write(path, lambda fh: fh.write(body))

    def mark_done(self, *, status: str = DONE_OK, note: str = "") -> None:
        if str(status) not in (DONE_OK, DONE_FAILED):
            raise ValueError(f"status must be {DONE_OK!r} or {DONE_FAILED!r}, got {status!r}")
        self._media_dir(self.client).mkdir(parents=True, exist_ok=True)
        meta = dict(
            media_id=self.client,
            status=str(status),
            note=str(note),
            n_published_blocks=int(self.n_published_blocks),
            last_orig_last=self.progress(self.client),
            cell=self.cell,
        )
        self._atomic_write(
            self._done_path(self.client),
            lambda fh: fh.write(json.dumps(meta, sort_keys=True).encode()),
        )

    def refresh(self) -> int:
        n_new = 0
        for media_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            idx = self._index.setdefault(media_dir.name, {})
            for win_dir in sorted(
                p for p in media_dir.iterdir() if p.is_dir() and p.name.startswith("win_")
            ):
                for npy in sorted(win_dir.glob("blk_*.npy")):
                    key = str(npy)
                    if key in self._seen:
                        continue
                    js = npy.with_suffix(".json")
                    if not js.exists():
                        continue  # the sidecar lands first: an .npy without one is not published
                    meta = json.loads(js.read_text())
                    self._seen.add(key)
                    idx[int(meta["orig_first"])] = (self._block_of(meta), npy, js)
                    n_new += 1
        return n_new

    def published_blocks(self, media_id: str) -> list[PublishedBlock]:
        idx = self._index.get(str(media_id)) or {}
        return [idx[o][0] for o in sorted(idx)]

    def read_steps(self, media_id: str, *, upto: int) -> list[StepRecord]:
        d = self._media_dir(media_id) / "steps"
        if not d.is_dir():
            return []
        cache = self._step_cache.setdefault(str(media_id), {})
        out = []
        for p in sorted(d.glob("step_*.json")):
            t = int(p.stem.split("_", 1)[1])
            if t > int(upto):
                continue
            if t not in cache:
                r = json.loads(p.read_text())
                cache[t] = StepRecord(
                    media_id=str(r["media_id"]),
                    t_target=int(r["t_target"]),
                    withdrawn=tuple(int(v) for v in r["withdrawn"]),
                    resident=tuple(int(v) for v in r["resident"]),
                )
            out.append(cache[t])
        return sorted(out, key=lambda r: int(r.t_target))

    def has_step(self, media_id: str, t_target: int) -> bool:
        return self._step_path(media_id, t_target).exists()

    def done_status(self, media_id: str) -> str | None:
        p = self._done_path(media_id)
        if not p.exists():
            return None
        return str(json.loads(p.read_text())["status"])

    def _read_block(self, block: PublishedBlock) -> np.ndarray:
        got = (self._index.get(block.media_id) or {}).get(int(block.orig_first))
        if got is None:
            raise PoolError(
                f"{block.media_id} @ orig {block.orig_first} is not in this client's index"
            )
        _, npy, js = got
        meta = json.loads(js.read_text())
        arr = np.load(npy)
        n = int(meta["n_latents"])
        if arr.ndim != 4 or int(arr.shape[0]) != n:
            raise PoolError(f"{npy}: latents are {arr.shape}, expected [{n}, C, H, W]")
        g32 = np.ascontiguousarray(arr, dtype=np.float32)
        got_sha = block_sha(g32)
        if got_sha != str(meta["latents_sha256"]):
            raise PoolError(
                f"{npy}: sha256 of the bytes read is {got_sha[:16]}, the writer stamped "
                f"{str(meta['latents_sha256'])[:16]} (truncated, overwritten or corrupted)"
            )
        return g32


#: ``(slot, media_id, window_start, f0) -> (block | None, reason)``: the reader's candidate for a block a peer
#: published at its own address (its recorded cameras ``c2w`` [4, 4, 4] and the slot material's data), or ``None``
#: with ``"dead"`` / ``"uncovered"``.
CandidateFn = Callable[[int, str, int, int], tuple[dict[str, Any] | None, str]]


class PeerBlocks:
    """The finished blocks one client knows: its own and the peers' blocks it admitted.

    Append-only; a block's index is its address in the scene state (:attr:`worldcast.scene_state.bank.MemoryEntry.block`).
    The order breaks retrieval ties: the six prefix blocks, then per block the peer blocks admitted at that step in
    ``(slot, media, orig_first)`` order, then the client's own new block.

    Args:
        ego_media, ego_slot: this client's media id and player slot.
        sources: ``{media_id: slot}`` of every other player whose published blocks may be admitted.
        candidate_at: :data:`CandidateFn`.
    """

    def __init__(
        self,
        *,
        ego_media: str,
        ego_slot: int,
        sources: Mapping[str, int],
        candidate_at: CandidateFn,
    ) -> None:
        self.ego_media = str(ego_media)
        self.ego_slot = int(ego_slot)
        self.sources: dict[str, int] = {str(m): int(s) for m, s in sources.items()}
        if self.ego_media in self.sources or self.ego_slot in self.sources.values():
            raise ValueError("sources must not contain this client's own media or slot")
        self.candidate_at = candidate_at
        self.blocks: list[dict[str, Any]] = []
        #: ``(media, orig_first)`` of published blocks never admitted (``candidate_at`` returned None).
        self.skipped: set[tuple[str, int]] = set()
        self._seen: dict[tuple[str, int], str] = {}  # -> "pending" | "admitted" | skip reason
        self._pending: dict[tuple[str, int], dict[str, Any]] = {}
        self.n_admitted = 0

    @property
    def peers(self) -> list[str]:
        """Every source media id, sorted (the writers whose withdrawals a client follows)."""
        return sorted(self.sources)

    def add_own(self, block: Mapping[str, Any]) -> int:
        """Append this client's generated block (``media_id``, ``slot``, ``window_start``, ``f0``, ``orig_first``,
        ``orig_last``, ``c2w``); returns its index."""
        if str(block["media_id"]) != self.ego_media:
            raise ValueError(
                f"add_own got a block of {block['media_id']!r}, this client is {self.ego_media!r}"
            )
        entry = dict(block, kind="generated", own_window=False, ego=True)
        entry.setdefault("payload_provenance", "generated")
        self.blocks.append(entry)
        return len(self.blocks) - 1

    def admit(
        self, pool: PoolBackend, *, t_target: int, peers: Sequence[str], max_wait_s: float
    ) -> int:
        """Wait for ``peers`` (:meth:`PoolBackend.wait_for_peers`), then admit every peer block published at its
        writer's address with ``orig_last < t_target``; returns the number admitted.

        Raises :class:`PoolError` when a candidate spans other frames than its published block.
        """
        if peers:
            pool.wait_for_peers(
                list(peers),
                need_orig_last=int(t_target) - int(pool.stride),
                need_step=int(t_target),
                max_wait_s=float(max_wait_s),
            )
        pool.refresh()
        n = 0
        for media, slot in sorted(self.sources.items(), key=lambda kv: (int(kv[1]), str(kv[0]))):
            for meta in pool.published_blocks(media):
                key = (str(media), int(meta.orig_first))
                if key not in self._seen:
                    cand, why = self.candidate_at(
                        int(slot), str(media), int(meta.window_start), int(meta.f0)
                    )
                    if cand is not None and (
                        int(cand["orig_first"]) != int(meta.orig_first)
                        or int(cand["orig_last"]) != int(meta.orig_last)
                    ):
                        raise PoolError(
                            f"{key}: the candidate built at the writer's address spans "
                            f"{cand['orig_first']}..{cand['orig_last']}, the published block "
                            f"{meta.orig_first}..{meta.orig_last}"
                        )
                    if cand is None:
                        self._seen[key] = str(why)
                        self.skipped.add(key)
                        continue
                    self._seen[key] = "pending"
                    self._pending[key] = dict(
                        cand,
                        own_window=False,
                        payload_provenance="generated",
                        address="published",
                        published_orig_first=int(meta.orig_first),
                    )
                if self._seen[key] != "pending":
                    continue
                b = self._pending[key]
                if not pool.covers(
                    media_id=str(b["media_id"]),
                    window_start=int(b["window_start"]),
                    f0=int(b["f0"]),
                    t_target=int(t_target),
                ):
                    continue  # its causal cut has not passed yet (the writer is ahead)
                del self._pending[key]
                self._seen[key] = "admitted"
                self.blocks.append(dict(b, kind="generated", own_window=False, ego=False))
                self.n_admitted += 1
                n += 1
        return n

    def eligible(self, *, t_target: int) -> list[int]:
        """Indices a memory slot may come from for the block at ``t_target``: own blocks with ``orig_last <
        t_target`` and, as deployed, peer blocks with ``orig_last <= t_target``."""

        def ok(b: Mapping[str, Any]) -> bool:
            own = str(b["media_id"]) == self.ego_media or int(b["slot"]) == self.ego_slot
            last = int(b["orig_last"])
            return last < int(t_target) if own else last <= int(t_target)

        return [i for i, b in enumerate(self.blocks) if ok(b)]
