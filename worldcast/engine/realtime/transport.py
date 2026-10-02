"""What an engine sends to and receives from its peers, and the in-memory pool that turns messages into the release
pool protocol (:class:`worldcast.engine.inference.pool.PoolBackend`).

Per block an engine emits (paper App. "Deployment protocol"):

* :class:`PlayerStateMessage`: its player's state over the block (positions at the four latent frames, facing,
  alive) and the block's controls;
* :class:`SceneBlockMessage`: the block's four clean latents, once the block is finished (a memory entry; peers
  recompute its depth themselves);
* :class:`StepMessage`: the own memory entries it withdrew at the block's write and the ones it holds afterwards;
* :class:`PositionMessage` (closed loop only): its own position estimate at the latent frames of the previous block
  (the state model and Eq. 6), sent before the block starts;
* :class:`DoneMessage` when it stops.

The network layer delivers peers' messages with :meth:`MessagePool.receive` (any thread) and ships the engine's own
messages from its ``on_message`` callback. Messages are plain dataclasses of numpy arrays and ints; :func:`encode`
and :func:`decode` give a compact binary form.
"""

import hashlib
import io
import pickle
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from worldcast.engine.inference.pool import (
    BLOCK,
    DONE_FAILED,
    DONE_OK,
    PoolBackend,
    PoolError,
    PublishedBlock,
    StepRecord,
)

__all__ = [
    "PlayerStateMessage",
    "SceneBlockMessage",
    "StepMessage",
    "PositionMessage",
    "DoneMessage",
    "Message",
    "MessagePool",
    "MessageStateExchange",
    "encode",
    "decode",
]


@dataclass
class PlayerStateMessage:
    """A player's state over one block: ``states`` ``[4, 6]`` (x, y, z, yaw, pitch, alive) at the block's four
    latent frames, ``controls`` the block's 16 rows (buttons ``[16, 11]``, encoded turn ``[16, 2]``, weapon
    ``[16]``) as a dict of arrays."""

    media_id: str
    slot: int
    f0: int
    orig_first: int
    states: np.ndarray
    controls: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SceneBlockMessage:
    """A finished block of the writer: four clean latents ``[4, 48, 24, 42]`` float32 at its own address."""

    media_id: str
    window_start: int
    f0: int
    orig_first: int
    orig_last: int
    latents: np.ndarray
    latents_sha256: str = ""

    def __post_init__(self) -> None:
        self.latents = np.ascontiguousarray(self.latents, dtype=np.float32)
        if not self.latents_sha256:
            self.latents_sha256 = hashlib.sha256(self.latents.tobytes()).hexdigest()


@dataclass
class StepMessage:
    """The writer's step record for its block at ``t_target`` (published ``orig_first`` lists)."""

    media_id: str
    t_target: int
    withdrawn: tuple[int, ...]
    resident: tuple[int, ...]


@dataclass
class PositionMessage:
    """A closed-loop client's position record for its block at ``t_target``: ``knots`` (latent frames) and their
    estimated feet positions ``xyz`` ``[n, 3]`` (:class:`worldcast.player_state.closed_loop.StateExchange` as a message).
    """

    media_id: str
    t_target: int
    knots: tuple[int, ...]
    xyz: np.ndarray


@dataclass
class DoneMessage:
    media_id: str
    status: str = DONE_OK
    note: str = ""


Message = Any


def encode(message: Message) -> bytes:
    """Bytes of a message (pickle of the dataclass; arrays as raw buffers)."""
    return pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)


def decode(data: bytes) -> Message:
    """The message encoded by :func:`encode` (only these four classes are accepted)."""
    allowed = {
        c.__name__: c
        for c in (PlayerStateMessage, SceneBlockMessage, StepMessage, PositionMessage, DoneMessage)
    }

    class _Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if module == __name__ and name in allowed:
                return allowed[name]
            if module.startswith("numpy") or (
                module == "builtins" and name in ("tuple", "dict", "list")
            ):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"refusing {module}.{name}")

    return _Restricted(io.BytesIO(data)).load()


class MessagePool(PoolBackend):
    """The release pool protocol over messages (no shared file system).

    Own publishes are kept and handed to ``outbox``; peers' messages arrive through :meth:`receive` and become
    visible at the next :meth:`refresh` (the lock-step wait polls it). ``player_states`` keeps the latest
    :class:`PlayerStateMessage` of every peer.
    """

    def __init__(
        self,
        *,
        client: str,
        stride: int,
        outbox: Callable[[Message], None] | None = None,
        poll_s: float = 0.002,
        on_timeout: str = "fatal",
    ) -> None:
        super().__init__(client=client, stride=stride, poll_s=poll_s, on_timeout=on_timeout)
        self.outbox = outbox
        self._inbox: list[Message] = []
        self._lock = threading.Lock()
        self._blocks: dict[str, dict[int, tuple[PublishedBlock, np.ndarray]]] = {}
        self._steps: dict[str, dict[int, StepRecord]] = {}
        self._done: dict[str, str] = {}
        self.player_states: dict[str, PlayerStateMessage] = {}
        self.positions: dict[tuple[str, int], PositionMessage] = {}

    # -- delivery -----------------------------------------------------------------------------------------------
    def receive(self, message: Message) -> None:
        """Deliver a peer's message (thread-safe; applied at the next refresh)."""
        if getattr(message, "media_id", None) == self.client:
            return
        with self._lock:
            self._inbox.append(message)

    def _send(self, message: Message) -> None:
        if self.outbox is not None:
            self.outbox(message)

    def _apply(self, m: Message) -> int:
        if isinstance(m, SceneBlockMessage):
            if (
                hashlib.sha256(np.ascontiguousarray(m.latents, np.float32).tobytes()).hexdigest()
                != m.latents_sha256
            ):
                raise PoolError(f"{m.media_id} @ {m.orig_first}: latents do not match their sha256")
            if int(m.orig_first) in self._blocks.get(m.media_id, {}):
                return 0
            meta = PublishedBlock(
                media_id=m.media_id,
                window_start=int(m.window_start),
                f0=int(m.f0),
                orig_first=int(m.orig_first),
                orig_last=int(m.orig_last),
                n_latents=BLOCK,
                latents_sha256=m.latents_sha256,
            )
            self._blocks.setdefault(m.media_id, {})[int(m.orig_first)] = (meta, m.latents)
            return 1
        if isinstance(m, StepMessage):
            self._steps.setdefault(m.media_id, {})[int(m.t_target)] = StepRecord(
                media_id=m.media_id,
                t_target=int(m.t_target),
                withdrawn=tuple(int(v) for v in m.withdrawn),
                resident=tuple(int(v) for v in m.resident),
            )
        elif isinstance(m, DoneMessage):
            self._done[m.media_id] = str(m.status)
        elif isinstance(m, PositionMessage):
            self.positions[(m.media_id, int(m.t_target))] = m
        elif isinstance(m, PlayerStateMessage):
            prev = self.player_states.get(m.media_id)
            if prev is None or int(m.f0) >= int(prev.f0):
                self.player_states[m.media_id] = m
        return 0

    # -- writer -------------------------------------------------------------------------------------------------
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
        a = (
            latents.detach().cpu().float().numpy()
            if hasattr(latents, "detach")
            else np.asarray(latents, np.float32)
        )
        if a.ndim != 4 or int(a.shape[0]) != BLOCK:
            raise ValueError(
                f"a published block is {BLOCK} latents [4, C, H, W], got {tuple(a.shape)}"
            )
        if int(orig_first) != int(window_start) + self.stride * int(f0):
            raise ValueError("orig_first disagrees with the stride map")
        msg = SceneBlockMessage(
            media_id=self.client,
            window_start=int(window_start),
            f0=int(f0),
            orig_first=int(orig_first),
            orig_last=int(orig_last),
            latents=a,
        )
        self._apply(msg)
        self._send(msg)

    def publish_step(
        self, *, t_target: int, withdrawn: Sequence[int], resident: Sequence[int]
    ) -> None:
        msg = StepMessage(
            media_id=self.client,
            t_target=int(t_target),
            withdrawn=tuple(sorted(int(v) for v in withdrawn)),
            resident=tuple(sorted(int(v) for v in resident)),
        )
        self._apply(msg)
        self._send(msg)

    def publish_player_state(self, message: PlayerStateMessage) -> None:
        self._send(message)

    def publish_position(self, message: PositionMessage) -> None:
        self._apply(message)
        self._send(message)

    def mark_done(self, *, status: str = DONE_OK, note: str = "") -> None:
        if str(status) not in (DONE_OK, DONE_FAILED):
            raise ValueError(f"status must be {DONE_OK!r} or {DONE_FAILED!r}")
        msg = DoneMessage(media_id=self.client, status=str(status), note=str(note))
        self._apply(msg)
        self._send(msg)

    # -- reader -------------------------------------------------------------------------------------------------
    def refresh(self) -> int:
        with self._lock:
            pending, self._inbox = self._inbox, []
        return sum(self._apply(m) for m in pending)

    def published_blocks(self, media_id: str) -> list[PublishedBlock]:
        idx = self._blocks.get(str(media_id)) or {}
        return [idx[o][0] for o in sorted(idx)]

    def read_steps(self, media_id: str, *, upto: int) -> list[StepRecord]:
        recs = self._steps.get(str(media_id)) or {}
        return [recs[t] for t in sorted(recs) if t <= int(upto)]

    def has_step(self, media_id: str, t_target: int) -> bool:
        return int(t_target) in (self._steps.get(str(media_id)) or {})

    def done_status(self, media_id: str) -> str | None:
        return self._done.get(str(media_id))

    def _read_block(self, block: PublishedBlock) -> np.ndarray:
        got = (self._blocks.get(block.media_id) or {}).get(int(block.orig_first))
        if got is None:
            raise PoolError(f"{block.media_id} @ orig {block.orig_first} has not arrived")
        return got[1]


class MessageStateExchange:
    """:class:`worldcast.player_state.closed_loop.StateExchange` over a :class:`MessagePool` (same methods)."""

    def __init__(self, pool: MessagePool, *, fatal: bool = True) -> None:
        self.pool, self.client, self.fatal = pool, pool.client, bool(fatal)

    def publish(self, t_target: int, knots: Sequence[int], xyz: np.ndarray) -> None:
        self.pool.publish_position(
            PositionMessage(
                media_id=self.client,
                t_target=int(t_target),
                knots=tuple(int(k) for k in knots),
                xyz=np.asarray(xyz, np.float64).reshape(-1, 3),
            )
        )

    def read(self, client: str, t_target: int):
        """``(knots, xyz [n, 3])`` of ``client`` at block ``t_target``, or ``None`` if it has not arrived."""
        self.pool.refresh()
        m = self.pool.positions.get((str(client), int(t_target)))
        return (
            None
            if m is None
            else ([int(k) for k in m.knots], np.asarray(m.xyz, np.float64).reshape(-1, 3))
        )

    def wait(self, peers: Sequence[str], t_target: int, *, max_wait_s: float, is_done) -> None:
        import time

        start = time.monotonic()
        while True:
            self.pool.refresh()
            late = [
                p
                for p in peers
                if not is_done(p) and (str(p), int(t_target)) not in self.pool.positions
            ]
            if not late:
                return
            if time.monotonic() - start >= float(max_wait_s):
                if self.fatal:
                    raise TimeoutError(f"no position record of {late} for block {t_target}")
                return
            time.sleep(self.pool.poll_s)
