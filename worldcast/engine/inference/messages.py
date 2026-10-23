"""The shared world state over the network (App. B, "Deployment protocol"): the live demo.

The paper's client publishes one message per block. Clients without a shared file system send it
in parts, each as soon as it exists: the new memory entry when a block is finished, the withdrawals
and the estimated position before the next one. In live play (``player_state.source: live``) a
block's player state leaves as soon as the block's controls are taken: the rows its client's camera
uses, with the block's controls (the ``live`` part). A part is a :class:`MessagePart`; each client
keeps the shared world state in memory (:class:`MessageWorldState`).

The times of a ``live`` part are session time: s on the clock all clients of a session share (the
live demo's room clock), so the parts of different clients can be paired by time.
"""

import bisect
import json
import math
import struct
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

import numpy as np

from worldcast.data.controls import button_bits, check_weapon_ids
from worldcast.data.latents import VIDEO_FRAMES_PER_BLOCK
from worldcast.utils.fingerprints import sha256_float32

from .world_state import PublishedBlock, StepRecord, WorldState

__all__ = ["LIVE_STATES_PER_ENTRY", "MessagePart", "MessageWorldState"]

#: What a part of each kind carries, besides the client's ``media_id``.
_CARRIES = {
    "scene": ("block", "latents"),
    "step": ("step",),
    "position": ("t_target", "latent_frames", "xyz"),
    "done": ("status",),
    "live": ("f0", "times", "rows", "buttons", "turn", "weapon"),
}
#: The type of each field a part can carry.
_TYPES = {
    "f0": int,
    "block": PublishedBlock,
    "latents": np.ndarray,
    "step": StepRecord,
    "t_target": int,
    "latent_frames": tuple,
    "xyz": np.ndarray,
    "status": str,
    "times": np.ndarray,
    "rows": np.ndarray,
    "buttons": np.ndarray,
    "turn": np.ndarray,
    "weapon": np.ndarray,
}
#: The dtype and shape of each array of a ``live`` part: one row per video frame of the block.
_LIVE_ARRAYS = {
    "times": (np.dtype(np.float64), (VIDEO_FRAMES_PER_BLOCK,)),
    "rows": (np.dtype(np.float32), (VIDEO_FRAMES_PER_BLOCK, 5)),
    "buttons": (np.dtype(np.uint16), (VIDEO_FRAMES_PER_BLOCK,)),
    "turn": (np.dtype(np.float32), (VIDEO_FRAMES_PER_BLOCK, 2)),
    "weapon": (np.dtype(np.uint8), (VIDEO_FRAMES_PER_BLOCK,)),
}
#: The length of the JSON header, before it.
_HEADER_LENGTH = struct.Struct(">I")
#: The live states kept of each player, per memory entry kept. A live state leaves at its block's
#: controls cut, before the block's memory entry, and the first six entries leave together, after
#: the sixth block: twice as many live states as entries hold the cameras of every kept entry.
LIVE_STATES_PER_ENTRY = 2


@dataclass
class MessagePart:
    """One part of a client's message, as it travels over the network.

    Attributes:
        media_id (str): the client.
        kind (str): which part, and the fields it carries: ``scene`` (``block``, ``latents``), a
            new memory entry; ``step`` (``step``), the withdrawals; ``position`` (``t_target``,
            ``latent_frames``, ``xyz``), the estimated positions of the closed loop; ``done``
            (``status``); ``live`` (``f0``, ``times``, ``rows``, ``buttons``, ``turn``,
            ``weapon``), the player state of a block in live play, with the block's controls.
        f0 (int | None): the first latent frame of the block the ``live`` rows belong to.
        block (PublishedBlock | None): the block published as a memory entry.
        latents (np.ndarray | None): ``[4, 48, 24, 42]`` float32, its latents.
        step (StepRecord | None): the memory entries the client withdrew.
        t_target (int | None): the source frame of the block ``xyz`` is published for.
        latent_frames (tuple[int, ...] | None): the latent frames of ``xyz``.
        xyz (np.ndarray | None): ``[n, 3]`` float64 estimated positions, u.
        status (str | None): ``ok`` or ``failed``: the client publishes nothing more.
        times (np.ndarray | None): ``[16]`` float64, the session time of the block's video
            frames, s, increasing.
        rows (np.ndarray | None): ``[16, 5]`` float32 (x, y, z, yaw, pitch) at the block's video
            frames, u and degrees: the player state the client's camera follows.
        buttons (np.ndarray | None): ``[16]`` uint16, the held buttons of each frame, bit ``i``
            for ``CONTROL_BUTTONS[i]`` (:func:`~worldcast.data.controls.buttons_from_bits`).
        turn (np.ndarray | None): ``[16, 2]`` float32, degrees turned during each frame (pitch,
            yaw).
        weapon (np.ndarray | None): ``[16]`` uint8, the held weapon's id at each frame
            (:data:`~worldcast.data.controls.OPENCS2_WEAPONS`).

    Raises:
        ValueError: the part carries other fields than those of its kind, a field of another
            type, an array of a ``live`` part of another dtype or shape, a non-finite time, row
            or turn, times that do not increase, a weapon id outside the vocabulary, or a record
            of another client.
    """

    media_id: str
    kind: str
    f0: int | None = None
    block: PublishedBlock | None = None
    latents: np.ndarray | None = None
    step: StepRecord | None = None
    t_target: int | None = None
    latent_frames: tuple[int, ...] | None = None
    xyz: np.ndarray | None = None
    status: str | None = None
    times: np.ndarray | None = None
    rows: np.ndarray | None = None
    buttons: np.ndarray | None = None
    turn: np.ndarray | None = None
    weapon: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.media_id, str) or self.kind not in _CARRIES:
            raise ValueError(
                f"a part has a client and a kind of {tuple(_CARRIES)}, got {self.media_id!r} and"
                f" {self.kind!r}"
            )
        carried = {name: getattr(self, name) for name in _TYPES if getattr(self, name) is not None}
        if set(carried) != set(_CARRIES[self.kind]):
            raise ValueError(
                f"a {self.kind!r} part carries {_CARRIES[self.kind]}, got {sorted(carried)}"
            )
        wrong = [name for name, value in carried.items() if not isinstance(value, _TYPES[name])]
        if wrong:
            raise ValueError(f"a {self.kind!r} part with {wrong} of another type")
        if self.kind == "live":
            wrong = [
                f"{name} {carried[name].dtype} {carried[name].shape}"
                for name, spec in _LIVE_ARRAYS.items()
                if (carried[name].dtype, carried[name].shape) != spec
            ]
            if wrong:
                expected = ", ".join(f"{n} {d} {s}" for n, (d, s) in _LIVE_ARRAYS.items())
                raise ValueError(f"a 'live' part carries {expected}; got {wrong}")
            infinite = [
                name
                for name, (dtype, _) in _LIVE_ARRAYS.items()
                if dtype.kind == "f" and not np.isfinite(carried[name]).all()
            ]
            if infinite:
                raise ValueError(f"a 'live' part with non-finite {infinite}")
            if (np.diff(self.times) <= 0).any():
                raise ValueError(f"the times of a 'live' part increase; got {self.times}")
            check_weapon_ids(self.weapon)
        record = self.block or self.step
        if record is not None and record.media_id != self.media_id:
            raise ValueError(f"a part of {self.media_id!r} with a record of {record.media_id!r}")

    def encode(self) -> bytes:
        """The part's bytes: the length of a JSON header, the header (the client, the kind and
        the fields of the kind; for an array its dtype and shape), then the arrays' bytes."""
        header: dict[str, Any] = {"media_id": self.media_id, "kind": self.kind}
        arrays = []
        for name in _CARRIES[self.kind]:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                arrays.append(np.ascontiguousarray(value))
                value = (arrays[-1].dtype.str, arrays[-1].shape)
            header[name] = asdict(value) if is_dataclass(value) else value
        head = json.dumps(header, separators=(",", ":")).encode()
        return b"".join([_HEADER_LENGTH.pack(len(head)), head, *(a.tobytes() for a in arrays)])

    @classmethod
    def decode(cls, data: bytes) -> "MessagePart":
        """The part of :meth:`encode`. The bytes are parsed as JSON and numeric arrays, nothing
        else.

        Raises:
            ValueError: ``data`` is not a part.
        """
        try:
            (length,) = _HEADER_LENGTH.unpack_from(data)
            offset = _HEADER_LENGTH.size + length
            header = json.loads(data[_HEADER_LENGTH.size : offset])
            carried: dict[str, Any] = {}
            for name in _CARRIES[header["kind"]]:
                value, kind = header[name], _TYPES[name]
                if kind is np.ndarray:
                    value = _array(data, offset, *value)
                    offset += value.nbytes
                elif kind is tuple:
                    value = _int_tuple(value)
                elif is_dataclass(kind):
                    value = _record(kind, value)
                elif type(value) is not kind:
                    raise ValueError(f"{name} is {value!r}")
                carried[name] = value
            if offset != len(data) or len(header) != 2 + len(carried):
                raise ValueError("bytes or header fields the kind does not carry")
            return cls(header["media_id"], header["kind"], **carried)
        except (KeyError, TypeError, RecursionError, struct.error) as error:
            raise ValueError(f"not a message part: {error!r}") from error


def _int_tuple(value: Any) -> tuple[int, ...]:
    """A JSON list of integers, as a tuple."""
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"not a list of integers: {value!r}")
    return tuple(value)


def _array(data: bytes, offset: int, dtype: Any, shape: Any) -> np.ndarray:
    """A copy of the numeric array of ``dtype`` and ``shape`` whose bytes start at ``offset``."""
    shape = _int_tuple(shape)
    if not isinstance(dtype, str) or np.dtype(dtype).kind not in "fiu" or min(shape, default=0) < 0:
        raise ValueError(f"not a numeric array: dtype {dtype!r}, shape {shape}")
    count = math.prod(shape)
    if count * np.dtype(dtype).itemsize > len(data) - offset:
        raise ValueError(f"an array of shape {shape} needs more bytes than the part has")
    return np.frombuffer(data, dtype, count, offset).reshape(shape).copy()


def _time_order(part: MessagePart) -> tuple[float, int]:
    """A ``live`` part's place among its client's: the session time of its last frame, then its
    block."""
    return float(part.times[-1]), part.f0


def _record(record: type, values: Any) -> Any:
    """``record(**values)`` for a JSON object with exactly the fields of the dataclass ``record``,
    each an integer, a string or a list of integers as the field is annotated."""
    if not isinstance(values, dict) or sorted(values) != sorted(f.name for f in fields(record)):
        raise ValueError(f"not a {record.__name__}: {values!r}")
    typed = {}
    for field in fields(record):
        value = values[field.name]
        if field.type not in (int, str):
            value = _int_tuple(value)
        elif type(value) is not field.type:
            raise ValueError(f"{record.__name__}.{field.name} is {value!r}")
        typed[field.name] = value
    return record(**typed)


class MessageWorldState(WorldState):
    """The shared world state in memory, over messages (no shared file system).

    What this client publishes is applied locally and handed to ``outbox`` as
    :class:`MessagePart` records; the other clients' parts arrive through :meth:`receive` and are
    applied at the next :meth:`refresh`, which the lockstep wait polls.

    ``scene`` parts are first-wins by ``(client, t_first)``. ``live`` parts are never dropped, so a
    memory entry admitted late still finds the rows of its cameras, and are kept twice:

    * by their block, for the cameras of the block's memory entry (:meth:`live_part`): the
      block's newest part in session time until the entry's ``scene`` part is applied, then the
      part it found. A client's ``live`` part leaves before its ``scene`` part, on one channel,
      so an entry and its cameras come from the same session, even after the seat is taken
      again;
    * by the session time of their last frame, for the rows (:meth:`recent_live`).

    A seat taken again starts its rollout over at its first block. So a ``live`` part newer in
    session time than every part of its client applied before, at a block not after the newest
    one's, clears the client's ``done``, and :meth:`~WorldState.is_done` turns False again. A
    leaver's part that arrives after its ``done`` continues its blocks and leaves ``done`` set.

    A session without an end keeps only the newest ``retain`` blocks of each client: applying a
    part drops, from every index, what is older (the ``scene``, ``step`` and ``position`` parts
    by their blocks and the ``live`` parts by their session time); a block's ``live`` part that
    keyed its memory entry goes with the entry's ``scene`` part, not before. What a reader still
    needs of an older block it copied when it admitted it
    (:class:`~worldcast.engine.inference.memory_blocks.LiveMemoryBlocks`).

    Args:
        client (str): this client's media id (:class:`WorldState`).
        poll_s (float): the lockstep poll period, s.
        outbox (Callable[[MessagePart], None] | None): sends a part to the other clients.
        retain (int | None): blocks kept of each client; ``None``: all.
    """

    def __init__(
        self,
        *,
        client: str,
        poll_s: float,
        outbox: Callable[[MessagePart], None] | None = None,
        retain: int | None = None,
    ) -> None:
        super().__init__(client=client, poll_s=poll_s)
        self.outbox, self.retain = outbox, retain
        self._inbox: list[tuple[MessagePart, str | None]] = []
        self._lock = threading.Lock()
        self._blocks: dict[str, dict[int, MessagePart]] = {}
        self._received_sha256: dict[tuple[str, int], str] = {}
        self._steps: dict[str, dict[int, StepRecord]] = {}
        self._positions: dict[tuple[str, int], MessagePart] = {}
        self._done: dict[str, str] = {}
        self._live: dict[tuple[str, int], MessagePart] = {}
        self._entry_live: dict[tuple[str, int], MessagePart] = {}
        self._live_by_time: dict[str, list[MessagePart]] = {}

    def receive(self, part: MessagePart) -> None:
        """Deliver another client's part (any thread; applied at the next refresh). A memory
        entry's latents are hashed here, on the delivering thread, and the hash is checked against
        the one their owner stamped when they are read (:meth:`WorldState.block_latents`)."""
        if part.media_id != self.client:
            received = sha256_float32(part.latents) if part.kind == "scene" else None
            with self._lock:
                self._inbox.append((part, received))

    def _send(self, kind: str, **carried: Any) -> None:
        """Apply one of this client's parts and hand it to the outbox."""
        part = MessagePart(self.client, kind, **carried)
        self._apply(part)
        if self.outbox is not None:
            self.outbox(part)

    def _apply(self, part: MessagePart, received: str | None = None) -> None:
        if part.kind == "scene":
            entries = self._blocks.setdefault(part.media_id, {})
            if part.block.t_first not in entries:
                entries[part.block.t_first] = part
                if received is not None:
                    self._received_sha256[(part.media_id, part.block.t_first)] = received
                key = (part.media_id, part.block.f0)
                if key in self._live:
                    self._entry_live[key] = self._live[key]
        elif part.kind == "step":
            self._steps.setdefault(part.media_id, {})[part.step.t_target] = part.step
        elif part.kind == "position":
            self._positions[(part.media_id, part.t_target)] = part
        elif part.kind == "done":
            self._done[part.media_id] = part.status
        elif part.kind == "live":
            by_time = self._live_by_time.setdefault(part.media_id, [])
            at = bisect.bisect_left(by_time, _time_order(part), key=_time_order)
            if at < len(by_time) and _time_order(by_time[at]) == _time_order(part):
                return  # the same part, delivered again
            if at == len(by_time) and (not by_time or part.f0 <= by_time[-1].f0):
                self._done.pop(part.media_id, None)  # the seat is taken again
            by_time.insert(at, part)
            key = (part.media_id, part.f0)
            if key not in self._live or _time_order(self._live[key]) < _time_order(part):
                self._live[key] = part
        if self.retain is not None:
            self._retain_newest(part.media_id)

    def _retain_newest(self, media_id: str) -> None:
        """Drop from every index what is older than the newest ``retain`` blocks of ``media_id``."""
        keep = self.retain
        scenes = self._blocks.get(media_id, {})
        for t_first in sorted(scenes)[:-keep]:
            self._entry_live.pop((media_id, scenes.pop(t_first).block.f0), None)
            self._received_sha256.pop((media_id, t_first), None)
        steps = self._steps.get(media_id, {})
        for t_target in sorted(steps)[:-keep]:
            del steps[t_target]
        for t_target in sorted(t for media, t in self._positions if media == media_id)[:-keep]:
            del self._positions[(media_id, t_target)]
        lives = self._live_by_time.get(media_id, [])
        for part in lives[:-keep]:
            if self._live.get((media_id, part.f0)) is part:
                del self._live[(media_id, part.f0)]
        del lives[:-keep]

    # -------------------------------------------------------------------------------- publishing
    def _store_block(self, block: PublishedBlock, latents: np.ndarray) -> None:
        self._send("scene", block=block, latents=latents)

    def _store_step(self, record: StepRecord) -> None:
        self._send("step", step=record)

    def _store_position(self, t_target: int, latent_frames: list[int], xyz: np.ndarray) -> None:
        self._send("position", t_target=t_target, latent_frames=tuple(latent_frames), xyz=xyz)

    def _store_done(self, status: str, note: str) -> None:
        """The status alone travels: a failed client's note stays in its own log."""
        self._send("done", status=status)

    def publish_live_state(
        self,
        f0: int,
        *,
        times: np.ndarray,
        rows: np.ndarray,
        buttons: np.ndarray,
        turn: np.ndarray,
        weapon: np.ndarray,
    ) -> None:
        """Publish this client's player state of block ``f0`` in live play
        (``player_state.source: live``): the rows its camera uses at the block's video frames,
        with the block's controls. This is the paper's player state of the block (App. B,
        "Deployment protocol") as rows: they go on from the positions of the block before, with
        the facing, the controls and the weapon of the block.

        Args:
            f0 (int): the block's first latent frame.
            times (np.ndarray): ``[16]`` the session time of the block's video frames, s,
                increasing.
            rows (np.ndarray): ``[16, 5]`` (x, y, z, yaw, pitch) of the block's video frames, u
                and degrees.
            buttons (np.ndarray): ``[16, 11]`` 0/1 held buttons in the order of
                :data:`~worldcast.data.controls.CONTROL_BUTTONS`.
            turn (np.ndarray): ``[16, 2]`` degrees turned during each frame (pitch, yaw).
            weapon (np.ndarray): ``[16]`` the held weapon's id.

        Raises:
            ValueError: an array of another shape, a non-finite time, row or turn, times that do
                not increase, or a weapon id outside the vocabulary.
        """
        weapon = np.asarray(weapon, np.int64)
        check_weapon_ids(weapon)  # before the uint8 cast, which would wrap it
        self._send(
            "live",
            f0=int(f0),
            times=np.array(times, np.float64),
            rows=np.array(rows, np.float32),
            buttons=button_bits(buttons),
            turn=np.array(turn, np.float32),
            weapon=weapon.astype(np.uint8),
        )

    # ----------------------------------------------------------------------------------- reading
    def refresh(self) -> None:
        """Apply the parts received since the last call."""
        with self._lock:
            pending, self._inbox = self._inbox, []
        for part, received in pending:
            self._apply(part, received)

    def published_blocks(self, media_id: str) -> list[PublishedBlock]:
        """The blocks of the ``scene`` parts applied so far, by ``t_first``."""
        index = self._blocks.get(str(media_id)) or {}
        return [index[t].block for t in sorted(index)]

    def read_steps(self, media_id: str, *, upto: int) -> list[StepRecord]:
        """The records of the ``step`` parts applied so far, up to ``upto``."""
        records = self._steps.get(str(media_id)) or {}
        return [records[t] for t in sorted(records) if t <= int(upto)]

    def has_step(self, media_id: str, t_target: int) -> bool:
        """Whether the ``step`` part of ``t_target`` has been applied."""
        return int(t_target) in (self._steps.get(str(media_id)) or {})

    def read_position(self, media_id: str, t_target: int) -> tuple[list[int], np.ndarray] | None:
        """The ``position`` part of ``t_target``, after a refresh; None while none has arrived."""
        self.refresh()
        part = self._positions.get((str(media_id), int(t_target)))
        if part is None:
            return None
        return list(part.latent_frames), np.asarray(part.xyz, np.float64).reshape(-1, 3)

    def done_status(self, media_id: str) -> str | None:
        """The status of the ``done`` part applied, or None; None again once the seat is taken
        again."""
        return self._done.get(str(media_id))

    def live_part(self, media_id: str, f0: int) -> MessagePart | None:
        """The ``live`` part of ``media_id``'s block ``f0`` that keys the block's memory entry,
        or None: the block's newest part in session time, and from the entry's ``scene`` part on
        the part it found."""
        key = (str(media_id), int(f0))
        return self._entry_live.get(key, self._live.get(key))

    def recent_live(self, media_id: str, n: int = 2) -> list[MessagePart]:
        """The newest ``n`` ``live`` parts of ``media_id`` applied so far, by the session time of
        their last frame (not by arrival), oldest first."""
        parts = self._live_by_time.get(str(media_id)) or []
        return parts[max(len(parts) - int(n), 0) :]

    def _read_block(self, block: PublishedBlock) -> np.ndarray:
        return self._blocks[block.media_id][block.t_first].latents

    def _read_sha256(self, block: PublishedBlock, latents: np.ndarray) -> str:
        """The hash of the block's latents taken when its part was received."""
        received = self._received_sha256.get((block.media_id, block.t_first))
        return super()._read_sha256(block, latents) if received is None else received
