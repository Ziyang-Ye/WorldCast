"""The GPU engine behind :class:`demo.engine.EngineProtocol`: ``worldcast.engine.realtime.Engine``, one per worker.

Written against the first version of ``worldcast.engine.realtime`` (``Engine.start(RoundSpec, slot)``, ``Engine.step(controls
callable)``, ``Engine.receive(message)``, the ``on_message`` callback and ``transport.encode`` / ``decode``); see
docs/demo.md, "GPU integration", for what to check when that package changes.

The engines of a room exchange the paper's messages (player state with the block's controls, scene-state block,
step record, done) as opaque :class:`demo.engine.PeerMessage` payloads; the coordinator only relays them. Positions
for the HUD come from the same player-state messages: the own one this engine publishes and the peers' ones it
receives, so nothing on screen uses a recorded position unless ``own_state`` is ``gt``.
"""

import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from demo.config import WorldCastEngineConfig, merge_dicts
from demo.engine import BlockActions, Frame, PeerMessage, PlayerState
from demo.library import Library, RoundStart, Seat

#: Message class name -> PeerMessage kind (``scene`` blocks are kept by the coordinator for late joiners).
KINDS = {
    "PlayerStateMessage": "state",
    "SceneBlockMessage": "scene",
    "StepMessage": "step",
    "DoneMessage": "done",
}
FRAMES_PER_LATENT = 4
#: Longest wait, s, for the engine to take a block's controls (it does before the ladder).
CONTROLS_TIMEOUT_S = 120.0


class Track:
    """A player's published positions, one row ``[x, y, z, yaw, pitch, alive]`` per latent frame."""

    def __init__(self, seat: int) -> None:
        self.seat, self.rows = seat, {}

    def add(self, f0: int, states: np.ndarray) -> None:
        for i, row in enumerate(np.asarray(states, dtype=np.float64)):
            self.rows[int(f0) + i] = row

    def at(self, frame: int, t: float) -> PlayerState | None:
        """The state at video frame ``frame`` (its latent frame, or the newest one before it)."""
        latent = (int(frame) + FRAMES_PER_LATENT - 1) // FRAMES_PER_LATENT
        known = [k for k in self.rows if k <= latent]
        if not known:
            return None
        x, y, z, yaw, pitch, alive = (float(v) for v in self.rows[max(known)])
        return PlayerState(
            seat=self.seat, t=t, x=x, y=y, z=z, yaw=yaw, pitch=pitch, alive=alive > 0.5
        )


class WorldCastEngine:
    """:class:`demo.engine.EngineProtocol` over ``worldcast.engine.realtime.Engine`` (models loaded once, in ``__init__``)."""

    def __init__(self, config: WorldCastEngineConfig, library: Library, fps: float = 16.0) -> None:
        import yaml

        from worldcast.config.inference import config_from_dict, with_overrides
        from worldcast.engine.realtime.config import RealtimeConfig
        from worldcast.engine.realtime.controls import FRAMES_PER_BLOCK
        from worldcast.engine.realtime.engine import Engine

        data: dict[str, Any] = {}
        for path in config.configs:
            with open(path, encoding="utf-8") as fh:
                data = merge_dicts(data, yaml.safe_load(fh) or {})
        inference = with_overrides(config_from_dict(data), config.overrides)
        self.fps, self.frames_per_step = float(fps), int(FRAMES_PER_BLOCK)
        self.library = library
        self.engine = Engine(
            inference,
            RealtimeConfig(**config.realtime),
            own_state=config.own_state,
            on_message=self._emitted,
        )
        self._lock = threading.Lock()
        self._outbox: list[PeerMessage] = []
        self._tracks: dict[int, Track] = {}
        self.counts = {"scene_out": 0, "scene_in": 0}

    def start(self, round_start: RoundStart, seat: Seat, peers: Sequence[Seat]) -> PlayerState:
        from worldcast.engine.realtime.engine import RoundSpec

        clients = tuple(sorted({seat.seat, *(p.seat for p in peers)}))
        spec = RoundSpec(
            match_id=round_start.match_id,
            map_name=round_start.map,
            round=round_start.round,
            start_frame=round_start.start_frame,
            clients=clients,
        )
        self.seat = seat.seat
        with self._lock:
            self._outbox.clear()
            self._tracks = {seat.seat: Track(seat.seat)}
            self._tracks[seat.seat].add(0, np.array([[*seat.spawn, 1.0]]))
        self.engine.start(spec, seat.seat)
        x, y, z, yaw, pitch = seat.spawn
        return PlayerState(seat=seat.seat, t=0.0, x=x, y=y, z=z, yaw=yaw, pitch=pitch)

    def step(self, controls: Callable[[], BlockActions]) -> Iterator[Frame]:
        from worldcast.engine.realtime.controls import BlockControls

        taken: dict[str, BlockActions] = {}
        ready = threading.Event()

        def block_controls():
            actions = taken["actions"] = controls()
            ready.set()
            return BlockControls(
                buttons=actions.buttons, camera=actions.camera, weapon=actions.weapon
            )

        for j, frame in enumerate(self.engine.step(block_controls)):
            # a session's frame 0 can be decoded before its first block takes the controls
            if not ready.wait(CONTROLS_TIMEOUT_S):
                raise RuntimeError("the engine did not take the block's controls")
            times = taken["actions"].times
            t = float(times[min(j, len(times) - 1)])
            with self._lock:
                me = self._tracks[self.seat].at(frame.index, t)
                peers = [
                    s
                    for k, track in self._tracks.items()
                    if k != self.seat
                    for s in [track.at(frame.index, t)]
                    if s is not None
                ]
            image = frame.image
            encoded = isinstance(image, (bytes, bytearray))
            yield Frame(
                index=frame.index,
                state=me,
                peers=peers,
                jpeg=bytes(image) if encoded else None,
                rgb=None if encoded else np.asarray(image),
            )

    def _emitted(self, message: Any) -> None:
        """The engine's own messages (any engine thread): keep our track, queue them for the room."""
        from worldcast.engine.realtime import transport

        kind = KINDS[type(message).__name__]
        block = int(getattr(message, "f0", getattr(message, "t_target", 0)))
        with self._lock:
            if kind == "state":
                self._tracks[self.seat].add(message.f0, message.states)
            self._outbox.append(
                PeerMessage(
                    seat=self.seat,
                    block=block,
                    kind=kind,
                    meta={},
                    payload=transport.encode(message),
                )
            )
        self.counts["scene_out"] += kind == "scene"

    def receive_message(self, message: PeerMessage) -> None:
        from worldcast.engine.realtime import transport

        decoded = transport.decode(message.payload)
        self.engine.receive(decoded)
        if message.kind == "state":
            with self._lock:
                self._tracks.setdefault(message.seat, Track(message.seat)).add(
                    decoded.f0, decoded.states
                )
        self.counts["scene_in"] += message.kind == "scene"

    def receive_state(self, state: PlayerState) -> None:
        """Unused: the peers' full player-state messages (with their controls) arrive through receive_message."""

    def remove_peer(self, seat: int) -> None:
        with self._lock:
            self._tracks.pop(int(seat), None)

    def take_messages(self) -> list[PeerMessage]:
        with self._lock:
            out, self._outbox = self._outbox, []
        return out

    def stats(self) -> dict[str, int]:
        return dict(self.counts)
