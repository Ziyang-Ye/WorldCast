"""A GPU worker: one engine (models loaded once), one player at a time.

It takes the player's input ticks from the browser, steps the engine block by block, streams each frame to the
browser (and to spectators) as it is decoded, and exchanges the shared world state with the room's other workers
through the coordinator: its own state once per block out, the peers' states in, the engines' peer messages (scene-state
blocks, step records) both ways.

Pacing: a block is started just in time for its first frame to reach the browser when the previous block runs
out, so inputs are cut as late as possible (lowest latency) and a fast engine does not run ahead of the display.
Sync: ``async`` (default) steps on the peers' latest states; ``lockstep`` first waits, up to
``worker.lockstep_timeout_s``, for every peer's previous block, as in the paper's evaluation.
"""

import asyncio
import itertools
import logging
import socket
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from aiohttp import ClientSession, WSMsgType, web

from demo.actions import InputBuffer, InputTick
from demo.config import DemoConfig
from demo.engine import BlockActions, EngineProtocol, Frame, PeerMessage, PlayerState
from demo.library import Library, RoundStart, Seat
from demo.protocol import encode_jpeg, pack, unpack
from worldcast.data.actions import encode_weapon

log = logging.getLogger("demo.worker")
#: Extra head start of a block over its measured critical delay, s (absorbs network jitter).
PACING_MARGIN_S = 0.04
#: A block starts as early as the slowest of the last this many blocks needed (its critical delay).
LEAD_WINDOW = 16
#: Input ticks remembered for the latency breakdown.
ARRIVALS_KEPT = 512
#: Seconds a newly seated player has to open the play socket (the engine's start counts against it).
FIRST_CONNECT_S = 30.0


@dataclass
class Session:
    """One player on this worker: the seat, its browser, its inputs and the peers it hears from."""

    room: str
    round: RoundStart
    seat: Seat
    name: str
    token: str
    sync: str
    t0: float  # server clock at room time 0
    roster: dict[int, dict] = field(default_factory=dict)
    buffer: InputBuffer | None = None
    arrivals: dict[int, float] = field(default_factory=dict)
    player: web.WebSocketResponse | None = None
    watchers: set[web.WebSocketResponse] = field(default_factory=set)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    connected: asyncio.Event = field(default_factory=asyncio.Event)
    peer_block: dict[int, int] = field(default_factory=dict)
    peer_news: asyncio.Event = field(default_factory=asyncio.Event)
    block: int = 0  # the next block, on the room clock
    frame: int = 0
    loop_task: asyncio.Task | None = None
    grace_task: asyncio.Task | None = None
    critical: deque = field(default_factory=lambda: deque(maxlen=LEAD_WINDOW))

    @property
    def lead(self) -> float:
        """Seconds from a block's start until its frames are all due-ready (the max of the recent blocks)."""
        return max(self.critical, default=0.0)

    @property
    def peers(self) -> set[int]:
        return {s for s in self.roster if s != self.seat.seat}


class Worker:
    def __init__(self, config: DemoConfig, engine: EngineProtocol, library: Library) -> None:
        self.cfg, self.engine, self.library = config, engine, library
        wc = config.worker
        self.id = wc.worker_id or f"{socket.gethostname().split('.')[0]}:{wc.port}"
        self.session: Session | None = None
        self.coordinator: web.WebSocketResponse | None = None
        self.clock_offset = 0.0  # server clock - local monotonic clock
        self._best_rtt = float("inf")
        self.engine_thread = ThreadPoolExecutor(1, thread_name_prefix="engine")
        self.encoders = ThreadPoolExecutor(2, thread_name_prefix="jpeg")

    # ----------------------------------------------------------------------------------------------- clocks
    def room_time(self, session: Session) -> float:
        return time.monotonic() + self.clock_offset - session.t0

    # -------------------------------------------------------------------------------------------------- app
    def app(self) -> web.Application:
        app = web.Application()
        app.add_routes(
            [
                web.get("/ws/play", self.play_ws),
                web.get("/ws/watch", self.watch_ws),
                web.get("/health", self.health),
            ]
        )
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _startup(self, app) -> None:
        self.http = ClientSession()
        self._link = asyncio.create_task(self._coordinator_link())

    async def _cleanup(self, app) -> None:
        self._link.cancel()
        await self.http.close()

    async def health(self, request) -> web.Response:
        s = self.session
        return web.json_response(
            {
                "id": self.id,
                "engine": self.cfg.engine.kind,
                "busy": s is not None,
                "room": s.room if s else None,
                "seat": s.seat.seat if s else None,
            }
        )

    # ------------------------------------------------------------------------------------- coordinator link
    async def _coordinator_link(self) -> None:
        url = self.cfg.worker.coordinator_url.rstrip("/") + "/ws/worker"
        delay = 0.5
        while True:
            try:
                async with self.http.ws_connect(url, heartbeat=10, max_msg_size=64 << 20) as ws:
                    self.coordinator, delay = ws, 0.5
                    self._best_rtt = float("inf")
                    await ws.send_json(
                        {
                            "t": "hello",
                            "id": self.id,
                            "port": self.cfg.worker.port,
                            "advertise_url": self.cfg.worker.advertise_url,
                            "gpu": self.cfg.worker.gpu_label or self.cfg.engine.kind,
                            "engine": self.cfg.engine.kind,
                            "fps": self.engine.fps,
                            "frames_per_step": self.engine.frames_per_step,
                        }
                    )
                    log.info("worker %s connected to %s", self.id, url)
                    clock = asyncio.create_task(self._clock_pings(ws))
                    try:
                        async for msg in ws:
                            if msg.type == WSMsgType.TEXT:
                                await self._from_coordinator(msg.json())
                            elif msg.type == WSMsgType.BINARY:
                                self._message_in(msg.data)
                    finally:
                        clock.cancel()
            except (OSError, asyncio.TimeoutError, ConnectionError) as exc:
                log.warning("coordinator %s unreachable (%s); retrying in %.1f s", url, exc, delay)
            self.coordinator = None
            if self.session is not None:
                await self._end_session("the coordinator went away")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)

    async def _clock_pings(self, ws) -> None:
        for i in itertools.count():
            await ws.send_json({"t": "clock", "c": time.monotonic()})
            await asyncio.sleep(0.2 if i < 5 else 5.0)

    async def _to_coordinator(self, message: dict) -> None:
        ws = self.coordinator
        if ws is not None and not ws.closed:
            await ws.send_json(message)

    async def _from_coordinator(self, msg: dict) -> None:
        kind = msg.get("t")
        if kind == "clock":
            now = time.monotonic()
            rtt = now - float(msg["c"])
            if rtt <= self._best_rtt * 1.5:
                self._best_rtt = min(self._best_rtt, rtt)
                self.clock_offset = float(msg["s"]) - (float(msg["c"]) + now) / 2
        elif kind == "assign":
            await self._assign(msg)
        s = self.session
        if s is None or msg.get("room") != s.room:
            return
        if kind == "roster":
            gone = s.peers - {int(p["seat"]) for p in msg["roster"]}
            s.roster = {int(p["seat"]): p for p in msg["roster"]}
            for seat in gone:
                self.engine.remove_peer(seat)
                s.peer_block.pop(seat, None)
            s.peer_news.set()
            await self._to_browsers(s, {"t": "roster", "roster": msg["roster"]})
        elif kind == "state":
            seat = int(msg["seat"])
            if seat in s.peers:
                self.engine.receive_state(PlayerState.from_wire(msg["frames"][-1]))
                s.peer_block[seat] = int(msg["block"])
                s.peer_news.set()

    def _message_in(self, data: bytes) -> None:
        header, payload = unpack(data)
        s = self.session
        if s is not None and header.get("room") == s.room and int(header["seat"]) != s.seat.seat:
            self.engine.receive_message(
                PeerMessage(
                    seat=int(header["seat"]),
                    block=int(header["block"]),
                    kind=str(header["kind"]),
                    meta=header.get("meta", {}),
                    payload=payload,
                )
            )

    # ---------------------------------------------------------------------------------------------- sessions
    async def _assign(self, msg: dict) -> None:
        reply = {"t": "assigned", "room": msg["room"], "seat": msg["seat"]}
        try:
            if self.session is not None:
                raise RuntimeError("this worker is already playing")
            round_start = self.library.get(msg["round"])
            seat = round_start.seat(int(msg["seat"]))
        except (KeyError, RuntimeError) as exc:
            await self._to_coordinator({**reply, "error": str(exc)})
            return
        s = Session(
            room=msg["room"],
            round=round_start,
            seat=seat,
            name=msg["name"],
            token=msg["token"],
            sync=msg["sync"],
            t0=float(msg["t0"]),
            roster={int(p["seat"]): p for p in msg["roster"]},
        )
        s.buffer = InputBuffer(
            fps=self.engine.fps,
            mapping=self.cfg.play.action_mapping,
            weapon=encode_weapon(seat.loadout[0]) if seat.loadout else 0,
        )
        # blocks are numbered on the room clock, so a late joiner's block b is everyone's block b (lock-step)
        s.block = max(0, int(self.room_time(s) * self.engine.fps / self.engine.frames_per_step))
        self.session = s
        await self._to_coordinator({**reply, "error": None})
        s.loop_task = asyncio.create_task(self._run(s))
        s.grace_task = asyncio.create_task(
            self._grace(s, FIRST_CONNECT_S, "the player did not connect")
        )

    async def _run(self, s: Session) -> None:
        """Start the engine on the session's seat, then play; a failure ends the session, not the worker."""
        peers = [s.round.seat(p) for p in s.peers]
        try:
            await asyncio.get_running_loop().run_in_executor(
                self.engine_thread, self.engine.start, s.round, s.seat, peers
            )
            s.ready.set()
            await self._status(s, "ready")
            await self._play(s)
        except Exception as exc:  # noqa: BLE001 - reported to the room, the worker stays up
            log.exception("session failed")
            if self.session is s:
                await self._end_session(f"the engine failed: {exc}")

    async def _status(self, s: Session, status: str, message: str = "") -> None:
        await self._to_coordinator(
            {"t": "status", "room": s.room, "seat": s.seat.seat, "status": status}
        )
        await self._to_browsers(s, {"t": "status", "status": status, "message": message})

    async def _grace(self, s: Session, seconds: float, reason: str) -> None:
        """End the session if the player's browser stays away for ``seconds``."""
        await asyncio.sleep(seconds)
        if self.session is s and s.player is None:
            await self._end_session(reason)

    async def _end_session(self, reason: str) -> None:
        s, self.session = self.session, None
        if s is None:
            return
        log.info("session %s seat %d ended: %s", s.room, s.seat.seat, reason)
        for task in (s.loop_task, s.grace_task):
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        await self._to_browsers(s, {"t": "status", "status": "ended", "message": reason})
        for ws in [s.player, *s.watchers]:
            if ws is not None:
                await ws.close()
        await self._to_coordinator(
            {"t": "left", "room": s.room, "seat": s.seat.seat, "reason": reason}
        )

    # --------------------------------------------------------------------------------------- browser sockets
    async def play_ws(self, request) -> web.WebSocketResponse:
        s = self.session
        if s is None or request.query.get("token") != s.token:
            raise web.HTTPForbidden(text="no seat on this worker for that token")
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        if s.player is not None:
            await s.player.close(message=b"replaced by a new connection")
        s.player = ws
        s.connected.set()
        await ws.send_json(self._welcome(s))
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                data = msg.json()
                kind = data.get("t")
                if kind == "in":
                    self._input(s, data)
                elif kind == "ping":
                    await ws.send_json({"t": "pong", "c": data["c"]})
                elif kind == "bye":
                    await self._end_session("the player left")
                    break
        finally:
            if s.player is ws:
                s.player = None
                s.connected.clear()
                if self.session is s:
                    s.grace_task = asyncio.create_task(
                        self._grace(
                            s, self.cfg.worker.reconnect_grace_s, "the player's browser went away"
                        )
                    )
        return ws

    async def watch_ws(self, request) -> web.WebSocketResponse:
        s = self.session
        if s is None or request.query.get("room") != s.room:
            raise web.HTTPNotFound(text="this worker is not playing in that room")
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        s.watchers.add(ws)
        await ws.send_json(self._welcome(s))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT and msg.json().get("t") == "ping":
                    await ws.send_json({"t": "pong", "c": msg.json()["c"]})
        finally:
            s.watchers.discard(ws)
        return ws

    def _welcome(self, s: Session) -> dict:
        return {
            "t": "welcome",
            "worker": self.id,
            "room": s.room,
            "seat": s.seat.public(),
            "name": s.name,
            "round": s.round.public(),
            "roster": list(s.roster.values()),
            "sync": s.sync,
            "fps": self.engine.fps,
            "frames_per_step": self.engine.frames_per_step,
            "status": "ready" if s.ready.is_set() else "starting",
            "play": asdict(self.cfg.play),
            "weapon_ids": {name: encode_weapon(name) for name in s.seat.loadout},
        }

    def _input(self, s: Session, data: dict) -> None:
        seq = int(data["q"])
        s.arrivals[seq] = time.monotonic()
        if len(s.arrivals) > ARRIVALS_KEPT:
            for old in sorted(s.arrivals)[: len(s.arrivals) - ARRIVALS_KEPT]:
                del s.arrivals[old]
        s.buffer.add(
            InputTick(
                seq=seq,
                arrival=self.room_time(s),
                buttons=int(data.get("b", 0)),
                dpitch=float(data.get("dp", 0.0)),
                dyaw=float(data.get("dy", 0.0)),
                weapon=int(data.get("w", s.buffer.weapon)),
            )
        )

    async def _to_browsers(self, s: Session, message: dict) -> None:
        for ws in [s.player, *s.watchers]:
            if ws is not None and not ws.closed:
                try:
                    await ws.send_json(message)
                except (ConnectionResetError, RuntimeError):
                    pass

    # --------------------------------------------------------------------------------------------- the loop
    async def _play(self, s: Session) -> None:
        """Step the engine block after block while the player is connected."""
        fps, frames = self.engine.fps, self.engine.frames_per_step
        origin: float | None = None  # local time at which session frame 0 was due on the browser
        await s.connected.wait()
        await self._status(s, "playing")
        while self.session is s:
            if not s.connected.is_set():
                await s.connected.wait()
                origin = None  # the browser rebuffers after a reconnect
            if origin is not None:
                wait = origin + s.frame / fps - s.lead - PACING_MARGIN_S - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            if s.sync == "lockstep":
                await self._wait_for_peers(s, s.block - 1)
            cut = {}

            def controls(
                block=s.block, first=s.frame
            ):  # called by the engine right before its ladder
                cut["at"] = time.monotonic()
                cut["actions"] = s.buffer.cut(
                    now=self.room_time(s), block=block, first_frame=first, frames=frames
                )
                return cut["actions"]

            called = time.monotonic()
            critical, late, states = 0.0, 0.0, []
            async for frame in self._step(controls):
                j, ready = len(states), time.monotonic()
                await self._send_frame(s, cut["actions"], j, frame, cut["at"], ready)
                sent = time.monotonic()
                if origin is None:
                    origin = sent - (s.frame + j) / fps
                critical = max(critical, sent - called - j / fps)
                late = max(late, sent - origin - (s.frame + j) / fps)
                states.append(frame.state.wire())
            if not states:
                await self._end_session("the round is over")
                return
            s.critical.append(critical)
            origin += late  # a frame missed its slot: the browser stalled and re-anchored
            s.block += 1
            s.frame += len(states)
            await self._to_coordinator(
                {
                    "t": "state",
                    "room": s.room,
                    "seat": s.seat.seat,
                    "block": s.block - 1,
                    "frames": states,
                }
            )
            for message in self.engine.take_messages():
                await self._message_out(s, message)

    async def _wait_for_peers(self, s: Session, block: int) -> None:
        deadline = time.monotonic() + self.cfg.worker.lockstep_timeout_s
        while any(s.peer_block.get(p, -1) < block for p in s.peers):
            s.peer_news.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                late = [p for p in s.peers if s.peer_block.get(p, -1) < block]
                log.warning("lock-step: stepping block %d without seats %s", block + 1, late)
                return
            try:
                await asyncio.wait_for(s.peer_news.wait(), remaining)
            except asyncio.TimeoutError:
                pass

    async def _step(self, controls: Callable[[], BlockActions]) -> AsyncIterator[Frame]:
        """The engine's frames of one block, handed over from the engine thread as they are decoded."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = object()

        def work():
            try:
                for frame in self.engine.step(controls):
                    loop.call_soon_threadsafe(queue.put_nowait, frame)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the event loop
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            loop.call_soon_threadsafe(queue.put_nowait, done)

        future = loop.run_in_executor(self.engine_thread, work)
        while (item := await queue.get()) is not done:
            if isinstance(item, BaseException):
                raise item
            yield item
        await future

    async def _send_frame(
        self, s: Session, actions: BlockActions, j: int, frame: Frame, taken: float, ready: float
    ) -> None:
        """Encode (unless the engine did) and send one frame with its timing: ``tm`` = [input tick to controls taken,
        controls taken to frame ready, encoding] in ms."""
        jpeg = frame.jpeg
        if jpeg is None:
            loop = asyncio.get_running_loop()
            jpeg = await loop.run_in_executor(
                self.encoders, encode_jpeg, frame.rgb, self.cfg.worker.jpeg_quality
            )
        encoded = time.monotonic()
        j = min(
            j, actions.num_frames - 1
        )  # an engine's first block may also carry the round's first frame
        seq = int(actions.input_seq[j])
        arrival = s.arrivals.get(seq)
        header = {
            "k": frame.index,
            "b": actions.block,
            "q": seq,
            "rt": round(float(actions.times[j]), 4),
            "tm": [
                round(1000 * (taken - arrival), 1) if arrival else None,
                round(1000 * (ready - taken), 1),
                round(1000 * (encoded - ready), 1),
            ],
            "s": frame.state.wire(),
            "p": [p.wire() for p in frame.peers],
        }
        if frame.index % 16 == 0:
            header["x"] = {**self.engine.stats(), "lead_ms": round(1000 * s.lead, 1)}
        data = pack(header, jpeg)
        for ws in [s.player, *s.watchers]:
            if ws is not None and not ws.closed:
                try:
                    await ws.send_bytes(data)
                except (ConnectionResetError, RuntimeError):
                    pass

    async def _message_out(self, s: Session, message: PeerMessage) -> None:
        ws = self.coordinator
        if ws is not None and not ws.closed:
            header = {
                "t": "msg",
                "room": s.room,
                "seat": s.seat.seat,
                "block": message.block,
                "kind": message.kind,
                "meta": message.meta,
            }
            await ws.send_bytes(pack(header, message.payload))


def make_engine(config: DemoConfig, library: Library) -> EngineProtocol:
    if config.engine.kind == "mock":
        from demo.mock_engine import MockEngine

        return MockEngine(config.engine.mock, library, fps=config.play.fps)
    from demo.worldcast_engine import WorldCastEngine

    return WorldCastEngine(config.engine.worldcast, library, fps=config.play.fps)


def run(config: DemoConfig) -> None:
    library = Library.load(config.library)
    engine = make_engine(config, library)
    web.run_app(
        Worker(config, engine, library).app(),
        host=config.worker.host,
        port=config.worker.port,
        print=None,
        access_log=None,
    )
