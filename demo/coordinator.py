"""The coordinator: lobby and rooms, worker assignment, the relay of the shared world state, the room clock.

One CPU process. Browsers load the UI and the lobby API from it; every GPU worker keeps one WebSocket to it. Per
room it relays what the paper's clients exchange and nothing else: each player's own state (once per block) and the
engines' peer messages (scene-state blocks and step records, pushed to every other worker of the room; scene-state
blocks are also kept for late joiners). Video passes through it only when ``coordinator.media_route`` is ``proxy``.
"""

import asyncio
import itertools
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aiohttp import ClientError, ClientSession, WSMsgType, web

from demo.config import SYNC_MODES, DemoConfig
from demo.library import MAP_LABELS, Library, RoundStart
from demo.protocol import new_token, pack, unpack

log = logging.getLogger("demo.coordinator")
STATIC = Path(__file__).resolve().parent / "static"
LIBRARY_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
#: Seconds a join waits for the worker to accept the assignment.
ASSIGN_TIMEOUT_S = 10.0
_room_ids = itertools.count(1)


@dataclass
class WorkerLink:
    id: str
    ws: web.WebSocketResponse
    url: str  # WebSocket base browsers use to reach the worker
    info: dict
    room: str | None = None
    seat: int | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "gpu": self.info.get("gpu"),
            "engine": self.info.get("engine"),
            "frames_per_step": self.info.get("frames_per_step"),
            "busy": self.room is not None,
        }


@dataclass
class Player:
    seat: int
    name: str
    worker: str
    token: str
    status: str = "starting"  # starting | ready | playing


@dataclass
class Room:
    id: str
    name: str
    round: RoundStart
    sync: str
    t0: float  # server clock at the room's start: room time 0
    players: dict[int, Player] = field(default_factory=dict)
    scene: dict[int, deque[tuple[dict, bytes]]] = field(default_factory=dict)
    empty_since: float | None = None


class Coordinator:
    def __init__(self, config: DemoConfig, library: Library) -> None:
        self.cfg, self.library = config, library
        self.workers: dict[str, WorkerLink] = {}
        self.rooms: dict[str, Room] = {}
        self.lobby_sockets: set = set()
        self.pending: dict[tuple[str, int], asyncio.Future] = {}
        self._lobby_dirty = asyncio.Event()
        self.http: ClientSession | None = None

    # ---------------------------------------------------------------------------------------------------- app
    def app(self) -> web.Application:
        app = web.Application(client_max_size=4 << 20)
        app.add_routes(
            [
                web.get("/", self.index),
                web.get("/api/lobby", self.lobby),
                web.post("/api/rooms", self.create_room),
                web.get("/api/rooms/{room}", self.room_info),
                web.post("/api/rooms/{room}/join", self.join),
                web.get("/library/{path:.+}", self.library_file),
                web.get("/ws/lobby", self.lobby_ws),
                web.get("/ws/worker", self.worker_ws),
                web.get("/ws/proxy/{worker}/{kind}", self.proxy_ws),
            ]
        )
        app.router.add_static("/static/", STATIC)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _startup(self, app) -> None:
        self.http = ClientSession()
        self._tasks = [
            asyncio.create_task(self._lobby_pusher()),
            asyncio.create_task(self._reaper()),
        ]

    async def _cleanup(self, app) -> None:
        for task in self._tasks:
            task.cancel()
        await self.http.close()

    @staticmethod
    def now() -> float:
        return time.monotonic()

    # --------------------------------------------------------------------------------------------- snapshots
    def room_public(self, room: Room) -> dict:
        return {
            "id": room.id,
            "name": room.name,
            "round": room.round.id,
            "map": room.round.map,
            "label": room.round.label,
            "sync": room.sync,
            "clock": round(self.now() - room.t0, 2),
            "players": [
                {
                    "seat": p.seat,
                    "name": p.name,
                    "status": p.status,
                    "watch_url": self._url(p, "watch"),
                }
                for p in sorted(room.players.values(), key=lambda p: p.seat)
            ],
        }

    def lobby_public(self) -> dict:
        free = sum(w.room is None for w in self.workers.values())
        return {
            "t": "lobby",
            "rooms": [self.room_public(r) for r in self.rooms.values()],
            "workers": {
                "total": len(self.workers),
                "free": free,
                "list": [w.public() for w in self.workers.values()],
            },
        }

    def _url(self, player: Player, kind: str) -> str | None:
        """Where a browser opens ``kind`` (play | watch) for this player's worker; relative = through us."""
        worker = self.workers.get(player.worker)
        if worker is None:
            return None
        query = f"token={player.token}" if kind == "play" else f"room={worker.room}"
        if self.cfg.coordinator.media_route == "proxy":
            return f"/ws/proxy/{worker.id}/{kind}?{query}"
        return f"{worker.url}/ws/{kind}?{query}"

    def touch_lobby(self) -> None:
        self._lobby_dirty.set()

    async def _lobby_pusher(self) -> None:
        while True:
            await self._lobby_dirty.wait()
            self._lobby_dirty.clear()
            message = self.lobby_public()
            for ws in list(self.lobby_sockets):
                await _send(ws, message)
            await asyncio.sleep(0.25)

    async def _reaper(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            limit = self.cfg.coordinator.empty_room_s
            for room in list(self.rooms.values()):
                if room.empty_since is not None and self.now() - room.empty_since > limit:
                    del self.rooms[room.id]
                    log.info("room %s closed (empty)", room.id)
                    self.touch_lobby()

    # ------------------------------------------------------------------------------------------------- HTTP
    async def index(self, request) -> web.FileResponse:
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    async def lobby(self, request) -> web.Response:
        play = asdict(self.cfg.play)
        return web.json_response(
            {
                **self.lobby_public(),
                "rounds": self.library.public(),
                "play": play,
                "sync_modes": list(SYNC_MODES),
            }
        )

    async def library_file(self, request) -> web.FileResponse:
        try:
            path = self.library.path(request.match_info["path"])
        except FileNotFoundError:
            raise web.HTTPNotFound()
        if path.suffix.lower() not in LIBRARY_SUFFIXES or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "max-age=3600"})

    async def create_room(self, request) -> web.Response:
        body = await request.json()
        try:
            round_start = self.library.get(str(body.get("round")))
        except KeyError as exc:
            raise web.HTTPNotFound(text=str(exc))
        sync = str(body.get("sync") or self.cfg.play.default_sync)
        if sync not in SYNC_MODES:
            raise web.HTTPBadRequest(text=f"sync must be one of {SYNC_MODES}")
        if len(self.rooms) >= self.cfg.coordinator.max_rooms:
            raise web.HTTPServiceUnavailable(text="the server has as many rooms as it allows")
        room_id = f"{round_start.map.replace('de_', '')}-{next(_room_ids)}"
        label = f"{MAP_LABELS.get(round_start.map, round_start.map)} · {round_start.label}"
        name = str(body.get("name") or label)[:60]
        room = Room(
            id=room_id,
            name=name,
            round=round_start,
            sync=sync,
            t0=self.now(),
            empty_since=self.now(),
        )
        self.rooms[room_id] = room
        log.info("room %s created: %s, %s", room_id, round_start.id, sync)
        self.touch_lobby()
        return web.json_response(self.room_public(room))

    def _room(self, request) -> Room:
        room = self.rooms.get(request.match_info["room"])
        if room is None:
            raise web.HTTPNotFound(text="this room has closed")
        return room

    async def room_info(self, request) -> web.Response:
        room = self._room(request)
        return web.json_response({**self.room_public(room), "round_start": room.round.public()})

    async def join(self, request) -> web.Response:
        room = self._room(request)
        body = await request.json()
        seat_index = int(body.get("seat", -1))
        try:
            seat = room.round.seat(seat_index)
        except KeyError as exc:
            raise web.HTTPBadRequest(text=str(exc))
        if seat_index in room.players:
            raise web.HTTPConflict(text="that seat is taken")
        worker = next((w for w in self.workers.values() if w.room is None), None)
        if worker is None:
            raise web.HTTPServiceUnavailable(
                text="every GPU is busy; try again when a player leaves"
            )
        name = str(body.get("name") or f"Player {seat_index + 1}").strip()[:24]
        player = Player(seat=seat_index, name=name, worker=worker.id, token=new_token())
        worker.room, worker.seat = room.id, seat_index
        room.players[seat_index] = player
        room.empty_since = None
        ack = asyncio.get_running_loop().create_future()
        self.pending[(room.id, seat_index)] = ack
        await _send(
            worker.ws,
            {
                "t": "assign",
                "room": room.id,
                "round": room.round.id,
                "seat": seat_index,
                "name": name,
                "token": player.token,
                "sync": room.sync,
                "t0": room.t0,
                "roster": self._roster(room),
            },
        )
        try:
            error = await asyncio.wait_for(ack, ASSIGN_TIMEOUT_S)
        except asyncio.TimeoutError:
            error = "the GPU worker did not answer"
        finally:
            self.pending.pop((room.id, seat_index), None)
        if error:
            self._drop_player(room, seat_index)
            raise web.HTTPServiceUnavailable(text=f"could not start your client: {error}")
        backlog = [
            item for kept in room.scene.values() for item in kept
        ]  # blocks keep arriving while we send
        await self._broadcast_roster(room)
        for header, payload in backlog:
            await _send_bytes(worker.ws, pack(header, payload))
        self.touch_lobby()
        log.info("room %s: %s joined seat %d on worker %s", room.id, name, seat_index, worker.id)
        return web.json_response(
            {
                "room": self.room_public(room),
                "round_start": room.round.public(),
                "seat": seat.public(),
                "name": name,
                "play_url": self._url(player, "play"),
                "play": asdict(self.cfg.play),
                "worker": worker.public(),
            }
        )

    # ------------------------------------------------------------------------------------------- WebSockets
    async def lobby_ws(self, request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.lobby_sockets.add(ws)
        await _send(ws, self.lobby_public())
        try:
            async for _ in ws:
                pass
        finally:
            self.lobby_sockets.discard(ws)
        return ws

    async def worker_ws(self, request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=64 << 20)
        await ws.prepare(request)
        hello = await ws.receive_json(timeout=10)
        if hello.get("t") != "hello":
            await ws.close()
            return ws
        url = hello.get("advertise_url") or f"ws://{request.remote}:{int(hello['port'])}"
        worker = WorkerLink(id=str(hello["id"]), ws=ws, url=url.rstrip("/"), info=hello)
        old = self.workers.get(worker.id)
        if old is not None:
            await self._worker_gone(old)
        self.workers[worker.id] = worker
        log.info(
            "worker %s connected (%s, %s), browsers reach it at %s",
            worker.id,
            hello.get("gpu"),
            hello.get("engine"),
            worker.url,
        )
        self.touch_lobby()
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._from_worker(worker, msg.json())
                elif msg.type == WSMsgType.BINARY:
                    await self._message_from_worker(worker, msg.data)
        finally:
            if self.workers.get(worker.id) is worker:
                await self._worker_gone(worker)
        return ws

    async def _from_worker(self, worker: WorkerLink, msg: dict) -> None:
        kind = msg.get("t")
        if kind == "clock":
            await _send(worker.ws, {"t": "clock", "c": msg["c"], "s": self.now()})
            return
        room = self.rooms.get(str(msg.get("room")))
        if room is None or worker.room != room.id:
            return
        seat = int(msg.get("seat", worker.seat))
        if kind == "assigned":
            future = self.pending.get((room.id, seat))
            if future is not None and not future.done():
                future.set_result(msg.get("error"))
        elif kind == "state":
            for other in self._room_workers(room, exclude=seat):
                await _send(other.ws, msg)
        elif kind == "status" and seat in room.players:
            room.players[seat].status = str(msg["status"])
            self.touch_lobby()
        elif kind == "left":
            log.info("room %s: seat %d left (%s)", room.id, seat, msg.get("reason", ""))
            self._drop_player(room, seat)
            await self._broadcast_roster(room)
            self.touch_lobby()

    async def _message_from_worker(self, worker: WorkerLink, data: bytes) -> None:
        """An engine's peer message: relay it to the room's other workers; keep scene blocks for late joiners."""
        header, payload = unpack(data)
        room = self.rooms.get(str(header.get("room")))
        if room is None or worker.room != room.id:
            return
        seat = int(header["seat"])
        if header.get("kind") == "scene":
            kept = room.scene.setdefault(
                seat, deque(maxlen=self.cfg.coordinator.scene_blocks_per_player)
            )
            kept.append((header, payload))
        for other in self._room_workers(room, exclude=seat):
            await _send_bytes(other.ws, data)

    def _room_workers(self, room: Room, exclude: int | None = None) -> list[WorkerLink]:
        out = []
        for player in room.players.values():
            worker = self.workers.get(player.worker)
            if player.seat != exclude and worker is not None:
                out.append(worker)
        return out

    def _roster(self, room: Room) -> list[dict]:
        return [
            {"seat": p.seat, "name": p.name, "team": room.round.seat(p.seat).team}
            for p in sorted(room.players.values(), key=lambda p: p.seat)
        ]

    async def _broadcast_roster(self, room: Room) -> None:
        message = {"t": "roster", "room": room.id, "roster": self._roster(room)}
        for worker in self._room_workers(room):
            await _send(worker.ws, message)

    def _drop_player(self, room: Room, seat: int) -> None:
        player = room.players.pop(seat, None)
        room.scene.pop(seat, None)
        if player is not None:
            worker = self.workers.get(player.worker)
            if worker is not None and worker.room == room.id and worker.seat == seat:
                worker.room = worker.seat = None
        if not room.players:
            room.empty_since = self.now()

    async def _worker_gone(self, worker: WorkerLink) -> None:
        log.info("worker %s disconnected", worker.id)
        self.workers.pop(worker.id, None)
        room = self.rooms.get(worker.room) if worker.room else None
        if room is not None and worker.seat in room.players:
            self._drop_player(room, worker.seat)
            await self._broadcast_roster(room)
        self.touch_lobby()

    async def proxy_ws(self, request) -> web.WebSocketResponse:
        """Browser <-> worker through the coordinator, for workers browsers cannot reach directly."""
        worker = self.workers.get(request.match_info["worker"])
        kind = request.match_info["kind"]
        if worker is None or kind not in ("play", "watch"):
            raise web.HTTPNotFound()
        browser = web.WebSocketResponse(max_msg_size=8 << 20)
        await browser.prepare(request)

        async def pump(src, dst):
            async for msg in src:
                if msg.type == WSMsgType.TEXT:
                    await dst.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await dst.send_bytes(msg.data)
            await dst.close()

        try:
            async with self.http.ws_connect(
                f"{worker.url}/ws/{kind}?{request.query_string}", max_msg_size=8 << 20
            ) as upstream:
                await asyncio.gather(
                    pump(browser, upstream), pump(upstream, browser), return_exceptions=True
                )
        except (ClientError, OSError) as exc:
            log.warning("proxy to worker %s failed: %s", worker.id, exc)
            await browser.close(message=b"the GPU worker is unreachable")
        return browser


async def _send(ws, message: dict) -> None:
    if not ws.closed:
        try:
            await ws.send_json(message)
        except (ConnectionResetError, RuntimeError):
            pass


async def _send_bytes(ws, data: bytes) -> None:
    if not ws.closed:
        try:
            await ws.send_bytes(data)
        except (ConnectionResetError, RuntimeError):
            pass


def run(config: DemoConfig) -> None:
    library = Library.load(config.library)
    log.info(
        "library: %d round start(s)%s",
        len(library.rounds),
        "" if config.library else " (synthetic arena)",
    )
    web.run_app(
        Coordinator(config, library).app(),
        host=config.coordinator.host,
        port=config.coordinator.port,
        print=None,
        access_log=None,
    )
