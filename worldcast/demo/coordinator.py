"""The coordinator: lobby, rooms, worker assignment, the relay of the shared world state, the room
clock.

One CPU process. Browsers load the UI and the lobby API from it; every GPU worker keeps one
WebSocket to it. Per room it relays what the paper's clients exchange and nothing else: the
clients' messages (player states, memory entries, withdrawals), pushed to every other worker of the
room. For late joiners it keeps the newest ``scene_state.bound`` memory entries of each player and
twice as many of its live states, which a joiner receives first.
Video passes through it only when ``coordinator.frame_route`` is ``proxy``. The wire formats are
:mod:`worldcast.demo.wire`.
"""

import asyncio
import itertools
import json
import logging
import math
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import ClientError, ClientSession, WSMsgType, web

from worldcast.data.game import (
    ASPECT_TAN_RATIO,
    HFOV_DEGREES,
    PITCH_MAX_DEGREES,
    PITCH_MIN_DEGREES,
)
from worldcast.data.latents import FRAME_SIZE
from worldcast.engine.inference.messages import LIVE_STATES_PER_ENTRY

from .clients import LIVE_STATE, MEMORY_ENTRY
from .config import DemoConfig
from .library import PLAYER_COLORS, Library, RoundStart, Seat
from .map_picker import MapPicker
from .wire import HEARTBEAT_S, LINK_MESSAGE_BYTES, WebSocket, pack, send, unpack

__all__ = ["Coordinator", "run"]

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"
LIBRARY_SUFFIXES = {".jpg", ".jpeg", ".png"}
#: Seconds a join waits for the worker to accept the assignment, and a new worker link for its
#: ``hello``.
ASSIGN_TIMEOUT_S, HELLO_TIMEOUT_S = 10.0, 10.0
#: Seconds between two pushes of the lobby to the browsers, between two pings of a lobby socket,
#: and between two looks for empty rooms.
LOBBY_PUSH_S, LOBBY_HEARTBEAT_S, REAP_PERIOD_S = 0.25, 20.0, 5.0
#: Largest message of a proxied browser socket, bytes (a JPEG frame is about 43 kB).
BROWSER_MESSAGE_BYTES = 8 << 20
#: Longest player name, characters.
PLAYER_NAME_LENGTH = 24


@dataclass
class WorkerLink:
    """A connected worker: its WebSocket, how browsers reach it, its client, the player it
    serves."""

    id: str
    ws: web.WebSocketResponse
    url: str  # WebSocket base browsers use to reach the worker
    client: str  # mock | worldcast
    room: str | None = None
    slot: int | None = None

    def public(self) -> dict:
        """What the page reads of a worker."""
        return {"id": self.id, "client": self.client}


@dataclass
class Player:
    """A seated player of a room and its worker."""

    slot: int
    name: str
    worker: str
    token: str


@dataclass
class Room:
    """A room: its round start, its clock origin, its players and their kept live states and
    memory entries."""

    id: str
    name: str
    round_start: RoundStart
    t0: float  # server clock at the room's start: room time 0
    players: dict[int, Player] = field(default_factory=dict)
    live: dict[int, deque[tuple[dict, bytes]]] = field(default_factory=dict)
    entries: dict[int, deque[tuple[dict, bytes]]] = field(default_factory=dict)
    empty_since: float | None = None


async def _json(request: web.Request) -> dict:
    """The request's JSON object; anything else is a bad request."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = None
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="the request body must be a JSON object")
    return body


async def _revalidate_static(request: web.Request, response: web.StreamResponse) -> None:
    """The page's files are revalidated on every load, as the page is: a browser never runs
    scripts older than the workers it talks to (their protocol, docs/demo.md)."""
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"


async def _pump(source: WebSocket, target: WebSocket) -> None:
    """Forward every WebSocket message of ``source`` to ``target``, then close ``target``.

    A proxied socket has one pump per direction, so a slow browser never holds up its inputs,
    pings and acknowledgements, and nothing is dropped: the browser acknowledges the frames it
    received through the proxy, so the worker's in-flight budget (``worker.max_inflight_ms``)
    also bounds the frames the proxy holds.
    """
    async for msg in source:
        if msg.type == WSMsgType.TEXT:
            await target.send_str(msg.data)
        elif msg.type == WSMsgType.BINARY:
            await target.send_bytes(msg.data)
    await target.close()


class Coordinator:
    """The lobby, the rooms and the relay between the workers of a room."""

    def __init__(self, config: DemoConfig, library: Library) -> None:
        self.cfg, self.library = config, library
        self.picker = MapPicker.load(library)
        self.workers: dict[str, WorkerLink] = {}
        self.rooms: dict[str, Room] = {}
        self.lobby_sockets: set = set()
        self.pending: dict[tuple[str, int], asyncio.Future] = {}
        self.http: ClientSession | None = None
        self._room_ids = itertools.count(1)
        self._lobby_dirty = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    def app(self) -> web.Application:
        """The coordinator's web application."""
        app = web.Application()
        app.add_routes(
            [
                web.get("/", self.index),
                web.get("/api/lobby", self.lobby),
                web.get("/api/maps", self.maps),
                web.post("/api/start", self.start),
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
        app.on_response_prepare.append(_revalidate_static)
        app.on_startup.append(self._startup)
        app.on_shutdown.append(self._shutdown)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _startup(self, app: web.Application) -> None:
        self.http = ClientSession()
        self._tasks = [
            asyncio.create_task(self._lobby_pusher()),
            asyncio.create_task(self._reaper()),
        ]

    async def _shutdown(self, app: web.Application) -> None:
        for ws in [*self.lobby_sockets, *(worker.ws for worker in self.workers.values())]:
            await ws.close()

    async def _cleanup(self, app: web.Application) -> None:
        for task in self._tasks:
            task.cancel()
        await self.http.close()

    # ------------------------------------------------------------------------------------ the lobby
    def room_public(self, room: Room) -> dict:
        """What the page reads of a room: its round start's id and its players, by slot."""
        players = sorted(room.players.values(), key=lambda p: p.slot)
        return {
            "id": room.id,
            "name": room.name,
            "round_start": room.round_start.id,
            "players": [
                {"slot": p.slot, "name": p.name, "watch_url": self._url(p, "watch")}
                for p in players
            ],
        }

    def lobby_public(self) -> dict:
        """The ``lobby`` message: the rooms and the workers (total, free, each one's client)."""
        return {
            "t": "lobby",
            "rooms": [self.room_public(room) for room in self.rooms.values()],
            "workers": {
                "total": len(self.workers),
                "free": sum(worker.room is None for worker in self.workers.values()),
                "list": [worker.public() for worker in self.workers.values()],
            },
        }

    def _url(self, player: Player, kind: str) -> str | None:
        """Where a browser opens ``kind`` (play | watch) of the player's worker."""
        worker = self.workers.get(player.worker)
        if worker is None:
            return None
        query = f"token={player.token}" if kind == "play" else f"room={worker.room}"
        if self.cfg.coordinator.frame_route == "proxy":
            return f"/ws/proxy/{worker.id}/{kind}?{query}"
        return f"{worker.url}/ws/{kind}?{query}"

    def touch_lobby(self) -> None:
        """The lobby changed: push it to the browsers."""
        self._lobby_dirty.set()

    async def _lobby_pusher(self) -> None:
        while True:
            await self._lobby_dirty.wait()
            self._lobby_dirty.clear()
            message = self.lobby_public()
            for ws in list(self.lobby_sockets):
                await send(ws, message)
            await asyncio.sleep(LOBBY_PUSH_S)

    def close_empty_rooms(self) -> None:
        """Close the rooms that have been empty for ``coordinator.empty_room_s``."""
        limit = self.cfg.coordinator.empty_room_s
        for room in list(self.rooms.values()):
            if room.empty_since is not None and time.monotonic() - room.empty_since > limit:
                del self.rooms[room.id]
                log.info("room %s closed (empty)", room.id)
                self.touch_lobby()

    async def _reaper(self) -> None:
        while True:
            await asyncio.sleep(REAP_PERIOD_S)
            self.close_empty_rooms()

    # ----------------------------------------------------------------------------------------- HTTP
    async def index(self, request: web.Request) -> web.FileResponse:
        """``GET /``: the page."""
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    async def lobby(self, request: web.Request) -> web.Response:
        """``GET /api/lobby``: the lobby message, the library's round starts and the constants
        the page takes from the server; ``camera_tans`` and ``pitch_range_degrees`` are the
        camera's (:mod:`worldcast.data.camera`), for the play view's instant turn."""
        across = math.tan(math.radians(HFOV_DEGREES) / 2.0)
        page = {
            "player_colors": PLAYER_COLORS,
            "player_name_length": PLAYER_NAME_LENGTH,
            "frame_pixels": [FRAME_SIZE[1], FRAME_SIZE[0]],
            "camera_tans": [across, across * ASPECT_TAN_RATIO],
            "pitch_range_degrees": [PITCH_MIN_DEGREES, PITCH_MAX_DEGREES],
        }
        lobby = {**self.lobby_public(), "round_starts": self.library.public(), "page": page}
        return web.json_response(lobby)

    async def library_file(self, request: web.Request) -> web.FileResponse:
        """``GET /library/<file>``: an image of the library; anything else is not found."""
        try:
            path = self.library.path(request.match_info["path"])
        except FileNotFoundError:
            raise web.HTTPNotFound()
        if path.suffix.lower() not in LIBRARY_SUFFIXES or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "max-age=3600"})

    async def create_room(self, request: web.Request) -> web.Response:
        """``POST /api/rooms`` ``{round_start}``: open a room at a round start."""
        body = await _json(request)
        try:
            round_start = self.library.get(str(body.get("round_start")))
        except KeyError as error:
            raise web.HTTPNotFound(text=error.args[0])
        return web.json_response(self.room_public(self._open_room(round_start)))

    def _open_room(self, round_start: RoundStart) -> Room:
        """A new room at a round start; 503 when the server has as many as it allows."""
        if len(self.rooms) >= self.cfg.coordinator.max_rooms:
            raise web.HTTPServiceUnavailable(text="the server has as many rooms as it allows")
        room_id = f"{round_start.map.removeprefix('de_')}-{next(self._room_ids)}"
        name = f"{round_start.map_label} · {round_start.label}"
        now = time.monotonic()
        room = Room(room_id, name, round_start, t0=now, empty_since=now)
        self.rooms[room_id] = room
        log.info("room %s created: %s", room_id, round_start.id)
        self.touch_lobby()
        return room

    async def maps(self, request: web.Request) -> web.Response:
        """``GET /api/maps``: the seats of the library on the overview of each map
        (:meth:`MapPicker.public`)."""
        return web.json_response({"maps": self.picker.public()})

    async def start(self, request: web.Request) -> web.Response:
        """``POST /api/start`` ``{map, u, v, floor, name}``: open a room at the seat nearest to a
        point of a map's overview and seat the player there.

        ``u`` and ``v`` are the point as fractions of the overview's width and height, ``floor``
        the index of its floor (``null``: all). Answers as :meth:`join` does, with the new ``room``.
        """
        body = await _json(request)
        try:
            floor = body.get("floor")
            round_start_id, slot = self.picker.nearest(
                str(body["map"]),
                float(body["u"]),
                float(body["v"]),
                None if floor is None else int(floor),
            )
        except (KeyError, TypeError, ValueError):
            raise web.HTTPBadRequest(text="a start needs a map and a point on it: u and v")
        except LookupError as error:
            raise web.HTTPNotFound(text=str(error))
        if all(worker.room is not None for worker in self.workers.values()):
            raise web.HTTPServiceUnavailable(
                text="every GPU is busy; try again when a player leaves"
            )
        round_start = self.library.get(round_start_id)
        room = self._open_room(round_start)
        try:
            joined = await self._take_seat(room, round_start.seat(slot), body.get("name"))
        except web.HTTPException:
            del self.rooms[room.id]  # nobody is in it
            self.touch_lobby()
            raise
        return web.json_response({**joined, "room": self.room_public(room)})

    def _room(self, request: web.Request) -> Room:
        room = self.rooms.get(request.match_info["room"])
        if room is None:
            raise web.HTTPNotFound(text="this room has closed")
        return room

    async def room_info(self, request: web.Request) -> web.Response:
        """``GET /api/rooms/<id>``: the room, its ``round_start`` as the whole record."""
        room = self._room(request)
        return web.json_response(
            {**self.room_public(room), "round_start": room.round_start.public()}
        )

    def _free_worker(self, room: Room) -> WorkerLink:
        """A free worker whose client is the room's: clients of two kinds share no messages."""
        kinds = {self.workers[p.worker].client for p in room.players.values()}
        for worker in self.workers.values():
            if worker.room is None and (not kinds or worker.client in kinds):
                return worker
        raise web.HTTPServiceUnavailable(text="every GPU is busy; try again when a player leaves")

    async def join(self, request: web.Request) -> web.Response:
        """``POST /api/rooms/<id>/join`` ``{slot, name}``: seat the player on a free worker.

        Answers with the player's ``play_url`` once the worker has accepted the seat; the worker
        then receives the live states, then the memory entries, kept of the room's other players.
        """
        room = self._room(request)
        body = await _json(request)
        try:
            seat = room.round_start.seat(int(body.get("slot", -1)))
        except (KeyError, TypeError, ValueError):
            raise web.HTTPBadRequest(text=f"round start {room.round_start.id} has no such seat")
        return web.json_response(await self._take_seat(room, seat, body.get("name")))

    async def _take_seat(self, room: Room, seat: Seat, name: object) -> dict:
        """Seat a player on a free worker; what :meth:`join` answers."""
        if seat.slot in room.players:
            raise web.HTTPConflict(text="that seat is taken")
        worker = self._free_worker(room)
        name = str(name or f"Player {seat.slot + 1}").strip()[:PLAYER_NAME_LENGTH]
        player = Player(seat.slot, name, worker.id, token=secrets.token_urlsafe(18))
        worker.room, worker.slot = room.id, seat.slot
        room.players[seat.slot] = player
        room.empty_since = None
        error = await self._assign(room, player, worker)
        if error:
            await send(worker.ws, {"t": "release", "room": room.id, "slot": seat.slot})
            self._drop_player(room, seat.slot)
            raise web.HTTPServiceUnavailable(text=f"could not start your client: {error}")
        held = [*room.live.values(), *room.entries.values()]
        kept = [message for messages in held for message in messages]
        await self._broadcast_roster(room)
        for header, payload in kept:
            await send(worker.ws, pack(header, payload))
        self.touch_lobby()
        log.info(
            "room %s: %s joined as player %d on worker %s", room.id, name, seat.slot, worker.id
        )
        return {
            "round_start": room.round_start.public(),
            "seat": seat.public(),
            "name": name,
            "play_url": self._url(player, "play"),
            "worker": worker.public(),
        }

    async def _assign(self, room: Room, player: Player, worker: WorkerLink) -> str | None:
        """Assign the player's seat to the worker; its error, or ``None`` once it accepted."""
        accepted = asyncio.get_running_loop().create_future()
        self.pending[(room.id, player.slot)] = accepted
        assign = {
            "t": "assign",
            "room": room.id,
            "round_start": room.round_start.id,
            "slot": player.slot,
            "token": player.token,
            "t0": room.t0,
            "roster": self._roster(room),
        }
        await send(worker.ws, assign)
        try:
            return await asyncio.wait_for(accepted, ASSIGN_TIMEOUT_S)
        except asyncio.TimeoutError:
            return "the GPU worker did not answer"
        finally:
            del self.pending[(room.id, player.slot)]

    # ----------------------------------------------------------------------------------- WebSockets
    async def lobby_ws(self, request: web.Request) -> web.WebSocketResponse:
        """``/ws/lobby``: the lobby message now and whenever it changes."""
        ws = web.WebSocketResponse(heartbeat=LOBBY_HEARTBEAT_S)
        await ws.prepare(request)
        self.lobby_sockets.add(ws)
        await send(ws, self.lobby_public())
        try:
            async for _ in ws:
                pass
        finally:
            self.lobby_sockets.discard(ws)
        return ws

    async def worker_ws(self, request: web.Request) -> web.WebSocketResponse:
        """``/ws/worker``: a worker's link, from its ``hello`` until it goes away."""
        ws = web.WebSocketResponse(heartbeat=HEARTBEAT_S, max_msg_size=LINK_MESSAGE_BYTES)
        await ws.prepare(request)
        hello = await ws.receive_json(timeout=HELLO_TIMEOUT_S)
        if hello.get("t") != "hello":
            await ws.close()
            return ws
        url = hello.get("advertise_url") or f"ws://{request.remote}:{int(hello['port'])}"
        worker = WorkerLink(str(hello["id"]), ws, url.rstrip("/"), str(hello["client"]))
        old = self.workers.get(worker.id)
        if old is not None:
            await self._worker_gone(old)
        self.workers[worker.id] = worker
        proxied = self.cfg.coordinator.frame_route == "proxy"
        reached = "through the coordinator" if proxied else f"at {worker.url}"
        log.info(
            "worker %s connected (%s), browsers reach it %s", worker.id, worker.client, reached
        )
        self.touch_lobby()
        try:
            async for msg in ws:
                try:
                    if msg.type == WSMsgType.TEXT:
                        await self._from_worker(worker, msg.json())
                    elif msg.type == WSMsgType.BINARY:
                        await self._message_from_worker(worker, msg.data)
                except Exception:  # a bad message must not end the worker's link
                    log.exception("a message of worker %s failed", worker.id)
        finally:
            if self.workers.get(worker.id) is worker:
                await self._worker_gone(worker)
        return ws

    async def _from_worker(self, worker: WorkerLink, msg: dict) -> None:
        kind = msg.get("t")
        if kind == "clock":
            await send(worker.ws, {"t": "clock", "c": msg["c"], "s": time.monotonic()})
            return
        room = self.rooms.get(str(msg.get("room")))
        if room is None or worker.room != room.id:
            return
        if kind == "assigned":
            accepted = self.pending.get((room.id, worker.slot))
            if accepted is not None and not accepted.done():
                accepted.set_result(msg.get("error"))
        elif kind == "left":
            log.info("room %s: player %d left (%s)", room.id, worker.slot, msg.get("reason", ""))
            self._drop_player(room, worker.slot)
            await self._broadcast_roster(room)
            self.touch_lobby()

    async def _message_from_worker(self, worker: WorkerLink, data: bytes) -> None:
        """Relay a client's message to the room; keep live states and memory entries for late
        joiners."""
        header, payload = unpack(data)
        room = self.rooms.get(str(header.get("room")))
        if room is None or worker.room != room.id:
            return
        bound, kind = self.cfg.scene_state.bound, header.get("kind")
        if kind == LIVE_STATE:
            kept = room.live.setdefault(worker.slot, deque(maxlen=LIVE_STATES_PER_ENTRY * bound))
            kept.append((header, payload))
        elif kind == MEMORY_ENTRY:
            kept = room.entries.setdefault(worker.slot, deque(maxlen=bound))
            kept.append((header, payload))
        for other in self._room_workers(room, exclude=worker.slot):
            await send(other.ws, data)

    def _room_workers(self, room: Room, exclude: int | None = None) -> list[WorkerLink]:
        workers = [self.workers.get(p.worker) for p in room.players.values() if p.slot != exclude]
        return [worker for worker in workers if worker is not None]

    def _roster(self, room: Room) -> list[dict]:
        return [
            {"slot": p.slot, "name": p.name, "team": room.round_start.seat(p.slot).team}
            for p in sorted(room.players.values(), key=lambda p: p.slot)
        ]

    async def _broadcast_roster(self, room: Room) -> None:
        message = {"t": "roster", "room": room.id, "roster": self._roster(room)}
        for worker in self._room_workers(room):
            await send(worker.ws, message)

    def _drop_player(self, room: Room, slot: int) -> None:
        player = room.players.pop(slot, None)
        room.live.pop(slot, None)
        room.entries.pop(slot, None)
        if player is not None:
            worker = self.workers.get(player.worker)
            if worker is not None and worker.room == room.id and worker.slot == slot:
                worker.room = worker.slot = None
        if not room.players:
            room.empty_since = time.monotonic()

    async def _worker_gone(self, worker: WorkerLink) -> None:
        log.info("worker %s disconnected", worker.id)
        self.workers.pop(worker.id, None)
        room = self.rooms.get(worker.room) if worker.room else None
        if room is not None and worker.slot in room.players:
            self._drop_player(room, worker.slot)
            await self._broadcast_roster(room)
        self.touch_lobby()

    async def proxy_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Browser <-> worker through the coordinator, for workers the browsers cannot reach."""
        worker = self.workers.get(request.match_info["worker"])
        kind = request.match_info["kind"]
        if worker is None or kind not in ("play", "watch"):
            raise web.HTTPNotFound()
        browser = web.WebSocketResponse(max_msg_size=BROWSER_MESSAGE_BYTES)
        await browser.prepare(request)
        url = f"{worker.url}/ws/{kind}?{request.query_string}"
        try:
            async with self.http.ws_connect(url, max_msg_size=BROWSER_MESSAGE_BYTES) as upstream:
                await asyncio.gather(
                    _pump(browser, upstream), _pump(upstream, browser), return_exceptions=True
                )
        except (ClientError, OSError) as error:
            log.warning("proxy to worker %s failed: %s", worker.id, error)
            await browser.close(message=b"the GPU worker is unreachable")
        return browser


def run(config: DemoConfig) -> None:
    """Serve the coordinator until interrupted."""
    library = Library.load(config.library)
    web.run_app(
        Coordinator(config, library).app(),
        host=config.coordinator.host,
        port=config.coordinator.port,
        print=None,
        access_log=None,
    )
