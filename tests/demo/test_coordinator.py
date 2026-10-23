"""The demo end to end on the CPU: a coordinator, mock workers and scripted browsers in one process,
the relay of the clients' messages, what the coordinator keeps for a late joiner, and its HTTP
API.

Run: ``python -m pytest tests/demo -q`` (needs ``pip install -e ".[test]"``).
"""

import asyncio
import math
import threading

import pytest

from tests.demo.support import FORWARD, RecordingClient, create_room, demo, take_seat, until
from worldcast.demo.clients import LIVE_STATE, MEMORY_ENTRY, ClientMessage
from worldcast.demo.library import Library, synthetic_library
from worldcast.engine.inference.field_rows import FieldRows

#: Longest a gated client holds a block's last frame for its player state to reach the other
#: client, s.
GATE_S = 5.0


def moved(states) -> float:
    return math.hypot(states[-1].x - states[0].x, states[-1].y - states[0].y)


def live(states) -> list:
    """The states of those a frame shows that another client published."""
    return [state for state in states if state.src == FieldRows.LIVE]


async def three_players(route: str):
    async with demo(3, **{"coordinator.frame_route": route}) as (local, http):
        coordinator = local.coordinator
        room = await create_room(local, http)
        a = await take_seat(local, http, room, 0, hold=FORWARD)
        b = await take_seat(local, http, room, 2)
        taken = await http.post(f"{local.url}/api/rooms/{room}/join", json={"slot": 0})
        # A holds forward: its own state moves, and B's frames show it moving
        await until(lambda: a.frames and moved(a.states()) > 300, "A to move")
        await until(lambda: b.states(0) and moved(b.states(0)) > 300, "B to see A move")
        await until(lambda: live(a.states(2)), "A to see B")
        kept = coordinator.rooms[room].entries
        await until(lambda: kept.get(0) and kept.get(2), "a memory entry of A and of B")
        c = await take_seat(local, http, room, 3)
        late = next(w.client for w in local.workers if w.session.seat.slot == 3)
        await until(
            lambda: {s.slot for s in live(c.states(0) + c.states(2))} == {0, 2}, "C to see both"
        )
        await until(lambda: {(0, (0, 16)), (2, (0, 16))} <= late.entries, "C's kept entries")
        for browser in (a, b, c):
            await browser.close()
        await until(lambda: not coordinator.rooms[room].players, "the seats to be free")
        lobby = await (await http.get(f"{local.url}/api/lobby")).json()
        return taken.status, a, lobby


@pytest.mark.parametrize("route", ["direct", "proxy"])
def test_three_players_share_a_room(route):
    taken, a, lobby = asyncio.run(three_players(route))
    assert taken == 409  # a taken seat is refused
    # every frame names the newest input it folds in, and the inputs reach the frames in order
    newest = [header["q"] for header in a.frames]
    assert newest == sorted(newest) and newest[-1] > 0
    assert all(len(header["tm"]) == 3 for header in a.frames)
    # everyone left: the workers are free again
    workers = lobby["workers"]
    assert (workers["total"], workers["free"]) == (3, 3)
    assert sorted(w["id"] for w in workers["list"]) == ["local-1", "local-2", "local-3"]
    assert lobby["rooms"][0]["players"] == []


class GatedClient(RecordingClient):
    """A mock client that, once the room's other client plays (``peer``), holds each block's last
    frame until that client has received the block's live state, for at most :data:`GATE_S`;
    ``late`` counts the blocks whose state had not arrived by then."""

    def __init__(self, config, library):
        super().__init__(config, library)
        self.peer: GatedClient | None = None
        self.states: dict[int, float] = {}  # player slot -> room time of its newest state
        self.received = threading.Condition()
        self.gated = self.late = 0

    def receive_message(self, message):
        super().receive_message(message)
        if message.kind == LIVE_STATE:
            with self.received:
                self.states[message.slot] = message.meta["rows"][-1][1]
                self.received.notify_all()

    def step(self, controls):
        for frame in super().step(controls):
            peer = self.peer
            if (frame.index + 1) % self.frames_per_block == 0 and peer.sessions:
                t = round(frame.state.t, 4)  # the room time its state carries on the wire
                with peer.received:
                    arrived = peer.received.wait_for(
                        lambda: peer.states.get(self.slot, -1.0) >= t, GATE_S
                    )
                self.gated, self.late = self.gated + 1, self.late + (not arrived)
            yield frame


async def two_gated_players():
    async with demo(2, client=GatedClient) as (local, http):
        first, second = local.clients
        first.peer, second.peer = second, first
        room = await create_room(local, http)
        browsers = [await take_seat(local, http, room, slot) for slot in (0, 2)]
        await until(lambda: min(first.gated, second.gated) >= 4, "four gated blocks of each")
        for browser in browsers:
            await browser.close()
        return first.late, second.late


def test_a_message_reaches_the_room_before_its_blocks_last_frame():
    """The worker sends its client's messages as the client queues them, not after the block."""
    assert asyncio.run(two_gated_players()) == (0, 0)


class LeavingClient(RecordingClient):
    """A mock client that queues a ``done`` when its session stops, as the WorldCast client
    does."""

    def __init__(self, config, library):
        super().__init__(config, library)
        self.playing = False

    def start(self, round_start, seat):
        super().start(round_start, seat)
        self.playing = True

    def stop(self):
        super().stop()
        if self.playing:
            self.playing = False
            self.put(ClientMessage(self.slot, "done", {}, b""))


async def a_player_leaves():
    async with demo(2, client=LeavingClient) as (local, http):
        room = await create_room(local, http)
        leaver = await take_seat(local, http, room, 0)
        stayer = await take_seat(local, http, room, 2)
        staying = next(w.client for w in local.workers if w.session.seat.slot == 2)
        await until(lambda: (LIVE_STATE, 0) in staying.events, "a state of the leaver")
        await leaver.close()
        await until(lambda: ("removed", 0) in staying.events, "the leaver to leave the roster")
        await stayer.close()
        return [event for event in staying.events if event[1] == 0]


def test_a_leavers_done_reaches_the_room_before_it_leaves():
    """The leaver's worker sends its client's last messages before ``left``, after which the
    coordinator relays nothing of it."""
    assert asyncio.run(a_player_leaves())[-2:] == [("done", 0), ("removed", 0)]


class QuietClient(RecordingClient):
    """A mock client that sends the messages of its first twelve blocks of 4 frames, twelve live
    states and three memory entries, and nothing after them."""

    def start(self, round_start, seat):
        super().start(round_start, seat)
        self.put_count = 0

    def put(self, message):
        self.put_count += 1
        if self.put_count <= 12 + 3:
            super().put(message)


async def a_late_joiner_gets_what_was_kept():
    """With the bound B = 2, player 0 sends what it sends; player 2 joins, then player 0
    leaves."""
    async with demo(2, client=QuietClient, **{"scene_state.bound": 2}) as (local, http):
        room = await create_room(local, http)
        first = await take_seat(local, http, room, 0)
        kept = local.coordinator.rooms[room]

        def last_kept() -> list | None:
            entries = kept.entries.get(0)
            return entries and entries[-1][0]["meta"]["frames"]

        await until(lambda: last_kept() == [32, 48], "player 0's last message")
        counts = len(kept.live[0]), len(kept.entries[0])
        late = await take_seat(local, http, room, 2)
        joiner = next(w.client for w in local.workers if w.session.seat.slot == 2)
        await until(lambda: len(joiner.events) >= 6, "the kept messages")
        await first.close()
        await until(lambda: 0 not in kept.players, "player 0 to leave")
        dropped = 0 not in kept.live and 0 not in kept.entries
        await late.close()
        return counts, joiner.events, dropped


def test_a_late_joiner_gets_the_kept_live_states_before_the_memory_entries():
    """The coordinator keeps the newest 2 B live states and B memory entries of each player and
    sends a joiner the live states first, so the joiner draws the player at once and keys each kept
    entry at its cameras; it drops them when the player leaves."""
    counts, events, dropped = asyncio.run(a_late_joiner_gets_what_was_kept())
    assert counts == (4, 2)
    assert events[:6] == [(LIVE_STATE, 0)] * 4 + [(MEMORY_ENTRY, 0)] * 2
    assert dropped


async def api_errors(tmp_path):
    (tmp_path / "preview.jpg").write_bytes(b"image")
    (tmp_path / "notes.txt").write_text("not an image")
    library = synthetic_library()
    library.root = tmp_path
    async with demo(2, library, **{"coordinator.max_rooms": 1}) as (local, http):
        coordinator, rooms = local.coordinator, f"{local.url}/api/rooms"

        async def post(url, **kwargs):
            response = await http.post(url, **kwargs)
            return response.status, await response.text()

        status = {
            "an unknown round start": await post(rooms, json={"round_start": "nowhere"}),
            "a body that is not JSON": await post(rooms, data=b"round_start=arena"),
        }
        room = await create_room(local, http)
        status["more rooms than allowed"] = await post(rooms, json={"round_start": "arena"})
        join = f"{rooms}/{room}/join"
        status["a seat the round start lacks"] = await post(join, json={"slot": 7})
        status["a seat that is not a number"] = await post(join, json={"slot": "left"})
        # a worker that cannot start the client: the join fails and the seat stays free
        for worker in local.workers:
            worker.library = Library()
        status["a worker without the round start"] = await post(join, json={"slot": 0})
        # the room's workers run one kind of client: the free worker of another kind is not taken
        for worker in local.workers:
            worker.library = library
        coordinator.workers["local-2"].client = "worldcast"
        first = await take_seat(local, http, room, 0)
        status["no free worker of the room's kind"] = await post(join, json={"slot": 1})
        await first.close()
        await until(lambda: not coordinator.rooms[room].players, "the seat to be free")
        files = {
            name: (await http.get(f"{local.url}/library/{name}")).status
            for name in ("preview.jpg", "notes.txt", "missing.png", "%2e%2e/library.json")
        }
        coordinator.cfg.coordinator.empty_room_s = 0.0
        coordinator.close_empty_rooms()
        return status, files, list(coordinator.rooms)


def test_the_api_refuses_what_it_cannot_serve(tmp_path):
    status, files, rooms = asyncio.run(api_errors(tmp_path))
    assert status == {
        "an unknown round start": (404, "unknown round start 'nowhere'"),
        "a body that is not JSON": (400, "the request body must be a JSON object"),
        "more rooms than allowed": (503, "the server has as many rooms as it allows"),
        "a seat the round start lacks": (400, "round start arena has no such seat"),
        "a seat that is not a number": (400, "round start arena has no such seat"),
        "a worker without the round start": (
            503,
            "could not start your client: unknown round start 'arena'",
        ),
        "no free worker of the room's kind": (
            503,
            "every GPU is busy; try again when a player leaves",
        ),
    }
    assert files == {
        "preview.jpg": 200,
        "notes.txt": 404,
        "missing.png": 404,
        "%2e%2e/library.json": 404,
    }
    assert rooms == []  # the empty room was closed
