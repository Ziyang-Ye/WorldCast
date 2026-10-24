"""A worker's life: its client's warm-up before its link to the coordinator, the link, and a
session's start, grace and end; the pacing of its blocks; the flow control of its browser sockets,
alone and with a browser on a slow link."""

import asyncio
import logging
import threading
import time
import types

import numpy as np
import pytest
from aiohttp import web

from tests.demo.support import (
    FORWARD,
    TIMEOUT_S,
    Browser,
    PipelinedClient,
    RecordingClient,
    SlowFirstBlock,
    create_room,
    demo,
    sized_jpeg,
    stamp,
    take_seat,
    until,
    watch,
)
from worldcast.data.controls import CONTROL_BUTTONS
from worldcast.demo import coordinator as coordinator_module
from worldcast.demo import worker as worker_module
from worldcast.demo.clients import LIVE_STATE, MEMORY_ENTRY, ClientMessage
from worldcast.demo.config import load_config
from worldcast.demo.library import DEFAULT_LOADOUT, synthetic_library
from worldcast.demo.local import LocalDemo, listen
from worldcast.demo.mock import MockClient
from worldcast.demo.pacing import BACKLOG_FRAMES, PACING_MARGIN_S, Pacing
from worldcast.demo.worker import BrowserSocket


async def refused_then_served(monkeypatch):
    """A worker whose coordinator first refuses the upgrade, then is replaced by a real one."""
    monkeypatch.setattr(worker_module, "RECONNECT_S", 0.01)
    monkeypatch.setattr(worker_module, "RECONNECT_MAX_S", 0.01)
    refused = []

    async def refuse(request):
        refused.append(request.path)
        raise web.HTTPNotFound()

    app = web.Application()
    app.add_routes([web.get("/ws/worker", refuse)])
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    sock = listen("127.0.0.1", 0)
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    library = synthetic_library()
    settings = {"worker.coordinator_url": f"ws://127.0.0.1:{port}", "worker.worker_id": "w"}
    config = load_config((), settings)
    worker = worker_module.Worker(config, MockClient(config.client.mock, library), library)
    worker_runner = web.AppRunner(worker.app())
    await worker_runner.setup()
    try:
        await until(lambda: len(refused) >= 2, "the worker to retry")
        alive = not worker._link.done()
        await runner.cleanup()
        config = load_config((), {"coordinator.host": "127.0.0.1", "coordinator.port": port})
        async with LocalDemo(config, library, []) as local:
            await until(lambda: "w" in local.coordinator.workers, "the worker to reconnect")
        return alive
    finally:
        await worker_runner.cleanup()


def test_the_link_outlives_a_coordinator_that_refuses_it(monkeypatch):
    assert asyncio.run(refused_then_served(monkeypatch))


class WarmingClient(RecordingClient):
    """A mock client whose warm-up takes a moment and notes the workers the coordinator of
    ``local`` knows when it ends."""

    local: LocalDemo

    def warm_up(self):
        time.sleep(0.2)
        self.events.append(("warm_up", sorted(self.local.coordinator.workers)))


async def warms_up_first() -> tuple[list, list]:
    settings = {"coordinator.host": "127.0.0.1", "coordinator.port": 0, "worker.port": 0}
    config = load_config((), settings)
    library = synthetic_library()
    client = WarmingClient(config.client.mock, library)
    client.local = LocalDemo(config, library, [client])
    async with client.local as local:
        await until(lambda: local.coordinator.workers, "the worker's hello")
        return client.events, sorted(local.coordinator.workers)


def test_a_worker_warms_its_client_up_before_it_says_hello():
    events, registered = asyncio.run(warms_up_first())
    assert events == [("warm_up", [])] and registered == ["local-1"]


async def bad_message_then_join() -> tuple[dict, float]:
    async with demo(1, **{"play.release_after_s": 0.5}) as (local, http):
        link = local.coordinator.workers["local-1"].ws
        await link.send_json({"t": "assign"})  # no room, no slot: the handler raises
        await link.send_bytes(b"\x00\x00\x00\x02{}")  # a client message without a slot
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 1)
        await until(lambda: browser.frames, "a frame")
        release_after_s = local.workers[0].session.inputs.release_after_s
        await browser.close()
        return browser.welcome, release_after_s


def test_a_message_that_fails_does_not_end_the_link():
    welcome, release_after_s = asyncio.run(bad_message_then_join())
    assert release_after_s == 0.5  # play.release_after_s: the player's inputs are cut with it
    assert welcome["seat"]["slot"] == 1 and welcome["fps"] == 16.0  # play.fps
    assert welcome["frames_per_block"] == 4
    assert welcome["buttons"] == list(CONTROL_BUTTONS)  # bit i of an input sample
    # the page's play settings (play.jitter_buffer_frames, play.margin_frames,
    # play.mouse_degrees_per_pixel)
    play = (
        welcome["jitter_buffer_frames"],
        welcome["margin_frames"],
        welcome["mouse_degrees_per_pixel"],
    )
    assert play == (1, 1.5, 0.03)
    assert welcome["backlog_frames"] == BACKLOG_FRAMES  # what the pacing's display model keeps to
    assert list(welcome["weapon_ids"]) == welcome["seat"]["loadout"] == list(DEFAULT_LOADOUT["T"])


async def player_never_connects(monkeypatch):
    monkeypatch.setattr(worker_module, "FIRST_CONNECT_S", 0.05)
    async with demo(1) as (local, http):
        coordinator = local.coordinator
        room = await create_room(local, http)
        joined = await http.post(f"{local.url}/api/rooms/{room}/join", json={"slot": 0})
        await until(lambda: local.workers[0].session is None, "the session to end")
        await until(lambda: not coordinator.rooms[room].players, "the seat to be free")
        return joined.status


def test_a_seat_nobody_opens_is_released(monkeypatch):
    assert asyncio.run(player_never_connects(monkeypatch)) == 200


async def comes_back():
    """A browser drops its connection and opens it again."""
    async with demo(1, **{"worker.reconnect_grace_s": 600.0}) as (local, http):
        worker = local.workers[0]
        room = await create_room(local, http)
        first = await take_seat(local, http, room, 0)
        session = worker.session
        await until(lambda: first.frames, "a frame")
        await first.close(bye=False)
        await until(lambda: session.player is None, "the worker to notice")
        waiting = not session.grace.done()
        second = await Browser.open(http, first.url)
        await until(lambda: second.frames, "a frame after the reconnect")
        await until(session.grace.cancelled, "the grace to be cancelled")
        resumed = worker.session is session
        await second.close()
        return waiting, resumed


def test_a_browser_that_comes_back_keeps_its_seat():
    assert asyncio.run(comes_back()) == (True, True)


async def turns_and_comes_back() -> tuple[Browser, Browser, float]:
    """A player holds forward and turns 0.5 degrees of yaw per input sample; its browser drops the
    connection, opens it again and turns on. Also the yaw of the samples the worker received before
    the second socket."""
    async with demo(1, **{"worker.reconnect_grace_s": 600.0}) as (local, http):
        worker = local.workers[0]
        room = await create_room(local, http)
        first = await take_seat(local, http, room, 0, hold=FORWARD)
        session = worker.session
        await until(lambda: len(first.frames) >= 16, "four blocks")
        await first.close(bye=False)
        await until(lambda: session.player is None, "the worker to notice")
        received = 0.5 * max(session.arrivals)  # the samples are numbered from 1
        second = await Browser.open(http, first.url, hold=FORWARD)
        await until(lambda: len(second.frames) >= 16, "four blocks after the reconnect")
        await second.close()
        return first, second, received


def test_frames_say_how_far_the_player_has_turned_and_a_new_socket_where_to_count_from():
    """A frame's ``tn`` is the turn of the samples folded into the controls through the frame,
    with the camera's pitch at the frame, ``cp`` (the mock's own state is each frame's); the
    welcome's ``tn`` is the turn of the samples the session received before its socket, from which
    a page that comes back counts on."""
    first, second, received = asyncio.run(turns_and_comes_back())
    assert first.welcome["tn"] == [0.0, 0.0] and second.welcome["tn"] == [0.0, received]
    assert [h["cp"] for h in first.frames] == [h["s"][6] for h in first.frames]
    blocks: dict[int, list[dict]] = {}
    for header in first.frames:
        blocks.setdefault(header["k"] // 4, []).append(header)
    whole = [headers for headers in blocks.values() if len(headers) == 4]  # none skipped
    assert len(whole) >= 3
    for headers in whole:
        # the block's last frame holds all of its samples' turn, spread over the block's 4 frames
        # (the mock's block is what ``front`` spreads over: any mapping cuts it so)
        assert headers[-1]["tn"] == [0.0, 0.5 * max(headers[-1]["q"], 0)]
        steps = np.diff([h["tn"][1] for h in headers])
        assert np.allclose(steps, steps[0])
    # the second page's samples, numbered from 1 again, after those of the first
    last = next(h for h in reversed(second.frames) if h["k"] % 4 == 3)
    assert last["tn"] == [0.0, received + 0.5 * last["q"]] and last["q"] <= len(second.sent)


async def first_blocks() -> Browser:
    async with demo(1) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0)
        await until(lambda: len(browser.frames) >= 12, "three blocks")
        await browser.close()
        return browser


def test_a_blocks_first_frame_says_so():
    """The page drops the frames before a block's first frame that wait beyond its backlog: the
    header's ``f`` is true on the first frame of each block (``static/stream.js``)."""
    browser = asyncio.run(first_blocks())
    assert [header["f"] for header in browser.frames[:12]] == [True, False, False, False] * 3


async def replaced():
    """A second play socket opens on the seat while the first is still open (a second tab)."""
    async with demo(1) as (local, http):
        worker = local.workers[0]
        room = await create_room(local, http)
        first = await take_seat(local, http, room, 0)
        await until(lambda: first.frames, "a frame")
        session, old = worker.session, worker.session.player
        second = await Browser.open(http, first.url)
        await until(lambda: first.ws.closed, "the worker to close the first socket")
        await until(lambda: len(second.frames) >= 8, "frames on the second socket")
        kept = worker.session is session and session.player not in (None, old)
        await second.close()
        await first.close(bye=False)
        return kept, first.ws.close_code


def test_a_new_play_socket_replaces_an_open_one():
    assert asyncio.run(replaced()) == (True, 1000)


async def replaced_while_turning(monkeypatch) -> Browser:
    """A second play socket opens on the seat of a player who turns 0.5 degrees of yaw per input
    sample; the first socket takes 0.3 s to close, and sends samples on meanwhile."""
    close = BrowserSocket.close

    async def slow_close(sock, *args, **kwargs):
        await asyncio.sleep(0.3)
        await close(sock, *args, **kwargs)

    monkeypatch.setattr(BrowserSocket, "close", slow_close)
    async with demo(1) as (local, http):
        room = await create_room(local, http)
        first = await take_seat(local, http, room, 0, hold=FORWARD)
        await until(lambda: len(first.frames) >= 4, "a block")
        second = await Browser.open(http, first.url, hold=FORWARD)
        sent = len(first.sent)
        await until(lambda: first.ws.closed, "the worker to close the first socket")
        assert len(first.sent) - sent >= 5  # samples sent after the second socket's welcome
        await until(lambda: len(second.frames) >= 32, "eight blocks on the second socket")
        await second.close()
        await first.close(bye=False)
        return second


def test_a_replaced_sockets_samples_count_no_more(monkeypatch):
    """The second socket's welcome counts the turn of the samples received before it; what the
    first still sends is not the player's, or the page would show every frame turned by it, for
    good."""
    second = asyncio.run(replaced_while_turning(monkeypatch))
    last = [h for h in second.frames if h["k"] % 4 == 3][-1]
    assert last["tn"] == [0.0, second.welcome["tn"][1] + 0.5 * last["q"]]
    assert last["q"] <= len(second.sent)


async def stays_away():
    async with demo(1, **{"worker.reconnect_grace_s": 0.05}) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0)
        await browser.close(bye=False)
        await until(lambda: local.workers[0].session is None, "the session to end")
        await until(lambda: not local.coordinator.rooms[room].players, "the seat to be free")
        return local.workers[0].client.sessions


def test_a_browser_that_stays_away_loses_its_seat():
    assert asyncio.run(stays_away()) == 1


class UnsendableFirst(RecordingClient):
    """A mock client whose first session as player 0 queues a message the worker cannot send: its
    meta is not JSON."""

    def start(self, round_start, seat):
        super().start(round_start, seat)
        if self.sessions == 1 and seat.slot == 0:
            self.put(ClientMessage(seat.slot, LIVE_STATE, {"rows": {0}}, b""))


async def relay_fails():
    """Player 0's relay fails in its first session; it leaves and takes the seat again."""
    async with demo(2, client=UnsendableFirst) as (local, http):
        room = await create_room(local, http)
        other = await take_seat(local, http, room, 2)
        first = await take_seat(local, http, room, 0)
        await until(lambda: first.frames, "a frame")
        await first.close()
        await until(lambda: 0 not in local.coordinator.rooms[room].players, "the seat to be free")
        receiver = next(w.client for w in local.workers if w.session.seat.slot == 2)
        before = (LIVE_STATE, 0) in receiver.events
        again = await take_seat(local, http, room, 0)
        await until(lambda: (LIVE_STATE, 0) in receiver.events, "player 0's state")
        await again.close()
        await other.close()
        return before


def test_a_relay_that_fails_frees_the_seat_and_the_next_session_relays(caplog):
    with caplog.at_level(logging.ERROR, logger=worker_module.__name__):
        relayed_before = asyncio.run(relay_fails())
    assert not relayed_before
    assert [r.getMessage() for r in caplog.records] == ["room arena-1: player 0's relay failed"]


async def release(monkeypatch):
    """The coordinator gives up on an assignment the worker answers too late."""
    monkeypatch.setattr(coordinator_module, "ASSIGN_TIMEOUT_S", 0.0)
    async with demo(1) as (local, http):
        room = await create_room(local, http)
        worker, ended = local.workers[0], []
        end_session = worker._end_session

        async def record(reason):
            ended.append(reason)
            await end_session(reason)

        monkeypatch.setattr(worker, "_end_session", record)
        joined = await http.post(f"{local.url}/api/rooms/{room}/join", json={"slot": 0})
        await until(lambda: ended and worker.session is None, "the worker to end the session")
        text, link = await joined.text(), local.coordinator.workers["local-1"]
        return joined.status, text, link.room, list(ended)


def test_an_assignment_that_times_out_is_released(monkeypatch):
    status, text, room, ended = asyncio.run(release(monkeypatch))
    assert (status, text) == (503, "could not start your client: the GPU worker did not answer")
    assert room is None and ended == ["the coordinator released the seat"]


async def markers_on_request() -> Browser:
    async with demo(2) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0)
        other = await take_seat(local, http, room, 3)
        await until(lambda: any(header["p"] for header in browser.frames), "a frame of player 3")
        await browser.show_markers()
        await until(lambda: "m" in browser.frames[-1], "a frame with the field's markers")
        await browser.show_markers(on=False)
        await until(lambda: "m" not in browser.frames[-1], "a frame without them")
        shown = len(browser.frames)
        await until(lambda: len(browser.frames) >= shown + 8, "two blocks more")
        await browser.close()
        await other.close()
        return browser


def test_frames_carry_the_fields_markers_while_the_page_asks_for_them():
    """The player's page turns the markers on and off with its ``M`` key; every frame names how
    its player's own row is known (``st``) and carries the players as the field reads them."""
    frames = asyncio.run(markers_on_request()).frames
    assert steps(["m" in header for header in frames]) == [False, True, False]
    assert {header["st"] for header in frames} == {"live"}
    # player 0 at its spawn; the only other player in its world is player 3, who plays: the
    # synthetic arena's players 1 and 2, whom nobody plays, are never in it
    assert frames[0]["s"][2:7] == [-600.0, -300.0, 0.0, 30.0, 0.0]
    seen = [p for header in frames for p in header["p"]]
    assert {(p[0], p[10]) for p in seen} == {(3, 1)}
    marked = next(header for header in frames if "m" in header)
    assert [m[0] for m in marked["m"]] == [3]


class StartingClient(RecordingClient):
    """A mock client whose session as player 2 starts once ``go`` is set."""

    def __init__(self, config, library):
        super().__init__(config, library)
        self.go = threading.Event()

    def start(self, round_start, seat):
        if seat.slot == 2:
            assert self.go.wait(TIMEOUT_S)
        super().start(round_start, seat)


async def comings_and_goings() -> list:
    """Player 0 plays; player 2 takes a seat and, while its client starts, player 0 leaves and
    player 1 takes a seat; player 1 leaves."""
    async with demo(3, client=StartingClient) as (local, http):
        room = await create_room(local, http)

        def client(slot: int) -> StartingClient:
            return next(
                w.client for w in local.workers if w.session and w.session.seat.slot == slot
            )

        first = await take_seat(local, http, room, 0)
        await until(lambda: first.frames, "a frame of player 0")
        joined = await http.post(f"{local.url}/api/rooms/{room}/join", json={"slot": 2})
        assert joined.status == 200  # the worker accepts the seat before its client starts
        late = client(2)
        await first.close()
        second = await take_seat(local, http, room, 1)
        starting = next(w for w in local.workers if w.client is late)
        await until(lambda: set(starting.session.roster) == {1, 2}, "the roster to change")
        late.go.set()
        await until(lambda: starting.session.ready.is_set(), "player 2's client to start")
        await second.close()
        await until(lambda: ("removed", 1) in late.events, "player 1 to leave")
        return late.events


def test_the_client_is_told_who_leaves():
    """A client is told who left while it started once it has started, then as the roster
    changes; a player who takes a seat tells the client itself, by its first state."""
    events = asyncio.run(comings_and_goings())
    assert [event for event in events if event[0] == "removed"] == [("removed", 0), ("removed", 1)]


async def leaves_while_another_starts() -> tuple[list[dict], list]:
    """Player 0 walks; player 2 takes a seat and, while its client starts with what reached it of
    player 0 (the kept live states, then what player 0 sends), player 0 leaves."""
    async with demo(2, client=StartingClient) as (local, http):
        room = await create_room(local, http)
        first = await take_seat(local, http, room, 0, hold=FORWARD)
        await until(lambda: len(first.frames) >= 8, "two blocks of player 0")
        late = await take_seat(local, http, room, 2)
        starting = next(w for w in local.workers if w.session and w.session.seat.slot == 2)
        early = starting.session.early
        await until(lambda: any(m.kind == LIVE_STATE for m in early), "player 0's live states")
        await first.close()
        await until(lambda: set(starting.session.roster) == {2}, "player 0 to leave the roster")
        starting.client.go.set()
        await until(lambda: len(late.frames) >= 8, "two blocks of player 2")
        await late.close()
        return late.frames, starting.client.events


def test_a_player_who_leaves_while_another_starts_stays_gone():
    """The messages that came while the client started reach it before the roster that came
    after them: player 0's live states, then its departure. Player 2 never shows player 0."""
    frames, events = asyncio.run(leaves_while_another_starts())
    of_0 = [kind for kind, slot in events if slot == 0]
    assert LIVE_STATE in of_0 and of_0[-1] == "removed" and of_0.count("removed") == 1
    assert [p for header in frames for p in header["p"] if p[0] == 0] == []


class UnremovableOne(RecordingClient):
    """A mock client of player 0 that fails on the departure of player 1, as the WorldCast client
    does on a seat without a recording."""

    def remove_player(self, slot):
        if self.slot == 0 and slot == 1:
            raise KeyError("player slot 1 has no recording")
        super().remove_player(slot)


async def a_departure_the_client_fails_on() -> tuple[list, list[list[int]]]:
    """Player 0 plays; players 1 and 2 take a seat, then player 1 leaves, then player 2."""
    async with demo(3, client=UnremovableOne) as (local, http):
        room = await create_room(local, http)
        first = await take_seat(local, http, room, 0)
        await until(lambda: first.frames, "a frame of player 0")
        client = next(w.client for w in local.workers if w.session and w.session.seat.slot == 0)
        one = await take_seat(local, http, room, 1)
        two = await take_seat(local, http, room, 2)
        await until(lambda: [0, 1, 2] in first.rosters, "the roster of all three")
        await one.close()
        await until(lambda: [0, 2] in first.rosters, "player 1 to leave the roster")
        await two.close()
        await until(lambda: ("removed", 2) in client.events, "player 0's client to learn 2 left")
        shown = len(first.frames)
        await until(lambda: len(first.frames) >= shown + 8, "player 0 to play on")
        rosters = list(first.rosters)
        await first.close()
        return client.events, rosters


def test_a_departure_the_client_fails_on_does_not_hold_the_others(caplog):
    """Player 0's client fails on player 1's departure at each roster after it: the page still
    gets every roster, and the client still learns who else leaves."""
    with caplog.at_level(logging.ERROR, logger=worker_module.__name__):
        events, rosters = asyncio.run(a_departure_the_client_fails_on())
    assert [event for event in events if event[0] == "removed"] == [("removed", 2)]
    assert rosters == [[0, 1], [0, 1, 2], [0, 2], [0]]
    failed = "room arena-1: the client's remove_player(1) failed"
    assert [record.getMessage() for record in caplog.records] == [failed] * 2


# ---------------------------------------------------------------------------------------- pacing
class PreparingClient(RecordingClient):
    """A mock client that, before it takes the controls of its third block, prepares the block
    for a second."""

    blocks = 0

    def step(self, controls):
        self.blocks += 1
        if self.blocks == 3:
            time.sleep(1.0)
        return super().step(controls)


async def waits_for_a_prepare() -> list[float]:
    async with demo(1, client=PreparingClient) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0)
        worker = local.workers[0]
        await until(lambda: worker.client.blocks >= 5, "five blocks")
        critical = list(worker.session.pacing.critical)
        await browser.close()
        return critical


def test_a_block_whose_start_waited_for_the_prepare_does_not_raise_the_lead():
    critical = asyncio.run(waits_for_a_prepare())
    # each block's critical delay counts from its controls: the fake denoising (40 ms) and the
    # four frames, never the second the third block waited before them
    assert len(critical) >= 4 and max(critical) < 0.5


async def slow_first_block(monkeypatch) -> tuple[Browser, list[tuple[list[float], list[float]]]]:
    """The browser of a session whose first block is slow, and what the worker's pacing did with
    each block: when its frames were sent and when its display shows them."""
    blocks = []
    record = Pacing.record

    def spy(pacing, taken, arrivals):
        shown = record(pacing, taken, arrivals)
        blocks.append((list(arrivals), shown))
        return shown

    monkeypatch.setattr(Pacing, "record", spy)
    settings = {**BLOCKS_OF_16, "play.fps": 30.0}
    async with demo(1, client=SlowFirstBlock, **settings) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0)
        # the pacing records a block as its last frame is queued on the socket, the browser has
        # it later
        await until(lambda: len(blocks) >= 12 and len(browser.frames) >= 12 * 16, "twelve blocks")
        await browser.close()
        return browser, blocks[:12]


def test_a_slow_first_block_does_not_hold_the_blocks_after_it_early(monkeypatch):
    browser, blocks = asyncio.run(slow_first_block(monkeypatch))
    assert [header["k"] for header in browser.frames[: 12 * 16]] == list(range(12 * 16))
    arrived = np.array([t for sent, _ in blocks for t in sent])
    shown = np.array([t for _, shown in blocks for t in shown])
    # from the third block on, over the next 5.3 s, the display the worker paces by (play.fps = 30)
    # shows 30 frames a second ...
    steady = slice(2 * 16, 12 * 16)
    assert shown[-1] - shown[2 * 16] > 0.95 * 5.3
    assert (len(shown[steady]) - 1) / (shown[-1] - shown[2 * 16]) > 0.95 * 30
    # ... the block after the slow one comes while the display holds the slow block's frames:
    # the page drops them but a few, so its first frame is shown that many frames after it
    # arrived, and the display does not keep them in its schedule, which would hold every block
    # after it that much longer than the pacing margin; the HUD's latency stays at its steady
    # value, not the seconds the slow block's lead would keep for 16 blocks as the slowest
    waits = (shown - arrived)[::16]
    assert waits[1] == pytest.approx(BACKLOG_FRAMES / 30)
    assert waits[2:].max() < PACING_MARGIN_S + 2 / 30  # with the spread of the fake denoising


# ------------------------------------------------------------------ a decoder of its own
#: Blocks of 16 frames shown at 20 fps (0.8 s a block), denoised in 0.5 s, then decoded by a decoder
#: of its own, a group of four frames at a time.
OWN_DECODER = {
    "client.mock.frames_per_block": 16,
    "client.mock.denoise_ms": 500,
    "client.mock.decode_ms": 0,
    "play.fps": 20.0,
}


async def decoded_apart(
    decode_s: float, blocks: int, **settings
) -> tuple[Browser, PipelinedClient]:
    """A player's browser and the worker's client after ``blocks`` blocks of a client whose decoder
    takes ``decode_s`` per group of four frames."""
    client = type("Decoding", (PipelinedClient,), {"decode_s": decode_s})
    async with demo(1, client=client, **{**OWN_DECODER, **settings}) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0)
        await until(lambda: len(browser.frames) >= blocks * 16, f"{blocks} blocks")
        worker = local.workers[0]
        assert worker.frame_thread is not worker.client_thread
        await browser.close()
        return browser, worker.client


def block_starts(browser: Browser) -> tuple[np.ndarray, np.ndarray]:
    """When each block's first frame arrived, and the ms from its controls to its decode."""
    firsts = [i for i, header in enumerate(browser.frames) if header["f"]]
    return (
        np.array([browser.arrived[i] for i in firsts]),
        np.array([browser.frames[i]["tm"][1] for i in firsts]),
    )


def test_the_next_block_starts_while_the_decoder_gives_the_frames_of_this_one():
    """A block takes 0.5 s to denoise and 0.64 s to decode, against 0.8 s of display: one after the
    other they would take 1.14 s a block; the worker starts the next block while the decoder is
    still at the last, so the blocks come at the display's rate."""
    browser, client = asyncio.run(decoded_apart(0.16, 8))
    assert [header["k"] for header in browser.frames[: 8 * 16]] == list(range(8 * 16))
    requested, decoded = client.requested, client.decoded
    assert sum(requested[k + 1] < decoded[k] for k in range(len(decoded) - 1)) >= 5
    arrived, _ = block_starts(browser)
    assert 0.7 < np.median(np.diff(arrived[1:])) < 1.0
    # never more than two blocks started whose frames are not all out
    assert all(decoded[k - 2] <= requested[k] for k in range(2, len(requested)))


def test_a_decode_slower_than_the_display_plays_at_the_rate_of_the_decode_without_a_backlog():
    """The display wants a block every 0.267 s (60 fps) and the decoder gives one in 0.4 s: the
    worker never starts a third block, so one block waits for the decoder at the most; every frame
    comes, in order, at the decoder's rate (denoising and decoding in turn would take 0.7 s a
    block), and the time from a block's controls to its first frame does not grow."""
    settings = {"play.fps": 60.0, "client.mock.denoise_ms": 300}
    browser, client = asyncio.run(decoded_apart(0.1, 12, **settings))
    count = 12 * 16
    assert [header["k"] for header in browser.frames[:count]] == list(range(count))
    requested, decoded = client.requested, client.decoded
    assert all(decoded[k - 2] <= requested[k] for k in range(2, len(requested)))
    assert len(requested) - len(decoded) <= 2
    arrived, delays = block_starts(browser)
    assert 0.35 < np.median(np.diff(arrived[2:])) < 0.55
    assert np.median(delays[-4:]) - np.median(delays[2:6]) < 300  # ms


async def leaves_with_blocks_in_flight() -> tuple[Browser, int]:
    client = type("Decoding", (PipelinedClient,), {"decode_s": 0.1})
    async with demo(1, client=client, **OWN_DECODER) as (local, http):
        room = await create_room(local, http)
        first = await take_seat(local, http, room, 0)
        await until(lambda: len(first.frames) >= 20, "a block and more")
        await first.close()  # the next block is under way
        await until(lambda: not local.coordinator.rooms[room].players, "the seat to be free")
        second = await take_seat(local, http, room, 0)
        await until(lambda: len(second.frames) >= 16, "the worker to serve the next player")
        await second.close()
        return second, local.workers[0].client.sessions


def test_a_session_that_ends_with_a_block_in_flight_frees_the_worker_for_the_next_player():
    second, sessions = asyncio.run(leaves_with_blocks_in_flight())
    assert sessions == 2 and [h["k"] for h in second.frames[:16]] == list(range(16))


# ---------------------------------------------------------------------------------- flow control
class FakeSocket:
    """A browser's socket that keeps what is sent on it: a text message's type, a frame's text
    (``"k@quality"``)."""

    closed = False

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_json(self, message: dict) -> None:
        self.sent.append(message["t"])

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data.decode().rstrip())

    async def close(self, message: bytes = b"") -> None:
        self.closed = True


def frame(k: int, qualities=(85, 75, 60, 50), size: int = 0) -> dict[int, bytes]:
    """Frame ``k``'s message at each of ``qualities``: ``"k@quality"``, padded to ``size``
    bytes."""
    return {quality: f"{k}@{quality}".encode().ljust(size) for quality in qualities}


@pytest.fixture
def clock(monkeypatch) -> types.SimpleNamespace:
    """The worker's ``time.monotonic()``, set by the test (``clock.now``)."""
    clock = types.SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    monkeypatch.setattr(worker_module, "time", clock)
    return clock


async def settle() -> None:
    """Let a socket's task send what it may."""
    for _ in range(3):
        await asyncio.sleep(0)


def socket_of(ws: FakeSocket, qualities: tuple[int, ...], player: bool) -> BrowserSocket:
    """A socket at 15 fps that allows 0.25 s to deliver a frame (three frames of display time
    until the first acknowledgement), with blocks of 4 frames."""
    return BrowserSocket(ws, qualities, 0.25, 1 / 15, 4, player=player)


async def first_frames() -> list[str]:
    ws = FakeSocket()
    sock = socket_of(ws, (85,), player=True)
    for k in range(6):
        sock.send_frame(k, frame(k), first=k == 0)
    sock.send({"t": "pong"})
    await settle()
    return ws.sent


def test_until_the_first_acknowledgement_a_frame_takes_its_display_time(clock):
    # text messages go first; the third frame is delivered within 3/15 s, the fourth would not be
    assert asyncio.run(first_frames()) == ["pong", "0@85", "1@85", "2@85"]


async def over_a_slow_link(clock: types.SimpleNamespace) -> list[list[str]]:
    """Frames of 1000 bytes over a link that delivers one each 0.1 s."""
    ws = FakeSocket()
    sock = socket_of(ws, (85,), player=True)
    for k in range(6):
        sock.send_frame(k, frame(k, size=1000), first=k == 0)
    await settle()
    went = [list(ws.sent)]
    for k in range(3):
        clock.now += 0.1
        sock.ack(k)
        await settle()
        went.append(ws.sent[sum(map(len, went)) :])
    return went


def test_a_frame_goes_when_the_link_would_deliver_it_within_the_budget(clock):
    went = asyncio.run(over_a_slow_link(clock))
    # the link's rate measured, 10 kB/s: a frame goes once it would arrive within 0.25 s, behind
    # one frame in flight
    assert went == [["0@85", "1@85", "2@85"], [], ["3@85"], ["4@85"]]


async def a_long_round_trip(clock: types.SimpleNamespace, blocks: int) -> tuple[list[str], int]:
    """Blocks of 16 frames of 43 kB, each block's frames ready at once, over a link with a round
    trip of 0.35 s and no limit on its rate."""
    ws = FakeSocket()
    sock = BrowserSocket(ws, (85, 75, 60, 50), 0.25, 1 / 15, 16, player=True)
    block, acks = 0, []  # acks: (when it reaches the worker, frame), on the way back
    while block < blocks or acks:
        if block < blocks and (not acks or block * 16 / 15 <= acks[0][0]):
            clock.now = block * 16 / 15
            for k in range(16 * block, 16 * block + 16):
                sock.send_frame(k, frame(k, size=43_000), first=k % 16 == 0)
            block += 1
        else:
            clock.now, k = acks.pop(0)
            sock.ack(k)
        sent = len(ws.sent)
        await settle()
        acks += [(clock.now + 0.35, int(text.split("@")[0])) for text in ws.sent[sent:]]
    return ws.sent, sock.quality


def test_a_link_with_a_long_round_trip_gets_every_frame_at_the_best_quality(clock):
    sent, quality = asyncio.run(a_long_round_trip(clock, blocks=6))
    # at least two round trips of frames in flight: a block's 16 frames within its 16/15 s
    assert sent == [f"{k}@85" for k in range(6 * 16)] and quality == 85


async def a_player_behind(clock: types.SimpleNamespace) -> tuple[list[str], int]:
    ws = FakeSocket()
    sock = socket_of(ws, (85, 75, 60, 50), player=True)
    for k in range(8):  # a block at 0 s: three frames go, five wait
        sock.send_frame(k, frame(k), first=k == 0)
    await settle()
    clock.now = 1.0  # the next block, while five frames wait: they are dropped
    sock.send_frame(8, frame(8), first=True)
    sock.ack(3)
    await settle()
    for k in (8, 9):  # each block arrives at once: the socket is idle for most of the next
        clock.now += 0.1
        sock.ack(k)
        clock.now += 0.9
        sock.send_frame(k + 1, frame(k + 1), first=True)
        await settle()
    return ws.sent, sock.quality


def test_a_player_behind_skips_to_the_next_block_at_a_lower_quality(clock):
    sent, quality = asyncio.run(a_player_behind(clock))
    # frames 3-7 never go and the next block's frame goes a quality lower; two calm blocks later
    # the quality is back up
    assert sent == ["0@85", "1@85", "2@85", "8@75", "9@75", "10@85"]
    assert quality == 85 and worker_module.CALM_BLOCKS == 2


async def blocks_behind(clock: types.SimpleNamespace) -> tuple[list[str], list[int]]:
    """Blocks of 6 frames over a link that delivers nothing, each encoded at the quality the socket
    had when it began, so the encode lags the socket's quality by a block."""
    ws = FakeSocket()
    sock = socket_of(ws, (85, 75, 60, 50), player=True)
    quality, steps = 85, []
    for b in range(4):
        clock.now = float(b)
        for k in range(6 * b, 6 * b + 6):
            sock.send_frame(k, frame(k, (quality,)), first=k % 6 == 0)
        steps.append(sock.quality)
        quality = sock.quality
        await settle()
    return ws.sent, steps


def test_a_block_encoded_before_a_step_down_does_not_step_down_again(clock):
    sent, steps = asyncio.run(blocks_behind(clock))
    # block 1, encoded at 85 before the step to 75, waits too: the quality stays 75 until a block
    # encoded at 75 still waits
    assert steps == [85, 75, 75, 60]
    assert sent == ["0@85", "1@85", "2@85"]


async def a_spectator_behind(clock: types.SimpleNamespace) -> tuple[list[str], int]:
    ws = FakeSocket()
    sock = socket_of(ws, (75, 60, 50), player=False)
    for k in range(4):  # three frames go
        sock.send_frame(k, frame(k), first=k == 0)
    await settle()
    for k in range(4, 10):  # of the seven that wait, the socket keeps a block: four
        sock.send_frame(k, frame(k), first=False)
    await settle()
    clock.now = 1.0
    sock.send_frame(10, frame(10), first=True)  # behind: a quality lower, the oldest dropped
    sock.ack(3)
    await settle()
    return ws.sent, sock.quality


def test_a_spectator_behind_keeps_the_newest_block_at_a_lower_quality(clock):
    sent, quality = asyncio.run(a_spectator_behind(clock))
    assert sent == ["0@75", "1@75", "2@75", "7@75", "8@75", "9@75", "10@60"]
    assert quality == 60


async def closing() -> FakeSocket:
    ws = FakeSocket()
    sock = socket_of(ws, (85,), player=True)
    for k in range(6):
        sock.send_frame(k, frame(k), first=k == 0)
    await settle()
    sock.send({"t": "roster"})
    await sock.close({"t": "ended"})
    return ws


def test_a_closing_socket_drops_its_frames_and_sends_its_text_messages(clock):
    ws = asyncio.run(closing())
    assert ws.sent == ["0@85", "1@85", "2@85", "roster", "ended"] and ws.closed


#: The paper's blocks of 16 frames, shown at the demo's 15 fps, each block's frames ready within
#: about 0.1 s of each other, as on a GPU.
BLOCKS_OF_16 = {
    "client.mock.frames_per_block": 16,
    "client.mock.denoise_ms": 200,
    "client.mock.decode_ms": 5,
}


async def turns_through_blocks_of_16(turn: float) -> list[tuple[float, list[float]]]:
    """A player holds forward and turns ``turn`` degrees of yaw per input sample, one every 16 ms,
    under the ``hybrid`` mapping: of each whole block of 16 frames that followed a whole block, the
    yaw it turned (the samples folded into its controls) and the step of the turn the page counts
    (``tn``) at each of its frames."""
    async with demo(1, **BLOCKS_OF_16, **{"play.input_mapping": "hybrid"}) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0, hold=FORWARD, turn=turn)
        await until(lambda: len(browser.frames) >= 6 * 16, "six blocks")
        await browser.close()
    blocks: dict[int, list[float]] = {}
    for header in browser.frames:
        blocks.setdefault(header["k"] // 16, []).append(header["tn"][1])
    return [
        (yaws[-1] - blocks[b - 1][-1], np.diff([blocks[b - 1][-1], *yaws]).tolist())
        for b, yaws in blocks.items()
        if len(yaws) == 16 and len(blocks.get(b - 1, [])) == 16
    ]


@pytest.mark.parametrize(("turn", "spread"), [(0.1, 4), (1.0, 16)])
def test_a_block_of_16_frames_turns_in_its_first_four_below_20_degrees_and_in_all_above(
    turn, spread
):
    """The ``hybrid`` mapping puts a turn of under 20 degrees in the block's first latent
    frame and spreads one of 20 or more over its 16 frames: a block of about 66 samples turns
    6.6 degrees at 0.1 per sample and 66 at 1.0 (``tests/demo/test_inputs.py``)."""
    blocks = asyncio.run(turns_through_blocks_of_16(turn))
    assert len(blocks) >= 3
    for total, steps in blocks:
        assert (total < 20.0) == (spread == 4)
        assert np.allclose(steps, [total / spread] * spread + [0.0] * (16 - spread), atol=1e-9)


class TimedClient(RecordingClient):
    """A mock client that records when it queues each of its memory entries and when each of the
    other players' reaches it, by ``(slot, first frame)``."""

    def __init__(self, config, library):
        super().__init__(config, library)
        self.queued, self.reached = {}, {}

    def put(self, message):
        if message.kind == MEMORY_ENTRY:
            self.queued[message.slot, message.meta["frames"][0]] = time.monotonic()
        super().put(message)

    def receive_message(self, message):
        if message.kind == MEMORY_ENTRY:
            self.reached[message.slot, message.meta["frames"][0]] = time.monotonic()
        super().receive_message(message)


def ages(browser: Browser) -> np.ndarray:
    """How long after its encoding each frame reached the browser, s."""
    return np.array([t - stamp(jpeg)[0] for t, jpeg in zip(browser.arrived, browser.jpegs)])


def qualities(browser: Browser) -> list[int]:
    """The JPEG quality of each frame the browser received."""
    return [stamp(jpeg)[1] for jpeg in browser.jpegs]


def steps(values: list[int]) -> list[int]:
    """``values`` without repeats: the qualities the frames stepped through, in order."""
    return [v for i, v in enumerate(values) if i == 0 or v != values[i - 1]]


def whole_blocks(browser: Browser, quality: int) -> int:
    """The blocks of which the browser received all 16 frames, each at ``quality``."""
    blocks: dict[int, list[int]] = {}
    for header, q in zip(browser.frames, qualities(browser)):
        blocks.setdefault(header["k"] // 16, []).append(q)
    return sum(q == [quality] * 16 for q in blocks.values())


async def slow_player(route: str, monkeypatch):
    """Player 0's browser reads its frames over a 3 Mbit/s link until they come in whole blocks at
    the lowest quality, then over a fast one; player 2's reads at once."""
    monkeypatch.setattr(worker_module, "encode_jpeg", sized_jpeg)
    inputs: list[int] = []
    add_input = worker_module.Session.add_input

    def record(session, data, *, room_time):
        if session.seat.slot == 0:
            inputs.append(int(data["q"]))
        add_input(session, data, room_time=room_time)

    monkeypatch.setattr(worker_module.Session, "add_input", record)
    settings = {**BLOCKS_OF_16, "coordinator.frame_route": route}
    async with demo(2, client=TimedClient, **settings) as (local, http):
        room = await create_room(local, http)
        other = await take_seat(local, http, room, 2)
        slow = await take_seat(local, http, room, 0, mbit=3.0)
        await until(lambda: whole_blocks(slow, 50) >= 3, "whole blocks over the slow link")
        throttled, sent = len(slow.frames), list(slow.sent)
        slow.mbit = None
        await until(lambda: qualities(slow)[-1] == 85, "the best quality over the fast link")
        await until(lambda: len(inputs) >= len(sent), "the inputs sent over the slow link")
        sender, receiver = (
            next(w.client for w in local.workers if w.session.seat.slot == slot) for slot in (0, 2)
        )
        lags = [receiver.reached[key] - t for key, t in sender.queued.items() if key[1] < 16 * 8]
        await slow.close()
        await other.close()
        return slow, throttled, inputs[: len(sent)] == sent, lags


@pytest.mark.parametrize("route", ["direct", "proxy"])
def test_a_player_on_a_slow_link(route, monkeypatch):
    slow, throttled, inputs_in_order, lags = asyncio.run(slow_player(route, monkeypatch))
    # its frames step down to the quality the link carries, and back up over the fast link
    assert steps(qualities(slow)) == [85, 75, 60, 50, 60, 75, 85]
    # no backlog builds up: a frame is never 1.5 s old, and from the first frame at the quality
    # the link carries on, under a block (16/15 s) and the 0.25 s a frame may take on the link
    age = ages(slow)[:throttled]
    assert age.max() < 1.5 and age[qualities(slow).index(50) :].max() < 1.3
    # its client's memory entries of the first eight blocks reach the other player's at once,
    # and every input sample of its browser reaches the worker, in order
    assert len(lags) == 8 and max(lags) < 0.1
    assert inputs_in_order


async def player_between_qualities(monkeypatch) -> Browser:
    """Player 0's browser reads over a 4.6 Mbit/s link, which carries quality 75 (34 kB frames,
    4.1 Mbit/s) but not 85 (43 kB, 5.2 Mbit/s), for eight blocks."""
    monkeypatch.setattr(worker_module, "encode_jpeg", sized_jpeg)
    async with demo(1, **BLOCKS_OF_16) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0, mbit=4.6)
        await until(lambda: browser.frames and browser.frames[-1]["k"] >= 8 * 16, "eight blocks")
        await browser.close()
        return browser


def test_a_player_on_a_link_between_two_qualities_steps_down_once(monkeypatch):
    browser = asyncio.run(player_between_qualities(monkeypatch))
    # the block encoded at 85 before the step to 75 does not step it down again: the link
    # carries whole blocks at 75 from then on
    assert steps(qualities(browser)) == [85, 75]
    assert whole_blocks(browser, 75) >= 4


async def slow_spectator(route: str, monkeypatch):
    """A spectator of player 0 reads over a 3 Mbit/s link until its frames come in whole blocks at
    the lowest quality; the player's browser reads at once."""
    monkeypatch.setattr(worker_module, "encode_jpeg", sized_jpeg)
    settings = {**BLOCKS_OF_16, "coordinator.frame_route": route}
    async with demo(1, **settings) as (local, http):
        room = await create_room(local, http)
        player = await take_seat(local, http, room, 0)
        await until(lambda: player.frames, "the player's first frame")
        spectator = await watch(local, http, room, 0, mbit=3.0)
        since = len(player.frames)
        await until(lambda: whole_blocks(spectator, 50) >= 2, "the spectator's whole blocks")
        await spectator.close(bye=False)
        await player.close()
        return player, since, spectator


@pytest.mark.parametrize("route", ["direct", "proxy"])
def test_a_spectator_on_a_slow_link(route, monkeypatch):
    player, since, spectator = asyncio.run(slow_spectator(route, monkeypatch))
    # the player's stream keeps its frame rate, its freshness and its quality
    arrived = player.arrived[since:]
    assert (len(arrived) - 1) / (arrived[-1] - arrived[0]) >= 0.9 * 15
    assert np.percentile(ages(player)[since:], 95) < 0.05
    assert set(qualities(player)) == {85}
    # the spectator's own encode starts a step lower and steps down for its link alone
    assert steps(qualities(spectator)) == [75, 60, 50]


# ------------------------------------------------------------------------- a session has no end
class NoFrameFirst(RecordingClient):
    """A mock client that yields nothing at its third block, as a client that finished would, in
    its first session only."""

    def step(self, controls):
        self.blocks = getattr(self, "blocks", 0) + 1
        if self.sessions == 1 and self.blocks == 3:
            return iter(())
        return super().step(controls)


async def yields_nothing():
    async with demo(1, client=NoFrameFirst) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0)
        await until(lambda: browser.ended is not None, "the session to end")
        await browser.close(bye=False)  # the worker closed its socket
        await until(lambda: not local.coordinator.rooms[room].players, "the seat to be free")
        second = await take_seat(local, http, room, 0)
        await until(lambda: len(second.frames) >= 8, "the worker to serve the next player")
        await second.close()
        return browser.ended, local.workers[0].client.sessions


def test_a_block_that_yields_no_frame_ends_the_session_with_the_clients_failure():
    """A client that yields nothing is not the end of a session (a session has none): it failed.
    The worker says so, frees the seat, and serves the next player."""
    ended, sessions = asyncio.run(yields_nothing())
    assert ended == "the client failed: the client yielded no frame" and sessions == 2


async def plays_past_a_recorded_round():
    async with demo(1, **{"play.fps": 60.0, "client.mock.frames_per_block": 16}) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0, hold=FORWARD, turn=0.0)
        await until(lambda: len(browser.frames) >= 130, "frames past a recorded round's 113")
        ended = browser.ended
        await browser.close()
        return ended


def test_a_session_goes_on_past_the_length_of_the_recorded_round():
    """Seven seconds of frames (113: the first frame and 16 a block, seven blocks) were once the
    whole session of the WorldCast client; a session has no length of its own."""
    assert asyncio.run(plays_past_a_recorded_round()) is None


AWAY = {"play.away_s": 3.0, "play.idle_s": 3.0, "client.mock.frames_per_block": 4}


async def is_released(hold: int, *, hidden: bool = False, moves: bool = False, seconds: float):
    """Whether a player is released within ``seconds`` of its first frame: it holds the buttons
    ``hold`` in every sample, the page may be ``hidden`` (told as it is by ``vis``, again every
    second, as every new socket of a hidden page does), and its mouse may move (a sample of nothing
    but a mouse movement)."""
    async with demo(1, **AWAY) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0, hold=hold, turn=0.0)
        await until(lambda: len(browser.frames) >= 4, "a block")
        deadline = time.monotonic() + seconds
        seq, told = 10_000, 0.0
        while time.monotonic() < deadline and browser.ended is None:
            if hidden and time.monotonic() >= told:
                await browser.ws.send_json({"t": "vis", "h": True})
                told = time.monotonic() + 1.0
            if moves:
                seq += 1
                sample = {"t": "in", "q": seq, "b": 0, "dp": 0.0, "dy": 0.5, "w": 2}
                await browser.ws.send_json(sample)
            await asyncio.sleep(0.05)
        ended = browser.ended
        await browser.close(bye=ended is None)
        return ended


def test_a_page_that_stays_hidden_loses_its_seat_however_often_it_says_so():
    assert asyncio.run(is_released(FORWARD, hidden=True, seconds=10.0)) == "you were away"


async def shown_again():
    async with demo(1, **AWAY) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0, hold=FORWARD, turn=0.0)
        await until(lambda: len(browser.frames) >= 4, "a block")
        for _ in range(5):  # hidden for 1 s, shown again, over and over: never 3 s away
            await browser.ws.send_json({"t": "vis", "h": True})
            await asyncio.sleep(1.0)
            await browser.ws.send_json({"t": "vis", "h": False})
            await asyncio.sleep(0.3)
        ended = browser.ended
        await browser.close()
        return ended


def test_a_page_shown_again_keeps_its_seat():
    assert asyncio.run(shown_again()) is None


async def released_after(*shown_at: float) -> float:
    """Seconds from the first block to the release of a player that sends no key and no mouse
    movement, its page shown again at each of ``shown_at`` seconds."""
    async with demo(1, **AWAY) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0, hold=0, turn=0.0)
        await until(lambda: len(browser.frames) >= 4, "a block")
        begun = time.monotonic()
        for at in shown_at:
            await asyncio.sleep(max(0.0, begun + at - time.monotonic()))
            await browser.ws.send_json({"t": "vis", "h": False})
        await until(lambda: browser.ended is not None, "the release")
        released = time.monotonic() - begun
        await browser.close(bye=False)
        return released


def test_a_page_shown_again_starts_the_idle_time_over():
    """No key and no mouse movement for 3 s lose the seat; a page shown at 2 s is as good as a key
    at 2 s: the seat is lost 3 s after it, not before."""
    assert asyncio.run(released_after(2.0)) >= 2.0 + 3.0


def test_a_player_who_sends_no_key_and_no_mouse_movement_loses_its_seat():
    assert asyncio.run(is_released(0, seconds=10.0)) == "you were away"


def test_a_player_who_plays_is_never_released():
    assert asyncio.run(is_released(FORWARD, seconds=4.0)) is None  # a key held
    assert asyncio.run(is_released(0, moves=True, seconds=4.0)) is None  # the mouse moving


async def sent_again(clock: types.SimpleNamespace) -> dict:
    """Frames 0 and 1 of 1000 bytes, then frame 1 again (a newer version, ``play.swap_sent``); the
    acknowledgements come in the order the frames went, 0.1 s after them and 0.2 s."""
    ws = FakeSocket()
    sock = socket_of(ws, (85,), player=True)
    for k in range(2):
        sock.send_frame(k, frame(k, size=1000), first=k == 0)
    await settle()
    sock.send_frame(1, frame(1, size=1000), first=False)
    await settle()
    seen = {"sent": list(ws.sent), "delay": [sock.delivery_delay()]}
    clock.now = 0.1
    for k in (0, 1):
        sock.ack(k)
    seen["unacked"] = [k for k, _, _ in sock._unacked]
    seen["round_trip"] = sock._round_trip
    seen["delay"].append(sock.delivery_delay())
    clock.now = 0.2
    sock.ack(1)
    seen["left"] = [k for k, _, _ in sock._unacked]
    return seen


def test_a_frame_sent_again_is_acknowledged_by_its_own_acknowledgement(clock):
    """The acknowledgement of a frame takes the frames that went up to it, not a later copy of it
    sent again behind them: the round trip stays the time from a frame's send to its own
    acknowledgement. The time a frame queued now takes to reach the page is the bytes waiting and
    in flight at the link's rate, and half its round trip (none before the first
    acknowledgement)."""
    seen = asyncio.run(sent_again(clock))
    assert seen["sent"] == ["0@85", "1@85", "1@85"]
    assert seen["unacked"] == [1] and seen["left"] == []
    assert seen["round_trip"] == pytest.approx(0.1)
    # 2000 bytes delivered in the 0.1 s the link was busy; 1000 bytes still in flight
    assert seen["delay"] == [0.0, pytest.approx(1000 / 20_000 + 0.05)]
