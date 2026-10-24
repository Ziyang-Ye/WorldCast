"""A worker that refreshes the block on screen (``play.refresh``), on the CPU with the mock client.

* The send queue: a frame goes shortly before the pacing's display shows it, a refresh's frame
  takes the place of one not yet sent (or not yet decoded) and never of one sent, a block that came
  early drops what the display would show after it, and the queue says which frame will not have
  gone by when.
* A key pressed in the middle of a block shows in that block: the frames not yet sent are the
  refresh's, from the first not yet sent on, every frame comes once and in order, a block's frames
  come a frame period apart, and the block's live state leaves once, with the newest controls; so
  too when a refresh takes the decode of the block's frames over (the engine's client with a
  decoder): its frames come from the block's iterator, again for frames already put.
* Refreshes coalesce and stay clear of the next block: none runs past the start of the next
  block's step, which commits the block on screen, and the pacing counts a block's critical delay
  from that start.
* A flick of the mouse larger than a block holds asks for a refresh at once, and the block on screen
  and the next do it within the generator's trained range, the turn conserved.
* Without ``play.refresh`` the worker sends a block's frames as they are decoded, as it always did.
"""

import asyncio
import copy
import logging
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from tests.demo.support import RecordingClient, create_room, demo, take_seat, until
from worldcast.data.controls import CONTROL_BUTTONS
from worldcast.data.latents import FPS, VIDEO_FRAMES_PER_LATENT
from worldcast.demo.clients import LIVE_STATE
from worldcast.demo.config import REFRESH_YAW_GAIN
from worldcast.demo.wire import pack, unpack
from worldcast.demo.worker import SendQueue, Worker, _marked_again

FORWARD = 1 << CONTROL_BUTTONS.index("forward")
#: The send queue's lead, s: ``play.send_lead_ms``'s default.
LEAD_S = 0.04
#: Blocks of 16 frames at 15 fps (1.07 s a block), denoised in 0.1 s.
REFRESH = {
    "play.refresh": True,
    "client.mock.frames_per_block": 16,
    "client.mock.denoise_ms": 100,
}


async def queued() -> tuple[list, dict]:
    sent = []
    queue = SendQueue(
        0.2, lambda k, messages, first, again: sent.append((k, messages, first, now())), LEAD_S
    )
    t0 = time.monotonic()

    def now() -> float:
        return time.monotonic() - t0

    shown = [queue.put(k, {85: b"%d" % k}, k == 0, t0) for k in range(4)]
    seen = {"unsent": [queue.replaceable(t0 + t) for t in (0.1, 0.5, 0.7, 0.95)]}
    seen["replaced"] = queue.replace(2, {85: b"two"})
    seen["early"] = queue.replace(5, {85: b"five"})  # a refresh's frame before the block's own
    # the next block comes as the display needs it: its first frame is shown a period after the last
    shown += [queue.put(k, {85: b"%d" % k}, k == 4, t0 + 0.4) for k in (4, 5)]
    seen["shown"] = [t - t0 for t in shown]
    await asyncio.sleep(0.05)
    seen["late"] = queue.replace(0, {85: b"zero"})  # went at once
    await asyncio.sleep(1.1)
    queue.close()
    return sent, seen


def test_the_send_queue_sends_each_frame_shortly_before_the_display_shows_it():
    sent, seen = asyncio.run(queued())
    np.testing.assert_allclose(seen["shown"], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], atol=1e-9)
    # frames 0-3 go at 0, 0.16, 0.36 and 0.56 s; the next ones would follow a period apart
    assert seen["unsent"] == [1, 3, 4, 5]
    assert seen["replaced"] and seen["early"] and not seen["late"]
    assert [k for k, *_ in sent] == list(range(6))
    assert [first for *_, first, _ in sent] == [True, False, False, False, True, False]
    assert [m[85] for _, m, _, _ in sent] == [b"0", b"1", b"two", b"3", b"4", b"five"]
    gone = np.array([t for *_, t in sent])
    np.testing.assert_allclose(gone, [0.0] + [0.2 * k - LEAD_S for k in range(1, 6)], atol=0.03)


async def early_block() -> list[int]:
    sent = []
    queue = SendQueue(0.05, lambda k, messages, first, again: sent.append(k), LEAD_S)
    t0 = time.monotonic()
    for k in range(6):  # shown 0 to 0.25 s
        queue.put(k, {85: b""}, k == 0, t0)
    # the next block's first frame, ready as early: the display shows it three frame periods after
    # it is ready and drops the frames it would show after it, as the page does
    assert queue.put(6, {85: b""}, True, t0 + 0.001) == t0 + 0.001 + 3 * 0.05
    await asyncio.sleep(0.5)
    queue.close()
    return sent


def test_a_block_that_came_early_drops_the_frames_the_display_would_show_after_it():
    assert asyncio.run(early_block()) == [0, 1, 2, 3, 6]


async def swapped(delay: float = 0.0) -> tuple[list, dict]:
    sent = []
    # shown a fifth of a second apart, the page showing each 0.3 s after the display less the lead;
    # a frame sent again goes half a frame period (0.1 s) and ``delay`` before that
    queue = SendQueue(
        0.2,
        lambda k, messages, first, again: sent.append((k, messages, first, again)),
        0.04,
        0.3,
        lambda: delay,
    )
    t0 = time.monotonic()
    for k in range(4):
        queue.put(k, {85: b"%d" % k}, k == 0, t0)
    seen = {"replaceable": [queue.replaceable(t0 + t) for t in (0.1, 0.5, 0.7)]}
    await asyncio.sleep(0.25)  # frames 0 and 1 went, the page shows frame 1 at 0.46 s
    seen["again"] = queue.replace(1, {85: b"one"})
    seen["queued"] = queue.replace(3, {85: b"three"})
    await asyncio.sleep(0.25)  # the page has shown frame 1
    seen["late"] = queue.replace(1, {85: b"uno"})
    await asyncio.sleep(0.4)
    seen["resent"] = [k for k, _ in queue.resent]
    queue.close()
    return sent, seen


def test_with_swap_a_newer_version_of_a_frame_sent_goes_while_it_reaches_the_page_in_time():
    """``swap``: a frame that went may be replaced, at once, while it would reach the page before
    the page shows it: the page's margin after the display less the lead, less half a frame
    period and the time a frame queued now takes to reach the page (``play.swap_sent``); it goes
    as a block's first frame of none, marked as sent again. The frames a refresh would replace
    start at the first that a newer version would still reach in time, not at the first not
    sent."""
    sent, seen = asyncio.run(swapped())
    # frame 0 (shown at 0, sent again until 0.16 s), frame 1 (0.2 s, 0.36 s), frame 2 (0.4, 0.56)
    assert seen["replaceable"] == [0, 2, 3]
    assert seen["again"] and seen["queued"] and not seen["late"]
    assert [(k, m[85], first, again) for k, m, first, again in sent] == [
        (0, b"0", True, False),
        (1, b"1", False, False),
        (1, b"one", False, True),
        (2, b"2", False, False),
        (3, b"three", False, False),
    ]
    assert seen["resent"] == [1]
    # a link that takes 0.15 s to deliver a frame queued now: frame 1 can no longer go again in
    # time at 0.25 s (until 0.21 s)
    sent, seen = asyncio.run(swapped(delay=0.15))
    assert seen["replaceable"] == [1, 3, 4] and not seen["again"] and seen["resent"] == []


def test_a_frame_sent_again_says_so_in_its_header():
    message = pack({"k": 7, "f": False, "pv": False, "v": 2}, b"jpeg")
    header, payload = unpack(_marked_again(message))
    assert header == {"k": 7, "f": False, "pv": False, "v": 2, "r": 1} and payload == b"jpeg"


def room(**state) -> tuple:
    """A worker whose client has ``cards`` cards (one free, or not), and a session whose block on
    screen has frames left to send and a refresh that fits before the next block."""
    cards, free, generating = state["cards"], state["free"], state["generating"]
    client = types.SimpleNamespace(
        can_refresh=True, refresh_cards=cards, refresh_card_free=free, frames_per_block=16
    )
    worker = types.SimpleNamespace(client=client)
    session = types.SimpleNamespace(
        generating={(1, 1)} if generating else set(),
        refresh_span=0.1,
        decode_lead=0.01,
        newest=1,
        sender=types.SimpleNamespace(replaceable=lambda at: 5),
    )
    return worker, session


@pytest.mark.parametrize(
    "cards, free, generating, room_for_it",
    [
        (1, True, False, True),
        (1, True, True, False),
        (2, True, True, True),
        (2, False, False, False),
    ],
)
def test_with_more_cards_a_refresh_waits_for_a_free_card_not_for_the_one_generated(
    cards, free, generating, room_for_it
):
    """On one card a refresh waits for the one being generated; with more (the WorldCast client's
    ``refresh_devices``) it waits only while no card is free."""
    worker, session = room(cards=cards, free=free, generating=generating)
    assert Worker._room_to_refresh(worker, session, 0.0, 1.0) == room_for_it


class Refreshing(RecordingClient):
    """The mock client with ``refresh``, recording when each step began and gave its first frame,
    when each refresh was asked for, began and ended, the rows of each version of a block, and the
    rows of each live state it sends."""

    def __init__(self, config, library):
        super().__init__(config, library)
        self.refreshes = True
        self.steps, self.first_out, self.asked, self.refreshed = [], [], [], []
        self.versions, self.live, self.preview_steps = [], [], []

    def step(self, controls):
        self.steps.append(time.monotonic())
        self.versions.append([])
        frames = super().step(controls)
        first = next(frames)  # the controls are taken: the block's rows are known
        self.first_out.append(time.monotonic())
        yield first
        self.versions[-1].append([view.wire() for view in self._open[2]])
        yield from frames

    def refresh(self, controls, switch, generated=None, preview_step=0):
        self.asked.append(time.monotonic())
        self.preview_steps.append(preview_step)
        return self._refresh(controls, switch, generated, preview_step)

    def _refresh(self, controls, switch, generated, preview_step=0):
        began = time.monotonic()
        yield from super().refresh(controls, switch, generated, preview_step)
        self.versions[-1].append([view.wire() for view in self._open[2]])
        self.refreshed.append((began, time.monotonic()))

    def put(self, message):
        if message.kind == LIVE_STATE:
            self.live.append(message.meta["rows"])
        super().put(message)


class RefreshingApart(Refreshing):
    """:class:`Refreshing` taken as a client whose decoder runs on a device of its own: the
    worker requests the next block while the frames of this one are taken on another thread."""

    blocks_in_flight = 2


class TakingOver(RefreshingApart):
    """:class:`RefreshingApart` as the engine's client with a decoder on a card of its own: its
    steps and refreshes are generated in turn on a thread of their own (a step prepares its block
    for 0.25 s before it takes the controls, as the engine's commits the block before and
    prepares), its frames decoded as they are taken, 15 ms more a frame (60 ms a latent frame, the
    Wan2.2 VAE's). A refresh generated while its block's frames are taken takes the decode over at
    a latent frame (every fourth frame): the block's iterator yields the refresh's frames, from its
    switch on but not past the first frame not yet yielded (those before come again), and the
    refresh's own iterator yields none."""

    #: A frame's decode, and a step's prepare, s.
    decode_s, prepare_s = 0.015, 0.25

    def __init__(self, config, library):
        super().__init__(config, library)
        self.generation = ThreadPoolExecutor(1)
        self.pending, self.taken_over = [], 0
        self.lock = threading.Lock()

    @property
    def latent_s(self) -> float:
        """A latent frame's decode, s."""
        return self.decode_s * VIDEO_FRAMES_PER_LATENT

    def _views(self, body, taken):
        return [
            body.advance(taken.buttons[j], taken.turn[j], float(taken.times[j]), 1 / FPS)
            for j in range(taken.num_frames)
        ]

    def step(self, controls):
        self.steps.append(time.monotonic())
        versions, block = [], {}
        self.versions.append(versions)

        def generate():
            time.sleep(self.prepare_s)
            self._publish_open()
            before, taken = copy.deepcopy(self.body), controls()
            views = self._views(self.body, taken)
            block.update(first=self.frame_index, taken=taken, views=views)
            self._open = before, self.frame_index, views
            versions.append([view.wire() for view in views])
            self.frame_index += len(views)
            time.sleep(self.cfg.denoise_ms / 1000.0)

        return self._block(self.generation.submit(generate), block)

    def _block(self, generated, block):
        generated.result()
        start, taken, views, version = block["first"], block["taken"], block["views"], 0
        j = 0
        while j < len(views):
            refresh = self._take(start) if j % VIDEO_FRAMES_PER_LATENT == 0 else None
            if refresh is not None:
                version, views, taken = refresh["version"], refresh["views"], refresh["taken"]
                self.taken_over += 1
                j = min(max(refresh["switch"](self.latent_s) - start, 0), j)
            time.sleep(self.decode_s)
            if j == 0 and not version:
                self.first_out.append(time.monotonic())
            yield replace(self._frame(start + j, taken, j, views[j]), version=version)
            j += 1

    def refresh(self, controls, switch, generated=None, preview_step=0):
        self.asked.append(time.monotonic())
        before, first, _ = self._open
        refresh, versions = {"switch": switch, "first": first}, self.versions[-1]

        def generate():
            began, taken = time.monotonic(), controls()
            body = copy.deepcopy(before)
            views = self._views(body, taken)
            self.body, self._open = body, (before, first, views)
            versions.append([view.wire() for view in views])
            time.sleep(self.cfg.denoise_ms / 1000.0)
            with self.lock:
                version = 1 + sum(1 for r in self.pending if r["first"] == first)
                refresh.update(views=views, taken=taken, version=version)
                self.pending.append(refresh)
            self.refreshed.append((began, time.monotonic()))
            if generated is not None:
                generated()

        return self._own(self.generation.submit(generate), refresh)

    def _take(self, first):
        """The newest refresh of the block of ``first`` generated and not taken over, with those
        before it."""
        with self.lock:
            ready = [r for r in self.pending if r["first"] == first and not r.get("over")]
            for r in ready:
                r["over"] = True
        return ready[-1] if ready else None

    def _own(self, generated, refresh):
        """The refresh's own frames: none if taken over or a newer one of the block came."""
        generated.result()
        first = refresh["first"]
        with self.lock:
            later = self.pending[self.pending.index(refresh) + 1 :]
            skip = refresh.get("over") or any(r["first"] == first for r in later)
            refresh["over"] = True
        if skip:
            return
        views, taken = refresh["views"], refresh["taken"]
        for j in range(max(refresh["switch"](self.latent_s) - first, 0), len(views)):
            time.sleep(self.decode_s)
            yield replace(self._frame(first + j, taken, j, views[j]), version=refresh["version"])


async def presses(
    refresh: bool,
    *,
    count: int = 4,
    turning: bool = False,
    client: type = Refreshing,
    **settings,
):
    """A player who stands, then presses forward ``count`` times, each 1.7 s after the one before,
    letting go 0.85 s after it (a press lands at another place of a block each time); with
    ``turning`` it turns 3 degrees with every sample, so that every sample asks for a refresh.
    Returns the browser, the time of each press by its sample, and the worker's client and
    session."""
    settings = {**REFRESH, "play.refresh": refresh, **settings}
    async with demo(1, client=client, **settings) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0, hold=None)
        session = local.workers[0].session
        await until(lambda: len(browser.frames) >= 3 * 16, "three blocks")
        pressed, seq, start = {}, 0, time.monotonic()
        for i in range(count * 2):
            held = FORWARD if i % 2 == 0 else 0
            ends = start + 0.85 * (i + 1) + 0.13 * (i % 3)
            while time.monotonic() < ends:
                seq += 1
                if held and not any(t > ends - 1.0 for t in pressed.values()):
                    pressed[seq] = time.monotonic()
                dy = 3.0 if turning else 0.0
                sample = {"t": "in", "q": seq, "b": held, "dp": 0.0, "dy": dy, "w": 2}
                await browser.ws.send_json(sample)
                await asyncio.sleep(0.016)
        await asyncio.sleep(1.5)
        await browser.close()
        return browser, pressed, local.workers[0].client, session


def latencies(browser, pressed: dict[int, float]) -> tuple[list[float], list[bool]]:
    """Per press: from its sample to the arrival of the first frame that holds it, and whether that
    frame is the first of its block."""
    waits, firsts = [], []
    for seq, at in pressed.items():
        i = next(i for i, header in enumerate(browser.frames) if header["q"] >= seq)
        waits.append(browser.arrived[i] - at)
        firsts.append(browser.frames[i]["f"])
    return waits, firsts


@pytest.mark.parametrize("client", [Refreshing, RefreshingApart], ids=["one-thread", "apart"])
def test_a_key_pressed_in_a_block_shows_in_that_block(client):
    browser, pressed, client, _ = asyncio.run(presses(True, client=client))
    waits, firsts = latencies(browser, pressed)
    # the press reaches the frames not yet sent of the block on screen: about the fake denoising
    # and the frames already on their way, never a wait for the next block
    assert max(waits) < 0.6 and np.median(waits) < 0.45, waits
    assert not all(firsts)  # a press shows in the middle of a block
    assert len(client.refreshed) >= len(pressed)
    # every frame once, in order (a decoder apart has the second block of a session come at once,
    # early, and the frames of the first block shown after it are dropped, as the page drops them
    # without refresh); from the third block on every frame, a frame period apart, not in a burst
    frames = [header["k"] for header in browser.frames]
    assert frames == sorted(set(frames))
    assert [k for k in frames if k >= 2 * 16] == list(range(2 * 16, frames[-1] + 1))
    assert np.median(np.diff(browser.arrived[3 * 16 :])) > 0.5 / 15
    # one live state a block, as the next block starts (the last one's as the client stops), with
    # the rows of the block's newest version
    assert len(client.live) == len(client.steps)
    for rows, versions in zip(client.live, client.versions):
        assert rows == versions[-1]


def test_a_refresh_with_a_preview_sends_the_previews_frames_first_marked():
    """With ``play.preview_step`` the worker asks each refresh for its preview: the mock gives the
    frames of the switch's latent frame, marked, half its denoising in; the version's own frames
    take their places in the send queue where they have not gone, so the browser still gets every
    frame once, in order, and a key pressed shows as soon as without the preview."""
    browser, pressed, client, _ = asyncio.run(presses(True, **{"play.preview_step": 2}))
    assert client.preview_steps and set(client.preview_steps) == {2}
    frames = [header["k"] for header in browser.frames]
    assert frames == sorted(set(frames))
    assert all("pv" in header for header in browser.frames)
    waits, _ = latencies(browser, pressed)
    assert max(waits) < 0.6, waits


def test_a_refresh_that_takes_the_decode_over_keeps_every_frame_once_in_order():
    """Every sample turns, and a refresh generated while its block's frames are decoded takes the
    decode over (:class:`TakingOver`): its frames come from the block's iterator, again for those
    put already, which the send queue replaces; the browser still gets every frame once, in order,
    with no stall, and each block's live state leaves with its newest version's rows."""
    browser, pressed, client, _ = asyncio.run(
        presses(True, count=3, turning=True, client=TakingOver)
    )
    assert client.taken_over >= 3
    # every frame once, in order; from the third block on every frame (the second block of a
    # session comes at once, early, and the display drops the first's after it)
    frames = [header["k"] for header in browser.frames]
    assert frames == sorted(set(frames))
    assert [k for k in frames if k >= 2 * 16] == list(range(2 * 16, frames[-1] + 1))
    assert np.diff(browser.arrived[3 * 16 :]).max() < 2.5 / 15  # no stall
    waits, _ = latencies(browser, pressed)
    assert max(waits) < 1.067 + 0.6, waits  # never a wait for a block after the next
    assert len(client.live) == len(client.steps)
    for rows, versions in zip(client.live, client.versions):
        assert rows == versions[-1]


def test_refreshes_coalesce_and_stay_clear_of_the_next_block():
    """Every sample turns: a refresh starts as soon as the one before it ended, and the newest
    input goes with it, the first of a block asked for as soon as the block is (before its first
    frame, while its denoising runs); none runs into the step of the next block, and the blocks
    keep coming at the frame rate."""
    browser, _, client, session = asyncio.run(presses(True, count=3, turning=True))
    spans = sorted(client.refreshed)
    assert len(spans) >= 6
    early = [
        any(began <= asked < out for asked in client.asked)
        for began, out in zip(client.steps, client.first_out)
    ]
    assert sum(early) >= 3  # a block refreshed whole, before any of its frames came
    assert all(before[1] <= after[0] for before, after in zip(spans, spans[1:]))
    for began, ended in spans:
        following = [t for t in client.steps if t > began]
        assert not following or ended <= following[0]
    assert [header["k"] for header in browser.frames] == list(range(len(browser.frames)))
    assert np.diff(browser.arrived[3 * 16 :]).max() < 2.5 / 15  # no stall
    # each block's critical delay counts from the start of its step: the fake denoising at least
    assert min(session.pacing.critical) >= 0.1


def test_without_refresh_a_press_waits_for_the_next_block():
    browser, pressed, client, _ = asyncio.run(presses(False))
    waits, firsts = latencies(browser, pressed)
    assert all(firsts) and not client.refreshed  # it shows from a block's first frame
    assert len(client.live) == len(client.steps)  # each block's live state at its cut


class FailingRefresh(Refreshing):
    """:class:`Refreshing` whose refreshes fail once their frames are taken."""

    def _refresh(self, controls, switch, generated, preview_step=0):
        yield from ()
        raise RuntimeError("the refresh failed")


def test_a_refresh_that_fails_is_said_and_the_blocks_play_on(caplog):
    """A refresh whose frames fail leaves the block's frames as they are: the worker says so, and
    a press shows from the next block, as without refresh."""
    with caplog.at_level(logging.ERROR, logger=Worker.__module__):
        browser, pressed, client, _ = asyncio.run(presses(True, count=2, client=FailingRefresh))
    failed = [r for r in caplog.records if r.name == Worker.__module__]
    assert failed and all("'s refresh of block " in r.getMessage() for r in failed)
    assert all(isinstance(r.exc_info[1], RuntimeError) for r in failed)
    assert client.asked and not client.refreshed
    _, firsts = latencies(browser, pressed)
    assert all(firsts)


async def two_refreshing_players() -> tuple[list, list, Refreshing]:
    """Two players of a room, each on a worker that refreshes; the first walks."""
    async with demo(2, client=Refreshing, **REFRESH) as (local, http):
        room = await create_room(local, http)
        walker = await take_seat(local, http, room, 0, hold=FORWARD)
        other = await take_seat(local, http, room, 1)
        clients = {w.session.seat.slot: w.client for w in local.workers}
        await until(lambda: len(other.frames) >= 5 * 16, "five blocks of the second player")
        await walker.close()
        await other.close()
        received = [slot for kind, slot in clients[1].events if kind == LIVE_STATE]
        return other.states(0), received, clients[0]


def test_players_who_refresh_see_each_other_from_the_live_states_their_blocks_leave_with():
    """The walker's blocks are refreshed while it walks (its samples turn); each block's live
    state leaves once, as the next block starts, and the other player draws the walker from them,
    moving."""
    seen, received, walker = asyncio.run(two_refreshing_players())
    assert walker.refreshed and len(walker.live) == len(walker.steps)
    assert received and set(received) == {0} and len(received) <= len(walker.live)
    xs = [state.x for state in seen]
    assert len(xs) > 16 and max(xs) - min(xs) > 50.0  # the walker moves in the other's view


class SlowFirstRefreshing(RefreshingApart):
    """:class:`RefreshingApart` whose session's first block denoises for 2 s, as a cold GPU's
    would: the second block, asked for at once, waits for it too."""

    def step(self, controls):
        first = self.frame_index == 0

        def take():
            taken = controls()
            if first:
                time.sleep(2.0)
            return taken

        return super().step(take)


def test_a_slow_first_block_leaves_no_backlog_that_holds_the_newest_block_back():
    """The two first blocks' slow controls-to-frames keep the pacing's lead long for 16 blocks,
    and the blocks after them start early: the send queue drops what the display would show after
    them, as the page does without refresh, so a press a few blocks in still shows within the
    block it lands in, not after a backlog of the blocks that came early."""
    browser, pressed, client, _ = asyncio.run(presses(True, client=SlowFirstRefreshing))
    waits, _ = latencies(browser, pressed)
    assert max(waits) < 0.6, waits
    frames = [header["k"] for header in browser.frames]
    assert frames == sorted(set(frames))


class Turning(Refreshing):
    """:class:`Refreshing` that keeps the controls of every cut and refresh (``cuts``)."""

    def __init__(self, config, library):
        super().__init__(config, library)
        self.cuts = []

    def step(self, controls):
        return super().step(lambda: self._kept(controls()))

    def _refresh(self, controls, switch, generated, preview_step=0):
        return super()._refresh(lambda: self._kept(controls()), switch, generated, preview_step)

    def _kept(self, taken):
        self.cuts.append(taken)
        return taken


async def flicked() -> tuple[float, Turning]:
    """A player who stands, flicks the mouse 400 degrees to the left in one sample four frames
    into a block, and stands again. Returns when the flick was sent and the client."""
    async with demo(1, client=Turning, **REFRESH) as (local, http):
        room = await create_room(local, http)
        browser = await take_seat(local, http, room, 0, hold=None)
        await until(lambda: len(browser.frames) >= 3 * 16, "three blocks")
        await until(lambda: browser.frames[-1]["k"] % 16 == 4, "a block's fourth frame")
        seq, sent = 0, 0.0
        for i in range(150):
            seq += 1
            dy = 400.0 if i == 0 else 0.0
            await browser.ws.send_json({"t": "in", "q": seq, "b": 0, "dp": 0.0, "dy": dy, "w": 2})
            sent = sent or time.monotonic()
            await asyncio.sleep(0.016)
        await browser.close()
        return sent, local.workers[0].client


def test_a_flick_refreshes_at_once_and_goes_on_into_the_next_block_within_the_trained_range():
    """A flick of 400 degrees asks for a refresh at once; the block on screen does from its switch
    on what its rows hold at 20 / ``REFRESH_YAW_GAIN`` degrees a row (fed ``REFRESH_YAW_GAIN``
    times: the generator's trained 20), and the next block's cut does the rest from its first row:
    the blocks as committed turn 400 in all, no row past the range."""
    sent, client = asyncio.run(flicked())
    assert min(t for t in client.asked if t >= sent) - sent < 0.1
    newest = {}
    for taken in client.cuts:  # the newest version of each block
        newest[float(taken.times[0])] = taken
    yaw = np.concatenate([newest[t].turn[:, 1] for t in sorted(newest)])
    assert yaw.sum() == pytest.approx(400.0, abs=1e-3)
    assert np.abs(yaw).max() <= 20.0 / REFRESH_YAW_GAIN + 1e-6
    turning = [t for t in sorted(newest) if newest[t].turn[:, 1].any()]
    assert len(turning) >= 2 and newest[turning[1]].turn[0, 1] == pytest.approx(
        20.0 / REFRESH_YAW_GAIN
    )
