"""A GPU worker: one client (models loaded once), one player at a time.

It takes the player's input samples from the browser (:mod:`worldcast.demo.inputs`), steps the
client block by block, streams each frame to the browser (and to spectators) as it is decoded, and
relays the client's messages (player states, memory entries, withdrawals) to and from the room's
other workers through the coordinator, each one as soon as the client queues it. A client steps on
the other clients' latest messages, so one slow player does not stall the others: the paper's
deployment. (Its evaluation runs the clients in lockstep: ``tools/run_session.py``.) The room's
roster tells the client who left (:meth:`ClientProtocol.remove_player`); a player who took a seat
tells the client itself, with its first state.

Pacing (:mod:`worldcast.demo.pacing`): a block is started just in time for its first frame to reach
the browser when the previous block runs out, so inputs are cut as late as possible (lowest latency)
and a fast client does not run ahead of the display. The worker warms its client up before it
registers with the coordinator (:meth:`ClientProtocol.warm_up`), so that its first player's blocks
are as fast as the later ones.

Flow control (:class:`BrowserSocket`): each browser socket sends from a queue of its own, and a
frame goes when the browser's link would deliver it within ``worker.max_inflight_ms``, so a slow
browser stalls neither the block loop nor the other browsers nor the room's messages; its frames
are skipped or dropped and their JPEG quality steps down instead.

Refresh (``play.refresh``): the frames wait in a :class:`SendQueue` until shortly before the
display shows them, and while the block on screen still has frames to send, an input that changed
cuts it again and the client denoises it again (:meth:`ClientProtocol.refresh`); its new frames
take the places of those not yet sent. The next block starts, and the block on screen is committed,
at the time the pacing gives: as late as its measured critical delay allows, counted from the commit
of the block before it.
"""

import asyncio
import contextlib
import itertools
import logging
import math
import socket
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import ClassVar

from aiohttp import ClientError, ClientSession, WSMessage, WSMsgType, web

from worldcast.data.controls import CONTROL_BUTTONS, encode_weapon
from worldcast.data.latents import VIDEO_FRAMES_PER_LATENT

from .clients import ClientMessage, ClientProtocol, LiveControls, LiveFrame, make_client
from .config import DemoConfig
from .inputs import INSPECT_WEAPONS, InputBuffer, InputSample
from .library import Library, RoundStart, Seat
from .pacing import BACKLOG_FRAMES, LEAD_WINDOW, Pacing, percentile
from .wire import (
    HEARTBEAT_S,
    LINK_MESSAGE_BYTES,
    WebSocket,
    encode_jpeg,
    load_jpeg_encoder,
    pack,
    send,
    unpack,
)

__all__ = [
    "BlockTrace",
    "BrowserSocket",
    "RefreshTrace",
    "SendQueue",
    "Session",
    "Worker",
    "block_frames",
    "run",
]

log = logging.getLogger(__name__)
#: Input samples remembered for the latency breakdown.
ARRIVALS_KEPT = 512
#: Seconds a newly seated player has to open the play socket (the client's start counts against it).
FIRST_CONNECT_S = 30.0
#: The coordinator link: the first and the longest wait before a reconnect, s.
RECONNECT_S, RECONNECT_MAX_S = 0.5, 10.0
#: Clock pings to the coordinator: this many in quick succession, then one per the second period.
CLOCK_BURST, CLOCK_BURST_S, CLOCK_PERIOD_S = 5, 0.2, 5.0
#: The JPEG qualities below ``worker.jpeg_quality`` that a browser's frames step down through while
#: its socket falls behind, and back up through once it keeps up (:class:`BrowserSocket`).
JPEG_QUALITY_STEPS = (75, 60, 50)

#: How soon a thread holding the interpreter lock hands it to a waiting one, s (Python's default is
#: 5 ms): the generation thread takes it back this soon after each call that released it while the
#: frames, the event loop and the decoder run beside it.
SWITCH_INTERVAL_S = 0.0005
#: A socket's quality steps back up after this many calm blocks in a row: blocks for which it was
#: idle at least this share of the time.
CALM_BLOCKS, CALM_IDLE = 2, 1 / 3
#: A browser's link rate is measured over its newest acknowledgements: about a second of frames.
RATE_ACKS = 16
#: A socket allows at least this many round trips of its link on the way, so that a link with a
#: long round trip still carries the full frame rate.
ROUND_TRIPS = 2
#: With ``play.swap_sent`` a newer version of a frame that went goes while the page would still
#: take it this many frame periods before it shows the frame (the page's display ticks, the link's
#: jitter).
SWAP_GUARD_FRAMES = 0.5
#: The entries a session keeps of what its send queue and its blocks and refreshes did
#: (:attr:`SendQueue.log`, :attr:`Session.trace`): about a minute of play.
TRACE_LENGTH = 1024


@dataclass
class BlockTrace:
    """What a block did with ``play.refresh``, for the stalls of ``tools/bench_live.py``; its times
    (``_t``) are ``time.monotonic()``.

    Attributes:
        kind (str): ``"block"``, of the class.
        first (int): its first frame.
        due_t (float | None): when its start was due (``None``: the display held no frame).
        started_t (float): when its step was asked for.
        cut_t (float): when its controls were cut.
        first_out_t (float): when its first frame was put in the send queue.
        last_out_t (float): when its last frame was.
    """

    kind: ClassVar[str] = "block"
    first: int
    due_t: float | None
    started_t: float
    cut_t: float
    first_out_t: float
    last_out_t: float


@dataclass
class RefreshTrace:
    """What a refresh did, filled in as it goes; its times (``_t``) are ``time.monotonic()``, 0
    until reached.

    Attributes:
        kind (str): ``"refresh"``, of the class.
        first (int): its block's first frame.
        asked_t (float): when it was asked for.
        cut_t (float): when its controls were cut.
        generated_t (float): when its latents were ready.
        switch_t (float): when the decoder took them.
        first_out_t (float): when its first frame was ready.
        last_out_t (float): when its last frame was.
        replaced (int): the first frame it replaced, -1 for none.
        frames (int): the frames it gave.
        preview_out_t (float): when its preview's first frame was ready.
    """

    kind: ClassVar[str] = "refresh"
    first: int
    asked_t: float
    cut_t: float = 0.0
    generated_t: float = 0.0
    switch_t: float = 0.0
    first_out_t: float = 0.0
    last_out_t: float = 0.0
    replaced: int = -1
    frames: int = 0
    preview_out_t: float = 0.0


async def block_frames(
    client: ClientProtocol,
    thread: ThreadPoolExecutor,
    controls: Callable[[], LiveControls],
    qualities: Sequence[int],
    frames_thread: ThreadPoolExecutor | None = None,
    *,
    request: Callable[[], Iterator[LiveFrame]] | None = None,
) -> AsyncIterator[tuple[LiveFrame, dict[int, bytes], float, float]]:
    """The client's frames of one block as they are decoded.

    The client steps and the frames are encoded on ``thread``; the event loop only sends. With a
    ``frames_thread`` the client only requests the block on ``thread``, and the frames are taken
    and encoded on ``frames_thread``, so that the next block can be requested meanwhile.

    Args:
        client (ClientProtocol): the client to step.
        thread (ThreadPoolExecutor): its thread.
        controls (Callable[[], LiveControls]): the block's controls, taken on ``thread``.
        qualities (Sequence[int]): the JPEG qualities to encode each frame at.
        frames_thread (ThreadPoolExecutor | None): a thread of its own to take the frames on; its
            blocks are taken in the order they are requested.
        request (Callable | None): what asks the client for the frames, on ``thread``; default
            ``client.step(controls)`` (a refresh: ``client.refresh``).

    Yields:
        tuple[LiveFrame, dict[int, bytes], float, float]: a frame, its JPEG at each quality, and
        the ``time.monotonic()`` when it was decoded and when it was encoded.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def work(frames: Callable[[], Iterator[LiveFrame]]):
        try:
            for frame in frames():
                ready = time.monotonic()
                jpegs = {quality: encode_jpeg(frame.rgb, quality) for quality in qualities}
                item = (frame, jpegs, ready, time.monotonic())
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as error:  # re-raised on the event loop
            loop.call_soon_threadsafe(queue.put_nowait, error)
        loop.call_soon_threadsafe(queue.put_nowait, done)

    request = request or (lambda: client.step(controls))
    if frames_thread is None:
        future = loop.run_in_executor(thread, work, request)
    else:
        requested = await loop.run_in_executor(thread, request)
        future = loop.run_in_executor(frames_thread, work, lambda: requested)
    while (item := await queue.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield item
    await future


class BrowserSocket:
    """A browser's socket on this worker, the player's or a spectator's, with a task of its own
    that sends on it: queueing a message never waits for the browser.

    Text messages go out first, in order, and are never dropped. Frames start with the first block
    that begins after the socket opened, so that each block it gets was encoded for it. The page
    acknowledges each frame it receives, and a frame goes when the link would deliver it, behind
    the frames in flight (sent and not yet acknowledged), within ``inflight_s``, or within
    :data:`ROUND_TRIPS` round trips of the link if that is longer. The link is measured from the
    acknowledgements: its rate, the bytes it delivered per second it had frames in flight over the
    newest :data:`RATE_ACKS` acknowledgements, and its round trip, the shortest time from a frame's
    send to its acknowledgement. Until the first acknowledgement, each frame is taken to need its
    display time. The other frames wait here, and one always goes when none is in flight.

    The browser is behind when a block's first frame comes while frames of an earlier block still
    wait: a player's socket then drops them, so that the new block shows as soon as the link
    allows, and a spectator's keeps the newest block of frames. Its JPEG quality then steps down,
    unless the frames that wait are of a quality above its own: those were encoded before its last
    step down, so the link has not yet been tried at the new quality. A block is calm when the
    socket was idle for at least :data:`CALM_IDLE` of it, nothing waiting and nothing in flight:
    the next quality up, 1.2-1.3 times the bytes at the default steps, then still fits the link.
    After :data:`CALM_BLOCKS` calm blocks in a row the quality steps back up.

    Args:
        ws (web.WebSocketResponse): the socket.
        qualities (Sequence[int]): the JPEG qualities its frames step through, the best first.
        inflight_s (float): how long the link may take to deliver a frame behind those in flight
            (at least :data:`ROUND_TRIPS` round trips), s.
        period_s (float): a frame's display time, s.
        frames_per_block (int): the frames of a block.
        player (bool): the player's socket, else a spectator's.
    """

    def __init__(
        self,
        ws: web.WebSocketResponse,
        qualities: Sequence[int],
        inflight_s: float,
        period_s: float,
        frames_per_block: int,
        *,
        player: bool,
    ) -> None:
        self.ws, self.qualities, self.player = ws, tuple(qualities), player
        self.inflight_s, self.period_s = inflight_s, period_s
        self.frames_per_block = frames_per_block
        self.rung = 0  # index of its quality in ``qualities``
        self._texts: deque[dict] = deque()
        self._frames: deque[tuple[int, int, bytes]] = deque()  # (frame index, quality, message)
        self._unacked: deque[tuple[int, int, float]] = deque()  # (frame index, bytes, when sent)
        self._waiting_bytes = self._unacked_bytes = 0  # of the frames that wait, and in flight
        self._rate = 0.0  # bytes a second the link delivers, of the newest acknowledgements
        # (bytes, s the link had frames in flight) of the newest acknowledgements
        self._acked: deque[tuple[int, float]] = deque(maxlen=RATE_ACKS)
        self._acked_at = -math.inf  # when the newest acknowledgement came
        self._round_trip = math.inf
        self._idle_since: float | None = time.monotonic()  # nothing waits and nothing is in flight
        self._block_at: float | None = None  # when the newest block's first frame came
        self._calm = 0
        self._closing = False
        self._ready = asyncio.Event()
        self._task = asyncio.create_task(self._send())

    @property
    def quality(self) -> int:
        """The JPEG quality of its frames now."""
        return self.qualities[self.rung]

    def send(self, message: dict) -> None:
        """Queue a text message."""
        self._texts.append(message)
        self._ready.set()

    def send_frame(self, k: int, messages: Mapping[int, bytes], first: bool) -> None:
        """Queue frame ``k``.

        Args:
            k (int): the frame's index in the session's video.
            messages (Mapping[int, bytes]): the frame's binary message at each JPEG quality it was
                encoded at; the socket takes the best one not above its own quality.
            first (bool): the frame is its block's first.
        """
        now = time.monotonic()
        if first:
            self._next_block(now)
        elif self._block_at is None:  # opened during a block: it starts with the next one
            return
        quality = max((q for q in messages if q <= self.quality), default=min(messages))
        self._frames.append((k, quality, messages[quality]))
        self._waiting_bytes += len(messages[quality])
        if not self.player:
            while len(self._frames) > self.frames_per_block:
                self._waiting_bytes -= len(self._frames.popleft()[2])
        self._idle_since = None
        self._ready.set()

    def ack(self, k: int) -> None:
        """The page received frame ``k``, and every frame sent before it: the acknowledgements
        come in the order the frames went, so it is the first of ``k`` in flight (a frame sent
        again goes after later frames); without one, the frames in flight up to ``k``."""
        now = time.monotonic()
        indices = [entry[0] for entry in self._unacked]
        if k in indices:
            count = indices.index(k) + 1
        else:
            count = next((i for i, index in enumerate(indices) if index > k), len(indices))
        acked = [self._unacked.popleft() for _ in range(count)]
        if acked:
            # the link had frames in flight since the acknowledgement before, or since the oldest
            # of these went into an idle link
            busy = now - max(self._acked_at, acked[0][2])
            size = sum(size for _, size, _ in acked)
            self._unacked_bytes -= size
            self._acked.append((size, busy))
            self._acked_at = now
            self._round_trip = min(self._round_trip, now - acked[-1][2])
            delivered, busy = (sum(column) for column in zip(*self._acked))
            self._rate = delivered / busy if busy > 0 else 0.0
        if not self._unacked and not self._frames and self._idle_since is None:
            self._idle_since = now
        self._ready.set()

    def delivery_delay(self) -> float:
        """How long a frame queued now takes to reach the page, s: the frames that wait and those in
        flight at the link's rate, and half its round trip (0 before the first acknowledgement; any
        thread)."""
        if not self._rate:
            return 0.0
        return (self._waiting_bytes + self._unacked_bytes) / self._rate + self._round_trip / 2

    def _next_block(self, now: float) -> None:
        """A block's first frame came: step the quality down if the browser is behind on frames of
        its quality, up after :data:`CALM_BLOCKS` calm blocks."""
        if self._frames:
            if self._frames[-1][1] <= self.quality:
                self.rung = min(self.rung + 1, len(self.qualities) - 1)
            if self.player:
                self._frames.clear()
                self._waiting_bytes = 0
            self._calm = 0
        elif self._block_at is not None:
            idle = 0.0 if self._idle_since is None else now - self._idle_since
            self._calm = self._calm + 1 if idle >= CALM_IDLE * (now - self._block_at) else 0
            if self._calm == CALM_BLOCKS:
                self.rung, self._calm = max(self.rung - 1, 0), 0
        self._block_at = now

    def _may_send_frame(self) -> bool:
        if not self._frames:
            return False
        if not self._unacked:
            return True
        if not self._acked:
            return (len(self._unacked) + 1) * self.period_s <= self.inflight_s
        in_flight = sum(size for _, size, _ in self._unacked) + len(self._frames[0][2])
        delivered, busy = (sum(column) for column in zip(*self._acked))
        # in_flight / rate <= the allowed time, with rate = delivered / busy
        allowed = max(self.inflight_s, ROUND_TRIPS * self._round_trip)
        return in_flight * busy <= allowed * delivered

    async def _send(self) -> None:
        while not self._closing or self._texts:
            await self._ready.wait()
            self._ready.clear()
            while self._texts or self._may_send_frame():
                if self._texts:
                    await send(self.ws, self._texts.popleft())
                else:
                    k, _, message = self._frames.popleft()
                    self._waiting_bytes -= len(message)
                    self._unacked_bytes += len(message)
                    self._unacked.append((k, len(message), time.monotonic()))
                    await send(self.ws, message)

    async def close(self, last: dict | None = None, reason: bytes = b"") -> None:
        """Drop the frames that wait, send the text messages and ``last``, close the socket."""
        self._frames.clear()
        self._waiting_bytes = 0
        if last is not None:
            self._texts.append(last)
        self._closing = True
        self._ready.set()
        await asyncio.wait({self._task})
        await self.ws.close(message=reason)

    def cancel(self) -> None:
        """The browser closed the socket: stop sending, unless :meth:`close` is under way."""
        if not self._closing:
            self._task.cancel()


@dataclass
class _Queued:
    """A frame in a :class:`SendQueue`: when it goes, its messages, whether it is its block's
    first, when the display shows it, when it was put, and how many refreshes replaced it."""

    go: float
    messages: Mapping[int, bytes]
    first: bool
    shown: float
    ready: float
    replaced: int


class SendQueue:
    """The frames of a session that wait for their time to go (``play.refresh``).

    A block's frames are put as they are ready, in order. Each is shown when the pacing's display
    shows it (:class:`~worldcast.demo.pacing.Pacing`: when it is ready, never sooner than a frame
    period after the frame before it, and a block's first at most
    :data:`~worldcast.demo.pacing.BACKLOG_FRAMES` frame periods after it is ready) and goes
    ``lead`` before that (``play.send_lead_ms``), or when it is ready if that is later. As the
    display drops the frames before a block's first frame that it would show after it (a block that
    came early, with the lead the first blocks of a session leave), so does the queue, of those not
    gone: the schedule stays the pacing's, and no backlog holds the newest block back. Until a frame
    goes, a refresh may replace it (:meth:`replace`), also before it is put; a frame that went
    stays, or with ``swap`` (``play.swap_sent``) a newer version of it goes again at once (marked:
    ``Session.send_frame``) as long as it would reach the page before the page shows the frame:
    ``swap`` after the pacing's display less the lead (the page's playout margin), less
    :data:`SWAP_GUARD_FRAMES` frame periods and the time a frame queued now takes to reach the page
    (``delay()``); the page shows it in place of the one it holds (``Playout.add`` in
    ``static/stream.js``). The frames go in order, from a task of the queue's own, through
    ``send``.

    Args:
        period (float): the display's frame period, s.
        send (Callable[[int, Mapping[int, bytes], bool, bool], None]): sends frame ``k`` (its
            messages, whether it is its block's first, whether it goes again),
            :meth:`Session.send_frame`.
        lead (float): how long before the display shows a frame it goes, s.
        swap (float): how long after the display, less the lead, the page shows a frame, s:
            ``play.margin_frames`` frame periods with ``play.swap_sent``; 0: none goes again.
        delay (Callable[[], float] | None): how long a frame queued now takes to reach the page, s
            (:meth:`BrowserSocket.delivery_delay` of the player's socket); ``None``: no time.
    """

    def __init__(
        self,
        period: float,
        send: Callable[[int, Mapping[int, bytes], bool, bool], None],
        lead: float,
        swap: float = 0.0,
        delay: Callable[[], float] | None = None,
    ) -> None:
        self.period, self.send = float(period), send
        self.lead, self.swap = float(lead), float(swap)
        self.delay = delay or (lambda: 0.0)
        self._lock = threading.Lock()  # the client's thread asks what goes when
        # every frame put -> (when it goes, when the display shows it)
        self._times: dict[int, tuple[float, float]] = {}
        #: Per newer version of a frame that went (``swap``): its index and when it went.
        self.resent: deque[tuple[int, float]] = deque(maxlen=TRACE_LENGTH)
        self._frames: dict[int, _Queued] = {}
        #: Per frame sent: its index, whether it is its block's first, when it was put, when it was
        #: to go, when the display shows it, when it went, how many refreshes replaced it.
        self.log: deque[tuple[int, bool, float, float, float, float, int]] = deque(
            maxlen=TRACE_LENGTH
        )
        self._early: dict[int, Mapping[int, bytes]] = {}  # a refresh's frames not yet put
        self._shown: float | None = None  # when the display shows the newest frame put
        self._newest = -1  # the newest frame put
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._send())

    def put(self, k: int, messages: Mapping[int, bytes], first: bool, ready: float) -> float:
        """Queue frame ``k`` of a block's first version, ready to go at ``ready`` (a refresh's
        frame that came first takes its place); returns when the display shows it."""
        with self._lock:
            due = ready if self._shown is None else self._shown + self.period
            if first:
                due = min(due, ready + BACKLOG_FRAMES * self.period)
            shown = self._shown = max(ready, due)
            for later in [j for j, entry in self._frames.items() if entry.shown >= shown]:
                del self._frames[later], self._times[later]
            replaced = int(k) in self._early
            messages = self._early.pop(int(k), messages)
            go = max(ready, shown - self.lead)
            self._frames[int(k)] = _Queued(go, messages, first, shown, ready, int(replaced))
            self._times = {j: times for j, times in self._times.items() if times[1] > ready - 1.0}
            self._times[int(k)] = go, shown
            self._newest = int(k)
        self._wake.set()
        return shown

    def replace(self, k: int, messages: Mapping[int, bytes]) -> bool:
        """Frame ``k`` of a refresh in the place of the one queued, or of the one to come, unless
        that one went: with ``swap`` it then goes again at once if it would reach the page before
        the page shows the frame; returns whether it took its place or went."""
        k, now = int(k), time.monotonic()
        late = self._late()
        with self._lock:
            entry = self._frames.get(k)
            if entry is not None:
                entry.messages = messages
                entry.replaced += 1
                return True
            if k > self._newest:
                self._early[k] = messages
                return True
            times = self._times.get(k)
            again = times is not None and now < self._until(*times, late)
        if again:
            self.resent.append((k, now))
            self.send(k, messages, False, True)
        return again

    def replaceable(self, at: float) -> int:
        """The first frame whose newer version would still take its place at ``at``
        (``time.monotonic()``): one that will not have gone by then, or with ``swap`` that would
        reach the page before the page shows it; the queued frames go at their times, the frames
        after the newest put a frame period apart (any thread)."""
        late = self._late()
        with self._lock:
            for k in sorted(self._times):
                if self._until(*self._times[k], late) > at:
                    return k
            if self._shown is None:
                return self._newest + 1
            ahead = (at - self._until(-math.inf, self._shown, late)) / self.period
            return self._newest + max(math.floor(ahead) + 1, 1)

    def _late(self) -> float:
        """How much earlier than the page shows it a newer version of a frame must go, s."""
        return SWAP_GUARD_FRAMES * self.period + self.delay() if self.swap else 0.0

    def _until(self, go: float, shown: float, late: float) -> float:
        """Until when a newer version of a frame that goes at ``go`` and is shown at ``shown``
        takes its place: until it goes, or with ``swap`` while it would reach the page in time."""
        if not self.swap:
            return go if go > -math.inf else shown - self.lead
        return max(go, shown - self.lead + self.swap - late)

    def rebuffer(self) -> None:
        """The display starts over (a browser reconnected): the next frame is shown when ready."""
        with self._lock:
            self._shown = None

    def close(self) -> None:
        """Stop sending; the frames that wait are dropped."""
        self._task.cancel()

    async def _send(self) -> None:
        while True:
            with self._lock:
                k = min(self._frames, default=None)
                when = None if k is None else self._frames[k].go
            if k is None or when > time.monotonic():
                self._wake.clear()
                timeout = None if when is None else when - time.monotonic()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout)
                except asyncio.TimeoutError:
                    pass
                continue
            with self._lock:
                frame = self._frames.pop(k)
                self._early = {j: m for j, m in self._early.items() if j > k}
            went = time.monotonic()
            self.log.append(
                (k, frame.first, frame.ready, frame.go, frame.shown, went, frame.replaced)
            )
            self.send(k, frame.messages, frame.first, False)


@dataclass
class Session:
    """One player's part in a room's session, on this worker: the seat, its browser, its
    inputs."""

    room: str
    round_start: RoundStart
    seat: Seat
    token: str
    t0: float  # server clock at room time 0
    inputs: InputBuffer
    pacing: Pacing
    roster: dict[int, dict] = field(default_factory=dict)
    known: set[int] = field(default_factory=set)  # the other players the roster has shown
    markers: bool = False  # the player's page draws the field's markers
    arrivals: dict[int, float] = field(default_factory=dict)  # input sample -> when it arrived
    hidden_since: float | None = None  # when the player's page was hidden, if it is
    last_input: float = field(default_factory=time.monotonic)  # when a key or the mouse last moved
    player: BrowserSocket | None = None
    watchers: set[BrowserSocket] = field(default_factory=set)
    ready: asyncio.Event = field(default_factory=asyncio.Event)  # the client has started
    connected: asyncio.Event = field(default_factory=asyncio.Event)
    early: list[ClientMessage] = field(default_factory=list)  # received while the client starts
    task: asyncio.Task | None = None  # starts the client, then plays
    grace: asyncio.Task | None = None  # ends the session if the player's browser stays away
    relay: asyncio.Task | None = None  # sends the client's messages to the coordinator
    last: list[ClientMessage] | None = None  # the client's messages from its stop
    # play.refresh: the frames waiting for their time; an input came, a block's step was asked for
    # or its frames were all taken; the first frame of the newest block stepped, the frame after
    # the last of the blocks taken to the end; what refreshes took from their cut to their latents
    # ready, and from those to their first frame
    sender: SendQueue | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    newest: int | None = None
    completed: int = 0
    refresh_end_s: deque[float] = field(default_factory=lambda: deque(maxlen=LEAD_WINDOW))
    decode_s: deque[float] = field(default_factory=lambda: deque(maxlen=LEAD_WINDOW))
    latents_out_s: deque[float] = field(default_factory=lambda: deque(maxlen=LEAD_WINDOW))
    #: What a refresh's latent frame takes from one to the next as its frames come (decoded and
    #: encoded), s: the median of the newest; 0 before any.
    latent_out_s: float = 0.0
    block_s: float = 0.0  # the newest block's cut to its first frame
    #: The refreshes asked for whose latents are not generated yet, (block's first frame, version).
    generating: set[tuple[int, int]] = field(default_factory=set)
    refreshing: set[asyncio.Task] = field(default_factory=set)  # refreshes whose frames come
    #: What the blocks and refreshes did, for the stalls of ``tools/bench_live.py``.
    trace: deque[BlockTrace | RefreshTrace] = field(
        default_factory=lambda: deque(maxlen=TRACE_LENGTH)
    )
    due: float | None = None  # when the newest block was due to start
    #: The refreshes asked for of the newest blocks: (block's first frame, version) -> its trace
    #: entry (the n-th refresh of a block is its version n).
    versions: dict[tuple[int, int], RefreshTrace] = field(default_factory=dict)

    @property
    def others(self) -> set[int]:
        """The player slots of the room's other players."""
        return {slot for slot in self.roster if slot != self.seat.slot}

    def sockets(self) -> list[BrowserSocket]:
        """The socket of the player's browser, while it is connected, and the spectators'."""
        return [sock for sock in (self.player, *self.watchers) if sock is not None]

    def broadcast(self, message: dict) -> None:
        """Queue a text message for the player's browser and the spectators'."""
        for sock in self.sockets():
            sock.send(message)

    def qualities(self) -> tuple[int, ...]:
        """The JPEG qualities to encode the next block's frames at: the player's socket's, and
        the lowest of the spectators' sockets'."""
        wanted = [] if self.player is None else [self.player.quality]
        if self.watchers:
            wanted.append(min(sock.quality for sock in self.watchers))
        return tuple(dict.fromkeys(wanted))

    def send_frame(
        self, k: int, messages: Mapping[int, bytes], first: bool, again: bool = False
    ) -> None:
        """Queue frame ``k`` for the browsers (:meth:`BrowserSocket.send_frame`); ``again``: a
        newer version of a frame that went (``play.swap_sent``), its header marked ``r``."""
        if not messages:  # none when nobody watched as the block began
            return
        if again:
            messages = {quality: _marked_again(m) for quality, m in messages.items()}
        for sock in self.sockets():
            sock.send_frame(k, messages, first)

    def send_delay(self) -> float:
        """How long a frame queued now takes to reach the player's page, s (0 without one)."""
        return 0.0 if self.player is None else self.player.delivery_delay()

    def set_hidden(self, hidden: bool) -> None:
        """The page was hidden or shown (it says so with every new socket too): a page hidden
        already stays hidden since it first was; a page shown again starts the idle time over, as
        a key does."""
        now = time.monotonic()
        if hidden:
            self.hidden_since = now if self.hidden_since is None else self.hidden_since
        else:
            self.hidden_since, self.last_input = None, now

    @property
    def refresh_span(self) -> float:
        """What a refresh takes from its cut to its latents ready, s (the generation is busy with
        it until then): :data:`~worldcast.demo.pacing.LEAD_PERCENTILE` of the newest
        :data:`~worldcast.demo.pacing.LEAD_WINDOW`, as the pacing's lead counts a block's; before
        the first refresh, what the newest block took from its cut to its first frame (its
        latents and a latent frame's decode; the first blocks of a session, slower, are not kept
        to stand for a refresh)."""
        return percentile(self.refresh_end_s) if self.refresh_end_s else self.block_s

    @property
    def decode_lead(self) -> float:
        """Likewise, what a refresh takes from its latents ready to its first frame ready, s: the
        decoder's latent frame on hand (a newer version takes the decoder over at the next latent
        frame), its first and the frame's encoding; 0 before the first refresh."""
        return percentile(self.decode_s)

    @property
    def refresh_lead(self) -> float:
        """What a refresh takes from its cut to its first frame ready, as a refresh cuts its
        controls from the first frame then unsent (:meth:`Worker._refresh`), s: the median of
        each part; one that comes later shows the input from a later frame on, one that comes
        sooner would show it from a later frame than it could (the newest
        :data:`~worldcast.demo.pacing.LEAD_WINDOW`; before the first refresh, :attr:`block_s`)."""
        if not self.refresh_end_s:
            return self.block_s
        return percentile(self.refresh_end_s, 50) + percentile(self.decode_s, 50)

    def add_input(self, data: dict, *, room_time: float) -> None:
        """Queue a browser input sample for the next block (or a refresh of the block on
        screen)."""
        seq = int(data["q"])
        self.arrivals[seq] = time.monotonic()
        if data.get("b") or data.get("dp") or data.get("dy"):
            self.last_input = self.arrivals[seq]
        if len(self.arrivals) > ARRIVALS_KEPT:
            for old in sorted(self.arrivals)[: len(self.arrivals) - ARRIVALS_KEPT]:
                del self.arrivals[old]
        self.inputs.add(
            InputSample(
                seq=seq,
                arrival=room_time,
                buttons=int(data.get("b", 0)),
                dpitch=float(data.get("dp", 0.0)),
                dyaw=float(data.get("dy", 0.0)),
                weapon=int(data.get("w", self.inputs.weapon)),
            )
        )
        self.wake.set()

    def frame_message(
        self,
        frame: LiveFrame,
        jpeg: bytes,
        ready: float,
        encoded: float,
        first: bool,
        taken: float,
    ) -> bytes:
        """A frame as a binary message.

        The header's ``tm`` is, in ms, [the frame's newest input sample to the controls taken at
        ``taken``, the controls taken to the frame decoded at ``ready``, its encoding until
        ``encoded``]; ``tn`` and ``cp``, how far the player had turned in the session through the
        frame and the camera's pitch at it, how much of the player's turn the frame holds
        (:meth:`LiveFrame.look_wire`); ``b`` and ``w``, the buttons held and the weapon in the
        frame's row of the controls (:meth:`LiveFrame.controls_wire`: the page's instant turn keeps
        a frame of a moving viewmodel out of its panorama); ``m``, the field's markers, while the
        player's page asks for them; ``f``, whether the frame is its block's first (``first``): at
        it the page drops the older frames of the blocks before it and keeps up to
        :data:`~worldcast.demo.pacing.BACKLOG_FRAMES` (``Playout.add`` in ``static/stream.js``);
        ``pv``, whether it is of a refresh's preview (``play.preview_step``), and ``v``, the
        version of its block it is of (the page keeps the newest of a frame it gets again); a
        frame sent again has ``r`` too (:meth:`send_frame`).
        """
        arrival = self.arrivals.get(frame.input_seq)
        header = {
            "k": frame.index,
            "f": first,
            "pv": frame.preview,
            "v": frame.version,
            "q": frame.input_seq,
            "tm": [
                None if arrival is None else round(1000 * (taken - arrival), 1),
                round(1000 * (ready - taken), 1),
                round(1000 * (encoded - ready), 1),
            ],
            "s": frame.state.wire(),
            "p": [other.wire() for other in frame.others],
            "st": frame.status,
            **frame.look_wire(),
            **frame.controls_wire(),
        }
        if self.markers:
            header["m"] = frame.marker_wire()
        return pack(header, jpeg)


class Worker:
    """One GPU's server: one client, one player at a time (see the module docstring)."""

    def __init__(self, config: DemoConfig, client: ClientProtocol, library: Library) -> None:
        self.cfg, self.client, self.library = config, client, library
        self.id = config.worker.worker_id or (
            f"{socket.gethostname().split('.')[0]}:{config.worker.port}"
        )
        self.session: Session | None = None
        self.coordinator: web.WebSocketResponse | None = None
        self.http: ClientSession | None = None
        self.clock_offset = 0.0  # server clock - local monotonic clock
        self.client_thread = ThreadPoolExecutor(1, thread_name_prefix="client")
        # a client that decodes on its own device has its frames taken on a thread of their own,
        # so that the next block is requested meanwhile
        self.frame_thread = (
            self.client_thread
            if client.blocks_in_flight == 1
            else ThreadPoolExecutor(1, thread_name_prefix="frames")
        )
        self._best_rtt = float("inf")
        self._link: asyncio.Task | None = None
        self._queued = asyncio.Event()  # the client queued a message for the room
        self._relay: asyncio.Task | None = None  # the newest session's

    def room_time(self, session: Session) -> float:
        """The session's room time now, s."""
        return time.monotonic() + self.clock_offset - session.t0

    def app(self) -> web.Application:
        """The worker's web application."""
        app = web.Application()
        app.add_routes(
            [
                web.get("/ws/play", self.play_ws),
                web.get("/ws/watch", self.watch_ws),
                web.get("/health", self.health),
            ]
        )
        app.on_startup.append(self._startup)
        app.on_shutdown.append(self._shutdown)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _startup(self, app: web.Application) -> None:
        self.http = ClientSession()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.client_thread, self.client.warm_up)
        await loop.run_in_executor(self.client_thread, load_jpeg_encoder)
        self.client.notify = lambda: loop.call_soon_threadsafe(self._queued.set)
        self._link = asyncio.create_task(self._coordinator_link())

    async def _shutdown(self, app: web.Application) -> None:
        await self._end_session("the worker shut down")

    async def _cleanup(self, app: web.Application) -> None:
        self._link.cancel()
        self.client.notify = None
        await self.http.close()

    async def health(self, request: web.Request) -> web.Response:
        """``GET /health``: the worker's id and client, and the room and player slot it serves."""
        s = self.session
        return web.json_response(
            {
                "id": self.id,
                "client": self.cfg.client.kind,
                "busy": s is not None,
                "room": s.room if s else None,
                "slot": s.seat.slot if s else None,
            }
        )

    # ----------------------------------------------------------------------------- coordinator link
    async def _coordinator_link(self) -> None:
        """Stay connected to the coordinator: reconnect when the link breaks, for as long as the
        worker lives."""
        url = self.cfg.worker.coordinator_url.rstrip("/") + "/ws/worker"
        hello = {
            "t": "hello",
            "id": self.id,
            "port": self.cfg.worker.port,
            "advertise_url": self.cfg.worker.advertise_url,
            "client": self.cfg.client.kind,
        }
        delay = RECONNECT_S
        while True:
            try:
                async with self.http.ws_connect(
                    url, heartbeat=HEARTBEAT_S, max_msg_size=LINK_MESSAGE_BYTES
                ) as ws:
                    self.coordinator, delay = ws, RECONNECT_S
                    self._best_rtt = float("inf")
                    await ws.send_json(hello)
                    log.info("worker %s connected to %s", self.id, url)
                    clock = asyncio.create_task(self._clock_pings(ws))
                    try:
                        async for msg in ws:
                            await self._from_coordinator(msg)
                    finally:
                        clock.cancel()
            except (OSError, asyncio.TimeoutError, ClientError) as error:
                log.warning(
                    "coordinator %s unreachable (%s); retrying in %.1f s", url, error, delay
                )
            except Exception:  # the link is the worker's lifeline: nothing may end it
                log.exception("the link to coordinator %s failed; retrying in %.1f s", url, delay)
            self.coordinator = None
            await self._end_session("the coordinator went away")
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_S)

    async def _clock_pings(self, ws: WebSocket) -> None:
        for i in itertools.count():
            await send(ws, {"t": "clock", "c": time.monotonic()})
            await asyncio.sleep(CLOCK_BURST_S if i < CLOCK_BURST else CLOCK_PERIOD_S)

    async def _from_coordinator(self, msg: WSMessage) -> None:
        """Handle one WebSocket message of the coordinator; a message that fails is logged."""
        try:
            if msg.type == WSMsgType.TEXT:
                await self._control_in(msg.json())
            elif msg.type == WSMsgType.BINARY:
                self._message_in(msg.data)
        except Exception:  # a bad message must not end the link
            log.exception("worker %s: a message of the coordinator failed", self.id)

    async def _control_in(self, msg: dict) -> None:
        kind = msg.get("t")
        if kind == "clock":
            now = time.monotonic()
            rtt = now - float(msg["c"])
            if rtt <= self._best_rtt * 1.5:  # a ping about as fast as the fastest so far
                self._best_rtt = min(self._best_rtt, rtt)
                self.clock_offset = float(msg["s"]) - (float(msg["c"]) + now) / 2
        elif kind == "assign":
            await self._assign(msg)
        s = self.session
        if s is None or msg.get("room") != s.room:
            return
        if kind == "roster":
            s.roster = {int(p["slot"]): p for p in msg["roster"]}
            s.broadcast({"t": "roster", "roster": msg["roster"]})
            if s.ready.is_set():
                self._follow_roster(s)
        elif kind == "release" and int(msg["slot"]) == s.seat.slot:
            await self._end_session("the coordinator released the seat")

    def _follow_roster(self, s: Session) -> None:
        """Tell the client of the players who left since it was told last (a player who took a
        seat tells it itself, by its first state). A change the client fails on is logged and tried
        again at the next roster; the others still reach it."""
        for slot in sorted(s.known - s.others):
            if self._tell(s, self.client.remove_player, slot):
                s.known.discard(slot)
        s.known |= s.others

    @staticmethod
    def _tell(s: Session, change: Callable[[int], None], slot: int) -> bool:
        """Whether the client took in a roster change of ``slot``; a failure is logged."""
        try:
            change(slot)
        except Exception:  # one seat's failure must not end the session or stall the others
            log.exception("room %s: the client's %s(%d) failed", s.room, change.__name__, slot)
            return False
        return True

    def _message_in(self, data: bytes) -> None:
        """A message of another client of the room, for this worker's client."""
        header, payload = unpack(data)
        s = self.session
        if s is None or header.get("room") != s.room or int(header["slot"]) == s.seat.slot:
            return
        message = ClientMessage(
            int(header["slot"]), str(header["kind"]), header.get("meta", {}), payload
        )
        if s.ready.is_set():
            self.client.receive_message(message)
        else:  # the client is still starting: the kept live states and memory entries, and more
            s.early.append(message)

    async def _relay_messages(self, s: Session, previous: asyncio.Task | None) -> None:
        """Send the client's messages of session ``s`` to the coordinator as soon as it queues
        them, in order, after those of the ``previous`` session; once the client has stopped, its
        last ones (:meth:`_end_session`)."""
        if previous is not None:
            await asyncio.wait({previous})
        while s.last is None:
            self._queued.clear()
            for message in self.client.take_messages():
                await self._message_out(s, message)
            if s.last is None:
                await self._queued.wait()
        for message in s.last:
            await self._message_out(s, message)

    async def _message_out(self, s: Session, message: ClientMessage) -> None:
        """A message of this worker's client, for the room's other clients."""
        header = {"room": s.room, "slot": s.seat.slot, "kind": message.kind, "meta": message.meta}
        await send(self.coordinator, pack(header, message.payload))

    # ------------------------------------------------------------------------------------- sessions
    async def _assign(self, msg: dict) -> None:
        reply = {"t": "assigned", "room": msg["room"]}
        try:
            if self.session is not None:
                raise RuntimeError("this worker is already playing")
            round_start = self.library.get(msg["round_start"])
            seat = round_start.seat(int(msg["slot"]))
        except (KeyError, RuntimeError) as error:
            await send(self.coordinator, {**reply, "error": error.args[0]})
            return
        s = Session(
            room=msg["room"],
            round_start=round_start,
            seat=seat,
            token=msg["token"],
            t0=float(msg["t0"]),
            inputs=InputBuffer(
                fps=self.cfg.play.fps,
                mapping=self.cfg.play.mapping,
                weapon=encode_weapon(seat.loadout[0]) if seat.loadout else 0,
                release_after_s=self.cfg.play.release_after_s,
                pitch_gain=self.cfg.play.gains[0],
                yaw_gain=self.cfg.play.gains[1],
                gather_s=self.cfg.play.gather_ms / 1000.0,
            ),
            pacing=Pacing(self.cfg.play.fps),
            roster={int(p["slot"]): p for p in msg["roster"]},
        )
        self.session = s
        s.relay = self._relay = asyncio.create_task(self._relay_messages(s, self._relay))
        await send(self.coordinator, {**reply, "error": None})
        s.task = asyncio.create_task(self._run(s))
        self._await_player(s, FIRST_CONNECT_S, "the player did not connect")

    async def _run(self, s: Session) -> None:
        """Start the client on the seat, hand it the messages that came meanwhile, then tell it
        who left meanwhile, then play; a failure ends the session, not the worker.

        The messages go first, as they came: a player who left during the start sent its last
        ones before the roster that dropped it, and its removal must come after them.
        """
        s.known = s.others
        try:
            await asyncio.get_running_loop().run_in_executor(
                self.client_thread, self.client.start, s.round_start, s.seat
            )
            for message in s.early:
                self.client.receive_message(message)
            self._follow_roster(s)
            s.ready.set()
            await self._play(s)
        except Exception as error:  # reported to the room, the worker stays up
            log.exception("session failed")
            if self.session is s:
                await self._end_session(f"the client failed: {error}")

    def _await_player(self, s: Session, seconds: float, reason: str) -> None:
        """End the session with ``reason`` unless the player's browser connects within
        ``seconds``."""

        async def grace() -> None:
            await asyncio.sleep(seconds)
            if self.session is s:
                await self._end_session(reason)

        s.grace = asyncio.create_task(grace())

    async def _end_session(self, reason: str) -> None:
        """End the session: tell its browsers, stop the client, send its last messages (the
        WorldCast client's ``done``) after all the others, then ``left``, which frees the seat
        (the coordinator relays nothing of a player who left)."""
        s, self.session = self.session, None
        if s is None:
            return
        log.info("room %s: the session of player %d ended: %s", s.room, s.seat.slot, reason)
        for task in (s.task, s.grace):
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        ended = {"t": "ended", "message": reason}
        await asyncio.gather(*(sock.close(ended) for sock in s.sockets()))
        # after the blocks the client is generating, and before the next session's start
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.frame_thread, bool)
        s.last = await loop.run_in_executor(self.client_thread, self._stop_client)
        self._queued.set()
        await asyncio.wait({s.relay})  # a relay that failed still frees the seat
        if (error := s.relay.exception()) is not None:
            log.error("room %s: player %d's relay failed", s.room, s.seat.slot, exc_info=error)
        await send(self.coordinator, {"t": "left", "room": s.room, "reason": reason})

    def _stop_client(self) -> list[ClientMessage]:
        """Stop the client (on its thread) and take the messages it queued last."""
        try:
            self.client.stop()
        except Exception:  # the session's own failure, raised again when it closes
            log.exception("the client's session closed with an error")
        return self.client.take_messages()

    # ------------------------------------------------------------------------------ browser sockets
    async def play_ws(self, request: web.Request) -> web.WebSocketResponse:
        """``/ws/play?token=``: the player's socket, input samples (and whether the page draws the
        field's markers) in, frames out; a new connection replaces the old one, whose messages
        count no more from then on (the new one's welcome counted the turn up to there), and a
        lost one starts the reconnect grace."""
        s = self.session
        if s is None or request.query.get("token") != s.token:
            raise web.HTTPForbidden(text="no seat on this worker for that token")
        ws = web.WebSocketResponse(heartbeat=HEARTBEAT_S)
        await ws.prepare(request)
        sock = self._browser_socket(ws, player=True)
        old, s.player = s.player, sock
        s.grace.cancel()
        s.connected.set()
        sock.send(self._welcome(s))
        if old is not None:
            await old.close(reason=b"replaced by a new connection")
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                if s.player is not sock:
                    continue
                data = msg.json()
                kind = data.get("t")
                if kind == "in":
                    s.add_input(data, room_time=self.room_time(s))
                elif kind == "markers":
                    s.markers = bool(data.get("on"))
                elif kind == "vis":
                    s.set_hidden(bool(data.get("h")))
                elif kind == "bye":
                    await self._end_session("the player left")
                    break
                else:
                    self._from_browser(sock, data)
        finally:
            sock.cancel()
            if s.player is sock:
                s.player = None
                s.connected.clear()
                if self.session is s:
                    self._await_player(
                        s, self.cfg.worker.reconnect_grace_s, "the player's browser went away"
                    )
        return ws

    async def watch_ws(self, request: web.Request) -> web.WebSocketResponse:
        """``/ws/watch?room=``: a spectator's socket, the player's frames out."""
        s = self.session
        if s is None or request.query.get("room") != s.room:
            raise web.HTTPNotFound(text="this worker is not playing in that room")
        ws = web.WebSocketResponse(heartbeat=HEARTBEAT_S)
        await ws.prepare(request)
        sock = self._browser_socket(ws, player=False)
        s.watchers.add(sock)
        sock.send(self._welcome(s))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    self._from_browser(sock, msg.json())
        finally:
            sock.cancel()
            s.watchers.discard(sock)
        return ws

    def _browser_socket(self, ws: web.WebSocketResponse, *, player: bool) -> BrowserSocket:
        """A browser's socket: the player's frames start at ``worker.jpeg_quality``, a
        spectator's a step lower."""
        best = self.cfg.worker.jpeg_quality
        qualities = (best, *(q for q in JPEG_QUALITY_STEPS if q < best))
        return BrowserSocket(
            ws,
            qualities if player else qualities[1:] or qualities,
            self.cfg.worker.max_inflight_ms / 1000.0,
            1.0 / self.cfg.play.fps,
            self.client.frames_per_block,
            player=player,
        )

    @staticmethod
    def _from_browser(sock: BrowserSocket, data: dict) -> None:
        """A browser's acknowledgement of a frame, or its clock ping."""
        kind = data.get("t")
        if kind == "ack":
            sock.ack(int(data["r"]))
        elif kind == "ping":
            sock.send({"t": "pong", "c": data["c"]})

    def _welcome(self, s: Session) -> dict:
        """A browser socket's first message. ``tn`` is the turn of the input samples the session
        received before the socket opened: the page adds the turn it sends to it, and a frame
        header's ``tn`` counts the same samples."""
        return {
            "t": "welcome",
            "seat": s.seat.public(),
            "roster": list(s.roster.values()),
            "weapon_ids": {name: encode_weapon(name) for name in s.seat.loadout},
            "inspect_weapons": sorted(INSPECT_WEAPONS),
            "buttons": CONTROL_BUTTONS,
            "jitter_buffer_frames": self.cfg.play.jitter_buffer_frames,
            "margin_frames": self.cfg.play.margin_frames,
            "backlog_frames": BACKLOG_FRAMES,
            "mouse_degrees_per_pixel": self.cfg.play.mouse_degrees_per_pixel,
            "fps": self.cfg.play.fps,
            "frames_per_block": self.client.frames_per_block,
            "tn": s.inputs.received.tolist(),
        }

    # ------------------------------------------------------------------------------------- the loop
    def _away(self, s: Session) -> bool:
        """Whether the player is gone from a seat that would otherwise stay for ever: the page
        hidden for ``play.away_s``, or no key and no mouse movement for ``play.idle_s``."""
        now = time.monotonic()
        hidden = s.hidden_since is not None and now - s.hidden_since >= self.cfg.play.away_s
        return hidden or now - s.last_input >= self.cfg.play.idle_s

    async def _play(self, s: Session) -> None:
        """Step the client block after block while the player is connected; the session has no
        end of its own (:meth:`_away` releases a seat that nobody plays).

        A client with ``blocks_in_flight`` 2 has the next block start while the frames of the one
        before are still being decoded; never a third, so that at most one block waits for the
        decoder: a decode slower than the display holds the blocks back, and the display plays at
        the rate the frames come. With ``play.refresh`` the session plays as
        :meth:`_play_refreshing` says.
        """
        await s.connected.wait()
        if self.cfg.play.refresh:
            await self._play_refreshing(s)
            return
        flight: deque[asyncio.Task] = deque()
        away = False
        try:
            while self.session is s and not away:
                if not s.connected.is_set():
                    await s.connected.wait()
                    s.pacing.rebuffer()
                ahead = len(flight) * self.client.frames_per_block * s.pacing.period
                start = s.pacing.next_start(ahead)
                if start is not None:
                    await asyncio.sleep(max(0.0, start - time.monotonic()))
                if away := self._away(s):
                    continue
                flight.append(asyncio.create_task(self._block(s)))
                while len(flight) >= self.client.blocks_in_flight:
                    await flight.popleft()
        finally:
            for task in flight:
                task.cancel()
        if away:
            await self._end_session("you were away")

    async def _block(self, s: Session) -> None:
        """One block: its frames to the browsers as they are decoded, then its place in the
        pacing."""
        taken = 0.0

        def controls() -> LiveControls:  # the client calls it right before the block's denoising
            nonlocal taken
            taken = time.monotonic()
            return s.inputs.cut(now=self.room_time(s), frames=self.client.frames_per_block)

        sent = []
        frames_thread = None if self.frame_thread is self.client_thread else self.frame_thread
        frames = block_frames(
            self.client, self.client_thread, controls, s.qualities(), frames_thread
        )
        async for frame, jpegs, ready, encoded in frames:
            first = not sent
            messages = {
                q: s.frame_message(frame, j, ready, encoded, first, taken) for q, j in jpegs.items()
            }
            s.send_frame(frame.index, messages, first=first)
            sent.append(time.monotonic())
        if not sent:
            raise RuntimeError("the client yielded no frame")
        s.pacing.record(taken, sent)

    async def _play_refreshing(self, s: Session) -> None:
        """:meth:`_play` with ``play.refresh``: the frames go through a :class:`SendQueue`, and the
        newest block is refreshed when the input changes (:meth:`_refresh_until`) from as soon as
        its controls are cut, while its first denoising still runs, until the next block's step,
        which commits it. The pacing counts a block's critical delay from the start of its step.
        """
        play = self.cfg.play
        swap = play.margin_frames * s.pacing.period if play.swap_sent else 0.0
        lead = play.send_lead_ms / 1000.0
        s.sender = SendQueue(s.pacing.period, s.send_frame, lead, swap, s.send_delay)
        flight: deque[asyncio.Task] = deque()
        away = False
        loop = asyncio.get_running_loop()
        self.client.card_freed = lambda: loop.call_soon_threadsafe(s.wake.set)
        try:
            while self.session is s and not away:
                if not s.connected.is_set():
                    await s.connected.wait()
                    s.pacing.rebuffer()
                    s.sender.rebuffer()
                await self._refresh_until(s, flight)
                if away := self._away(s):
                    continue
                while len(flight) >= self.client.blocks_in_flight or (flight and flight[0].done()):
                    await flight.popleft()
                # the newest block's first frame: the frames of the blocks taken to the end, then
                # a block's frames for each one still in flight (none is a session's first); no
                # block is refreshed from here until the new one's controls are cut: a refresh of
                # the block before would hold its step back
                first = s.completed + len(flight) * self.client.frames_per_block
                s.newest = None
                flight.append(asyncio.create_task(self._refreshed_block(s, first, s.due)))
        finally:
            self.client.card_freed = None
            for task in (*flight, *s.refreshing):
                task.cancel()
            s.sender.close()
        if away:
            await self._end_session("you were away")

    async def _refreshed_block(self, s: Session, first: int, due: float | None) -> None:
        """One block with ``play.refresh`` (its first frame ``first``, its start due at ``due``):
        its step asked for, the newest block refreshed from its controls' cut on (an input cut into
        a refresh before it would be the cut's anyway), its frames to the send queue as they are
        decoded, then its place in the pacing, its critical delay counted from the start of its
        step. A refresh that takes the block's decode over has its frames come here: in the places
        of those put, or put in their turn (:meth:`ClientProtocol.step`)."""
        loop = asyncio.get_running_loop()
        taken, started = 0.0, time.monotonic()

        def cut() -> None:
            s.newest = first
            s.wake.set()

        def controls() -> LiveControls:  # the client calls it right before the block's denoising
            nonlocal taken
            taken = time.monotonic()
            taken_controls = s.inputs.cut(
                now=self.room_time(s), frames=self.client.frames_per_block
            )
            loop.call_soon_threadsafe(cut)
            return taken_controls

        # asked for at once (it only queues the block), not behind the frames a refresh decodes
        requested = self.client.step(controls)
        frames = block_frames(
            self.client,
            self.client_thread,
            controls,
            s.qualities(),
            self.frame_thread,
            request=lambda: requested,
        )
        sent: list[float] = []
        start = None  # the block's first frame, as it came (a session's first: frame 0 too)
        async for frame, jpegs, ready, encoded in frames:
            k = frame.index
            start = k if start is None else start
            entry = s.versions.get((first, frame.version)) if frame.version else None
            cut_at = taken if entry is None else entry.cut_t
            messages = {
                q: s.frame_message(frame, j, ready, encoded, k == start, cut_at)
                for q, j in jpegs.items()
            }
            if k < start + len(sent):  # a newer version of a frame put
                replaced = s.sender.replace(k, messages)
            else:
                s.sender.put(k, messages, k == start, time.monotonic())
                sent.append(time.monotonic())
                replaced = False
            if entry is not None:
                _refresh_frame(s, entry, k, replaced, frame.preview)
        if start is None:
            raise RuntimeError("the client yielded no frame")
        s.block_s = sent[0] - taken
        s.completed = start + len(sent)
        s.trace.append(BlockTrace(start, due, started, taken, sent[0], sent[-1]))
        s.pacing.record(started, sent)
        s.wake.set()

    async def _refresh_until(self, s: Session, flight: deque[asyncio.Task]) -> None:
        """Refresh the newest block until the next block is due to start
        (:meth:`~worldcast.demo.pacing.Pacing.next_start` after the blocks in ``flight``), when its
        step commits it; with no start yet (the display holds no frame), until a block may start.

        A refresh starts when the input changed since the block's last cut
        (:meth:`InputBuffer.changed`, ``play.refresh_threshold_deg``), the block's controls have
        been cut, and it has room (:meth:`_room_to_refresh`); one asked for by a turn that starts
        waits a little for more of it (:meth:`InputBuffer.gathering`, ``play.gather_ms``). Its
        frames are taken in the background (:meth:`_refresh`): the next block's step waits for no
        refresh's frames.
        """
        threshold = self.cfg.play.refresh_threshold_deg
        frames, period = self.client.frames_per_block, s.pacing.period
        while True:
            s.wake.clear()
            while flight and flight[0].done():
                await flight.popleft()
            start = s.due = s.pacing.next_start(len(flight) * frames * period)
            now = time.monotonic()
            if start is None and len(flight) < self.client.blocks_in_flight:
                return
            if start is not None and now >= start:
                return
            if start is not None and s.newest is not None and self._room_to_refresh(s, now, start):
                if s.inputs.changed(threshold):
                    gather = s.inputs.gathering(self.room_time(s))
                    if gather <= 0.0:
                        self._refresh(s, start)
                        continue
                    with contextlib.suppress(asyncio.TimeoutError):  # the turn comes on meanwhile
                        await asyncio.wait_for(s.wake.wait(), min(gather, start - now))
                    continue
            try:
                await asyncio.wait_for(s.wake.wait(), None if start is None else start - now)
            except asyncio.TimeoutError:
                return

    def _room_to_refresh(self, s: Session, now: float, start: float) -> bool:
        """Whether a refresh asked for now would have room: none being generated (the newest
        input waits for the one before and goes with the next), or with more cards than one
        (the WorldCast client's ``refresh_devices``) a card free, its latents ready
        (:attr:`Session.refresh_span`) and its first frame decoded (:attr:`Session.decode_lead`)
        before the next block's step starts, and the block still having frames to send by then.

        The decoder is done with refreshes by that start, also on a device of its own: a decode
        that runs on into the next step slows its prepare down on the host (and on the
        generator's device its denoising), and so its first frame."""
        if not self.client.can_refresh:
            return False
        if self.client.refresh_cards > 1:
            if not self.client.refresh_card_free:
                return False
        elif s.generating:
            return False
        ready = now + s.refresh_span + s.decode_lead
        if ready > start:
            return False
        return s.sender.replaceable(ready) < s.newest + self.client.frames_per_block

    def _refresh(self, s: Session, free_by: float) -> None:
        """Ask for a refresh of the newest block at once (it waits for the block's first
        denoising), cut again from the first frame expected unsent when its first frame is ready
        (:attr:`Session.refresh_lead`); its frames are taken in the background, in the places of
        those not yet sent (or not yet decoded).

        Its decode is held to what the decoder has room for before ``free_by``, the next block's
        start (:meth:`_room_to_refresh`): it starts at the first frame that will not have been
        sent, or later, at the latent frame from which it still ends in time; one that has no
        room decodes nothing, and the frames shown stay the version's before it (but those not
        yet decoded, which it gives).
        """
        loop = asyncio.get_running_loop()
        frames, first = self.client.frames_per_block, s.newest
        version = 1 + sum(1 for block, _ in s.versions if block == first)
        entry = RefreshTrace(first, time.monotonic())
        s.versions = {key: e for key, e in s.versions.items() if key[0] >= first - frames}
        s.versions[(first, version)] = entry
        s.trace.append(entry)
        s.generating.add((first, version))

        def controls() -> LiveControls:  # right before the block's denoising again
            entry.cut_t = time.monotonic()
            row = s.sender.replaceable(entry.cut_t + s.refresh_lead) - first
            return s.inputs.recut(now=self.room_time(s), frames=frames, from_row=row)

        def done_generating() -> None:
            if (first, version) in s.generating:
                s.generating.discard((first, version))
                s.refresh_end_s.append(entry.generated_t - entry.cut_t)
            s.wake.set()

        def generated() -> None:  # its latents are ready, on the generation thread
            entry.generated_t = time.monotonic()
            loop.call_soon_threadsafe(done_generating)

        def switch(seconds: float) -> int:  # the decoder takes its latents
            now = entry.switch_t = time.monotonic()
            frame = s.sender.replaceable(now + seconds)
            latent = max(seconds, s.latent_out_s)  # the frames' encoding too
            if latent > 0:
                room = math.floor((free_by - now) / latent) * VIDEO_FRAMES_PER_LATENT
                frame = max(frame, first + frames - max(room, 0))
            return frame

        def taken(task: asyncio.Task) -> None:  # the frames it did not replace stay the block's
            s.refreshing.discard(task)
            if not task.cancelled() and (error := task.exception()) is not None:
                log.error(
                    "room %s: player %d's refresh of block %d failed",
                    s.room,
                    s.seat.slot,
                    first,
                    exc_info=error,
                )

        # asks at once, on the loop's thread
        requested = self.client.refresh(controls, switch, generated, self.cfg.play.preview_step)
        task = asyncio.create_task(self._refresh_frames(s, requested, entry, version))
        s.refreshing.add(task)
        task.add_done_callback(taken)

    async def _refresh_frames(
        self, s: Session, requested: Iterator[LiveFrame], entry: RefreshTrace, version: int
    ) -> None:
        """The frames of refresh ``version`` (its trace ``entry``), to the send queue; with more
        cards than one they may be another version's of the block (each frame goes with its
        own's)."""
        first = entry.first
        new = block_frames(
            self.client,
            self.client_thread,
            lambda: None,
            s.qualities(),
            self.frame_thread,
            request=lambda: requested,
        )
        done = False
        try:
            async for frame, jpegs, ready, encoded in new:
                own = s.versions.get((first, frame.version), entry)
                on = frame.index == first
                messages = {
                    q: s.frame_message(frame, j, ready, encoded, on, own.cut_t)
                    for q, j in jpegs.items()
                }
                replaced = s.sender.replace(frame.index, messages)
                _refresh_frame(s, own, frame.index, replaced, frame.preview)
            done = True
        finally:
            if not done:  # it may have ended without its latents (an error, a session's end)
                s.generating.discard((first, version))
            s.wake.set()


def _marked_again(message: bytes) -> bytes:
    """A frame's message with ``r`` (sent again) in its header."""
    header, payload = unpack(message)
    return pack({**header, "r": 1}, payload)


def _refresh_frame(s: Session, entry: RefreshTrace, k: int, replaced: bool, preview: bool) -> None:
    """A frame ``k`` of the refresh of trace ``entry`` came (``replaced``: in the place of one
    queued; ``preview``: of its preview): what it took, in the trace and in the session's
    estimates (of the refresh's own frames)."""
    now = time.monotonic()
    if preview:
        entry.preview_out_t = entry.preview_out_t or now
        return
    if not entry.first_out_t:
        entry.first_out_t = now
        if entry.generated_t:
            s.decode_s.append(now - entry.generated_t)
    entry.last_out_t = now
    entry.frames += 1
    if replaced and entry.replaced < 0:
        entry.replaced = k
    count = entry.frames - 1
    if count and count % VIDEO_FRAMES_PER_LATENT == 0:  # a latent frame's frames later
        per_frame = (now - entry.first_out_t) / count
        s.latents_out_s.append(per_frame * VIDEO_FRAMES_PER_LATENT)
        s.latent_out_s = percentile(s.latents_out_s, 50)


def run(config: DemoConfig) -> None:
    """Serve a worker until interrupted."""
    sys.setswitchinterval(SWITCH_INTERVAL_S)
    library = Library.load(config.library)
    web.run_app(
        Worker(config, make_client(config, library), library).app(),
        host=config.worker.host,
        port=config.worker.port,
        print=None,
        access_log=None,
    )
