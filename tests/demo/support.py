"""Helpers of the demo tests: a block's controls, a scripted browser (its link throttled or not),
waiting for a condition, a local demo of mock workers (their client one that records what reaches
it, or one whose first block is slow), a room and a seat in it, stand-in JPEGs of a real frame's
size, the Chrome that runs the page in the tests that need it (its mouse, keyboard and fingers; any
Chrome, Chromium or headless shell: ``WORLDCAST_CHROME`` names it), a recorded session for the page
to play.

The end-to-end tests wait for what they assert (a frame that shows another player, a kept memory
entry) instead of sleeping: they do not depend on how fast the machine is.
"""

import asyncio
import base64
import json
import os
import queue
import shutil
import socket
import struct
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest
from aiohttp import ClientError, ClientSession, WSMsgType

from tests.engine.inference.support import media_id
from worldcast.data.controls import CONTROL_BUTTONS, quantize_camera_delta
from worldcast.demo.clients import MEMORY_ENTRY, LiveControls, LivePlayerState
from worldcast.demo.config import load_config
from worldcast.demo.coordinator import STATIC
from worldcast.demo.library import Library, synthetic_library
from worldcast.demo.local import LocalDemo
from worldcast.demo.mock import MockClient
from worldcast.demo.wire import unpack

#: Longest wait for a condition, s: far above what any machine needs, so a timeout is a failure.
TIMEOUT_S = 120.0
FORWARD = 1 << CONTROL_BUTTONS.index("forward")
#: Bytes of a 672 x 384 frame of generated video at each JPEG quality the worker steps through.
FRAME_BYTES = {85: 43_000, 75: 34_000, 60: 26_000, 50: 22_000}
_STAMP = struct.Struct(">dB")
MAC_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
#: The environment variable that names the browser the page tests run in: a Chrome, a Chromium or a
#: ``chrome-headless-shell`` (its libraries on ``LD_LIBRARY_PATH``).
CHROME_ENV = "WORLDCAST_CHROME"


def find_chrome() -> str | None:
    """The browser of the page tests: ``$WORLDCAST_CHROME``, else Chrome or Chromium on the path
    (or the Mac's), else none."""
    if path := os.environ.get(CHROME_ENV):
        return path
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if path := shutil.which(name):
            return path
    return str(MAC_CHROME) if MAC_CHROME.exists() else None


CHROME = find_chrome()
needs_chrome = pytest.mark.skipif(CHROME is None, reason="needs Chrome or Chromium to run the page")


def chrome_flags(directory: Path) -> list[str]:
    """The flags every test's headless Chrome runs with, its profile in ``directory``."""
    flags = ["--headless=new", "--no-first-run", f"--user-data-dir={directory / 'profile'}"]
    flags.append("--allow-file-access-from-files")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        flags.append("--no-sandbox")
    return flags


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


#: The keys the tests press: ``code`` -> (``key``, Windows virtual key code, the text it types).
KEYS = {
    "ArrowLeft": ("ArrowLeft", 37, ""),
    "ArrowUp": ("ArrowUp", 38, ""),
    "ArrowRight": ("ArrowRight", 39, ""),
    "ArrowDown": ("ArrowDown", 40, ""),
    "Escape": ("Escape", 27, ""),
    "Digit1": ("1", 49, "1"),
    "Digit2": ("2", 50, "2"),
    "Digit3": ("3", 51, "3"),
    "Digit4": ("4", 52, "4"),
    "KeyM": ("m", 77, "m"),
    "KeyQ": ("q", 81, "q"),
    "KeyW": ("w", 87, "w"),
    "KeyA": ("a", 65, "a"),
    "KeyS": ("s", 83, "s"),
    "KeyD": ("d", 68, "d"),
    "KeyF": ("f", 70, "f"),
    "ShiftLeft": ("Shift", 16, ""),
    "MetaLeft": ("Meta", 91, ""),
}


class ChromePage:
    """A Chrome page driven over DevTools: JavaScript in, the mouse as a person has it, a press
    and, a tenth of a second later, its release (a click made of one event can hide a press and a
    release that land on different elements), the keyboard's presses and releases, and fingers."""

    def __init__(self, ws) -> None:
        self.ws, self.calls = ws, 0
        self.events: list[dict] = []

    async def call(self, method: str, **params) -> dict:
        """The result of a DevTools command; the events that arrive meanwhile are kept in
        :attr:`events`."""
        self.calls += 1
        await self.ws.send_json({"id": self.calls, "method": method, "params": params})
        async for message in self.ws:
            data = json.loads(message.data)
            if data.get("id") == self.calls:
                return data["result"]
            if "method" in data:
                self.events.append(data)
        raise AssertionError("Chrome closed the page")

    async def eval(self, expression: str):
        """The value of a JavaScript expression (a promise is awaited)."""
        result = await self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        assert "exceptionDetails" not in result, result["exceptionDetails"]
        return result["result"].get("value")

    async def wait_for(self, expression: str) -> None:
        """Wait until a JavaScript expression holds; fail after :data:`TIMEOUT_S`."""
        deadline = time.monotonic() + TIMEOUT_S
        while not await self.eval(expression):
            assert time.monotonic() < deadline, f"timed out waiting for {expression}"
            await asyncio.sleep(0.05)

    async def move(self, x: float, y: float) -> None:
        """Move the mouse to ``x, y`` (CSS pixels)."""
        await self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)

    async def click(self, x: float, y: float, hold: float = 0.12, button: str = "left") -> None:
        """Move the mouse to ``x, y`` (CSS pixels), press ``button`` and release it ``hold`` s
        later."""
        point = dict(x=x, y=y, button=button, clickCount=1)
        await self.move(x, y)
        await self.call("Input.dispatchMouseEvent", type="mousePressed", **point)
        await asyncio.sleep(hold)
        await self.call("Input.dispatchMouseEvent", type="mouseReleased", **point)

    async def key(self, code: str, down: bool, modifiers: int = 0) -> None:
        """Press (``down``) or release the key ``code`` (a key of :data:`KEYS`) as a keyboard
        does, with the ``modifiers`` held (1 Alt, 2 Ctrl, 4 Meta, 8 Shift)."""
        key, virtual, text = KEYS[code]
        kind = ("keyDown" if text else "rawKeyDown") if down else "keyUp"
        await self.call(
            "Input.dispatchKeyEvent",
            type=kind,
            code=code,
            key=key,
            windowsVirtualKeyCode=virtual,
            text=text if down else "",
            modifiers=modifiers,
        )

    async def touch(self, kind: str, fingers: dict[int, tuple[float, float]]) -> None:
        """A touch event with the ``fingers``, each at ``x, y`` (CSS pixels) under its own id:
        ``touchStart`` with every finger now on the screen (the new one goes down),
        ``touchMove`` with every finger (those that moved send pointer events), ``touchEnd`` with
        the fingers that lift (none: all of them)."""
        points = [dict(x=x, y=y, id=id) for id, (x, y) in fingers.items()]
        await self.call("Input.dispatchTouchEvent", type=kind, touchPoints=points)

    async def tap(self, x: float, y: float, hold: float = 0.12) -> None:
        """A finger on ``x, y`` for ``hold`` s."""
        await self.touch("touchStart", {0: (x, y)})
        await asyncio.sleep(hold)
        await self.touch("touchEnd", {})


@asynccontextmanager
async def chrome_page(
    page: str | None,
    window: tuple[int, int, int],
    directory: Path,
    touch: bool = False,
    url: str | None = None,
    webgl: bool = True,
    commands: Sequence[tuple[str, dict]] = (),
):
    """A page open in a Chrome with a window of ``(width, height, device pixels per CSS pixel)``
    and software WebGL (``webgl`` false: no GPU, so no WebGL): a harness page (the HTML ``page``,
    which links ``static/``), or ``url`` (a demo's); ``touch``: a phone's, whose pointer is coarse;
    ``commands``: DevTools commands ``(method, params)`` run before the page loads (a user agent, a
    media feature, a script to run first). Yields its :class:`ChromePage`."""
    width, height, scale = window
    if url is None:
        (directory / "static").symlink_to(STATIC)
        file = directory / "harness.html"
        file.write_text(page)
        url = file.as_uri()
    port = free_port()
    gpu = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] if webgl else ["--disable-gpu"]
    chrome_args = [
        *chrome_flags(directory),
        f"--remote-debugging-port={port}",
        "--use-mock-keychain",
        *gpu,
    ]
    chrome = subprocess.Popen(
        [CHROME, *chrome_args, "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        async with ClientSession() as http:
            targets: list = []
            for _ in range(300):
                assert chrome.poll() is None, "Chrome exited"
                try:
                    async with http.get(f"http://127.0.0.1:{port}/json/list") as response:
                        targets = [t for t in await response.json() if t["type"] == "page"]
                except ClientError:
                    pass
                if targets:
                    break
                await asyncio.sleep(0.1)
            assert targets, "Chrome opened no page"
            async with http.ws_connect(targets[0]["webSocketDebuggerUrl"], max_msg_size=0) as ws:
                chrome_page = ChromePage(ws)
                size = dict(width=width, height=height, deviceScaleFactor=scale, mobile=touch)
                await chrome_page.call("Emulation.setDeviceMetricsOverride", **size)
                if touch:
                    await chrome_page.call(
                        "Emulation.setTouchEmulationEnabled", enabled=True, maxTouchPoints=5
                    )
                for method, params in commands:
                    await chrome_page.call(method, **params)
                await chrome_page.call("Page.navigate", url=url)
                yield chrome_page
    finally:
        chrome.kill()
        chrome.wait()


async def run_harness(page: str, window: tuple[int, int, int], directory: Path) -> dict:
    """The report of a harness page that runs alone and leaves it as JSON in the text of ``#out``
    (``running`` until it is done), in a :func:`chrome_page`."""
    async with chrome_page(page, window, directory) as chrome:
        await chrome.wait_for("document.getElementById('out')?.textContent !== 'running'")
        report = json.loads(await chrome.eval("document.getElementById('out').textContent"))
        assert "error" not in report, report["error"]
        return report


def sized_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """A stand-in for :func:`worldcast.demo.wire.encode_jpeg`: incompressible bytes of a real
    frame's size at ``quality`` (:data:`FRAME_BYTES`), stamped with when it was encoded and
    ``quality`` (:func:`stamp`)."""
    head = b"\xff\xd8" + _STAMP.pack(time.monotonic(), quality)
    return head + os.urandom(FRAME_BYTES[quality] - len(head))


def stamp(jpeg: bytes) -> tuple[float, int]:
    """When a :func:`sized_jpeg` was encoded (``time.monotonic()``) and its quality."""
    return _STAMP.unpack_from(jpeg, 2)


def write_video(path: Path, frames: int) -> None:
    """A rendered video of ``frames`` frames of one shade each (672 x 384, 16 fps)."""
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    writer = imageio_ffmpeg.write_frames(str(path), (672, 384))
    writer.send(None)
    for i in range(frames):
        writer.send(np.full((384, 672, 3), 5 * i, np.uint8).tobytes())
    writer.close()


def write_labels(
    root: Path,
    slot: int,
    visible_at: dict[int, tuple[int, ...]],
    unknown_at: dict | None = None,
    behind_at: dict | None = None,
) -> None:
    """The GT visibility labels of the recording of ``slot`` of the synthetic round, 8.5 s at
    32 fps (``tests/engine/inference/support.py``), as the dataset's: every other player is in the
    view and its label known, not visible, but for those listed per source frame: the players
    ``visible_at[frame]`` are visible, the labels of ``unknown_at[frame]`` are unknown (in the view,
    not valid) and ``behind_at[frame]`` are out of the view (unknown too). The recording's own row
    is in no view and never valid."""
    shape = (10, 272)
    visible, valid, frustum = np.zeros(shape, bool), np.ones(shape, bool), np.ones(shape, bool)
    valid[slot] = frustum[slot] = False
    for frame, players in visible_at.items():
        visible[list(players), frame] = True
    for frame, players in (unknown_at or {}).items():
        valid[list(players), frame] = False
    for frame, players in (behind_at or {}).items():
        valid[list(players), frame] = frustum[list(players), frame] = False
    root.mkdir(exist_ok=True)
    np.savez(
        root / f"{media_id(slot)}.npz",
        _binary_visible=visible,
        _binary_eval_valid=valid,
        _in_frustum=frustum,
    )


def live_controls(
    frames: int = 16,
    *,
    first: int = 0,
    forward: bool = False,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> LiveControls:
    """A block's controls: nothing held (or ``forward``), turning ``pitch`` and ``yaw`` degrees
    per frame since the session's start; its frames follow frame ``first`` of the session."""
    buttons = np.zeros((frames, len(CONTROL_BUTTONS)), np.float32)
    buttons[:, CONTROL_BUTTONS.index("forward")] = float(forward)
    turn = np.tile(np.array([pitch, yaw], np.float32), (frames, 1))
    after = first + 1 + np.arange(frames)
    return LiveControls(
        buttons=buttons,
        view_deltas=np.stack([quantize_camera_delta(row, clip=False) for row in turn]),
        weapon=np.zeros(frames, np.int64),
        times=after / 16.0,
        turn=turn,
        input_seq=np.full(frames, -1, np.int64),
        turned=np.column_stack([pitch * after, yaw * after]),
    )


async def until(condition: Callable[[], bool], what: str) -> None:
    """Wait until ``condition()`` holds; fail with ``what`` after :data:`TIMEOUT_S`."""
    deadline = time.monotonic() + TIMEOUT_S
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.02)


class Browser:
    """A scripted browser on a player's play socket (an input sample every 16 ms, turning ``turn``
    degrees of yaw per sample while it holds a button) or on a spectator's watch socket; it
    acknowledges each frame as it arrives, as the page does.

    Attributes:
        url (str): the socket's address.
        welcome (dict): the worker's first message.
        frames (list[dict]): the header of every frame received.
        jpegs (list[bytes]): their JPEGs.
        arrived (list[float]): when each arrived, ``time.monotonic()``.
        sent (list[int]): the input samples sent.
        rosters (list[list[int]]): the slots of each ``roster`` the worker sent.
        mbit (float | None): the rate of the browser's link, Mbit/s; ``None``: as fast as the
            machine. A frame of ``n`` bytes arrives ``8 n / rate`` after the one before it has; the
            frames behind it wait in the socket's buffers, as on a slow link.
        ended (str | None): why the worker ended the session.
    """

    def __init__(
        self, ws, url: str, welcome: dict, hold: int | None, mbit: float | None, turn: float
    ) -> None:
        self.ws, self.url, self.welcome, self.mbit, self.turn = ws, url, welcome, mbit, turn
        self.frames: list[dict] = []
        self.jpegs: list[bytes] = []
        self.arrived: list[float] = []
        self.sent: list[int] = []
        self.rosters: list[list[int]] = []
        self.ended: str | None = None
        self._tasks = [asyncio.create_task(self._receive())]
        if hold is not None:
            self._tasks.append(asyncio.create_task(self._send_inputs(hold)))

    @classmethod
    async def open(
        cls,
        http: ClientSession,
        url: str,
        hold: int | None = 0,
        mbit: float | None = None,
        turn: float = 0.5,
    ) -> "Browser":
        """Open the socket at ``url``; ``hold``: the buttons held throughout (``None``: a
        spectator, who sends no inputs); ``mbit``: the link's rate; ``turn``: degrees of yaw per
        input sample while it holds a button."""
        ws = await http.ws_connect(url)
        welcome = await ws.receive_json(timeout=TIMEOUT_S)
        assert welcome["t"] == "welcome"
        return cls(ws, url, welcome, hold, mbit, turn)

    async def _send_inputs(self, hold: int) -> None:
        dyaw = self.turn if hold else 0.0
        for seq in range(1, 1 << 30):
            sample = {"t": "in", "q": seq, "b": hold, "dp": 0.0, "dy": dyaw, "w": 2}
            await self.ws.send_json(sample)
            self.sent.append(seq)
            await asyncio.sleep(0.016)

    async def _receive(self) -> None:
        async for msg in self.ws:
            if msg.type == WSMsgType.BINARY:
                if self.mbit is not None:
                    await asyncio.sleep(8 * len(msg.data) / (self.mbit * 1e6))
                header, jpeg = unpack(msg.data)
                assert jpeg[:2] == b"\xff\xd8"
                self.frames.append(header)
                self.jpegs.append(jpeg)
                self.arrived.append(time.monotonic())
                await self.ws.send_json({"t": "ack", "r": header["k"]})
            elif msg.type == WSMsgType.TEXT:
                data = msg.json()
                if data["t"] == "ended":
                    self.ended = data["message"]
                elif data["t"] == "roster":
                    self.rosters.append([p["slot"] for p in data["roster"]])

    async def show_markers(self, on: bool = True) -> None:
        """Ask the worker for the field's markers in every frame, or no more (``on``), as the
        page's ``M`` does."""
        await self.ws.send_json({"t": "markers", "on": on})

    def states(self, slot: int | None = None) -> list[LivePlayerState]:
        """The player's own state at every frame, or the states the frames carry of the player of
        ``slot``."""
        if slot is None:
            return [LivePlayerState.from_wire(header["s"]) for header in self.frames]
        states = [p for header in self.frames for p in header["p"] if p[0] == slot]
        return [LivePlayerState.from_wire(p) for p in states]

    async def close(self, bye: bool = True) -> None:
        """Leave the seat (``bye``), or just drop the connection."""
        for task in self._tasks[1:]:
            task.cancel()
        if bye:
            await self.ws.send_json({"t": "bye"})
        await self.ws.close()
        await asyncio.gather(*self._tasks, return_exceptions=True)


class RecordingClient(MockClient):
    """A mock client that counts its sessions and records the memory entries it receives,
    ``(slot, frames)``, and what reaches it of the other players, in order: ``(kind, slot)`` of
    each message and ``("removed", slot)`` of each departure."""

    def __init__(self, config, library):
        super().__init__(config, library)
        self.sessions, self.entries, self.events = 0, set(), []

    def start(self, round_start, seat):
        self.sessions, self.entries, self.events = self.sessions + 1, set(), []
        super().start(round_start, seat)

    def receive_message(self, message):
        self.events.append((message.kind, message.slot))
        if message.kind == MEMORY_ENTRY:
            self.entries.add((message.slot, tuple(message.meta["frames"])))
        super().receive_message(message)

    def remove_player(self, slot):
        self.events.append(("removed", slot))
        super().remove_player(slot)


class SlowFirstBlock(RecordingClient):
    """A mock client whose session's first block denoises for 3 s, as a cold GPU's would."""

    def step(self, controls):
        first = self.frame_index == 0

        def take():
            taken = controls()
            if first:
                time.sleep(3.0)
            return taken

        return super().step(take)


class PipelinedClient(RecordingClient):
    """A mock client whose decoder is a stage of its own, as the WorldCast client's is on a second
    card: ``step`` requests the block (its denoising, ``client.mock.denoise_ms``, runs on a thread
    of its own, one block at a time) and returns the block's frames, which the decoder gives, one
    block after the other, ``decode_s`` per group of four once the block has its latents.

    Attributes:
        requested (list[float]): when each block was requested.
        decoded (list[float]): when the decoder gave each block's last frame.
    """

    blocks_in_flight = 2
    decode_s = 0.0

    def __init__(self, config, library):
        super().__init__(config, library)
        self.requested, self.decoded = [], []
        self._generating = threading.Lock()

    def step(self, controls):
        self.requested.append(time.monotonic())
        latents: queue.Queue = queue.Queue()

        def generate():
            with self._generating:
                latents.put(list(MockClient.step(self, controls)))

        threading.Thread(target=generate, daemon=True).start()
        return self._decode(latents)

    def _decode(self, latents):
        frames = latents.get()
        for i, frame in enumerate(frames):
            if i % 4 == 0:
                time.sleep(self.decode_s)
            yield frame
        self.decoded.append(time.monotonic())


@asynccontextmanager
async def demo(
    workers: int, library: Library | None = None, client: type = RecordingClient, **settings
):
    """A local demo of mock workers (each with a ``client``) on free ports, and an HTTP session to
    it."""
    overrides = {
        "coordinator.host": "127.0.0.1",
        "coordinator.port": 0,
        "worker.port": 0,
        "client.mock.denoise_ms": 40,
        "client.mock.jitter_ms": 0,
        "client.mock.decode_ms": 1,
        **settings,
    }
    config = load_config((), overrides)
    library = library or synthetic_library()
    clients = [client(config.client.mock, library) for _ in range(workers)]
    async with LocalDemo(config, library, clients) as local, ClientSession() as http:
        await until(lambda: len(local.coordinator.workers) == workers, "the workers to connect")
        yield local, http


async def create_room(local: LocalDemo, http: ClientSession, round_start: str = "arena") -> str:
    """Open a room at ``round_start``; returns its id."""
    response = await http.post(f"{local.url}/api/rooms", json={"round_start": round_start})
    assert response.status == 200, await response.text()
    return (await response.json())["id"]


def browser_url(local: LocalDemo, url: str) -> str:
    """A WebSocket address the coordinator gave, as a browser opens it."""
    return url if url.startswith("ws") else local.url.replace("http", "ws", 1) + url


async def take_seat(
    local: LocalDemo,
    http: ClientSession,
    room: str,
    slot: int,
    hold: int = 0,
    mbit: float | None = None,
    turn: float = 0.5,
) -> Browser:
    """Join ``room`` as the player of ``slot`` and open its play socket (:meth:`Browser.open`)."""
    response = await http.post(f"{local.url}/api/rooms/{room}/join", json={"slot": slot})
    assert response.status == 200, await response.text()
    url = browser_url(local, (await response.json())["play_url"])
    return await Browser.open(http, url, hold, mbit, turn)


async def watch(
    local: LocalDemo, http: ClientSession, room: str, slot: int, mbit: float | None = None
) -> Browser:
    """Open a spectator's socket on the player of ``slot`` of ``room`` (``mbit``: the link's
    rate)."""
    info = await (await http.get(f"{local.url}/api/rooms/{room}")).json()
    url = next(p["watch_url"] for p in info["players"] if p["slot"] == slot)
    return await Browser.open(http, browser_url(local, url), hold=None, mbit=mbit)


async def recorded_play(frames: int) -> dict:
    """What the page of a player takes from a mock demo, recorded: the lobby's page constants
    (``page``), the join, the worker's ``welcome``, and the header and base64 JPEG of each of the
    first ``frames`` frames."""
    async with demo(1) as (local, http):
        lobby = await (await http.get(f"{local.url}/api/lobby")).json()
        room = await create_room(local, http)
        join = await http.post(
            f"{local.url}/api/rooms/{room}/join", json={"slot": 0, "name": "Ada"}
        )
        join = await join.json()
        browser = await Browser.open(http, browser_url(local, join["play_url"]), hold=0)
        await until(lambda: len(browser.frames) >= frames, f"{frames} frames")
        await browser.close()
    return {
        "page": lobby["page"],
        "join": join,
        "welcome": browser.welcome,
        "frames": [
            [header, base64.b64encode(jpeg).decode()]
            for header, jpeg in zip(browser.frames[:frames], browser.jpegs[:frames])
        ],
    }
