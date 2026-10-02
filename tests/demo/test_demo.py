"""End-to-end check of the demo backend on CPU: a coordinator, three mock workers and three scripted browsers
(two from the start, one joining late).

Run: ``python -m pytest tests/demo -q`` (needs ``pip install -r requirements/demo.txt``).
"""

import asyncio
import math
import socket
import statistics
import time

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import ClientSession, WSMsgType, web  # noqa: E402

from demo.config import load_config  # noqa: E402
from demo.coordinator import Coordinator  # noqa: E402
from demo.engine import PlayerState  # noqa: E402
from demo.library import synthetic_library  # noqa: E402
from demo.mock_engine import MockEngine  # noqa: E402
from demo.protocol import unpack  # noqa: E402
from demo.worker import Worker  # noqa: E402

FORWARD = 1  # bit 0 = PAPER_ACTION_BUTTONS[0] = "forward"


def moved(states) -> float:
    return math.hypot(states[-1].x - states[0].x, states[-1].y - states[0].y)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def serve(app: web.Application, port: int) -> web.AppRunner:
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


async def start_demo(sync: str, route: str):
    cport = free_port()
    overrides = [
        f"coordinator.port={cport}",
        f"worker.coordinator_url=ws://127.0.0.1:{cport}",
        "engine.mock.step_ms=40",
        "engine.mock.jitter_ms=0",
        "engine.mock.decode_ms=1",
        "engine.mock.scene_block_bytes=4096",
        f"play.default_sync={sync}",
        f"coordinator.media_route={route}",
    ]
    library = synthetic_library()
    runners = [await serve(Coordinator(load_config(None, overrides), library).app(), cport)]
    for i in range(3):
        port = free_port()
        cfg = load_config(None, overrides + [f"worker.port={port}", f"worker.worker_id=w{i}"])
        runners.append(
            await serve(Worker(cfg, MockEngine(cfg.engine.mock, library), library).app(), port)
        )
    return f"http://127.0.0.1:{cport}", runners


async def play(http: ClientSession, url: str, seconds: float, hold: int, delay: float = 0.0):
    """A scripted browser: sends one input tick per 16 ms, returns the frame headers with their receive times."""
    await asyncio.sleep(delay)
    frames, sent = [], {}
    async with http.ws_connect(url) as ws:
        welcome = await ws.receive_json(timeout=10)
        assert welcome["t"] == "welcome"

        async def inputs():
            seq = 0
            while True:
                seq += 1
                sent[seq] = time.monotonic()
                await ws.send_json(
                    {"t": "in", "q": seq, "b": hold, "dp": 0.0, "dy": 0.5 if hold else 0.0, "w": 2}
                )
                await asyncio.sleep(0.016)

        task = asyncio.create_task(inputs())
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            msg = await ws.receive(timeout=5)
            if msg.type == WSMsgType.BINARY:
                header, jpeg = unpack(msg.data)
                assert jpeg[:2] == b"\xff\xd8"
                frames.append((time.monotonic(), header))
        task.cancel()
        await ws.send_json({"t": "bye"})
    return frames, sent


async def scenario(sync: str, route: str):
    base, runners = await start_demo(sync, route)

    def ws_url(path):
        return path if path.startswith("ws") else base.replace("http", "ws", 1) + path

    try:
        async with ClientSession() as http:
            for _ in range(100):
                lobby = await (await http.get(f"{base}/api/lobby")).json()
                if lobby["workers"]["free"] == 3:
                    break
                await asyncio.sleep(0.05)
            assert lobby["workers"]["free"] == 3, lobby["workers"]
            room = await (
                await http.post(f"{base}/api/rooms", json={"round": "arena", "sync": sync})
            ).json()
            url = f"{base}/api/rooms/{room['id']}/join"
            joins = [
                await (await http.post(url, json={"seat": seat, "name": f"p{seat}"})).json()
                for seat in (0, 2)
            ]
            taken = await http.post(url, json={"seat": 0})
            assert taken.status == 409

            async def late_joiner():
                await asyncio.sleep(1.5)
                join = await (await http.post(url, json={"seat": 3, "name": "late"})).json()
                return await play(http, ws_url(join["play_url"]), 2.0, 0)

            (frames_a, sent_a), (frames_b, _), (frames_c, _) = await asyncio.gather(
                play(http, ws_url(joins[0]["play_url"]), 4.0, FORWARD),
                play(http, ws_url(joins[1]["play_url"]), 4.0, 0),
                late_joiner(),
            )
            await asyncio.sleep(0.5)
            lobby = await (await http.get(f"{base}/api/lobby")).json()
        return frames_a, sent_a, frames_b, frames_c, lobby
    finally:
        for runner in reversed(runners):
            await runner.cleanup()


@pytest.mark.parametrize("sync, route", [("async", "direct"), ("lockstep", "proxy")])
def test_three_players(sync, route):
    frames_a, sent_a, frames_b, frames_c, lobby = asyncio.run(scenario(sync, route))
    assert len(frames_a) > 40 and len(frames_b) > 40 and len(frames_c) > 15
    # each player's frames carry the other player's state-model position
    seen_by_b = [PlayerState.from_wire(p) for _, h in frames_b for p in h["p"] if p[0] == 0]
    seen_by_a = [PlayerState.from_wire(p) for _, h in frames_a for p in h["p"] if p[0] == 2]
    assert seen_by_a and seen_by_b
    # player A holds forward: B sees A move, and A's own state moves
    assert moved(seen_by_b) > 300
    own = [PlayerState.from_wire(h["s"]) for _, h in frames_a]
    assert moved(own) > 300
    # input-to-frame latency of A (receive time of the frame that first folds in each input)
    latency = [1000 * (t - sent_a[h["q"]]) for t, h in frames_a if h["q"] in sent_a]
    assert latency and statistics.median(latency) < 1000
    print(
        f"{sync}: {len(frames_a)} frames, input-to-frame median {statistics.median(latency):.0f} ms"
    )
    # the late joiner sees the others and got the scene-state blocks published before it joined
    assert {p[0] for _, h in frames_c for p in h["p"]} == {0, 2}
    assert max(h["x"]["scene_in"] for _, h in frames_c if "x" in h) >= 2
    # everyone left: the workers are free again
    assert lobby["workers"]["free"] == 3
