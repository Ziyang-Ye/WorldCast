"""Starting from a map: the coordinator's ``/api/maps`` and ``/api/start`` with mock workers.

Run: ``python -m pytest tests/demo/test_start_from_map.py -q`` (needs ``pip install -e ".[test]"``).
"""

import asyncio

import numpy as np
from aiohttp import web

from tests.demo.support import demo, until
from worldcast.demo.library import synthetic_library


def point_of(local, slot: int) -> dict:
    """The body of a click right on the arena's seat ``slot``."""
    arena = synthetic_library().get("arena")
    x, y = arena.seat(slot).spawn[:2]
    u, v = local.coordinator.picker.maps["arena"].overview.to_unit(np.array(x), np.array(y))
    return {"map": "arena", "u": float(u), "v": float(v), "floor": None}


async def clicks():
    async with demo(2) as (local, http):
        coordinator = local.coordinator
        maps = await (await http.get(f"{local.url}/api/maps")).json()
        first = await http.post(
            f"{local.url}/api/start", json={**point_of(local, 2), "name": "Ada"}
        )
        started = await first.json()
        second = await http.post(f"{local.url}/api/start", json=point_of(local, 3))
        started_too = await second.json()
        third = await http.post(f"{local.url}/api/start", json=point_of(local, 0))
        busy = (third.status, await third.text(), len(coordinator.rooms))
        lobby = await (await http.get(f"{local.url}/api/lobby")).json()
        return maps, (first.status, started), (second.status, started_too), busy, lobby


def test_a_click_opens_a_room_and_seats_the_player_at_the_nearest_seat():
    maps, (status, started), (status_too, started_too), busy, lobby = asyncio.run(clicks())
    (arena,) = maps["maps"]
    assert arena["map"] == "arena" and len(arena["seats"]) == 4
    assert status == 200
    assert started["seat"]["slot"] == 2 and started["name"] == "Ada"
    assert started["room"]["round_start"] == "arena"
    assert [p["slot"] for p in started["room"]["players"]] == [2]
    assert started["play_url"] and started["worker"]["id"]
    # a second click opens a room of its own, on the other worker
    assert status_too == 200 and started_too["seat"]["slot"] == 3
    assert started_too["room"]["id"] != started["room"]["id"]
    assert started_too["worker"]["id"] != started["worker"]["id"]
    assert started_too["name"] == "Player 4"
    # no worker is free: refused, and no room is left behind
    assert busy == (503, "every GPU is busy; try again when a player leaves", 2)
    assert len(lobby["rooms"]) == 2 and lobby["workers"]["free"] == 0


async def refusals():
    async with demo(1) as (local, http):
        coordinator = local.coordinator
        post = lambda body: http.post(f"{local.url}/api/start", json=body)  # noqa: E731
        answers = {
            "an unknown map": await post({**point_of(local, 0), "map": "nowhere"}),
            "no point": await post({"map": "arena"}),
            "a point that is not a number": await post({"map": "arena", "u": "left", "v": 0.5}),
            "a floor the map lacks": await post({**point_of(local, 0), "floor": 3}),
        }
        statuses = {name: (answer.status, await answer.text()) for name, answer in answers.items()}

        async def refuse(room, seat, name):
            raise web.HTTPServiceUnavailable(text="the worker did not answer")

        coordinator._take_seat = refuse
        failed = await post(point_of(local, 0))
        statuses["a seat that cannot be taken"] = (failed.status, await failed.text())
        await until(lambda: coordinator.workers, "the worker")
        return statuses, dict(coordinator.rooms)


def test_a_start_that_cannot_be_served_is_refused_and_leaves_no_room():
    statuses, rooms = asyncio.run(refusals())
    assert statuses == {
        "an unknown map": (404, "no map 'nowhere'"),
        "no point": (400, "a start needs a map and a point on it: u and v"),
        "a point that is not a number": (400, "a start needs a map and a point on it: u and v"),
        "a floor the map lacks": (404, "map 'arena' has no seat on floor 3"),
        "a seat that cannot be taken": (503, "the worker did not answer"),
    }
    assert rooms == {}
