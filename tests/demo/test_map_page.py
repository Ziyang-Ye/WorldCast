"""The lobby's map picker, in headless Chrome against a mock demo.

A click on a map starts a room at the seat nearest to it, on the floor the page shows; the tabs
switch the map and the floor buttons the floor. The tests skip without Chrome or Chromium.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np

from tests.demo.support import chrome_page, demo, needs_chrome, until
from worldcast.demo.library import DEFAULT_LOADOUT, Library, RoundStart, Seat

pytestmark = needs_chrome

WINDOW = (1440, 900, 1)
#: Two floors of a map, split at z = -500.
OVERVIEWS = {
    "overviews": {
        "de_nuke": {
            "origin": [-1000, 1000],
            "scale": 2.0,
            "size": [1000, 1000],
            "floors": [
                {"name": "Upper", "image": None, "z": [-500, 1e9]},
                {"name": "Lower", "image": None, "z": [-1e9, -500]},
            ],
        }
    }
}


def seat(slot: int, x: float, y: float, z: float = 0.0) -> Seat:
    return Seat(slot, "T", (x, y, z, 90.0, 0.0), DEFAULT_LOADOUT["T"])


def library(root: Path) -> Library:
    """Nuke: two floors, a seat on each at the same place and another on the upper floor; Mirage:
    three seats on one."""
    (root / "overviews.json").write_text(json.dumps(OVERVIEWS))
    nuke = RoundStart(
        "nuke", "de_nuke", "Nuke", 0, (seat(0, 0, 0), seat(1, 0, 0, -800.0), seat(2, 600, 600))
    )
    mirage = RoundStart(
        "mirage", "de_mirage", "Mirage", 0, (seat(0, -300, 0), seat(1, 0, 0), seat(2, 300, 0))
    )
    return Library({"nuke": nuke, "mirage": mirage}, root)


async def clicks(root: Path) -> dict:
    """What the page makes of clicks on the maps: the seat each one starts at."""
    report: dict = {}
    async with demo(3, library(root)) as (local, http):
        coordinator = local.coordinator
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            async with chrome_page(None, WINDOW, Path(directory), url=local.url) as chrome:

                async def box(selector: str, index: int = 0) -> list[float]:
                    """The element's box in the window, css px, after it is scrolled into view."""
                    return await chrome.eval(
                        f"(() => {{ const e = document.querySelectorAll('{selector}')[{index}];"
                        " e.scrollIntoView({block: 'center'}); const b = e.getBoundingClientRect();"
                        " return [b.left, b.top, b.width, b.height]; })()"
                    )

                async def press(selector: str, index: int = 0) -> None:
                    left, top, width, height = await box(selector, index)
                    await chrome.click(left + width / 2, top + height / 2)

                async def start(map: str, x: float, y: float) -> tuple[str, int]:
                    """Click the place of a world point; the round start and seat it opens."""
                    u, v = coordinator.picker.maps[map].overview.to_unit(np.array(x), np.array(y))
                    left, top, width, height = await box("#map-view")
                    before = set(coordinator.rooms)
                    await chrome.click(left + float(u) * width, top + float(v) * height)
                    await until(lambda: set(coordinator.rooms) - before, "a room to open")
                    (room,) = (coordinator.rooms[r] for r in set(coordinator.rooms) - before)
                    await until(lambda: room.players, "the player to sit")
                    await chrome.wait_for("location.hash.startsWith('#/play/')")
                    await chrome.eval("location.hash = '#/'")
                    await chrome.wait_for("!!document.querySelector('#map-view')")
                    return room.round_start.id, next(iter(room.players))

                def texts(selector: str) -> str:
                    return f"[...document.querySelectorAll('{selector}')].map((b) => b.textContent)"

                await chrome.wait_for("document.querySelectorAll('.map-tabs button').length === 2")
                report["tabs"] = await chrome.eval(texts(".map-tabs button"))
                # the first map is Nuke (the library's order): two floors, so floor buttons
                await chrome.wait_for(
                    "document.querySelectorAll('.map-floors button').length === 2"
                )
                report["floors"] = await chrome.eval(texts(".map-floors button"))
                report["hint"] = await chrome.eval(texts("#map-hint"))
                # beside the upper floor's seat at the origin: that seat, though the lower
                # floor's is as near
                report["upper"] = await start("de_nuke", 20, 20)
                await press(".map-floors button", 1)
                await chrome.wait_for(
                    "document.querySelectorAll('.map-floors button')[1].ariaPressed === 'true'"
                )
                report["lower"] = await start("de_nuke", 20, 20)
                # the other map, a click away from every seat: the nearest anyway
                await press(".map-tabs button", 1)
                await chrome.wait_for(
                    "document.querySelectorAll('.map-floors button').length === 0"
                )
                report["mirage"] = await start("de_mirage", 280, 400)
    return report


def test_a_click_on_a_map_starts_at_the_nearest_seat_of_the_floor_shown():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        report = asyncio.run(clicks(Path(directory)))
    assert report["tabs"] == ["Nuke", "Mirage"]
    assert report["floors"] == ["Upper", "Lower"]
    assert report["hint"] == ["Click anywhere on the map"]
    assert report["upper"] == ("nuke", 0)
    assert report["lower"] == ("nuke", 1)
    assert report["mirage"] == ("mirage", 2)
