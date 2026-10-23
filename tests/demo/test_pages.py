"""The lobby and the seat picker, in headless Chrome against a mock demo.

The page as served (``static/app.js``, ``app.css``) at 1440 x 900: the round starts' tiles line up
whatever their notes, the button that makes a room names the round start chosen, an empty list of
rooms fills its card, and the seat picker lays its seats out in the columns that leave the last
row fullest. The tests skip without Chrome or Chromium.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from tests.demo.support import chrome_page, demo, needs_chrome
from worldcast.demo.library import DEFAULT_LOADOUT, Library, RoundStart, Seat

pytestmark = needs_chrome

WINDOW = (1440, 900, 1)
#: The round starts: how many seats each has; one has a note that wraps, one a single seat.
SEATS = {"r3": 3, "r4": 4, "r6": 6, "r7": 7, "r10": 10, "r1": 1}
NOTE = "Match 2392812, the round that the paper's third table plays, from its first frame on"
#: What the test's script measures in the page: the elements of a selector, their boxes, and the
#: texts of the summary of the round start chosen.
HELPERS = """window.t = {
    all: (selector) => [...document.querySelectorAll(selector)],
    box: (el) => (el ? el.getBoundingClientRect().toJSON() : null),
    sum: () => [...document.querySelectorAll('#create-sum > *')]
        .map((e) => e.textContent.replace(/\\s+/g, ' ')),
}"""
LOBBY = """{
    eyebrow: document.querySelector('.eyebrow').textContent,
    tops: t.all('.round .thumb').map((e) => t.box(e).top),
    sum: t.sum(),
    card: t.box(document.querySelector('.card-rooms')),
    empty: t.box(document.querySelector('.card-rooms .empty')),
    empty_text: document.querySelector('.card-rooms .empty').textContent,
}"""
ROOMS = """{
    empty: t.all('.card-rooms .empty').length,
    cls: document.querySelector('#rooms').className,
}"""


def library() -> Library:
    def seat(slot: int) -> Seat:
        team = "T" if slot < 5 else "CT"
        return Seat(slot, team, (0.0, 0.0, 0.0, 0.0, 0.0), DEFAULT_LOADOUT[team])

    starts = {
        id: RoundStart(
            id=id,
            map="de_nuke",
            label=f"Round {count}",
            start_frame=0,
            seats=tuple(seat(slot) for slot in range(count)),
            note=NOTE if id == "r6" else "",
        )
        for id, count in SEATS.items()
    }
    return Library(starts)


async def page_report() -> dict:
    """The lobby before and after a click on a round start and a room, and the seat picker of
    each round start."""
    report: dict = {}
    async with demo(1, library()) as (local, http):
        with tempfile.TemporaryDirectory() as directory:
            async with chrome_page(None, WINDOW, Path(directory), url=local.url) as chrome:

                async def measure(expression: str):
                    return json.loads(await chrome.eval(f"JSON.stringify({expression})"))

                await chrome.wait_for(
                    f"document.querySelectorAll('.round').length === {len(SEATS)}"
                )
                await asyncio.sleep(0.5)
                await chrome.eval(HELPERS)
                report["lobby"] = await measure(LOBBY)

                async def choose(id: str) -> list[str]:
                    x, y = await chrome.eval(
                        "(() => { const e = document.querySelector("
                        f"'.round[data-id={id}]'); e.scrollIntoView({{block: 'center'}});"
                        " const r = t.box(e); return [r.left + r.width / 2, r.top + 40]; })()"
                    )
                    await chrome.click(x, y)
                    await asyncio.sleep(0.2)
                    return await measure("t.sum()")

                report["chosen"] = await choose("r4")
                report["single"] = await choose("r1")
                report["seats"], report["subs"] = {}, {}
                for id, count in SEATS.items():
                    answer = await http.post(f"{local.url}/api/rooms", json={"round_start": id})
                    room = (await answer.json())["id"]
                    await chrome.eval(f"location.hash = '#/room/{room}'")
                    await chrome.wait_for(f"t.all('.seat').length === {count}")
                    await asyncio.sleep(0.2)
                    report["seats"][id] = await measure("t.all('.seat').map(t.box)")
                    report["subs"][id] = await chrome.eval("t.all('#room-sub')[0].textContent")
                await chrome.eval("location.hash = '#/'")
                await chrome.wait_for(f"t.all('.room-row').length === {len(SEATS)}")
                report["rooms"] = await measure(ROOMS)
    return report


@pytest.fixture(scope="module")
def pages() -> dict:
    return asyncio.run(page_report())


def test_the_lobby_names_what_it_is_and_what_it_makes(pages):
    assert pages["lobby"]["eyebrow"] == "Distributed multiplayer world models"
    assert pages["lobby"]["sum"] == ["Nuke · Round 3", "3 seats, each on a GPU of its own"]
    assert pages["chosen"] == ["Nuke · Round 4", "4 seats, each on a GPU of its own"]


def test_a_round_start_of_one_seat_says_seat_not_seats(pages):
    assert pages["single"] == ["Nuke · Round 1", "1 seat, each on a GPU of its own"]
    assert pages["subs"]["r1"] == "Nuke · Round 1 · 0 of 1 seat taken"
    assert pages["subs"]["r3"] == "Nuke · Round 3 · 0 of 3 seats taken"


def test_the_round_starts_line_up_whatever_their_notes(pages):
    """The thumbnails of the round starts stand on one line to a row (three, then three), though a
    note wraps under the third."""
    tops = pages["lobby"]["tops"]
    assert tops[:3] == [tops[0]] * 3 and tops[3:] == [tops[3]] * 3 and tops[3] > tops[0]


def test_an_empty_list_of_rooms_fills_its_card_and_says_what_to_do(pages):
    lobby = pages["lobby"]
    card, empty = lobby["card"], lobby["empty"]
    assert empty["height"] > 0.5 * card["height"] and empty["bottom"] < card["bottom"]
    assert "No rooms yet" in lobby["empty_text"] and "share its link" in lobby["empty_text"]
    assert pages["rooms"] == {"empty": 0, "cls": "rooms"}  # the rooms replace it


@pytest.mark.parametrize(
    "id, columns",
    [("r3", 3), ("r4", 4), ("r6", 3), ("r7", 4), ("r10", 5)],
    ids=["r3", "r4", "r6", "r7", "r10"],
)
def test_the_seat_picker_has_the_columns_that_leave_the_last_row_fullest(pages, id, columns):
    """3 seats in 3 columns, 4 in 4, 6 in 3 (two full rows, not 5 and 1), 7 in 4, 10 in 5; the
    tiles of a row are as wide as each other."""
    rects = pages["seats"][id]
    first_row = [r for r in rects if r["top"] == rects[0]["top"]]
    assert len(first_row) == columns
    assert len({round(r["width"]) for r in first_row}) == 1 and first_row[0]["width"] > 200
    assert len({r["top"] for r in rects}) == -(-len(rects) // columns)
