"""Where everything of the play view is, in headless Chrome.

The page's own script and style (``static/play.js``, ``app.css``) run on the welcome and the frames
of a recorded mock session (``play_layout.html``), driven with the mouse as a person has it (a press
and its release a tenth of a second later); at ten window sizes the test reads where the stage,
the HUD and the panels are. The frame is shown at its own pixel size, centred, in every window of
1000 x 700 or more, with the latency breakdown open or not, and nothing but the crosshair and the
field's markers is over it while one plays. The tests skip without Chrome or Chromium.
"""

import asyncio
import json
import re
from pathlib import Path

import pytest

from tests.demo.support import chrome_page, needs_chrome, recorded_play
from worldcast.data.latents import FRAME_SIZE

HARNESS = Path(__file__).with_name("play_layout.html")
pytestmark = needs_chrome

HEIGHT, WIDTH = FRAME_SIZE
#: The window sizes (width, height, device pixels per CSS pixel): a laptop, a large monitor, a
#: laptop's retina screen, the narrowest and lowest windows that dock the breakdown, the smallest
#: window that shows the frame whole with its surroundings at the sides, a short laptop window, two
#: narrower than 1000 (the cards below the key caps; the second too low for the frame whole), one
#: too small for the frame, a phone's width.
WINDOWS = {
    "laptop": (1440, 900, 1),
    "monitor": (1920, 1080, 1),
    "retina": (1440, 900, 2),
    "dock": (1260, 700, 1),
    "narrow": (1000, 700, 1),
    "short": (1366, 650, 1),
    "mid": (900, 820, 1),
    "mid-short": (900, 700, 1),
    "small": (1000, 540, 1),
    "phone": (390, 844, 3),
}
#: Where the stats button opens the latency breakdown: docked at the right of the view in a window
#: of 1260 or more (landscape), a popover under the stats in the others.
DOCKED = ("laptop", "monitor", "retina", "dock", "short")
#: The windows that show the frame whole, with the surroundings at its sides.
NATIVE = ("laptop", "monitor", "retina", "dock", "narrow", "short")
#: The windows narrower than 1000, with the cards in a row below the key caps.
BELOW = ("mid", "mid-short")
#: The touch devices, with a coarse pointer and the on-screen pad: a phone upright, a phone on its
#: side, a small phone on its side, a tablet on its side.
TOUCH = {
    "phone-upright": (390, 844, 3),
    "phone-sideways": (844, 390, 3),
    "phone-small": (667, 375, 3),
    "tablet-sideways": (1024, 768, 2),
}
#: A recorded session's frames the page plays: more than the jitter buffer holds.
FRAMES = 24
#: What lies in the stage: the view, the markers' canvas (hidden until M) and the crosshair.
IN_STAGE = ["canvas.view", "canvas.view marks[hidden]", "svg"]
MARKED_STAGE = ["canvas.view", "canvas.view marks", "svg"]
#: The markers every frame carries, [slot, u, v, depth]: a player near, one far, one very far.
MARKERS = [[3, 0.25, 0.75, 100.0], [4, 0.5, 0.6, 1000.0], [5, 0.75, 0.45, 20000.0]]
#: The other players of a room: one joins (``pair``), then all four (``crowd``: five in the room).
GUESTS = [
    {"slot": 3, "name": "Grace", "team": "CT"},
    {"slot": 4, "name": "Linus", "team": "CT"},
    {"slot": 5, "name": "Ken", "team": "T"},
    {"slot": 6, "name": "Margaret", "team": "T"},
]
#: How the own position is known on every frame: what the breakdown shows and the bar does not.
POSITION = "degraded"
CLOSE = pytest.approx


async def recorded_session() -> dict:
    """What the page of a player takes from a mock demo: the lobby's constants, the join, the
    worker's welcome, and the header and JPEG of each frame (carrying the markers and a position
    status)."""
    session = await recorded_play(FRAMES)
    frames = [[{**h, "m": MARKERS, "st": POSITION}, jpeg] for h, jpeg in session["frames"]]
    return {**session, "frames": frames, "markers": MARKERS, "guests": GUESTS}


async def drive(setup: dict, window: tuple[int, int, int], directory: Path, touch=False) -> dict:
    """Play the page in a window, as a person does, and report where everything is at each stage:
    ``starting``, ``paused``, ``latency`` and ``playing`` (after a click on Play, with the latency
    breakdown opened while paused: it is there if it stays, and not after it), ``pair`` (a second
    player in the room), ``crowd`` (five), ``marked`` (M pressed), ``dropped`` (the socket closed),
    ``ended``, ``hidden`` (the tab hidden); and under
    ``controls`` what the mouse did: whether Play stayed the same element under the frames, what
    the stats button reached and opened, and what a click away from the breakdown did. A touch
    device taps Play and has only ``paused``, ``playing`` and ``crowd``."""
    page = HARNESS.read_text(encoding="utf-8").replace("__SETUP__", json.dumps(setup))
    height = window[1]
    async with chrome_page(page, window, directory, touch=touch) as chrome:
        await chrome.wait_for("window.ready === true")
        assert (
            await chrome.eval("window.harness ? (harness.out.error ?? null) : 'no harness'") is None
        )

        async def report(stage: str) -> None:
            await chrome.eval(f"harness.report({stage!r})")

        async def state() -> dict:
            return await chrome.eval("harness.state()")

        async def click(selector: str) -> None:
            x, y = await chrome.eval(f"harness.centre({selector!r})")
            await chrome.click(x, y)

        playing = "harness.state().fps > 0 && harness.state().clock !== '0:00:00'"
        if touch:
            await chrome.wait_for("!!document.querySelector('#go')")
            await asyncio.sleep(0.1)
            await report("paused")
            x, y = await chrome.eval("harness.centre('#go')")
            await chrome.tap(x, y)
            await chrome.wait_for(playing)
            await asyncio.sleep(0.4)
            await report("playing")
            await chrome.eval("harness.roster(4)")
            await asyncio.sleep(0.15)
            await report("crowd")
            return await chrome.eval("harness.out.stages")
        controls = {}
        await chrome.wait_for("!!document.querySelector('#go')")
        await asyncio.sleep(0.1)
        await chrome.eval("harness.keep()")
        await report("paused")
        await asyncio.sleep(1.2)
        controls["go_stays"] = await chrome.eval("harness.kept()")
        controls["frames_under_the_panel"] = (await state())["fps"] > 0
        controls["stats_reached"] = (await state())["stats"]["reached"]
        await click("#stats")
        await asyncio.sleep(0.3)
        controls["opened"] = await state()
        await chrome.click(8, height - 8)  # on the dimmed page, away from the panel and the stats
        await asyncio.sleep(0.2)
        controls["away"] = await state()
        await chrome.wait_for("harness.reached('#go')")
        await click("#go")
        await chrome.wait_for(playing)
        await asyncio.sleep(0.4)
        await report("latency")
        if (await state())["panel"]:
            await chrome.eval("document.querySelector('#stats').click()")  # close it, no press
            await asyncio.sleep(0.1)
        await report("playing")
        await chrome.eval("harness.roster(1)")
        await asyncio.sleep(0.15)
        await report("pair")
        await chrome.eval("harness.roster(4)")
        await asyncio.sleep(0.15)
        await report("crowd")
        await chrome.eval("harness.key('KeyM')")
        await chrome.wait_for("!document.querySelector('#marks').hidden")
        await asyncio.sleep(0.4)
        await report("marked")
        controls["rings"] = await chrome.eval("harness.marked()")
        await chrome.eval("harness.key('KeyM')")
        await chrome.wait_for("document.querySelector('#marks').hidden")
        await chrome.eval("harness.drop()")
        await asyncio.sleep(0.2)
        await report("dropped")
        await chrome.eval("harness.end()")
        await asyncio.sleep(0.2)
        await report("ended")
        controls["ended"] = await state()
        await chrome.eval("harness.hide()")
        await report("hidden")
        return {**await chrome.eval("harness.out.stages"), "controls": controls}


@pytest.fixture(scope="module")
def session() -> dict:
    """The recorded mock session the page plays (:func:`recorded_session`)."""
    return asyncio.run(recorded_session())


@pytest.fixture(scope="module")
def stages(session, tmp_path_factory) -> dict[str, dict]:
    """The report of every window size (:func:`drive`)."""
    setup = session
    return {
        name: asyncio.run(drive(setup, window, tmp_path_factory.mktemp(name)))
        for name, window in WINDOWS.items()
    } | {
        name: asyncio.run(drive(setup, window, tmp_path_factory.mktemp(name), touch=True))
        for name, window in TOUCH.items()
    }


@pytest.fixture
def windows(stages) -> dict[str, dict]:
    """The reports of the windows with a mouse."""
    return {name: report for name, report in stages.items() if name in WINDOWS}


def intersects(a: list[float], b: list[float]) -> bool:
    """Whether two ``[x, y, width, height]`` rectangles share area."""
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def test_the_frame_is_shown_at_its_own_size_in_every_window_of_1000_by_700_or_more(windows):
    """Wherever the window is 1000 x 700 or more, or a laptop's short 1366 x 650, the stage is the
    frame's 672 x 384 CSS pixels, centred across, and the page does not scroll; a larger window
    leaves it as it is, and the roomiest also centre it down."""
    assert (WIDTH, HEIGHT) == (672, 384)
    for name in NATIVE:
        stage = windows[name]["playing"]
        width, height, _ = stage["viewport"]
        x, y, w, h = stage["stage"]
        assert (w, h) == (WIDTH, HEIGHT) and x == CLOSE((width - WIDTH) / 2, abs=0.5), name
        assert stage["native"] and 0 < y and y + h < height, name
        scroll_width, scroll_height, client_width, client_height = stage["playScroll"]
        assert (scroll_width, scroll_height) == (client_width, client_height), name
    for name in ("laptop", "monitor", "retina"):
        stage = windows[name]["playing"]
        width, height, _ = stage["viewport"]
        assert stage["stage"] == CLOSE(
            [(width - WIDTH) / 2, (height - HEIGHT) / 2, WIDTH, HEIGHT], abs=0.5
        )


def test_opening_the_docked_breakdown_moves_nothing(windows):
    """Docked (a window of 1260 or more, landscape) the breakdown lies in the loadout's place at
    the right of the view and out of the grid's sizing: the stage stays where it was, the play view
    does not scroll, and the breakdown is whole in the window, clear of the view, the loadout
    stepping aside."""
    for name in DOCKED:
        closed, opened = windows[name]["playing"], windows[name]["latency"]
        assert opened["stage"] == closed["stage"], name
        for stage in (closed, opened):
            assert stage["playScroll"][:2] == stage["playScroll"][2:], (name, stage["playScroll"])
        x, y, width, height = opened["hud"]["#lat-panel"]
        stage_x, _, stage_width, _ = opened["stage"]
        viewport = opened["viewport"]
        assert stage_x + stage_width <= x and x + width <= viewport[0], name
        assert 0 <= y and y + height <= viewport[1], name
        assert closed["hud"][".weapons-card"] is not None, name
        assert opened["hud"][".weapons-card"] is None, name


def test_the_play_view_does_not_scroll_and_the_frame_stays_put_whoever_is_in_the_room(windows):
    """The cards below the key caps (a window narrower than 1000) are a row of 126 px: the players'
    card with the line for a player who is alone fits it, and a room of five scrolls inside its
    card. Nowhere does the play view scroll, or the frame move, whether the player is alone, with
    one more or with four."""
    for name, window in windows.items():
        if name == "phone":
            continue  # the page scrolls, down its column
        for stage in ("playing", "pair", "crowd"):
            scroll = window[stage]["playScroll"]
            assert scroll[:2] == scroll[2:], (name, stage, scroll)
            assert window[stage]["stage"] == window["playing"]["stage"], (name, stage)
    for name in BELOW:
        for stage in ("playing", "pair", "crowd"):
            hud = windows[name][stage]["hud"]
            assert hud[".flank-left"][3] == hud[".flank-right"][3] == CLOSE(126, abs=0.5), name


def test_the_play_button_stays_the_same_element_while_frames_arrive_under_the_panel(windows):
    """A click is a press and its release on one element, about a tenth of a second apart, here a
    real mouse's: the panel is drawn once, not at every frame, and the click reaches Play."""
    for name, window in windows.items():
        assert window["controls"]["frames_under_the_panel"], name
        assert window["controls"]["go_stays"], name
        assert window["playing"]["overlay"] is None, name


def test_the_frame_shrinks_in_its_aspect_ratio_where_the_window_is_too_small(windows):
    """A window of 1000 x 540 and one 390 wide cannot hold the frame whole: it shrinks, as does one
    of 900 x 700, whose cards below the key caps take 126 px; one of 900 x 820 can hold it."""
    for name in ("small", "mid-short", "phone"):
        stage = windows[name]["playing"]
        x, y, width, height = stage["stage"]
        assert width < WIDTH and width / height == CLOSE(WIDTH / HEIGHT, abs=0.01), name
        assert stage["scroll"][0] == stage["viewport"][0], name  # no sideways scrolling
        assert x == CLOSE((stage["viewport"][0] - width) / 2, abs=0.5), name
    assert windows["small"]["playing"]["stage"][2] == CLOSE(535.5, abs=0.5)  # 540 - 234, x 7/4
    assert windows["mid-short"]["playing"]["stage"][2] == CLOSE(518, abs=0.5)  # 700 - 404, x 7/4
    assert windows["mid"]["playing"]["stage"][2:] == [WIDTH, HEIGHT]  # 900 x 820 holds it


def test_the_canvas_holds_the_frame_at_its_own_pixels_whatever_the_screen(windows):
    """No device pixel ratio or window size makes the view's backing store larger than the frame:
    a frame stretched into a larger store is only blurrier. On a screen of two device pixels per
    CSS pixel the browser then shows each pixel as a square."""
    for name, window in windows.items():
        sizes = {tuple(stage["view"]) for stage in window.values() if "view" in stage}
        assert sizes == {(WIDTH, HEIGHT)}, name
    assert (
        windows["retina"]["playing"]["viewport"][2] == 2 and windows["retina"]["playing"]["native"]
    )
    assert windows["laptop"]["playing"]["native"] and not windows["small"]["playing"]["native"]
    assert not windows["phone"]["playing"]["native"]  # shrunk to the phone's width: smoothed


def test_nothing_but_the_crosshair_and_the_markers_is_over_the_frame_while_one_plays(windows):
    for name, window in windows.items():
        for stage in ("playing", "latency"):
            assert window[stage]["inside"] == IN_STAGE, name
            assert window[stage]["over"] == [] and window[stage]["overlay"] is None, (name, stage)
            for selector, box in window[stage]["hud"].items():
                assert box is None or not intersects(box, window[stage]["stage"]), (name, selector)


def test_the_markers_canvas_and_the_crosshair_lie_on_the_frame(windows):
    """After M the markers' canvas is the stage's rectangle, in a store of the screen's pixels up to
    two per CSS pixel, and the crosshair's middle is the stage's middle."""
    for name, window in windows.items():
        marked = window["marked"]
        x, y, width, height = marked["stage"]
        assert marked["inside"] == MARKED_STAGE and marked["over"] == [], name
        assert marked["marks"]["rect"] == CLOSE([x, y, width, height], abs=0.5), name
        scale = min(2, marked["viewport"][2])
        assert marked["marks"]["store"] == CLOSE([width * scale, height * scale], abs=1), name
        if name == "phone":
            assert marked["crosshair"] is None  # a phone has no crosshair
        else:
            left, top, size, _ = marked["crosshair"]
            middle = (x + width / 2, y + height / 2)
            assert (left + size / 2, top + size / 2) == CLOSE(middle, abs=0.5)
    assert windows["retina"]["marked"]["marks"]["store"] == [2 * WIDTH, 2 * HEIGHT]
    assert windows["laptop"]["marked"]["marks"]["store"] == [WIDTH, HEIGHT]


def test_a_marker_is_a_ring_with_a_label_that_shows_at_any_distance(windows):
    """A player 100 u away is marked by a ring 60 px across (the figure's size close up), one at
    1,000 u or at 20,000 u by a ring at least 24 px across and 14 px high: an outline on a dark one
    (the pixels reach 2.5 px past the ring), with its label above."""
    for name, window in windows.items():
        near, middle, far = window["controls"]["rings"]
        for ring in (near, middle, far):
            left, top, right, bottom = ring["box"]
            x, y = ring["at"]
            assert (left + right) / 2 == CLOSE(x, abs=1.5), name
            assert y - top > 19 and bottom - y == CLOSE(15 if ring is near else 8.5, abs=1.5), name
        assert near["box"][2] - near["box"][0] == CLOSE(63, abs=2), name
        for ring in (middle, far):
            assert ring["box"][2] - ring["box"][0] == CLOSE(27, abs=2), name


def test_the_latency_breakdown_opens_with_the_mouse_while_paused(windows):
    """The mouse is free only while paused, when a panel covers the page: a pointer at the middle
    of the stats button reaches it, and a click opens the breakdown. At the right of the view in a
    window of 1260 or more, where it stays when one plays and a click away leaves it; elsewhere a
    popover that takes the pause panel's place, closed by a click away and gone when one plays."""
    for name, window in windows.items():
        controls = window["controls"]
        assert controls["stats_reached"], name
        assert window["paused"]["overlay"]["title"] == "Click to play", name
        opened, away = controls["opened"], controls["away"]
        assert opened["panel"] and "INPUT → PHOTON" in opened["panel"]["text"], name
        assert opened["card"] is (name in DOCKED), name
        if name in DOCKED:
            assert away["panel"] and away["card"], name
            assert window["latency"]["hud"]["#lat-panel"] is not None, name
        else:
            assert away["panel"] is None and away["card"], name
            assert window["latency"]["hud"]["#lat-panel"] is None, name
        assert window["playing"]["hud"]["#lat-panel"] is None, name
        x, y, width, height = opened["panel"]["rect"]
        viewport = window["paused"]["viewport"]
        assert 0 <= x and x + width <= viewport[0] and 0 <= y and y + height <= viewport[1], name


def test_the_breakdown_says_how_the_own_position_is_known_not_the_bar(windows):
    """How the own position is known (the frame header's ``st``) is no longer in the bar above the
    view, where a player saw a red word for most of a session: the latency breakdown says it, with
    what it means in a sentence."""
    for name, window in windows.items():
        bar = window["playing"]["text"]["bar"]
        assert (
            POSITION not in bar.lower() and "guarded" not in bar and "reacquiring" not in bar
        ), name
        assert bar.startswith("Ada P1 ") and re.search(r"\d+:\d\d:\d\d", bar), (name, bar)
        text = window["controls"]["opened"]["panel"]["text"]
        assert "YOUR POSITION · DEGRADED" in text, name
        assert (
            "so it is ignored for now and your position follows your movement as far as the view"
            " followed it." in text
        ), name
        assert "Press a key, click or move the mouse" in text, name  # nothing sent yet


def test_the_pause_and_start_panels_cover_the_page_and_go_when_one_plays(windows):
    for name, window in windows.items():
        width, height, _ = window["playing"]["viewport"]
        starting, paused = window["starting"]["overlay"], window["paused"]["overlay"]
        assert starting["title"] == "Starting your client" and paused["title"] == "Click to play"
        assert starting["rect"] == paused["rect"] == [0, 0, width, height], name
        assert window["playing"]["overlay"] is None


def test_the_panels_say_what_runs_the_client_and_what_a_player_can_do_next(windows):
    """The recorded session is the mock client's, which has no GPU: the start and pause panels say
    so, instead of calling it a GPU worker that generates the view. A finished session offers the
    lobby and the same room's seats again."""
    for name, window in windows.items():
        assert (
            window["starting"]["overlay"]["text"]
            == "Seat 1 on mock client local-1. Loading the round start…"
        ), name
        assert window["paused"]["overlay"]["text"] == (
            "The mock client local-1 plays pre-rendered frames: no GPU generates your view."
        ), name
        assert window["ended"]["overlay"]["text"] == "You were away.", name
        assert window["ended"]["overlay"]["links"] == [
            ["Back to the lobby", "#/"],
            ["Take a seat again", "#/room/room"],
        ], name


def test_a_lost_connection_names_what_ran_the_client(windows):
    """The recorded session is the mock client's: a socket that closes says so, not that a GPU
    worker was lost."""
    for name, window in windows.items():
        overlay = window["dropped"]["overlay"]
        assert overlay["title"] == "Session ended", name
        assert overlay["text"] == "Lost the connection to the mock client.", name


def test_a_finished_session_leaves_no_breakdown_and_a_dead_stats_button(windows):
    for name, window in windows.items():
        ended = window["controls"]["ended"]
        assert window["ended"]["overlay"]["title"] == "Session ended", name
        assert ended["panel"] is None and not ended["stats"]["reached"], name


def test_the_page_tells_the_worker_when_it_is_hidden(windows):
    """The play view says with the welcome that it is shown, and again when the tab is hidden (the
    worker releases a seat whose page stays hidden)."""
    for name, window in windows.items():
        assert window["playing"]["visibility"] == [False], name
        assert window["hidden"]["visibility"] == [False, True], name


def test_a_player_alone_is_told_that_others_appear_when_they_take_a_seat(windows):
    """The players' card says it, calmly, while the player is alone in the room, and the line goes
    when a second player is there."""
    line = "Others appear in your view when they take a seat in this room."
    for name, window in windows.items():
        assert window["playing"]["text"]["alone"] == line, name
        assert window["playing"]["text"]["roster"].startswith("Ada P1"), name
        assert window["pair"]["text"]["alone"] is None, name
        assert "Grace" in window["pair"]["text"]["roster"], name
        assert window["crowd"]["text"]["alone"] is None, name


def test_the_hud_sits_around_the_frame_on_a_desktop(windows):
    """Who plays and the clock above, the key caps below, the players on its left, the loadout on
    its right and, in a wide window, once it is open, the latency breakdown in the loadout's
    place."""
    for name in NATIVE:
        stage = windows[name]["latency"]
        x, y, width, height = stage["stage"]
        hud = stage["hud"]
        assert hud[".who"][1] + hud[".who"][3] <= y and hud[".stats"][1] + hud[".stats"][3] <= y
        assert hud[".keys"][1] >= y + height
        assert hud[".flank-left"][0] + hud[".flank-left"][2] <= x, name
        assert hud[".flank-right"][0] >= x + width, name
        assert (hud[".weapons-card"] is None) is (name in DOCKED)  # the breakdown has its place
        assert hud[".pad"] is None  # a mouse has no on-screen pad
        assert (hud["#lat-panel"] is not None) is (name in DOCKED), name
        assert "Ada" in stage["text"]["roster"]
        clock = windows[name]["playing"]["text"]["clock"]
        assert re.fullmatch(r"\d+:\d\d:\d\d", clock) and clock != "0:00:00"  # a frame set it


def test_in_a_window_narrower_than_1000_the_cards_go_below_the_key_caps(windows):
    hud = windows["mid"]["playing"]["hud"]
    assert hud[".flank-left"][1] >= hud[".keys"][1] + hud[".keys"][3]
    assert hud[".flank-right"][1] >= hud[".keys"][1] + hud[".keys"][3]


def test_on_a_phone_the_frame_is_on_top_at_the_full_width_and_the_hud_below(windows):
    """A narrow window with a mouse: the view on top, the HUD below it, and no on-screen pad, which
    a mouse does not need."""
    stage = windows["phone"]["playing"]
    x, y, width, height = stage["stage"]
    assert (x, y, width) == (0, 0, 390) and height == CLOSE(390 * HEIGHT / WIDTH, abs=0.05)
    hud = stage["hud"]
    assert hud[".who"][1] >= height and hud[".stats"][1] >= height
    assert hud[".keys"] is None and hud[".weapons-card"] is None and hud[".pad"] is None
    assert stage["scroll"][0] == 390 and stage["pointer"] == "fine"


def test_a_touch_device_taps_play_and_is_told_how_the_pad_plays(stages):
    """A phone has the on-screen pad: its pause panel (a tap on Play starts) says what the sticks
    and the buttons do, with the gamepad, and has no keyboard settings."""
    for name in TOUCH:
        paused = stages[name]["paused"]["overlay"]
        assert paused["title"] == "Tap to play", name
        assert paused["legend"] == [
            ["Left stick", "Move"],
            ["Right stick", "Look"],
            ["Buttons", "Jump, crouch, fire, next weapon"],
            [
                "Gamepad",
                "Sticks move and look; A jump, B crouch, X reload, Y inspect (knife); triggers "
                "fire; bumpers and D-pad take weapons",
            ],
        ], name
        assert paused["settings"] is False, name
        assert stages[name]["playing"]["overlay"] is None, name
        assert stages[name]["playing"]["pointer"] == "coarse", name


def test_the_pad_is_outside_the_frame_whatever_the_device_and_the_room(stages):
    """Nothing but the crosshair is over the frame while a touch device plays; the pad's sticks
    (at least 100 px across) and buttons (at least 44 px) lie whole in the window, clear of the
    frame, whoever is in the room, and the page does not scroll sideways. Its elements take no
    scroll or zoom from a finger."""
    for name, (width, height, _) in TOUCH.items():
        for stage in ("playing", "crowd"):
            report = stages[name][stage]
            assert report["inside"] == IN_STAGE and report["over"] == [], (name, stage)
            pad = report["pad"]
            assert len(pad["halves"]) == 2 and len(pad["sticks"]) == 2, name
            assert len(pad["buttons"]) == 4 and set(pad["touchAction"]) == {"none"}, name
            for box in (*pad["halves"], *pad["sticks"], *pad["buttons"]):
                assert not intersects(box, report["stage"]), (name, stage, box)
                assert 0 <= box[0] and box[0] + box[2] <= width, (name, stage, box)
                assert 0 <= box[1] and box[1] + box[3] <= height, (name, stage, box)
            assert all(stick[2] >= 100 and stick[3] >= 100 for stick in pad["sticks"]), name
            assert all(button[2] >= 44 and button[3] >= 44 for button in pad["buttons"]), name
            assert report["scroll"][0] == width, (name, stage)
            assert report["crosshair"] is not None, name


def test_the_pad_is_below_the_frame_in_a_portrait_window(stages):
    """Upright, the frame is on top at the full width, the HUD under it and the pad at the bottom
    of the window, a stick in each corner, each with its two buttons above it."""
    for stage in ("playing", "crowd"):
        report = stages["phone-upright"][stage]
        x, y, width, height = report["stage"]
        assert (x, y, width) == (0, 0, 390) and height == CLOSE(390 * HEIGHT / WIDTH, abs=0.05)
        pad = report["pad"]
        assert all(half[1] >= height for half in pad["halves"]), stage
        left, right = pad["halves"]
        assert left[0] + left[2] <= right[0], stage
        assert all(half[1] + half[3] >= 844 - 30 for half in pad["halves"]), stage
        assert pad["pause"] is not None and pad["pause"][1] >= height, stage


def test_the_pad_is_at_the_sides_of_the_frame_in_a_landscape_window(stages):
    """On its side, the left half of the pad (movement, jump, crouch) is left of the frame and the
    right half (look, fire, weapon) right of it, the frame centred between them and shrinking only
    where the window has no room: whole (672 x 384) on a tablet, 534 wide on a phone of 844 x 390,
    379 on one of 667 x 375."""
    for name, (width, _, _), frame in (
        ("phone-sideways", TOUCH["phone-sideways"], 534),
        ("phone-small", TOUCH["phone-small"], 379),
        ("tablet-sideways", TOUCH["tablet-sideways"], 672),
    ):
        for stage in ("playing", "crowd"):
            report = stages[name][stage]
            x, y, w, h = report["stage"]
            assert w == CLOSE(frame, abs=1) and w / h == CLOSE(WIDTH / HEIGHT, abs=0.01), name
            assert x + w / 2 == CLOSE(width / 2, abs=1), name
            left, right = report["pad"]["halves"]
            assert left[0] + left[2] <= x and right[0] >= x + w, name
            assert report["hud"][".who"][1] + report["hud"][".who"][3] <= y, name
            assert report["pad"]["pause"] is not None, name
    assert stages["tablet-sideways"]["playing"]["stage"][2:] == [WIDTH, HEIGHT]


def test_on_a_phone_every_card_is_as_tall_as_what_it_holds(stages):
    """Below the frame the players' card stands as tall as its text, whoever is in the room: the
    cards' row of the windows under 1000 px (a fixed height, a card that scrolls) is not the
    phone's."""
    for name, four_rows in (("phone-upright", 83), ("phone-sideways", 101)):  # the line of text
        for stage in ("playing", "crowd"):  # for one alone wraps once more in the narrow card
            report = stages[name][stage]
            assert all(shown == whole for shown, whole in report["plates"]), (
                name,
                stage,
                report["plates"],
            )
        alone, crowd = stages[name]["playing"], stages[name]["crowd"]
        assert crowd["hud"][".flank-left"][3] == CLOSE(
            alone["hud"][".flank-left"][3] + four_rows, abs=3
        ), name


async def cross(setup: dict, directory: Path) -> dict:
    """Open the breakdown with the mouse in a window of 1300 x 800 (docked), narrow the window to
    1000 x 800 (a popover layout), open it there, widen the window again and open it once more:
    after each step, whether the breakdown is open (the panel's rectangle), its inline place and
    the play view's classes."""
    page = HARNESS.read_text(encoding="utf-8").replace("__SETUP__", json.dumps(setup))
    async with chrome_page(page, (1300, 800, 1), directory) as chrome:
        await chrome.wait_for("window.ready === true")
        await chrome.wait_for("!!document.querySelector('#go')")
        steps = {}

        async def step(name: str, action: str, *args) -> None:
            if action == "stats":
                x, y = await chrome.eval("harness.centre('#stats')")
                await chrome.click(x, y)
            else:
                size = dict(width=args[0], height=800, deviceScaleFactor=1, mobile=False)
                await chrome.call("Emulation.setDeviceMetricsOverride", **size)
            await asyncio.sleep(0.4)
            steps[name] = await chrome.eval("harness.panelStyle()")
            steps[name]["panel"] = (await chrome.eval("harness.state()"))["panel"]

        await step("docked", "stats")
        await step("narrowed", "resize", 1000)
        await step("popover", "stats")
        await step("widened", "resize", 1300)
        await step("docked again", "stats")
        return steps


def test_a_window_that_crosses_between_the_layouts_closes_the_breakdown(session, tmp_path):
    """Docked, the breakdown would become a popover over a view being played when the window is
    narrowed; a popover leaves its inline place behind when the window is widened. Either way the
    breakdown closes, and what opens next is placed for the layout it opens in."""
    steps = asyncio.run(cross(session, tmp_path))
    docked, narrowed = steps["docked"], steps["narrowed"]
    assert docked["panel"] and "docked" in docked["classes"].split()
    assert (docked["top"], docked["right"]) == ("", "")
    stage_x, _, stage_width, _ = docked["stage"]
    assert docked["panel"]["rect"][0] == docked["flank"][0] >= stage_x + stage_width
    assert narrowed["panel"] is None and "docked" not in narrowed["classes"].split()
    assert (narrowed["top"], narrowed["right"]) == ("", "")
    popover = steps["popover"]
    assert popover["panel"] and "measuring" in popover["classes"].split()
    assert popover["top"].endswith("px") and popover["right"].endswith("px")
    widened = steps["widened"]
    assert widened["panel"] is None and "measuring" not in widened["classes"].split()
    assert (widened["top"], widened["right"]) == ("", "")
    again = steps["docked again"]
    assert again["panel"] and (again["top"], again["right"]) == ("", "")
    assert again["panel"]["rect"][:2] == again["flank"][:2]


#: The windows of a mouse (width, height) the pause panel is swept over: from a phone's width to a
#: laptop's, from a window that is a bit of a screen's height to a tall one.
SWEEP_WIDTHS = (360, 480, 600, 683, 768, 900, 1100)
SWEEP_HEIGHTS = (400, 500, 600, 700, 800)
#: What a sweep reads at each window size: the panel, its list and its two buttons, the stats
#: button, whether the buttons and the stats are what a pointer at their middle reaches, and the
#: window.
SWEEP = """(() => {
  const q = (selector) => document.querySelector(selector);
  const rect = (el) => {
    const b = el.getBoundingClientRect();
    return [b.left, b.top, b.width, b.height];
  };
  const reached = (selector) => {
    const [x, y, w, h] = rect(q(selector));
    return !!document.elementFromPoint(x + w / 2, y + h / 2)?.closest(selector);
  };
  return {
    panel: rect(q('.panel-pause')), list: rect(q('.panel-scroll')), go: rect(q('#go')),
    leave: rect(q('#leave')), stats: rect(q('#stats')), goReached: reached('#go'),
    leaveReached: reached('#leave'), statsReached: reached('#stats'),
    viewport: [innerWidth, innerHeight], cramped: q('.play').classList.contains('cramped'),
  };
})()"""


async def sweep(setup: dict, directory: Path) -> dict[tuple[int, int], dict]:
    """The pause panel of a mouse's page at each window of the sweep (the window is resized under
    the open panel, as a person drags a window's edge)."""
    page = HARNESS.read_text(encoding="utf-8").replace("__SETUP__", json.dumps(setup))
    reports = {}
    async with chrome_page(page, (1100, 800, 1), directory) as chrome:
        await chrome.wait_for("window.ready === true")
        await chrome.wait_for("!!document.querySelector('#go')")
        for height in SWEEP_HEIGHTS:
            for width in SWEEP_WIDTHS:
                size = dict(width=width, height=height, deviceScaleFactor=1, mobile=False)
                await chrome.call("Emulation.setDeviceMetricsOverride", **size)
                report = None
                for _ in range(
                    40
                ):  # the layout has settled when two reads agree, whatever the load
                    await asyncio.sleep(0.1)
                    read = await chrome.eval(SWEEP)
                    if read == report:
                        break
                    report = read
                reports[(width, height)] = report
    return reports


def inside(box: list[float], viewport: list[float]) -> bool:
    """Whether the ``[x, y, width, height]`` box lies whole in the window."""
    return (
        box[0] >= 0
        and box[1] >= 0
        and box[0] + box[2] <= viewport[0]
        and box[1] + box[3] <= viewport[1]
    )


def test_the_pause_panel_and_its_play_button_are_in_reach_in_every_window_of_a_mouse(
    session, tmp_path
):
    """Whatever the window (360 to 1100 wide, 400 to 800 high, the page of a mouse): the panel, its
    Play and Leave buttons lie whole in the window and are what a pointer at their middle reaches,
    the list of controls has room to show some of itself (the panel keeps at least 300 px, its list
    100), and where the frame rate and latency stay clickable above the panel the panel keeps clear
    of them; where there is no room beside them they go under the panel (`cramped`)."""
    reports = asyncio.run(sweep(session, tmp_path))
    assert len(reports) == len(SWEEP_WIDTHS) * len(SWEEP_HEIGHTS)
    for size, report in reports.items():
        viewport = report["viewport"]
        assert viewport == list(size), size
        for part in ("panel", "go", "leave"):
            assert inside(report[part], viewport), (size, part, report[part])
        assert report["goReached"] and report["leaveReached"], size
        assert report["panel"][3] >= min(300, size[1] - 40) and report["list"][3] >= 100, size
        if report["statsReached"]:
            assert not report["cramped"], size
            assert not intersects(report["panel"], report["stats"]), size
    # the roomy windows keep the stats above the panel, the narrow and low ones give it the window
    assert reports[(1100, 800)]["statsReached"] and not reports[(1100, 800)]["cramped"]
    assert reports[(683, 600)]["cramped"] or reports[(683, 600)]["statsReached"]
    assert any(report["cramped"] for report in reports.values())
