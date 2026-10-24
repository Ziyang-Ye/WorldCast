"""Looking and playing, in headless Chrome: the mouse, the arrow keys, a
gamepad (of the standard mapping, or by the layout of an Xbox, DualShock 4 or DualSense pad), the
on-screen pad, the mouse look switch and the pointer lock, the keys and mouse buttons the page
takes, the inspect button, which holds with a knife only, and the rate of the samples.

The page's own script (``static/play.js``) runs on the welcome and a few frames of a recorded mock
session (``play_input.html``) and is driven as a person drives it: keys held for a second with a
keyboard's press and release, the mouse's moves, its press and its release a tenth of a second
later, fingers on the pad (several at once). What the page sends is what the worker gets: an input
sample at every animation frame of a display of 60 Hz or less (at most 60 a second on a faster
one, and at once when a button or the weapon changes), with the buttons held and the degrees
turned since the last sample, pitch then yaw (yaw > 0 turns left, pitch > 0 looks down). The mouse
turns the view as far as it moves, as in the game: its turn is sent whole, with no cap, but for the
first movement after the pointer lock takes hold; the arrow keys and the sticks turn at a rate, at
most the maximum turn speed, so the samples' turn is checked against those numbers, frame by
frame. A view turned for a time is checked to within the longest frame time the page saw. The tests
skip without Chrome or Chromium, and on a CI runner (``CI`` set), which does not keep up with them
in real time.
"""

import asyncio
import json
import math
import re
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from tests.demo.support import chrome_page, needs_chrome, realtime, recorded_play
from worldcast.data.controls import CONTROL_BUTTONS

HARNESS = Path(__file__).with_name("play_input.html")
pytestmark = [needs_chrome, realtime]

BIT = {name: 1 << i for i, name in enumerate(CONTROL_BUTTONS)}
CLOSE = pytest.approx
PITCH_SHARE = 0.8
#: The rate sources (``static/input.js``): the arrow keys and the sticks turn at most 240 degrees a
#: second of yaw and 0.8 of that of pitch (the Max turn speed setting); a held arrow key turns 120
#: degrees a second (96 for the pitch). The mouse has no cap.
MAX_YAW, MAX_PITCH = 240.0, 192.0
ARROW_YAW, ARROW_PITCH = 120.0, 96.0
#: The mouse's sensitivity until the player sets one, degrees per pixel (the worker's
#: ``play.mouse_degrees_per_pixel``).
MOUSE_DEGREES_PER_PIXEL = 0.03
#: The sticks: a direction held beyond 0.4 of the travel; a look that asks for 25 degrees a second
#: just past the dead zone (0.12) and for the maximum turn speed at full deflection (the square of
#: the deflection past the dead zone between them).
FLOOR = 25.0
HALF_DEFLECTION = FLOOR + (MAX_YAW - FLOOR) * ((0.5 - 0.12) / 0.88) ** 2  # 65.1
#: A turn after a rest this long is a new input (``TURN_ONSET_GAP_MS``, in ``static/input.js``).
TURN_ONSET_GAP_S = 0.15
#: The tolerance of a rate read back from a sample (four decimals of a degree over a frame).
RATE_TOLERANCE = 0.05
#: A rate of a plateau is measured over windows of at least this long (s), and counts as at the
#: plateau within this share of it.
WINDOW_S, PLATEAU = 0.1, 0.03
#: Fewer windows than this in a push is a page the machine starved of frames.
STARVED = 4


@pytest.fixture(scope="module")
def recorded() -> dict:
    """A recorded mock session (three frames) for the page to play."""
    return asyncio.run(recorded_play(3))


@asynccontextmanager
async def play(
    recorded: dict,
    directory: Path,
    window=(1000, 700, 1),
    touch=False,
    storage: dict | None = None,
    gamepad_api: bool = True,
    pointer_lock: bool = True,
    refresh: int | None = None,
):
    """The play view in a Chrome of ``window``, its localStorage holding ``storage``, the motion
    feedback off (these tests time the turn by the animation frames of a software 2D canvas, which
    a move redrawn at every frame would stretch; ``test_motion_feedback.py`` runs it) and Walk by
    default off (a movement key holds just its button here; ``test_walk.py`` runs it), waiting at
    the pause panel (``gamepad_api`` false: a browser without ``navigator.getGamepads``;
    ``pointer_lock`` false: one without ``requestPointerLock``; ``refresh``: a display of that
    many Hz, whose animation frames come from a timer)."""
    setup = {
        **recorded,
        "storage": {"wc.motion": '"off"', "wc.walk": "false", **(storage or {})},
        "gamepadApi": gamepad_api,
        "pointerLock": pointer_lock,
        "refresh": refresh,
    }
    page = HARNESS.read_text(encoding="utf-8").replace("__SETUP__", json.dumps(setup))
    async with chrome_page(page, window, directory, touch=touch) as chrome:
        await chrome.wait_for("window.ready === true")
        assert (
            await chrome.eval("window.harness ? (harness.out.error ?? null) : 'no harness'") is None
        )
        await chrome.wait_for("!!document.querySelector('#go')")
        await asyncio.sleep(0.1)
        yield chrome


async def engage(chrome, touch=False) -> None:
    """Press Play, with the mouse or a finger, and wait until the pause panel is gone and the page
    has drawn the frames that follow."""
    x, y = await chrome.eval("harness.centre('#go')")
    await (chrome.tap(x, y) if touch else chrome.click(x, y))
    await chrome.wait_for("document.querySelector('#overlay').hidden")
    await asyncio.sleep(0.5)


def at_rate(rates: list[float], rate: float, top: float | None = None) -> bool:
    """Whether the turn measured, a rate over each window of at least 100 ms that turned
    (:func:`sent`), is ``rate`` degrees a second at its plateau: a third of the windows at it or
    above it by no more than the cap (not under it by more than 3 percent), the windows of the
    start and the end taking the rest, and none above ``top``, the rate sources' cap for the axis
    (``rate`` itself where the source asks for the cap); the turn made in all, checked by the
    tests that call this, is the turn asked for. A page the machine starved (fewer than 4 windows
    in a push: its frames ran over 0.25 s and are no measure of a rate; the turn made in all,
    which the callers check, still is) is held to the cap alone. Returns True or fails with the
    rates."""
    top = abs(rate) if top is None else top
    near = [m for m in rates if m * rate > 0 and abs(m) >= (1 - PLATEAU) * abs(rate)]
    assert max(abs(measured) for measured in rates) <= top + RATE_TOLERANCE, (rate, top, rates)
    assert len(near) >= len(rates) / 3 or len(rates) < STARVED, (rate, rates)
    return True


def keeps_coming(measured: dict) -> bool:
    """Whether the input samples (:func:`sent`) kept coming: one at about every animation frame the
    page saw (a display of 60 Hz or less sends one at each; headless Chrome under load runs fewer
    frames, not fewer samples per frame)."""
    return measured["count"] >= 0.5 * measured["frames"] and (
        measured["count"] or not measured["frames"]
    )


def within_caps(profile: list[list[float]]) -> bool:
    """Whether no sample of a profile (:func:`sent`) commands more than the caps: 240 degrees a
    second of yaw, 192 of pitch."""
    return all(
        abs(pitch) <= MAX_PITCH + RATE_TOLERANCE and abs(yaw) <= MAX_YAW + RATE_TOLERANCE
        for pitch, yaw, _ in profile
    )


def made(measured: dict, axis: str, rate: float) -> bool:
    """Whether a window of a push (:func:`sent`, its ``seconds``) holds the turn ``rate`` makes
    over it, to within a frame's turn."""
    asked = abs(rate) * measured["seconds"]
    slack = abs(rate) * measured["frame"] + 0.5
    return measured[axis] * rate > 0 and asked - slack <= abs(measured[axis]) <= asked + slack


def steady(profile: list[list[float]], axis: int, rate: float) -> bool:
    """Whether every sample of a profile (:func:`sent`) that turns on ``axis`` turns at ``rate``
    over the frames it covers, to within a few percent (a frame's time as the page and the harness
    read it), and the last one turns no more: a rate source turns at its rate at once, with no ramp,
    and stops at once."""
    turning = [row[axis] for row in profile if row[axis]]
    return (
        len(turning) >= 4
        and all(abs(r - rate) <= 0.05 * abs(rate) + RATE_TOLERANCE for r in turning)
        and profile[-1][axis] == 0
    )


def windows(steady: list[dict], axis: str) -> list[float]:
    """The rate of the turn ``axis`` (``dp``, ``dy``) over consecutive windows of the samples, each
    at least :data:`WINDOW_S` long, of those that turned: the sum of their turns over the sum of
    their times (a rate over one sample's time would read a frame that ran short as a fast turn)."""
    rates: list[float] = []
    turned = seconds = 0.0
    for sample in steady:
        turned, seconds = turned + sample[axis], seconds + sample["step"]
        if seconds >= WINDOW_S:
            if turned:
                rates.append(turned / seconds)
            turned = seconds = 0.0
    return rates


async def sent(chrome, since: float, until: float | None = None) -> dict:
    """What the input samples sent from the page's time ``since`` (``until``) hold: the degrees
    turned (``pitch``, ``yaw``), the largest yaw of one sample, the rate of every sample in order
    (``profile``: pitch, yaw, the seconds of the frames it covers), the rate over each window of
    100 ms or more (``yaw_rates``, ``pitch_rates``), the buttons' masks (``bits``, in order of
    first appearance, and ``last``, of the last sample), the weapon ids, the number of samples and
    of animation frames, the inputs
    the latency meter was told began (``onsets``, the page's time of each) and the longest frame
    time (s) of the page since."""
    samples = await chrome.eval(f"harness.samples({since})")
    samples = [s for s in samples if until is None or s["at"] <= until]
    onsets = await chrome.eval(f"harness.onsets({since})")
    bits = list(dict.fromkeys(s["b"] for s in samples))
    # the rate of every sample that turned, over the frames it covers, from the frame of the
    # sample before it: a frame that sends no sample (the sample rate's credit, a frame that came
    # early) leaves its turn to the next one, so one frame's time is not the sample's; a span with
    # a stalled frame, over 0.25 s, turns for 0.25 s of it: no rate
    steady = [s for s in samples if s["step"] is not None and s["step"] <= 0.25]
    return {
        "profile": [[s["dp"] / s["step"], s["dy"] / s["step"], s["step"]] for s in steady],
        "yaw_rates": windows(steady, "dy"),
        "pitch_rates": windows(steady, "dp"),
        "pitch": sum(s["dp"] for s in samples),
        "yaw": sum(s["dy"] for s in samples),
        "largest": max((abs(s["dy"]) for s in samples), default=0.0),
        "bits": bits,
        "last": samples[-1]["b"] if samples else None,
        "weapons": list(dict.fromkeys(s["w"] for s in samples)),
        "count": len(samples),
        "frames": await chrome.eval(f"harness.animation({since})"),
        "onsets": [t for o in onsets if until is None or o["now"] <= until for t in o["began"]],
        "frame": await chrome.eval(f"harness.gaps({since})") / 1000,
    }


async def hold_keys(chrome, codes: list[str], seconds=1.0, during=None) -> dict:
    """Hold the keys ``codes`` for ``seconds`` s (``during``: instead, an async function run 0.2 s
    after the press, and the release 0.1 s after it), then what the page sent meanwhile
    (:func:`sent`), ``seconds``: how long the page saw the keys held, and ``taken``: whether it
    took each key's press (its default prevented, so the page does not scroll)."""
    since = await chrome.eval("harness.now()")
    for code in codes:
        await chrome.key(code, True)
    if during is None:
        await asyncio.sleep(seconds)
    else:
        await asyncio.sleep(0.2)
        await during(chrome)
        await asyncio.sleep(0.1)
    for code in codes:
        await chrome.key(code, False)
    await asyncio.sleep(0.4)  # the turn comes to rest over a quarter of a second at most
    keys = await chrome.eval(f"harness.keys.filter((k) => k.t >= {since})")
    now = await chrome.eval("harness.now()")
    down = min((k["t"] for k in keys if k["type"] == "keydown"), default=since)
    up = max((k["t"] for k in keys if k["type"] == "keyup"), default=now)
    return {
        **await sent(chrome, since),
        "seconds": (up - down) / 1000,
        "taken": [k["taken"] for k in keys if k["type"] == "keydown"],
        "pressed": [k["t"] for k in keys if k["type"] == "keydown"],
    }


async def keyboard_session(recorded: dict, directory: Path) -> dict:
    """The arrow keys, one second each, in a Chrome of 1000 x 700."""
    report: dict = {}
    async with play(recorded, directory) as chrome:
        report["paused"] = await hold_keys(chrome, ["ArrowLeft"], seconds=0.5)
        await chrome.key("Escape", False)
        await engage(chrome)
        report["still"] = await hold_keys(chrome, [])
        for name, codes in {
            "left": ["ArrowLeft"],
            "right": ["ArrowRight"],
            "up": ["ArrowUp"],
            "down": ["ArrowDown"],
            "both": ["ArrowLeft", "ArrowRight"],
        }.items():
            report[name] = await hold_keys(chrome, codes)
        report["held"] = await hold_keys(
            chrome,
            ["ArrowUp", "KeyW"],
            during=lambda chrome: chrome.eval("JSON.stringify(harness.state().held)"),
        )
        report["stalled"] = await hold_keys(
            chrome, ["ArrowLeft"], during=lambda chrome: chrome.eval("harness.stall(800)")
        )
        # three presses: the second 20 ms after the first lets go, the third after a rest of 0.4 s
        since = await chrome.eval("harness.now()")
        for rest in (0.02, 0.4, 0.0):
            await chrome.key("ArrowLeft", True)
            await asyncio.sleep(0.3)
            await chrome.key("ArrowLeft", False)
            await asyncio.sleep(rest)
        await asyncio.sleep(0.4)
        report["presses"] = {
            **await sent(chrome, since),
            "pressed": await chrome.eval(
                "harness.keys.filter((k) => k.type === 'keydown' && "
                f"k.t >= {since}).map((k) => k.t)"
            ),
        }
        # focus lost with a key held: nothing is held, the view does not turn and is not covered
        await chrome.key("ArrowLeft", True)
        await asyncio.sleep(0.3)
        blurred = await chrome.eval(
            "(() => { window.dispatchEvent(new Event('blur')); return harness.now(); })()"
        )
        await asyncio.sleep(0.4)
        report["blur"] = {
            **await sent(chrome, blurred),
            "state": await chrome.eval("harness.state()"),
        }
        await chrome.key("ArrowLeft", False)
        # a click on the uncovered view plays again; Escape pauses with the panel
        x, y, width, height = await chrome.eval("harness.stage()")
        await chrome.click(x + width / 2, y + height / 2)
        report["refocus"] = await chrome.eval("harness.state()")
        await chrome.key("Escape", True)
        await chrome.key("Escape", False)
        await chrome.wait_for("!document.querySelector('#overlay').hidden")
        # the Turn speed slider, with the keyboard: 120 and two steps of 5
        await chrome.eval("document.querySelector('#turn').focus()")
        for _ in range(2):
            await chrome.key("ArrowRight", True)
            await chrome.key("ArrowRight", False)
        report["slider"] = await chrome.eval("harness.state()")
        await engage(chrome)
        report["fast"] = await hold_keys(chrome, ["ArrowLeft"])
        # the Max turn speed slider: 240 and twelve steps of 10 down, below the arrows' 130
        await chrome.key("Escape", True)
        await chrome.key("Escape", False)
        await chrome.wait_for("!document.querySelector('#overlay').hidden")
        await chrome.eval("document.querySelector('#maxturn').focus()")
        for _ in range(12):
            await chrome.key("ArrowLeft", True)
            await chrome.key("ArrowLeft", False)
        report["max_slider"] = await chrome.eval("harness.state()")
        await engage(chrome)
        report["capped"] = await hold_keys(chrome, ["ArrowLeft"])
        report["capped_pitch"] = await hold_keys(chrome, ["ArrowUp"])
    return report


@pytest.fixture(scope="module")
def keys(recorded, tmp_path_factory) -> dict:
    """What the page sent for each arrow key held (:func:`keyboard_session`)."""
    return asyncio.run(keyboard_session(recorded, tmp_path_factory.mktemp("keys")))


def test_an_arrow_held_for_a_second_turns_the_yaw_at_the_rate_within_a_frame(keys):
    """Left asks for 120 degrees a second of yaw, right the other way: the samples turn the view at
    that rate between a ramp up and a ramp down, and over the time the page saw the key held in all
    by what the key asked for, to within the longest frame time (the turn is integrated per
    animation frame). The press was taken, so the page does not scroll."""
    for name, sign in (("left", 1), ("right", -1)):
        measured = keys[name]
        assert at_rate(measured["yaw_rates"], sign * ARROW_YAW, MAX_YAW), name
        assert 0.95 <= measured["seconds"] <= 1.6, name
        assert measured["yaw"] == CLOSE(
            sign * ARROW_YAW * measured["seconds"], abs=ARROW_YAW * measured["frame"] + 0.5
        )
        assert measured["pitch"] == 0 and measured["taken"] == [True], name
        assert measured["bits"] == [0], name  # no button: the turn is not a key of the model


def test_an_arrow_turns_at_its_rate_from_its_press_to_its_release(keys):
    """Every frame the key is held turns the view at the key's rate, from the first to the last,
    and nothing turns after the key is released: no ramp, nothing left over."""
    for name, axis, rate in (
        ("left", 1, ARROW_YAW),
        ("right", 1, -ARROW_YAW),
        ("up", 0, -ARROW_PITCH),
        ("down", 0, ARROW_PITCH),
    ):
        assert steady(keys[name]["profile"], axis, rate), name


def test_a_turn_after_a_rest_is_one_input_that_began_with_its_key_press(keys):
    """The latency meter is told of an arrow's turn once, with the time of the key press that began
    it, not of the frame after (the first that turns): a turn held for a
    second is one onset, a press 20 ms after a release continues that turn, one after a rest of
    0.4 s is a new input. No key: no onset. Opposite arrows ask for nothing, but for the frame that
    may fall between the two presses (two events): at most that one onset."""
    for name in ("left", "right", "up", "down"):
        assert keys[name]["onsets"] == [CLOSE(keys[name]["pressed"][0], abs=0.01)], name
    assert len(keys["both"]["onsets"]) <= 1 and keys["still"]["onsets"] == []
    first, second, third = keys["presses"]["pressed"]
    assert keys["presses"]["onsets"] == [CLOSE(first, abs=0.01), CLOSE(third, abs=0.01)]
    assert second - first < 400 and third - second > 400


def test_without_a_key_there_is_no_drift_and_the_samples_keep_coming(keys):
    still = keys["still"]
    assert (still["pitch"], still["yaw"], still["bits"]) == (0, 0, [0])
    assert keeps_coming(still)  # samples keep coming while one plays


def test_the_pitch_turns_at_eight_tenths_of_the_yaw_rate(keys):
    """Up looks up (pitch down), down looks down: 96 degrees a second."""
    for name, sign in (("up", -1), ("down", 1)):
        measured = keys[name]
        assert at_rate(measured["pitch_rates"], sign * ARROW_PITCH, MAX_PITCH), name
        assert measured["yaw_rates"] == [], name
        assert measured["pitch"] == CLOSE(
            sign * ARROW_PITCH * measured["seconds"], abs=ARROW_PITCH * measured["frame"] + 0.5
        ), name
        assert measured["yaw"] == 0, name


def test_opposite_arrows_cancel_and_an_arrow_beside_a_button_turns_and_holds(keys):
    """The two presses are two events, so a frame may see one of them alone: that frame's turn
    (120 degrees a second over it) is all the net turn can be."""
    both = keys["both"]
    assert abs(both["yaw"]) <= ARROW_YAW * both["frame"] + 0.01 and both["pitch"] == 0
    held = keys["held"]
    assert held["pitch"] < -2 and BIT["forward"] in held["bits"]


def test_a_stalled_frame_turns_the_view_for_a_quarter_of_a_second_at_most(keys):
    """The page is busy for 0.8 s with the key held: the frame after it turns what 0.25 s of the
    key does (30 degrees), not 96; the turn over the whole press is half of what an uncapped one
    would be."""
    stalled = keys["stalled"]
    assert 0 < stalled["largest"] <= ARROW_YAW * 0.25 + 0.01
    assert max(abs(rate) for rate in stalled["yaw_rates"]) <= MAX_YAW + RATE_TOLERANCE
    assert 40 < stalled["yaw"] < 96


def test_the_arrows_do_nothing_while_the_game_is_paused_and_the_page_does_not_take_them(keys):
    """A paused page sends no samples, so an arrow turns nothing, and leaves the key to the page
    (the slider's own); the arrows of a player who plays are taken."""
    assert keys["paused"]["count"] == 0 and keys["paused"]["taken"] == [False]
    assert keys["left"]["taken"] == [True]


def test_a_window_that_loses_the_focus_stands_with_nothing_held_and_the_view_uncovered(keys):
    """Another window of the room is where its person plays: the view stays clear, says so, and a
    click on it plays again."""
    blur, refocus = keys["blur"], keys["refocus"]
    assert blur["yaw"] == 0 and blur["state"]["held"] == []
    assert not blur["state"]["card"] and blur["state"]["unfocused"]
    assert not refocus["card"] and not refocus["unfocused"]


def test_the_turn_speed_slider_sets_the_arrows_rate_and_is_kept(keys):
    """Two steps of 5 on the slider (default 120, from 20 to 320) make 130 degrees a second, shown
    and kept in localStorage; an arrow held after Play asks for it."""
    slider = keys["slider"]
    assert (slider["turn"], slider["turnLabel"]) == ("130", "130°/s")
    assert slider["maxTurn"] == "240" and slider["maxTurnLabel"] == "240°/s"
    assert slider["store"] == {
        "wc.arrowturn": "130",
        "wc.turncap": None,
        "wc.mouselook": None,
        "wc.sens": None,
    }
    fast = keys["fast"]
    assert at_rate(fast["yaw_rates"], 130.0, MAX_YAW)
    assert fast["yaw"] == CLOSE(130.0 * fast["seconds"], abs=130.0 * fast["frame"] + 0.5)


def test_the_max_turn_speed_slider_caps_every_source_and_is_kept(keys):
    """Twelve steps of 10 down from 240 make 120 degrees a second, shown and kept in localStorage:
    the arrow key is set to 130 (the pitch to 104), the view turns at 120 (96 for the pitch), and
    no turn is asked for beyond that to be caught up with after the key is released."""
    slider = keys["max_slider"]
    assert (slider["maxTurn"], slider["maxTurnLabel"]) == ("120", "120°/s")
    assert slider["store"]["wc.turncap"] == "120" and slider["store"]["wc.arrowturn"] == "130"
    capped = keys["capped"]
    assert at_rate(capped["yaw_rates"], 120.0)
    assert capped["yaw"] == CLOSE(120.0 * capped["seconds"], abs=120.0 * capped["frame"] + 0.5)
    assert at_rate(keys["capped_pitch"]["pitch_rates"], -96.0)


async def mouse_session(recorded: dict, directory: Path) -> dict:
    """The mouse look on (the default of a fine pointer): a move, wide moves of the real mouse,
    events that report a jump and a pile of events in one task, then the switch off."""
    report: dict = {}
    async with play(recorded, directory) as chrome:
        report["start"] = await chrome.eval("harness.state()")
        await engage(chrome)
        report["locks_on"] = await chrome.eval("harness.locks()")
        await chrome.move(500, 500)
        await asyncio.sleep(1.0)
        since = await chrome.eval("harness.now()")
        await chrome.move(600, 550)
        await asyncio.sleep(0.8)
        report["moved_on"] = await sent(chrome, since)
        # wide moves of the real mouse, to the edges of the window and back: events of up to 960 px
        since = await chrome.eval("harness.now()")
        for x in (20, 980, 20, 980, 500):
            await chrome.move(x, 350)
            await asyncio.sleep(0.03)
        await asyncio.sleep(3.0)
        report["wide"] = await sent(chrome, since)
        # one event that reports 20,000 pixels right and 9,000 up: a pointer that jumps
        since = await chrome.eval("harness.now()")
        await chrome.eval("harness.burst(20000, -9000)")
        await asyncio.sleep(3.0)
        report["jump"] = await sent(chrome, since)
        # twelve events in one task, each 1,000 pixels left and down: 360 degrees of each asked for
        since = await chrome.eval("harness.now()")
        await chrome.eval(
            "(() => { for (let i = 0; i < 12; i += 1) harness.burst(-1000, 1000); })()"
        )
        await asyncio.sleep(1.0)
        report["pile"] = await sent(chrome, since)
        # a lock that takes hold, a jump of the pointer to it, then a move
        since = await chrome.eval("harness.now()")
        await chrome.eval("harness.lock(); harness.burst(3000, -800)")
        await asyncio.sleep(0.3)
        report["locked_jump"] = await sent(chrome, since)
        since = await chrome.eval("harness.now()")
        await chrome.eval("harness.burst(100, 0)")
        await asyncio.sleep(0.3)
        report["after_lock"] = await sent(chrome, since)
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(0.6)
        report["rest"] = await sent(chrome, since)
        await chrome.key("Escape", False)  # a key's release: nothing
        await chrome.key("Escape", True)
        await chrome.key("Escape", False)
        await chrome.wait_for("!document.querySelector('#overlay').hidden")
        # the switch, clicked: off, kept, and the mouse's sensitivity is no setting then
        sx, sy = await chrome.eval("harness.centre('.switch')")
        await chrome.click(sx, sy)
        await asyncio.sleep(0.1)
        report["switched_off"] = await chrome.eval("harness.state()")
        await engage(chrome)
        report["locks_off"] = await chrome.eval("harness.locks()")
        report["cursor"] = await chrome.eval(
            "getComputedStyle(document.querySelector('#stage')).cursor"
        )
        since = await chrome.eval("harness.now()")
        for x, y in ((300, 300), (400, 350), (380, 330)):
            await chrome.move(x, y)
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)
        report["moved_off"] = await sent(chrome, since)
        await chrome.move(300, 300)
        # the stage clicked: the left button fires, the right one alt-fires
        x, y = await chrome.eval(
            "(() => { const [x, y, w, h] = harness.stage(); return [x + w / 2, y + h / 2]; })()"
        )
        for name, button in (("fire", "left"), ("alt", "right")):
            since = await chrome.eval("harness.now()")
            await chrome.click(x, y, hold=0.25, button=button)
            await asyncio.sleep(0.15)
            report[name] = await sent(chrome, since)
        # the switch, clicked again: on, kept; Play requests the pointer lock again
        await chrome.key("Escape", True)
        await chrome.key("Escape", False)
        await chrome.wait_for("!document.querySelector('#overlay').hidden")
        await chrome.click(sx, sy)
        await asyncio.sleep(0.1)
        report["switched_on"] = await chrome.eval("harness.state()")
        await engage(chrome)
        report["locks_back"] = await chrome.eval("harness.locks()")
    return report


@pytest.fixture(scope="module")
def mouse(recorded, tmp_path_factory) -> dict:
    return asyncio.run(mouse_session(recorded, tmp_path_factory.mktemp("mouse")))


def test_a_fine_pointer_looks_with_the_mouse_by_default_and_asks_for_the_pointer_lock(mouse):
    """The mouse look starts on: the switch is on, the sensitivity slider is live, nothing is kept
    in localStorage yet, and Play requests the pointer lock."""
    start = mouse["start"]
    assert start["switch"] is True and start["sensDisabled"] is False
    assert start["store"]["wc.mouselook"] is None
    assert mouse["locks_on"] == 1


def test_the_mouse_turns_the_view_by_the_welcomes_degrees_per_pixel(mouse, recorded):
    """The sensitivity is the worker's 0.03 degrees per pixel: 100 px right and 50 down is 3
    degrees right (yaw -3) and 1.5 down."""
    per_pixel = recorded["welcome"]["mouse_degrees_per_pixel"]
    assert per_pixel == MOUSE_DEGREES_PER_PIXEL
    assert mouse["moved_on"]["yaw"] == CLOSE(-100 * per_pixel, abs=0.01)
    assert mouse["moved_on"]["pitch"] == CLOSE(50 * per_pixel, abs=0.01)


def test_the_mouse_turns_the_view_as_far_as_it_moves_however_far_and_fast(mouse):
    """As in the game: wide moves of the real mouse (events of up to 960 px, 28.8 degrees), one
    event that reports 20,000 pixels and twelve events of 1,000 in one task are each sent whole,
    in the next sample, with no cap: the turn is the movement times the sensitivity, and it stops
    when the mouse does."""
    per_pixel = MOUSE_DEGREES_PER_PIXEL
    wide = mouse["wide"]  # from (600, 550) to (500, 350), by the edges of the window
    assert wide["yaw"] == CLOSE(100 * per_pixel, abs=0.01)
    assert wide["pitch"] == CLOSE(-200 * per_pixel, abs=0.01)
    # a sample carries a move's turn at once: far more than a rate source's 4 degrees a frame
    assert wide["largest"] > 10
    jump = mouse["jump"]
    assert jump["yaw"] == CLOSE(-20000 * per_pixel, abs=0.01)
    assert jump["pitch"] == CLOSE(-9000 * per_pixel, abs=0.01)
    assert jump["largest"] == CLOSE(20000 * per_pixel, abs=0.01)  # in one sample
    pile = mouse["pile"]
    assert pile["yaw"] == CLOSE(12 * 1000 * per_pixel, abs=0.01)
    assert pile["pitch"] == CLOSE(12 * 1000 * per_pixel, abs=0.01)
    for name in ("wide", "jump", "pile"):
        assert mouse[name]["profile"][-1][:2] == [0, 0], name


def test_the_first_movement_after_the_pointer_lock_takes_hold_turns_nothing(mouse):
    """A lock that takes hold may report the pointer's jump to it as a movement: the page drops
    the first movement after it, and the next turns the view as ever."""
    assert mouse["locked_jump"]["yaw"] == 0 and mouse["locked_jump"]["pitch"] == 0
    assert mouse["after_lock"]["yaw"] == CLOSE(-100 * MOUSE_DEGREES_PER_PIXEL, abs=0.01)


def test_a_mouse_at_rest_does_not_drift(mouse):
    rest = mouse["rest"]
    assert (rest["yaw"], rest["pitch"]) == (0, 0) and keeps_coming(rest)
    assert rest["onsets"] == []


def test_the_mouse_look_switch_turns_the_mouse_off_and_on_and_is_kept(mouse):
    """Off: kept in localStorage, the sensitivity slider is disabled, Play requests no further
    pointer lock and mouse movement turns nothing, the cursor stays an ordinary pointer (a click
    fires along the view's middle, where the page's crosshair is); on again, it is kept and the
    pointer lock is requested."""
    off = mouse["switched_off"]
    assert off["switch"] is False and off["sensDisabled"] is True
    assert off["store"]["wc.mouselook"] == "false"
    assert mouse["locks_off"] == 1 and mouse["cursor"] == "default"
    assert mouse["moved_off"]["yaw"] == 0 and mouse["moved_off"]["pitch"] == 0
    assert mouse["moved_off"]["count"] >= 5
    on = mouse["switched_on"]
    assert on["switch"] is True and on["sensDisabled"] is False
    assert on["store"]["wc.mouselook"] == "true"
    assert mouse["locks_back"] == 2


def test_with_the_mouse_look_off_the_mouse_buttons_still_fire(mouse):
    """A click on the stage holds attack (left) or attack2 (right) while it lasts and lets go."""
    for name, button in (("fire", "attack"), ("alt", "attack2")):
        assert set(mouse[name]["bits"]) == {0, BIT[button]} and mouse[name]["last"] == 0, name


#: The keys that carry the weapons and the markers, the page takes them while one plays: Ctrl+1-4
#: switch tabs in Chrome and Edge on Windows and Linux, and Firefox starts a find on a letter.
WEAPON_AND_MARKER_KEYS = ["Digit1", "Digit2", "Digit3", "Digit4", "KeyQ", "KeyM"]
#: DevTools' modifiers: Ctrl, and Meta (Cmd on a Mac, the Windows key).
CTRL, META = 2, 4


async def taken_session(recorded: dict, directory: Path) -> dict:
    """The keys the page takes and lets go, and the mouse buttons it takes: the weapon keys, Q and M
    with and without Ctrl, paused and playing; Cmd with keys held and pressed; the middle, back and
    forward buttons of a mouse."""
    report: dict = {}
    async with play(recorded, directory) as chrome:
        await chrome.eval("harness.keys.length = 0")
        for code in WEAPON_AND_MARKER_KEYS:
            await chrome.key(code, True)
            await chrome.key(code, False)
        report["paused"] = list(await chrome.eval("harness.keys"))
        await engage(chrome)
        await chrome.eval("harness.keys.length = 0")
        for modifiers in (0, CTRL):
            for code in WEAPON_AND_MARKER_KEYS:
                await chrome.key(code, True, modifiers)
                await chrome.key(code, False, modifiers)
            await asyncio.sleep(0.1)
        report["engaged"] = list(await chrome.eval("harness.keys"))
        report["weapon"] = (await chrome.eval("harness.state()"))["weapon"]
        # Cmd with W held: macOS sends the key's release to no one, so Cmd lets go of it; a key
        # pressed with Cmd held, W and an arrow, holds nothing and turns nothing
        since = await chrome.eval("harness.now()")
        await chrome.key("KeyW", True)
        await asyncio.sleep(0.3)
        report["before"] = await sent(chrome, since)
        report["lit_before"] = (await chrome.eval("harness.state()"))["held"]
        await chrome.key("MetaLeft", True, META)
        await asyncio.sleep(0.1)
        since = await chrome.eval("harness.now()")
        await chrome.key("KeyW", True, META)
        await chrome.key("ArrowLeft", True, META)
        await asyncio.sleep(0.5)
        report["cmd"] = await sent(chrome, since)
        report["lit_cmd"] = (await chrome.eval("harness.state()"))["held"]
        await chrome.key("ArrowLeft", False, META)
        await chrome.key("KeyW", False, META)
        await chrome.key("MetaLeft", False)
        # the keys work again once Cmd is up
        since = await chrome.eval("harness.now()")
        await chrome.key("KeyW", True)
        await asyncio.sleep(0.3)
        await chrome.key("KeyW", False)
        report["after"] = await sent(chrome, since)
        # the mouse's middle, back and forward buttons, then the left one
        x, y, width, height = await chrome.eval("harness.stage()")
        await chrome.eval("harness.buttons.length = 0")
        since = await chrome.eval("harness.now()")
        for button in ("middle", "back", "forward", "left"):
            # a back that the page did not take leaves it: the click then gets no answer
            click = chrome.click(x + width / 2, y + height / 2, hold=0.1, button=button)
            await asyncio.wait_for(click, 20)
        await asyncio.sleep(0.2)
        report["buttons"] = list(await chrome.eval("harness.buttons"))
        report["pressed"] = await sent(chrome, since)
        # paused again, the page leaves the middle button to the browser (back would leave the page)
        await chrome.key("Escape", True)
        await chrome.key("Escape", False)
        await chrome.wait_for("!document.querySelector('#overlay').hidden")
        await chrome.eval("harness.buttons.length = 0")
        await chrome.click(5, 5, hold=0.05, button="middle")
        report["buttons_paused"] = list(await chrome.eval("harness.buttons"))
    return report


@pytest.fixture(scope="module")
def taken(recorded, tmp_path_factory) -> dict:
    return asyncio.run(taken_session(recorded, tmp_path_factory.mktemp("taken")))


def test_the_weapon_keys_q_and_m_are_taken_while_one_plays_and_left_alone_paused(taken):
    """Ctrl+1-4 switch browser tabs on Windows and Linux, so with Ctrl held to crouch the weapon
    keys, Q and M are the page's, with Ctrl or without; paused the page leaves them to the
    browser."""
    assert [k["taken"] for k in taken["paused"]] == [False] * 12
    downs = [k for k in taken["engaged"] if k["type"] == "keydown"]
    assert [k["code"] for k in downs] == WEAPON_AND_MARKER_KEYS * 2
    assert all(k["taken"] for k in taken["engaged"])
    assert taken["weapon"].endswith("Knife")  # 4 and then Q: back to the third weapon


def test_cmd_lets_go_of_the_held_keys_and_a_key_pressed_with_it_holds_nothing(taken):
    """macOS delivers no keyup for the other keys while Cmd is down: W held, then Cmd, leaves W
    up at once (the sample, the key cap); W and an arrow pressed with Cmd held hold and turn
    nothing; once Cmd is up the keys work."""
    assert taken["before"]["last"] == BIT["forward"] and taken["lit_before"] == ["forward"]
    cmd = taken["cmd"]
    assert cmd["bits"] == [0] and keeps_coming(cmd) and taken["lit_cmd"] == []
    assert cmd["yaw"] == 0 and cmd["pitch"] == 0
    assert set(taken["after"]["bits"]) == {BIT["forward"], 0}


def test_the_middle_back_and_forward_buttons_are_taken_while_one_plays_and_fire_nothing(taken):
    """Back and forward leave the play view and drop the seat, the middle button scrolls on its
    own or pastes: every event of those buttons (down, up, auxclick) is the page's while one
    plays (the page is still there after a back click), and the browser's paused; the left button
    still fires."""
    events = {(b["type"], b["button"]): b["taken"] for b in taken["buttons"]}
    for button in (1, 3, 4):
        assert all(events[(kind, button)] for kind in ("mousedown", "mouseup", "auxclick")), button
    assert events[("mousedown", 0)] and events[("mouseup", 0)]
    assert set(taken["pressed"]["bits"]) == {0, BIT["attack"]} and taken["pressed"]["last"] == 0
    assert taken["buttons_paused"] and not any(b["taken"] for b in taken["buttons_paused"])


#: What the page says when the browser refuses the pointer lock.
LOCK_REFUSED = "The mouse was not captured: click the view to look with it"
RAW_LOCK = {"id": "stage", "options": {"unadjustedMovement": True}}
PLAIN_LOCK = {"id": "stage", "options": None}


async def inspect_session(recorded: dict, directory: Path) -> dict:
    """F with the AK-47 (the first weapon), F with the knife (key 3), F held through a switch from
    the knife to the AK-47, and F's cap and line of the legend with each."""
    report: dict = {}

    async def held_f(seconds: float = 0.3, during=None) -> dict:
        since = await chrome.eval("harness.now()")
        await chrome.key("KeyF", True)
        await asyncio.sleep(0.15)
        state = (await chrome.eval("harness.state()"))["inspect"]
        if during is not None:
            await chrome.key(during, True)
            await chrome.key(during, False)
        await asyncio.sleep(seconds)
        await chrome.key("KeyF", False)
        await asyncio.sleep(0.15)
        return {**await sent(chrome, since), "held": state}

    async with play(recorded, directory) as chrome:
        report["paused"] = (await chrome.eval("harness.state()"))["inspect"]
        await engage(chrome)
        report["gun"] = await held_f()
        await chrome.key("Digit3", True)
        await chrome.key("Digit3", False)
        await asyncio.sleep(0.1)
        report["knife_state"] = (await chrome.eval("harness.state()"))["inspect"]
        report["knife"] = await held_f()
        report["switched"] = await held_f(during="Digit1")
        await chrome.key("Escape", True)
        await chrome.key("Escape", False)
        await chrome.wait_for("!document.querySelector('#overlay').hidden")
        report["paused_gun"] = (await chrome.eval("harness.state()"))["inspect"]
    return report


@pytest.fixture(scope="module")
def inspect(recorded, tmp_path_factory) -> dict:
    return asyncio.run(inspect_session(recorded, tmp_path_factory.mktemp("inspect")))


def test_f_inspects_with_the_knife_only(inspect, recorded):
    """F with the AK-47 sends nothing (the samples hold no button) and its cap neither lights nor
    looks available; with the knife it holds the inspect button while it is down; held through a
    switch to the AK-47 it lets go with the switch. The worker would clear it anyway
    (``test_inputs.py``)."""
    ids = recorded["welcome"]["weapon_ids"]
    assert ids["knife"] in recorded["welcome"]["inspect_weapons"]
    assert ids["ak47"] not in recorded["welcome"]["inspect_weapons"]
    gun = inspect["gun"]
    assert gun["bits"] == [0] and gun["held"]["greyed"] and not gun["held"]["lit"]
    assert gun["weapons"] == [ids["ak47"]]
    knife = inspect["knife"]
    assert set(knife["bits"]) == {0, BIT["look_at_weapon"]} and knife["last"] == 0
    assert knife["held"]["lit"] and not knife["held"]["greyed"]
    assert knife["weapons"] == [ids["knife"]] and not inspect["knife_state"]["greyed"]
    switched = inspect["switched"]
    assert switched["bits"][-1] == 0 and BIT["look_at_weapon"] in switched["bits"]
    assert switched["weapons"] == [ids["knife"], ids["ak47"]]


def test_f_says_it_inspects_with_a_knife_and_is_greyed_with_another_weapon(inspect):
    """F's key cap is titled "F — inspect (knife)", and its line of the legend says "Inspect
    (knife)"; both are greyed while the weapon held is not a knife."""
    for name in ("paused", "paused_gun"):
        state = inspect[name]
        assert state["title"] == "F — inspect (knife)" and state["greyed"], name
        assert state["legend"] == [["F", True], ["Inspect (knife)", True]], name
    assert inspect["knife_state"]["title"] == "F — inspect (knife)"
    assert not inspect["knife_state"]["greyed"]


async def mouse_moved(chrome) -> dict:
    """What the page sends for a move of the real mouse, 100 px right and 30 down."""
    await chrome.move(300, 300)
    await asyncio.sleep(1.0)  # the turn the move to here asks for is made first
    since = await chrome.eval("harness.now()")
    await chrome.move(400, 330)
    await asyncio.sleep(0.5)
    return await sent(chrome, since)


async def pause_and_play(chrome, *answers: str) -> None:
    """Pause with Escape and press Play again, the browser answering the pointer lock requests
    with ``answers`` (:func:`play_input.html`'s lockPlan)."""
    await chrome.key("Escape", True)
    await chrome.key("Escape", False)
    await chrome.wait_for("!document.querySelector('#overlay').hidden")
    await chrome.eval(f"harness.lockPlan({json.dumps(list(answers))[1:-1]})")
    await engage(chrome)


async def lock_session(recorded: dict, directory: Path) -> dict:
    """The pointer lock the browser refuses: for raw movement only, for the plain request too
    (Chrome does for about a second after an Esc exit), and by the event alone (Firefox, Safari);
    then a click on the view."""
    report: dict = {}
    async with play(recorded, directory) as chrome:
        await chrome.eval("harness.lockPlan('reject')")
        await engage(chrome)
        report["raw_refused"] = {
            "requests": await chrome.eval("harness.lockRequests()"),
            "toasts": await chrome.eval("harness.toasts()"),
            "moved": await mouse_moved(chrome),
            "state": await chrome.eval("harness.state()"),
        }
        await pause_and_play(chrome, "reject", "reject")
        report["refused"] = {
            "requests": await chrome.eval("harness.lockRequests()"),
            "toasts": await chrome.eval("harness.toasts()"),
            "moved": await mouse_moved(chrome),
            "arrow": await hold_keys(chrome, ["ArrowLeft"]),
            "state": await chrome.eval("harness.state()"),
        }
        x, y, width, height = await chrome.eval("harness.stage()")
        await chrome.click(x + width / 2, y + height / 2)
        await asyncio.sleep(0.2)
        report["clicked"] = {
            "requests": await chrome.eval("harness.lockRequests()"),
            "toasts": await chrome.eval("harness.toasts()"),
            "moved": await mouse_moved(chrome),
        }
        await pause_and_play(chrome, "event")
        report["event"] = {
            "requests": await chrome.eval("harness.lockRequests()"),
            "toasts": await chrome.eval("harness.toasts()"),
            "moved": await mouse_moved(chrome),
            "state": await chrome.eval("harness.state()"),
        }
        await chrome.click(x + width / 2, y + height / 2)
        await asyncio.sleep(0.2)
        report["event_clicked"] = {
            "requests": await chrome.eval("harness.lockRequests()"),
            "toasts": await chrome.eval("harness.toasts()"),
            "moved": await mouse_moved(chrome),
        }
    return report


@pytest.fixture(scope="module")
def lock(recorded, tmp_path_factory) -> dict:
    return asyncio.run(lock_session(recorded, tmp_path_factory.mktemp("lock")))


def test_a_lock_refused_for_raw_movement_is_asked_for_again_plainly_and_says_nothing(lock):
    """A system without raw input rejects the first request (and Chromium fires the event):
    the plain one is made, granted, the mouse looks and no notice is shown."""
    refused = lock["raw_refused"]
    assert refused["requests"] == [RAW_LOCK, PLAIN_LOCK]
    assert LOCK_REFUSED not in refused["toasts"]
    assert refused["moved"]["yaw"] == CLOSE(-100 * MOUSE_DEGREES_PER_PIXEL, abs=0.01)


def test_a_lock_the_browser_refuses_leaves_the_keyboard_playing_and_says_so_once(lock):
    """Chrome refuses a lock within a second of an Esc exit, for the plain request too: the
    player is told once (the promise and the event both say it), is not covered by the pause
    panel, plays with the keyboard (an arrow turns the view at its rate, the mouse turns
    nothing) and the mouse look is still on in the settings."""
    refused = lock["refused"]
    assert refused["requests"][2:] == [RAW_LOCK, PLAIN_LOCK]
    assert refused["toasts"].count(LOCK_REFUSED) == 1
    assert not refused["state"]["card"] and refused["state"]["switch"] is True
    assert refused["moved"]["yaw"] == 0 and refused["moved"]["pitch"] == 0
    assert keeps_coming(refused["moved"])
    assert at_rate(refused["arrow"]["yaw_rates"], ARROW_YAW, MAX_YAW)


def test_a_click_on_the_view_asks_for_the_lock_again_and_the_mouse_looks_once_it_is_granted(lock):
    clicked = lock["clicked"]
    assert clicked["requests"][4:] == [RAW_LOCK]
    assert clicked["toasts"].count(LOCK_REFUSED) == 1
    assert clicked["moved"]["yaw"] == CLOSE(-100 * MOUSE_DEGREES_PER_PIXEL, abs=0.01)


def test_a_browser_that_says_a_refusal_by_the_event_alone_gets_one_request_and_one_notice(lock):
    """Firefox and Safari return no promise: the event is the answer, and there is no plain
    request to follow it."""
    event = lock["event"]
    assert event["requests"][5:] == [RAW_LOCK]
    assert event["toasts"].count(LOCK_REFUSED) == 2
    assert event["moved"]["yaw"] == 0 and not event["state"]["card"]
    clicked = lock["event_clicked"]
    assert clicked["requests"][6:] == [RAW_LOCK] and clicked["toasts"].count(LOCK_REFUSED) == 2
    assert clicked["moved"]["yaw"] == CLOSE(-100 * MOUSE_DEGREES_PER_PIXEL, abs=0.01)


async def settings_session(recorded: dict, directory: Path) -> dict:
    """Settings kept in the browser before the page runs: out of range, and the mouse look off;
    then a browser that cannot lock the pointer."""
    report: dict = {}
    kept = {"wc.arrowturn": "5", "wc.turncap": "500", "wc.mouselook": "false", "wc.sens": "0.1"}
    async with play(recorded, directory, storage=kept) as chrome:
        report["card"] = await chrome.eval("harness.state()")
        await engage(chrome)
        report["locks"] = await chrome.eval("harness.locks()")
        await chrome.move(300, 300)
        await asyncio.sleep(0.1)
        since = await chrome.eval("harness.now()")
        await chrome.move(400, 330)
        await asyncio.sleep(0.5)
        report["moved"] = await sent(chrome, since)
        report["lowest"] = await hold_keys(chrome, ["ArrowLeft"])
    kept = {"wc.mouselook": "true"}
    async with play(recorded, directory / "unlockable", storage=kept, pointer_lock=False) as chrome:
        report["unlockable"] = await chrome.eval("harness.state()")
        await engage(chrome)
        report["unlockable_locks"] = await chrome.eval("harness.locks()")
        report["unlockable_moved"] = await mouse_moved(chrome)
    return report


@pytest.fixture(scope="module")
def settings(recorded, tmp_path_factory) -> dict:
    directory = tmp_path_factory.mktemp("settings")
    (directory / "unlockable").mkdir()
    return asyncio.run(settings_session(recorded, directory))


def test_the_settings_kept_in_the_browser_are_read_within_their_ranges(settings):
    """A turn speed under the range's low end is its low end (20), a max turn speed over its high
    end the high end (320); the mouse look the player turned off stays off, with the player's
    sensitivity, and the arrow key turns at the 20 (under the max): no pointer lock, no mouse turn.
    """
    card = settings["card"]
    assert (card["turn"], card["maxTurn"]) == ("20", "320")
    assert card["switch"] is False and card["sensDisabled"] is True
    assert settings["locks"] == 0
    assert settings["moved"]["yaw"] == 0 and settings["moved"]["pitch"] == 0
    assert at_rate(settings["lowest"]["yaw_rates"], 20.0, MAX_YAW)


def test_a_browser_that_cannot_lock_the_pointer_has_the_mouse_look_off_whatever_was_kept(settings):
    """iPadOS Safari has no pointer lock: the mouse look is off even where a setting kept in the
    browser says on, the switch is there, off and disabled (and the setting is left as it was);
    Play requests nothing and the mouse turns nothing."""
    unlockable = settings["unlockable"]
    assert unlockable["switch"] is False and unlockable["switchDisabled"] is True
    assert unlockable["sensDisabled"] is True and unlockable["store"]["wc.mouselook"] == "true"
    assert settings["unlockable_locks"] == 0
    assert settings["unlockable_moved"]["yaw"] == 0 and settings["unlockable_moved"]["pitch"] == 0


async def pushed(chrome, axes: list[float], pressed=(), seconds=1.0, rest=(0, 0, 0, 0)) -> dict:
    """Push the gamepad's sticks to ``axes`` and press its buttons ``pressed`` for ``seconds`` s,
    then what the page sent from the push on (:func:`sent`) and how long the page saw it
    (``seconds``). The pad is let go to ``rest``."""
    start = await chrome.eval(f"harness.pad.set({axes}, {list(pressed)})")
    await asyncio.sleep(seconds)
    end = await chrome.eval(f"harness.pad.set({list(rest)})")
    await asyncio.sleep(0.45)  # the turn comes to rest over a third of a second at most
    return {**await sent(chrome, start), "seconds": (end - start) / 1000, "start": start}


async def gamepad_session(recorded: dict, directory: Path) -> dict:
    report: dict = {}
    async with play(recorded, directory) as chrome:
        report["paused"] = await pushed(chrome, [0, 0, 1, 0], seconds=0.4)
        await engage(chrome)
        await asyncio.sleep(0.5)
        report["polls_alone"] = await chrome.eval("harness.pad.polls()")
        await chrome.eval("harness.pad.connect()")
        await chrome.eval("harness.pad.connect()")
        report["toasts"] = await chrome.eval("harness.toasts()")
        report["dead"] = await pushed(chrome, [0, 0, 0.1, 0])
        report["floor"] = await pushed(chrome, [0, 0, 0.13, 0])
        report["half"] = await pushed(chrome, [0, 0, 0.5, 0])
        report["full"] = await pushed(chrome, [0, 0, -1, 0])
        report["down"] = await pushed(chrome, [0, 0, 0, 1])
        report["up_diagonal"] = await pushed(chrome, [0, 0, 0.6, -0.6])
        report["forward"] = await pushed(chrome, [0, -1, 0, 0])
        report["diagonal"] = await pushed(chrome, [0.5, -0.5, 0, 0])
        report["weak"] = await pushed(chrome, [0.3, -0.3, 0, 0])
        for name, index in {
            "jump": 0,
            "duck": 1,
            "reload": 2,
            "look_at_weapon": 3,
            "attack": 7,
            "attack2": 6,
        }.items():
            report[name] = await pushed(chrome, [0, 0, 0, 0], pressed=[index], seconds=0.3)
        # Y again with the knife (key 3), then the AK-47 again (key 1)
        for code in ("Digit3", "Digit1"):
            await chrome.key(code, True)
            await chrome.key(code, False)
            if code == "Digit3":
                report["inspect_knife"] = await pushed(
                    chrome, [0, 0, 0, 0], pressed=[3], seconds=0.3
                )
        report["weapon0"] = (await chrome.eval("harness.state()"))["weapon"]
        report["next"] = await pushed(chrome, [0, 0, 0, 0], pressed=[5], seconds=0.3)
        report["weapon1"] = (await chrome.eval("harness.state()"))["weapon"]
        report["previous"] = await pushed(chrome, [0, 0, 0, 0], pressed=[4], seconds=0.3)
        report["weapon_back"] = (await chrome.eval("harness.state()"))["weapon"]
        report["dpad_left"] = await pushed(chrome, [0, 0, 0, 0], pressed=[14], seconds=0.3)
        report["weapon_left"] = (await chrome.eval("harness.state()"))["weapon"]
        report["dpad_up"] = await pushed(chrome, [0, 0, 0, 0], pressed=[12], seconds=0.3)
        report["weapon_up"] = (await chrome.eval("harness.state()"))["weapon"]
        await chrome.eval("harness.pad.disconnect()")
        polls = await chrome.eval("harness.pad.polls()")
        report["gone"] = await pushed(chrome, [0, 0, 1, 0], seconds=0.4)
        report["polls_after"] = [polls, await chrome.eval("harness.pad.polls()")]
        report["toasts_after"] = await chrome.eval("harness.toasts()")
    return report


@pytest.fixture(scope="module")
def pad(recorded, tmp_path_factory) -> dict:
    return asyncio.run(gamepad_session(recorded, tmp_path_factory.mktemp("pad")))


def test_a_gamepad_is_polled_only_once_one_is_connected_and_its_connection_toasts_once(pad):
    """The page asks navigator.getGamepads once at the start (a pad that connected before the page)
    and not at every frame while there is none; the connection is said once, though the event came
    twice, and the page stops asking when the pad goes."""
    assert pad["polls_alone"] == 1
    assert pad["toasts"] == ["Test Pad connected"]
    assert pad["toasts_after"][-1] == "Test Pad disconnected"
    assert pad["polls_after"][0] == pad["polls_after"][1]


def test_the_right_stick_turns_from_a_floor_above_what_the_generator_follows_to_the_max_speed(pad):
    """Within the dead zone (0.1) nothing turns; just beyond it (0.13) the stick asks for 25
    degrees a second, at half deflection for 65.1, at full deflection for the maximum turn speed
    (240 degrees a second of yaw, and 192 of pitch, the pitch's cap): right is yaw down, down looks
    down, and the speed is the same whichever way the stick points, within the caps."""
    assert pad["dead"]["yaw"] == 0 and pad["dead"]["pitch"] == 0
    for name, axis, rate in (
        ("floor", "yaw", -FLOOR),
        ("half", "yaw", -HALF_DEFLECTION),
        ("full", "yaw", MAX_YAW),
        ("down", "pitch", MAX_PITCH),
    ):
        measured = pad[name]
        top = MAX_YAW if axis == "yaw" else MAX_PITCH
        assert at_rate(measured[f"{axis}_rates"], rate, top), name
        assert measured[axis] == CLOSE(
            rate * measured["seconds"], abs=abs(rate) * measured["frame"] + 0.5
        ), name
        assert pad[name]["bits"] == [0], name
    # (0.6, -0.6) is a deflection of 0.849, 172.4 degrees a second, 1 / sqrt(2) of it each way:
    # 121.9 on each axis, under both caps
    diagonal = pad["up_diagonal"]
    each = (FLOOR + (MAX_YAW - FLOOR) * ((math.hypot(0.6, 0.6) - 0.12) / 0.88) ** 2) / math.sqrt(2)
    assert each == CLOSE(121.9, abs=0.05) and each < MAX_PITCH
    assert at_rate(diagonal["yaw_rates"], -each, MAX_YAW)
    assert at_rate(diagonal["pitch_rates"], -each, MAX_PITCH)
    for axis, rate in (("yaw", each), ("pitch", each)):
        assert diagonal[axis] == CLOSE(
            -rate * diagonal["seconds"], abs=rate * diagonal["frame"] + 0.5
        ), axis


def test_a_stick_pushed_and_let_go_turns_at_its_rate_and_stops(pad):
    """A stick turns at the rate of its deflection, within the caps, and turns nothing once it is
    let go: no turn to catch up with."""
    for name in ("floor", "half", "full", "down", "up_diagonal"):
        profile = pad[name]["profile"]
        assert within_caps(profile), name
        assert profile[-1][:2] == [0, 0], name


def test_a_stick_turn_after_a_rest_is_one_input_for_the_latency_meter(pad):
    """Each push of the right stick starts a turn after a rest of 0.2 s or more: one onset, at the
    frame that turned first (within a frame or two of the push), whatever the deflection."""
    for name in ("floor", "half", "full", "down", "up_diagonal"):
        onsets = pad[name]["onsets"]
        assert len(onsets) == 1 or pad[name]["frame"] > TURN_ONSET_GAP_S, name
        assert pad[name]["start"] - 250 <= onsets[0] <= pad[name]["start"] + 400, name
    assert pad["dead"]["onsets"] == []  # within the dead zone: no turn, no input


def test_the_left_stick_holds_the_movement_buttons_in_eight_directions(pad):
    """Forward beyond 0.4 of its travel; forward and right in the diagonal; nothing at 0.3."""
    assert set(pad["forward"]["bits"]) == {0, BIT["forward"]}
    assert set(pad["diagonal"]["bits"]) == {0, BIT["forward"] | BIT["move_right"]}
    assert pad["weak"]["bits"] == [0]
    for name in ("forward", "diagonal"):
        assert pad[name]["yaw"] == 0 and pad[name]["pitch"] == 0


def test_the_gamepad_buttons_hold_the_models_buttons_while_they_are_down(pad):
    """A jump, B crouch, X reload, Y inspect (with the knife; with the AK-47 nothing), the right
    trigger fire, the left alt-fire."""
    for name in ("jump", "duck", "reload", "attack", "attack2"):
        assert set(pad[name]["bits"]) == {BIT[name], 0} and pad[name]["last"] == 0, name
    assert pad["look_at_weapon"]["bits"] == [0]
    knife = pad["inspect_knife"]
    assert set(knife["bits"]) == {BIT["look_at_weapon"], 0} and knife["last"] == 0


def test_the_bumpers_and_the_dpad_take_weapons(pad, recorded):
    """The right bumper takes the weapon after the one held and the left the one before; the D-pad's
    left (the fourth) and up (the first) pick by place: the loadout shows it and the samples carry
    its id."""
    assert pad["weapon0"].endswith("AK-47") and pad["weapon1"].endswith("Glock-18")
    assert pad["weapon_back"].endswith("AK-47")
    assert pad["weapon_left"].endswith("HE Grenade") and pad["weapon_up"].endswith("AK-47")
    ids = [
        recorded["welcome"]["weapon_ids"][name] for name in recorded["welcome"]["seat"]["loadout"]
    ]
    assert len(set(ids)) == 4
    assert pad["next"]["weapons"] == [ids[1]] and pad["previous"]["weapons"] == [ids[0]]
    assert pad["dpad_left"]["weapons"] == [ids[3]] and pad["dpad_up"]["weapons"] == [ids[0]]


def test_a_paused_page_and_a_gone_pad_turn_nothing(pad):
    assert pad["paused"]["count"] == 0
    assert pad["gone"]["yaw"] == 0 and pad["gone"]["pitch"] == 0


async def odd_pad_session(recorded: dict, directory: Path) -> dict:
    """A gamepad that has no standard mapping, and a browser that has no gamepad API."""
    report: dict = {}
    async with play(recorded, directory) as chrome:
        await engage(chrome)
        polls = await chrome.eval("harness.pad.polls()")
        await chrome.eval("harness.pad.connect('')")
        report["toasts"] = await chrome.eval("harness.toasts()")
        report["pushed"] = await pushed(chrome, [0.5, -1, 1, 0.5], pressed=[0, 1, 7], seconds=0.5)
        report["polls"] = [polls, await chrome.eval("harness.pad.polls()")]
        await chrome.eval("harness.pad.disconnect()")
        report["disconnected"] = await chrome.eval("harness.toasts()")
    async with play(recorded, directory / "none", gamepad_api=False) as chrome:
        report["no_api_card"] = await chrome.eval("harness.state()")
        await engage(chrome)
        report["no_api"] = await hold_keys(chrome, ["ArrowLeft"])
        report["no_api_toasts"] = await chrome.eval("harness.toasts()")
    return report


@pytest.fixture(scope="module")
def odd(recorded, tmp_path_factory) -> dict:
    directory = tmp_path_factory.mktemp("odd")
    (directory / "none").mkdir()
    return asyncio.run(odd_pad_session(recorded, directory))


def test_a_pad_without_the_standard_mapping_is_said_not_to_be_used_and_not_polled(odd):
    """Its connection says so, nothing of its sticks or buttons reaches the samples, and the page
    does not ask for it at every frame."""
    assert odd["toasts"] == ["Test Pad has no standard mapping: not used"]
    pushed_ = odd["pushed"]
    assert keeps_coming(pushed_) and pushed_["bits"] == [0]
    assert pushed_["yaw"] == 0 and pushed_["pitch"] == 0
    assert odd["polls"][0] == odd["polls"][1]
    assert odd["disconnected"][-1] == "Test Pad disconnected"


def test_a_browser_without_the_gamepad_api_still_plays_with_the_keyboard(odd):
    """A page served over plain http from a host that is not localhost has no navigator.getGamepads
    in some browsers: the play view still renders, Play starts, samples go out and an arrow key
    turns the view at its rate."""
    assert odd["no_api_card"]["card"]
    measured = odd["no_api"]
    assert keeps_coming(measured)
    assert at_rate(measured["yaw_rates"], ARROW_YAW, MAX_YAW)
    assert odd["no_api_toasts"] == []


#: What a pad does, by action: the axes it moves to (index -> value) and the buttons it presses.
#: Pads of no standard mapping as Linux drivers report them: an Xbox pad through xpad (axes left x,
#: y, left trigger, right x, y, right trigger, D-pad x, y; the triggers rest at -1), a DualShock 4
#: (left x, y, right x, left trigger, right trigger, right y; a hat on axis 9 that rests at 9/7 and
#: reads -1 up, then 2/7 more for each direction clockwise) and a DualSense of eight axes (the
#: D-pad on axes 6 and 7).
Action = tuple[dict[int, float], list[int]]
XBOX_ID = "Microsoft X-Box 360 pad (Vendor: 045e Product: 028e)"
XBOX_REST = [0, 0, -1, 0, 0, -1, 0, 0]
XBOX: dict[str, Action] = {
    "first_weapon": ({7: -1}, []),
    "look_left": ({3: -1}, []),
    "look_down": ({4: 1}, []),
    "forward": ({1: -1}, []),
    "diagonal": ({0: 0.5, 1: -0.5}, []),
    "jump": ({}, [0]),
    "duck": ({}, [1]),
    "reload": ({}, [2]),
    "attack": ({5: 1}, []),
    "attack2": ({2: 1}, []),
    "unmoved_triggers": ({2: 0, 5: 0}, []),
    "back_and_start": ({}, [6, 7]),
    "next": ({}, [5]),
    "previous": ({}, [4]),
    "dpad_left": ({6: -1}, []),
    "dpad_up": ({7: -1}, []),
    "dpad_right": ({6: 1}, []),
    "dpad_down": ({7: 1}, []),
    "look_at_weapon": ({}, [3]),
}
DUALSHOCK_ID = "Wireless Controller (Vendor: 054c Product: 09cc)"
DUALSHOCK_REST = [0, 0, 0, -1, -1, 0, 0, 0, 0, 9 / 7]
DUALSHOCK: dict[str, Action] = {
    "first_weapon": ({9: -1}, []),
    "look_left": ({2: -1}, []),
    "look_down": ({5: 1}, []),
    "forward": ({1: -1}, []),
    "diagonal": ({0: 0.5, 1: -0.5}, []),
    "jump": ({}, [0]),
    "duck": ({}, [1]),
    "reload": ({}, [3]),
    "attack": ({4: 1}, []),
    "attack2": ({3: 1}, []),
    "attack_button": ({}, [7]),
    "attack2_button": ({}, [6]),
    "unmoved_triggers": ({3: 0, 4: 0}, []),
    "next": ({}, [5]),
    "previous": ({}, [4]),
    "dpad_left": ({9: 5 / 7}, []),
    "dpad_up": ({9: -1}, []),
    "dpad_right": ({9: -3 / 7}, []),
    "dpad_down": ({9: 1 / 7}, []),
    "look_at_weapon": ({}, [2]),
    "dpad_up_left": ({9: 1}, []),
}
DUALSENSE_ID = "DualSense Wireless Controller (Vendor: 054c Product: 0ce6)"
DUALSENSE_REST = [0, 0, 0, -1, -1, 0, 0, 0]
DUALSENSE: dict[str, Action] = {
    "first_weapon": ({7: -1}, []),
    "look_left": ({2: -1}, []),
    "attack": ({4: 1}, []),
    "dpad_left": ({6: -1}, []),
    "dpad_up": ({7: -1}, []),
    "dpad_right": ({6: 1}, []),
    "dpad_down": ({7: 1}, []),
}
#: The weapon the loadout shows after the weapon actions (the D-pad's up first: the weapon before it
#: is the last pad's).
AFTER = {
    "first_weapon": "AK-47",
    "next": "Glock-18",
    "previous": "AK-47",
    "dpad_left": "HE Grenade",
    "dpad_up": "AK-47",
    "dpad_right": "Glock-18",
    "dpad_down": "Knife",
}


async def act(chrome, rest: list[float], actions: dict[str, Action]) -> dict:
    """Each of ``actions`` done on the gamepad in turn (a look for a second, the others for 0.3
    s), with what the page sent (:func:`pushed`) and the weapon it shows after."""
    report: dict = {}
    for name, (moved, pressed) in actions.items():
        axes = [moved.get(i, value) for i, value in enumerate(rest)]
        seconds = 1.0 if name.startswith("look") else 0.3
        report[name] = await pushed(chrome, axes, pressed, seconds, rest)
        report[name]["weapon"] = (await chrome.eval("harness.state()"))["weapon"]
    return report


async def raw_pad_session(recorded: dict, directory: Path) -> dict:
    """A pad with no standard mapping that the page knows by its vendor and product: an Xbox pad
    (one named the way Firefox names it too), a DualShock 4, a DualSense; then pads it does not
    know."""
    report: dict = {}
    async with play(recorded, directory) as chrome:
        await engage(chrome)
        await asyncio.sleep(0.3)
        await chrome.eval(f"harness.pad.connect('', {json.dumps(XBOX_ID)})")
        report["xbox_toasts"] = await chrome.eval("harness.toasts()")
        report["xbox"] = await act(chrome, XBOX_REST, XBOX)
        await chrome.eval("harness.pad.disconnect()")
        await chrome.eval(f"harness.pad.connect('', {json.dumps(DUALSHOCK_ID)})")
        report["dualshock_toasts"] = await chrome.eval("harness.toasts()")
        report["dualshock"] = await act(chrome, DUALSHOCK_REST, DUALSHOCK)
        await chrome.eval("harness.pad.disconnect()")
        await chrome.eval(f"harness.pad.connect('', {json.dumps(DUALSENSE_ID)})")
        report["dualsense"] = await act(chrome, DUALSENSE_REST, DUALSENSE)
        await chrome.eval("harness.pad.disconnect()")
        await chrome.eval("harness.pad.connect('', '045e-028e-Microsoft X-Box 360 pad')")
        report["firefox_toasts"] = await chrome.eval("harness.toasts()")
        report["firefox"] = await act(chrome, XBOX_REST, {"jump": XBOX["jump"]})
        await chrome.eval("harness.pad.disconnect()")
        report["refused"] = {}
        look_and_forward = {**XBOX["look_left"][0], **XBOX["forward"][0]}
        axes = [look_and_forward.get(i, value) for i, value in enumerate(XBOX_REST)]
        for name, pad_id in {
            "unknown": "Generic USB Joystick (Vendor: 1234 Product: abcd)",
            "sidewinder": "Microsoft SideWinder Precision 2 (Vendor: 045e Product: 0027)",
            "dualshock3": "Sony Controller (Vendor: 054c Product: 0268)",
            "no_ids": "Some Pad",
        }.items():
            await chrome.eval(f"harness.pad.connect('', {json.dumps(pad_id)})")
            report["refused"][name] = {
                "toasts": await chrome.eval("harness.toasts()"),
                "pushed": await pushed(chrome, axes, [0, 7], seconds=0.4, rest=XBOX_REST),
            }
            await chrome.eval("harness.pad.disconnect()")
    return report


@pytest.fixture(scope="module")
def raw(recorded, tmp_path_factory) -> dict:
    return asyncio.run(raw_pad_session(recorded, tmp_path_factory.mktemp("raw")))


def check_layout(report: dict, *, trigger_buttons: bool = False) -> None:
    """What every layout does as the standard mapping does: the right stick turns at the maximum
    turn speed, the left holds the movement buttons, the face buttons and the triggers hold theirs
    (Y inspects with the knife the D-pad's down took) and the bumpers and the D-pad take the
    weapons."""
    look = report["look_left"]
    assert at_rate(look["yaw_rates"], MAX_YAW) and look["pitch"] == 0
    assert within_caps(look["profile"]) and look["profile"][-1][:2] == [0, 0]
    down = report["look_down"]
    assert at_rate(down["pitch_rates"], MAX_PITCH) and down["yaw"] == 0
    for name in ("jump", "duck", "reload", "look_at_weapon", "attack", "attack2"):
        assert set(report[name]["bits"]) == {BIT[name], 0}, name
    assert set(report["forward"]["bits"]) == {0, BIT["forward"]}
    assert set(report["diagonal"]["bits"]) == {0, BIT["forward"] | BIT["move_right"]}
    if trigger_buttons:
        assert set(report["attack_button"]["bits"]) == {BIT["attack"], 0}
        assert set(report["attack2_button"]["bits"]) == {BIT["attack2"], 0}
    assert report["unmoved_triggers"]["bits"] == [0]  # a trigger that reads 0 until it moves
    for name, weapon in AFTER.items():
        assert report[name]["weapon"].endswith(weapon), name


def test_an_xbox_pad_of_no_standard_mapping_plays_by_the_xpad_layout(raw):
    """Chrome on Linux gives a pad xpad drives mapping "": the page knows it by its vendor and
    product, says so, and reads the sticks (axes 0, 1 and 3, 4), the triggers (axes 2 and 5, at -1
    at rest), the face buttons and bumpers (0-5) and the D-pad (axes 6 and 7) as it reads the
    standard mapping's; Back and Start (buttons 6 and 7) do nothing."""
    assert raw["xbox_toasts"] == ["Microsoft X-Box 360 pad connected (Xbox layout)"]
    check_layout(raw["xbox"])
    assert raw["xbox"]["back_and_start"]["bits"] == [0]


def test_a_dualshock_4_and_a_dualsense_play_by_their_layouts(raw):
    """Sticks on axes 0, 1 and 2, 5, triggers on axes 3 and 4 (and buttons 6 and 7), Square and
    Triangle the other way round from the Xbox pad's X and Y, and the D-pad as a hat on axis 9 (or
    axes 6 and 7 on a pad of eight axes): up, right, down and left take the weapons, a diagonal
    (up and left) takes the last of its two."""
    assert raw["dualshock_toasts"][-1] == "Wireless Controller connected (PlayStation layout)"
    check_layout(raw["dualshock"], trigger_buttons=True)
    assert raw["dualshock"]["dpad_up_left"]["weapon"].endswith("HE Grenade")
    sense = raw["dualsense"]
    assert at_rate(sense["look_left"]["yaw_rates"], MAX_YAW)
    assert set(sense["attack"]["bits"]) == {BIT["attack"], 0}
    for name in ("first_weapon", "dpad_left", "dpad_up", "dpad_right", "dpad_down"):
        assert sense[name]["weapon"].endswith(AFTER[name]), name


def test_a_pad_named_the_way_firefox_names_it_is_known_too(raw):
    assert raw["firefox_toasts"][-1] == "Microsoft X-Box 360 pad connected (Xbox layout)"
    assert set(raw["firefox"]["jump"]["bits"]) == {BIT["jump"], 0}


def test_a_pad_the_page_does_not_know_stays_refused_with_the_same_notice(raw):
    """An unknown vendor, a known vendor's other product and no vendor in the id: the notice is
    that of any pad of no standard mapping, and nothing of its sticks or buttons reaches the
    samples."""
    names = {
        "unknown": "Generic USB Joystick",
        "sidewinder": "Microsoft SideWinder Precision 2",
        "dualshock3": "Sony Controller",
        "no_ids": "Some Pad",
    }
    for name, shown in names.items():
        refused = raw["refused"][name]
        assert refused["toasts"][-1] == f"{shown} has no standard mapping: not used", name
        assert refused["pushed"]["bits"] == [0], name
        assert refused["pushed"]["yaw"] == 0 and refused["pushed"]["pitch"] == 0, name


async def touch_session(recorded: dict, directory: Path) -> dict:
    """The on-screen pad on a phone of 390 x 844, with fingers: both sticks and a button at once."""
    report: dict = {}
    async with play(recorded, directory, (390, 844, 3), touch=True) as chrome:
        report["card"] = await chrome.eval("harness.state()")
        await engage(chrome, touch=True)
        move, look = await chrome.eval("[harness.stick('move'), harness.stick('look')]")
        fire = await chrome.eval("harness.centre('[data-b=attack]')")
        jump = await chrome.eval("harness.centre('[data-b=jump]')")
        fingers = {0: (move["x"], move["y"])}
        await chrome.touch("touchStart", fingers)
        fingers[0] = (move["x"], move["y"] - move["travel"])
        await chrome.touch("touchMove", fingers)
        fingers[1] = (look["x"], look["y"])
        await chrome.touch("touchStart", fingers)
        fingers[1] = (look["x"] + 0.5 * look["travel"], look["y"])
        await chrome.touch("touchMove", fingers)
        fingers[2] = tuple(fire)
        await chrome.touch("touchStart", fingers)
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(1.0)
        until = await chrome.eval("harness.now()")
        report["together"] = {**await sent(chrome, since, until), "seconds": (until - since) / 1000}
        report["lit"] = await chrome.eval("harness.state()")
        # the look finger slides far off its stick: the stick is at full deflection, still its
        fingers[1] = (look["x"] + 300, look["y"] + 12)
        await chrome.touch("touchMove", fingers)
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(0.8)
        until = await chrome.eval("harness.now()")
        report["slid"] = {**await sent(chrome, since, until), "seconds": (until - since) / 1000}
        report["knobs"] = (await chrome.eval("harness.state()"))["knob"]
        # the fire finger lifts, then a second button goes down beside the held sticks
        await chrome.touch("touchEnd", {2: fingers.pop(2)})
        fingers[3] = tuple(jump)
        await chrome.touch("touchStart", fingers)
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(0.4)
        report["jump"] = await sent(chrome, since)
        # every finger lifts: nothing stays held, the sticks rest, the view stops
        await chrome.touch("touchEnd", {})
        await asyncio.sleep(1.0)  # the turn asked for and not yet made is made, and comes to rest
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(0.4)
        report["after"] = {
            **await sent(chrome, since),
            "state": await chrome.eval("harness.state()"),
        }
        # the weapon button takes the next weapon; the pause button pauses
        before = (await chrome.eval("harness.state()"))["padWeapon"]
        wx, wy = await chrome.eval("harness.centre('[data-next]')")
        await chrome.tap(wx, wy)
        await asyncio.sleep(0.2)
        report["weapon"] = [before, (await chrome.eval("harness.state()"))["padWeapon"]]
        await chrome.tap(*await chrome.eval("harness.centre('#pause')"))
        await asyncio.sleep(0.2)
        report["paused"] = await chrome.eval("harness.state()")
    return report


@pytest.fixture(scope="module")
def touch(recorded, tmp_path_factory) -> dict:
    return asyncio.run(touch_session(recorded, tmp_path_factory.mktemp("touch")))


def test_both_sticks_and_a_button_work_at_once_with_three_fingers(touch):
    """The left stick up holds forward, the right stick at half deflection turns the yaw right at
    31.5 degrees a second (the curve of the gamepad's), and the fire finger holds attack: all in
    the same samples, the sticks and the button lit meanwhile."""
    together = touch["together"]
    assert together["last"] == BIT["forward"] | BIT["attack"]
    assert set(together["bits"]) <= {0, BIT["forward"], BIT["forward"] | BIT["attack"]}
    assert (
        at_rate(together["yaw_rates"], -HALF_DEFLECTION, MAX_YAW) and together["pitch_rates"] == []
    )
    assert made(together, "yaw", -HALF_DEFLECTION)
    assert together["pitch"] == 0
    assert sorted(touch["lit"]["lit"]) == ["attack", "look", "move"]


def test_a_finger_that_slides_off_its_stick_keeps_it_at_full_deflection(touch):
    """The pointer is captured: the look finger 300 px past the stick asks for the top rate, the
    maximum turn speed (right and a little down: its 12 px below the centre), none of it over the
    cap; the knob stays at the edge of the stick's travel, 37 px from the centre."""
    slid = touch["slid"]
    down = MAX_YAW * 12 / math.hypot(300, 12)
    assert within_caps(slid["profile"])
    assert at_rate(slid["pitch_rates"], down, MAX_PITCH)
    yaw = [row[1] for row in slid["profile"]]
    assert -MAX_YAW - RATE_TOLERANCE <= yaw[-1] < -0.9 * MAX_YAW
    move, look = (math.hypot(*map(float, re.findall(r"-?[\d.]+", k))) for k in touch["knobs"])
    assert (move, look) == (CLOSE(37.0, abs=0.01), CLOSE(37.0, abs=0.01))
    assert slid["last"] == BIT["forward"] | BIT["attack"]


def test_a_second_button_goes_down_beside_the_held_sticks(touch):
    """The fire finger lifted and a jump finger came down: forward and jump are held, not attack."""
    assert touch["jump"]["last"] == BIT["forward"] | BIT["jump"]


def test_lifting_every_finger_leaves_nothing_held_and_the_view_still(touch):
    after = touch["after"]
    assert after["bits"] == [0] and after["yaw"] == 0 and after["pitch"] == 0
    assert after["state"]["lit"] == [] and after["state"]["knob"] == ["", ""]
    assert after["state"]["scroll"] == [0, 0]


def test_the_weapon_button_and_the_pause_button_of_the_pad(touch):
    assert touch["weapon"] == ["AK-47", "Glock-18"]
    assert touch["paused"]["card"] and touch["paused"]["lit"] == []
    assert touch["card"]["card"] and touch["card"]["switch"] is None  # no keyboard settings


async def touch_edge_session(recorded: dict, directory: Path) -> dict:
    """The pad's edges, on a phone of 390 x 844: a tap on the view, a thumb that lands off the
    middle of a stick, two fingers on one button, and every finger cancelled at once."""
    report: dict = {}
    async with play(recorded, directory, (390, 844, 3), touch=True) as chrome:
        await engage(chrome, touch=True)
        look = await chrome.eval("harness.stick('look')")
        fire = await chrome.eval("harness.centre('[data-b=attack]')")
        stage_x, stage_y, stage_w, stage_h = await chrome.eval("harness.stage()")
        since = await chrome.eval("harness.now()")
        await chrome.tap(stage_x + stage_w / 2, stage_y + stage_h / 2, hold=0.25)
        await asyncio.sleep(0.2)
        report["tap"] = await sent(chrome, since)
        # a thumb lands 50 px right of the middle of the stick, past the knob's travel (37 px)
        landed = (look["x"] + 50, look["y"])
        await chrome.touch("touchStart", {0: landed})
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(0.5)
        report["landed"] = await sent(chrome, since)
        moved = (landed[0] + 0.5 * look["travel"], landed[1])
        await chrome.touch("touchMove", {0: moved})
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(0.8)
        until = await chrome.eval("harness.now()")
        report["landed_moved"] = {
            **await sent(chrome, since, until),
            "seconds": (until - since) / 1000,
        }
        await chrome.touch("touchEnd", {})
        await asyncio.sleep(0.2)
        # two fingers on Fire: it is held until the second lifts
        fingers = {1: tuple(fire), 2: (fire[0] + 6, fire[1] + 6)}
        await chrome.touch("touchStart", {1: fingers[1]})
        await chrome.touch("touchStart", fingers)
        await asyncio.sleep(0.3)
        await chrome.touch("touchEnd", {1: fingers.pop(1)})
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(0.3)
        report["one_lifted"] = await sent(chrome, since)
        await chrome.touch("touchEnd", {})
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(0.3)
        report["both_lifted"] = await sent(chrome, since)
        # every finger cancelled (a system gesture, a palm): sticks at rest, nothing held
        await chrome.touch("touchStart", {0: (look["x"], look["y"])})
        fingers = {0: (look["x"] + look["travel"], look["y"]), 1: tuple(fire)}
        await chrome.touch("touchMove", fingers)
        await chrome.touch("touchStart", fingers)
        await asyncio.sleep(0.4)
        report["before_cancel"] = await chrome.eval("harness.state()")
        await chrome.touch("touchCancel", {})
        await asyncio.sleep(0.6)  # the turn comes to rest over a third of a second at most
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(0.4)
        report["cancelled"] = {
            **await sent(chrome, since),
            "state": await chrome.eval("harness.state()"),
        }
    return report


@pytest.fixture(scope="module")
def edges(recorded, tmp_path_factory) -> dict:
    return asyncio.run(touch_edge_session(recorded, tmp_path_factory.mktemp("edges")))


def test_a_tap_on_the_view_fires_nothing_on_a_touch_device(edges):
    """The pad has Fire: the mouse events that follow a tap on the view are not the mouse's."""
    assert keeps_coming(edges["tap"]) and edges["tap"]["bits"] == [0]


def test_a_thumb_that_lands_off_the_middle_of_a_stick_turns_nothing_until_it_moves(edges):
    """The stick works from where the thumb lands: 50 px off the middle (past its 37 px of
    travel) the view stays; a move of half the travel from there turns it at the half-deflection
    rate, not at the top rate."""
    assert edges["landed"]["yaw"] == 0 and keeps_coming(edges["landed"])
    moved = edges["landed_moved"]
    assert at_rate(moved["yaw_rates"], -HALF_DEFLECTION, MAX_YAW)
    assert made(moved, "yaw", -HALF_DEFLECTION)


def test_a_button_with_two_fingers_on_it_is_held_until_the_last_lifts(edges):
    assert edges["one_lifted"]["bits"] == [BIT["attack"]]
    assert edges["both_lifted"]["last"] == 0


def test_cancelled_touches_leave_the_sticks_at_rest_and_nothing_held(edges):
    """The system cancels every finger (pointercancel): the look stick lets go, the knob goes home
    and Fire is up."""
    before = edges["before_cancel"]
    assert sorted(before["lit"]) == ["attack", "look"]
    cancelled = edges["cancelled"]
    assert cancelled["bits"] == [0] and cancelled["yaw"] == 0 and cancelled["pitch"] == 0
    assert cancelled["state"]["lit"] == [] and cancelled["state"]["knob"] == ["", ""]


#: The rate of the samples that go out at the animation frames, a second (``SAMPLE_RATE``, in
#: ``static/input.js``): the engine reads the latest sample of a block, a display of any rate sends
#: this many at most (and a button or the weapon changing is sent at once besides).
SAMPLE_RATE = 60
#: The most samples one second may hold: the rate and the credit that a sample can be sent early on.
MOST_SAMPLES = 62
DISPLAYS = (30, 60, 75, 90, 120, 144, 165, 240, 360)


def scripted(hz: int, seconds: float = 4.0, **run) -> dict:
    """A run of ``harness.cap`` over ``seconds`` s of a display of ``hz`` Hz: the yaw at 36 degrees
    a second and the pitch at 21.6 from 0.3 s to 1.8 s, as a slow arrow key and the up key ask."""
    rates = {round(0.3 * hz): [21.6, 36], round(1.8 * hz): [0, 0]}
    return {"hz": hz, "frames": round(seconds * hz), "rates": rates, **run}


def button_changes(hz: int) -> list[tuple[int, str, bool]]:
    """Buttons changing on a display of ``hz`` Hz, by frame: a tap of forward (down, and up in the
    next frame), jump held for 0.4 s, then a tap of duck."""
    return [
        (round(0.5 * hz), "forward", True),
        (round(0.5 * hz) + 1, "forward", False),
        (round(1.0 * hz), "jump", True),
        (round(1.4 * hz), "jump", False),
        (round(1.4 * hz) + 1, "duck", True),
        (round(1.4 * hz) + 2, "duck", False),
    ]


async def cap_session(recorded: dict, directory: Path) -> dict:
    """``PlayerInput`` alone (``harness.cap``) on displays of 30 to 360 Hz, 60 Hz frames that
    differ by 8 ms in turns, and a 144 Hz display whose page stalls for a quarter of a second, with
    a turn asked for and with buttons changing."""
    report: dict = {"turn": {}, "edges": {}}
    async with play(recorded, directory) as chrome:

        async def cap(**run) -> dict:
            return await chrome.eval(f"harness.cap({json.dumps(run)})")

        for hz in DISPLAYS:
            report["turn"][hz] = await cap(**scripted(hz))
            changes = {frame: [name, down] for frame, name, down in button_changes(hz)}
            report["edges"][hz] = await cap(**scripted(hz, changes=changes))
        report["jitter"] = await cap(**scripted(60, jitter=4))
        report["stall"] = await cap(**scripted(144, stall={str(round(0.9 * 144)): 250}))
    return report


@pytest.fixture(scope="module")
def capped(recorded, tmp_path_factory) -> dict:
    return asyncio.run(cap_session(recorded, tmp_path_factory.mktemp("capped")))


def in_a_second(times: list[float]) -> int:
    """The most of ``times`` (ms) that fall in a second."""
    return max(sum(t <= u < t + 1000 for u in times) for t in times)


def turns(run: dict) -> list[float]:
    """The pitch and the yaw the samples of a ``harness.cap`` run carry in all."""
    return [sum(m[axis] for m in run["messages"]) for axis in ("dp", "dy")]


def test_a_display_of_any_rate_sends_at_most_sixty_samples_a_second(capped):
    """30 to 360 Hz: no second holds more than 62 samples; a display of 60 Hz or less sends one at
    every frame, a faster one 60 a second on average."""
    for hz, run in capped["turn"].items():
        times = [m["time"] for m in run["messages"]]
        assert in_a_second(times) <= MOST_SAMPLES, hz
        if hz <= SAMPLE_RATE:
            assert len(times) == run["frames"], hz
        else:
            assert len(times) == CLOSE(SAMPLE_RATE * run["frames"] / hz, abs=2), hz


def test_frames_of_unequal_length_all_send_at_sixty_hertz_and_a_stall_costs_one_sample_more(capped):
    """A 60 Hz display whose frames are 12.7 and 20.7 ms long by turns drops no sample; a 144 Hz
    one whose page stalls a quarter of a second keeps its second under the bound, with no more than
    three samples in the four frames after the stall (the credit that one stall earns is capped)."""
    jitter = capped["jitter"]
    assert len(jitter["messages"]) == jitter["frames"]
    stall = capped["stall"]
    assert in_a_second([m["time"] for m in stall["messages"]]) <= MOST_SAMPLES
    frame = round(0.9 * 144)
    assert len([m for m in stall["messages"] if frame <= m["frame"] < frame + 4]) <= 3


def test_the_turn_commanded_in_the_frames_between_samples_goes_with_the_next_sample(capped):
    """Whatever the display, the samples carry all of the turn the input makes (the pitch and
    the yaw asked for are made, none lost in a frame without a sample and none counted twice), and
    nothing is left unsent once the turn is over."""
    for hz, run in capped["turn"].items():
        assert turns(run) == CLOSE(run["commanded"], abs=0.01), hz
        assert turns(run) == CLOSE(run["asked"], abs=0.01), hz
        assert run["unsent"] == [0, 0], hz
        assert turns(run)[1] == CLOSE(36 * 1.5, abs=0.5), hz  # 36 degrees a second for 1.5 s
    for run in (capped["jitter"], capped["stall"]):
        assert turns(run) == CLOSE(run["commanded"], abs=0.01)
        assert turns(run) == CLOSE(run["asked"], abs=0.01)


def test_a_button_or_weapon_change_is_sent_in_the_frame_it_happens_on_any_display(capped):
    """A tap, a key held for 0.4 s and another tap: each change is a sample of its own in the frame
    it happens in, whatever the rate of the display, and every sample holds what is held at its
    time; on a display of 60 Hz a change takes the place of its frame's sample (one sample a frame
    as before), on a slower one it comes besides; the turn is all sent."""
    bits = {name: BIT[name] for name in ("forward", "jump", "duck")}
    for hz, run in capped["edges"].items():
        plan = iter(button_changes(hz))
        held = 0
        for message in run["messages"]:
            if message["changed"]:
                frame, name, down = next(plan)
                assert message["frame"] == frame, hz
                held = held | bits[name] if down else held & ~bits[name]
            assert message["b"] == held, hz
        assert next(plan, None) is None, hz
        assert turns(run) == CLOSE(run["commanded"], abs=0.01), hz
        if hz <= SAMPLE_RATE:
            besides = len(button_changes(hz)) if hz < SAMPLE_RATE else 0
            assert len(run["messages"]) == run["frames"] + besides, hz


async def refresh_session(recorded: dict, directory: Path, hz: int) -> dict:
    """The page on a display of ``hz`` Hz: idle, an arrow held for a second, W pressed and
    released."""
    report: dict = {}
    async with play(recorded, directory, refresh=hz) as chrome:
        await engage(chrome)
        since = await chrome.eval("harness.now()")
        await asyncio.sleep(2.0)
        now = await chrome.eval("harness.now()")
        samples = await chrome.eval(f"harness.samples({since}).filter((s) => s.at <= {now})")
        seconds = (now - since) / 1000
        report["idle"] = {
            "times": [s["at"] for s in samples],
            "seconds": seconds,
            "animation": await chrome.eval(f"harness.animation({since})") / seconds,
        }
        before = await chrome.eval("harness.commanded()")
        since = await chrome.eval("harness.now()")
        report["arrow"] = await hold_keys(chrome, ["ArrowLeft"])
        samples = await chrome.eval(f"harness.samples({since})")
        after = await chrome.eval("harness.commanded()")
        report["commanded"] = [a - b for a, b in zip(after, before)]
        report["carried"] = [sum(s[axis] for s in samples) for axis in ("dp", "dy")]
        report["arrow_times"] = [s["at"] for s in samples]
        since = await chrome.eval("harness.now()")
        await chrome.key("KeyW", True)
        await asyncio.sleep(0.5)
        await chrome.key("KeyW", False)
        await asyncio.sleep(0.3)
        report["keys"] = await chrome.eval(f"harness.keys.filter((k) => k.t >= {since})")
        report["edge_samples"] = await chrome.eval(f"harness.samples({since})")
    return report


@pytest.fixture(scope="module")
def refreshed(recorded, tmp_path_factory) -> dict:
    async def both() -> dict:
        return {
            hz: await refresh_session(recorded, tmp_path_factory.mktemp(f"hz{hz}"), hz)
            for hz in (144, 240)
        }

    return asyncio.run(both())


def test_the_page_on_a_display_of_144_or_240_hz_sends_at_most_sixty_samples_a_second(refreshed):
    """The animation frames come 100 or more times a second (150 or more at 240 Hz); idle and
    with an arrow held, the page sends about 60 samples a second and never more than 62 in a
    second."""
    for hz, report in refreshed.items():
        assert report["idle"]["animation"] > (150 if hz == 240 else 100), hz
        idle = report["idle"]["times"]
        assert len(idle) / report["idle"]["seconds"] == CLOSE(SAMPLE_RATE, abs=8), hz
        assert in_a_second(idle) <= MOST_SAMPLES, hz
        assert in_a_second(report["arrow_times"]) <= MOST_SAMPLES, hz


def test_the_page_on_a_fast_display_sends_all_the_turn_the_input_makes(refreshed):
    """An arrow held for a second: the samples carry exactly the turn the input made (the frames
    between samples lose none), which is what the key asked for."""
    for hz, report in refreshed.items():
        arrow = report["arrow"]
        assert report["carried"] == CLOSE(report["commanded"], abs=0.01), hz
        assert report["carried"][1] == CLOSE(
            ARROW_YAW * arrow["seconds"], abs=ARROW_YAW * arrow["frame"] + 0.5
        ), hz
        assert arrow["pitch"] == 0 and arrow["taken"] == [True], hz


def test_a_key_press_and_release_are_sent_within_a_frame_on_a_fast_display(refreshed):
    """W goes down and up: the sample that holds forward goes out in the animation frame of the
    key's press (or the next), the one that lets it go in that of the release."""
    for hz, report in refreshed.items():
        down, up = [k for k in report["keys"] if k["code"] == "KeyW"]
        samples = report["edge_samples"]
        pressed = next(s for s in samples if s["b"] & BIT["forward"])
        let_go = next(s for s in samples if s["at"] > pressed["at"] and not s["b"] & BIT["forward"])
        assert 0 <= pressed["frame"] - down["frame"] <= 1, hz
        assert 0 <= let_go["frame"] - up["frame"] <= 1, hz
