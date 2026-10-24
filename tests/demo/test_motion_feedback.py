"""The play view's motion feedback, in headless Chrome, with real key events.

The page's own script (``static/play.js``) runs on the welcome of a recorded mock session and a
stand-in frame stream (``motion_play.html``) that sends a frame, a red ramp across and a green one
down, only when the test says, with the newest input sample the frame holds. A movement key is
pressed and the frame that folds it in is held back: the view answers at once, scaled up (forward)
or shifted against the move (a strafe) along the ramp to the cap, as the red of the canvas shows,
and goes on to hold the cap; when the frame is delivered the view falls to the frame's own over a
fifth of a second, along the way and not in a step, and stays there for as long as the key is held.
A key let go before its frame falls the same way, and the arrow keys and the mouse turn the view
as before, with no move. The Motion feedback setting in the pause panel is off by default (every
frame is drawn as it was generated), runs here at Light, and is kept in localStorage. The browser
has no WebGL: the frames are drawn on the 2D canvas (the instant turn draws the feedback's move
alike, ``test_instant_turn.py``). The tests skip without Chrome or Chromium, and on a CI runner
(``CI`` set), which does not keep up with them in real time.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from tests.demo.support import chrome_page, needs_chrome, realtime, recorded_play
from worldcast.data.controls import CONTROL_BUTTONS

HARNESS = Path(__file__).with_name("motion_play.html")
pytestmark = [needs_chrome, realtime]

BIT = {name: 1 << i for i, name in enumerate(CONTROL_BUTTONS)}
#: The caps at the Light setting (``static/feedback.js``) and the ramp time (s).
IN, SHIFT, RAMP, HANDOVER = 0.04, 0.025, 0.4, 0.2
#: The arrow keys' yaw rate (degrees a second) and the mouse's degrees per pixel (the defaults).
ARROW_YAW, MOUSE_DEGREES_PER_PIXEL = 120.0, 0.03
#: The setting the scenarios run with, as localStorage keeps it.
LIGHT = {"wc.motion": '"light"'}


def ease(seconds: float) -> float:
    """The share of the cap ``seconds`` after a press."""
    return 1 - (1 - min(max(seconds / RAMP, 0.0), 1.0)) ** 2


def fall(seconds: float) -> float:
    """The share of what a ramp held that is left ``seconds`` after the frame that folds it in."""
    x = min(max(seconds / HANDOVER, 0.0), 1.0)
    return 1 - x * x * (3 - 2 * x)


@pytest.fixture(scope="module")
def recorded() -> dict:
    return asyncio.run(recorded_play(3))


@asynccontextmanager
async def motion_play(recorded: dict, directory: Path, storage=None, canvas2d: bool = True):
    """The play view on the stand-in frame stream, three frames shown, waiting at the pause
    panel, in a browser with no WebGL (``canvas2d`` false: none that gives a 2D context either;
    ``storage``: what localStorage holds)."""
    setup = {
        "page": recorded["page"],
        "join": recorded["join"],
        "welcome": recorded["welcome"],
        "header": recorded["frames"][0][0],
        "storage": storage or {},
        "canvas2d": canvas2d,
        "webgl": False,
    }
    page = HARNESS.read_text(encoding="utf-8").replace("__SETUP__", json.dumps(setup))
    async with chrome_page(page, (1000, 700, 1), directory) as chrome:
        await chrome.wait_for("window.ready === true")
        assert (
            await chrome.eval("window.harness ? (harness.out.error ?? null) : 'no harness'") is None
        )
        await chrome.eval("harness.deliver(-1, 3)")
        await chrome.wait_for("!!document.querySelector('#go')")
        await asyncio.sleep(0.2)
        yield chrome


async def engage(chrome) -> None:
    """Press Play and wait until the pause panel is gone and the view has settled."""
    x, y = await chrome.eval("harness.centre('#go')")
    await chrome.click(x, y)
    await chrome.wait_for("document.querySelector('#overlay').hidden")
    await asyncio.sleep(0.4)


async def pause(chrome) -> None:
    await chrome.key("Escape", True)
    await chrome.key("Escape", False)
    await chrome.wait_for("!document.querySelector('#overlay').hidden")


async def pressed_at(chrome, code: str) -> float:
    """The time of the page's newest keydown of ``code``."""
    keys = await chrome.eval("harness.keys")
    return max(k["t"] for k in keys if k["code"] == code and k["type"] == "keydown")


async def press(chrome, code: str, seconds: float, button: str) -> dict:
    """Press ``code`` and hold it for ``seconds`` s with the frame that folds it in held back:
    the page's drawings since the press (``waiting``), the press (``at``) and the input sample
    that carried it (``q``)."""
    since = await chrome.eval("harness.now()")
    await chrome.key(code, True)
    await asyncio.sleep(seconds)
    sample = next(s for s in await chrome.eval(f"harness.samples({since})") if s["b"] & BIT[button])
    return {
        "waiting": await chrome.eval(f"harness.draws({since})"),
        "at": await pressed_at(chrome, code),
        "q": sample["q"],
    }


async def fold(chrome, held: int, seconds: float) -> dict:
    """Deliver the frame that holds the input samples up to ``held`` and watch ``seconds`` s: the
    drawings from then on (``after``) and the one of that frame (``frame``)."""
    since = await chrome.eval("harness.now()")
    await chrome.eval(f"harness.deliver({held})")
    await asyncio.sleep(seconds)
    after = await chrome.eval(f"harness.draws({since})")
    return {"after": after, "frame": next(d for d in after if d["q"] == held and d["keep"])}


def zoom_of(draw: dict) -> float:
    """The scale of the ramp the canvas shows, from its red at 0.1 and 0.9 of the width."""
    red = draw["red"]
    return 0.8 * 255 / (red[1] - red[0])


async def feedback_session(recorded: dict, directory: Path) -> dict:
    report: dict = {}
    async with motion_play(recorded, directory, storage=LIGHT) as chrome:
        await engage(chrome)
        report["rest"] = await chrome.eval("harness.draws(0)")
        # forward, its frame held back for 0.7 s, then delivered; the key held on for 0.8 s
        report["forward"] = await press(chrome, "KeyW", 0.7, "forward")
        report["forward"] |= await fold(chrome, report["forward"]["q"], 0.8)
        since = await chrome.eval("harness.now()")
        await chrome.key("KeyW", False)
        await asyncio.sleep(0.5)
        report["forward_up"] = await chrome.eval(f"harness.draws({since})")
        # a tap: down for 0.2 s and up, no frame to fold it in
        since = await chrome.eval("harness.now()")
        await chrome.key("KeyW", True)
        await asyncio.sleep(0.2)
        await chrome.key("KeyW", False)
        await asyncio.sleep(0.6)
        report["tap"] = {"draws": await chrome.eval(f"harness.draws({since})"), "since": since}
        report["tap"]["at"] = await pressed_at(chrome, "KeyW")
        keys = await chrome.eval("harness.keys")
        report["tap"]["held"] = (
            max(k["t"] for k in keys if k["code"] == "KeyW" and k["type"] == "keyup")
            - report["tap"]["at"]
        ) / 1000
        # back
        report["back"] = await press(chrome, "KeyS", 0.7, "back")
        report["back"] |= await fold(chrome, report["back"]["q"], 0.5)
        await chrome.key("KeyS", False)
        # strafe left, then right
        for name, code, button in (("left", "KeyA", "move_left"), ("right", "KeyD", "move_right")):
            report[name] = await press(chrome, code, 0.7, button)
            report[name] |= await fold(chrome, report[name]["q"], 0.5)
            await chrome.key(code, False)
            await asyncio.sleep(0.2)
        # forward and left together, folded in by one frame
        since = await chrome.eval("harness.now()")
        await chrome.key("KeyW", True)
        await chrome.key("KeyA", True)
        await asyncio.sleep(0.7)
        samples = await chrome.eval(f"harness.samples({since})")
        both = BIT["forward"] | BIT["move_left"]
        report["both"] = {
            "waiting": await chrome.eval(f"harness.draws({since})"),
            "q": next(s["q"] for s in samples if s["b"] & both == both),
        }
        report["both"] |= await fold(chrome, report["both"]["q"], 0.5)
        await chrome.key("KeyW", False)
        await chrome.key("KeyA", False)
        await asyncio.sleep(0.3)
        # an arrow key and the mouse turn the view as before, with no move
        since = await chrome.eval("harness.now()")
        await chrome.key("ArrowLeft", True)
        await asyncio.sleep(0.8)
        await chrome.key("ArrowLeft", False)
        await asyncio.sleep(0.5)
        report["arrow"] = {
            "draws": await chrome.eval(f"harness.draws({since})"),
            "samples": await chrome.eval(f"harness.samples({since})"),
            "keys": await chrome.eval("harness.keys"),
        }
        since = await chrome.eval("harness.now()")
        await chrome.eval("harness.burst(-300, 0)")
        await asyncio.sleep(1.0)
        report["mouse"] = {
            "draws": await chrome.eval(f"harness.draws({since})"),
            "samples": await chrome.eval(f"harness.samples({since})"),
        }
    return report


@pytest.fixture(scope="module")
def session(recorded, tmp_path_factory) -> dict:
    """What the page drew in each scenario (:func:`feedback_session`), at Light."""
    return asyncio.run(feedback_session(recorded, tmp_path_factory.mktemp("light")))


def test_forward_scales_the_view_up_at_once_along_the_ramp_while_its_frame_is_held_back(session):
    """The drawn frame's scale is that of an ease-out from the press to 1.04 at the time of its
    animation frame, at 0.4 s at the cap and held there, never above it; the canvas shows it (the
    ramp's red at 0.1 and 0.9 of the width is the frame's at 0.5 -/+ 0.4 / 1.04)."""
    forward = session["forward"]
    waiting, at = forward["waiting"], forward["at"]
    assert all(d["zoom"] == 1.0 for d in session["rest"]) and len(waiting) > 8
    zooms = [d["zoom"] for d in waiting]
    assert zooms == sorted(zooms) and max(zooms) == pytest.approx(1 + IN)
    for draw in waiting:
        assert draw["zoom"] == pytest.approx(1 + IN * ease((draw["raf"] - at) / 1000), abs=1e-9)
        assert draw["shift"] == 0.0
    early = [d for d in waiting if 30 < d["t"] - at < 150]
    assert all(d["zoom"] > 1.0 for d in early)  # at once, not at the next block
    assert zoom_of(waiting[-1]) == pytest.approx(1 + IN, abs=0.012)
    assert zoom_of(session["rest"][-1]) == pytest.approx(1.0, abs=0.012)


def test_the_frame_that_folds_the_key_in_hands_over_in_a_fall_not_a_step(session):
    """The frame is drawn at the scale the view had (no jump), the scale falls to 1 along a smooth
    step over the next fifth of a second, through values between and not in a step, and stays 1
    while the key is held (0.6 s more) and after it is let go; the canvas shows the frame's own
    scale again."""
    forward = session["forward"]
    frame = forward["frame"]
    assert frame["zoom"] == pytest.approx(1 + IN)
    later = [d for d in forward["after"] if d["t"] > frame["t"]]
    zooms = [d["zoom"] for d in later]
    assert zooms == sorted(zooms, reverse=True) and zooms[-1] == 1.0
    for draw in forward["after"]:
        seconds = (draw["raf"] - frame["raf"]) / 1000
        assert draw["zoom"] == pytest.approx(1 + IN * fall(seconds), abs=1e-9)
    assert any(1.0 < z < 1 + IN for z in zooms)
    assert zoom_of(forward["after"][-1]) == pytest.approx(1.0, abs=0.012)
    assert all(d["zoom"] == 1.0 and d["shift"] == 0.0 for d in session["forward_up"])


def test_a_key_let_go_before_its_frame_falls_the_same_way_and_is_not_answered(session):
    """Down for 0.2 s with no frame: the scale rose (to about 1.03), fell back to 1 by itself and
    never went under 1 for the stop."""
    draws = session["tap"]["draws"]
    zooms = [d["zoom"] for d in draws]
    assert max(zooms) > 1 + 0.6 * IN and min(zooms) == 1.0 and zooms[-1] == 1.0
    assert max(zooms) <= 1 + IN * ease(session["tap"]["held"])
    peak = zooms.index(max(zooms))
    assert zooms[peak:] == sorted(zooms[peak:], reverse=True)


def test_back_scales_the_view_down_and_leaves_a_filled_band(session):
    """Back: the scale falls to 0.97 along the ramp, the canvas shows the ramp smaller (the red at
    0.1 and 0.9 of the width is the frame's at 0.5 -/+ 0.4 / 0.97), and the frame that folds it in
    hands over to 1."""
    back = session["back"]
    zooms = [d["zoom"] for d in back["waiting"]]
    assert zooms == sorted(zooms, reverse=True) and min(zooms) == pytest.approx(0.97)
    assert zoom_of(back["waiting"][-1]) == pytest.approx(0.97, abs=0.012)
    assert back["frame"]["zoom"] == pytest.approx(0.97) and back["after"][-1]["zoom"] == 1.0


def test_a_strafe_shifts_the_view_against_the_move_and_hands_over_to_the_frames(session):
    """Left moves the view to the right by 2.5 percent of its width (the ramp's red at a fixed
    place falls by 6), right to the left; neither scales; the frame that folds the key in hands
    over to no shift."""
    for name, sign in (("left", 1), ("right", -1)):
        strafe = session[name]
        shifts = [sign * d["shift"] for d in strafe["waiting"]]
        assert shifts == sorted(shifts) and max(shifts) == pytest.approx(SHIFT)
        assert all(d["zoom"] == 1.0 for d in strafe["waiting"]), name
        red = strafe["waiting"][-1]["red"]
        assert red[0] == pytest.approx(255 * (0.1 - sign * SHIFT), abs=3), name
        assert zoom_of(strafe["waiting"][-1]) == pytest.approx(1.0, abs=0.012), name
        assert strafe["frame"]["shift"] == pytest.approx(sign * SHIFT), name
        assert strafe["after"][-1]["shift"] == 0.0, name


def test_keys_held_together_add_and_one_frame_folds_both_in(session):
    """Forward and left held together: the scale and the shift at their caps together, both handed
    over to nothing by the one frame that holds the sample of the two."""
    both = session["both"]
    last = both["waiting"][-1]
    assert last["zoom"] == pytest.approx(1 + IN) and last["shift"] == pytest.approx(SHIFT)
    assert both["after"][-1]["zoom"] == 1.0 and both["after"][-1]["shift"] == 0.0


def test_the_arrow_keys_and_the_mouse_turn_the_view_as_before_with_no_move(session):
    """An arrow key held for 0.8 s turns the yaw at its rate and a mouse move of 300 pixels by
    9 degrees, and the view is neither scaled nor shifted meanwhile: the frame is drawn as it is."""
    arrow = session["arrow"]
    assert all(d["zoom"] == 1.0 and d["shift"] == 0.0 for d in arrow["draws"])
    keys = [k["t"] for k in arrow["keys"] if k["code"] == "ArrowLeft"][-2:]
    seconds = (keys[1] - keys[0]) / 1000
    yaw = sum(s["dy"] for s in arrow["samples"])
    assert yaw == pytest.approx(ARROW_YAW * seconds, abs=ARROW_YAW * 0.1 + 0.5)
    assert all(s["b"] == 0 for s in arrow["samples"])
    mouse = session["mouse"]
    assert all(d["zoom"] == 1.0 and d["shift"] == 0.0 for d in mouse["draws"])
    assert sum(s["dy"] for s in mouse["samples"]) == pytest.approx(
        300 * MOUSE_DEGREES_PER_PIXEL, abs=0.01
    )
    reds = [d["red"] for d in arrow["draws"] + mouse["draws"]]
    assert all(red == reds[0] for red in reds)


async def setting_session(recorded: dict, directory: Path) -> dict:
    """The setting in the pause panel: its default (Off), a click on Strong (kept), Play and a
    key; Off with a key pressed, folded in and let go; a value kept before the page runs (Light),
    and one that is none."""
    report: dict = {}
    async with motion_play(recorded, directory) as chrome:
        report["default"] = await chrome.eval("harness.state()")
        await chrome.eval(
            "document.querySelector('input[name=motionfb][value=strong]').scrollIntoView()"
        )
        x, y = await chrome.eval("harness.centre('input[name=motionfb][value=strong]')")
        await chrome.click(x, y)
        report["clicked"] = await chrome.eval("harness.state()")
        await engage(chrome)
        report["strong"] = await press(chrome, "KeyW", 0.7, "forward")
        await chrome.key("KeyW", False)
        await pause(chrome)
        report["paused"] = await chrome.eval("harness.state()")
    async with motion_play(recorded, directory / "light", storage=LIGHT) as chrome:
        report["kept_light"] = await chrome.eval("harness.state()")
    async with motion_play(recorded, directory / "off") as chrome:
        report["kept_off"] = await chrome.eval("harness.state()")
        await engage(chrome)
        report["off"] = await press(chrome, "KeyW", 0.7, "forward")
        report["off"] |= await fold(chrome, report["off"]["q"], 0.4)
        await chrome.key("KeyW", False)
    async with motion_play(
        recorded, directory / "bogus", storage={"wc.motion": '"loud"'}
    ) as chrome:
        report["bogus"] = await chrome.eval("harness.state()")
    return report


@pytest.fixture(scope="module")
def setting(recorded, tmp_path_factory) -> dict:
    directory = tmp_path_factory.mktemp("setting")
    for name in ("light", "off", "bogus"):
        (directory / name).mkdir()
    return asyncio.run(setting_session(recorded, directory))


def test_the_setting_is_off_by_default_and_strong_doubles_the_caps_and_is_kept(setting):
    """Nothing kept: Off is on and nothing is stored; a click on Strong stores it
    (``wc.motion``, in localStorage as a JSON string) and the next key scales the view to 1.08,
    twice the cap; Light kept in the browser is read."""
    assert setting["default"]["choice"] == [["off", True], ["light", False], ["strong", False]]
    assert setting["kept_light"]["choice"] == [["off", False], ["light", True], ["strong", False]]
    assert setting["default"]["stored"] is None
    assert setting["clicked"]["choice"] == [["off", False], ["light", False], ["strong", True]]
    assert setting["clicked"]["stored"] == '"strong"'
    zooms = [d["zoom"] for d in setting["strong"]["waiting"]]
    assert max(zooms) == pytest.approx(1 + 2 * IN) and zooms == sorted(zooms)
    assert setting["paused"]["choice"][2] == ["strong", True]


def test_off_draws_the_frames_as_they_are(setting):
    """Off, the default: a key pressed, folded in and let go never changes the scale or the
    shift of a drawn frame, which the canvas shows as it is; an unknown stored value is Off."""
    assert setting["kept_off"]["choice"] == [["off", True], ["light", False], ["strong", False]]
    off = setting["off"]
    draws = off["waiting"] + off["after"]
    assert all(d["zoom"] == 1.0 and d["shift"] == 0.0 for d in draws)
    assert all(d["red"] == draws[0]["red"] for d in draws)
    assert setting["bogus"]["choice"] == [["off", True], ["light", False], ["strong", False]]


async def bare_session(recorded: dict, directory: Path) -> dict:
    """A browser that gives no 2D context, the setting at Light: a key pressed, folded in, let
    go."""
    async with motion_play(recorded, directory, storage=LIGHT, canvas2d=False) as chrome:
        await engage(chrome)
        report = await press(chrome, "KeyW", 0.5, "forward")
        report |= await fold(chrome, report["q"], 0.5)
        await chrome.key("KeyW", False)
        await asyncio.sleep(0.2)
        report["errors"] = await chrome.eval("harness.errors")
    return report


def test_a_browser_with_no_canvas_context_breaks_nothing(recorded, tmp_path):
    """With no 2D context nothing is drawn and nothing is raised: the page goes on
    asking the controller for its move (the scale still rises along the ramp and falls to 1 when
    the frame comes) and the key's sample still goes out."""
    report = asyncio.run(bare_session(recorded, tmp_path))
    assert report["errors"] == []
    zooms = [d["zoom"] for d in report["waiting"]]
    assert zooms == sorted(zooms) and max(zooms) == pytest.approx(1 + IN)
    assert report["frame"]["zoom"] == pytest.approx(1 + IN) and report["after"][-1]["zoom"] == 1.0
