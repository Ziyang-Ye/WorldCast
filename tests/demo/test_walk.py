"""Walk by default, in headless Chrome, with real key events.

The page's own script (``static/play.js``) runs on the welcome and a few frames of a recorded mock
session (``play_input.html``). With the pause panel's Walk by default on (the default), the page
holds the walk button (``speed``) while a movement key (W, A, S, D; an arrow key does not move the
player) or the left stick is held, and Shift, the key that runs, lets go of it; Shift alone holds
nothing. The button changes with the key, in the sample that carries the key. With it off, Shift
walks as it did. The key cap of Shift lights for the key, whatever the setting. The tests skip
without Chrome or Chromium, and on a CI runner (``CI`` set), which does not keep up with them in
real time.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from tests.demo.support import chrome_page, needs_chrome, realtime, recorded_play
from worldcast.data.controls import CONTROL_BUTTONS

HARNESS = Path(__file__).with_name("play_input.html")
pytestmark = [needs_chrome, realtime]

BIT = {name: 1 << i for i, name in enumerate(CONTROL_BUTTONS)}
FORWARD, BACK, LEFT, RIGHT, SPEED = (
    BIT[name] for name in ("forward", "back", "move_left", "move_right", "speed")
)
#: What a step of a scenario waits for the page to send.
SETTLE_S = 0.3


@pytest.fixture(scope="module")
def recorded() -> dict:
    return asyncio.run(recorded_play(3))


@asynccontextmanager
async def play(recorded: dict, directory: Path, storage: dict | None = None):
    """The play view of the input harness with the localStorage ``storage`` (Walk by default is
    on where it holds nothing), waiting at the pause panel."""
    setup = {
        **recorded,
        "storage": {"wc.motion": '"off"', **(storage or {})},
        "gamepadApi": True,
        "pointerLock": True,
        "refresh": None,
    }
    page = HARNESS.read_text(encoding="utf-8").replace("__SETUP__", json.dumps(setup))
    async with chrome_page(page, (1000, 700, 1), directory) as chrome:
        await chrome.wait_for("window.ready === true")
        assert (
            await chrome.eval("window.harness ? (harness.out.error ?? null) : 'no harness'") is None
        )
        await chrome.wait_for("!!document.querySelector('#go')")
        await asyncio.sleep(0.1)
        yield chrome


async def engage(chrome) -> None:
    x, y = await chrome.eval("harness.centre('#go')")
    await chrome.click(x, y)
    await chrome.wait_for("document.querySelector('#overlay').hidden")
    await asyncio.sleep(0.5)


async def masks(chrome, since: float) -> list[int]:
    """The button masks of the samples sent since ``since``, in order, one for each change."""
    samples = await chrome.eval(f"harness.samples({since})")
    masks = [s["b"] for s in samples]
    return [m for i, m in enumerate(masks) if i == 0 or m != masks[i - 1]]


def within(sent: list[int], *allowed: int) -> bool:
    """Whether every mask sent is one of ``allowed``: a periodic sample before a key's event is
    the state before it, which a step may begin with."""
    return set(sent) <= set(allowed)


async def step(chrome, *keys: tuple[str, bool], seconds: float = SETTLE_S) -> list[int]:
    """Press and release ``keys`` ((code, down) in order), wait, and the masks the page sent
    meanwhile, in order, each once."""
    since = await chrome.eval("harness.now()")
    for code, down in keys:
        await chrome.key(code, down)
    await asyncio.sleep(seconds)
    return await masks(chrome, since)


async def cap_lit(chrome, button: str) -> bool:
    return await chrome.eval(
        f"document.querySelector('.cap[data-b={button}]').classList.contains('on')"
    )


async def keyboard_session(recorded: dict, directory: Path, storage: dict | None) -> dict:
    """The keys of one setting: Shift alone, W, W then Shift and Shift off, Shift then W, W let go
    with Shift held and with it not, S, A and D, an arrow key and the mouse."""
    report: dict = {}
    async with play(recorded, directory, storage) as chrome:
        report["panel"] = await chrome.eval("""({
              on: document.querySelector('#walk').checked,
              says: document.querySelector('#walk-says').textContent,
              legend: document.querySelector('#shift-says').textContent,
              title: document.querySelector('.cap[data-b=speed]').title,
              stored: localStorage.getItem('wc.walk'),
            })""")
        await engage(chrome)
        report["shift"] = await step(chrome, ("ShiftLeft", True))
        report["shift_cap"] = await cap_lit(chrome, "speed")
        report["shift_up"] = await step(chrome, ("ShiftLeft", False))
        report["w"] = await step(chrome, ("KeyW", True))
        report["w_cap"] = [await cap_lit(chrome, "forward"), await cap_lit(chrome, "speed")]
        report["w_shift"] = await step(chrome, ("ShiftLeft", True))
        report["w_shift_cap"] = [await cap_lit(chrome, "forward"), await cap_lit(chrome, "speed")]
        report["w_unshift"] = await step(chrome, ("ShiftLeft", False))
        report["w_up"] = await step(chrome, ("KeyW", False))
        report["shift_w"] = await step(chrome, ("ShiftLeft", True), ("KeyW", True))
        report["shift_w_up"] = await step(chrome, ("KeyW", False))
        report["shift_released"] = await step(chrome, ("ShiftLeft", False))
        report["s"] = await step(chrome, ("KeyS", True), ("KeyS", False))
        report["a_d"] = await step(
            chrome, ("KeyA", True), ("KeyD", True), ("KeyA", False), ("KeyD", False)
        )
        since = await chrome.eval("harness.now()")
        await chrome.key("ArrowLeft", True)
        await asyncio.sleep(0.5)
        await chrome.key("ArrowLeft", False)
        await asyncio.sleep(0.4)
        report["arrow"] = {
            "masks": await masks(chrome, since),
            "yaw": sum(s["dy"] for s in await chrome.eval(f"harness.samples({since})")),
        }
        since = await chrome.eval("harness.now()")
        await chrome.eval("harness.burst(-300, 0)")
        await asyncio.sleep(1.0)
        samples = await chrome.eval(f"harness.samples({since})")
        report["mouse"] = {
            "masks": await masks(chrome, since),
            "yaw": sum(s["dy"] for s in samples),
        }
        # the panel: the switch, clicked; the legend and the key cap say what Shift does; it is kept
        await chrome.key("Escape", True)
        await chrome.key("Escape", False)
        await chrome.wait_for("!document.querySelector('#overlay').hidden")
        x, y = await chrome.eval("harness.centre('#walk + .switch')")
        await chrome.click(x, y)
        await asyncio.sleep(0.1)
        report["clicked"] = await chrome.eval("""({
              on: document.querySelector('#walk').checked,
              says: document.querySelector('#walk-says').textContent,
              legend: document.querySelector('#shift-says').textContent,
              title: document.querySelector('.cap[data-b=speed]').title,
              stored: localStorage.getItem('wc.walk'),
            })""")
        await engage(chrome)
        report["w_after"] = await step(chrome, ("KeyW", True), ("KeyW", False))
    return report


@pytest.fixture(scope="module")
def walking(recorded, tmp_path_factory) -> dict:
    """What the page sent with Walk by default on (nothing kept), then clicked off."""
    return asyncio.run(keyboard_session(recorded, tmp_path_factory.mktemp("walking"), None))


@pytest.fixture(scope="module")
def running(recorded, tmp_path_factory) -> dict:
    """What the page sent with Walk by default kept off, then clicked on."""
    storage = {"wc.walk": "false"}
    return asyncio.run(keyboard_session(recorded, tmp_path_factory.mktemp("running"), storage))


def test_walk_by_default_is_on_and_its_panel_says_what_shift_does(walking):
    """Nothing kept: the switch is on, the row says the player walks while a movement key is held
    and Shift runs, the legend says Shift runs and so does the key cap's title; a click turns it
    off, kept as ``wc.walk`` false, and all three say Shift walks."""
    panel, clicked = walking["panel"], walking["clicked"]
    assert panel["on"] is True and panel["stored"] is None
    assert panel["says"] == "You walk while a movement key is held; Shift runs"
    assert (panel["legend"], panel["title"]) == ("run", "Shift runs")
    assert clicked["on"] is False and clicked["stored"] == "false"
    assert clicked["says"] == "You run; Shift walks"
    assert (clicked["legend"], clicked["title"]) == ("walk", "Shift walks")


def test_a_movement_key_holds_the_walk_button_in_the_sample_that_carries_it(walking):
    """W alone: the first sample that holds forward holds the walk button with it, and so do all
    that follow; W let go lets go of both in one sample. S, and A and D, the same."""
    assert within(walking["w"], 0, FORWARD | SPEED) and walking["w"][-1] == FORWARD | SPEED
    assert within(walking["w_up"], FORWARD | SPEED, 0) and walking["w_up"][-1] == 0
    assert within(walking["s"], 0, BACK | SPEED) and walking["s"][-2:] == [BACK | SPEED, 0]
    # A, then D, then A let go, then D: the walk button goes with them while one is held
    assert walking["a_d"][-4:] == [LEFT | SPEED, LEFT | RIGHT | SPEED, RIGHT | SPEED, 0]


def test_shift_runs_by_letting_go_of_the_walk_button_and_alone_holds_nothing(walking):
    """Shift alone: no button. W held and Shift pressed: the walk button is let go at once (the
    sample is forward alone); Shift let go: it is held again. Shift down, then W: forward alone, the
    walk button never held."""
    assert within(walking["shift"], 0) and within(walking["shift_up"], 0)
    assert (
        within(walking["w_shift"], FORWARD | SPEED, FORWARD) and walking["w_shift"][-1] == FORWARD
    )
    assert within(walking["w_unshift"], FORWARD, FORWARD | SPEED)
    assert walking["w_unshift"][-1] == FORWARD | SPEED
    assert within(walking["shift_w"], 0, FORWARD) and walking["shift_w"][-1] == FORWARD
    assert within(walking["shift_w_up"], FORWARD, 0) and walking["shift_w_up"][-1] == 0
    assert within(walking["shift_released"], 0)  # W let go with Shift still held: nothing


def test_the_key_cap_of_shift_lights_for_the_key_and_not_for_the_walk_the_page_holds(walking):
    """The caps show what the player holds: Shift alone lights Shift; W lights W and not Shift,
    though the samples hold the walk button; W with Shift lights both."""
    assert walking["shift_cap"] is True
    assert walking["w_cap"] == [True, False]
    assert walking["w_shift_cap"] == [True, True]


def test_an_arrow_key_and_the_mouse_turn_the_view_and_do_not_move_the_player(walking):
    """An arrow key is no movement key: the samples hold no button while it turns the view, and
    nor do they for the mouse's turn."""
    arrow, mouse = walking["arrow"], walking["mouse"]
    assert arrow["masks"] == [0] and arrow["yaw"] > 10
    assert mouse["masks"] == [0] and mouse["yaw"] == pytest.approx(9.0, abs=0.01)


def test_with_walk_by_default_off_shift_walks_as_before(running):
    """Kept off: a movement key holds its button alone, Shift alone holds the walk button, Shift
    with W holds both and W let go leaves Shift's; the switch is off and the panel says Shift walks.
    """
    panel = running["panel"]
    assert panel["on"] is False and panel["stored"] == "false"
    assert panel["says"] == "You run; Shift walks"
    assert (panel["legend"], panel["title"]) == ("walk", "Shift walks")
    assert within(running["w"], 0, FORWARD) and running["w"][-1] == FORWARD
    assert within(running["w_up"], FORWARD, 0) and running["w_up"][-1] == 0
    assert within(running["shift"], 0, SPEED) and running["shift"][-1] == SPEED
    assert within(running["shift_up"], SPEED, 0) and running["shift_up"][-1] == 0
    assert within(running["w_shift"], FORWARD, FORWARD | SPEED)
    assert running["w_shift"][-1] == FORWARD | SPEED and running["w_unshift"][-1] == FORWARD
    assert running["shift_w"][-2:] == [SPEED, FORWARD | SPEED]
    assert within(running["shift_w_up"], FORWARD | SPEED, SPEED)
    assert running["shift_w_up"][-1] == SPEED and running["shift_released"][-1] == 0
    assert running["arrow"]["masks"] == [0] and running["mouse"]["masks"] == [0]


def test_a_click_on_the_switch_turns_walking_on_and_it_is_kept(running):
    """Off kept, then clicked on: ``wc.walk`` true, and the next movement key holds the walk
    button."""
    clicked = running["clicked"]
    assert clicked["on"] is True and clicked["stored"] == "true"
    assert clicked["says"] == "You walk while a movement key is held; Shift runs"
    assert within(running["w_after"], 0, FORWARD | SPEED)
    assert running["w_after"][-2:] == [FORWARD | SPEED, 0]


async def stick_session(recorded: dict, directory: Path) -> dict:
    """The left stick of a gamepad: inside its dead zone of movement, past it forward, let go."""
    report: dict = {}
    async with play(recorded, directory) as chrome:
        await engage(chrome)
        await chrome.eval("harness.pad.connect()")
        for name, axes in (
            ("inside", [0, -0.3, 0, 0]),
            ("forward", [0, -0.6, 0, 0]),
            ("diagonal", [0.6, -0.6, 0, 0]),
            ("let_go", [0, 0, 0, 0]),
        ):
            since = await chrome.eval(f"harness.pad.set({axes}, [])")
            await asyncio.sleep(SETTLE_S)
            report[name] = await masks(chrome, since)
    return report


def test_the_left_stick_walks_once_it_holds_a_direction_and_stops_with_it(recorded, tmp_path):
    """Inside the stick's dead zone for movement (0.4 of its travel) nothing is held, the walk
    button included; past it the direction and the walk button go together, a second direction
    joins them, and the stick let go lets go of all."""
    report = asyncio.run(stick_session(recorded, tmp_path))
    assert within(report["inside"], 0)
    assert (
        within(report["forward"], 0, FORWARD | SPEED) and report["forward"][-1] == FORWARD | SPEED
    )
    assert within(report["diagonal"], FORWARD | SPEED, FORWARD | RIGHT | SPEED)
    assert report["diagonal"][-1] == FORWARD | RIGHT | SPEED
    assert within(report["let_go"], FORWARD | RIGHT | SPEED, 0) and report["let_go"][-1] == 0
