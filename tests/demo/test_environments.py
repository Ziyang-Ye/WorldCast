"""The page in other people's browsers, in headless Chrome against a mock demo.

The page as served (``static/``) plays a mock session in a Chrome made to look like other
machines: a Linux and a Windows user agent and platform, a device pixel ratio of 1 to 2, windows
from 1280 x 720 to an ultrawide one, classic scrollbars that take room from the page, no WebGL at
all, a display that refreshes at 240 Hz, reduced motion, a forced-colours theme. In each of them it
opens, shows frames, takes keys, keeps the stage, the HUD and the pause panel apart and inside the
window, and reports no error to the console. It draws the frames by the instant turn (WebGL2) or,
with no WebGL, as they were generated on a 2D canvas; where storage is blocked it still seats the
player, and a view the browser cannot open says so; a tab that closes while one plays asks first,
and the keys the browsers act on (Ctrl and Alt with a letter or an arrow) are the page's while one
plays. The tests skip without Chrome or Chromium.
"""

import asyncio
import base64
import io
import tempfile
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from PIL import Image

from tests.demo.support import (
    STATIC,
    ChromePage,
    chrome_page,
    create_room,
    demo,
    needs_chrome,
    take_seat,
    until,
)
from worldcast.data.latents import FRAME_SIZE

pytestmark = needs_chrome

HEIGHT, WIDTH = FRAME_SIZE
CLOSE = pytest.approx
#: The most a frame is drawn per second: the blocks' rate, which a faster display does not raise.
MAX_FPS = 24
#: The most input samples sent per second, whatever the display's rate: 60, and a sample more.
MAX_SAMPLES = 62

#: A user agent, the platform string of ``navigator.platform`` and the platform of its client hints.
LINUX = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36",
    "Linux x86_64",
    "Linux",
)
WINDOWS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36",
    "Win32",
    "Windows",
)
#: Scrollbars that take 15 px from the page, as Windows and Linux draw them (a Mac's overlay ones
#: take none).
CLASSIC_SCROLLBARS = (
    "document.addEventListener('DOMContentLoaded', () => document.head.append(Object.assign("
    "document.createElement('style'), {textContent: '::-webkit-scrollbar { width: 15px;"
    " height: 15px; background: #222 } ::-webkit-scrollbar-thumb { background: #666 }'})));"
)
#: A display that refreshes 240 times a second: the page's animation frames come that often.
HIGH_REFRESH = """
window.requestAnimationFrame = (f) => setTimeout(() => f(performance.now()), 1000 / 240);
window.cancelAnimationFrame = (id) => clearTimeout(id);
"""
#: A browser that gives no WebGL context (a headless shell with no GPU still has a software one).
NO_WEBGL = """
const getContext = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
  return /webgl/.test(type) ? null : getContext.call(this, type, ...rest);
};
"""
#: What a page script collects for the tests: every key event with whether the page took it, the
#: input samples sent, and the animation frames.
WATCH = """
window.__keys = [];
for (const type of ['keydown', 'keyup']) {
  window.addEventListener(type, (e) => window.__keys.push({ type, code: e.code,
    taken: e.defaultPrevented }));
}
window.__samples = 0;
const send = WebSocket.prototype.send;
WebSocket.prototype.send = function (data) {
  if (typeof data === 'string' && data.includes('"t":"in"')) window.__samples += 1;
  return send.call(this, data);
};
window.__animation = 0;
requestAnimationFrame(function count() { window.__animation += 1; requestAnimationFrame(count); });
"""
#: The measurements of the page: where its boxes are (null: not shown), the window, how far each
#: scrolls, what the stage is, whether Play can be reached, how the page is painted.
LAYOUT = """(() => {
  const q = (s) => document.querySelector(s);
  const box = (el) => {
    if (!el) return null;
    const style = getComputedStyle(el);
    const b = el.getBoundingClientRect();
    return style.display === 'none' || style.visibility === 'hidden' || !b.width || !b.height
      ? null : [b.left, b.top, b.width, b.height];
  };
  const view = document.getElementById('view');
  const go = box(q('#go'));
  const held = q('.cap.on');
  return {
    viewport: [innerWidth, innerHeight, devicePixelRatio],
    boxes: Object.fromEntries(['.who', '.stats', '.bezel', '#stage', '.keys', '.flank-left',
      '.flank-right'].map((s) => [s, box(q(s))])),
    panel: box(q('.overlay .panel')),
    go,
    reach: go ? document.elementFromPoint(go[0] + go[2] / 2, go[1] + go[3] / 2)?.closest('#go')
      !== null : null,
    document: [document.documentElement.scrollWidth, document.documentElement.clientWidth],
    play: [q('.play').scrollWidth, q('.play').clientWidth, q('.play').scrollHeight,
      q('.play').clientHeight, getComputedStyle(q('.play')).position === 'fixed'],
    native: q('#stage').classList.contains('native'),
    view: [view.width, view.height],
    context: view.getContext('webgl2') ? 'webgl2' : view.getContext('2d') ? '2d' : null,
    fps: Number(q('#fps').textContent),
    scrollbar: innerWidth - document.documentElement.clientWidth,
    colours: [getComputedStyle(document.body).backgroundColor,
      getComputedStyle(document.body).color],
    held: held ? getComputedStyle(held).backgroundColor : null,
    idle: getComputedStyle(q('.cap:not(.on)')).backgroundColor,
    fade: getComputedStyle(q('.toast')).transitionDuration,
    blur: getComputedStyle(q('.overlay')).backdropFilter,
    media: ['(forced-colors: active)', '(prefers-reduced-motion: reduce)',
      '(prefers-color-scheme: light)'].map((m) => matchMedia(m).matches),
  };
})()"""
#: The stage's rectangle.
STAGE = (
    "(() => { const b = document.querySelector('#stage').getBoundingClientRect();"
    " return [b.left, b.top, b.width, b.height]; })()"
)
PLAYING = "Number(document.querySelector('#fps').textContent) > 0"
HELD = {
    "KeyW": ("w", 87, ".cap[data-b=forward]"),
    "ArrowLeft": ("ArrowLeft", 37, ".cap[data-k=ArrowLeft]"),
}


@dataclass(frozen=True)
class Environment:
    """A machine the page runs on: the window ``(width, height, device pixels per CSS pixel)``,
    its user agent, whether WebGL is there, the media features it asks for, whether its scrollbars
    take room, and the scripts that run before the page's."""

    window: tuple[int, int, int | float]
    agent: tuple[str, str, str] | None = None
    webgl: bool = True
    media: dict[str, str] = field(default_factory=dict)
    scrollbars: bool = False
    scripts: Sequence[str] = ()


#: The machines: a Linux laptop at 1280 x 720, a Linux monitor at 125% (1536 x 864 CSS pixels), a
#: Windows laptop at 150% (1280 x 720), a Windows QHD monitor, an ultrawide, a retina window, a
#: Linux laptop whose browser has no WebGL, a display of 240 Hz, a narrow window (portrait), one too
#: short for the page, a half screen, a tablet-sized window, a reader of reduced motion who prefers
#: a light page, and a forced-colours theme.
ENVIRONMENTS = {
    "linux-720": Environment((1280, 720, 1), LINUX),
    "linux-125": Environment((1536, 864, 1.25), LINUX, scrollbars=True),
    "windows-150": Environment((1280, 720, 1.5), WINDOWS, scrollbars=True),
    "windows-qhd": Environment((2560, 1440, 1), WINDOWS),
    "ultrawide": Environment((3440, 1440, 1), LINUX),
    "retina": Environment((1440, 900, 2), WINDOWS),
    "no-webgl": Environment((1366, 650, 1), LINUX, webgl=False, scripts=[NO_WEBGL]),
    "high-refresh": Environment((1280, 720, 1), LINUX, scripts=[HIGH_REFRESH]),
    "narrow": Environment((600, 800, 1), WINDOWS, scrollbars=True),
    "narrow-short": Environment((600, 560, 1), WINDOWS, scrollbars=True),
    "short": Environment((1280, 500, 1), WINDOWS, scrollbars=True),
    "half-screen": Environment((960, 1000, 1), LINUX, scrollbars=True),
    "tablet": Environment((1024, 768, 1), WINDOWS, scrollbars=True),
    "quiet": Environment(
        (1280, 720, 1),
        LINUX,
        media={
            "prefers-reduced-motion": "reduce",
            "prefers-reduced-transparency": "reduce",
            "prefers-color-scheme": "light",
        },
    ),
    "forced-colors": Environment((1280, 720, 1), WINDOWS, media={"forced-colors": "active"}),
}


def commands(environment: Environment) -> list[tuple[str, dict]]:
    """The DevTools commands that make Chrome the machine of ``environment``."""
    found: list[tuple[str, dict]] = [
        ("Page.enable", {}),
        ("Runtime.enable", {}),
        ("Log.enable", {}),
    ]
    if environment.agent:
        agent, platform, family = environment.agent
        metadata = dict(
            platform=family,
            platformVersion="6.5.0",
            architecture="x86",
            model="",
            mobile=False,
            brands=[dict(brand="Chromium", version="140")],
        )
        args = dict(userAgent=agent, platform=platform, userAgentMetadata=metadata)
        found.append(("Emulation.setUserAgentOverride", args))
    if environment.media:
        features = [dict(name=name, value=value) for name, value in environment.media.items()]
        found.append(("Emulation.setEmulatedMedia", dict(features=features)))
    if environment.scrollbars:
        found.append(("Emulation.setScrollbarsHidden", dict(hidden=False)))
    scripts = [*environment.scripts, WATCH, *([CLASSIC_SCROLLBARS] * environment.scrollbars)]
    return found + [
        ("Page.addScriptToEvaluateOnNewDocument", dict(source=f"(() => {{ {s} }})();"))
        for s in scripts
    ]


def errors(chrome: ChromePage) -> list[str]:
    """What the page said to the console as an error, and the exceptions it threw."""
    found = []
    for event in chrome.events:
        params = event["params"]
        if event["method"] == "Runtime.exceptionThrown":
            found.append(params["exceptionDetails"].get("exception", {}).get("description", "?"))
        elif event["method"] == "Runtime.consoleAPICalled" and params["type"] == "error":
            found.append(
                " ".join(str(a.get("value", a.get("description"))) for a in params["args"])
            )
        elif event["method"] == "Log.entryAdded" and params["entry"]["level"] == "error":
            found.append(f"{params['entry']['text']} {params['entry'].get('url', '')}")
    return found


async def click(chrome: ChromePage, selector: str) -> None:
    """A click on the middle of the element ``selector``."""
    x, y = await chrome.eval(
        f"(() => {{ const b = document.querySelector({selector!r}).getBoundingClientRect();"
        " return [b.left + b.width / 2, b.top + b.height / 2]; })()"
    )
    await chrome.click(x, y)


async def take_first_seat(chrome: ChromePage, room: str) -> None:
    """Open the room's seat picker and join at the first free seat."""
    await chrome.wait_for("document.querySelectorAll('.round').length > 0")
    await chrome.eval(f"location.hash = '#/room/{room}'")
    await chrome.wait_for("document.querySelectorAll('.seat:not([disabled])').length > 0")
    await asyncio.sleep(0.2)
    await click(chrome, ".seat:not([disabled])")
    await click(chrome, "#join")


async def sit(chrome: ChromePage, room: str) -> None:
    """Take the first seat and wait at the pause panel."""
    await take_first_seat(chrome, room)
    await chrome.wait_for("!!document.querySelector('#go')")
    await asyncio.sleep(0.2)


async def play(chrome: ChromePage) -> None:
    """Press Play and wait until frames show."""
    await click(chrome, "#go")
    await chrome.wait_for(PLAYING + " && document.querySelector('#overlay').hidden")
    await asyncio.sleep(0.3)


async def painted(chrome: ChromePage) -> float:
    """The spread of the stage's pixels as the screen shows them (standard deviation of their
    luminance, 0-255): a blank view has none."""
    x, y, width, height = await chrome.eval(STAGE)
    clip = dict(x=x, y=y, width=width, height=height, scale=1)
    shot = await chrome.call("Page.captureScreenshot", format="png", clip=clip)
    pixels = list(Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("L").getdata())
    mean = sum(pixels) / len(pixels)
    return (sum((p - mean) ** 2 for p in pixels) / len(pixels)) ** 0.5


async def press(
    chrome: ChromePage, code: str, down: bool, name: str, virtual: int, modifiers: int = 0
) -> None:
    """A key down or up (``modifiers``: 1 Alt, 2 Ctrl, 4 Meta, 8 Shift) as a keyboard sends it."""
    await chrome.call(
        "Input.dispatchKeyEvent",
        type="rawKeyDown" if down else "keyUp",
        code=code,
        key=name,
        windowsVirtualKeyCode=virtual,
        modifiers=modifiers,
    )


async def tap(chrome: ChromePage, code: str, name: str, virtual: int, modifiers: int = 0) -> None:
    """A key pressed and let go."""
    await press(chrome, code, True, name, virtual, modifiers)
    await press(chrome, code, False, name, virtual, modifiers)


async def rates(chrome: ChromePage) -> dict[str, float]:
    """Animation frames and input samples per second while one plays."""
    read = "[window.__animation, window.__samples, performance.now()]"
    before = await chrome.eval(read)
    await asyncio.sleep(1.0)
    after = await chrome.eval(read)
    seconds = (after[2] - before[2]) / 1000
    return {
        "animation": (after[0] - before[0]) / seconds,
        "samples": (after[1] - before[1]) / seconds,
    }


async def visit(environment: Environment, local, http, room: str) -> dict:
    """The page in ``environment``: its layout paused, playing, with three more players and paused
    again, what a frame looks like, what two keys light, and what it said to the console."""
    report: dict = {}
    with tempfile.TemporaryDirectory() as directory:
        async with chrome_page(
            None,
            environment.window,
            Path(directory),
            url=local.url,
            webgl=environment.webgl,
            commands=commands(environment),
        ) as chrome:
            await sit(chrome, room)
            report["paused"] = await chrome.eval(LAYOUT)
            await play(chrome)
            report["playing"] = await chrome.eval(LAYOUT)
            report["painted"] = await painted(chrome)
            report["rates"] = await rates(chrome)
            report["keys"], report["lit"] = {}, {}
            for code, (name, virtual, cap) in HELD.items():
                await press(chrome, code, True, name, virtual)
                await asyncio.sleep(0.25)
                lit = f"document.querySelector({cap!r}).classList.contains('on')"
                down = await chrome.eval(lit)
                report["lit"][code] = (await chrome.eval(LAYOUT))["held"]
                await press(chrome, code, False, name, virtual)
                await asyncio.sleep(0.25)
                report["keys"][code] = (down, await chrome.eval(lit))
            others = [await take_seat(local, http, room, slot) for slot in (1, 2, 3)]
            await chrome.wait_for("document.querySelectorAll('#roster li').length === 4")
            await asyncio.sleep(0.3)
            report["crowd"] = await chrome.eval(LAYOUT)
            for other in others:
                await other.close()
            await tap(chrome, "Escape", "Escape", 27)
            await chrome.wait_for("!document.querySelector('#overlay').hidden")
            report["escape"] = await chrome.eval(LAYOUT)
            await click(chrome, "#leave")
            await until(
                lambda: all(w.room is None for w in local.coordinator.workers.values()),
                "the workers to be free",
            )
            report["errors"] = errors(chrome)
    return report


async def environment_reports() -> dict[str, dict]:
    reports = {}
    async with demo(4) as (local, http):
        for name, environment in ENVIRONMENTS.items():
            reports[name] = await visit(environment, local, http, await create_room(local, http))
    return reports


@pytest.fixture(scope="module")
def reports() -> dict[str, dict]:
    """What the page did in every environment (:func:`visit`)."""
    return asyncio.run(environment_reports())


@asynccontextmanager
async def session(scripts: Sequence[str] = (), webgl: bool = True, seat: bool = True):
    """A page in a Chrome on a mock demo, after the scripts ``scripts`` and seated at the first
    seat of a room (``seat`` false: waiting at the lobby), at its pause panel. Yields the Chrome,
    the demo, its HTTP session and the room."""
    window = (1280, 720, 1)
    environment = Environment(window, LINUX, webgl=webgl, scripts=scripts)
    with tempfile.TemporaryDirectory() as directory:
        async with demo(1) as (local, http):
            room = await create_room(local, http)
            async with chrome_page(
                None,
                window,
                Path(directory),
                url=local.url,
                webgl=webgl,
                commands=commands(environment),
            ) as chrome:
                if seat:
                    await sit(chrome, room)
                else:
                    await chrome.wait_for("document.querySelectorAll('.round').length > 0")
                yield chrome, local, http, room


def run(scenario):
    """Run an async scenario of :func:`session`."""
    return asyncio.run(scenario())


def intersects(a: list[float], b: list[float]) -> bool:
    """Whether two ``[x, y, width, height]`` rectangles share area, but for half a pixel."""
    across = a[0] + 0.5 < b[0] + b[2] and b[0] + 0.5 < a[0] + a[2]
    return across and a[1] + 0.5 < b[1] + b[3] and b[1] + 0.5 < a[1] + a[3]


def test_the_page_plays_the_mock_session_on_every_machine(reports):
    """In each environment frames show (at the blocks' rate whatever the display's), input samples
    go out at most 60 a second (the display of 240 Hz included), the stage is painted, no error
    reaches the console, and the view draws the frames by the instant turn on a WebGL2 canvas of
    their own pixels, or with no WebGL as they are on a 2D one."""
    for name, report in reports.items():
        playing = report["playing"]
        assert report["errors"] == [], name
        assert 0 < playing["fps"] <= MAX_FPS and report["painted"] > 5, name
        assert playing["view"] == [WIDTH, HEIGHT], name
        assert playing["context"] == ("webgl2" if ENVIRONMENTS[name].webgl else "2d"), name
        assert 0 < report["rates"]["samples"] <= MAX_SAMPLES, name
    assert reports["high-refresh"]["rates"]["animation"] > 150


def test_the_keys_light_their_caps_in_every_environment(reports):
    """W and the left arrow light their key caps while held and let them go, and a held cap is
    told from an idle one by its colours, in a forced-colours theme too."""
    for name, report in reports.items():
        assert report["keys"] == {"KeyW": (True, False), "ArrowLeft": (True, False)}, name
        for held in report["lit"].values():
            assert held is not None and held != report["playing"]["idle"], name


def test_the_stage_the_hud_and_the_panels_hold_in_every_window(reports):
    """The stage keeps the frame's aspect ratio and, in a window of 1000 x 650 or more, its own
    size; the HUD, the key caps and the cards stand clear of each other and of the stage, inside
    the page's width (its scrollbar counted), and the page does not scroll sideways; a window that
    is not a phone's does not scroll the play view; the pause panel is whole in the window and its
    Play button reaches the mouse."""
    for name, report in reports.items():
        for stage in ("paused", "playing", "crowd", "escape"):
            layout = report[stage]
            width, height, ratio = layout["viewport"]
            page = layout["document"][1]
            where = (name, stage)
            assert layout["document"][0] <= page and layout["play"][0] <= layout["play"][1], where
            x, y, w, h = layout["boxes"]["#stage"]
            assert w / h == CLOSE(WIDTH / HEIGHT, abs=0.01) and w <= WIDTH + 0.5, where
            if width >= 1000 and height >= 650:
                assert [w, h] == CLOSE([WIDTH, HEIGHT], abs=0.5), where
            assert layout["native"] == (abs(w - WIDTH) < 0.01 and ratio == int(ratio)), where
            scroll_height, client_height, fixed = layout["play"][2:]
            assert not fixed or scroll_height <= client_height, where
            boxes = {s: b for s, b in layout["boxes"].items() if b and s != "#stage"}
            for s, (bx, by, bw, bh) in boxes.items():
                assert bx >= -0.5 and bx + bw <= page + 0.5 and bh > 0, (where, s)
            names = list(boxes)
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    assert not intersects(boxes[a], boxes[b]), (where, a, b)
        panel, go = report["paused"]["panel"], report["paused"]["go"]
        assert report["paused"]["reach"] and report["escape"]["reach"], name
        for box in (panel, go):
            assert box[0] >= 0 and box[0] + box[2] <= width and box[1] >= 0, name
            assert box[1] + box[3] <= height, name


def test_the_page_follows_the_settings_of_the_machine(reports):
    """The page stays dark on a machine that prefers a light page, makes its transitions
    instant for a reader of reduced motion and draws no blurred backdrop for one of reduced
    transparency, and in a forced-colours theme still shows which key is held."""
    quiet = reports["quiet"]["playing"]
    assert quiet["media"] == [False, True, True] and quiet["blur"] == "none"
    assert reports["linux-720"]["playing"]["blur"] != "none"
    assert quiet["colours"] == ["rgb(5, 5, 6)", "rgb(245, 245, 247)"] and quiet["fade"] == "1e-05s"
    assert reports["forced-colors"]["playing"]["media"][0]
    assert reports["linux-720"]["playing"]["media"][:2] == [False, False]
    assert reports["linux-720"]["playing"]["fade"] == "0.3s"


def test_the_page_reads_no_user_agent_or_platform():
    """The page is the same on every machine: it asks the browser for features, never for who it
    is (a gamepad's id names the device, not the browser, and ``pad.js`` reads its vendor and
    product)."""
    for path in sorted(STATIC.glob("*.js")):
        text = path.read_text(encoding="utf-8")
        names = ("userAgent", "navigator.platform", "navigator.vendor")
        assert not any(name in text for name in names), path


def test_a_browser_that_keeps_nothing_still_seats_the_player_and_plays():
    """Blocked storage (cookies refused, a private window) throws at every read: the seat is kept
    in memory, the settings are not kept."""
    script = """
    for (const name of ['localStorage', 'sessionStorage']) {
      Object.defineProperty(window, name, {
        get() { throw new DOMException('no', 'SecurityError'); },
      });
    }
    """

    async def scenario():
        async with session([script]) as (chrome, *_):
            await play(chrome)
            assert (await chrome.eval(LAYOUT))["fps"] > 0 and await painted(chrome) > 5
            assert errors(chrome) == []

    run(scenario)


def test_a_view_the_browser_cannot_open_says_so_and_leads_back_to_the_lobby():
    """A browser that refuses the play socket (a page on https and a worker on http, a bad
    address) throws at its construction: the page says so, where it was blank."""
    script = """
    const Socket = window.WebSocket;
    window.WebSocket = function (url, ...rest) {
      if (String(url).includes('/play')) throw new DOMException('Mixed content', 'SecurityError');
      return new Socket(url, ...rest);
    };
    window.WebSocket.OPEN = Socket.OPEN;
    """

    async def scenario():
        async with session([script], seat=False) as (chrome, _, __, room):
            await take_first_seat(chrome, room)
            await chrome.wait_for(
                "document.querySelector('.page-title')?.textContent" " === 'This view did not open'"
            )
            assert "Mixed content" in await chrome.eval(
                "document.querySelector('.page-sub').textContent"
            )
            assert await chrome.eval("document.body.classList.contains('is-play')") is False
            await click(chrome, ".back")
            await chrome.wait_for("document.querySelectorAll('.round').length > 0")

    run(scenario)


def test_a_tab_that_closes_while_one_plays_asks_first():
    """Ctrl+W is the browser's, with Ctrl held to crouch: the page cannot take it, so it asks
    before the tab goes while one plays, and not while paused."""
    ask = (
        "(() => { const e = new Event('beforeunload', { cancelable: true });"
        " window.dispatchEvent(e); return e.defaultPrevented; })()"
    )

    async def scenario():
        async with session() as (chrome, *_):
            assert await chrome.eval(ask) is False
            await play(chrome)
            assert await chrome.eval(ask) is True
            await tap(chrome, "Escape", "Escape", 27)
            await chrome.wait_for("!document.querySelector('#overlay').hidden")
            assert await chrome.eval(ask) is False

    run(scenario)


#: Keys, by code: the key and its virtual key code.
KEYBOARD = {
    "KeyW": ("w", 87),
    "KeyA": ("a", 65),
    "KeyS": ("s", 83),
    "KeyD": ("d", 68),
    "KeyR": ("r", 82),
    "KeyF": ("f", 70),
    "KeyC": ("c", 67),
    "Space": (" ", 32),
    "ArrowLeft": ("ArrowLeft", 37),
    "ArrowUp": ("ArrowUp", 38),
    "ArrowRight": ("ArrowRight", 39),
    "ArrowDown": ("ArrowDown", 40),
    "F5": ("F5", 116),
    "F11": ("F11", 122),
    "F12": ("F12", 123),
}
#: What a browser acts on that the page takes while one plays (the code, the modifiers: 1 Alt, 2
#: Ctrl): Ctrl+D does not bookmark, Ctrl+S save, Ctrl+A select, Ctrl+R reload, Ctrl+F find, Alt+Left
#: go back, and Space does not scroll.
TAKEN = [(code, 2) for code in ("KeyW", "KeyA", "KeyS", "KeyD", "KeyR", "KeyF", "KeyC")] + [
    ("ArrowLeft", 2),
    ("ArrowRight", 2),
    ("ArrowUp", 1),
    ("ArrowLeft", 1),
    ("ArrowRight", 1),
    ("Space", 0),
    ("ArrowDown", 0),
]
#: The browser's own keys, which the page leaves alone: reload, fullscreen, the developer tools.
LEFT = ["F5", "F11", "F12"]


def test_the_keys_browsers_act_on_are_the_pages_while_one_plays_and_the_browsers_own_are_not():
    async def scenario():
        async with session() as (chrome, *_):
            await play(chrome)
            await chrome.eval("window.__keys.length = 0")
            for code, modifiers in [*TAKEN, *[(code, 0) for code in LEFT]]:
                await tap(chrome, code, *KEYBOARD[code], modifiers)
            await asyncio.sleep(0.2)
            seen = await chrome.eval("window.__keys.filter((k) => k.type === 'keydown')")
            assert [(k["code"], k["taken"]) for k in seen] == [
                *[(code, True) for code, _ in TAKEN],
                *[(code, False) for code in LEFT],
            ]

    run(scenario)


def test_a_window_that_loses_the_focus_lets_go_of_the_keys():
    """Alt-Tab, the Super key and a click on another window blur the page, whose keys' releases
    never reach it: it lets go of what was held and leaves the view uncovered."""

    async def scenario():
        async with session() as (chrome, *_):
            await play(chrome)
            await press(chrome, "KeyW", True, "w", 87)
            await asyncio.sleep(0.25)
            lit = "document.querySelector('.cap[data-b=forward]').classList.contains('on')"
            assert await chrome.eval(lit) is True
            await chrome.eval("window.dispatchEvent(new Event('blur'))")
            await asyncio.sleep(0.2)
            assert await chrome.eval(lit) is False
            assert await chrome.eval(
                "document.querySelector('.play').classList.contains('unfocused')"
            )
            assert await chrome.eval("document.querySelector('#overlay').hidden")
            await press(chrome, "KeyW", False, "w", 87)

    run(scenario)


async def reload_with(chrome: ChromePage, url: str, replaced: dict[str, str]) -> None:
    """Load ``url`` with the page's scripts of the names ``replaced`` served as the texts given."""
    patterns = [dict(urlPattern=f"*/static/{name}") for name in replaced]
    await chrome.call("Fetch.enable", patterns=patterns)
    await chrome.call("Page.navigate", url=url)
    notice = "document.getElementById('unsupported')?.hidden === false"
    while not await chrome.eval(notice):
        paused = [e for e in chrome.events if e["method"] == "Fetch.requestPaused"]
        for event in paused:
            chrome.events.remove(event)
            name = event["params"]["request"]["url"].rsplit("/", 1)[1]
            body = base64.b64encode(replaced[name].encode()).decode()
            headers = [dict(name="Content-Type", value="text/javascript")]
            await chrome.call(
                "Fetch.fulfillRequest",
                requestId=event["params"]["requestId"],
                responseCode=200,
                responseHeaders=headers,
                body=body,
            )
        await asyncio.sleep(0.05)


def test_a_browser_that_cannot_load_or_parse_the_pages_scripts_says_so():
    """A script that is blocked and one that does not parse (a browser too old for the page's
    syntax) leave no blank page: the shell says the browser cannot run it, where the page, once
    its scripts ran, has no notice."""

    async def scenario():
        async with session(seat=False) as (chrome, local, *_):
            assert await chrome.eval("document.documentElement.dataset.ready") == "1"
            assert await chrome.eval("document.getElementById('unsupported')") is None
            await chrome.call("Network.enable")
            await chrome.call("Network.setBlockedURLs", urls=["*/static/app.js"])
            await chrome.call("Page.navigate", url=local.url)
            await chrome.wait_for("document.getElementById('unsupported')?.hidden === false")
            text = await chrome.eval("document.getElementById('unsupported').textContent")
            assert "cannot run WorldCast Live" in text
            await chrome.call("Network.setBlockedURLs", urls=[])
            await reload_with(chrome, local.url, {"stream.js": "export class { static = ;\n"})
            assert await chrome.eval("document.documentElement.dataset.ready") is None

    run(scenario)


def test_the_screen_stays_awake_while_the_view_is_open_and_is_let_go_after():
    """A gamepad or the on-screen pad keeps no display from sleeping: the play view asks for a
    screen wake lock, and gives it back when it closes."""
    script = """
    window.__wake = { asked: [], released: 0 };
    Object.defineProperty(navigator, 'wakeLock', { value: { request: async (kind) => {
      window.__wake.asked.push(kind);
      return { release: async () => { window.__wake.released += 1; } };
    } } });
    """

    async def scenario():
        async with session([script]) as (chrome, *_):
            assert await chrome.eval("window.__wake.asked") == ["screen"]
            assert await chrome.eval("window.__wake.released") == 0
            await click(chrome, "#leave")
            await chrome.wait_for("window.__wake.released === 1")

    run(scenario)


def test_the_latency_meter_remembers_a_sample_by_its_age_at_any_display_rate():
    """A display of 240 Hz sends 240 samples a second: the meter keeps a sample's send time for
    10 s, not for 600 samples (2.5 s there), so a frame that folds in a sample sent three seconds
    ago still gives the end-to-end delay and the breakdown."""
    meter = """(async () => {
      const { LatencyMeter } = await import('/static/latency.js');
      const meter = new LatencyMeter();
      for (let q = 1; q <= 1500; q += 1) meter.sample(q, q * 4.2, q === 1 ? [0] : []);
      meter.frame({ q: 1, tm: [10, 300, 20] }, { shown: 3000, recv: 2990 });
      return [meter.fresh.length, meter.parts.length, meter.sent.size];
    })()"""

    async def scenario():
        async with session(seat=False) as (chrome, *_):
            fresh, parts, remembered = await chrome.eval(meter)
            assert (fresh, parts) == (1, 1) and remembered == 1500

    run(scenario)
