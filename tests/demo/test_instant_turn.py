"""The instant turn of the play view (``static/turn.js``) in headless Chrome with software WebGL2.

The play view's frame stream (``static/stream.js``) runs on a stand-in socket
(``instant_turn.html``) that sends frames of a known world, a sphere of colour around the eye, each
taken at the orientation its header says (``tn``, ``cp``) and with a viewmodel of a colour the
world never has, fixed on the frame. The player's turn is set at an animation frame and the view is
read back: where the frames shown have seen what the turned view shows, the next animation frame
draws the view turned by all of the turn, the world's own pixels where they belong; where they have
not, it turns as far as they reach. No pixel of the view is from nowhere, the viewmodel stays the
newest frame's own pixels and nothing of it goes into the panorama; a frame that holds the whole
turn is drawn as it was generated. A turn of half a circle, either way, shows at once where the
frames went round. Frames older than the panorama keeps them, or taken elsewhere, are not used, and
with the setting off every frame is drawn as it was generated. The tests skip without Chrome or
Chromium.

``WORLDCAST_TURN_VIDEO`` names a directory for the recording of a scripted session (an mp4 of the
canvas) and stills of turns into what the view has and has not seen, for a person to look at.
"""

import asyncio
import base64
import json
import math
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.demo.support import chrome_page, needs_chrome, recorded_play
from worldcast.data.controls import CONTROL_BUTTONS
from worldcast.data.game import (
    ASPECT_TAN_RATIO,
    HFOV_DEGREES,
    PITCH_MAX_DEGREES,
    PITCH_MIN_DEGREES,
)
from worldcast.data.latents import FRAME_SIZE
from worldcast.demo.inputs import INSPECT_WEAPONS

pytestmark = needs_chrome

HARNESS = Path(__file__).with_name("instant_turn.html")
MOTION_HARNESS = Path(__file__).with_name("motion_play.html")
ACROSS = math.tan(math.radians(HFOV_DEGREES) / 2)
PAGE = {
    "frame_pixels": [FRAME_SIZE[1], FRAME_SIZE[0]],
    "camera_tans": [ACROSS, ACROSS * ASPECT_TAN_RATIO],
    "pitch_range_degrees": [PITCH_MIN_DEGREES, PITCH_MAX_DEGREES],
}
WELCOME = {
    "t": "welcome",
    "fps": 16,
    "frames_per_block": 16,
    "jitter_buffer_frames": 1,
    "margin_frames": 1.5,
    "backlog_frames": 24,
    "buttons": list(CONTROL_BUTTONS),
    "inspect_weapons": sorted(INSPECT_WEAPONS),
    "tn": [0, 0],
}
#: A knife carried at a run: a frame row's buttons and weapon (its header's b and w).
KNIFE_RUN = {"b": 1 << CONTROL_BUTTONS.index("forward"), "w": min(INSPECT_WEAPONS)}
#: The most a view drawn from the frames differs from the world, mean per colour (0-255): JPEG and
#: the panorama's texels (about 1-2.5); a degree off is several times more on the world's checker.
WORLD_ERROR = 5.0
#: The most the viewmodel differs from the newest frame's own pixels, mean per colour: JPEG.
STILL_ERROR = 4.0
#: The magnification that hides a sliver at the view's edges (static/turn.js MAX_MAGNIFY).
MAX_MAGNIFY = 1.03
#: How far a turn into what the view has not seen may go, degrees: the sliver the magnification
#: hides.
UNSEEN_DEGREES = 1.5
VIDEO_ENV = "WORLDCAST_TURN_VIDEO"


@asynccontextmanager
async def turning(directory: Path, instant: bool = True):
    """The harness in a Chrome, ready for frames."""
    setup = {"page": PAGE, "welcome": WELCOME, "instant": instant}
    page = HARNESS.read_text(encoding="utf-8").replace("__SETUP__", json.dumps(setup))
    async with chrome_page(page, (700, 420, 1), directory) as chrome:
        await chrome.wait_for("window.ready === true")
        error = await chrome.eval("window.harness ? (harness.out.error ?? null) : 'no harness'")
        assert error is None
        assert await chrome.eval("!!stream.turn"), "the instant turn did not start (no WebGL2?)"
        yield chrome
        assert await chrome.eval("harness.errors") == []


async def show(chrome, poses, controls: dict | None = None) -> None:
    """Send a frame at each pose (yaw, pitch[, position]) and wait until it is on screen;
    `controls`: its row's buttons and weapon, and whether a knife is swung outside the viewmodel's
    pieces in it (instant_turn.html, harness.deliver)."""
    for yaw, pitch, *at in poses:
        position = json.dumps(at[0] if at else [0, 0, 0])
        given = json.dumps(controls or {})
        k = await chrome.eval(f"harness.deliver({yaw}, {pitch}, {position}, {given})")
        await chrome.wait_for(f"harness.shown() >= {k}")


async def turn(chrome, pitch: float, yaw: float) -> dict:
    """Turn the player to (pitch, yaw) at an animation frame. Returns the first drawing that has
    the turn, with `frames` the animation frames from the turn's to it (1: the next one)."""
    at = await chrome.eval(f"harness.turnTo({pitch}, {yaw})")
    wanted = f"(d) => d.look.yaw === {yaw} && Math.abs(d.look.pitch - {pitch}) < 1e-9"
    await chrome.wait_for(f"harness.draws({at['t']}).some({wanted})")
    draws = await chrome.eval(f"harness.draws({at['t']} - 200)")
    gaps = sorted(b["raf"] - a["raf"] for a, b in zip(draws, draws[1:]) if b["raf"] > a["raf"])
    first = await chrome.eval(f"harness.draws({at['t']}).find({wanted})")
    first["frames"] = round((first["raf"] - at["raf"]) / gaps[len(gaps) // 2], 2)
    return first


async def check(chrome, view: dict, source: str = "any") -> dict:
    """The canvas against the world at the view's orientation (instant_turn.html,
    harness.check)."""
    return await chrome.eval(f"harness.check({view['yaw']}, {view['pitch']}, '{source}')")


def looked_around(first: int, last: int, step: int = 10) -> list[tuple[float, float]]:
    """The poses of a look from `first` to `last` degrees of yaw and back, every `step`."""
    there = list(range(first, last + 1, step))
    return [(yaw, 0) for yaw in there + there[-2::-1]]


def test_a_turn_back_to_what_the_frames_have_seen_shows_at_the_next_animation_frame(tmp_path):
    async def run():
        async with turning(tmp_path) as chrome:
            await show(chrome, looked_around(0, 120))
            drawn = await turn(chrome, 0, 100)
            assert drawn["frames"] < 1.5
            assert drawn["view"] == {"yaw": 100, "pitch": 0}
            result = await check(chrome, drawn["view"])
            counts = result["counts"]
            assert counts["nowhere"] == 0
            assert counts["panorama"] > 0.9 * (counts["frame"] + counts["panorama"])
            assert result["error"] < WORLD_ERROR
            assert result["stray"] == 0 and result["still"] < STILL_ERROR
            assert result["kept"] >= 0.99 * result["held"] > 0
            # a degree off is far off the world: the check sees the turn
            assert (await chrome.eval("harness.check(101, 0)"))["error"] > 3 * result["error"]
            drawn = await turn(chrome, 0, 40)
            assert drawn["frames"] < 1.5 and drawn["view"] == {"yaw": 40, "pitch": 0}
            result = await check(chrome, drawn["view"])
            assert result["counts"]["nowhere"] == 0 and result["error"] < WORLD_ERROR

    asyncio.run(run())


def test_a_turn_into_what_the_frames_have_not_seen_goes_as_far_as_they_reach(tmp_path):
    async def run():
        async with turning(tmp_path) as chrome:
            await show(chrome, looked_around(0, 120))
            # right, where no frame looked: the view stays on the newest frame but for a sliver
            # the magnification hides
            drawn = await turn(chrome, 0, -100)
            assert drawn["frames"] < 1.5
            assert -UNSEEN_DEGREES <= drawn["view"]["yaw"] <= 0
            assert drawn["magnify"] <= MAX_MAGNIFY
            result = await check(chrome, drawn["view"])
            assert result["counts"]["nowhere"] == 0 and result["error"] < WORLD_ERROR
            # left past the frames: as far as the last of them, 120 degrees
            drawn = await turn(chrome, 0, 200)
            assert 120 - UNSEEN_DEGREES <= drawn["view"]["yaw"] <= 120 + UNSEEN_DEGREES
            result = await check(chrome, drawn["view"])
            assert result["counts"]["nowhere"] == 0 and result["counts"]["frame"] == 0
            assert result["error"] < WORLD_ERROR and result["stray"] == 0

    asyncio.run(run())


def test_a_flick_of_half_a_turn_shows_at_once_where_the_frames_went_round(tmp_path):
    async def run():
        async with turning(tmp_path) as chrome:
            await show(chrome, [(yaw, 0) for yaw in range(0, 360, 10)] + [(0, 0)])
            for yaw in (180, -180, 0):
                drawn = await turn(chrome, 0, yaw)
                assert drawn["frames"] < 1.5 and drawn["view"] == {"yaw": yaw, "pitch": 0}
                result = await check(chrome, drawn["view"])
                assert result["counts"]["nowhere"] == 0 and result["error"] < WORLD_ERROR
                assert result["stray"] == 0
                # the viewmodel stays as the newest frame has it; the world around it in its
                # pieces, which the frames have seen beside it, turns with the rest
                assert result["kept"] >= 0.99 * result["held"] > 0
                assert result["counts"]["viewmodel"] < 1.05 * result["held"]
            # from 50 u away nothing of them is used: half a turn shows none of it
            await show(chrome, [(0, 0, [50, 0, 0])])
            drawn = await turn(chrome, 0, 180)
            assert abs(drawn["view"]["yaw"]) <= UNSEEN_DEGREES
            result = await check(chrome, drawn["view"])
            assert result["counts"]["nowhere"] == 0 and result["counts"]["panorama"] == 0

    asyncio.run(run())


def test_a_turn_up_or_down_goes_as_far_as_the_frames_reach(tmp_path):
    async def run():
        async with turning(tmp_path) as chrome:
            # looked down to 30 degrees and back (a frame every 5 degrees, a turn of 80 degrees a
            # second): a look down shows at once, a look up only as far as the frames reach
            pitches = list(range(0, 31, 5)) + list(range(25, -1, -5))
            await show(chrome, [(0, pitch) for pitch in pitches])
            drawn = await turn(chrome, 25, 0)
            assert drawn["frames"] < 1.5 and drawn["view"] == {"yaw": 0, "pitch": 25}
            result = await check(chrome, drawn["view"])
            assert result["counts"]["nowhere"] == 0 and result["error"] < WORLD_ERROR
            drawn = await turn(chrome, -30, 0)
            assert -UNSEEN_DEGREES <= drawn["view"]["pitch"] <= 0
            result = await check(chrome, drawn["view"])
            assert result["counts"]["nowhere"] == 0
            # straight down, round the pole: turning there shows the frames that looked down there
            await show(chrome, [(yaw, 89) for yaw in range(0, 181, 15)] + [(0, 89)])
            drawn = await turn(chrome, 89, 90)
            assert drawn["view"] == {"yaw": 90, "pitch": 89}
            result = await check(chrome, drawn["view"])
            assert result["counts"]["nowhere"] == 0 and result["error"] < WORLD_ERROR

    asyncio.run(run())


def test_the_frames_that_catch_up_are_drawn_as_generated(tmp_path):
    async def run():
        async with turning(tmp_path) as chrome:
            await show(chrome, looked_around(0, 120))
            drawn = await turn(chrome, 0, 100)
            assert drawn["view"]["yaw"] == 100
            # the frames that hold the turn come: the view is each of them as it is, the
            # magnification eased out
            await show(chrome, [(yaw, 0) for yaw in (20, 50, 80, 100)])
            await chrome.wait_for("stream.turn.magnify === 1")
            last = await chrome.eval("harness.lastDraw()")
            assert last["view"] == last["base"] == {"yaw": 100, "pitch": 0}
            result = await check(chrome, last["view"])
            assert result["counts"]["panorama"] == 0 and result["counts"]["nowhere"] == 0
            assert result["error"] < STILL_ERROR and result["still"] < STILL_ERROR

    asyncio.run(run())


def test_the_panorama_keeps_nothing_of_the_viewmodel(tmp_path):
    async def run():
        async with turning(tmp_path) as chrome:
            around = [(yaw, pitch) for pitch in (0, 30) for yaw in range(0, 360, 20)]
            await show(chrome, around + [(0, 0)])
            panorama = 0
            for yaw in range(0, 360, 30):
                for pitch in (-20, 0, 30, 60):
                    seen = await chrome.eval(f"harness.panoramaStray({yaw}, {pitch})")
                    assert seen["stray"] == 0, (yaw, pitch, seen)
                    panorama += seen["panorama"]
            assert panorama > 30 * 0.5 * PAGE["frame_pixels"][0] * PAGE["frame_pixels"][1]

    asyncio.run(run())


def test_the_panorama_keeps_no_knife_swung_at_a_run(tmp_path):
    """A knife carried at a run swings outside the viewmodel's pieces: while a frame's row holds a
    knife and a movement button, the panorama takes only the band of it above where the knife
    reaches, so no turn shows it where it was. A frame whose row says nothing of it would keep
    the knife: the gating is what keeps the panorama clean."""

    async def strays(controls: dict) -> int:
        directory = tmp_path / ("run" if controls else "none")
        directory.mkdir()
        async with turning(directory) as chrome:
            await show(chrome, looked_around(0, 120), {**controls, "knife": True})
            await show(chrome, [(0, 0)])  # at rest: the knife in its piece
            stray = 0
            for yaw in range(20, 121, 20):
                seen = await chrome.eval(f"harness.panoramaStray({yaw}, 0)")
                stray += seen["stray"]
            # a turn back is drawn from the band the panorama took: as far as it reaches
            drawn = await turn(chrome, 0, 100)
            result = await check(chrome, drawn["view"])
            assert result["counts"]["nowhere"] == 0
            return stray

    async def run():
        assert await strays(KNIFE_RUN) == 0
        assert await strays({}) > 1000

    asyncio.run(run())


def test_frames_shown_long_ago_are_not_used(tmp_path):
    async def run():
        async with turning(tmp_path) as chrome:
            await show(chrome, looked_around(0, 120))
            await chrome.eval("harness.skip(9)")
            drawn = await turn(chrome, 0, 100)
            assert abs(drawn["view"]["yaw"]) <= UNSEEN_DEGREES
            result = await check(chrome, drawn["view"])
            assert result["counts"]["nowhere"] == 0 and result["counts"]["panorama"] == 0

    asyncio.run(run())


def test_with_the_setting_off_every_frame_is_drawn_as_generated(tmp_path):
    async def run():
        async with turning(tmp_path, instant=False) as chrome:
            await show(chrome, looked_around(0, 120))
            await chrome.eval("harness.turnTo(0, 100)")
            await asyncio.sleep(0.2)
            drawn = await chrome.eval("harness.lastDraw()")
            assert drawn["look"] == drawn["view"] == {"yaw": 0, "pitch": 0}
            assert drawn["magnify"] == 1
            result = await check(chrome, drawn["view"])
            assert result["counts"]["panorama"] == 0 and result["error"] < STILL_ERROR
            await chrome.eval("harness.setInstant(true)")
            await chrome.wait_for("harness.lastDraw().view.yaw === 100")

    asyncio.run(run())


def test_the_motion_feedback_moves_the_turned_view_as_the_flat_canvas_moves_a_frame(tmp_path):
    """The motion feedback's move (feedback.js) scales the turned view about its centre and shifts
    it, as the 2D canvas does a frame (``test_motion_feedback.py``)."""

    async def run():
        async with turning(tmp_path) as chrome:
            await show(chrome, looked_around(0, 60))
            await turn(chrome, 0, 30)
            for zoom, shift in ((1.04, 0), (0.97, 0), (1, 0.025), (1, -0.025)):
                moved = await chrome.eval(f"harness.moved({zoom}, {shift})")
                assert moved["error"] < 6, (zoom, shift, moved)
                assert moved["pixels"] > 0.8 * 672 * 384, (zoom, shift, moved)

    asyncio.run(run())


@pytest.fixture(scope="module")
def recorded() -> dict:
    return asyncio.run(recorded_play(3))


def test_the_mouse_turns_the_play_view_at_the_next_animation_frame(recorded, tmp_path):
    """The play view as served (``static/play.js``, on the stand-in stream of
    ``motion_play.html``, with WebGL): frames taken looking left to 60 degrees and back come, and a
    jump of the mouse to the left (30 degrees: the mouse turns as far as it moves) turns the view at
    the next animation frame by all of it, before any frame holds it; a mouse turn right, where no
    frame looked, does not turn it."""
    setup = {
        "page": recorded["page"],
        "join": recorded["join"],
        "welcome": recorded["welcome"],
        "header": recorded["frames"][0][0],
        "storage": {"wc.instantturn": "true"},  # off by default
        "canvas2d": True,
        "webgl": True,
    }
    page = MOTION_HARNESS.read_text(encoding="utf-8").replace("__SETUP__", json.dumps(setup))

    async def run():
        async with chrome_page(page, (1000, 700, 1), tmp_path) as chrome:
            await chrome.wait_for("window.ready === true")
            for yaw in [0, 10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 0]:
                extra = {"tn": [0, yaw], "cp": 0}
                await chrome.eval(f"harness.deliver(-1, 1, {json.dumps(extra)})")
                await asyncio.sleep(0.12)
            await chrome.wait_for("!!document.querySelector('#go')")
            x, y = await chrome.eval("harness.centre('#go')")
            await chrome.click(x, y)
            await chrome.wait_for("document.querySelector('#overlay').hidden")
            await asyncio.sleep(0.3)
            last = await chrome.eval("harness.draws(0).at(-1)")
            assert last["view"] == {"yaw": 0, "pitch": 0}
            # 0.03 degrees a pixel: 30
            since = await chrome.eval(
                "(() => { const t = harness.now(); harness.burst(-1000, 0); return t; })()"
            )
            await chrome.wait_for(f"harness.draws({since}).length > 0")
            first = (await chrome.eval(f"harness.draws({since})"))[0]
            assert first["view"]["yaw"] == pytest.approx(30) and first["view"]["pitch"] == 0
            assert first["k"] == last["k"]  # no frame since: the newest is still the one at 0
            await asyncio.sleep(0.6)
            # right: 30 back to 0, then 30 more where none looked
            since = await chrome.eval(
                "(() => { const t = harness.now(); harness.burst(1000, 0); harness.burst(1000, 0);"
                " return t; })()"
            )
            await asyncio.sleep(0.3)
            draws = await chrome.eval(f"harness.draws({since})")
            assert min(d["view"]["yaw"] for d in draws) >= -UNSEEN_DEGREES
            assert draws[0]["view"]["yaw"] == pytest.approx(0, abs=UNSEEN_DEGREES)
            assert await chrome.eval("harness.errors") == []

    asyncio.run(run())


#: A session in real time, [t (s), pitch, yaw] with the turn linear between: at rest, a turn left of
#: 120 degrees over a second (where no frame has looked), back over a second (where they have), a
#: flick right of 90 degrees (unseen), and one of 180 degrees left (seen).
SESSION = [
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 120],
    [2.5, 0, 120],
    [3.5, 0, 0],
    [4, 0, 0],
    [4.2, 0, -90],
    [5.2, 0, -90],
    [5.4, 0, 90],
    [6.4, 0, 90],
]


def test_a_session_of_turns_in_real_time(tmp_path):
    """The turn follows a path at every animation frame and the frames come 16 a second, each where
    the turn was 0.3 s before (``harness.simulate``): a turn back over what the frames have seen is
    drawn where the hand is at the next animation frame, a turn into what they have not seen waits
    for them, and half a turn back to where they looked shows at once. The canvas is recorded at
    every animation frame; the recording (an mp4) and stills of it go to ``$WORLDCAST_TURN_VIDEO``
    if it is set."""
    out = Path(os.environ.get(VIDEO_ENV) or tmp_path)
    out.mkdir(parents=True, exist_ok=True)

    async def run():
        async with turning(tmp_path) as chrome:
            timeline = await chrome.eval(f"harness.simulate({json.dumps(SESSION)}, 6.4)")
            result = await check(chrome, (await chrome.eval("harness.lastDraw()"))["view"])
            assert result["counts"]["nowhere"] == 0
            return timeline, await chrome.eval("harness.recorded()")

    timeline, frames = asyncio.run(run())
    # each animation frame draws the turn of the one before
    pairs = [(a["turn"][1], b) for a, b in zip(timeline, timeline[1:]) if b.get("view")]
    times = [b["t"] for a, b in zip(timeline, timeline[1:]) if b.get("view")]

    def during(t0, t1):
        return [(hand, b) for (hand, b), t in zip(pairs, times) if t0 <= t <= t1]

    back = during(2.6, 3.4)
    assert sum(abs(b["view"]["yaw"] - hand) < 1 for hand, b in back) > 0.8 * len(back)
    # into the unseen, left and right: the view is the newest frame's but for the sliver, and turns
    # as the frames come
    for _, b in during(1.0, 2.2) + during(4.0, 4.8):
        assert abs(b["view"]["yaw"] - b["base"]["yaw"]) <= UNSEEN_DEGREES
    flick = during(5.45, 5.6)
    assert all(abs(b["view"]["yaw"] - hand) < 1 for hand, b in flick)
    assert all(b["base"]["yaw"] < 0 for _, b in flick[:2])  # the frames have not turned yet
    stills = {"seen-turn-back": 3.0, "unseen-turn-right": 4.35, "seen-half-turn": 5.5}
    for name, at in stills.items():
        index = min(range(len(timeline)), key=lambda i: abs(timeline[i]["t"] - at))
        url = frames[min(index, len(frames) - 1)]
        (out / f"turn-{name}.jpg").write_bytes(base64.b64decode(url.split(",", 1)[1]))
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    video = out / "instant-turn.mp4"
    writer = imageio_ffmpeg.write_frames(str(video), tuple(PAGE["frame_pixels"]), fps=60)
    writer.send(None)
    for url in frames:
        image = Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
        writer.send(np.asarray(image).tobytes())
    writer.close()
    assert video.stat().st_size > 10_000
