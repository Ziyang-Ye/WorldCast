"""The page's playout (``stream.js``) at the start of a session and when its display falls behind,
in headless Chrome.

The page's own ``Playout`` (the script as served, ``playout_replay.html``) runs on a simulated
clock, so what it shows is the same on every machine. The sessions are those of a GPU worker at
15 fps with blocks of 16 frames, whose frames arrive within a few ms of each other: one on a warm
GPU, one whose first block is slow (a cold GPU's: the block after it comes at once, the one after
that when the display needs it). Each runs on a display that keeps up (60 Hz) and on one that
falls behind in the first seconds. The drop at a block's first frame is also called alone, on a
table of (display interval, frames waiting). The tests skip without Chrome or Chromium, and
wherever Chrome fails to dump the page (it hangs, or exits without): a browser fault is a skip, a
page that fails is a failure. The repository's tests do not need a browser: the page's rule is
pinned in text by ``test_static.py`` and the schedule's by ``test_pacing.py``.
"""

import html
import json
import re
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from tests.demo.support import CHROME, chrome_flags, needs_chrome
from worldcast.demo import latency
from worldcast.demo.config import load_config
from worldcast.demo.coordinator import STATIC
from worldcast.demo.latency import shown_by_page
from worldcast.demo.pacing import BACKLOG_FRAMES, Pacing

REPLAY = Path(__file__).with_name("playout_replay.html")
pytestmark = needs_chrome
#: How long Chrome has to dump the page, s: far above what any machine needs.
DUMP_TIMEOUT_S = 120

PLAY = load_config((), {}).play
#: The display rate of the recorded GPU sessions these traces follow, whatever the demo's default.
FPS = 15.0
PER_BLOCK = 16
FRAME_MS = 1000 / FPS
BLOCK_MS = PER_BLOCK * FRAME_MS
BLOCKS = 40
#: When each block's frames start to arrive, ms: paced; or, after a first block of 3 s, the second
#: block 0.3 s after it, the third 1.2 s after that, the rest paced.
WARM = [n * BLOCK_MS for n in range(BLOCKS)]
COLD = [0, 310, 1537, *(1537 + n * BLOCK_MS for n in range(1, BLOCKS - 2))]
#: A display that shows a frame every 2.25 frame periods (150 ms at 15 fps), as the pages of the
#: recorded GPU sessions did for their first 25 frames, for the first 4 s.
SLOW_FOR_MS, SLOW_EVERY_MS = 4000, 2.25 * FRAME_MS
#: A page hidden from the 8th s to the 38th.
AWAY_MS = [8000, 38000]
#: Frames sent again (``play.swap_sent``): every fourth frame of the warm session, version 1, this
#: many ms after it first arrived; a third of them too late for the page.
AGAIN_MS = (40.0, 80.0, 400.0)


def arrivals(starts: Sequence[float]) -> list[list[float]]:
    """The frames of a session, ``[ms from the first, k, whether it starts its block]``: block
    ``n``'s frames arrive 2 ms apart from ``starts[n]``."""
    return [
        [start + 2 * j, n * PER_BLOCK + j, j == 0]
        for n, start in enumerate(starts)
        for j in range(PER_BLOCK)
    ]


def trace(frames: list[list[float]], **display) -> dict:
    """The page's playout on ``frames``: the worker's welcome gives its settings; ``display`` adds
    ``slow`` or ``away`` (``playout_replay.html``)."""
    return {
        "fps": FPS,
        "per_block": PER_BLOCK,
        "buffer": PLAY.jitter_buffer_frames,
        "backlog": BACKLOG_FRAMES,
        "margin": PLAY.margin_frames,
        "hz": 60,
        "slow": None,
        "away": None,
        "arrivals": frames,
        **display,
    }


#: A block's first frame arrives with frames of earlier blocks waiting, the display showing frames
#: ``every`` frame periods apart: ``(every, waiting, dropped)``. The frames before it are kept
#: while the display keeps up with the frame period (its interval within the playout's 1.25), at
#: most 3 of them, and as many as it shows in 3 frame periods when it is slower.
RULES = [
    (1.0, 3, 0),
    (1.0, 5, 2),
    (1.0, 9, 6),
    (0.8, 5, 2),
    (1.05, 3, 0),  # a measured interval a few percent above the frame period is no slower display
    (1.2, 3, 0),
    (1.3, 3, 1),
    (1.5, 3, 1),
    (2.0, 3, 2),
    (2.25, 3, 2),
    (4.0, 3, 3),
]


def again(frames: list[list[float]]) -> list[list[float]]:
    """``frames`` with every fourth frame sent again (version 1), :data:`AGAIN_MS` after it."""
    resent = [
        [t + AGAIN_MS[(k // 4) % len(AGAIN_MS)], k, False, 1, 1] for t, k, _ in frames if k % 4 == 2
    ]
    return [*frames, *resent]


def mixed(frames: list[list[float]]) -> list[list[float]]:
    """``frames`` but every eighth (``k % 8 == 5``), whose version 1 is sent all the same 40 ms
    after it, as a socket that dropped a frame sends a refresh's frame again; and every fourth
    frame's version 1 sent again 40 ms after it, its version 0 again 20 ms after that (decodes that
    finish out of order)."""
    kept = [frame for frame in frames if frame[1] % 8 != 5]
    resent = []
    for t, k, _ in frames:
        if k % 8 == 5:
            resent.append([t + 40.0, k, False, 1, 1])
        if k % 4 == 2:
            resent += [[t + 40.0, k, False, 1, 1], [t + 60.0, k, False, 0, 1]]
    return [*kept, *resent]


def replay_data() -> dict:
    warm, cold = arrivals(WARM), arrivals(COLD)
    return {
        "traces": {
            "warm": trace(warm),
            "warm, frames sent again": trace(again(warm)),
            "warm, frames sent again out of order": trace(mixed(warm)),
            "cold": trace(cold),
            "warm, slow display": trace(warm, slow=[SLOW_FOR_MS, SLOW_EVERY_MS]),
            "cold, slow display": trace(cold, slow=[SLOW_FOR_MS, SLOW_EVERY_MS]),
            "warm, page hidden": trace(warm, away=AWAY_MS),
        },
        "rules": [
            {
                "fps": FPS,
                "per_block": PER_BLOCK,
                "backlog": BACKLOG_FRAMES,
                "margin": PLAY.margin_frames,
                "every": every * FRAME_MS,
                "ahead": ahead,
            }
            for every, ahead, _ in RULES
        ],
    }


def run_page(data: dict, directory: Path) -> dict:
    """The replay page's result in headless Chrome."""
    for name in ("stream.js", "common.js", "turn.js", "input.js"):
        (directory / name).symlink_to(STATIC / name)
    page, dom = directory / "replay.html", directory / "dom.html"
    page.write_text(REPLAY.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data)))
    flags = [*chrome_flags(directory), "--disable-gpu", "--dump-dom"]
    exited = None
    with dom.open("wb") as out:
        chrome = subprocess.Popen(
            [CHROME, *flags, page.as_uri()], stdout=out, stderr=subprocess.DEVNULL
        )
        try:  # Chrome may stay up after it dumped the page
            deadline = time.monotonic() + DUMP_TIMEOUT_S
            while "</html>" not in dom.read_text(errors="replace"):
                exited = chrome.poll()
                if exited is not None or time.monotonic() > deadline:
                    break
                time.sleep(0.2)
        finally:
            chrome.kill()
            chrome.wait()
    result = re.search(r'<pre id="out">(.*?)</pre>', dom.read_text(errors="replace"), re.S)
    if not result:  # a browser fault, not the page's
        pytest.skip(
            f"Chrome left no page dump (exit code {exited}, waited {DUMP_TIMEOUT_S} s at most)"
        )
    assert result.group(1) != "running", "the replay page did not run: stream.js failed to load"
    out = json.loads(html.unescape(result.group(1)))
    assert "error" not in out, out["error"]
    return out


class Replays:
    """What the page showed: ``waits[name]``, how long the first frame of each block waited from
    its arrival to its display, ms (block 0 first; nan where a later block dropped it); ``dropped``,
    ``holds`` and ``every`` (the display's interval the playout measured, after each frame it
    showed), by trace; ``rules``, the frames dropped in each of :data:`RULES`."""

    def __init__(self, data: dict, result: dict) -> None:
        self.waits, self.dropped, self.holds, self.every = {}, {}, {}, {}
        self.shown, self.versions, self.inorder = {}, {}, result["inorder"]
        for name, replay in result["traces"].items():
            self.shown[name] = {int(k): t for k, t in replay["shown"].items()}
            self.versions[name] = {int(k): v for k, v in replay["versions"].items()}
            frames = data["traces"][name]["arrivals"]
            shown = replay["shown"]
            self.waits[name] = np.array(
                [shown.get(str(k), np.nan) - t for t, k, first, *_ in frames if first]
            )
            self.dropped[name], self.holds[name] = replay["dropped"], replay["holds"]
            self.every[name] = np.array(replay["every"])
        self.rules = result["rules"]


def schedule_waits(frames: list[list[float]]) -> np.ndarray:
    """How long the schedule's display (:class:`~worldcast.demo.pacing.Pacing`) makes the first
    frame of each block of ``frames`` wait, ms."""
    pacing, waits = Pacing(FPS), []
    for n in range(0, len(frames), PER_BLOCK):
        sent = [t / 1000 for t, _, _ in frames[n : n + PER_BLOCK]]
        waits.append(1000 * (pacing.record(sent[0], sent)[0] - sent[0]))
    return np.array(waits)


def replay(directory: Path) -> Replays:
    data = replay_data()
    return Replays(data, run_page(data, directory))


@pytest.fixture(scope="module")
def replays(tmp_path_factory) -> Replays:
    return replay(tmp_path_factory.mktemp("playout"))


def test_the_drop_keeps_the_backlog_of_a_display_that_keeps_up_and_what_a_slower_one_shows(replays):
    """At a block's first frame the page keeps the frames its display shows in the backlog's frame
    periods: ``BACKLOG_FRAMES`` of them while it keeps up, whatever a measured interval a few
    percent above the frame period says; fewer, down to none, on a slower display."""
    assert BACKLOG_FRAMES == 3
    assert replays.rules == [dropped for _, _, dropped in RULES]


def test_a_display_that_keeps_up_waits_for_the_frames_kept_and_its_next_frame(replays):
    """On a display that keeps up, each block's first frame waits for the playout's margin, and
    the block after a slow first block, which comes early, for the 3 frames kept before it and the
    display's next frame time (3.6 frame periods, under the 3.75 of ``BACKLOG_FRAMES``' bound),
    then the later ones for the margin again; the
    warm session drops nothing, the cold one the 8 frames that wait beyond the 3 kept (the page
    shows its first frame on arrival, ``play.jitter_buffer_frames``)."""
    warm, cold = replays.waits["warm"], replays.waits["cold"]
    assert replays.dropped["warm"] == 0 and replays.dropped["cold"] == 8
    assert (warm[1:] < 115).all()
    assert 195 < cold[1] < 3.75 * FRAME_MS
    assert cold[2] < 115 and (cold[3:7] < 135).all() and (cold[7:] < 115).all()


def test_a_display_that_fell_behind_in_the_first_seconds_is_back_within_the_first_blocks(replays):
    """A display that shows a frame every 2.25 frame periods for the first 4 s: per block, the
    first frame of the first blocks waits for at most the frames its display shows in 3 frame
    periods (1 of them, 3 before its interval is known) and its next frame time, the later ones
    as on a display that keeps up; the frames dropped are those beyond what it shows."""
    for name, dropped in (("warm, slow display", 32), ("cold, slow display", 38)):
        waits = replays.waits[name]
        assert replays.dropped[name] == dropped
        assert waits[1] <= 3 * SLOW_EVERY_MS
        assert (waits[2:5] <= 2 * SLOW_EVERY_MS).all()
        assert (waits[5:8] < 150).all() and (waits[8:] < 115).all()


def test_a_page_hidden_for_a_while_leaves_the_measured_display_interval_alone(replays):
    """A page hidden for 30 s shows its first frame after it with a gap of a block's periods: the
    interval at which the display shows its frames stays within what the playout stretches to, so
    the blocks after the page is back keep their frames as before, and the page dropped the
    blocks that came while it was hidden but the 3 frames in front of each. The first block shown
    is stale: the 3 frames kept before it play first, then it, until the next block's first frame
    drops what is left."""
    every = replays.every["warm, page hidden"]
    assert every.max() < 1.25 * FRAME_MS
    waits = replays.waits["warm, page hidden"]
    assert 800 < waits[35] < 900  # 12.5 frame periods, 0.8 s at 15 fps
    assert waits[36] < 3.5 * FRAME_MS and (waits[37:] < 125).all()
    assert replays.dropped["warm, page hidden"] == 446


def test_the_schedule_and_the_page_agree_on_how_long_a_block_that_came_early_waits(replays):
    """The schedule's display and the page's playout are two implementations of the drop: a
    block that arrives with frames of the block before it waiting is shown 3 frame periods after
    it arrived by the schedule, and by the page within a frame period; where blocks come when
    the display needs them, the schedule shows the first frame on arrival and the page after its
    margin of 1.5 frame periods or two."""
    cold = schedule_waits(arrivals(COLD))
    assert cold[1] == pytest.approx(BACKLOG_FRAMES * FRAME_MS)
    assert abs(replays.waits["cold"][1] - cold[1]) < FRAME_MS
    warm = schedule_waits(arrivals(WARM))
    assert (warm[1:] < 1).all()
    assert (replays.waits["warm"][2:] - warm[2:] < 2 * FRAME_MS).all()


def test_a_frame_sent_again_shows_in_place_of_the_one_waiting_if_it_comes_in_time(replays):
    """A newer version of a frame (``play.swap_sent``) is shown in its place when it arrives before
    the frame is due, and dropped when it comes after: every frame is shown once, at the times of
    the session without the frames sent again."""
    name = "warm, frames sent again"
    shown, versions = replays.shown[name], replays.versions[name]
    assert shown == replays.shown["warm"] and replays.dropped[name] == len(shown) // 4
    for k, t in shown.items():
        arrived = WARM[k // PER_BLOCK] + 2 * (k % PER_BLOCK) + AGAIN_MS[(k // 4) % len(AGAIN_MS)]
        assert versions[k] == (k % 4 == 2 and arrived <= t)


def test_a_frame_sent_again_never_takes_the_place_of_a_newer_one_nor_of_one_never_received(
    replays,
):
    """A version older than the one waiting (a decode that finished late) is dropped, and so is a
    frame sent again that the page never received: the page shows each frame it received once, at
    most its newest version, and none of those it did not; the frames go to the playout in the order
    they arrived, whatever order their decodes finish in (``InOrder``, a failed decode skipped)."""
    name = "warm, frames sent again out of order"
    shown, versions = replays.shown[name], replays.versions[name]
    assert not [k for k in shown if k % 8 == 5]
    for k, t in shown.items():
        arrived = WARM[k // PER_BLOCK] + 2 * (k % PER_BLOCK) + 40.0
        assert versions[k] == (k % 4 == 2 and arrived <= t)
    assert replays.inorder == ["first", "second", "fourth"]


def test_the_benchs_model_of_the_page_shows_what_the_page_shows(replays):
    """``tools/bench_live.py``'s model of the page (:func:`~worldcast.demo.latency.shown_by_page`,
    the rules of ``Playout`` in Python) shows each frame of these sessions when the page does, and
    the same version of a frame sent again."""
    play = load_config((), {"play.jitter_buffer_frames": 1, "play.fps": FPS}).play
    traces = (
        ("cold", arrivals(COLD)),
        ("warm, frames sent again", again(arrivals(WARM))),
        ("warm, frames sent again out of order", mixed(arrivals(WARM))),
    )
    for name, frames in traces:
        frames = sorted(frames, key=lambda f: (f[0], f[1]))
        sent = [
            (frame[1], frame[0] / 1000 - latency.NETWORK_S, frame[2], *_version(frame))
            for frame in frames
        ]
        shown, at, _ = shown_by_page(sent, play, PER_BLOCK)
        assert {sent[i][0]: 1000 * t for i, t in zip(shown, at)} == pytest.approx(
            replays.shown[name], abs=1e-6
        )
        versions = {sent[i][0]: (frames[i][3] if len(frames[i]) > 3 else 0) for i in shown}
        assert versions == replays.versions[name]


def _version(frame: list) -> tuple[int, bool]:
    """The rank of a replayed frame's version and whether it was sent again."""
    version, resent = (frame[3], frame[4]) if len(frame) > 3 else (0, 0)
    return 2 * version + 1, bool(resent)
