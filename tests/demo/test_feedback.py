"""The motion feedback controller (``static/feedback.js``), run alone in headless Chrome over
scripted presses, releases and arrivals of frames (``feedback.html``).

A movement button's press starts one ramp of the view's change (a dolly for forward and back, a
shift for a strafe) that rises along an ease-out to a cap over 0.4 s and holds there until the
first frame that folds the press in is drawn, or the button is let go; from either it falls to
nothing over 0.2 s, along a smooth step, and a stop is not answered by a change the other way. The
change is a function of the buttons' state alone, within the caps: a held button gives one ramp,
a new press starts its ramp only when the last has fallen to nothing, and no sequence of them
builds anything up. The tests skip without Chrome or Chromium.
"""

import asyncio
import json
from pathlib import Path

import pytest

from tests.demo.support import STATIC, needs_chrome, run_harness

HARNESS = Path(__file__).with_name("feedback.html")
pytestmark = needs_chrome

#: The caps at the Light setting (``static/feedback.js``): forward scales the view up by 4 percent,
#: back down by 3, a strafe shifts it by 2.5 percent of its width against the move.
IN, OUT, SHIFT = 0.04, 0.03, 0.025
#: The ramp to the cap and the fall from a frame (or a release), s.
RAMP, HANDOVER = 0.4, 0.2


def ease(seconds: float) -> float:
    """The share of the cap ``seconds`` after a press."""
    x = min(max(seconds / RAMP, 0.0), 1.0)
    return 1 - (1 - x) ** 2


def fall(seconds: float) -> float:
    """The share of what a ramp held that is left ``seconds`` after it began to fall."""
    x = min(max(seconds / HANDOVER, 0.0), 1.0)
    return 1 - x * x * (3 - 2 * x)


def at(*times: float) -> list[list]:
    """Steps that record the view at ``times`` (ms)."""
    return [[t, "at"] for t in times]


def script(*steps: list) -> list[list]:
    """Steps in the order of their time, a press, release or arrival before a recording at the same
    time."""
    return sorted(steps, key=lambda step: (step[0], step[1] == "at"))


def runs() -> dict[str, dict]:
    """The scripted runs: ``setting`` and ``steps`` ([ms, kind, ...] in order)."""
    cycles: list[list] = []
    for i in range(30):
        start = 1000 * i
        cycles += [[start, "press", "forward", 2 * i + 1], [start + 300, "arrive", 2 * i + 1]]
        cycles += [[start + 500, "release", "forward"]]
        cycles += at(start + 10, start + 200, start + 400, start + 500, start + 600, start + 900)
    both = [[0, "press", "forward", 1], [0, "press", "back", 1]]
    pair = [[0, "press", "move_left", 1], [0, "press", "move_right", 1]]
    return {
        "forward": {
            "setting": "light",
            "steps": script([0, "press", "forward", 1], *at(0, 50, 100, 200, 300, 400, 500, 5000)),
        },
        "back": {
            "setting": "light",
            "steps": script([0, "press", "back", 1], *at(0, 100, 200, 400, 9000)),
        },
        "left": {
            "setting": "light",
            "steps": script([0, "press", "move_left", 1], *at(0, 100, 200, 400, 9000)),
        },
        "right": {
            "setting": "light",
            "steps": script([0, "press", "move_right", 1], *at(0, 100, 200, 400, 9000)),
        },
        "forward and left": {
            "setting": "light",
            "steps": script(
                [0, "press", "forward", 1], [0, "press", "move_left", 1], *at(100, 200, 400, 1000)
            ),
        },
        "forward and back": {"setting": "light", "steps": script(*both, *at(200, 400, 1000))},
        "left and right": {"setting": "light", "steps": script(*pair, *at(200, 400, 1000))},
        "second key": {
            "setting": "light",
            "steps": script(
                [0, "press", "forward", 1],
                [300, "press", "move_right", 2],
                *at(200, 300, 500, 700, 1000),
            ),
        },
        "arrival": {
            "setting": "light",
            "steps": script(
                [0, "press", "forward", 3],
                [500, "arrive", 2],
                [600, "arrive", 3],
                *at(500, 550, 600, 650, 700, 750, 800, 5000),
            ),
        },
        "release": {
            "setting": "light",
            "steps": script(
                [0, "press", "forward", 1],
                [200, "release", "forward"],
                *at(100, 200, 250, 300, 350, 400, 450, 5000),
            ),
        },
        "release back and strafe": {
            "setting": "light",
            "steps": script(
                [0, "press", "back", 1],
                [0, "press", "move_left", 1],
                [100, "release", "back"],
                [100, "release", "move_left"],
                *at(50, 100, 200, 300, 400, 5000),
            ),
        },
        "arrival after release": {
            "setting": "light",
            "steps": script(
                [0, "press", "forward", 1],
                [100, "release", "forward"],
                [150, "arrive", 1],
                *at(100, 150, 200, 300, 400),
            ),
        },
        "press during the fall": {
            "setting": "light",
            "steps": script(
                [0, "press", "forward", 1],
                [500, "arrive", 1],
                [550, "release", "forward"],
                [600, "press", "forward", 2],
                *at(500, 600, 650, 699, 700, 750, 800, 1100, 1500, 4000),
            ),
        },
        "press folded during the fall": {
            "setting": "light",
            "steps": script(
                [0, "press", "forward", 1],
                [500, "arrive", 1],
                [520, "release", "forward"],
                [540, "press", "forward", 2],
                [600, "arrive", 2],
                *at(600, 699, 700, 800, 1500, 4000),
            ),
        },
        "held": {
            "setting": "light",
            "steps": script(
                [0, "press", "forward", 1], [500, "arrive", 1], *at(699, 700, 2000, 10000, 100000)
            ),
        },
        "cycles": {"setting": "light", "steps": script(*cycles)},
        "off": {
            "setting": "off",
            "steps": script(
                [0, "press", "forward", 1],
                [0, "press", "move_left", 1],
                [1000, "arrive", 1],
                [1100, "release", "forward"],
                *at(0, 200, 400, 1000, 1100, 1500),
            ),
        },
        "strong": {
            "setting": "strong",
            "steps": script(
                [0, "press", "forward", 1],
                [0, "press", "move_right", 1],
                [1000, "arrive", 1],
                *at(200, 400, 1000, 1100, 1200),
            ),
        },
        "strong back": {
            "setting": "strong",
            "steps": script([0, "press", "back", 1], *at(400, 1000)),
        },
        "unknown": {
            "setting": "light",
            "steps": script(
                [0, "press", "jump", 1],
                [10, "release", "forward"],
                [20, "press", "attack", 2],
                *at(0, 500, 1000),
            ),
        },
        "set": {
            "setting": "light",
            "steps": script(
                [0, "press", "forward", 1],
                [500, "set", "strong"],
                [1000, "press", "forward", 2],
                [1500, "set", "off"],
                [1600, "press", "forward", 3],
                *at(400, 500, 1000, 1400, 1500, 2000),
            ),
        },
    }


@pytest.fixture(scope="module")
def seen(tmp_path_factory) -> dict[str, list[dict]]:
    """The view's ``time``, ``zoom`` and ``shift`` at each ``at`` step of every run."""
    page = HARNESS.read_text(encoding="utf-8").replace("__SETUP__", json.dumps({"runs": runs()}))
    report = asyncio.run(run_harness(page, (400, 300, 1), tmp_path_factory.mktemp("feedback")))
    return report["runs"]


def zoom(rows: list[dict]) -> list[float]:
    return [row["zoom"] for row in rows]


def shift(rows: list[dict]) -> list[float]:
    return [row["shift"] for row in rows]


def times(rows: list[dict]) -> list[float]:
    return [row["time"] for row in rows]


def test_the_caps_and_the_ramp_times_are_named_constants_in_one_place():
    text = (STATIC / "feedback.js").read_text(encoding="utf-8")
    for line in (
        "export const FEEDBACK_SHARE = { off: 0, light: 1, strong: 2 };",
        "export const FEEDBACK_DEFAULT = 'off';",
        "export const DOLLY_IN = 0.04;",
        "export const DOLLY_OUT = 0.03;",
        "export const STRAFE_SHIFT = 0.025;",
        "export const RAMP_S = 0.4;",
        "export const HANDOVER_S = 0.2;",
    ):
        assert line in text, line


def test_a_forward_press_ramps_along_an_ease_out_to_the_cap_and_holds_it(seen):
    """Starting at once, steepest at the start and flat at the cap, which it reaches at 0.4 s and
    keeps for as long as nothing folds the press in; the view scales, it does not shift."""
    rows = seen["forward"]
    assert times(rows) == [0, 50, 100, 200, 300, 400, 500, 5000]
    assert zoom(rows) == pytest.approx([1 + IN * ease(t / 1000) for t in times(rows)])
    assert zoom(rows)[0] == 1.0 and zoom(rows)[1] - 1 == pytest.approx(0.234 * IN, abs=1e-4)
    assert zoom(rows)[-3:] == [1 + IN] * 3 and max(zoom(rows)) == 1 + IN
    assert shift(rows) == [0.0] * len(rows)
    slopes = [
        (b - a) / (u - t)
        for a, b, t, u in zip(zoom(rows), zoom(rows)[1:], times(rows), times(rows)[1:])
    ]
    assert all(a > b for a, b in zip(slopes[:5], slopes[1:6]))  # slowing down into the cap


def test_back_scales_the_view_down_and_a_strafe_shifts_it_against_the_move(seen):
    """Back to 0.97 of the view; left (the player steps left, the world moves right) shifts the
    view 2.5 percent of its width to the right, right to the left; neither scales."""
    back, left, right = seen["back"], seen["left"], seen["right"]
    assert zoom(back) == pytest.approx([1 - OUT * ease(t / 1000) for t in times(back)])
    assert zoom(back)[-1] == pytest.approx(0.97) and shift(back) == [0.0] * len(back)
    assert shift(left) == pytest.approx([SHIFT * ease(t / 1000) for t in times(left)])
    assert shift(right) == pytest.approx([-SHIFT * ease(t / 1000) for t in times(right)])
    assert shift(left)[-1] == pytest.approx(0.025) and shift(right)[-1] == pytest.approx(-0.025)
    assert zoom(left) == [1.0] * len(left) and zoom(right) == [1.0] * len(right)


def test_buttons_held_together_add_and_opposite_ones_cancel_at_the_caps(seen):
    """Forward and left: the dolly and the shift together, each along its own ramp, a second key
    along a ramp of its own from its own press; forward and back: the two caps' sum; left and
    right: nothing."""
    both = seen["forward and left"]
    assert zoom(both) == pytest.approx([1 + IN * ease(t / 1000) for t in times(both)])
    assert shift(both) == pytest.approx([SHIFT * ease(t / 1000) for t in times(both)])
    assert (zoom(both)[-1], shift(both)[-1]) == pytest.approx((1.04, 0.025))
    assert zoom(seen["forward and back"])[-1] == pytest.approx(1 + IN - OUT)
    assert shift(seen["left and right"]) == pytest.approx([0.0] * 3)
    second = seen["second key"]
    assert times(second) == [200, 300, 500, 700, 1000]
    assert zoom(second) == pytest.approx([1 + IN * ease(t / 1000) for t in times(second)])
    assert shift(second) == pytest.approx([-SHIFT * ease((t - 300) / 1000) for t in times(second)])
    assert shift(second)[:2] == [0.0, 0.0] and shift(second)[-1] == pytest.approx(-SHIFT)


def test_the_frame_that_folds_the_press_in_hands_over_to_nothing_in_a_fifth_of_a_second(seen):
    """A frame that holds only earlier samples (q = 2 of a press carried by 3) changes nothing;
    the frame with q = 3 starts the fall from where the ramp is (at the cap here, so no jump),
    along a smooth step to nothing at 0.2 s, which a key still held does not start again."""
    rows = seen["arrival"]
    assert times(rows) == [500, 550, 600, 650, 700, 750, 800, 5000]
    assert zoom(rows)[:2] == [1 + IN] * 2  # q = 2 changed nothing
    assert zoom(rows[2:]) == pytest.approx(
        [1 + IN * fall((t - 600) / 1000) for t in times(rows[2:])]
    )
    assert zoom(rows)[2] == pytest.approx(1 + IN)  # no jump at the handover
    assert all(a > b for a, b in zip(zoom(rows)[2:6], zoom(rows)[3:7]))  # a fall, not a snap
    assert 1.0 < zoom(rows)[3] < 1 + IN
    assert zoom(rows)[-2:] == [1.0, 1.0] and shift(rows) == [0.0] * len(rows)


def test_a_held_key_whose_press_was_folded_in_stays_at_nothing(seen):
    """The ramp is one per press: after the fall the view is the frame's for as long as the key is
    held, for ever."""
    rows = seen["held"]
    assert times(rows) == [699, 700, 2000, 10000, 100000]
    assert zoom(rows)[0] > 1.0  # 1 ms short of the fall's end
    assert zoom(rows)[1:] == [1.0] * 4 and shift(rows) == [0.0] * 5


def test_a_release_before_the_frame_falls_to_nothing_the_same_way_and_is_not_answered(seen):
    """Released at 200 ms, 0.75 of the way up the ramp: the fall starts from where it was, reaches
    nothing 0.2 s later and the view never goes below the frame's, nor stays: a stop shows as
    the frames show it."""
    rows = seen["release"]
    assert times(rows) == [100, 200, 250, 300, 350, 400, 450, 5000]
    expected = [1 + IN * ease(0.1)] + [
        1 + IN * ease(0.2) * fall((t - 200) / 1000) for t in times(rows)[1:]
    ]
    assert zoom(rows) == pytest.approx(expected)
    assert zoom(rows)[1] == pytest.approx(1 + 0.75 * IN) and zoom(rows)[-3:] == [1.0, 1.0, 1.0]
    assert min(zoom(rows)) == 1.0 and all(a >= b for a, b in zip(zoom(rows)[1:], zoom(rows)[2:]))
    both = seen["release back and strafe"]
    assert zoom(both)[-1] == 1.0 and shift(both)[-1] == 0.0
    assert 0.97 <= min(zoom(both)) and max(zoom(both)) <= 1.0  # never a zoom in for a stop
    assert max(abs(s) for s in shift(both)) <= SHIFT and shift(both)[0] > 0
    # a frame that folds in a press already let go changes nothing more
    late = seen["arrival after release"]
    assert times(late) == [100, 150, 200, 300, 400]
    assert zoom(late) == pytest.approx(
        [1 + IN * ease(0.1) * fall((t - 100) / 1000) for t in times(late)]
    )


def test_a_press_during_a_fall_starts_its_ramp_when_the_fall_is_over_and_nothing_stacks(seen):
    """Pressed again at 600 ms, in the fall that began at 500 (and released at 550): the view
    goes on falling to nothing at 700 ms, no higher than it was at 600, and only then does the
    new press ramp, from nothing, to the cap: never above it."""
    rows = seen["press during the fall"]
    assert times(rows) == [500, 600, 650, 699, 700, 750, 800, 1100, 1500, 4000]
    z = zoom(rows)
    assert z[0] == pytest.approx(1 + IN) and z[1] == pytest.approx(1 + IN * fall(0.1))
    assert z[1] >= z[2] >= z[3] > 1.0 and z[4] == 1.0  # still falling, to nothing at 700
    assert z[5:] == pytest.approx([1 + IN * ease((t - 700) / 1000) for t in times(rows)[5:]])
    assert z[8:] == [1 + IN] * 2 and max(z) <= 1 + IN


def test_a_press_the_next_frame_has_already_folded_in_starts_no_ramp(seen):
    """Pressed again in the fall, but the frame that folds that press in (q = 2) is drawn before
    the fall ends: the generated frames show the move, the view makes none of its own."""
    rows = seen["press folded during the fall"]
    assert times(rows) == [600, 699, 700, 800, 1500, 4000]
    assert zoom(rows)[0] == pytest.approx(1 + IN * fall(0.1)) and zoom(rows)[1] > 1.0
    assert zoom(rows)[2:] == [1.0] * 4 and shift(rows) == [0.0] * len(rows)


def test_presses_and_frames_in_any_number_build_nothing_up(seen):
    """Thirty presses, each folded in at 300 ms and let go at 500: every press starts from nothing
    and gives the same view at the same time after it, the view stays in [1, 1.04] throughout and
    it is the frame's, exactly, in the end."""
    rows = seen["cycles"]
    z = zoom(rows)
    assert len(rows) == 30 * 6
    assert 1.0 <= min(z) and max(z) <= 1 + IN
    for position in range(6):
        assert z[position::6] == pytest.approx([z[position]] * 30, abs=1e-12), position
    expected = [1 + IN * ease(0.01), 1 + IN * ease(0.2), 1 + IN * ease(0.3) * fall(0.1)]
    assert z[:3] == pytest.approx(expected) and z[3:6] == [1.0, 1.0, 1.0]
    assert z[-1] == 1.0 and shift(rows) == [0.0] * len(rows)


def test_off_changes_nothing_and_strong_doubles_the_caps(seen):
    """Off: whatever is pressed, folded in or let go, the view is exactly the frame; Strong takes
    the caps twice (a dolly of 8 percent, back 6, a shift of 5 percent of the width) and the same
    ramp and fall."""
    off = seen["off"]
    assert zoom(off) == [1.0] * len(off) and shift(off) == [0.0] * len(off)
    strong = seen["strong"]
    assert times(strong) == [200, 400, 1000, 1100, 1200]
    assert zoom(strong)[:3] == pytest.approx([1 + 2 * IN * ease(t) for t in (0.2, 0.4, 1.0)])
    assert shift(strong)[:3] == pytest.approx([-2 * SHIFT * ease(t) for t in (0.2, 0.4, 1.0)])
    assert (zoom(strong)[2], shift(strong)[2]) == pytest.approx((1.08, -0.05))
    assert zoom(strong)[3] == pytest.approx(1 + 2 * IN * fall(0.1)) and zoom(strong)[4] == 1.0
    assert zoom(seen["strong back"])[-1] == pytest.approx(0.94)


def test_buttons_that_do_not_move_the_player_and_unpressed_ones_do_nothing(seen):
    rows = seen["unknown"]
    assert zoom(rows) == [1.0] * 3 and shift(rows) == [0.0] * 3


def test_a_new_setting_starts_over(seen):
    """Changed in the pause panel: nothing is under way after it, and the new setting's caps
    apply to the presses after it; Off ramps nothing."""
    rows = seen["set"]
    assert times(rows) == [400, 500, 1000, 1400, 1500, 2000]
    assert zoom(rows) == pytest.approx([1 + IN, 1.0, 1.0, 1 + 2 * IN, 1.0, 1.0])
