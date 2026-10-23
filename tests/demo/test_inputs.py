"""How a browser's input samples become a block's controls (the turn spread over the block, done in
its first frames within the generator's trained range and the rest carried to the next block,
chosen between the two by the block's turn with a gain on the yaw the generator is fed, or kept on
each sample's frame), how far they say the player has turned, what a cut holds once the samples
stop, the turn below which a block is idle (``live.idle_context``), the inspect button, which
reaches the generator with a knife only, and a refresh's cut of the block again
(``play.refresh``): what asks for one and when, the rows that stay, the rows that take the new
samples, and the turn conserved.
"""

import numpy as np
import pytest

from worldcast.data.controls import (
    CONTROL_BUTTONS,
    OPENCS2_WEAPONS,
    encode_weapon,
    quantize_camera_delta,
)
from worldcast.demo import inputs
from worldcast.demo.inputs import (
    FRONT_FRAMES,
    GATHER_DEGREES,
    GATHER_S,
    HYBRID_DEGREES,
    HYBRID_FRONT_FRAMES,
    HYBRID_YAW_GAIN,
    INSPECT_WEAPONS,
    InputBuffer,
    InputSample,
)
from worldcast.engine.inference.idle_context import is_idle

FORWARD, JUMP, RIGHT, INSPECT = (
    CONTROL_BUTTONS.index(name) for name in ("forward", "jump", "move_right", "look_at_weapon")
)
#: The rows of a block's controls.
ROWS = ("buttons", "view_deltas", "weapon", "times", "turn", "input_seq", "turned")


def _buffer(mapping: str) -> InputBuffer:
    """Three samples of one 16-frame block ending at room time 1: forward held, a jump tapped in
    the middle, 8 degrees of yaw in all."""
    buffer = InputBuffer(fps=16.0, mapping=mapping, weapon=2, release_after_s=1.0)
    forward, jump = 1 << FORWARD, 1 << JUMP
    for seq, (arrival, buttons, dyaw) in enumerate(
        [(0.10, forward, 2.0), (0.50, forward | jump, 2.0), (0.97, forward, 4.0)]
    ):
        buffer.add(InputSample(seq, arrival, buttons, dpitch=0.0, dyaw=dyaw, weapon=2))
    return buffer


def test_latest_mapping_shows_an_input_from_the_first_frame():
    controls = _buffer("latest").cut(now=1.0, frames=16)
    assert controls.num_frames == 16 and controls.times[-1] == 1.0
    assert controls.buttons[:, FORWARD].all()  # held at the cut: every frame
    assert controls.buttons[:, JUMP].tolist() == [1.0] + [0.0] * 15  # tapped: the first frame
    assert np.allclose(controls.turn[:, 1], 8.0 / 16)  # the turn spread over the block
    assert (controls.input_seq == 2).all() and (controls.weapon == 2).all()
    # the view deltas are the model's encoding of the turn (noclip)
    encoded = quantize_camera_delta(controls.turn[0], clip=False)
    assert np.array_equal(controls.view_deltas[0], encoded)


def test_front_mapping_turns_in_the_first_two_frames():
    """The block's turn over its first two frames, its buttons as ``latest`` holds them."""
    assert FRONT_FRAMES == 2
    controls = _buffer("front").cut(now=1.0, frames=16)
    assert controls.turn[:, 1].tolist() == [4.0] * 2 + [0.0] * 14 and not controls.turn[:, 0].any()
    assert controls.turned[:, 1].tolist() == [4.0] + [8.0] * 15
    latest = _buffer("latest").cut(now=1.0, frames=16)
    assert np.array_equal(controls.buttons, latest.buttons)
    encoded = [quantize_camera_delta(row, clip=False) for row in controls.turn]
    assert np.array_equal(controls.view_deltas, np.stack(encoded))


@pytest.mark.parametrize(
    ("pitch", "yaw", "frames", "per_frame"),
    [
        (-30.0, 100.0, 5, (-6.0, 20.0)),  # beyond 2 x 20 degrees: each frame within the trained 20
        (0.0, -80.0, 4, (0.0, -20.0)),
        (90.0, 10.0, 5, (18.0, 2.0)),  # the larger axis decides
        (0.0, 400.0, 16, (0.0, 20.0)),  # no more frames than the block's: the rest carried
    ],
)
def test_front_mapping_keeps_a_flick_within_the_trained_range(pitch, yaw, frames, per_frame):
    buffer = InputBuffer(fps=16.0, mapping="front", weapon=2, release_after_s=1.0)
    for seq, share in enumerate((0.25, 0.75)):
        buffer.add(InputSample(seq, 0.5, 0, dpitch=share * pitch, dyaw=share * yaw, weapon=2))
    controls = buffer.cut(now=1.0, frames=16)
    expected = np.zeros((16, 2), np.float32)
    expected[:frames] = per_frame
    np.testing.assert_allclose(controls.turn, expected, atol=1e-5)
    np.testing.assert_allclose(controls.turned[-1] + buffer.carry, (pitch, yaw), atol=1e-9)


def test_the_controls_say_how_far_the_player_has_turned():
    """Through each frame, since the session's start: the samples' turn folded into the frames so
    far. The buffer counts the turn it has received, cut or not, which a page that comes back
    counts on from."""
    buffer = _buffer("latest")
    assert buffer.received.tolist() == [0.0, 8.0] and buffer.turned.tolist() == [0.0, 0.0]
    first = buffer.cut(now=1.0, frames=16)
    assert first.turned.dtype == np.float64 and first.turned.shape == (16, 2)
    assert first.turned[:, 1].tolist() == [0.5 * (j + 1) for j in range(16)]
    buffer.add(InputSample(3, 1.5, 0, dpitch=-1.5, dyaw=-2.0, weapon=2))
    assert buffer.received.tolist() == [-1.5, 6.0] and buffer.turned.tolist() == [0.0, 8.0]
    second = buffer.cut(now=2.0, frames=16)
    np.testing.assert_allclose(second.turned[0], (-1.5 / 16, 8.0 - 2.0 / 16))
    assert second.turned[-1].tolist() == buffer.turned.tolist() == [-1.5, 6.0]
    idle = buffer.cut(now=3.0, frames=16)  # nothing new: the view stays where it is
    assert (idle.turned == [-1.5, 6.0]).all()
    # realtime: each sample's turn from its frame on
    realtime = _buffer("realtime").cut(now=1.0, frames=16)
    frame = [int(np.searchsorted(realtime.times, t)) for t in (0.10, 0.50, 0.97)]
    steps = np.diff(realtime.turned[:, 1], prepend=0.0)
    assert steps.nonzero()[0].tolist() == frame and realtime.turned[-1].tolist() == [0.0, 8.0]


def test_realtime_mapping_keeps_each_sample_on_its_frame():
    buffer = _buffer("realtime")
    controls = buffer.cut(now=1.0, frames=16)
    frame = [int(np.searchsorted(controls.times, t)) for t in (0.10, 0.50, 0.97)]
    assert controls.buttons[:, JUMP].nonzero()[0].tolist() == list(range(frame[1], frame[2] + 1))
    assert controls.turn[frame, 1].tolist() == [2.0, 2.0, 4.0] and controls.turn[:, 1].sum() == 8.0
    assert controls.input_seq[frame[0] - 1] == -1 and controls.input_seq[-1] == 2
    # nothing new within a second of the last sample: the next block holds what it left
    held = buffer.cut(now=1.95, frames=16)
    assert held.buttons[:, FORWARD].all() and not held.turn.any()


def test_the_held_buttons_are_released_once_the_samples_stop():
    """``play.release_after_s`` = 1 s after the newest sample, a cut holds no button: the page
    went away without releasing what it held."""
    buffer = InputBuffer(fps=16.0, mapping="latest", weapon=2, release_after_s=1.0)
    buffer.add(InputSample(0, 0.5, 1 << FORWARD, dpitch=0.0, dyaw=0.0, weapon=2))
    held = [buffer.cut(now=now, frames=16).buttons[:, FORWARD].all() for now in (1.0, 1.5)]
    released = [buffer.cut(now=now, frames=16).buttons.any() for now in (1.6, 3.0)]
    assert held == [True, True] and released == [False, False]
    # a sample after the gap holds again
    buffer.add(InputSample(1, 3.2, 1 << FORWARD, dpitch=0.0, dyaw=0.0, weapon=2))
    assert buffer.cut(now=3.5, frames=16).buttons[:, FORWARD].all()


def _cut(mapping: str, samples: list[InputSample], **options):
    """A buffer that took ``samples``, and its cut of a block of 16 frames."""
    buffer = InputBuffer(fps=16.0, mapping=mapping, weapon=2, release_after_s=1.0, **options)
    for sample in samples:
        buffer.add(sample)
    return buffer, buffer.cut(now=1.0, frames=16)


def _block(mapping: str, *, pitch: float = 0.0, yaw: float = 0.0, **options):
    """A buffer and its cut of a block whose four samples add up to ``pitch``, ``yaw``."""
    samples = [InputSample(k, 0.2 * k, 0, pitch / 4, yaw / 4, 2) for k in range(4)]
    return _cut(mapping, samples, **options)


@pytest.fixture
def latent_front(monkeypatch):
    """``front`` over a latent frame's rows, as ``hybrid`` does a block that turns little
    (:data:`HYBRID_FRONT_FRAMES`)."""
    monkeypatch.setattr(inputs, "FRONT_FRAMES", HYBRID_FRONT_FRAMES)


def _buckets(controls) -> np.ndarray:
    """``[16, 2]`` the view controls in the generator's 0.5-degree buckets (pitch, yaw)."""
    return np.round(controls.view_deltas * 40).astype(int)


@pytest.mark.parametrize(
    ("mapping", "samples", "pitch", "yaw"),
    [
        # a flick: 80 degrees of yaw and 10 up, in two samples
        ("latest", [(0.1, 0.0, 30.0), (0.6, -10.0, 50.0)], [-2] * 16, [16] * 16),
        ("front", [(0.1, 0.0, 30.0), (0.6, -10.0, 50.0)], [-9] * 4 + [0] * 12, [40] * 4 + [0] * 12),
        (
            "realtime",
            [(0.1, 0.0, 30.0), (0.6, -10.0, 50.0)],
            [0] * 9 + [-26] + [0] * 6,
            [0, 50] + [0] * 7 + [63] + [0] * 6,
        ),
        # a small correction: 8 degrees of yaw and 1 down
        ("latest", [(0.1, 0.0, 3.0), (0.6, 1.0, 5.0)], [0] * 16, [2] * 16),
        ("front", [(0.1, 0.0, 3.0), (0.6, 1.0, 5.0)], [2] * 2 + [0] * 14, [13] * 2 + [0] * 14),
        (
            "realtime",
            [(0.1, 0.0, 3.0), (0.6, 1.0, 5.0)],
            [0] * 9 + [4] + [0] * 6,
            [0, 10] + [0] * 7 + [16] + [0] * 6,
        ),
    ],
)
def test_the_other_mappings_feed_the_generator_the_turn_as_made(mapping, samples, pitch, yaw):
    """``latest``, ``front`` and ``realtime`` have no gain: the view controls are the code of each
    row of the turn, and the rows are what these mappings always cut (the buckets, 0.5 degree each).
    """
    samples = [InputSample(k, arrival, 0, dp, dy, 2) for k, (arrival, dp, dy) in enumerate(samples)]
    controls = _cut(mapping, samples)[1]
    assert _buckets(controls).T.tolist() == [pitch, yaw]
    codes = [quantize_camera_delta(row, clip=False) for row in controls.turn]
    assert np.array_equal(controls.view_deltas, np.stack(codes))


@pytest.mark.parametrize(
    ("pitch", "yaw", "concentrated"),
    [
        (0.0, 2.0, True),
        (0.0, 19.9, True),
        (0.0, 20.0, False),  # at the threshold: spread
        (0.0, -20.0, False),  # the sign does not count
        (0.0, -19.9, True),
        (0.0, 60.0, False),
        (19.9, 19.9, True),  # the larger axis decides, not their sum
        (-25.0, 3.0, False),
        (0.0, 0.0, True),  # no turn: no rows either way
    ],
)
def test_hybrid_mapping_does_a_block_under_the_threshold_in_its_first_frame_and_spreads_the_rest(
    pitch, yaw, concentrated, latent_front
):
    """The turn rows of ``front`` for a block that turns less than 20 degrees, of ``latest`` for
    one that turns that much or more."""
    assert HYBRID_DEGREES == 20.0
    _, hybrid = _block("hybrid", pitch=pitch, yaw=yaw)
    _, same = _block("front" if concentrated else "latest", pitch=pitch, yaw=yaw)
    for name in ("buttons", "weapon", "times", "turn", "input_seq", "turned"):
        assert np.array_equal(getattr(hybrid, name), getattr(same, name)), name
    assert np.count_nonzero(hybrid.turn[:, 1]) == (4 if concentrated else 16) * bool(yaw)


def test_hybrid_mapping_feeds_the_generator_a_tenth_more_yaw_and_the_camera_the_turn_made():
    """The gain is on the yaw of the view controls: not on the pitch, and not on the turn, the
    turn the page and the camera count."""
    assert HYBRID_YAW_GAIN == 1.1
    # 8 degrees of yaw, 1 down: 2.0 a frame in 4 frames, fed 2.2 (front: 2.0, bucket 7, not 8)
    buffer, small = _block("hybrid", pitch=1.0, yaw=8.0)
    assert small.turn[:, 1].tolist() == [2.0] * 4 + [0.0] * 12
    assert _buckets(small).T.tolist() == [[1] * 4 + [0] * 12, [8] * 4 + [0] * 12]
    # 80 degrees of yaw, 10 up: 5.0 a frame in 16 frames, fed 5.5 (latest: bucket 16, not 17)
    _, sweep = _block("hybrid", pitch=-10.0, yaw=80.0)
    assert sweep.turn[:, 1].tolist() == [5.0] * 16
    assert _buckets(sweep).T.tolist() == [[-2] * 16, [17] * 16]
    # a gain moves a code only across a bucket: 24 degrees, 1.5 a frame, fed 1.65, stays in 6
    assert _buckets(_block("hybrid", yaw=24.0)[1])[:, 1].tolist() == [6] * 16
    # no yaw: nothing to scale; pitch alone is as it is
    assert _buckets(_block("hybrid", pitch=12.0)[1]).T.tolist() == [[10] * 4 + [0] * 12, [0] * 16]
    # what the page and the pose count is the turn made
    assert small.turned[-1].tolist() == [1.0, 8.0]
    assert buffer.turned.tolist() == buffer.received.tolist() == [1.0, 8.0]


def test_yaw_gain_feeds_the_generator_more_yaw_on_every_mapping_and_the_camera_the_turn_made():
    """``play.yaw_gain`` scales the yaw of the view controls only: at 1 nothing changes, the pitch,
    the turn and the page's count are the turn made, and ``hybrid`` multiplies its own gain."""
    for mapping in ("latest", "front", "realtime", "hybrid"):
        _, same = _block(mapping, pitch=1.0, yaw=80.0)
        _, one = _block(mapping, pitch=1.0, yaw=80.0, yaw_gain=1.0)
        assert np.array_equal(one.view_deltas, same.view_deltas), mapping
    _, plain = _block("latest", pitch=1.0, yaw=80.0)
    buffer, gained = _block("latest", pitch=1.0, yaw=80.0, yaw_gain=1.25)
    assert np.array_equal(gained.turn, plain.turn) and np.array_equal(gained.turned, plain.turned)
    assert buffer.turned.tolist() == buffer.received.tolist() == [1.0, 80.0]
    assert _buckets(gained)[:, 0].tolist() == _buckets(plain)[:, 0].tolist()  # not the pitch
    assert _buckets(plain)[:, 1].tolist() == [16] * 16  # 5.0 degrees a frame: bucket 16
    assert _buckets(gained)[:, 1].tolist() == [19] * 16  # fed 6.25 degrees a frame: bucket 19
    _, hybrid = _block("hybrid", yaw=80.0, yaw_gain=1.25)
    _, hybrid_alone = _block("hybrid", yaw=80.0)
    assert (_buckets(hybrid)[:, 1] > _buckets(hybrid_alone)[:, 1]).all()  # 1.25 x 1.1 over 1.1


def test_hybrid_mapping_gain_takes_a_sweep_of_300_degrees_past_the_trained_range():
    """A block of 300 degrees of yaw is 18.75 degrees a frame, within the generator's trained 20
    (buckets of 0.5 degree: 40 is 20 degrees), and the gain feeds it 20.6 (bucket 41; the comment
    of ``HYBRID_YAW_GAIN`` names the band). ``latest`` feeds 39 for it, and a block of 320 degrees
    reaches 40 with no gain and 42 with it."""
    for yaw, plain, gained in [(300.0, 39, 41), (320.0, 40, 42)]:
        _, hybrid = _block("hybrid", yaw=yaw)
        assert hybrid.turn[:, 1].tolist() == [yaw / 16] * 16
        assert _buckets(hybrid)[:, 1].tolist() == [gained] * 16
        assert _buckets(_block("latest", yaw=yaw)[1])[:, 1].tolist() == [plain] * 16


def test_hybrid_mapping_holds_no_turn_back_so_the_page_counts_every_degree():
    """A turn under the threshold is done in its own block (it is not carried to a later one): the
    page's ``turned`` is the turn received, block by block."""
    buffer = InputBuffer(fps=16.0, mapping="hybrid", weapon=2, release_after_s=1.0)
    for block in range(6):  # 2 degrees a block
        buffer.add(InputSample(block, block + 0.5, 0, dpitch=0.0, dyaw=2.0, weapon=2))
        controls = buffer.cut(now=float(block + 1), frames=16)
        assert controls.turn[:, 1].sum() == 2.0 and controls.turned[-1, 1] == 2.0 * (block + 1)
        assert controls.turned[3, 1] == controls.turned[-1, 1]  # all in the first latent frame
        assert buffer.turned.tolist() == buffer.received.tolist()
    buffer.add(InputSample(9, 6.5, 0, dpitch=0.0, dyaw=40.0, weapon=2))  # a sweep: smooth
    sweep = buffer.cut(now=7.0, frames=16)
    assert np.allclose(np.diff(sweep.turned[:, 1], prepend=12.0), 40.0 / 16)


def test_hybrid_mapping_over_random_blocks_is_front_or_latest_by_the_total_with_the_yaw_gain(
    latent_front,
):
    """6,000 blocks of 1-5 samples of whole mouse pixels, a third of them with a total within a few
    pixels of the threshold, either sign on either axis. The turn rows are those of ``front`` where
    the larger axis of the block's total is under 20 degrees and of ``latest`` where it is not;
    the view controls are those of the same mapping given the block's yaw 1.1 times."""
    rng = np.random.default_rng(6000)
    degrees_per_pixel = 0.055
    for _ in range(6000):
        count = int(rng.integers(1, 6))
        if rng.random() < 1 / 3:
            totals = rng.choice([-1, 1], 2) * rng.integers(358, 370, 2)  # about 20 degrees
        else:
            scale = 10 ** rng.uniform(0.0, 3.5)  # up to 3,000 pixels = 170 degrees
            totals = np.round(rng.normal(0.0, scale, 2)).astype(int)
        pixels = np.zeros((count, 2), int)
        pixels[:] = totals // count
        pixels[-1] += totals - pixels.sum(axis=0)
        samples = [
            InputSample(k, 0.1 * k, 0, *(row * degrees_per_pixel), weapon=2)
            for k, row in enumerate(pixels)
        ]
        total = np.array([(s.dpitch, s.dyaw) for s in samples]).sum(axis=0)
        mapping = "front" if np.abs(total).max() < HYBRID_DEGREES else "latest"

        hybrid, as_made = (_cut(m, samples)[1] for m in ("hybrid", mapping))
        fed = _cut(mapping, [InputSample(0, 0.1, 0, total[0], HYBRID_YAW_GAIN * total[1], 2)])[1]
        for name in ("buttons", "weapon", "times", "turn", "turned"):
            assert np.array_equal(getattr(hybrid, name), getattr(as_made, name)), name
        assert np.array_equal(hybrid.view_deltas, fed.view_deltas)


@pytest.mark.parametrize(
    ("mapping", "axis", "idle", "moving"),
    [
        ("hybrid", 1, 0.44, 0.46),  # the yaw is fed 1.1 times: its 0.5-degree buckets start sooner
        ("hybrid", 0, 0.46, 0.5),
        ("front", 1, 0.23, 0.25),  # over the first 2 frames: under about 0.12 degrees a frame
        ("latest", 1, 1.9, 2.0),  # over the 16 frames
        ("realtime", 1, 0.12, 0.125),  # on the one frame of a sample
    ],
)
def test_a_block_is_idle_where_its_turn_encodes_to_zero_in_the_view_controls(
    mapping, axis, idle, moving
):
    """What ``is_idle`` (the context of a client whose player is idle, ``live.idle_context``) makes
    of the cut of one sample of ``idle`` or ``moving`` degrees: the mapping decides how far a turn
    is spread, and the view controls round a frame's turn of under about 0.12 degrees to zero."""
    for degrees, expected in ((idle, True), (moving, False)):
        turn = [0.0, 0.0]
        turn[axis] = degrees
        _, controls = _cut(mapping, [InputSample(0, 0.5, 0, *turn, weapon=2)])
        assert is_idle(controls) is expected, (mapping, degrees)


def test_a_block_without_a_sample_is_idle_until_a_button_is_held():
    buffer = InputBuffer(fps=16.0, mapping="hybrid", weapon=2, release_after_s=1.0)
    assert is_idle(buffer.cut(now=1.0, frames=16))
    buffer.add(InputSample(0, 1.5, 1 << FORWARD, dpitch=0.0, dyaw=0.0, weapon=2))
    assert [is_idle(buffer.cut(now=now, frames=16)) for now in (2.0, 2.5, 3.0)] == [
        False,
        False,
        True,
    ]


# ------------------------------------------------------------------------------------- refreshes
def _recut(mapping: str, from_row: int, *samples: InputSample):
    """:func:`_buffer`'s block of ``mapping`` cut at room time 1, then ``samples`` and its recut
    from ``from_row`` at room time 1.3; returns the buffer, the cut and the recut."""
    buffer = _buffer(mapping)
    first = buffer.cut(now=1.0, frames=16)
    for sample in samples:
        buffer.add(sample)
    return buffer, first, buffer.recut(now=1.3, frames=16, from_row=from_row)


def test_a_change_of_button_or_weapon_or_a_turn_past_the_threshold_asks_for_a_refresh():
    """The cut's newest sample turns left; samples that go on turning left ask for a refresh once
    they turn a degree since the last cut, or recut; a button or the weapon changing asks for one
    at once."""
    buffer = _buffer("latest")
    buffer.cut(now=1.0, frames=16)  # forward held, turning left
    assert not buffer.changed(1.0)
    buffer.add(InputSample(3, 1.1, 1 << FORWARD, dpitch=0.0, dyaw=0.6, weapon=2))
    assert not buffer.changed(1.0)  # the same button, 0.6 degrees the same way
    buffer.add(InputSample(4, 1.15, 1 << FORWARD, dpitch=0.0, dyaw=0.5, weapon=2))
    assert buffer.changed(1.0) and not buffer.changed(1.2)  # 1.1 degrees of yaw in all
    buffer.recut(now=1.15, frames=16, from_row=8)
    for seq in range(5, 13):  # a slow pan, 0.125 degrees a sample (7.5 degrees a second)
        assert not buffer.changed(1.0)
        buffer.add(InputSample(seq, 1.2, 1 << FORWARD, dpitch=0.0, dyaw=0.125, weapon=2))
    assert buffer.changed(1.0)  # a degree since the recut
    for change in (
        InputSample(13, 1.2, 1 << FORWARD | 1 << JUMP, 0.0, 0.1, weapon=2),
        InputSample(13, 1.2, 0, 0.0, 0.1, weapon=2),  # forward let go
        InputSample(13, 1.2, 1 << FORWARD, 0.0, 0.1, weapon=3),
    ):
        buffer.recut(now=1.2, frames=16, from_row=8)
        buffer.add(change)
        assert buffer.changed(90.0)


def test_a_turn_that_starts_stops_or_turns_back_asks_for_a_refresh():
    """However small: the sign of the newest sample's turn against the newest one the cuts folded
    in, on either axis. A turn that goes on the same way does not, until it passes the
    threshold."""
    buffer = InputBuffer(fps=16.0, mapping="front", weapon=2, release_after_s=1.0)
    buffer.add(InputSample(0, 0.5, 0, dpitch=0.0, dyaw=0.0, weapon=2))
    buffer.cut(now=1.0, frames=16)
    assert buffer.direction.tolist() == [0.0, 0.0] and not buffer.changed(1.0)
    seq = iter(range(1, 100))
    for turn, asks in (
        ((0.0, 0.0), False),  # at rest
        ((0.0, 0.05), True),  # starts turning left
        ((0.0, 0.05), False),  # goes on
        ((0.0, -0.05), True),  # turns back
        ((0.0, 0.0), True),  # stops
        ((0.05, 0.0), True),  # looks down
        ((-0.05, 0.0), True),  # and up
    ):
        buffer.add(InputSample(next(seq), 1.1, 0, *turn, weapon=2))
        assert buffer.changed(1.0) is asks, turn
        buffer.recut(now=1.1, frames=16, from_row=8)
        assert buffer.direction.tolist() == np.sign(turn).tolist()
    # a turn back within the samples of one cut counts by the newest sample
    buffer.add(InputSample(next(seq), 1.2, 0, 0.0, -0.3, weapon=2))
    buffer.add(InputSample(next(seq), 1.2, 0, -0.05, 0.0, weapon=2))
    assert not buffer.changed(1.0)  # the newest sample is the cut's: looking up
    buffer.add(InputSample(next(seq), 1.2, 0, -0.05, 0.1, weapon=2))
    assert buffer.changed(1.0)  # and now turning left too


def test_a_refresh_does_the_new_turn_in_the_first_rows_after_its_switch_with_front():
    """``front`` (the demo's mapping with refresh): a flick of 30 degrees that comes after the cut
    (with the 4 the cut left for the rows from row 9, none) is done in the two rows from the
    switch, one of 90 in five (within the trained 20 degrees a row), a degree in two rows; the rows
    before the switch keep their turn and the block's turn is every sample's. Near the block's end
    the rows left take it all."""
    for flick, rows, row in ((30.0, 2, 9), (90.0, 5, 6), (1.0, 2, 3), (12.0, 2, 14), (39.0, 3, 1)):
        buffer = InputBuffer(fps=16.0, mapping="front", weapon=2, release_after_s=1.0)
        buffer.add(InputSample(0, 0.5, 0, dpitch=0.0, dyaw=4.0, weapon=2))
        first = buffer.cut(now=1.0, frames=16)
        buffer.add(InputSample(1, 1.1, 0, dpitch=0.0, dyaw=flick, weapon=2))
        again = buffer.recut(now=1.1, frames=16, from_row=row)
        np.testing.assert_array_equal(again.turn[:row], first.turn[:row])
        left = first.turn[row:, 1].sum() + flick
        expected = np.zeros(16 - row)
        expected[:rows] = left / rows
        np.testing.assert_allclose(again.turn[row:, 1], expected, atol=1e-5)
        assert np.abs(again.turn[:, 1]).max() <= max(20.0, left / rows) + 1e-5
        np.testing.assert_allclose(again.turned[-1], (0.0, 4.0 + flick), atol=1e-9)
        np.testing.assert_allclose(buffer.turned, buffer.received, atol=1e-9)


@pytest.mark.parametrize("mapping", ["latest", "front", "hybrid"])
def test_a_recut_keeps_the_rows_on_their_way_and_gives_the_rest_the_new_input(mapping):
    """From row 6 on: the right strafe held now, the jump tapped since the cut in row 6, the newest
    sample's weapon and seq, and the turn the cut left for those rows plus the new samples' spread
    as the mapping spreads a block's; before it the cut's rows, bit for bit."""
    jump, right = 1 << JUMP, 1 << RIGHT
    buffer, first, again = _recut(
        mapping,
        6,
        InputSample(3, 1.1, right | jump, dpitch=1.0, dyaw=2.0, weapon=3),
        InputSample(4, 1.2, right, dpitch=0.0, dyaw=2.0, weapon=3),
    )
    for name in ROWS:
        assert np.array_equal(getattr(again, name)[:6], getattr(first, name)[:6]), name
    assert np.array_equal(again.times, first.times)
    assert again.buttons[6:, RIGHT].all() and not again.buttons[6:, FORWARD].any()
    assert again.buttons[6:, JUMP].tolist() == [1.0] + [0.0] * 9
    assert (again.weapon[6:] == 3).all() and (again.input_seq[6:] == 4).all()
    left = first.turn[6:].astype(np.float64).sum(axis=0) + (1.0, 4.0)
    np.testing.assert_allclose(again.turn[6:].sum(axis=0), left, atol=1e-5)
    spread = {"latest": 10, "front": 2, "hybrid": 4}[mapping]  # hybrid: under 20 degrees
    assert np.count_nonzero(again.turn[6:, 1]) == spread
    # the block's turn is every sample's: the page's count is the turn received
    np.testing.assert_allclose(again.turned[-1], (1.0, 12.0), atol=1e-9)
    assert buffer.turned.tolist() == again.turned[-1].tolist()
    np.testing.assert_allclose(buffer.turned, buffer.received, atol=1e-9)
    np.testing.assert_allclose(np.diff(again.turned, axis=0, prepend=0.0), again.turn, atol=1e-5)
    codes = [quantize_camera_delta(row, clip=False) for row in again.turn[6:]]
    if mapping != "hybrid":  # hybrid feeds the generator its yaw 1.1 times
        assert np.array_equal(again.view_deltas[6:], np.stack(codes))


def test_a_recut_goes_on_from_the_last_recut_and_the_next_block_from_it():
    buffer, first, again = _recut("latest", 4, InputSample(3, 1.1, 0, 0.0, 3.0, weapon=2))
    buffer.add(InputSample(4, 1.25, 1 << JUMP, 0.0, -1.0, weapon=2))
    third = buffer.recut(now=1.35, frames=16, from_row=10)
    for name in ROWS:
        assert np.array_equal(getattr(third, name)[:10], getattr(again, name)[:10]), name
    assert third.buttons[10:, JUMP].all() and not third.buttons[4:10, JUMP].any()
    np.testing.assert_allclose(third.turned[-1], (0.0, 10.0), atol=1e-9)
    buffer.add(InputSample(5, 1.8, 0, 0.0, 2.0, weapon=2))
    following = buffer.cut(now=2.0, frames=16)
    np.testing.assert_allclose(following.turned[0], (0.0, 10.0 + 2.0 / 16), atol=1e-9)
    assert not following.buttons[:, JUMP].any()  # let go after the recut took it


def test_a_recut_from_row_0_keeps_the_taps_of_the_cut_and_one_past_the_end_takes_the_last_row():
    jump = 1 << JUMP
    _, first, again = _recut("latest", 0, InputSample(3, 1.1, 0, 0.0, 0.0, weapon=2))
    # row 0 holds every button seen since the block before, the cut's taps and held keys too
    assert first.buttons[0, JUMP] and again.buttons[0, JUMP] and again.buttons[0, FORWARD]
    assert not again.buttons[1:, FORWARD].any()  # let go since the cut
    buffer, first, last = _recut("front", 16, InputSample(3, 1.1, jump, 0.0, 5.0, weapon=2))
    for name in ROWS:
        assert np.array_equal(getattr(last, name)[:15], getattr(first, name)[:15]), name
    assert last.buttons[15, JUMP] and last.turn[15, 1] == 5.0
    assert buffer.turned.tolist() == [0.0, 13.0]


def test_a_realtime_recut_puts_each_new_sample_on_its_frame_from_the_row_on():
    """``realtime``: the rows from ``from_row`` keep what the cut put there, and each new sample
    goes on the frame shown while it was taken, at ``from_row`` at the earliest."""
    _, first, again = _recut(
        "realtime",
        8,
        InputSample(3, 0.2, 1 << JUMP, 0.0, 1.0, weapon=2),  # taken before the row: lands on it
        InputSample(4, 0.9, 0, 0.0, 2.0, weapon=2),
    )
    late = int(np.searchsorted(first.times, 0.9))
    for name in ROWS:
        assert np.array_equal(getattr(again, name)[:8], getattr(first, name)[:8]), name
    expected = first.turn[:, 1].copy()
    expected[8] += 1.0
    expected[late] += 2.0
    assert again.turn[:, 1].tolist() == expected.tolist()
    # a sample counts for its frame and sets the buttons after it, as a cut's
    assert again.buttons[8 : late + 1, JUMP].all() and not again.buttons[late + 1 :, JUMP].any()
    assert again.turned[-1].tolist() == [0.0, 11.0]


def test_a_recut_needs_a_block_cut_before():
    buffer = InputBuffer(fps=16.0, mapping="latest", weapon=2, release_after_s=1.0)
    with pytest.raises(RuntimeError, match="no block was cut"):
        buffer.recut(now=1.0, frames=16, from_row=3)
    buffer.cut(now=1.0, frames=16)
    with pytest.raises(ValueError, match="16 rows, not 4"):
        buffer.recut(now=1.0, frames=4, from_row=3)


def test_the_inspect_button_reaches_the_generator_only_with_a_knife():
    """Every knife of the vocabulary (an alias of one included) and nothing else inspects. The
    inspect button held through a switch from the AK-47 to the knife and back holds in the knife's
    rows of the cut and of a recut, and nowhere else; the other buttons are kept."""
    knives = {name for i, name in enumerate(OPENCS2_WEAPONS) if i in INSPECT_WEAPONS}
    assert knives == {
        "bayonet",
        "knife",
        "knife_butterfly",
        "knife_karambit",
        "knife_m9_bayonet",
        "knife_stiletto",
    }
    assert {encode_weapon(n) for n in ("weapon_knife", "shadowdaggers", "m9bayonet")} <= set(
        INSPECT_WEAPONS
    )
    ak, knife = encode_weapon("ak47"), encode_weapon("knife")
    held = 1 << FORWARD | 1 << INSPECT
    for mapping in ("latest", "front", "hybrid"):
        buffer = InputBuffer(fps=16.0, mapping=mapping, weapon=ak, release_after_s=1.0)
        buffer.add(InputSample(1, 0.1, held, 0.0, 0.0, weapon=ak))
        gun = buffer.cut(now=1.0, frames=16)
        assert not gun.buttons[:, INSPECT].any() and gun.buttons[:, FORWARD].all(), mapping
        buffer.add(InputSample(2, 1.1, held, 0.0, 0.0, weapon=knife))
        drawn = buffer.recut(now=1.1, frames=16, from_row=8)
        assert drawn.buttons[:, INSPECT].tolist() == [0.0] * 8 + [1.0] * 8, mapping
        assert (drawn.weapon[8:] == knife).all() and drawn.buttons[:, FORWARD].all()
        kept = buffer.cut(now=2.0, frames=16)  # no new sample: the knife and the key stay held
        assert kept.buttons[:, INSPECT].all() and (kept.weapon == knife).all()
        buffer.add(InputSample(3, 2.5, held, 0.0, 0.0, weapon=ak))
        back = buffer.cut(now=3.0, frames=16)
        assert not back.buttons[:, INSPECT].any() and (back.weapon == ak).all(), mapping


def test_the_inspect_button_with_a_gun_asks_for_no_refresh():
    """With a gun the inspect button is no change of the controls; with the knife it is one."""
    ak, knife = encode_weapon("ak47"), encode_weapon("knife")
    for weapon, asks in ((ak, False), (knife, True)):
        buffer = InputBuffer(fps=16.0, mapping="front", weapon=weapon, release_after_s=1.0)
        buffer.add(InputSample(1, 0.1, 1 << FORWARD, 0.0, 0.0, weapon=weapon))
        buffer.cut(now=1.0, frames=16)
        buffer.add(InputSample(2, 1.1, 1 << FORWARD | 1 << INSPECT, 0.0, 0.0, weapon=weapon))
        assert buffer.changed(1.0) is asks


def _flick(buffer: InputBuffer, first_seq: int, at: float, turns) -> None:
    """Samples a frame apart from room time ``at``, turning ``turns`` of yaw."""
    for i, dyaw in enumerate(turns):
        buffer.add(InputSample(first_seq + i, at + i / 60, 0, 0.0, dyaw, 2))


def test_a_refresh_for_a_turn_that_starts_waits_a_little_for_more_of_it_unless_it_is_large():
    """A flick after a rest: its first sample asks for a refresh, which waits until ``GATHER_S``
    after that sample arrived, unless the turn has reached ``GATHER_DEGREES`` (a large flick goes
    at once); a button, a turn that goes on and a turn that only stops wait for nothing, and a
    turn back waits as one that starts."""
    assert (GATHER_S, GATHER_DEGREES) == (0.02, 20.0)
    buffer = InputBuffer(fps=16.0, mapping="front", weapon=2, release_after_s=1.0)
    buffer.cut(now=1.0, frames=16)
    _flick(buffer, 1, 1.0, [3.0])
    assert buffer.changed(1.0) and buffer.gathering(1.0) == pytest.approx(GATHER_S)
    assert buffer.gathering(1.005) == pytest.approx(GATHER_S - 0.005)
    assert buffer.gathering(1.03) == 0.0  # waited long enough
    _flick(buffer, 2, 1.0 + 1 / 60, [10.0, 7.0])  # 20 degrees: a large flick goes at once
    assert buffer.gathering(1.005) == 0.0
    # a large first sample goes at once
    buffer = InputBuffer(fps=16.0, mapping="front", weapon=2, release_after_s=1.0)
    buffer.cut(now=1.0, frames=16)
    _flick(buffer, 1, 1.0, [-35.0])
    assert buffer.changed(1.0) and buffer.gathering(1.0) == 0.0
    # a button pressed with a small turn goes at once
    buffer = InputBuffer(fps=16.0, mapping="front", weapon=2, release_after_s=1.0)
    buffer.cut(now=1.0, frames=16)
    _flick(buffer, 1, 1.0, [3.0])
    buffer.add(InputSample(9, 1.01, 1 << FORWARD, 0.0, 0.0, 2))
    assert buffer.gathering(1.01) == 0.0
    # a turn that goes on (past the threshold) and one that stops wait for nothing
    buffer = InputBuffer(fps=16.0, mapping="front", weapon=2, release_after_s=1.0)
    _flick(buffer, 1, 0.5, [3.0, 3.0])
    buffer.cut(now=1.0, frames=16)
    assert buffer.direction.tolist() == [0.0, 1.0]
    _flick(buffer, 3, 1.0, [1.0, 1.0])
    assert buffer.changed(1.0) and buffer.gathering(1.0) == 0.0
    buffer.recut(now=1.0, frames=16, from_row=4)
    _flick(buffer, 5, 1.0, [0.0])
    assert buffer.changed(1.0) and buffer.gathering(1.0) == 0.0
    # a turn back waits as one that starts
    buffer.recut(now=1.0, frames=16, from_row=4)
    _flick(buffer, 6, 1.1, [-1.0])
    assert buffer.gathering(1.1) == pytest.approx(GATHER_S)


@pytest.mark.parametrize("gather_s", [0.0, 0.04])
def test_a_refresh_for_a_turn_that_starts_waits_as_long_as_the_setting_says(gather_s):
    """``gather_s`` (``play.gather_ms``) in place of ``GATHER_S``: 0 waits for nothing."""
    buffer = InputBuffer(
        fps=16.0, mapping="front", weapon=2, release_after_s=1.0, gather_s=gather_s
    )
    buffer.cut(now=1.0, frames=16)
    _flick(buffer, 1, 1.0, [3.0])
    assert buffer.changed(1.0) and buffer.gathering(1.0) == pytest.approx(gather_s)
    assert buffer.gathering(1.01) == pytest.approx(max(gather_s - 0.01, 0.0))


def test_a_flick_the_block_cannot_hold_within_the_trained_range_is_carried_to_the_next_cut():
    """``front`` does a turn in as few rows as keep each within 20 degrees as the generator is fed
    it: 180 degrees of yaw in nine rows (in twelve fed 1.25 times); a flick of 400 in all sixteen
    rows, 20 a row, and the 80 left carried to the next block's cut, which does it first. A
    refresh near the block's end does what its rows hold and carries the rest. The page's count
    stays the turn received: the turn cut, and the carry once the next cut takes it."""
    for gain, rows in ((1.0, 9), (1.25, 12)):
        _, controls = _cut("front", [InputSample(0, 0.5, 0, 0.0, 180.0, 2)], yaw_gain=gain)
        assert np.count_nonzero(controls.turn[:, 1]) == rows, gain
        np.testing.assert_allclose(controls.turn[:rows, 1], 180.0 / rows)
        assert np.abs(controls.turn[:, 1]).max() * gain <= 20.0 + 1e-9
    buffer, flick = _cut("front", [InputSample(0, 0.5, 0, 20.0, 400.0, 2)])
    np.testing.assert_allclose(flick.turn, [[1.0, 20.0]] * 16)  # the pitch with it, in step
    np.testing.assert_allclose(buffer.carry, (4.0, 80.0))
    np.testing.assert_allclose(buffer.turned, (16.0, 320.0))
    np.testing.assert_allclose(buffer.received - buffer.turned, buffer.carry)
    buffer.add(InputSample(1, 1.5, 0, 0.0, 10.0, 2))
    after = buffer.cut(now=2.0, frames=16)
    np.testing.assert_allclose(after.turn[:5], [[0.8, 18.0]] * 5)  # 90 degrees in five rows
    assert not after.turn[5:].any() and not buffer.carry.any()
    np.testing.assert_allclose(buffer.turned, buffer.received)
    # a refresh near the end of the block
    buffer = InputBuffer(fps=16.0, mapping="front", weapon=2, release_after_s=1.0)
    first = buffer.cut(now=1.0, frames=16)
    buffer.add(InputSample(1, 1.1, 0, 0.0, -100.0, 2))
    again = buffer.recut(now=1.1, frames=16, from_row=13)
    np.testing.assert_allclose(again.turn[13:, 1], [-20.0] * 3)
    assert np.array_equal(again.turn[:13], first.turn[:13])
    np.testing.assert_allclose(buffer.carry, (0.0, -40.0))
    np.testing.assert_allclose(again.turned[-1], (0.0, -60.0))
    following = buffer.cut(now=2.0, frames=16)
    np.testing.assert_allclose(following.turn[:2, 1], [-20.0] * 2)
    np.testing.assert_allclose(buffer.turned, buffer.received)


def test_front_does_the_turn_in_as_few_rows_as_keep_the_fed_turn_within_the_trained_range():
    """``front`` does a turn in two rows when the turn the generator is fed fits, 20 degrees a row:
    34 degrees of yaw in two (17 a row), fed 1.25 times (42.5) in three; the turn is the turn
    made, the view controls the fed one, the pitch's gain on the pitch."""
    for gain, rows in ((1.0, 2), (1.25, 3)):
        _, controls = _cut("front", [InputSample(0, 0.5, 0, 4.0, 34.0, 2)], yaw_gain=gain)
        assert np.count_nonzero(controls.turn[:, 1]) == rows, gain
        np.testing.assert_allclose(controls.turn[:rows], [[4.0 / rows, 34.0 / rows]] * rows)
        fed = [quantize_camera_delta((4.0 / rows, gain * 34.0 / rows), clip=False)] * rows
        np.testing.assert_array_equal(controls.view_deltas[:rows], np.stack(fed))
    buffer, pitched = _cut("front", [InputSample(0, 0.5, 0, 4.0, 10.0, 2)], pitch_gain=1.5)
    np.testing.assert_array_equal(
        pitched.view_deltas[:2], np.stack([quantize_camera_delta((3.0, 5.0), clip=False)] * 2)
    )
    assert buffer.turned.tolist() == [4.0, 10.0] and pitched.turn[:2].tolist() == [[2.0, 5.0]] * 2
