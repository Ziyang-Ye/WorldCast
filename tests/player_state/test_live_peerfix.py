"""Two rules of the camera that keep another player in the view where the picture has it, both off
by default (``live.dead_zone_u``, ``live.hop``): the motion head is read for a tap and a slow walk
too, so that the position estimate moves with the camera only as far as the view followed it; and
the camera of a block starts where the block before it ended and runs into the estimate over the
block, so that what the view did not follow shows as a glide and not as a step at the block's start.
"""

import numpy as np
import pytest

from tests.player_state.support import PRIOR, SUBSTEPS, open_ground
from tests.player_state.test_live import (
    FORWARD,
    LATENTS,
    CameraReader,
    Room,
    keys,
    live_loop,
    own_rows,
    play,
)
from worldcast.config.inference import HOP_MODES, LiveConfig
from worldcast.data.latents import BLOCK, last_video_frame
from worldcast.player_state.live import HOP_WEIGHTS
from worldcast.player_state.live_position import (
    IDLE_STEP_U,
    STUCK_MIN_STEP_U,
    GuardedPosition,
)

#: The prior's top speed over a latent frame, u.
TOP_STEP_U = 245.0 / 4.0
DEAD_ZONE_U = 4.0


def position(dead_zone_u: float | None = None) -> GuardedPosition:
    kwargs = {} if dead_zone_u is None else {"dead_zone_u": dead_zone_u}
    return GuardedPosition(
        np.zeros(3), np.zeros(3), top_step_u=TOP_STEP_U, walkable=open_ground(), **kwargs
    )


def walk(pos: GuardedPosition, steps: np.ndarray, view: np.ndarray) -> np.ndarray:
    """The estimates after latent frames whose camera and prior move ``steps`` ``[n, 2]`` (u) and
    whose motion head reads the view moving ``view`` ``[n]`` u; the place head says nothing."""
    out = []
    for step, seen in zip(steps, view):
        prior = np.array([step[0], step[1], 0.0])
        read = np.array([seen / 64.0, 0.0, 0.0])
        out.append(pos.step(np.full(3, 1e6), read, prior, prior))
    return np.stack(out)


def test_the_new_settings_are_off_by_default_and_checked():
    assert (LiveConfig().dead_zone_u, LiveConfig().hop) == (0.0, "step")
    assert HOP_MODES == ("step", "spread")
    assert LiveConfig(dead_zone_u=4.0, hop="spread").hop == "spread"
    with pytest.raises(ValueError, match="live.dead_zone_u"):
        LiveConfig(dead_zone_u=-1.0)
    with pytest.raises(ValueError, match="live.hop"):
        LiveConfig(hop="glide")


def test_a_tap_while_the_view_stands_moves_the_estimate_by_at_most_the_dead_zone():
    steps = np.tile([[12.0, 0.0]], (4, 1))  # a tap: under STUCK_MIN_STEP_U a latent frame
    assert 12.0 < STUCK_MIN_STEP_U
    stands = np.array([0.0, 1.0, 3.0, 0.0])  # the head's noise, below the dead zone
    on = walk(position(DEAD_ZONE_U), steps, stands)
    assert np.linalg.norm(on[-1, :2]) <= DEAD_ZONE_U
    off = walk(position(), steps, stands)
    assert off[-1, 0] == pytest.approx(48.0)  # off: the camera's steps carry the estimate


def test_a_step_the_view_followed_carries_the_estimate_whole_with_the_dead_zone():
    steps = np.tile([[12.0, 0.0]], (4, 1))
    on = walk(position(DEAD_ZONE_U), steps, np.full(4, 12.0))
    assert on[-1, 0] == pytest.approx(48.0)


def test_a_view_that_followed_part_of_a_small_step_carries_part_of_it():
    steps = np.tile([[20.0, 0.0]], (6, 1))
    on = walk(position(DEAD_ZONE_U), steps, np.full(6, 5.0))  # 25 % of the camera's step
    moved = on[-1, 0]
    assert 0.0 < moved < 120.0


def test_a_view_below_the_dead_zone_reads_as_standing_and_one_above_it_does_not():
    steps = np.tile([[10.0, 0.0]], (3, 1))
    assert walk(position(DEAD_ZONE_U), steps, np.full(3, 3.0))[-1, 0] == 0.0
    assert walk(position(DEAD_ZONE_U), steps, np.full(3, 6.0))[-1, 0] > 0.0


def test_a_standing_client_and_a_running_one_are_the_same_with_the_dead_zone_on_or_off():
    rng = np.random.default_rng(0)
    for length in (0.0, 0.5, 80.0):  # no step, below IDLE_STEP_U, and over STUCK_MIN_STEP_U
        steps = np.tile([[length, 0.0]], (6, 1))
        view = rng.uniform(0.0, 90.0, 6)
        off = walk(position(), steps, view)
        assert np.array_equal(off, walk(position(0.0), steps, view))
        assert np.array_equal(off, walk(position(DEAD_ZONE_U), steps, view))
    assert length >= STUCK_MIN_STEP_U and 0.5 < IDLE_STEP_U


class SlowReader(CameraReader):
    """The state model of a view that stands: the motion head reads nothing, and the place head
    nothing of the camera's moves."""

    def read(self, latents, latent_frames):
        place, steps = super().read(latents, latent_frames)
        return np.repeat(self.motion.start_state[None, :3], len(place), 0), steps * 0.0


def tap(loop, blocks: int = 4) -> np.ndarray:
    """Block 1 pressed forward for four rows only, then blocks standing: where the estimate ends."""
    buttons = keys()
    buttons[:4, FORWARD] = 1.0
    for s in range(1, 1 + BLOCK * blocks, BLOCK):
        play(loop, s, buttons if s == 1 else keys())
    return loop.cameras.track.xyz[-1].copy()


def test_a_tap_in_a_live_loop_does_not_carry_the_estimate_when_the_view_stands():
    start = live_loop(Room(), 0, reader=SlowReader()).cameras.track.xyz[0]
    on = tap(live_loop(Room(), 0, reader=SlowReader(), dead_zone_u=DEAD_ZONE_U))
    off = tap(live_loop(Room(), 0, reader=SlowReader()))
    assert np.linalg.norm((on - start)[:2]) <= DEAD_ZONE_U
    assert np.linalg.norm((off - start)[:2]) > 2 * DEAD_ZONE_U


def test_the_loop_is_what_it_was_with_both_settings_off():
    rows = []
    for kwargs in ({}, {"dead_zone_u": 0.0, "hop": "step"}):
        loop = live_loop(Room(), 0, reader=CameraReader(), **kwargs)
        for s in range(1, 1 + 6 * BLOCK, BLOCK):
            play(loop, s, keys(FORWARD) if s < 14 else keys())
        rows.append(loop.batch["player_states"].clone())
        assert loop.motion.spread is None
    assert (rows[0] == rows[1]).all()


class HalfReader(CameraReader):
    """The state model of a view that follows 30 % of the camera: the estimate falls behind it."""

    def read(self, latents, latent_frames):
        place, steps = super().read(latents, latent_frames)
        return place + [2000.0, 2000.0, 0.0], steps * 0.3


def run_blocks(hop: str, blocks: int = 5):
    loop = live_loop(Room(), 0, reader=HalfReader(), hop=hop)
    for s in range(1, 1 + blocks * BLOCK, BLOCK):
        play(loop, s, keys(FORWARD))
    return loop


def starts_and_ends(loop, blocks: int):
    """Per block, ``(first row x, last row x)`` of the camera rows the field is given."""
    return [
        own_rows(loop, 1 + BLOCK * b)[[0, -1], 0].numpy().astype(np.float64) for b in range(blocks)
    ]


def test_the_camera_of_a_block_starts_where_the_block_before_it_ended_with_the_hop_spread():
    step, spread = run_blocks("step"), run_blocks("spread")
    a, b = starts_and_ends(step, 5), starts_and_ends(spread, 5)
    # one row of the camera is at most a few steps (the prior's running step over a video frame)
    row_u = TOP_STEP_U / SUBSTEPS
    hops_step = [a[i + 1][0] - a[i][1] for i in range(1, 4)]
    hops_spread = [b[i + 1][0] - b[i][1] for i in range(1, 4)]
    assert min(hops_step) < -3 * row_u  # the camera steps back at the block's start
    assert all(0.0 <= h <= 2 * row_u for h in hops_spread)
    # the camera of the last row is the estimate's, with the hop spread or not
    assert np.allclose([x[1] for x in a], [x[1] for x in b], atol=1e-3)


def test_the_rays_and_the_field_read_the_same_camera_with_the_hop_spread():
    loop = run_blocks("spread")
    table = loop.batch["player_states"][0, 0]
    for s in range(5, 1 + 5 * BLOCK, BLOCK):
        rows = loop.cameras.block_rows(s)[:, :3]
        last = [last_video_frame(s + k) for k in range(BLOCK)]
        assert np.allclose(rows, table[last, :3].numpy(), atol=1e-2)


def test_the_spread_is_a_hop_that_is_gone_by_the_last_row_of_the_block():
    assert HOP_WEIGHTS[-1] == 0.0 and (np.diff(HOP_WEIGHTS) < 0).all() and HOP_WEIGHTS[0] < 1.0
    loop = run_blocks("spread")
    spread = loop.motion.spread
    assert spread is not None and spread.shape == loop.motion.displacement.shape
    ends = [last_video_frame(s + BLOCK - 1) for s in range(1, 1 + 5 * BLOCK, BLOCK)]
    assert np.abs(spread[ends]).max() == 0.0


def test_a_standing_client_has_no_hop_to_spread():
    loop = live_loop(Room(), 0, reader=CameraReader(), hop="spread")
    for s in range(1, 1 + 4 * BLOCK, BLOCK):
        play(loop, s, keys())
    assert np.abs(loop.motion.spread).max() == 0.0
    assert LATENTS > 4 * BLOCK and PRIOR is not None
