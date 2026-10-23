"""The own position under live controls: the guarded estimator never jumps, never moves a standing
player, lands a jump in place where it took off, steps with the camera as far as the motion head
says the generated view followed it (the place head's say is not needed for a wall), falls at most
two body widths behind the camera within a block, follows the place head while it agrees, and
follows it to where it is when it disagrees for seconds without jumping, after agreeing lately, or
the estimate is at the map's edge; it moves no faster than a player runs and never off the map, and
survives the place head's failures; replayed on excerpts of logged head outputs from a live play on
a GPU; the wall ahead is read off the depth head's surface, the edge of the map off the cell
table."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.player_state.support import PRIOR, SUBSTEP_WIDTH, SUBSTEPS, map_extent, open_ground
from worldcast.player_state import live_position
from worldcast.player_state.extrapolation import JUMP_COLUMN, physics_prior_track
from worldcast.player_state.live_position import (
    BLOCK_LAG_U,
    CONTINUATION_FLOOR,
    PROBE_MIN_U,
    REACQUIRE_RATE_U,
    GuardedPosition,
    Walkable,
    wall_clearance,
)

#: The prior's running speed, u per latent frame (245 u/s).
RUN_U = 61.2
#: The estimate's fastest step in the tests that do not test it, u: no bound.
TOP_U = 1000.0
#: The prior's top speed over a latent frame, u: the farthest a player runs (245 u/s).
TOP_STEP_U = 245.0 / 4.0
#: A place estimate far outside the gate of a player near the origin, u: never followed.
FAR = np.array([10000.0, 0.0, 0.0])
#: Excerpts of the state model's head outputs logged per latent frame in live play on a GPU (place
#: head, motion head, the prior's step; the truth where the keys were the recording's; the stopped
#: step, the camera's before a continuation, and the estimate the run played with where a log has
#: them; ``a0``: the place head's first estimate).
REPLAY = json.loads(Path(__file__).with_name("live_position_replay.json").read_text())


def _guard(p0, a0, *, top: float = TOP_U, walkable: Walkable | None = None) -> GuardedPosition:
    """A position from ``p_0`` and ``A_0`` on open ground, bounded at ``top`` u per latent frame."""
    return GuardedPosition(p0, a0, top_step_u=top, walkable=walkable or open_ground())


@pytest.fixture
def unbudgeted(monkeypatch):
    """No lag budget: the estimate falls behind the camera as far as the view says, so that a rule
    of the step is seen over several latent frames of one block."""
    monkeypatch.setattr(live_position, "BLOCK_LAG_U", np.inf)


def _run(
    position: GuardedPosition,
    place: np.ndarray,
    displacement: np.ndarray,
    prior_steps: np.ndarray,
    camera_steps: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str | None], list[float | None]]:
    """``E_0 .. E_n`` (``E_0 = p_0``), and the status and blocked progress after each step; the
    camera's steps are the prior's unless given."""
    camera_steps = prior_steps if camera_steps is None else camera_steps
    estimates, statuses, blocked = [position.p0.copy()], [None], [None]
    for f in range(1, len(place)):
        estimates.append(position.step(place[f], displacement[f], prior_steps[f], camera_steps[f]))
        statuses.append(position.status)
        blocked.append(position.blocked_progress)
    return np.stack(estimates), statuses, blocked


def _runs(statuses: list[str | None]) -> list[tuple[str, int]]:
    """The statuses after each step as runs ``(status, count)``."""
    out: list[tuple[str, int]] = []
    for status in statuses[1:]:
        if out and out[-1][0] == status:
            out[-1] = (status, out[-1][1] + 1)
        else:
            out.append((status, 1))
    return out


def _client(scenario: str, frames: int = 80, place_noise: float = 20.0, motion_noise: float = 4.0):
    """A synthetic client of the E6 scenarios: GT ``[n, 3]``, the prior's steps, the place head and
    the motion head (each latent frame's noise drawn in that order).

    The keys run along x at :data:`RUN_U` (``burst``: the player stands). ``wall``: the player
    stops at x = 600 while the keys push on; ``stairs``: it climbs 32 u per latent frame over 2 s;
    ``burst``: the place head is 2,000 u off for 5 s; ``stuck``: it reports one cell 600 u ahead
    on the track for 10 s, ``stuck_beside`` 600 u beside it; ``a0``: its first estimate is 1,500 u
    off.
    """
    rng = np.random.default_rng(0)
    steps = np.zeros((frames, 3))
    steps[1:] = [0.0, 0.0, 0.0] if scenario == "burst" else [RUN_U, 0.0, 0.0]
    gt = np.zeros((frames, 3))
    for f in range(1, frames):
        gt[f] = gt[f - 1] + steps[f]
        if scenario == "wall":
            gt[f, 0] = min(gt[f, 0], 600.0)
        if scenario == "stairs" and 20 <= f < 28:
            gt[f, 2] += 32.0
    place, displacement = np.zeros((frames, 3)), np.zeros((frames, 3))
    for f in range(frames):
        place[f] = gt[f] + rng.normal(0.0, place_noise, 3) * [1.0, 1.0, 0.2]
        if f:
            displacement[f] = (gt[f] - gt[f - 1] + rng.normal(0.0, motion_noise, 3)) / 64.0
    if scenario == "burst":
        place[30:50] += [2000.0, 300.0, 0.0]
    if scenario == "stuck":
        place[30:70] = gt[30] + [600.0, 0.0, 0.0]
    if scenario == "stuck_beside":
        place[30:70] = gt[30] + [0.0, 600.0, 0.0]
    if scenario == "a0":
        place[0] += [1500.0, 0.0, 0.0]
    return gt, steps, place, displacement


def _jump_steps(jump_rows: range, latents: int) -> np.ndarray:
    """The physics prior's steps ``[latents + 1, 3]`` per latent frame of a player who holds no
    movement key and presses jump on the video frames ``jump_rows``."""
    rows = 1 + 4 * latents
    controls = torch.zeros(1, rows, SUBSTEPS, SUBSTEP_WIDTH + 1)
    controls[..., -1] = 1.0
    controls[0, list(jump_rows), :, JUMP_COLUMN] = 1.0
    track = physics_prior_track(controls, torch.zeros(1, rows), torch.zeros(1, 1, 3), prior=PRIOR)
    latent = track[0, ::4].double().numpy()
    return np.diff(latent, axis=0, prepend=latent[:1])


def _running(frames: int, speed: float = RUN_U) -> tuple[np.ndarray, np.ndarray]:
    """The prior's steps ``[frames, 3]`` of a player running along x, and its track from 0."""
    steps = np.tile([speed, 0.0, 0.0], (frames, 1))
    steps[0] = 0.0
    return steps, np.cumsum(steps, axis=0)


# ----------------------------------------------------------------------------- one frame by hand
def test_a_correction_is_gated_and_rate_limited():
    position = _guard(np.zeros(3), np.array([10.0, 0.0, 0.0]))  # A_0 10 u off: its bias
    run = np.array([64.0, 0.0, 0.0])
    one = np.array([1.0, 0.0, 0.0])
    # innovation (10, 0, 40): half of it, within 4 + 64/4 = 20 u (xy) and 4 + 64 * 3/4 = 52 u (z)
    assert position.step([84.0, 0.0, 40.0], one, run, run).tolist() == [69.0, 0.0, 20.0]
    # innovation (80, 0, -80): half is 40 u, limited to 20 u in xy; -40 u in height
    assert position.step([223.0, 0.0, -60.0], one, run, run).tolist() == [153.0, 0.0, -20.0]
    # 100 u in height is outside its gate: the prior's step alone in height, xy corrected on
    assert position.step([227.0, 0.0, 80.0], one, run, run).tolist() == [217.0, 0.0, -20.0]
    # 300 u in xy is another cell; the height, inside its own gate, is still corrected
    assert position.step([291.0, 300.0, 0.0], one, run, run).tolist() == [281.0, 0.0, -10.0]
    assert position.status == "guarded" and position.blocked_progress is None
    assert position.accepted_place is None
    # back inside the gate: the place term, 355 u less the 10 u bias of A_0, is accepted
    position.step([355.0, 0.0, -10.0], one, run, run)
    assert position.accepted_place.tolist() == [345.0, 0.0, -10.0]


def test_a_first_place_estimate_a_cell_away_is_replaced_by_the_start():
    run, one = np.array([64.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    far = _guard(np.zeros(3), np.array([300.0, 0.0, 0.0]))
    assert far.a0.tolist() == [0.0, 0.0, 0.0]
    assert far.step([74.0, 0.0, 0.0], one, run, run).tolist() == [69.0, 0.0, 0.0]
    near = _guard(np.zeros(3), np.array([200.0, 0.0, 0.0]))
    assert near.a0.tolist() == [200.0, 0.0, 0.0]
    # the place term is 104 - 200 = -96 u: inside the gate, corrected by at most 20 u
    assert near.step([104.0, 0.0, 0.0], one, run, run).tolist() == [44.0, 0.0, 0.0]
    # in height the gate is 96 u: a floor above is replaced; 250 u and 90 u, each inside, are kept
    assert _guard(np.zeros(3), [0.0, 0.0, 300.0]).a0.tolist() == [0.0, 0.0, 0.0]
    assert _guard(np.zeros(3), [250.0, 0.0, 90.0]).a0.tolist() == [250.0, 0.0, 90.0]


def test_the_step_is_as_far_as_the_view_followed_the_camera(unbudgeted):
    """Along the camera's 64 u step (the place head far, never followed) the motion head's
    displacement is a share of it: whole from 0.4 on, none below 0.15, linear in between. One
    latent frame of nothing is a glitch of the head: the step goes on; the second stops it, and
    the next block's camera keeps a quarter of its way at least."""
    position = _guard(np.zeros(3), np.zeros(3))
    push, short = np.array([64.0, 0.0, 0.0]), np.array([16.0, 0.0, 0.0])

    def step(motion: float, prior: np.ndarray = push) -> float:
        return position.step(FAR, [motion, 0.0, 0.0], prior, prior)[0]

    assert step(1.0) == 64.0 and position.blocked_progress is None
    assert step(0.0) == 128.0 and position.blocked_progress is None  # a glitch
    assert step(0.0) == 128.0 and position.blocked_progress == CONTINUATION_FLOOR  # stood still
    # a quarter of the step: (0.25 - 0.15) / (0.4 - 0.15) = 0.4 of it; the camera keeps 0.55
    assert step(0.25) == pytest.approx(128.0 + 0.4 * 64.0, abs=1e-9)
    assert position.blocked_progress == pytest.approx(0.25 + 0.75 * 0.4, abs=1e-12)
    assert step(0.5) == pytest.approx(128.0 + 0.4 * 64.0 + 64.0, abs=1e-9)  # moved: whole
    assert position.blocked_progress is None
    # a prior step shorter than 40 u is not read: the head reads a slow step less surely
    assert step(0.0, short) == pytest.approx(128.0 + 0.4 * 64.0 + 64.0 + 16.0, abs=1e-9)
    assert position.blocked_progress is None


def test_the_view_is_read_against_the_camera_not_against_the_prior(unbudgeted):
    """A wall stopped the camera at 48 u of the prior's 64 u step and the view moved 20 u: 0.42 of
    the camera's step, 0.31 of the prior's. The estimate steps with the camera, whole; a view that
    stands still holds it."""
    run, camera = np.array([64.0, 0.0, 0.0]), np.array([48.0, 0.0, 0.0])
    position = _guard(np.zeros(3), np.zeros(3))
    assert position.step(FAR, [20.0 / 64.0, 0.0, 0.0], run, camera).tolist() == [48.0, 0.0, 0.0]
    assert position.blocked_progress is None
    position = _guard(np.zeros(3), np.zeros(3))
    stood = [position.step(FAR, np.zeros(3), run, camera)[0] for _ in range(3)]
    assert stood == [48.0, 48.0, 48.0]  # the first latent frame of nothing is a glitch
    assert position.blocked_progress == CONTINUATION_FLOOR


def test_a_view_that_moved_across_the_camera_step_counts_as_moved():
    """The motion head's displacement is read by its length, in any direction: in live play it
    does not follow the keys' direction, and a view sliding along a wall moved. A displacement of
    64 u across the camera's 64 u step is a step moved: the estimate steps along the camera."""
    position = _guard(np.zeros(3), np.zeros(3))
    run = np.array([64.0, 0.0, 0.0])
    for _ in range(3):
        assert position.step(FAR, [0.0, 1.0, 0.0], run, run)[1] == 0.0
    assert position.blocked_progress is None and position.step(
        FAR, [0.0, -1.0, 0.0], run, run
    ).tolist() == [256.0, 0.0, 0.0]


def test_a_camera_at_its_floor_is_read_and_a_slower_one_is_not():
    """The view stands still along a camera slowed to the floor of a continuation, a quarter of
    the prior's 61.2 u step: read, the estimate holds. Along a step of 9 u, less than what the
    floor of a 40 u step moves, it is not read."""
    run = np.array([RUN_U, 0.0, 0.0])
    assert PROBE_MIN_U == 10.0
    position = _guard(np.zeros(3), np.zeros(3))
    slowed, crawl = CONTINUATION_FLOOR * run, np.array([9.0, 0.0, 0.0])
    held = [position.step(FAR, np.zeros(3), run, slowed)[0] for _ in range(3)]
    assert held == pytest.approx([15.3, 15.3, 15.3]) and position.blocked_progress == 0.25
    assert position.step(FAR, np.zeros(3), run, crawl)[0] == pytest.approx(24.3)
    assert position.blocked_progress is None


def test_the_estimate_falls_at_most_two_body_widths_behind_the_camera_within_a_block():
    """The camera runs 61.2 u per latent frame and the view stands still from the second: a block
    of four ends the estimate 180.8 u on of the camera's 244.8 (64 u behind), as the first latent
    frame is a glitch's and the third and fourth catch up what is more than 64 u. The block after,
    its camera slowed to the floor by the continuation, moves 15.3 u per latent frame and the
    estimate holds: the hop at its end is the 61.2 u of the camera, under the budget."""
    run = np.array([RUN_U, 0.0, 0.0])
    position = _guard(np.zeros(3), np.zeros(3))
    first = [position.step(FAR, np.zeros(3), run, run)[0] for _ in range(4)]
    assert first == pytest.approx([61.2, 61.2, 119.6, 180.8]) and BLOCK_LAG_U == 64.0
    later = [position.step(FAR, np.zeros(3), run, CONTINUATION_FLOOR * run)[0] for _ in range(8)]
    assert later == pytest.approx([180.8] * 8)


def test_a_wall_the_view_stood_still_at_holds_the_estimate_whatever_the_wall_leaves(unbudgeted):
    """The camera stands at a wall (steps of 0) and the motion head sees a fifth of the prior's
    61.2 u step: the estimate is bit-still. Where the wall leaves half of the step, the estimate
    steps with the camera's half until the head's fifth of the prior's step, 0.4 of the camera's
    step, whole from then on."""
    zero, run = np.zeros(3), np.array([RUN_U, 0.0, 0.0])
    position = _guard(zero, zero)
    estimates = [position.step(FAR, 0.2 * run / 64.0, run, zero) for _ in range(40)]
    assert (np.stack(estimates) == 0.0).all()
    position = _guard(zero, zero)
    estimates = [position.step(FAR, 0.2 * run / 64.0, run, 0.5 * run)[0] for _ in range(3)]
    assert estimates == pytest.approx([30.6, 61.2, 91.8])  # 0.4 of the camera's step: whole
    assert position.blocked_progress is None


def test_a_camera_held_at_a_wall_by_a_stop_stands_still_to_the_bit():
    """The camera stands at a wall the depth head saw (a step of a millionth of a unit, the rounding
    of its stop) while the keys push on: the estimate does not move by that, whatever the heads
    say, as long as the view does not move on past the stop."""
    position = _guard(np.zeros(3), np.zeros(3))
    run, stopped = np.array([RUN_U, 0.0, 0.0]), np.array([1e-6, 0.0, 0.0])
    estimates = [position.step(FAR, np.zeros(3), run, stopped) for _ in range(20)]
    assert (np.stack(estimates) == 0.0).all()


def test_the_view_moving_through_a_wall_the_depth_head_stopped_the_camera_at_is_followed(
    unbudgeted,
):
    """The depth head stopped the camera at 10 u of the prior's 64 u step: the estimate steps 10 u
    until the motion head says the view moved on for two latent frames, from half of the prior's
    step; it then steps as far as the head says, and the wall is a wall that is not there."""
    position = _guard(np.zeros(3), np.zeros(3))
    run, stopped = np.array([64.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])

    def step(motion: float) -> float:
        return position.step(FAR, [motion, 0.0, 0.0], run, stopped)[0]

    assert step(0.0) == 10.0 and not position.moved_past_wall  # nothing, then the stopped step
    assert step(1.0) == 20.0 and not position.moved_past_wall  # one latent frame is no verdict
    assert step(1.0) == 84.0 and position.moved_past_wall  # moved on twice: the prior's step
    assert step(0.5) == 116.0 and position.moved_past_wall  # half of it
    assert step(0.45) == 126.0 and not position.moved_past_wall  # less than half: the stopped step


def test_the_motion_head_stops_the_estimate_while_the_place_head_is_not_followed(unbudgeted):
    """The place head jumps a cell from frame to frame 1,000 u beside a running player: rejected on
    every latent frame, it is not followed from the 12th on. The motion head sees the view stand
    still from the 13th: the estimate steps with the prior until the second latent frame of it,
    and stands where it is."""
    frames = 24
    steps, track = _running(frames)
    place = track + [0.0, 1000.0, 0.0]
    place[1::2, 1] += 600.0
    motion = steps / 64.0
    motion[13:] = 0.0
    position = _guard(track[0], track[0])
    estimates, statuses, blocked = _run(position, place, motion, steps)
    assert _runs(statuses) == [("guarded", 11), ("degraded", 12)]
    assert np.array_equal(estimates[:14], track[:14])
    assert (estimates[14:] == track[13]).all()
    assert blocked == [None] * 14 + [CONTINUATION_FLOOR] * (frames - 14)
    assert position.accepted_place is None  # not followed


# ------------------------------------------------------------------------------ jumps and height
def test_jumps_in_place_land_at_their_take_off_height():
    """Ten jumps of the physics prior, one per block: each flight ends 10.15625 u higher than it
    started, the estimate where it took off."""
    steps = _jump_steps(range(1, 161, 16), latents=40)
    assert steps[1:5, 2].tolist() == [41.5625, -8.4375, -22.96875, 0.0]
    p0 = np.array([100.0, -200.0, 30.0])
    place = p0 + np.random.default_rng(0).normal(0.0, 15.0, (41, 3)) * [1.0, 1.0, 0.2]
    estimates, _, _ = _run(_guard(p0, p0), place, np.zeros((41, 3)), steps)
    assert (estimates[:, :2] == p0[:2]).all()
    assert estimates[:, 2].tolist() == [30.0, 71.5625, 63.125, 40.15625] * 10 + [30.0]


def test_hops_in_a_row_return_to_the_take_off_height_at_the_reacquire_rate():
    """Jump held for 2 s: four flights in a row, 40.625 u higher at the last landing; back on the
    ground the estimate returns 16 u per latent frame."""
    steps = _jump_steps(range(1, 33), latents=16)
    p0 = np.array([100.0, -200.0, 30.0])
    place = p0 + np.random.default_rng(0).normal(0.0, 15.0, (17, 3)) * [1.0, 1.0, 0.2]
    estimates, _, _ = _run(_guard(p0, p0), place, np.zeros((17, 3)), steps)
    assert estimates[10:, 2].tolist() == [70.625, 54.625, 38.625, 30.0, 30.0, 30.0, 30.0]


def test_a_player_who_stops_stands_at_the_height_it_ran_to():
    position = _guard(np.zeros(3), np.zeros(3))
    run, one, zero = np.array([64.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.zeros(3)
    # the place head 40 u higher on a slope: half of the innovation each latent frame
    assert position.step([64.0, 0.0, 40.0], one, run, run).tolist() == [64.0, 0.0, 20.0]
    assert position.step([128.0, 0.0, 40.0], one, run, run).tolist() == [128.0, 0.0, 30.0]
    for _ in range(3):
        assert position.step([128.0, 0.0, 40.0], zero, zero, zero).tolist() == [128.0, 0.0, 30.0]


def test_a_height_offset_a_place_head_that_agreed_holds_is_followed_to_where_it_is():
    """The place head agrees with a running player for 20 latent frames and then reads it 200 u
    higher (stairs the prior knows nothing of): outside the height gate, rejected, not followed
    after 12 latent frames; persistently apart after 14 of 16, it is followed at 16 u per latent
    frame until it is inside the gate, then corrected at the height rate. xy never moves from the
    track."""
    steps, track = _running(60, speed=64.0)
    place = track.copy()
    place[20:, 2] += 200.0
    estimates, statuses, _ = _run(_guard(track[0], track[0]), place, steps / 64.0, steps)
    assert (estimates[:, :2] == track[:, :2]).all()
    assert estimates[34:43, 2].tolist() == [0.0] + [16.0 * k for k in range(1, 8)] + [156.0]
    assert estimates[-1, 2] == pytest.approx(200.0, abs=0.01)
    assert _runs(statuses) == [
        ("guarded", 30),
        ("degraded", 4),
        ("reacquiring", 7),
        ("guarded", 18),
    ]


def test_stairs_are_followed_in_height():
    gt, steps, place, displacement = _client("stairs")
    estimates, _, _ = _run(_guard(gt[0], place[0]), place, displacement, steps)
    assert np.abs(estimates - gt)[:, 2].max() <= 40.0


# ---------------------------------------------------------------------- following the place head
def test_the_status_says_whether_the_place_head_is_followed():
    """The place head jumps between two far cells for 16 latent frames: never consistent, it is
    not followed after 12 rejections, and followed again once fewer than 12 of the last 16 were
    rejected."""
    frames = 40
    steps, track = _running(frames)
    jumping = track.copy()
    jumping[10:26, 1] += np.where(np.arange(16) % 2 == 0, 1000.0, -1000.0)
    estimates, statuses, _ = _run(_guard(track[0], track[0]), jumping, steps / 64.0, steps)
    assert _runs(statuses) == [("guarded", 20), ("degraded", 9), ("guarded", 10)]
    assert np.abs(estimates - track).max() == 0.0


def test_a_place_track_that_drifts_away_and_never_agreed_is_never_followed():
    """The place head follows the player 600 u beside it and drifts 15 u per latent frame further
    (a consistent track, a cell away, that agreed with the estimate on 11 latent frames only): it
    is rejected and never followed; the estimate is the dead reckoning to the bit."""
    frames = 60
    steps, track = _running(frames)
    place = track + [0.0, 600.0, 0.0] + np.arange(frames)[:, None] * [0.0, 15.0, 0.0]
    estimates, statuses, _ = _run(_guard(track[0], track[0]), place, steps / 64.0, steps)
    assert _runs(statuses) == [("guarded", 11), ("degraded", 48)]
    assert np.array_equal(estimates, track)


@pytest.mark.parametrize(("agreed", "followed"), [(15, False), (16, True), (30, True)])
def test_a_place_head_is_followed_to_where_it_is_after_agreeing_on_16_latent_frames(
    agreed, followed
):
    """The place head agrees with a running player on ``agreed`` latent frames and then reads it
    600 u beside the track, moving with it (a lost head holds exactly this level). It is followed
    only if it agreed lately on 16 of the last 64: then it is followed at 16 u per latent frame
    until it is inside the gate and the estimate ends on it; else the estimate is the dead
    reckoning to the bit, however long the head holds its level."""
    frames = 100
    steps, track = _running(frames)
    place = track.copy()
    place[agreed + 1 :] += [0.0, 600.0, 0.0]
    estimates, statuses, _ = _run(_guard(track[0], track[0]), place, steps / 64.0, steps)
    if not followed:
        assert np.array_equal(estimates, track) and "reacquiring" not in statuses
        return
    assert _runs(statuses)[1:3] == [("degraded", 5), ("reacquiring", 22)]
    assert np.linalg.norm((estimates - place)[-1, :2]) < 1e-3
    pull = np.linalg.norm((np.diff(estimates, axis=0) - steps[1:])[:, :2], axis=1)
    assert pull.max() == pytest.approx(4.0 + 0.25 * RUN_U) and pull.max() > REACQUIRE_RATE_U


def test_a_place_track_that_agreed_and_then_holds_its_own_is_followed_to_where_it_is():
    """The prior runs at a tenth of the player's speed for 2 s while the motion head and the place
    head see the player run: the estimate falls 421 u behind, outside the gate, after the place
    head agreed with it for 20 latent frames. The place head's own track holds that level for
    seconds, so it is followed, 16 u per latent frame at most, and the long disagreement is the
    prior's, and does not last: the estimate ends within the place head's noise of the truth."""
    frames = 80
    rng = np.random.default_rng(0)
    steps, track = _running(frames)
    prior = steps.copy()
    prior[20:28] *= 0.1
    place = track + rng.normal(0.0, 30.0, (frames, 3)) * [1.0, 1.0, 0.0]
    estimates, statuses, _ = _run(_guard(track[0], track[0]), place, steps / 64.0, prior)
    error = np.linalg.norm((estimates - track)[:, :2], axis=1)
    assert round(float(error[27]), 1) == 420.6 and round(float(error[-1]), 1) == 31.7
    assert _runs(statuses)[:3] == [("guarded", 34), ("degraded", 4), ("reacquiring", 7)]
    pull = np.linalg.norm((np.diff(estimates, axis=0) - prior[1:])[:, :2], axis=1)
    reacquired = [s == "reacquiring" for s in statuses[1:]]
    assert pull[reacquired].max() <= REACQUIRE_RATE_U + 1e-9


def test_a_place_head_that_holds_its_place_while_the_player_runs_is_never_followed():
    """The place head agrees with a running player on 30 latent frames and then sticks to one
    cell 600 u beside the track for 40 more: it moved less than a quarter of what the camera did
    over 16 latent frames. It is not followed, whatever it agreed on: the estimate is the dead
    reckoning."""
    gt, steps, place, displacement = _client("stuck_beside")
    estimates, statuses, _ = _run(_guard(gt[0], place[0]), place, displacement, steps)
    assert "reacquiring" not in statuses
    drag = np.linalg.norm((np.diff(estimates, axis=0) - steps[1:])[29:69, :2], axis=1)
    assert drag.max() < 1e-9


def test_a_place_track_held_while_standing_is_not_followed_when_the_player_runs_off():
    """The place head is stuck 1,000 u away while the player stands for 6 s, and stays there as it
    runs off: it never agreed. Never followed, and the estimate is the dead reckoning."""
    frames = 40
    steps = np.zeros((frames, 3))
    steps[25:] = [RUN_U, 0.0, 0.0]
    track = np.cumsum(steps, axis=0)
    place = np.tile(track[0] + [1000.0, 0.0, 0.0], (frames, 1))
    estimates, statuses, _ = _run(_guard(track[0], track[0]), place, steps / 64.0, steps)
    assert _runs(statuses) == [("guarded", 11), ("degraded", 28)]
    assert np.array_equal(estimates, track)


@pytest.mark.parametrize("leg", [1, 2, 4])
def test_a_place_head_stuck_while_the_player_strafes_to_and_fro_is_never_followed(leg):
    """The place head is right for 10 latent frames, then stuck 500 u away while the player strafes
    left and right in legs of ``leg`` latent frames for 20 s: it agreed on too few latent frames
    to be followed. Never followed, never re-acquired: the estimate is the truth."""
    frames = 80
    steps = np.zeros((frames, 3))
    steps[1:, 1] = RUN_U * np.where((np.arange(1, frames) // leg) % 2 == 0, -1.0, 1.0)
    truth = np.cumsum(steps, axis=0)
    place = truth.copy()
    place[10:] = truth[10] + [400.0, 300.0, 0.0]
    estimates, statuses, _ = _run(_guard(truth[0], truth[0]), place, steps / 64.0, steps)
    assert _runs(statuses) == [("guarded", 20), ("degraded", 59)]
    assert np.array_equal(estimates, truth)


def test_an_estimate_the_prior_leads_out_of_the_map_follows_a_place_head_that_never_agreed(
    monkeypatch,
):
    """The keys push on along a strip of the map that ends at x = 768 u and the camera stops at the
    edge, while the place head holds a steady track 300 u along it, 450 u from the estimate, and
    never agreed with it: the estimate at the edge is wrong by construction, so the head is
    followed, 16 u per latent frame, and the estimate leaves the edge; were the edge no evidence,
    it would stay."""
    strip = Walkable(np.array([[i, 0] for i in range(-2, 2)]))
    frames = 90
    steps, track = _running(frames)
    place = np.tile([300.0, 0.0, 0.0], (frames, 1))

    def run() -> tuple[np.ndarray, list[str | None]]:
        position = _guard(track[0], place[0], walkable=strip)
        estimates, statuses, stopped = [track[0]], [None], False
        for f in range(1, frames):
            camera = steps[f].copy()
            camera[0] = min(
                camera[0], strip.clearance(estimates[-1][:2], np.array([1.0, 0.0]), RUN_U)
            )
            stopped = stopped or estimates[-1][0] > 740.0
            if stopped:
                camera[:2] = 0.0
            estimates.append(position.step(place[f], camera / 64.0, steps[f], camera))
            statuses.append(position.status)
        return np.stack(estimates), statuses

    estimates, statuses = run()
    assert estimates[:, 0].max() == pytest.approx(755.65) and "reacquiring" in statuses
    assert estimates[-1, 0] == pytest.approx(351.65)
    monkeypatch.setattr(live_position, "EDGE_NEAR", 99)
    held, statuses = run()
    assert held[-30:, 0].tolist() == pytest.approx([755.65] * 30) and "reacquiring" not in statuses


# ------------------------------------------------------------------------------- the invariants
def _distance_to_segment(point: np.ndarray, end: np.ndarray) -> float:
    """The distance of ``point`` from the segment from the origin to ``end``."""
    along = np.clip(point @ end / max(end @ end, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(point - along * end))


def test_the_estimate_never_jumps():
    """The estimate leaves the segment of the latent frame's step (nothing to the prior's) by at
    most ``max(16, 4 + |camera_xy|/4)`` in xy and ``max(16, 4 + 3/4 |camera_xy|)`` in height,
    whatever the heads say and wherever a wall stops the step: the step is the camera's or the
    prior's, scaled by what the motion head saw. It moves no farther than the top step and stays
    on the map's extent."""
    rng = np.random.default_rng(0)
    ground = open_ground(8)  # 2,048 u each way: a long run leaves it
    seen, blocked_frames, moved_on_frames, edge_frames = set(), 0, 0, 0
    for _ in range(4):
        truth, base, offset = np.zeros(3), np.zeros(3), np.zeros(3)
        position = _guard(truth, truth + rng.normal(0.0, 20.0, 3), top=55.0, walkable=ground)
        regime, left = "stand", 0
        for _ in range(400):
            if left == 0:
                regime = rng.choice(["stand", "run", "wall", "offset", "outliers", "drift"])
                left = int(rng.integers(4, 40))
                heading = rng.uniform(0.0, 2 * np.pi)
                offset = rng.uniform(300.0, 2000.0) * np.array(
                    [np.cos(heading), np.sin(heading), 0.0]
                )
            left -= 1
            speed = 0.0 if regime == "stand" else rng.uniform(10.0, 70.0)
            prior = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
            prior[2] = rng.choice([0.0, 30.0, -30.0], p=[0.8, 0.1, 0.1])
            moved = np.zeros(3) if regime == "wall" else prior
            truth = truth + moved
            if regime == "drift":
                offset = offset + rng.normal(0.0, 20.0, 3) * [1.0, 1.0, 0.0]
            place = truth + rng.normal(0.0, 25.0, 3) * [1.0, 1.0, 0.3]
            if regime in ("offset", "drift"):
                place = place + offset
            if regime == "outliers":
                place = place + rng.uniform(-2000.0, 2000.0, 3) * [1.0, 1.0, 0.1]
            motion = (moved + rng.normal(0.0, 8.0, 3)) / 64.0
            share = 1.0 if rng.random() < 0.5 else rng.uniform(0.0, 1.0)
            camera = prior * [share, share, 1.0]
            estimate = position.step(place, motion, prior, camera)

            # the step is on the segment from nothing to the prior's step, in xy
            length = np.hypot(*camera[:2])
            reach = prior[:2] * (np.hypot(*prior[:2]) >= 1.0)
            move = estimate - base
            bound = max(REACQUIRE_RATE_U, 4.0 + 0.25 * length)
            assert _distance_to_segment(move[:2], reach) <= bound + 1e-9
            assert abs(move[2] - prior[2]) <= max(REACQUIRE_RATE_U, 4.0 + 0.75 * length) + 1e-9
            assert np.hypot(*move[:2]) <= 55.0 + 1e-9 and ground.contains(estimate[:2])
            edge_frames += not ground.contains(base[:2] + prior[:2])
            base = estimate
            seen.add(position.status)
            blocked_frames += position.blocked_progress is not None
            moved_on_frames += position.moved_past_wall
    assert seen == {"guarded", "reacquiring", "degraded"}
    assert blocked_frames > 0 and moved_on_frames > 0 and edge_frames > 0


def test_no_player_is_moved_faster_than_it_runs():
    """The place head 60 u ahead of a player running at 61.2 u per latent frame is inside the gate:
    half of the innovation, at most 4 + 61.2/4 = 19.3 u, would carry the estimate 80.5 u on. It
    moves the top step (245 u/s over a latent frame) and no more."""
    steps, track = _running(12)
    place = track + [60.0, 0.0, 0.0]
    place[0] = track[0]
    estimates, _, _ = _run(_guard(track[0], place[0], top=TOP_STEP_U), place, steps / 64.0, steps)
    assert estimates[:, 0].tolist() == pytest.approx([TOP_STEP_U * f for f in range(12)])
    free, _, _ = _run(_guard(track[0], place[0]), place, steps / 64.0, steps)
    assert free[1, 0] == pytest.approx(RUN_U + 19.3, abs=0.005)


def test_a_walkable_extent_is_the_columns_of_the_cells_and_a_margin():
    """Three columns of 256 u; a margin of one column around them."""
    walkable = Walkable(np.array([[0, 0], [1, 0], [1, 1]]))
    inside = [(128.0, 128.0), (-200.0, -200.0), (700.0, 700.0), (700.0, -200.0)]
    outside = [(-300.0, 0.0), (800.0, 0.0), (0.0, 800.0), (-300.0, 700.0)]
    assert [walkable.contains(np.array(p)) for p in inside] == [True] * 4
    assert [walkable.contains(np.array(p)) for p in outside] == [False] * 4


def test_the_clearance_is_how_far_a_player_walks_before_the_edge():
    walkable = Walkable(np.array([[0, 0], [1, 0], [1, 1]]))
    ahead, back = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    # the edge at x = 768: from 128 u the first sample outside is 640 u on, the one before it 624
    assert walkable.clearance(np.array([128.0, 128.0]), ahead, 1000.0) == 624.0
    assert walkable.clearance(np.array([128.0, 128.0]), ahead, 500.0) == np.inf
    # from outside it is free: the cell table says nothing of where a player outside may go
    assert walkable.clearance(np.array([900.0, 0.0]), ahead, 100.0) == np.inf
    assert walkable.clearance(np.array([900.0, 0.0]), back, 400.0) == np.inf


def test_the_estimate_stops_at_the_edge_of_the_walkable_extent():
    """A player runs along +x on a strip that ends at x = 768 u (the cells up to 512 u and a margin
    of one column): the estimate stops within a sample step of the edge and stands there, with a
    place head 200 u ahead of it that would carry it on."""
    strip = Walkable(np.array([[i, 0] for i in range(-2, 2)]))
    steps, track = _running(30)
    place = track + [200.0, 0.0, 0.0]
    place[0] = track[0]
    estimates, _, _ = _run(_guard(track[0], place[0], walkable=strip), place, steps / 64.0, steps)
    assert estimates[:, 0].max() <= 768.0 and strip.contains(estimates[-1, :2])
    assert (estimates[-10:] == estimates[-1]).all() and estimates[-1, 0] > 740.0


def test_a_standing_player_is_never_corrected():
    """Standing, the place head 60 u away (40 u higher) on every latent frame: the estimate does
    not move, in xy or in height; it is corrected once the player moves."""
    position = _guard(np.zeros(3), np.zeros(3))
    zero, place = np.zeros(3), np.array([60.0, 0.0, 40.0])
    assert {tuple(position.step(place, zero, zero, zero)) for _ in range(20)} == {(0.0, 0.0, 0.0)}
    run = np.array([4.0, 0.0, 0.0])
    assert position.step(place + run, run / 64.0, run, run).tolist() == [9.0, 0.0, 7.0]


def test_a_standing_player_stays_bit_still():
    rng = np.random.default_rng(0)
    p0 = np.array([100.0, -200.0, 30.0])
    place = p0 + rng.normal(0.0, 30.0, (41, 3)) * [1.0, 1.0, 0.2]
    place[0] = p0
    zero = np.zeros((41, 3))
    estimates, _, _ = _run(_guard(p0, place[0]), place, zero, zero)
    assert (estimates == p0).all()


# ------------------------------------------------------------------- the place head's failures
def test_a_burst_of_the_place_head_does_not_move_a_standing_player():
    gt, steps, place, displacement = _client("burst")
    estimates, statuses, _ = _run(_guard(gt[0], place[0]), place, displacement, steps)
    assert (estimates == gt[0]).all()
    # 2,000 u away and steady for 5 s: followed once persistent, but the player stands
    assert _runs(statuses) == [("guarded", 40), ("degraded", 5), ("guarded", 34)]


def test_a_confidently_stuck_place_head_ahead_on_the_track_drags_at_most_a_rate_limit():
    """Ahead on the track it comes inside the gate, where a correction is at most 4 u plus a
    quarter of the step."""
    gt, steps, place, displacement = _client("stuck")
    estimates, _, _ = _run(_guard(gt[0], place[0]), place, displacement, steps)
    drag = np.linalg.norm((np.diff(estimates, axis=0) - steps[1:])[29:69, :2], axis=1)
    assert drag.max() == pytest.approx(4.0 + 0.25 * RUN_U)


def test_a_first_place_estimate_far_off_is_not_followed():
    gt, steps, place, displacement = _client("a0")
    estimates, _, _ = _run(_guard(gt[0], place[0]), place, displacement, steps)
    assert np.linalg.norm((estimates - gt)[:, :2], axis=1).max() <= 40.0


def test_a_wall_the_view_stands_still_at_stops_the_estimate(unbudgeted):
    gt, steps, place, displacement = _client("wall", frames=40, place_noise=5.0, motion_noise=2.0)
    estimates, _, blocked = _run(_guard(gt[0], place[0]), place, displacement, steps)
    # latent frame 11 is the first the view stands still in: the second, 12, stops the estimate
    assert [f for f, progress in enumerate(blocked) if progress is not None] == list(range(12, 40))
    error = np.linalg.norm((estimates - gt)[:, :2], axis=1)
    assert round(float(error[12:].max()), 1) == 28.6 and error[-1] <= 5.0  # the prior: 1,787 u past


def test_a_wall_at_the_e6_noise_is_found_on_every_latent_frame_after_the_first_two(unbudgeted):
    """At place noise 20 u and motion noise 4 u the motion head's displacement is a tenth of the
    step at most: no latent frame of the 80 misses the wall, and the place head pulls the estimate
    back from the one step it took beyond it."""
    gt, steps, place, displacement = _client("wall")
    estimates, _, blocked = _run(_guard(gt[0], place[0]), place, displacement, steps)
    assert [f for f, progress in enumerate(blocked) if progress is None] == list(range(12))
    error = np.linalg.norm((estimates - gt)[:, :2], axis=1)
    assert error[12] == pytest.approx(29.46, abs=0.005)
    assert error[12:].max() <= 35.0 and error[-1] <= 10.0  # the prior alone: 4,234 u past it


def test_a_drifting_motion_head_cannot_lock_the_estimate_out():
    """The gate's prediction is the camera's, never the motion head's: a motion head 30 degrees off
    and three place estimates 2,000 u away leave the estimate on the track."""
    rng = np.random.default_rng(0)
    frames = 121
    heading = np.deg2rad(3.0 * np.arange(frames))
    steps = RUN_U * np.stack([np.cos(heading), np.sin(heading), np.zeros(frames)], axis=-1)
    steps[0] = 0.0
    gt = np.cumsum(steps, axis=0)
    turned = heading + np.deg2rad(30.0)
    motion = RUN_U * np.stack([np.cos(turned), np.sin(turned), np.zeros(frames)], axis=-1) / 64.0
    place = gt + rng.normal(0.0, 20.0, (frames, 3)) * [1.0, 1.0, 0.2]
    place[0] = gt[0]
    place[[17, 37, 51]] += [2000.0, 0.0, 0.0]
    # the prior runs 10 % fast: on its own it drifts up to 234 u off the circle
    estimates, _, _ = _run(_guard(gt[0], place[0]), place, motion, 1.1 * steps)
    assert np.linalg.norm((estimates - gt)[:, :2], axis=1).max() <= 64.0


# ------------------------------------------------------------------- replayed GPU head outputs
def _replay(
    name: str, camera_steps: np.ndarray | None = None
) -> tuple[dict, np.ndarray, list[str | None], list[float | None]]:
    """An excerpt of :data:`REPLAY` stepped from its first latent frame on, as the live loop steps
    it; the camera's steps are the prior's (no wall in the way) unless given."""
    excerpt = REPLAY[name]
    place, displacement, steps = (
        np.array(excerpt[key]) for key in ("place", "displacement", "prior_step")
    )
    position = _guard(np.array(excerpt["p0"]), place[0])
    return excerpt, *_run(position, place, displacement, steps, camera_steps)


def test_a_standing_player_never_moves_on_the_logged_heads():
    """50 of 60 latent frames standing while the place head is mostly rejected: the estimate does
    not move."""
    excerpt, estimates, statuses, _ = _replay("standing")
    steps = np.array(excerpt["prior_step"])
    standing = [
        f for f in range(1, len(steps)) if np.hypot(*steps[f, :2]) < 1.0 and not steps[f, 2]
    ]
    assert len(standing) == 50
    assert all(np.array_equal(estimates[f], estimates[f - 1]) for f in standing)
    assert _runs(statuses) == [("guarded", 21), ("degraded", 39)]


def test_a_place_head_a_cell_off_that_agreed_on_11_latent_frames_is_not_followed():
    """12 s on the recording's keys while the place head reports a track a cell or more off the
    truth that moves with the player: it agreed on 11 latent frames only, and it is lost, not
    right. Every place estimate is rejected, none followed, and the estimate is the dead
    reckoning, 229.3 u from the truth where the place head's track is 600 u and more."""
    excerpt, estimates, statuses, _ = _replay("lost")
    truth = np.array(excerpt["truth"])
    assert _runs(statuses) == [("guarded", 11), ("degraded", 37)]
    error = np.linalg.norm((estimates - truth)[:, :2], axis=1)
    place = np.array(excerpt["place"]) - np.array(excerpt["place"])[0] + np.array(excerpt["p0"])
    head = np.linalg.norm((place - truth)[:, :2], axis=1)
    assert round(float(error[-1]), 1) == 229.3 and head[-10:].min() > 600.0


def test_a_stall_of_the_view_stops_the_estimate_with_the_place_head_lost(unbudgeted):
    """A push forward for 26 latent frames, then strafes, while the place head is lost and not
    followed from latent frame 12: the motion head sees the view stand still over latent frames 19
    and 20. The first is a glitch the estimate steps through, the second stops it for that frame;
    elsewhere it steps with the prior."""
    excerpt, estimates, statuses, blocked = _replay("wall")
    steps = np.array(excerpt["prior_step"])
    assert _runs(statuses) == [("guarded", 11), ("degraded", 49)]
    assert [f for f, progress in enumerate(blocked) if progress is not None] == [20]
    assert (estimates[20] == estimates[19]).all()
    expected = steps.copy()
    standing = np.hypot(steps[:, 0], steps[:, 1]) < 1.0
    expected[standing, :2] = 0.0  # a standing step keeps only its height
    expected[20, :2] = 0.0
    np.testing.assert_allclose(np.diff(estimates[11:], axis=0), expected[12:], rtol=0.0, atol=1e-9)


def _path_length(track: np.ndarray) -> float:
    """The length of a track in xy, u."""
    return float(np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1).sum())


def _live(name: str, **patches) -> tuple[dict, np.ndarray, list[str | None], list[float | None]]:
    """An excerpt of a live play stepped from its first latent frame on the Dust2 extent, the
    estimate bounded at the top step as the live loop bounds it."""
    excerpt = REPLAY[name]
    place, displacement, steps, stopped = (
        np.array(excerpt[key]) for key in ("place", "displacement", "prior_step", "stopped_step")
    )
    position = _guard(
        np.array(excerpt["p0"]),
        np.array(excerpt["a0"]),
        top=TOP_STEP_U,
        walkable=map_extent("de_dust2"),
    )
    return excerpt, *_run(position, place, displacement, steps, stopped)


def test_the_keys_run_on_at_a_wall_the_view_stood_still_at_and_the_estimate_stays_on_the_map():
    """Latent frames 326-356 of a live play: the keys run at full speed (1,812 u), the motion head
    sees the view move 3 to 35 u per latent frame, and the place head is lost. The estimate the
    run played with ran the whole way and left the map's extent on 12 of the 31 latent frames; this
    one steps as far as the view allowed within the lag budget, 1,070 u, and stays on the map."""
    excerpt, estimates, statuses, blocked = _live("lost_view_still")
    steps, logged = np.array(excerpt["prior_step"]), np.array(excerpt["logged_estimate"])
    ground = map_extent("de_dust2")
    assert round(_path_length(steps.cumsum(0))) == 1812
    assert round(_path_length(logged)) == 1812 and round(_path_length(estimates)) == 1070
    assert [ground.contains(p) for p in logged].count(False) == 12
    assert all(ground.contains(p[:2]) for p in estimates)
    assert sum(progress is not None for progress in blocked) == 27
    assert _runs(statuses) == [("guarded", 11), ("degraded", 19)]  # the place head: not followed


def test_a_place_head_that_agreed_for_7_s_and_holds_a_track_away_is_followed_without_a_deadlock(
    monkeypatch,
):
    """Latent frames 116-190 of a live play: the place head agrees with the estimate for 28 latent
    frames and then holds a steady track away from it, 1,900 to 2,300 u, for the rest. The
    estimate the run played with stayed outside the gate for all of them (never accepted, not
    followed), a deadlock. The track is persistently apart and agreed lately, so it is followed,
    at most 16 u per latent frame, for 25 latent frames: the estimate ends 374 u nearer to it than
    it does where the head is never followed. No more: the keys run on along a heading the track
    does not follow."""
    excerpt, estimates, statuses, _ = _live("lost_place_track")
    place, logged = np.array(excerpt["place"]), np.array(excerpt["logged_estimate"])
    track = (place - np.array(excerpt["a0"]) + np.array(excerpt["p0"]))[:, :2]
    assert round(float(np.linalg.norm(logged[-1] - track[-1]))) == 2297
    assert (np.linalg.norm(logged - track, axis=1)[29:] > 256.0).all()
    assert _runs(statuses) == [
        ("guarded", 28),
        ("degraded", 11),
        ("reacquiring", 25),
        ("degraded", 10),
    ]
    now = np.linalg.norm(estimates[:, :2] - track, axis=1)
    assert round(float(now[-1])) == 1903
    monkeypatch.setattr(live_position, "PERSISTENT_NEAR", 99)
    _, held, held_statuses, _ = _live("lost_place_track")
    assert round(float(np.linalg.norm(held[-1, :2] - track[-1]))) == 2277
    assert "reacquiring" not in held_statuses
    assert "reacquiring" not in held_statuses


# --------------------------------------------------------------------------------- the wall ahead
def _wall(distance: float, *, rise: tuple[float, float] = (0.0, 100.0), width: float = 200.0):
    """Surface points of a wall ``distance`` u ahead along +x, from ``rise[0]`` to ``rise[1]`` u
    above the feet, ``width`` u wide, 4 u apart."""
    y, z = np.meshgrid(np.arange(-width / 2, width / 2 + 1, 4.0), np.arange(*rise, 4.0))
    return np.stack([np.full(y.size, distance), y.ravel(), z.ravel()], axis=-1)


def _ramp(degrees: float, length: float = 400.0) -> np.ndarray:
    """Surface points of a ramp rising at ``degrees`` along +x from the feet, 4 u apart."""
    x, y = np.meshgrid(np.arange(0.0, length, 4.0), np.arange(-100.0, 101.0, 4.0))
    return np.stack([x.ravel(), y.ravel(), x.ravel() * np.tan(np.deg2rad(degrees))], axis=-1)


def _stairs(rise: float, run: float = 16.0, first: float = 50.0) -> np.ndarray:
    """Surface points of stairs up along +x from ``first`` u, ``rise`` u a step, 4 u apart."""
    y = np.arange(-100.0, 101.0, 4.0)
    points = []
    for k in range(12):
        x0 = first + run * k
        for z in np.arange(rise * k, rise * (k + 1), 4.0):  # the riser
            points.append(np.stack([np.full(y.size, x0), y, np.full(y.size, z)], axis=-1))
        for x in np.arange(x0, x0 + run, 4.0):  # the tread
            points.append(np.stack([np.full(y.size, x), y, np.full(y.size, rise * (k + 1))], -1))
    return np.concatenate(points)


def test_a_wall_ahead_leaves_its_distance_less_half_the_hull():
    forward = np.array([1.0, 0.0])
    assert wall_clearance(_wall(100.0), forward) == 84.0
    assert wall_clearance(_wall(10.0), forward) == 0.0
    # met at 45 degrees: the hull's side reaches it (200 - 16 sqrt 2) / sqrt 2 - 16 = 109.4 u on,
    # the nearest point of the 4 u grid 111.3 u
    diagonal = np.array([1.0, 1.0]) / np.sqrt(2.0)
    assert wall_clearance(_wall(100.0, width=600.0), diagonal) == pytest.approx(111.28, abs=0.005)
    # the nearest of two walls
    assert wall_clearance(np.concatenate([_wall(300.0), _wall(150.0)]), forward) == 134.0


@pytest.mark.parametrize(
    "surface",
    [
        _wall(-100.0),  # behind
        _wall(100.0) + [0.0, 300.0, 0.0],  # beside the way
        _wall(100.0, rise=(0.0, 18.0)),  # a step the player walks up
        _wall(100.0, rise=(80.0, 200.0)),  # overhead
        _wall(100.0, rise=(18.0, 48.0)),  # rises less than 32 u in the hull's height
        _ramp(30.0),
        _ramp(45.0),
        _stairs(16.0),
        _stairs(8.0),
        np.concatenate([_ramp(0.0), _wall(100.0, rise=(0.0, 16.0))]),
        np.zeros((0, 3)),
    ],
    ids=[
        "behind",
        "beside",
        "step",
        "overhead",
        "low",
        "ramp-30",
        "ramp-45",
        "stairs-16",
        "stairs-8",
        "floor",
        "nothing",
    ],
)
def test_what_is_no_wall(surface):
    assert wall_clearance(surface, np.array([1.0, 0.0])) == float("inf")


def test_a_wall_seen_through_noisy_depth_is_found_where_most_of_it_is():
    """The depth head off by up to a tenth on most patches and by up to 60 % on some: the wall,
    104 u on, is found about a tenth early."""
    rng = np.random.default_rng(0)
    surface = _wall(120.0)
    error = np.where(
        rng.random(len(surface)) < 0.85,
        rng.uniform(0.9, 1.1, len(surface)),
        rng.uniform(0.6, 1.6, len(surface)),
    )
    surface[:, 0] *= error
    assert round(wall_clearance(surface, np.array([1.0, 0.0])), 2) == 91.98
