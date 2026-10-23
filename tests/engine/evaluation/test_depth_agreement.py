"""Depth agreement on known geometry, and the paper's cells when ``WORLDCAST_PAPER_REFS`` is set.

Cameras, reprojection, the float32 tolerance, the overlap gate, the rows of a unit and their kinds,
chance, the co-visibility mask and the segment ratios. The paper's rows are read from
``$WORLDCAST_PAPER_REFS/depth_agreement/<table>/<condition>.npz`` (``rows``: ``[n, 10]`` int32,
:data:`ROW_COLUMNS`, units numbered from 0): ``scene_agree`` (Table 19's 32 rounds: ``worldcast``,
``scene_state_off``, ``trained_without_scene_state``, ``only_own_earlier_blocks``,
``least_covering_memory``, ``recorded``), ``closed_loop`` (its 31 closed-loop rounds:
``worldcast``, ``scene_state_off``, ``recorded``) and ``view_agree`` (Table 10's 256 windows:
``worldcast``, ``frame_concatenation``, ``solaris``, ``recorded``).
"""

import dataclasses
import math
import os
from pathlib import Path

import numpy as np
import pytest

from tests.data.support import tick_table
from worldcast.engine.evaluation.depth_agreement import (
    BETWEEN_CLIENTS,
    CHANCE,
    CHANCE_LAG,
    GATE_DEPTHS_U,
    HIT_U,
    OWN_NEXT_FRAME,
    ROW_COLUMNS,
    SCENE_SPLIT_SECONDS,
    VIEW_SPLIT_SECONDS,
    Cameras,
    block_log_depth,
    chance_frames,
    compare_depths,
    covisibility_mask,
    frame_rotations,
    latent_depth,
    overlap_gate,
    pair_rows,
    recorded_cameras,
    recorded_depth,
    recorded_ticks,
    recorded_windows,
    segment_ratios,
)

GRID = (24, 42)
#: The paper's half-angle tangents: HFOV 106.26 degrees, 16:9.
TAN_H = math.tan(math.radians(106.26 / 2))
TAN_V = TAN_H * 9 / 16
REFS = os.environ.get("WORLDCAST_PAPER_REFS")


def cameras(yaws, positions=None, alive=None, eye_height: float = 0.0) -> Cameras:
    """Level cameras at ``positions`` (default the origin) with the eye ``eye_height`` above."""
    yaws = np.asarray(yaws, dtype=np.float64)
    n = len(yaws)
    positions = np.zeros((n, 3)) if positions is None else np.asarray(positions, np.float64)
    return Cameras(
        positions=positions,
        eye_heights=np.full(n, eye_height),
        rotations=frame_rotations(yaws, np.zeros(n)),
        alive=np.ones(n, bool) if alive is None else np.asarray(alive, bool),
    )


def exact_rotation(yaw: float, pitch: float) -> np.ndarray:
    """``world = camera @ R``: rows forward, right, up; pitch > 0 looks down."""
    y, p = math.radians(yaw), math.radians(pitch)
    forward = [math.cos(y) * math.cos(p), math.sin(y) * math.cos(p), -math.sin(p)]
    right = [math.sin(y), -math.cos(y), 0.0]
    up = [math.cos(y) * math.sin(p), math.sin(y) * math.sin(p), math.cos(p)]
    return np.array([forward, right, up])


def landed_rays(eye_b: np.ndarray, yaw_b: float) -> np.ndarray:
    """``[4, 1008]``: which rays of a level camera at the origin, yaw 0, placed at each gate depth,
    land strictly inside a level camera at ``eye_b`` (float64, exact rotations)."""
    x = (np.arange(GRID[1]) + 0.5) / GRID[1] * 2 - 1
    y = -((np.arange(GRID[0]) + 0.5) / GRID[0] * 2 - 1)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rays = np.stack([np.ones_like(xx), xx * TAN_H, yy * TAN_V], -1).reshape(-1, 3)
    out = []
    for depth in GATE_DEPTHS_U:
        local = (depth * rays @ exact_rotation(0.0, 0.0) - eye_b) @ exact_rotation(yaw_b, 0.0).T
        forward = local[:, 0]
        out.append(
            (forward > 1)
            & (np.abs(local[:, 1] / forward) < TAN_H)
            & (np.abs(local[:, 2] / forward) < TAN_V)
        )
    return np.array(out)


def test_rotations_are_fitted_to_float32_rays():
    """``world = camera @ R`` with forward, right and up (pitch > 0 looks down), fitted to the
    float32 rays: within 1e-7 of the exact rotation, and not equal to it."""
    yaws, pitches = np.array([0.0, 30.0, -120.0, 75.5]), np.array([0.0, 20.0, -45.0, 88.0])
    fitted = frame_rotations(yaws, pitches)
    exact = np.stack([exact_rotation(y, p) for y, p in zip(yaws, pitches)])
    assert fitted.shape == (4, 3, 3) and fitted.dtype == np.float64
    assert np.abs(fitted - exact).max() < 1e-7
    assert not np.array_equal(fitted, exact)
    np.testing.assert_allclose(fitted[1, 0], [0.8138, 0.4698, -0.3420], atol=1e-4)


def test_recorded_cameras_read_the_last_tick_and_the_crouch():
    """Latent frame k reads the tick at or before source frame 8 k (a 32 fps recording: k / 4 s);
    the eye is 64 u up, 46 u crouched; dead after death and 0.5 s after the last tick."""
    table = tick_table(4.0, dies_at=2.0)
    active = [["forward", "duck"] if 1.0 <= t < 1.5 else ["forward"] for t in table.t]
    cams = recorded_cameras(dataclasses.replace(table, active=active), 0, 20, 32.0)
    k = np.arange(20)
    row = np.minimum(k * 16, len(table) - 1)  # 64 Hz ticks, k / 4 s
    np.testing.assert_array_equal(recorded_ticks(table, 0, 20, 32.0), row)
    np.testing.assert_array_equal(cams.positions[:, 0], table.x[row])
    assert cams.positions.dtype == np.float64
    np.testing.assert_array_equal(cams.eye_heights, np.where((k >= 4) & (k < 6), 46.0, 64.0))
    np.testing.assert_array_equal(cams.alive, k < 8)
    np.testing.assert_array_equal(cams.rotations, frame_rotations(table.yaw[row], table.pitch[row]))
    np.testing.assert_array_equal(cams.eyes[:, 2], cams.positions[:, 2] + cams.eye_heights)

    later = recorded_cameras(tick_table(4.0), 80, 6, 32.0)  # frames 80 .. 120: 2.5 .. 3.75 s
    np.testing.assert_array_equal(later.positions[:, 0], tick_table(4.0).x[160 + 16 * k[:6]])
    gap = recorded_cameras(tick_table(1.0), 0, 8, 32.0)  # last tick at 63 / 64 s
    np.testing.assert_array_equal(gap.alive, np.arange(8) / 4 <= 63 / 64 + 0.5)


def test_latent_depth_is_the_last_video_frame_in_float32():
    log_depth = np.zeros((3, 4, *GRID), np.float16)
    log_depth[:, -1] = np.float16(5.0)
    log_depth[1, -1, 0, :3] = (np.float16(-1.0), np.float16(9.0), np.float16(np.nan))
    depth = latent_depth(log_depth)
    assert depth.shape == (3, *GRID) and depth.dtype == np.float32
    assert depth[0, 5, 5] == np.exp(np.float32(5.0))
    assert depth[1, 0, 0] == 1.0 and depth[1, 0, 1] == 4096.0 and np.isnan(depth[1, 0, 2])


def test_the_recorded_depth_takes_the_last_frame_of_the_middle_latent():
    """Windows at source frames 0, 80, 161 (odd: never on the latent grid) and 322 (its frames
    sit on the third video frame of a latent), given in the order 322, 161, 80, 0: each latent frame
    of the round takes the last video frame of a latent nearest a window's middle, else any; on a
    tie the window given first, not the earliest."""
    windows, sources = {}, {}
    for w, start in enumerate((322, 161, 80, 0)):
        log_depth = np.zeros((41, 4, *GRID), np.float16)
        for latent in range(41):
            for column in range(4):  # a log depth of its own per (window, latent, column)
                value = np.float16(1 + w + latent / 64 + column / 256)
                log_depth[latent, column] = value
                sources[float(np.exp(np.float32(value)))] = (start, latent, column)
        windows[start] = log_depth
    depth = recorded_depth(windows, 101)
    assert depth.shape == (101, *GRID) and depth.dtype == np.float32

    def source(k):
        return sources[float(depth[k, 0, 0])]

    assert source(10) == (0, 10, 3)  # window 80 has frame 80 as its first frame only
    assert source(20) == (0, 20, 3)
    assert source(25) == (80, 15, 3)  # a tie: latent 25 of window 0, latent 15 of window 80
    assert source(40) == (80, 30, 3)
    assert source(41) == (80, 31, 3)
    assert source(61) == (322, 21, 2)  # only window 322 reaches frame 488
    assert np.isnan(depth[100]).all()
    assert np.isnan(depth[0]).all()  # frame 0 is no window's video frame after its first


def test_the_recorded_windows_run_from_0_then_from_80_inside_the_recording():
    """A window spans 321 source frames: in 965 source frames the last fits at 644, in 964 at
    563; a recording shorter than a window has none."""
    assert recorded_windows(965) == [0, 161, 322, 483, 644, 80, 241, 402, 563]
    assert recorded_windows(964) == [0, 161, 322, 483, 80, 241, 402, 563]
    assert recorded_windows(321) == [0] and recorded_windows(320) == []


def test_block_log_depth_runs_the_first_frame_and_each_block_alone():
    calls = []

    def depth_fn(latents: np.ndarray) -> np.ndarray:
        calls.append(len(latents))
        return latents[:, :4] + len(latents)

    latents = np.random.default_rng(0).standard_normal((9, 48, 2, 3)).astype(np.float32)
    depth = block_log_depth(depth_fn, latents)
    assert calls == [1, 4, 4]
    np.testing.assert_array_equal(depth[0], latents[0, :4] + 1)
    np.testing.assert_array_equal(depth[1:], latents[1:, :4] + 4)
    with pytest.raises(ValueError, match="expected 1 \\+ 4 b"):
        block_log_depth(depth_fn, latents[:8])


def test_a_view_agrees_with_itself_where_it_has_a_surface():
    """The same camera and depth: every pixel reached and agreeing, except where the depth is at or
    beyond the hit limit (not reprojected) or behind the near plane."""
    cams = cameras([20.0], eye_height=64.0)
    depth = np.full((1, *GRID), 300.0, np.float32)
    depth[0, 3, :5] = HIT_U
    depth[0, 4, :2] = 0.5
    counts = compare_depths(depth, cams.eyes, cams.rotations, depth, cams.eyes, cams.rotations)
    assert counts.shape == (1, 5) and counts[0].tolist() == [1001, 1001, 1001, 0, 0]
    behind = cameras([200.0], eye_height=64.0)  # looking away
    counts = compare_depths(depth, cams.eyes, cams.rotations, depth, behind.eyes, behind.rotations)
    assert counts[0].tolist() == [0, 0, 0, 0, 0]


def test_the_tolerance_is_float32_against_a_float64_gap():
    """``max(24 u, 5% d)`` with ``d`` and 5% in float32: a gap between the float32 and the float64
    tolerance agrees, as in the paper's rows; one past the float32 tolerance does not."""
    target = np.float32(901.48)  # where the float32 and float64 tolerances differ by 2.3e-6 u
    tol32 = float(np.float32(0.05) * target)
    tol64 = 0.05 * float(target)
    assert tol32 > tol64 + 2e-6
    depth = np.full((1, *GRID), target, np.float32)
    behind = cameras([0.0])
    for gap, agrees in (((tol32 + tol64) / 2, True), (tol32 + 1e-6, False)):
        ahead = cameras([0.0], positions=[[gap, 0.0, 0.0]])  # A's points are ``gap`` farther
        counts = compare_depths(
            depth, ahead.eyes, ahead.rotations, depth, behind.eyes, behind.rotations
        )[0]
        reached, agree, wide, farther, nearer = counts.tolist()
        assert reached == wide > 800 and farther == 0
        assert (agree, nearer) == ((reached, 0) if agrees else (0, reached))
    assert ROW_COLUMNS[5:] == ("reached", "agree", "agree_wide", "farther", "nearer")


def test_the_overlap_gate_counts_rays_of_the_first_view_at_any_gate_depth():
    """At least 50 of A's rays land in B's frustum at one of 128, 256, 512 and 1024 u: here 66 land
    at some depth, at most 40 at any one; a sweep of B's yaw matches an exact-rotation count."""
    eye, yaw = np.array([1150.0, 900.0, 0.0]), 150.0
    landed = landed_rays(eye, yaw)
    assert landed.any(0).sum() == 66 and landed.sum(1).max() == 40
    a, b = cameras([0.0]), cameras([yaw], positions=[eye])
    assert overlap_gate(a.eyes, a.rotations, b.eyes, b.rotations).tolist() == [True]

    yaws = np.arange(80.0, 140.0, 0.5)
    b = cameras(yaws, positions=np.tile([0.0, -200.0, 0.0], (len(yaws), 1)))
    a = cameras(np.zeros(len(yaws)))
    gate = overlap_gate(a.eyes, a.rotations, b.eyes, b.rotations)
    count = np.array([landed_rays(np.array([0.0, -200.0, 0.0]), y).any(0).sum() for y in yaws])
    clear = np.abs(count - 49.5) > 3
    assert ((count > 0) & (count < 47)).any() and ((count > 52) & (count < 1008)).any()
    np.testing.assert_array_equal(gate[clear], count[clear] >= 50)


def test_chance_is_10_s_away_in_a_round_and_the_most_distant_instant_in_a_window():
    k = np.arange(1, 100)
    round_ = chance_frames(k, 100)
    np.testing.assert_array_equal(round_[:59], k[:59] + CHANCE_LAG)  # 10 s later
    np.testing.assert_array_equal(round_[59:], k[59:] - CHANCE_LAG)  # 10 s earlier at the end
    short = chance_frames(np.arange(1, 60), 60)  # a round of 15 s: some instants have no chance
    np.testing.assert_array_equal(short[:19], np.arange(41, 60))
    assert (short[19:40] == -1).all()
    np.testing.assert_array_equal(short[40:], np.arange(1, 20))
    window = chance_frames(np.arange(1, 41), 41, most_distant=True)
    np.testing.assert_array_equal(window, np.where(np.arange(1, 41) <= 20, 40, 1))
    assert chance_frames(np.array([21]), 42, most_distant=True).tolist() == [41]  # a tie: later


def test_the_rows_of_a_unit():
    """Ordered pairs of views (between clients, then their chance rows), then each view's own next
    latent frame; dead frames are not scored; the gate cameras select the frames, the cameras
    reproject; chance rows without a latent frame 10 s away are dropped in a round."""
    n = 50
    rng = np.random.default_rng(1)
    depths = [rng.uniform(150.0, 700.0, (n, *GRID)).astype(np.float32) for _ in range(3)]
    alive = np.ones(n, bool)
    alive[7] = False
    gate = [cameras(np.full(n, yaw)) for yaw in (0.0, 10.0, -15.0)]
    gate[1] = dataclasses.replace(gate[1], alive=alive)
    rows = pair_rows(depths, gate, unit=4)
    assert rows.dtype == np.int32 and rows.shape[1] == len(ROW_COLUMNS)
    assert (rows[:, 0] == 4).all()
    blocks = [tuple(r) for r in rows[:, 1:3]]
    order = list(dict.fromkeys(blocks))
    assert order == [(0, 1), (0, 2), (0, 0), (1, 0), (1, 2), (1, 1), (2, 0), (2, 1), (2, 2)]
    between = rows[(rows[:, 1] == 0) & (rows[:, 2] == 1) & (rows[:, 4] == BETWEEN_CLIENTS)]
    np.testing.assert_array_equal(between[:, 3], np.delete(np.arange(1, n), 6))
    chance = rows[(rows[:, 1] == 0) & (rows[:, 2] == 1) & (rows[:, 4] == CHANCE)]
    kept = between[:, 3][chance_frames(between[:, 3], n) >= 1]
    np.testing.assert_array_equal(chance[:, 3], kept)  # k 10 .. 40 dropped
    assert len(kept) < len(between)
    own = rows[(rows[:, 1] == 1) & (rows[:, 4] == OWN_NEXT_FRAME)]
    no_6_7 = np.delete(np.arange(1, n - 1), [5, 6])  # 6: its next frame is dead; 7: dead
    np.testing.assert_array_equal(own[:, 3], no_6_7)

    k, shifted = int(chance[-1, 3]), int(chance_frames(chance[-1:, 3], n)[0])
    assert (k, shifted) == (49, 9)
    a, b = gate[0], gate[1]
    at_b = (depths[1][[k]], b.eyes[[k]], b.rotations[[k]])
    one = compare_depths(depths[0][[shifted]], a.eyes[[k]], a.rotations[[k]], *at_b)
    np.testing.assert_array_equal(chance[-1, 5:], one[0])

    window = pair_rows(depths, gate, most_distant_chance=True)
    assert (window[:, 4] == CHANCE).sum() == (window[:, 4] == BETWEEN_CLIENTS).sum()

    published = list(gate)  # view 0 publishes another position, and its alive is not read
    published[0] = dataclasses.replace(
        gate[0], positions=gate[0].positions + [5.0, 0.0, 0.0], alive=np.zeros(n, bool)
    )
    moved = pair_rows(depths, published, gate=gate, unit=4)
    np.testing.assert_array_equal(moved[:, :5], rows[:, :5])
    changed = np.any(moved[:, 5:] != rows[:, 5:], 1)
    assert changed.any() and not changed[(rows[:, 1] != 0) & (rows[:, 2] != 0)].any()


def test_the_covisibility_mask_keeps_pairs_the_recorded_video_agrees_on():
    keys = [(0, 0, 1, 5, 0), (0, 0, 1, 5, 2), (0, 0, 1, 6, 0), (0, 0, 1, 6, 2), (0, 0, 0, 5, 1)]
    keys += [(1, 1, 0, 5, 0), (1, 1, 0, 5, 2)]
    rows = np.array([[*key, 10, 3, 4, 1, 1] for key in keys], np.int32)
    recorded = rows.copy()
    recorded[:, 6] = [2, 0, 0, 7, 9, 1, 0]  # agree: frame 6 of (0, 0, 1) has none between clients
    mask = covisibility_mask(rows, recorded)
    assert mask.tolist() == [True, True, False, False, False, True, True]
    with pytest.raises(ValueError, match="other keys"):
        covisibility_mask(rows, recorded[:-1])
    other = recorded.copy()
    other[0, 3] = 4
    with pytest.raises(ValueError, match="other keys"):
        covisibility_mask(rows, other)


def test_segment_ratios_are_ratios_of_sums_per_unit():
    """Latent frame k is at k / 4 s: 30 s is latent frame 120 of a round, 5 s latent frame 20 of a
    window; a unit without a reached pixel in a segment is NaN."""
    rows = np.array(
        [
            [0, 0, 1, 119, 0, 10, 5, 0, 0, 0],
            [0, 1, 0, 119, 0, 30, 5, 0, 0, 0],
            [0, 0, 1, 120, 0, 10, 1, 0, 0, 0],
            [0, 0, 1, 120, 2, 10, 9, 0, 0, 0],
            [2, 0, 1, 19, 0, 4, 4, 0, 0, 0],
            [2, 0, 1, 20, 0, 4, 1, 0, 0, 0],
        ],
        np.int32,
    )
    scene = segment_ratios(rows, BETWEEN_CLIENTS, 3, SCENE_SPLIT_SECONDS)
    assert scene.shape == (3, 2)
    np.testing.assert_array_equal(scene[0], [10 / 40, 1 / 10])
    assert np.isnan(scene[1]).all()
    assert scene[2, 0] == 5 / 8 and np.isnan(scene[2, 1])
    view = segment_ratios(rows, BETWEEN_CLIENTS, 3, VIEW_SPLIT_SECONDS)
    np.testing.assert_array_equal(view[2], [1.0, 0.25])
    assert segment_ratios(rows, CHANCE, 3, SCENE_SPLIT_SECONDS)[0, 1] == 0.9


# ------------------------------------------------------------------------- the paper's cells


def paper_rows(table: str, condition: str) -> np.ndarray:
    path = Path(REFS or "") / "depth_agreement" / table / f"{condition}.npz"
    if not REFS or not path.is_file():
        pytest.skip(f"the paper's rows are not there ({path}); set WORLDCAST_PAPER_REFS")
    with np.load(path) as z:
        return z["rows"]


def cells(ratios: np.ndarray, sign: str = "") -> list[str]:
    return [f"{value:{sign}.3f}" for value in np.nanmean(ratios, 0)]


def test_table19_with_gt_states():
    """Table 19, upper part: levels over the co-visible pairs, first 30 s / after 30 s."""
    recorded = paper_rows("scene_agree", "recorded")
    levels = {
        "worldcast": ["0.264", "0.206"],
        "scene_state_off": ["0.229", "0.133"],
        "trained_without_scene_state": ["0.259", "0.135"],
        "only_own_earlier_blocks": ["0.231", "0.147"],
        "least_covering_memory": ["0.225", "0.119"],
        "recorded": ["0.411", "0.365"],
    }
    between, own = {}, {}
    for condition, want in levels.items():
        rows = paper_rows("scene_agree", condition)
        masked = rows[covisibility_mask(rows, recorded)]
        between[condition] = segment_ratios(masked, BETWEEN_CLIENTS, 32, SCENE_SPLIT_SECONDS)
        own[condition] = segment_ratios(rows, OWN_NEXT_FRAME, 32, SCENE_SPLIT_SECONDS)
        assert cells(between[condition]) == want, condition
        if condition == "worldcast":
            chance = segment_ratios(masked, CHANCE, 32, SCENE_SPLIT_SECONDS)
            assert cells(chance) == ["0.113", "0.101"]
    gain = between["worldcast"] - between["scene_state_off"]
    assert cells(gain, "+") == ["+0.035", "+0.073"]
    assert cells(own["worldcast"] - own["scene_state_off"], "+") == ["+0.004", "+0.000"]


def test_table19_in_the_closed_loop_and_table4a():
    """Table 19, lower part, and the Depth row of Table 4a (scene state off / on)."""
    recorded = paper_rows("closed_loop", "recorded")
    between, own = {}, {}
    for condition in ("worldcast", "scene_state_off"):
        rows = paper_rows("closed_loop", condition)
        masked = rows[covisibility_mask(rows, recorded)]
        between[condition] = segment_ratios(masked, BETWEEN_CLIENTS, 31, SCENE_SPLIT_SECONDS)
        own[condition] = segment_ratios(rows, OWN_NEXT_FRAME, 31, SCENE_SPLIT_SECONDS)
    assert cells(between["worldcast"]) == ["0.247", "0.120"]
    assert cells(between["scene_state_off"]) == ["0.243", "0.092"]
    gain = between["worldcast"] - between["scene_state_off"]
    assert cells(gain, "+") == ["+0.004", "+0.028"]
    assert cells(own["worldcast"] - own["scene_state_off"], "+") == ["-0.001", "-0.002"]


def test_table10():
    """Table 10: the two views of each ten-second window, first 5 s / last 5 s, no mask."""
    levels = {
        "worldcast": ["0.257", "0.207"],
        "frame_concatenation": ["0.232", "0.170"],
        "solaris": ["0.233", "0.179"],
        "recorded": ["0.317", "0.309"],
    }
    between = {}
    for condition, want in levels.items():
        rows = paper_rows("view_agree", condition)
        between[condition] = segment_ratios(rows, BETWEEN_CLIENTS, 256, VIEW_SPLIT_SECONDS)
        assert cells(between[condition]) == want, condition
        if condition == "worldcast":
            chance = segment_ratios(rows, CHANCE, 256, VIEW_SPLIT_SECONDS)
            assert cells(chance) == ["0.117", "0.116"]
    for baseline, want in (
        ("frame_concatenation", ["+0.025", "+0.037"]),
        ("solaris", ["+0.023", "+0.028"]),
    ):
        assert cells(between["worldcast"] - between[baseline], "+") == want
