"""The player box of the entity criterion and the tick lookup behind it, also against the boxes
stored for a view of the paper's ten-second windows."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from worldcast.data.game import BODY_HEIGHT, HFOV_DEGREES, VFOV_DEGREES
from worldcast.data.latents import SOURCE_FRAMES_PER_VIDEO_FRAME
from worldcast.data.recordings import TickTable
from worldcast.engine.evaluation.boxes import (
    BOX_MARGIN,
    BOX_WIDTH_RATIO,
    clip_box,
    clipped_side,
    frame_states,
    player_box,
    table_boxes,
    tick_rows,
    tick_states,
    view_boxes,
)
from worldcast.engine.evaluation.entity import MIN_SIDE

#: A view of the paper's ten-second windows with its stored boxes and the tick rows its frames read
#: (``tick_view``: per player ``[t, x, y, z, yaw, pitch, is_alive]``, the tables' float32 values).
FIXTURE = Path(__file__).parent / "fixtures" / "entity" / "views.json.gz"
WIDTH, HEIGHT = 672, 384
TAN_H = math.tan(math.radians(HFOV_DEGREES / 2.0))
TAN_V = math.tan(math.radians(VFOV_DEGREES / 2.0))


def state(x: float, y: float, z: float = 0.0, yaw: float = 0.0, pitch: float = 0.0, alive=1.0):
    return np.array([x, y, z, yaw, pitch, alive], dtype=np.float64)


def ahead(depth: float, u: float) -> np.ndarray:
    """A player ``depth`` u ahead of an observer at the origin looking along +x, its head at the
    horizontal pixel ``u``."""
    return state(depth, -((u / WIDTH) - 0.5) * 2.0 * TAN_H * depth)


def test_a_standing_player_ahead():
    box = player_box(state(0, 0), state(100, 0), WIDTH, HEIGHT)
    v_feet = (0.5 + (64.0 / 100.0) / TAN_V * 0.5) * HEIGHT  # the eye is 64 u above the origin
    v_head = (0.5 - ((BODY_HEIGHT - 64.0) / 100.0) / TAN_V * 0.5) * HEIGHT
    half = BOX_WIDTH_RATIO * (v_feet - v_head) / 2.0
    expected = (WIDTH / 2 - half, v_head, WIDTH / 2 + half, v_feet, 100.0)
    assert box == pytest.approx(expected, rel=1e-9)
    assert VFOV_DEGREES == 73.74  # the rounded value, not derived from the horizontal one


def test_the_box_is_centred_on_the_head():
    observer = state(0, 0, pitch=30.0)  # looking down: feet and head lie at different depths
    target = state(150, -60)
    box = player_box(observer, target, WIDTH, HEIGHT)
    pitch = math.radians(30.0)
    forward = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])

    def u_of(point: np.ndarray) -> float:
        delta = point - np.array([0.0, 0.0, 64.0])
        return ((60.0 / (delta @ forward)) / TAN_H * 0.5 + 0.5) * WIDTH

    u_head = u_of(np.array([150.0, -60.0, BODY_HEIGHT]))
    assert (box[0] + box[2]) / 2 == pytest.approx(u_head, rel=1e-9)
    assert abs(u_head - u_of(np.array([150.0, -60.0, 0.0]))) > 10


def test_players_at_or_behind_the_camera_and_dead_players_have_no_box():
    assert player_box(state(0, 0), state(-100, 0), WIDTH, HEIGHT) is None
    assert player_box(state(0, 0), state(1.0, 0), WIDTH, HEIGHT) is None  # depth 1 u
    assert player_box(state(0, 0, alive=0.5), state(100, 0), WIDTH, HEIGHT) is None
    assert player_box(state(0, 0), state(100, 0, alive=0.0), WIDTH, HEIGHT) is None


def test_a_box_outside_the_frame_is_kept_unclipped_within_the_margin():
    centre = player_box(state(0, 0), ahead(100.0, WIDTH / 2), WIDTH, HEIGHT)
    half = (centre[2] - centre[0]) / 2.0  # the same at every horizontal position at this depth
    edge = WIDTH * (1.0 + BOX_MARGIN)
    inside = player_box(state(0, 0), ahead(100.0, edge - 0.5 + half), WIDTH, HEIGHT)
    assert inside is not None and inside[0] > WIDTH and inside[0] < edge  # not clipped
    assert player_box(state(0, 0), ahead(100.0, edge + 0.5 + half), WIDTH, HEIGHT) is None
    left = -WIDTH * BOX_MARGIN
    assert player_box(state(0, 0), ahead(100.0, left + 0.5 - half), WIDTH, HEIGHT) is not None
    assert player_box(state(0, 0), ahead(100.0, left - 0.5 - half), WIDTH, HEIGHT) is None


def test_tick_rows_take_the_last_tick_and_know_a_player_for_one_and_a_half_ticks():
    times = np.arange(10) / 64.0
    rows, known = tick_rows(times, np.array([0, 1, 4, 5, 6]))
    assert rows.tolist() == [0, 2, 8, 9, 9]  # source frame 1 is tick 2 exactly
    assert known.tolist() == [True, True, True, True, False]  # 10/64 s <= 9/64 + 1.5/64 < 12/64
    rows, known = tick_rows(times + 0.5, np.array([0]))
    assert rows.tolist() == [0] and known.tolist() == [True]  # no lower bound


def test_frame_states_are_dead_where_the_player_is_not_known():
    times = np.arange(4) / 64.0
    ticks = np.stack([state(float(k), 0) for k in range(4)])
    out = frame_states(times, ticks, np.array([0, 1, 3]))  # 6/64 s > 3/64 + 1.5/64
    assert out[:, 0].tolist() == [0.0, 2.0, 3.0]
    assert out[:, 5].tolist() == [1.0, 1.0, 0.0]


def test_view_boxes_skip_the_observer_and_dead_players():
    observer = np.stack([state(0, 0), state(0, 0, alive=0.0)])
    players = {
        2: np.stack([state(120, 10), state(120, 10)]),
        0: np.stack([state(100, 0), state(100, 0)]),
        1: observer,
        3: np.stack([state(100, 5, alive=0.0)] * 2),
    }
    boxes = view_boxes(observer, players, 1, WIDTH, HEIGHT)
    assert [b[0] for b in boxes[0]] == [0, 2]
    assert boxes[0][0][1:] == list(player_box(observer[0], players[0][0], WIDTH, HEIGHT))
    assert boxes[1] == []  # the observer is dead


def test_the_boxes_from_the_ticks_are_the_stored_boxes():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        item = json.load(handle)["tick_view"]
    window = item["window"]
    width, height, observer = window["width"], window["height"], window["obs_slot"]
    count = window["n_frames"]
    source = window["start_frame"] + SOURCE_FRAMES_PER_VIDEO_FRAME * np.arange(count)
    states = {}
    for slot, rows in item["ticks"].items():
        table = np.asarray(rows, dtype=np.float32).astype(np.float64)
        states[int(slot)] = frame_states(table[:, 0], table[:, 1:], source)
    boxes = view_boxes(states[observer], states, observer, width, height)
    assert [[b[0] for b in frame] for frame in boxes] == [
        [b[0] for b in frame] for frame in window["boxes"]
    ]
    for frame, stored in zip(boxes, window["boxes"]):
        for box, expected in zip(frame, stored):
            assert box[1:] == pytest.approx(expected[1:], rel=0, abs=1e-9)
            large = clipped_side(box[1:5], width, height) >= MIN_SIDE
            assert large == (clipped_side(expected[1:5], width, height) >= MIN_SIDE)
    sides = [clipped_side(b[1:5], width, height) for frame in boxes for b in frame]
    assert min(sides) < MIN_SIDE <= max(sides)  # both sides of the threshold are tested


def test_table_boxes_have_no_boxes_past_the_table():
    states = np.stack([np.stack([state(0, 0)] * 3), np.stack([state(100, 0)] * 3)])
    boxes = table_boxes(states, 1, 4, 0, WIDTH, HEIGHT)
    assert [len(frame) for frame in boxes] == [1, 1, 0, 0]


def test_clipping():
    assert clip_box([-10, 20, 30, 500], WIDTH, HEIGHT) == (0.0, 20.0, 30.0, float(HEIGHT))
    assert clip_box([700, 20, 800, 50], WIDTH, HEIGHT) is None
    assert clipped_side([-10, 0, 30, 30], WIDTH, HEIGHT) == pytest.approx(30.0)
    assert clipped_side([700, 20, 800, 50], WIDTH, HEIGHT) == 0.0


def test_tick_states_of_a_tick_table():
    floats = np.array([0.5, 1.25], dtype=np.float32).astype(np.float64)
    table = TickTable(
        t=np.array([0.0, 1 / 64]),
        x=floats,
        y=floats + 1,
        z=floats + 2,
        yaw=floats + 3,
        pitch=floats + 4,
        is_alive=np.array([True, False]),
        active=[None, None],
        delta_pitch=np.zeros(2),
        delta_yaw=np.zeros(2),
        input_weapon=[None, None],
        team_num=np.zeros(2, dtype=np.int64),
    )
    times, states = tick_states(table)
    assert times.dtype == states.dtype == np.float64
    assert states.tolist() == [[0.5, 1.5, 2.5, 3.5, 4.5, 1.0], [1.25, 2.25, 3.25, 4.25, 5.25, 0.0]]
