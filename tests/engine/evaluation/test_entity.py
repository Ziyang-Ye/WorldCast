"""The entity criterion: the matching, the recorded video's own hits, Unmatched, the folded gate,
the partner scope and the summaries, on synthetic views and on three views of the paper's
ten-second windows (their stored detections, the labels given to the generator and the paper's
counts)."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from worldcast.engine.evaluation.boxes import table_boxes
from worldcast.engine.evaluation.entity import (
    ViewCounts,
    assign_one_to_one,
    gt_visibility_gate,
    pool_by_crowding,
    recording_hits,
    score_view,
    summarize_entity,
    visibility_gate,
    window_of,
)

#: Two views with GT states and one with predicted states, with the paper's per-view counts.
FIXTURE = Path(__file__).parent / "fixtures" / "entity" / "views.json.gz"

LARGE = [100.0, 100.0, 160.0, 250.0]  # 60 x 150, centre (130, 175)


def view(boxes, real, generated, confirmed, n_slots):
    """A dumped view: per frame the boxes ``[slot, x0, y0, x1, y1]`` and the detections of the
    recorded and the generated video ``gen``; ``confirmed`` per slot and frame."""

    def dets(frames):
        return {str(i): [list(d) + [0.9] for d in frame] for i, frame in enumerate(frames)}

    return {
        "media_id": "m-p00@0",
        "width": 640,
        "height": 360,
        "n_frames": len(boxes),
        "n_slots": n_slots,
        "boxes": [[[slot, *box, 100.0] for slot, box in frame] for frame in boxes],
        "confirmed": confirmed,
        "dets_real": dets(real),
        "dets_gen": dets(generated),
    }


def counts(v: ViewCounts) -> tuple[int, int, int, int]:
    return (v.n_pairs, v.hits, v.n_dets, v.n_unmatched)


def test_the_paper_views_score_as_in_the_paper():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)
    for item in fixture["gt_views"]:
        window = item["window"]
        labels = np.array([[c == "1" for c in row] for row in item["labels"]])
        gate = visibility_gate(labels, 0, window["n_frames"])
        for arm, expected in item["expected"].items():
            assert list(counts(score_view(window, arm, gate))) == expected, (
                window["media_id"],
                arm,
            )
    item = fixture["predicted_view"]
    window = item["window"]
    labels = np.array([[c == "1" for c in row] for row in item["labels"]])
    boxes = table_boxes(
        np.array(item["states"]),
        0,
        window["n_frames"],
        window["obs_slot"],
        window["width"],
        window["height"],
    )
    gate = visibility_gate(labels, 0, window["n_frames"])
    for arm, expected in item["expected"].items():
        assert list(counts(score_view(window, arm, gate, boxes))) == expected


def test_the_matching_maximizes_the_pairs_before_the_cost():
    assert sorted(assign_one_to_one(np.array([[0.1, 0.9], [0.2, np.inf]]), 1.0)) == [(0, 1), (1, 0)]
    assert assign_one_to_one(np.array([[0.1], [0.2]]), 1.0) == [(0, 0)]  # one player per detection
    assert assign_one_to_one(np.array([[1.5]]), 1.0) == []
    assert assign_one_to_one(np.zeros((0, 2)), 1.0) == []


def test_the_size_gate_selects_after_the_matching():
    small = [140.0, 170.0, 150.0, 190.0]  # 10 x 20, its centre at the generated detection's
    window = view(
        boxes=[[(0, small), (1, LARGE)]],
        real=[[LARGE]],
        generated=[[[135.0, 160.0, 155.0, 200.0]]],  # within one body width of both
        confirmed=[[1], [1]],
        n_slots=2,
    )
    v = score_view(window, "gen", np.ones((2, 1), dtype=bool))
    assert counts(v) == (1, 0, 0, 0)  # the large player is scored and missed: the small one took it
    assert v.in_view.tolist() == [1] and v.frame_pairs.tolist() == [1]


def test_the_recorded_videos_own_hits_match_among_all_confirmed_players():
    other = [170.0, 100.0, 230.0, 250.0]  # centre (200, 175)
    boxes = [[(0, LARGE), (1, other)]]
    real = [[[130.0, 100.0, 190.0, 250.0]]]  # centre (160, 175): nearer the first player
    generated = [[other]]
    gate = np.array([[False], [True]])
    window = view(boxes, real, generated, confirmed=[[1], [1]], n_slots=2)
    assert recording_hits(window).tolist() == [[True], [False]]
    assert counts(score_view(window, "gen", gate)) == (0, 0, 1, 0)
    window = view(boxes, real, generated, confirmed=[[0], [1]], n_slots=2)
    assert counts(score_view(window, "gen", gate)) == (1, 1, 1, 0)


def test_the_partner_scope_drops_the_other_pairs_after_the_matching():
    beside = [150.0, 100.0, 210.0, 250.0]  # centre (180, 175)
    window = view(
        boxes=[[(0, LARGE), (1, beside)]],
        real=[[LARGE, beside]],
        generated=[[[110.0, 100.0, 170.0, 250.0]]],  # within one body width of both, nearer 0
        confirmed=[[1], [1]],
        n_slots=2,
    )
    gate = np.ones((2, 1), dtype=bool)
    assert counts(score_view(window, "gen", gate)) == (2, 1, 1, 0)
    partner = score_view(window, "gen", gate, targets={1})
    assert counts(partner) == (1, 0, 1, 0)  # player 0 took the detection
    assert partner.in_view.tolist() == [2] and partner.frame_pairs.tolist() == [1]
    assert counts(score_view(window, "gen", gate, targets={0})) == (1, 1, 1, 0)


def test_unmatched_is_not_one_to_one_and_needs_no_confirmation():
    tiny = [500.0, 100.0, 510.0, 250.0]  # short side 10: not expected
    generated = [
        LARGE,
        [110.0, 100.0, 170.0, 250.0],  # 10 px from the same expected centre
        [400.0, 100.0, 460.0, 250.0],  # nothing near
        [490.0, 100.0, 520.0, 250.0],  # only the tiny box near
        [300.0, 100.0, 360.0, 250.0],  # a recorded detection near
        [600.0, 100.0, 620.0, 250.0],  # short side 20: not counted
    ]
    window = view(
        boxes=[[(0, LARGE), (1, tiny)]],
        real=[[[300.0, 100.0, 360.0, 250.0]]],
        generated=[generated],
        confirmed=[[0], [0]],
        n_slots=2,
    )
    gate = np.ones((2, 1), dtype=bool)
    assert counts(score_view(window, "gen", gate)) == (0, 0, 5, 2)
    assert score_view(window, "gen", gate, radius=0.1).n_unmatched == 3  # 10 px > 0.1 x 60 px
    assert score_view(window, "gen", np.zeros((2, 1), dtype=bool)).n_unmatched == 4
    with pytest.raises(ValueError, match="frames"):
        score_view(window, "gen", np.ones((2, 2), dtype=bool))


def test_the_gate_folds_onto_latent_frames_from_the_rollout_start():
    labels = np.zeros((1, 12), dtype=bool)
    labels[0, 4] = True  # the last frame of latent frame 1 (frames 1-4)
    assert visibility_gate(labels, 0, 9)[0].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert visibility_gate(labels, 2, 3)[0].tolist() == [1, 1, 1]
    assert visibility_gate(labels, 4, 2)[0].tolist() == [1, 0]
    assert visibility_gate(np.ones((1, 3), dtype=bool), 0, 6)[0].tolist() == [1, 1, 1, 1, 1, 0]


def test_the_gt_gate_reads_visible_and_valid_from_the_rollout_start(tmp_path):
    visible = np.zeros((10, 1000), dtype=bool)
    valid = np.ones((10, 1000), dtype=bool)
    visible[3, 316] = True  # source frame 316: video frame 158 of a rollout from 0
    visible[4, 330], valid[4, 330] = True, False
    visible[5, 330] = True
    np.savez(tmp_path / "m-p00.npz", _binary_visible=visible, _binary_eval_valid=valid)
    whole = gt_visibility_gate(tmp_path, "m-p00", 320, 6, 0)  # frames 160-165 of the rollout
    assert whole[3].tolist() == [1, 0, 0, 0, 0, 0]  # latent frame 40 holds frames 157-160
    assert not whole[4].any() and whole[5].tolist() == [0, 0, 0, 0, 0, 1]
    window = gt_visibility_gate(tmp_path, "m-p00", 320, 6, 320)  # the window's own rollout
    assert not window[3].any() and window[5].tolist() == [0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="not a frame"):
        gt_visibility_gate(tmp_path, "m-p00", 321, 6, 0)


def test_summaries_take_the_views_with_at_least_eight():
    empty = np.zeros(0, dtype=np.int64)

    def counted(n_pairs, hits, n_dets, n_unmatched):
        return ViewCounts(n_pairs, hits, n_dets, n_unmatched, empty, empty, empty)

    views = [counted(7, 0, 7, 7), counted(8, 8, 8, 2), counted(10, 5, 0, 0)]
    summary = summarize_entity(views)
    assert (summary.rendered, summary.rendered_views) == (0.75, 2)  # the two middle values' mean
    assert (summary.unmatched, summary.unmatched_views) == (0.25, 1)
    assert math.isnan(summarize_entity([]).rendered)


def test_crowding_pools_the_pairs_of_the_frames():
    pooled = pool_by_crowding([0, 1, 1, 2, 4], [3, 1, 2, 3, 1], [3, 1, 1, 2, 1])
    assert {n: (b.frames, b.pairs, b.hits) for n, b in pooled.items()} == {
        1: (2, 3, 2),
        2: (1, 3, 2),
        4: (1, 1, 1),
    }
    assert pooled[1].rendered == pytest.approx(2 / 3)


def test_the_window_of_a_view():
    assert window_of("2392598-de_nuke-r03-p07@01368") == "2392598-de_nuke-r03@1368"
