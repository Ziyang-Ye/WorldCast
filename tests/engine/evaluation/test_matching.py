"""Agreement by feature matching with the matchers stubbed, and the paper's cells when
``WORLDCAST_PAPER_REFS`` is set.

The intrinsics, the camera rotations and RotErr on known geometry, the screen mask, the keys and
their strides, co-visibility, the per-round statistics and the self-check. The paper's per-pair
results are read from ``$WORLDCAST_PAPER_REFS/matching/<arm>.npz``
(``worldcast``, ``scene_state_off``, ``recorded``: ``gim_keys`` ``[n, 7]`` int64, the round then
:data:`KEY_COLUMNS`, with ``matched_pixels``; ``loftr_keys`` with ``inliers`` and
``rotation_error``; ``mask_keys`` with ``matched_pixels_masked`` (-1 off the Mat. Pix. keys),
``inliers_masked`` and ``rotation_error_masked``), the clients' latent frames from
``matching/latent_frames.npz`` (``latent_frames`` ``[31, 5]``) and the recorded pair rows from
``depth_agreement/closed_loop/recorded.npz``.
"""

import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from worldcast.engine.evaluation.depth_agreement import BETWEEN_CLIENTS, CHANCE, OWN_NEXT_FRAME
from worldcast.engine.evaluation.matching import (
    GIM_SIZE,
    GIM_STRIDE,
    KEY_COLUMNS,
    LOFTR_BATCH,
    LOFTR_STRIDE,
    camera_intrinsics,
    camera_rotations,
    covisible,
    in_screen_mask,
    load_gim,
    load_loftr,
    match_keys,
    matched_pixels,
    needed_frames,
    read_frames,
    relative_pose,
    rotation_error,
    round_statistics,
    score_matched_pixels,
    score_relative_poses,
    self_check,
)
from worldcast.engine.evaluation.stats import mean_over_units, unit_interval
from worldcast.hub import METRIC_WEIGHTS

REFS = os.environ.get("WORLDCAST_PAPER_REFS")
ARMS = ("worldcast", "scene_state_off", "recorded")


def pair_row(a: int, b: int, frame: int, kind: int = BETWEEN_CLIENTS, agree: int = 1) -> list[int]:
    """A pair row of round 0 (:data:`~worldcast.engine.evaluation.depth_agreement.ROW_COLUMNS`)."""
    return [0, a, b, frame, kind, 10, agree, agree, 0, 0]


def key(a: int, b: int, frame: int, kind: int = BETWEEN_CLIENTS, frame_a: int | None = None):
    return [kind, a, b, frame, frame if frame_a is None else frame_a, frame]


def test_intrinsics_follow_the_recordings_field_of_view():
    intrinsics = camera_intrinsics()
    tan_h = math.tan(math.radians(106.26 / 2))
    assert intrinsics[0, 0] == pytest.approx(252.0009, abs=1e-4)
    assert intrinsics[1, 1] == pytest.approx(256.0010, abs=1e-4)
    assert intrinsics[0, 0] == 336 / tan_h and intrinsics[1, 1] == 192 / (tan_h * 9 / 16)
    assert intrinsics[:, 2].tolist() == [336.0, 192.0, 1.0]
    assert intrinsics[0, 1] == intrinsics[1, 0] == 0.0


@pytest.mark.parametrize("yaw, pitch", [(0.0, 0.0), (90.0, 0.0), (-135.0, 20.0), (30.0, -45.0)])
def test_camera_rotations_take_the_view_to_opencv_axes(yaw, pitch):
    rotation = camera_rotations(np.array([yaw]), np.array([pitch]))[0]
    y, p = math.radians(yaw), math.radians(pitch)
    forward = [math.cos(y) * math.cos(p), math.sin(y) * math.cos(p), -math.sin(p)]
    right = [math.sin(y), -math.cos(y), 0.0]
    up = [math.cos(y) * math.sin(p), math.sin(y) * math.sin(p), math.cos(p)]
    assert rotation @ forward == pytest.approx([0, 0, 1], abs=1e-6)
    assert rotation @ right == pytest.approx([1, 0, 0], abs=1e-6)
    assert rotation @ up == pytest.approx([0, -1, 0], abs=1e-6)


def test_rotation_error_is_the_angle_between_rotations():
    turn = math.radians(30.0)
    about_z = np.array(
        [[math.cos(turn), -math.sin(turn), 0], [math.sin(turn), math.cos(turn), 0], [0, 0, 1]]
    )
    assert rotation_error(about_z, np.eye(3)) == pytest.approx(30.0)
    assert rotation_error(about_z, about_z) == pytest.approx(0.0, abs=1e-6)


def project(points: np.ndarray, rotation: np.ndarray, eye: np.ndarray) -> np.ndarray:
    camera = (points - eye) @ rotation.T
    pixels = camera @ camera_intrinsics().T
    return (pixels[:, :2] / pixels[:, 2:]).astype(np.float32)


def test_relative_pose_recovers_the_recorded_cameras_rotation():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    rotations = camera_rotations(np.array([10.0, 17.0]), np.array([5.0, -3.0]))
    eyes = np.array([[0.0, 0.0, 64.0], [0.0, -60.0, 70.0]])
    view = np.linalg.inv(rotations[0])
    local = np.column_stack(
        [rng.uniform(-0.9, 0.9, (200, 2)) * [1.33, 0.75], np.ones(200)]
    ) * rng.uniform(200, 2000, (200, 1))
    points = local @ view.T + eyes[0]
    first, second = project(points, rotations[0], eyes[0]), project(points, rotations[1], eyes[1])
    inliers, error = relative_pose(first, second, rotations[0], rotations[1])
    assert inliers == 200
    assert error < 0.05
    _, swapped = relative_pose(first, second, rotations[1], rotations[0])
    assert swapped > 10.0
    inliers, error = relative_pose(first[:7], second[:7], rotations[0], rotations[1])
    assert inliers == 0 and math.isnan(error)


def test_screen_mask_covers_the_lower_right_and_the_lower_left_corner():
    u = np.array([0.5, 0.49, 0.9, 0.3, 0.31, 0.2, 0.2, 0.6])
    v = np.array([0.45, 0.9, 0.44, 0.7, 0.9, 0.69, 1.0, 0.1])
    expected = [True, False, False, True, False, False, True, False]
    assert in_screen_mask(u, v).tolist() == expected
    assert in_screen_mask(torch.tensor(u), torch.tensor(v)).tolist() == expected


def test_match_keys_keep_the_stride_the_kinds_and_the_row_order():
    rows = []
    for k in range(1, 130):
        rows += [pair_row(0, 1, k), pair_row(0, 1, k, CHANCE), pair_row(0, 0, k, OWN_NEXT_FRAME)]
        rows += [pair_row(1, 0, k)]
    keys = match_keys(np.array(rows), [100, 300], GIM_STRIDE, (BETWEEN_CLIENTS, CHANCE))
    assert keys.dtype == np.int64 and keys.shape[1] == len(KEY_COLUMNS)
    assert keys[:, 3].tolist() == [32, 32, 32, 64, 64, 64, 96, 96, 96, 128, 128, 128]
    assert keys[:, 0].tolist() == [0, 2, 0] * 4
    chance = keys[keys[:, 0] == CHANCE]
    # 10 s later while the shorter client has it (N = 100), else 10 s earlier
    assert chance[:, 4].tolist() == [72, 24, 56, 88] and chance[:, 5].tolist() == [32, 64, 96, 128]
    between = keys[keys[:, 0] == BETWEEN_CLIENTS]
    assert (between[:, 4] == between[:, 3]).all() and (between[:, 5] == between[:, 3]).all()
    loftr = match_keys(np.array(rows), [100, 300], LOFTR_STRIDE)
    assert set(loftr[:, 3] % 4) == {0} and set(loftr[:, 0]) == {BETWEEN_CLIENTS}
    assert len(loftr) == 2 * len(range(4, 130, 4))


def test_needed_frames_are_each_clients_compared_frames():
    keys = np.array([key(0, 1, 32), key(0, 1, 32, CHANCE, 72), key(2, 0, 64)])
    assert needed_frames(keys) == {0: [32, 64, 72], 1: [32], 2: [64]}


def test_covisibility_is_directional_and_chance_keys_follow_their_pair():
    rows = np.array(
        [pair_row(0, 1, 8), pair_row(1, 0, 8, agree=0), pair_row(0, 1, 12, agree=0)]
        + [pair_row(0, 1, 8, CHANCE, agree=0), pair_row(1, 0, 12, CHANCE, agree=5)]
    )
    keys = np.array(
        [key(0, 1, 8), key(1, 0, 8), key(0, 1, 12), key(0, 1, 8, CHANCE, 48), key(1, 0, 12, CHANCE)]
    )
    assert covisible(keys, rows).tolist() == [True, False, False, True, False]
    assert covisible(keys[:0], rows).shape == (0,)


def test_round_statistics_split_at_30_s_over_covisible_same_instant_pairs():
    frames = [4, 8, 116, 120, 200, 240]
    keys = np.array([key(0, 1, k) for k in frames] + [key(0, 1, 8, CHANCE, 48)])
    values = np.array([1.0, 3.0, 5.0, 10.0, math.nan, 30.0, 1000.0])
    keep = np.array([True, True, True, True, True, False, True])
    assert round_statistics(keys, values, keep, "mean")[0] == 3.0
    assert math.isnan(round_statistics(keys, values, keep, "mean")[1])
    assert round_statistics(keys, values, keep, "median").tolist() == [3.0, 10.0]
    nothing = round_statistics(keys, values, np.zeros(7, bool), "median")
    assert np.isnan(nothing).all()
    with pytest.raises(ValueError):
        round_statistics(keys, values, keep, "max")


def test_self_check_takes_the_pairs_present_at_both_offsets():
    keys = np.array(
        [key(0, 1, 40), key(0, 1, 40, CHANCE, 80), key(0, 1, 44), key(1, 0, 48)]
        + [key(1, 0, 48, CHANCE, 88), key(1, 0, 52, CHANCE, 92)]
    )
    values = np.array([2.0, 10.0, 99.0, 4.0, 20.0, 99.0])
    keep = np.array([True, True, True, True, True, True])
    assert self_check(keys, values, keep, "mean") == (3.0, 15.0)
    keep[3] = False
    assert self_check(keys, values, keep, "mean") == (2.0, 10.0)


class StubGim:
    """``match`` returns a 6 x 8 certainty per frame; the query half's warp is set per test."""

    def __init__(self, certainty: torch.Tensor, warp: torch.Tensor) -> None:
        self.certainty, self.warp, self.images = certainty, warp, []

    def match(self, first: torch.Tensor, second: torch.Tensor):
        self.images.append((first, second))
        return self.warp, self.certainty


def warp_of(u: float, v: float) -> tuple[float, float]:
    """The matcher's coordinates in ``[-1, 1]`` of a position of a 384 x 672 frame padded to 504
    rows."""
    return 2 * u - 1, (v * 384 + 60) / 504 * 2 - 1


def test_matched_pixels_count_the_query_half_and_the_masked_cells():
    certainty = torch.zeros(6, 16)
    certainty[:, :8] = 0.06
    certainty[0, 0] = 0.05
    certainty[:, 8:] = 0.9
    warp = torch.zeros(6, 16, 4)
    warp[..., :2] = torch.tensor(warp_of(0.1, 0.1))
    warp[..., 2:] = torch.tensor(warp_of(0.1, 0.1))
    warp[1, :, 2:] = torch.tensor(warp_of(0.7, 0.6))
    warp[2, :2, :2] = torch.tensor(warp_of(0.2, 0.8))
    model = StubGim(certainty, warp)
    query, support = torch.rand(3, 384, 672), torch.rand(3, 384, 672)
    assert matched_pixels(model, query, support) == (47, 47 - 8 - 2)
    first, second = model.images[0]
    assert first.shape == second.shape == (1, 3, 504, 672)
    assert (first[0, :, :60] == 0).all() and (first[0, :, 444:] == 0).all()
    assert torch.equal(first[0, :, 60:444], query) and torch.equal(second[0, :, 60:444], support)
    assert GIM_SIZE == (672, 896)


def test_score_matched_pixels_compares_each_keys_frames(monkeypatch):
    frames = {
        c: {k: torch.full((3, 384, 672), float(c * 100 + k)) for k in (32, 72)} for c in (0, 1)
    }
    calls = []

    def stub(model, query, support):
        calls.append((float(query[0, 0, 0]), float(support[0, 0, 0])))
        return len(calls), 0

    keys = np.array([key(0, 1, 32), key(0, 1, 32, CHANCE, 72), key(1, 0, 72)])
    monkeypatch.setattr(sys.modules[score_matched_pixels.__module__], "matched_pixels", stub)
    out = score_matched_pixels(None, keys, frames)
    assert calls == [(32.0, 132.0), (72.0, 132.0), (172.0, 72.0)]
    assert out["matched_pixels"].tolist() == [1, 2, 3]
    assert out["matched_pixels_masked"].tolist() == [0, 0, 0]


class StubLoftr:
    """Two matches per pair inside the frame, one in the screen mask, one below the confidence."""

    def __init__(self) -> None:
        self.batches = []

    def __call__(self, data):
        first = data["image0"]
        self.batches.append(tuple(first.shape))
        count = len(first)
        points = torch.tensor([[100.0, 50.0], [600.0, 300.0], [10.0, 10.0]]).repeat(count, 1)
        points[0::3, 0] += first[:, 0, 0, 0] * 10
        return {
            "keypoints0": points,
            "keypoints1": points + 1,
            "confidence": torch.tensor([0.9, 0.8, 0.5]).repeat(count),
            "batch_indexes": torch.arange(count).repeat_interleave(3),
        }


def test_score_relative_poses_runs_loftr_in_batches_and_drops_masked_matches(monkeypatch):
    module = sys.modules[score_relative_poses.__module__]
    seen = []

    def stub(points_a, points_b, rotation_a, rotation_b):
        seen.append((points_a.copy(), points_b.copy(), rotation_a[0, 0], rotation_b[0, 0]))
        return len(points_a), float(points_a[0, 0])

    monkeypatch.setattr(module, "relative_pose", stub)
    keys = np.array([key(0, 1, 4 * i) for i in range(1, 21)])
    frames = {c: {4 * i: torch.full((3, 384, 672), 0.01 * i) for i in range(1, 21)} for c in (0, 1)}
    rotations = {
        c: np.arange(100, dtype=float)[:, None, None] * np.ones((3, 3)) + c for c in (0, 1)
    }
    model = StubLoftr()
    out = score_relative_poses(model, keys, frames, rotations)
    assert model.batches == [(LOFTR_BATCH, 1, 384, 672)] * 2 + [(4, 1, 384, 672)]
    assert out["inliers"].tolist() == [2] * 20 and out["inliers_masked"].tolist() == [1] * 20
    assert out["rotation_error"].dtype == np.float32
    assert out["rotation_error"][0] == pytest.approx(100.5 + 0.1)
    unmasked, masked = seen[0], seen[1]
    assert np.allclose(unmasked[0], [[100.6, 50.5], [600.5, 300.5]])
    assert np.allclose(unmasked[1], [[101.6, 51.5], [601.5, 301.5]])
    assert np.allclose(masked[0], [[100.6, 50.5]]) and np.allclose(masked[1], [[101.6, 51.5]])
    assert (unmasked[2], unmasked[3]) == (4.0, 5.0)


def test_read_frames_take_every_nth_video_frame(monkeypatch):
    asked = []

    class Reader:
        def __init__(self, path, num_threads):
            self.height, self.width = (384, 672) if "gen" in str(path) else (192, 336)

        def __len__(self):
            return 400

        def get_batch(self, indices):
            asked.append(list(indices))
            pixels = np.stack(
                [np.full((self.height, self.width, 3), i % 256, np.uint8) for i in indices]
            )
            return types.SimpleNamespace(asnumpy=lambda: pixels)

    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=Reader))
    frames = read_frames("gen.mp4", [1, 25], 4, "cpu")
    assert asked == [[4, 100]] and sorted(frames) == [1, 25]
    assert frames[25].shape == (3, 384, 672) and float(frames[25][0, 0, 0]) == pytest.approx(
        100 / 255
    )
    recorded = read_frames("rec.mp4", [2], 8, "cpu")
    assert asked[-1] == [16] and recorded[2].shape == (3, 384, 672)
    assert float(recorded[2].min()) == pytest.approx(16 / 255) and float(recorded[2].max()) <= 1.0
    with pytest.raises(ValueError, match="400 frames"):
        read_frames("gen.mp4", [100], 4, "cpu")


def test_the_loaders_check_the_weights(tmp_path):
    other = tmp_path / "weights.ckpt"
    other.write_bytes(b"not the weights")
    with pytest.raises(ValueError, match=METRIC_WEIGHTS["gim_dkm"].sha256):
        load_gim(other, "cpu")
    with pytest.raises(ValueError, match=METRIC_WEIGHTS["loftr"].sha256):
        load_loftr(other, "cpu")


def test_gims_matcher_runs_on_a_small_grid():
    pytest.importorskip("einops")
    pytest.importorskip("torchvision")
    from worldcast.third_party.gim_dkm.model import DKMv3

    torch.manual_seed(0)
    model = DKMv3(64, 96).eval()
    model.upsample_res = (96, 128)
    warp, certainty = model.match(torch.rand(1, 3, 48, 64), torch.rand(1, 3, 48, 64))
    assert warp.shape == (96, 256, 4) and certainty.shape == (96, 256)
    assert ((certainty >= 0) & (certainty <= 1)).all() and (warp.abs() <= 1).all()


#: tab:longhorizon(a), the paper's scorer: (first 30 s, after 30 s) of On (worldcast) and Off
#: (scene_state_off), and On - Off with its interval (point, low, high).
PAPER = {
    "matched_pixels": (
        (92131.9994133992, 44302.28028540685),
        (72018.82874788782, 31094.597003967174),
        (
            (20113.170665511378, 13207.683281439673),
            (5077.329002818584, 4508.4857788483905),
            (35674.83915917858, 22483.71546489634),
        ),
    ),
    "inliers": (
        (51.458150984718536, 27.71002860589022),
        (46.22447910268878, 18.749559677513172),
        (
            (5.233671882029752, 8.960468928377052),
            (1.1383865564347633, 5.06665237301616),
            (9.340272628414194, 13.147281215468318),
        ),
    ),
    "rotation_error": (
        (28.98596106806109, 55.62096614222373),
        (31.834435616770097, 59.05542035256663),
        (
            (-2.848474548709008, -3.4344542103428997),
            (-4.781840261336296, -6.651478439761746),
            (-0.8970740552871461, -0.3351040470984678),
        ),
    ),
}
#: The screen mask: On - Off (point, low, high) per segment; Mat. Pix. in thousands.
PAPER_MASKED = {
    "matched_pixels_masked": (
        (12.66083238182771, 7.996536093829098),
        (2.153710793455114, 2.1846399875059674),
        (23.36400161235236, 14.085135635784107),
    ),
    "inliers_masked": (
        (3.094405157615129, 5.219090762510119),
        (0.2031356536654369, 2.677663786976754),
        (5.937358464266558, 7.912951662720507),
    ),
    "rotation_error_masked": (
        (-1.8820616429851902, -12.93258414729949),
        (-4.183391334741346, -22.062658131507135),
        (-0.11324225875639149, -4.797164192507345),
    ),
}
#: The recorded video's self-check: same instant, 10 s apart (mean over rounds), and the rounds
#: where the same instant is better.
PAPER_SELF_CHECK = {
    "matched_pixels": (163075.691, 52334.437, 29),
    "inliers": (128.335, 36.358, 31),
    "rotation_error": (16.689, 64.901, 31),
}


@pytest.fixture(scope="module")
def paper():
    root = Path(REFS or "")
    paths = [root / "matching" / f"{c}.npz" for c in ARMS]
    paths += [root / "matching" / "latent_frames.npz"]
    paths += [root / "depth_agreement" / "closed_loop" / "recorded.npz"]
    missing = [p for p in paths if not p.is_file()]
    if not REFS or missing:
        pytest.skip(
            f"the paper's per-pair results are not there ({missing}); set WORLDCAST_PAPER_REFS"
        )
    results = {c: dict(np.load(p)) for c, p in zip(ARMS, paths)}
    rows = np.load(paths[-1])["rows"]
    rounds = len(np.load(paths[-2])["latent_frames"])
    return dict(
        results=results,
        latent_frames=np.load(paths[-2])["latent_frames"],
        rows=[rows[rows[:, 0] == r] for r in range(rounds)],
    )


def per_round(paper, arm, prefix, field):
    data = paper["results"][arm]
    for r, rows in enumerate(paper["rows"]):
        sel = data[f"{prefix}_keys"][:, 0] == r
        yield data[f"{prefix}_keys"][sel, 1:], data[field][sel], rows


def table(paper, arm, prefix, field, statistic, scale=1.0, stride=1):
    out = []
    for keys, values, rows in per_round(paper, arm, prefix, field):
        sel = (keys[:, 3] % stride == 0) & (values >= 0) if stride > 1 else np.ones(len(keys), bool)
        out.append(round_statistics(keys[sel], values[sel], covisible(keys[sel], rows), statistic))
    return np.array(out) / scale


def test_paper_keys_are_match_keys_of_the_recorded_rows(paper):
    for arm in ARMS:
        data = paper["results"][arm]
        for r, rows in enumerate(paper["rows"]):
            counts = paper["latent_frames"][r]
            stored = data["gim_keys"][data["gim_keys"][:, 0] == r, 1:]
            assert np.array_equal(stored, match_keys(rows, counts, GIM_STRIDE, (0, 2)))
            kinds = (0, 2) if arm == "recorded" else (0,)
            stored = data["loftr_keys"][data["loftr_keys"][:, 0] == r, 1:]
            assert np.array_equal(stored, match_keys(rows, counts, LOFTR_STRIDE, kinds))


@pytest.mark.parametrize(
    "field, prefix, statistic",
    [
        ("matched_pixels", "gim", "mean"),
        ("inliers", "loftr", "mean"),
        ("rotation_error", "loftr", "median"),
    ],
)
def test_paper_cells_and_differences(paper, field, prefix, statistic):
    on = table(paper, "worldcast", prefix, field, statistic)
    off = table(paper, "scene_state_off", prefix, field, statistic)
    expected_on, expected_off, (point, low, high) = PAPER[field]
    assert mean_over_units(on).tolist() == pytest.approx(expected_on, rel=1e-12)
    assert mean_over_units(off).tolist() == pytest.approx(expected_off, rel=1e-12)
    assert mean_over_units(on - off).tolist() == pytest.approx(point, rel=1e-12)
    bounds = unit_interval(on - off)
    assert bounds[0].tolist() == pytest.approx(low, rel=1e-12)
    assert bounds[1].tolist() == pytest.approx(high, rel=1e-12)


@pytest.mark.parametrize("field", list(PAPER_MASKED))
def test_paper_masked_differences(paper, field):
    pixels = field == "matched_pixels_masked"
    statistic = "median" if field.startswith("rotation") else "mean"
    options = dict(scale=1000.0, stride=GIM_STRIDE) if pixels else {}
    on = table(paper, "worldcast", "mask", field, statistic, **options)
    off = table(paper, "scene_state_off", "mask", field, statistic, **options)
    point, low, high = PAPER_MASKED[field]
    assert mean_over_units(on - off).tolist() == pytest.approx(point, rel=1e-12)
    bounds = unit_interval(on - off)
    assert bounds[0].tolist() == pytest.approx(low, rel=1e-12)
    assert bounds[1].tolist() == pytest.approx(high, rel=1e-12)


@pytest.mark.parametrize("field", list(PAPER_SELF_CHECK))
def test_paper_self_check(paper, field):
    prefix = "gim" if field == "matched_pixels" else "loftr"
    statistic = "median" if field == "rotation_error" else "mean"
    pairs = np.array(
        [
            self_check(keys, values, covisible(keys, rows), statistic)
            for keys, values, rows in per_round(paper, "recorded", prefix, field)
        ]
    )
    same, apart, better = PAPER_SELF_CHECK[field]
    assert round(float(pairs[:, 0].mean()), 3) == same
    assert round(float(pairs[:, 1].mean()), 3) == apart
    sign = -1 if field == "rotation_error" else 1
    assert int(np.sum(sign * (pairs[:, 0] - pairs[:, 1]) > 0)) == better
