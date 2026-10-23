"""VBench aggregation: the shard reader, the 20-minute and ten-second bins, and the sensitivity."""

import json
from pathlib import Path

import pytest

from worldcast.engine.evaluation.vbench import (
    DAMAGES,
    DIMENSIONS,
    hour_bins,
    read_shards,
    round_bins,
    sensitivity,
)


def write_shard(path: Path, clips: dict[str, dict]) -> Path:
    path.write_text(json.dumps({"dims": list(DIMENSIONS), "clips": clips}))
    return path


def test_shards_read_flags_as_1_or_0_and_drop_what_vbench_did_not_score(tmp_path):
    first = write_shard(
        tmp_path / "shard0.json",
        {
            "a__s00000.mp4": {"subject_consistency": 0.8, "dynamic_degree": True},
            "a__s00060.mp4": {"subject_consistency": 0.7, "dynamic_degree": None},
        },
    )
    second = write_shard(
        tmp_path / "shard1.json",
        {"a__s00060.mp4": {"dynamic_degree": False}, "b__s00000.mp4": {"imaging_quality": 71}},
    )
    scores = read_shards([first, second])
    assert list(scores) == ["a__s00000.mp4", "a__s00060.mp4", "b__s00000.mp4"]
    assert scores["a__s00000.mp4"] == {"subject_consistency": 0.8, "dynamic_degree": 1.0}
    assert scores["a__s00060.mp4"] == {"subject_consistency": 0.7, "dynamic_degree": 0.0}
    assert scores["b__s00000.mp4"] == {"imaging_quality": 71.0}


def test_one_hour_clips_fall_in_20_minute_bins_by_their_start():
    scores = {
        "hour-a__s00000.mp4": {"subject_consistency": 0.9, "dynamic_degree": 1.0},
        "hour-a__s01140.mp4": {"subject_consistency": 0.7, "dynamic_degree": 0.0},
        "hour-b__s01200.mp4": {"subject_consistency": 0.5},
        "hour-b__s03540.mp4": {"subject_consistency": 0.6, "imaging_quality": 70.0},
    }
    bins = hour_bins(scores)
    assert list(bins) == [0, 1, 2]
    assert bins[0] == {
        "clips": 2,
        "subject_consistency": pytest.approx(0.8),
        "dynamic_degree": 0.5,  # the fraction of clips classified as moving
    }
    assert bins[1] == {"clips": 1, "subject_consistency": 0.5}
    assert bins[2] == {"clips": 1, "subject_consistency": 0.6, "imaging_quality": 70.0}


def test_whole_round_windows_fall_in_ten_second_bins_of_the_first_60_s():
    scores = {
        f"2393218-de_mirage-r15-p0{p}__start{320 * k:06d}.mp4": {"aesthetic_quality": k + p / 10}
        for k in range(8)
        for p in (2, 3)
    }
    bins = round_bins(scores)
    assert list(bins) == [0, 1, 2, 3, 4, 5]  # windows from 60 s on are left out
    for k, means in bins.items():
        assert means == {"clips": 2, "aesthetic_quality": pytest.approx(k + 0.25)}


def test_the_sensitivity_is_a_ratio_of_means_over_clips_with_an_original():
    scores = {
        "orig__w0__start000000.mp4": {"subject_consistency": 1.0, "dynamic_degree": 1.0},
        "orig__w1__start000320.mp4": {"subject_consistency": 3.0, "dynamic_degree": 1.0},
        "blur1__w0__start000000.mp4": {"subject_consistency": 2.0, "dynamic_degree": 1.0},
        "blur1__w1__start000320.mp4": {"subject_consistency": 2.0, "dynamic_degree": 0.0},
        "blur1__w9__start002880.mp4": {"subject_consistency": 9.0},  # no original: left out
        "freeze3__w1__start000320.mp4": {"subject_consistency": 3.3, "imaging_quality": 1.0},
    }
    changes = sensitivity(scores)
    assert list(changes) == ["blur1", "freeze3"]
    # (2 + 2) / 2 against (1 + 3) / 2: no change, while the per-clip changes average +33 %
    assert changes["blur1"] == {"clips": 2, "subject_consistency": 0.0, "dynamic_degree": -50.0}
    # imaging quality: the original has none
    assert changes["freeze3"] == {"clips": 1, "subject_consistency": pytest.approx(10.0)}


def test_the_damages_of_table_14_are_named_by_their_clip_prefix():
    assert list(DAMAGES) == ["blur1", "blur3", "jpeg1", "jpeg3", "noise3", "exposure3", "freeze3"]
