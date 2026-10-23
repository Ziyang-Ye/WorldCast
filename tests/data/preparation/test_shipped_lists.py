"""The paper's files that ship with the repository: the lists the bucket steps take that OpenCS2
does not give (the held-out matches, the excluded media), the evaluation reserve and the validation
index training reads, and the evaluation sets (``configs/eval/``)."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from worldcast.engine.evaluation.protocols import UNIPC
from worldcast.utils.files import sha256_file

REPO = Path(__file__).resolve().parents[3]
#: Each shipped file and its lines.
FILES = {
    "configs/data/held_out_matches.txt": 40,
    "configs/data/excluded_media.txt": 227,
    "configs/data/eval_reserve.txt": 1591,
    "configs/data/validation_index.jsonl": 64,
    "configs/eval/windows.jsonl": 256,
    "configs/eval/window_views.jsonl": 512,
    "configs/eval/round_index.jsonl": 96,
}
MAPS = ("de_dust2", "de_mirage", "de_nuke", "de_ancient")
MEDIA_ID = r"(\d+)-(de_dust2|de_mirage|de_nuke|de_ancient)-r\d\d-p\d\d"


def lines(name: str) -> list[str]:
    return (REPO / name).read_text().split()


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (REPO / name).read_text().splitlines()]


def held_out() -> set[int]:
    return {int(match) for match in lines("configs/data/held_out_matches.txt")}


@pytest.mark.parametrize("name, count", FILES.items())
def test_a_file_is_what_the_docs_digest(name, count):
    path = REPO / name
    assert path.read_text().endswith("\n") and len(path.read_text().splitlines()) == count
    assert sha256_file(path) in (REPO / "docs" / "data.md").read_text()


def test_the_held_out_matches_are_ascending_match_ids():
    matches = lines("configs/data/held_out_matches.txt")
    assert all(re.fullmatch(r"\d+", match) for match in matches)
    assert matches == sorted(set(matches), key=int)


@pytest.mark.parametrize(
    "name", ["configs/data/excluded_media.txt", "configs/data/eval_reserve.txt"]
)
def test_the_media_lists_are_media_ids_of_the_papers_maps(name):
    media = lines(name)
    assert all(re.fullmatch(MEDIA_ID, media_id) for media_id in media)
    assert len(set(media)) == len(media)


def test_the_evaluation_reserve_is_de_dust2_recordings_of_training_matches():
    media = lines("configs/data/eval_reserve.txt")
    assert {re.fullmatch(MEDIA_ID, media_id)[2] for media_id in media} == {"de_dust2"}
    assert not {int(re.fullmatch(MEDIA_ID, media_id)[1]) for media_id in media} & held_out()


def test_the_validation_index_is_the_one_validation_checks():
    assert sha256_file(REPO / "configs" / "data" / "validation_index.jsonl") == UNIPC.index_sha256
    assert Counter(row["map_name"] for row in rows("configs/data/validation_index.jsonl")) == {
        name: 16 for name in MAPS
    }


@pytest.mark.parametrize(
    "name",
    [
        "configs/data/validation_index.jsonl",
        "configs/eval/windows.jsonl",
        "configs/eval/window_views.jsonl",
        "configs/eval/round_index.jsonl",
    ],
)
def test_every_window_and_round_is_of_a_held_out_match(name):
    assert {int(row["match_id"]) for row in rows(name)} <= held_out()


def test_the_windows_are_64_per_map_and_each_has_its_two_views():
    windows = rows("configs/eval/windows.jsonl")
    views = rows("configs/eval/window_views.jsonl")
    assert Counter(row["map_name"] for row in windows) == {name: 64 for name in MAPS}
    by_window = {row["group_id"]: row for row in windows}
    assert len(by_window) == 256
    assert Counter(row["group_id"] for row in views) == {group: 2 for group in by_window}
    for view in views:
        window = by_window[view["group_id"]]
        assert view["media_id"] in window["view_media_ids"]
        assert view["start_frame"] == window["start_frame"]


def test_the_round_index_is_32_rounds_of_three_clients():
    index = rows("configs/eval/round_index.jsonl")
    groups = Counter(row["group"] for row in index)
    assert len(groups) == 32 and set(groups.values()) == {3}
    assert Counter(row["map_name"] for row in index) == {name: 24 for name in MAPS}


def test_the_training_path_template_names_the_shipped_files():
    paths = yaml.safe_load((REPO / "examples" / "train_paths.yaml").read_text())
    assert paths["data"]["exclude_media_manifest"] == "configs/data/eval_reserve.txt"
    assert paths["validation"]["index"] == "configs/data/validation_index.jsonl"
