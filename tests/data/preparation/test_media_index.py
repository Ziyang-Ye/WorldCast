"""The media index: one row per player-round of OpenCS2."""

import argparse
import dataclasses
import hashlib
import json
import re

import pytest

from tests.data.preparation.support import FRAMES, SEATS, write_opencs2, write_ticks
from tests.tools.support import load_tool
from worldcast.data.preparation import media_index
from worldcast.data.preparation.media_index import build_media_index
from worldcast.data.recordings import MediaIndex, read_player_ticks
from worldcast.data.video import load_video_window

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

#: The fields the release reads of a media row (``worldcast.data.recordings.MediaRecord``).
FIELDS = {
    "media_id",
    "match_id",
    "map_name",
    "round",
    "player_slot",
    "fps",
    "video_frames",
    "video_path",
    "ticks_path",
    "ticks_rows",
    "ticks_file_size",
    "ticks_sha256",
    "capture_start_tick",
}
MAPS = ["de_dust2", "de_nuke"]


def expected_row(root, seat) -> dict:
    ticks = root / seat.directory / "ticks.parquet"
    return {
        "media_id": seat.media_id,
        "match_id": seat.match_id,
        "map_name": seat.map_name,
        "round": seat.round,
        "player_slot": seat.slot,
        "fps": 32.0,
        "video_frames": FRAMES,
        "video_path": f"{seat.directory}/video.mp4",
        "ticks_path": f"{seat.directory}/ticks.parquet",
        "ticks_rows": 2 * FRAMES,
        "ticks_file_size": ticks.stat().st_size,
        "ticks_sha256": hashlib.sha256(ticks.read_bytes()).hexdigest(),
        "capture_start_tick": seat.first_tick,
    }


def test_a_row_per_player_round_ordered_by_map_match_round_and_slot(opencs2):
    # the tiny download's index lists the player-rounds in reverse
    rows, refused = build_media_index(opencs2, MAPS, workers=1)
    assert refused == []
    assert rows == [expected_row(opencs2, seat) for seat in SEATS]
    assert all(set(row) == FIELDS for row in rows)


def test_the_rows_do_not_depend_on_the_workers(opencs2):
    assert build_media_index(opencs2, MAPS, workers=1) == build_media_index(
        opencs2, MAPS, workers=4
    )


def test_maps_choose_the_player_rounds_and_the_paper_maps_are_the_default(opencs2):
    rows, _ = build_media_index(opencs2, ["de_nuke"])
    assert [row["media_id"] for row in rows] == [
        seat.media_id for seat in SEATS if seat.map_name == "de_nuke"
    ]
    with pytest.raises(ValueError, match=r"has no media of \['de_inferno'\]"):
        build_media_index(opencs2, ["de_nuke", "de_inferno"])
    parser = argparse.ArgumentParser()
    media_index.add_arguments(parser)
    args = parser.parse_args(["--dataset-root", "a", "--out", "b"])
    assert args.maps == ["de_dust2", "de_mirage", "de_nuke", "de_ancient"] and args.workers >= 1


def test_the_tool_writes_json_lines_that_the_release_reads_back(opencs2, tmp_path, capsys):
    out = tmp_path / "data" / "media_index.jsonl"
    argv = ["media-index", "--dataset-root", str(opencs2), "--out", str(out), "--maps", *MAPS]
    assert load_tool("prepare_data").main(argv) == 0
    assert capsys.readouterr().out == f"media-index: {len(SEATS)} rows -> {out}\n"
    lines = out.read_text().splitlines()
    assert [json.loads(line)["media_id"] for line in lines] == [s.media_id for s in SEATS]
    for line in lines:
        assert line == json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"))

    index = MediaIndex.load(out)
    assert set(index.slots((11, "de_dust2", 1))) == {0, 1, 2}
    assert set(index.slots((11, "de_dust2", 2))) == {0}
    for seat in SEATS:
        media = index.media(seat.media_id)
        assert media.video_path == f"{seat.directory}/video.mp4"
        # the release's own check of the tick table: size, sha256, rows and the time base
        assert len(read_player_ticks(media, opencs2)) == 2 * FRAMES
        frames = load_video_window(
            opencs2 / media.video_path, 0, video_frames=3, source_frames=media.source_frames
        )
        assert frames.shape[:2] == (3, 3)
    out2 = tmp_path / "again.jsonl"
    argv[argv.index(str(out))] = str(out2)
    load_tool("prepare_data").main([*argv, "--workers", "3"])
    assert out2.read_bytes() == out.read_bytes()


def _append_byte(path):
    with path.open("ab") as handle:
        handle.write(b"\0")


def _other_capture_start(root, seat):
    write_ticks(root / seat.directory / "ticks.parquet", dataclasses.replace(seat, round=9))


def _no_ticks(root, seat):
    write_ticks(root / seat.directory / "ticks.parquet", dataclasses.replace(seat, rows=0))


#: reason, what to do to the files of the second player-round, what the message says
DAMAGE = [
    (
        "video_missing",
        lambda root, seat: (root / seat.directory / "video.mp4").unlink(),
        "not found",
    ),
    (
        "video_size",
        lambda root, seat: _append_byte(root / seat.directory / "video.mp4"),
        r"has \d+ bytes, the OpenCS2 index says \d+",
    ),
    (
        "video_frames",
        lambda root, seat: (root / seat.directory / "video.mp4").write_bytes(
            b"\0" * (root / seat.directory / "video.mp4").stat().st_size
        ),
        r"0 frames, expected 40",
    ),
    (
        "ticks_missing",
        lambda root, seat: (root / seat.directory / "ticks.parquet").unlink(),
        "not found",
    ),
    (
        "ticks_unreadable",
        lambda root, seat: (root / seat.directory / "ticks.parquet").write_bytes(b"no parquet"),
        r"cannot be read \(",
    ),
    ("ticks_empty", _no_ticks, "is empty"),
    ("ticks_start", _other_capture_start, r"starts at tick 5900, .* capture starts at 5100"),
]


@pytest.mark.parametrize("reason, damage, message", DAMAGE, ids=[d[0] for d in DAMAGE])
def test_a_player_round_that_fails_a_check_is_left_out_and_counted(
    opencs2, reason, damage, message
):
    seat = SEATS[1]
    damage(opencs2, seat)
    rows, refused = build_media_index(opencs2, MAPS)
    assert [media_id for media_id, _ in refused] == [seat.media_id]
    assert refused[0][1].reason == reason
    assert re.search(message, str(refused[0][1])) and seat.media_id in str(refused[0][1])
    assert [row["media_id"] for row in rows] == [s.media_id for s in SEATS if s != seat]


def test_a_video_that_has_not_the_indexed_frame_count_is_left_out(tmp_path):
    seats = SEATS[:2]
    write_opencs2(tmp_path, seats, index={seats[0].media_id: {"video_frames": FRAMES + 1}})
    rows, refused = build_media_index(tmp_path, ["de_dust2"])
    assert [row["media_id"] for row in rows] == [seats[1].media_id]
    assert refused[0][1].reason == "video_frames"
    assert f"{FRAMES} frames, expected {FRAMES + 1}" in str(refused[0][1])


def test_the_tool_names_what_it_left_out(opencs2, tmp_path, capsys):
    (opencs2 / SEATS[0].directory / "video.mp4").unlink()
    out = tmp_path / "media_index.jsonl"
    argv = ["media-index", "--dataset-root", str(opencs2), "--out", str(out), "--maps", *MAPS]
    assert load_tool("prepare_data").main(argv) == 0
    captured = capsys.readouterr()
    assert captured.out == f"media-index: {len(SEATS) - 1} rows -> {out}\n"
    assert (
        "left out 1 player-rounds" in captured.err and "video_missing: 1 (first: " in captured.err
    )
    assert len(out.read_text().splitlines()) == len(SEATS) - 1


def test_an_index_that_cannot_be_used_is_refused(opencs2, tmp_path):
    with pytest.raises(FileNotFoundError, match="OpenCS2 index does not exist"):
        build_media_index(tmp_path / "nowhere", MAPS)
    index = opencs2 / "index" / "pov_rounds.parquet"
    index.write_bytes(b"no parquet")
    with pytest.raises(ValueError, match="cannot read the OpenCS2 index"):
        build_media_index(opencs2, MAPS)
    pq.write_table(pa.table({"media_id": ["a"], "map_name": ["de_nuke"]}), index)
    with pytest.raises(ValueError, match="lacks the columns .*match_id"):
        build_media_index(opencs2, MAPS)
    twice = tmp_path / "twice"
    write_opencs2(twice, (SEATS[0], SEATS[0]))
    with pytest.raises(ValueError, match=f"repeats media {SEATS[0].media_id}"):
        build_media_index(twice, ["de_dust2"])


def test_a_download_without_a_usable_player_round_is_refused(opencs2, tmp_path, capsys):
    for seat in SEATS:
        (opencs2 / seat.directory / "video.mp4").unlink()
    with pytest.raises(ValueError, match=r"passes its checks:\n  video_missing: 6 \(first: "):
        build_media_index(opencs2, MAPS)
    out = tmp_path / "media_index.jsonl"
    argv = ["media-index", "--dataset-root", str(opencs2), "--out", str(out), "--maps", *MAPS]
    with pytest.raises(SystemExit) as stop:
        load_tool("prepare_data").main(argv)
    assert stop.value.code == 2  # a usage error
    assert "no player-round of" in capsys.readouterr().err
    assert not out.exists()
