"""The raw-video manifest of stages 1 and 1_long."""

import argparse
import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from tests.data.preparation.support import FRAMES, SEATS, Seat, write_opencs2, write_ticks
from tests.tools.support import load_tool
from worldcast.config.training import load_train_config
from worldcast.data.controls import TICK_GATE_SECONDS, AlignmentError, align_ticks_to_video_frames
from worldcast.data.latents import video_frame_count
from worldcast.data.preparation import manifest as manifest_step
from worldcast.data.preparation.manifest import (
    CONTROL_COLUMNS,
    STAGE_REQUIRES_ALIVE,
    build_manifest,
    stage_pixel_frames,
)
from worldcast.data.preparation.media_index import build_media_index
from worldcast.data.recordings import MediaRecord, read_tick_columns
from worldcast.data.training import RawVideoWindows
from worldcast.data.window import WindowSpec
from worldcast.utils.files import write_jsonl

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

REPO = Path(__file__).resolve().parents[3]
MAPS = ["de_dust2", "de_nuke"]
#: Two latent frames: five video frames from ten source frames, a window the tiny recordings hold
#: 16 times at stride 2.
PIXEL_FRAMES = 5
SPAN = 9
FIELDS = {
    "media_id",
    "match_id",
    "map_name",
    "round",
    "player_slot",
    "video_path",
    "ticks_path",
    "video_frames",
    "video_file_size",
    "ticks_file_size",
    "ticks_sha256",
    "ticks_rows",
    "fps",
    "pixel_frames",
    "skip_frame",
    "stride",
    "max_tick_gap_seconds",
    "start_frames",
    "n_windows",
}


def media_records(root, maps=MAPS) -> list[MediaRecord]:
    rows, _ = build_media_index(root, maps)
    return [MediaRecord.from_row(row) for row in rows]


def refreshed(root, media: MediaRecord) -> MediaRecord:
    """``media`` with the size and sha256 of its tick table as it is now."""
    path = root / media.ticks_path
    return dataclasses.replace(
        media,
        ticks_file_size=path.stat().st_size,
        ticks_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def reader_accepts(root, record: dict, start: int) -> bool:
    """Whether the stage-1 reader aligns the ticks of the window at ``start``."""
    ticks = read_tick_columns(root / record["ticks_path"], CONTROL_COLUMNS, record["ticks_rows"])
    try:
        align_ticks_to_video_frames(
            timestamps=ticks["t"],
            held_buttons=ticks["active"],
            delta_pitch=ticks["delta_pitch"],
            delta_yaw=ticks["delta_yaw"],
            input_weapons=ticks["input_weapon"],
            start_frame=start,
            num_frames=record["pixel_frames"],
            source_fps=record["fps"],
            max_tick_gap_seconds=record["max_tick_gap_seconds"],
            camera_encoding="clip",
        )
    except AlignmentError:
        return False
    return True


def test_the_defaults_are_the_papers_stage_settings():
    assert stage_pixel_frames("1") == 81 and stage_pixel_frames("1_long") == 161
    with pytest.raises(ValueError, match="stage must be one of"):
        stage_pixel_frames("2")
    parser = argparse.ArgumentParser()
    manifest_step.add_arguments(parser)
    args = parser.parse_args(["--dataset-root", "a", "--media-index", "b", "--out", "c"])
    assert args.stage == "1" and args.stride == 8 and args.workers >= 1
    for stage, config in (("1", "stage1"), ("1_long", "stage1_long")):
        cfg = load_train_config([REPO / "configs" / "train" / f"{config}.yaml"])
        assert video_frame_count(cfg.stage.latent_frames) == stage_pixel_frames(stage)
        assert cfg.stage.player_state_field is False


def test_a_record_per_recording_with_windows_on_the_stride(opencs2):
    records = media_records(opencs2)
    result = build_manifest(records, opencs2, pixel_frames=PIXEL_FRAMES, stride=2, workers=1)
    assert result.refused == [] and result.dropped_windows == {}
    # a window ends inside the video: starts 0, 2, ..., 30
    starts = list(range(0, FRAMES - SPAN + 1, 2))
    assert [record["media_id"] for record in result.records] == [s.media_id for s in SEATS]
    for record, seat in zip(result.records, SEATS):
        ticks, video = opencs2 / seat.directory / "ticks.parquet", opencs2 / seat.directory
        assert record == {
            "media_id": seat.media_id,
            "match_id": seat.match_id,
            "map_name": seat.map_name,
            "round": seat.round,
            "player_slot": seat.slot,
            "video_path": f"{seat.directory}/video.mp4",
            "ticks_path": f"{seat.directory}/ticks.parquet",
            "video_frames": FRAMES,
            "video_file_size": (video / "video.mp4").stat().st_size,
            "ticks_file_size": ticks.stat().st_size,
            "ticks_sha256": hashlib.sha256(ticks.read_bytes()).hexdigest(),
            "ticks_rows": 2 * FRAMES,
            "fps": 32.0,
            "pixel_frames": PIXEL_FRAMES,
            "skip_frame": 2,
            "stride": 2,
            "max_tick_gap_seconds": 1.5 / 64.0,
            "start_frames": starts,
            "n_windows": len(starts),
        }
    coarse = build_manifest(records, opencs2, pixel_frames=PIXEL_FRAMES, stride=8)
    assert {tuple(r["start_frames"]) for r in coarse.records} == {(0, 8, 16, 24)}
    assert all(r["stride"] == 8 and r["n_windows"] == 4 for r in coarse.records)


def test_the_windows_of_stage_1_and_stage_1_long_start_on_the_latent_grid(tmp_path):
    seat = Seat(11, "de_dust2", 1, 0, frames=330)
    write_opencs2(tmp_path, (seat,))
    records = media_records(tmp_path, ["de_dust2"])
    stage1 = build_manifest(records, tmp_path, pixel_frames=stage_pixel_frames("1")).records[0]
    # 81 video frames span 161 source frames: starts 0, 8, ..., 168 of 330
    assert stage1["pixel_frames"] == 81 and stage1["skip_frame"] == 2 and stage1["stride"] == 8
    assert stage1["start_frames"] == list(range(0, 330 - 161 + 1, 8)) and stage1["n_windows"] == 22
    long = build_manifest(records, tmp_path, pixel_frames=stage_pixel_frames("1_long")).records[0]
    # 161 video frames span 321 source frames: starts 0 and 8
    assert long["pixel_frames"] == 161 and long["start_frames"] == [0, 8]
    short = Seat(11, "de_dust2", 1, 1, frames=100)
    write_opencs2(tmp_path / "short", (short,))
    with pytest.raises(ValueError, match=r"no recording has a window of 81 video frames:\n"):
        build_manifest(
            media_records(tmp_path / "short", ["de_dust2"]), tmp_path / "short", pixel_frames=81
        )


def holds_row(row: int, start: int) -> bool:
    """Whether the tick row is among the ticks the window at ``start`` owns: the ticks after
    source frame ``start - 2`` up to source frame ``start + 8`` (two ticks per source frame)."""
    return 2 * (start - 2) < row <= 2 * (start + 8)


def test_a_window_is_kept_when_the_stage_1_reader_aligns_its_ticks(tmp_path):
    seats = (
        Seat(11, "de_dust2", 1, 0),
        Seat(11, "de_dust2", 1, 1, rows=50),  # the ticks end at frame 24
        Seat(11, "de_dust2", 1, 2, missing=tuple(range(30, 40))),  # a gap of 11 ticks
        Seat(11, "de_dust2", 1, 3, buttons=((20, "bogus"),)),  # a button OpenCS2 does not list
        Seat(11, "de_dust2", 1, 4, buttons=((20, "turn_left"), (60, "use"))),  # a keyboard turn
    )
    write_opencs2(tmp_path, seats)
    result = build_manifest(
        media_records(tmp_path, ["de_dust2"]), tmp_path, pixel_frames=PIXEL_FRAMES, stride=2
    )
    candidates = list(range(0, FRAMES - SPAN + 1, 2))
    without_row_20 = [start for start in candidates if not holds_row(20, start)]
    kept, records = {}, {}
    for record in result.records:
        slot = record["player_slot"]
        kept[slot], records[slot] = record["start_frames"], record
    assert [kept[0], kept[4], kept[3]] == [candidates, without_row_20, without_row_20]
    assert kept[1] == list(range(0, 17, 2))
    assert 0 < len(kept[2]) < len(candidates) and 0 < len(without_row_20) < len(candidates)
    # the stage's reader serves every window of the manifest and refuses what the ticks rule out;
    # it ignores the keyboard turn, which the manifest leaves out as the paper's do
    for slot in (0, 1, 2, 3):
        assert kept[slot] == [
            start for start in candidates if reader_accepts(tmp_path, records[slot], start)
        ]
    assert all(reader_accepts(tmp_path, records[4], start) for start in candidates)
    assert result.refused == []
    assert result.dropped_windows == {
        "buttons": 2 * (len(candidates) - len(without_row_20)),
        "ticks": 2 * len(candidates) - len(kept[1]) - len(kept[2]),
    }


def test_the_alive_gate_keeps_the_windows_in_which_the_player_lives_throughout(tmp_path):
    seats = (Seat(11, "de_dust2", 1, 0), Seat(11, "de_dust2", 1, 1, alive_rows=50))
    write_opencs2(tmp_path, seats)
    records = media_records(tmp_path, ["de_dust2"])
    kwargs = {"pixel_frames": PIXEL_FRAMES, "stride": 2}
    plain = build_manifest(records, tmp_path, **kwargs)
    gated = build_manifest(records, tmp_path, require_alive=True, **kwargs)
    candidates = list(range(0, FRAMES - SPAN + 1, 2))
    assert [r["start_frames"] for r in plain.records] == [candidates, candidates]
    # the player dies at tick row 50: a window owns the ticks of source frames start .. start + 8
    alive = [start for start in candidates if 2 * (start + 8) < 50]
    assert [r["start_frames"] for r in gated.records] == [candidates, alive]
    assert gated.dropped_windows == {"alive": len(candidates) - len(alive)}
    assert STAGE_REQUIRES_ALIVE == {"1": False, "1_long": True}


def _delete(path):
    path.unlink()


def _append_byte(path):
    with path.open("ab") as handle:
        handle.write(b"\0")


def _reverse_bytes(path):
    path.write_bytes(path.read_bytes()[::-1])


def _write_no_parquet(path):
    path.write_bytes(b"no parquet")


def _zero_times(path):
    table = pq.ParquetFile(path).read()
    pq.write_table(table.set_column(1, "t", pa.array([0.0] * table.num_rows, pa.float32())), path)


#: reason, what to do to the tick table of the second recording, what its record says afterwards
TICKS_DAMAGE = [
    ("ticks_missing", _delete, lambda m: m, "not found"),
    ("ticks_mismatch", _append_byte, lambda m: m, r"has \d+ bytes, the media index says"),
    ("ticks_mismatch", _reverse_bytes, lambda m: m, "does not match the media index sha256"),
    ("ticks_unreadable", _write_no_parquet, "refresh", "cannot read the tick table"),
    ("ticks_time", _zero_times, "refresh", "strictly increasing"),
    (
        "ticks_rows",
        lambda path: None,
        lambda m: dataclasses.replace(m, ticks_rows=m.ticks_rows + 1),
        "tick table has 80 rows, expected 81",
    ),
]


@pytest.mark.parametrize("reason, damage, update, message", TICKS_DAMAGE)
def test_a_recording_with_a_tick_table_that_fails_a_check_is_left_out_and_counted(
    opencs2, reason, damage, update, message
):
    records = media_records(opencs2)
    damaged = records[1]
    damage(opencs2 / damaged.ticks_path)
    records[1] = refreshed(opencs2, damaged) if update == "refresh" else update(damaged)
    result = build_manifest(records, opencs2, pixel_frames=PIXEL_FRAMES, stride=2)
    assert [media_id for media_id, _ in result.refused] == [damaged.media_id]
    assert result.refused[0][1].reason == reason
    assert re.search(message, str(result.refused[0][1]))
    assert [r["media_id"] for r in result.records] == [
        m.media_id for m in records if m != records[1]
    ]


def test_a_recording_without_its_video_or_a_window_is_left_out(tmp_path):
    seats = (
        Seat(11, "de_dust2", 1, 0),
        Seat(11, "de_dust2", 1, 1),
        Seat(11, "de_dust2", 1, 2, rows=2),  # the ticks end before any window does
        Seat(11, "de_dust2", 1, 3, frames=8),  # shorter than a window
    )
    write_opencs2(tmp_path, seats)
    records = media_records(tmp_path, ["de_dust2"])
    (tmp_path / records[1].video_path).unlink()
    result = build_manifest(records, tmp_path, pixel_frames=PIXEL_FRAMES, stride=2)
    assert [r["media_id"] for r in result.records] == [records[0].media_id]
    reasons = {media_id: refusal.reason for media_id, refusal in result.refused}
    assert reasons == {
        records[1].media_id: "video_missing",
        records[2].media_id: "no_window",
        records[3].media_id: "short_video",
    }
    assert "a window needs 9" in str(result.refused[2][1])


def test_the_decode_scan_leaves_out_the_videos_that_do_not_decode(opencs2):
    records = media_records(opencs2)
    garbage, truncated = records[1], records[2]
    path = opencs2 / garbage.video_path
    path.write_bytes(b"\0" * path.stat().st_size)
    path = opencs2 / truncated.video_path
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    records[3] = dataclasses.replace(records[3], source_frames=FRAMES + 1)
    plain = build_manifest(records, opencs2, pixel_frames=PIXEL_FRAMES, stride=2)
    assert plain.refused == [] and len(plain.records) == len(SEATS)
    scanned = build_manifest(
        records, opencs2, pixel_frames=PIXEL_FRAMES, stride=2, decode_scan=True
    )
    reasons = {media_id: refusal.reason for media_id, refusal in scanned.refused}
    assert reasons == {
        garbage.media_id: "decode_error",
        truncated.media_id: "decode_error",
        records[3].media_id: "decode_frames",
    }
    assert f"decodes to {FRAMES} frames, the media index says {FRAMES + 1}" in str(
        scanned.refused[2][1]
    )
    assert [r["media_id"] for r in scanned.records] == [
        m.media_id for m in records if m.media_id not in reasons
    ]


def test_the_records_do_not_depend_on_the_workers(opencs2):
    records = media_records(opencs2)
    one = build_manifest(records, opencs2, pixel_frames=PIXEL_FRAMES, stride=2, workers=1)
    four = build_manifest(records[::-1], opencs2, pixel_frames=PIXEL_FRAMES, stride=2, workers=4)
    assert one == four


def test_the_stage_1_dataset_reads_every_window_of_the_manifest(opencs2, tmp_path):
    result = build_manifest(media_records(opencs2), opencs2, pixel_frames=PIXEL_FRAMES, stride=2)
    path = tmp_path / "manifest.jsonl"
    write_jsonl(path, result.records)
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == result.records
    assert all(set(json.loads(line)) == FIELDS for line in lines)
    spec = WindowSpec(2, max_tick_gap_seconds=TICK_GATE_SECONDS, camera_encoding="clip")
    dataset = RawVideoWindows(path, spec=spec, dataset_root=opencs2)
    assert (
        len(dataset) == sum(r["n_windows"] for r in result.records)
        and dataset.sample_weights is None
    )
    starts = [(r["media_id"], s) for r in result.records for s in r["start_frames"]]
    for index, (media_id, start) in enumerate(starts):
        item = dataset[index]
        assert item["metadata"] == {
            "media_id": media_id,
            "start_frame": start,
            "dataset_index": index,
        }
        assert item["frames"].shape == (3, PIXEL_FRAMES, 384, 672)
        assert item["buttons"].shape == (PIXEL_FRAMES, 11)


def _tool(*argv):
    return load_tool("prepare_data").main(["manifest", *argv])


def _refused(capsys, *argv) -> str:
    """What the tool says on stderr when it ends with a usage error (status 2) for ``argv``."""
    capsys.readouterr()
    with pytest.raises(SystemExit) as stop:
        _tool(*argv)
    assert stop.value.code == 2
    return capsys.readouterr().err


def _long_download(root) -> list[Seat]:
    seats = [Seat(11, "de_dust2", 1, 0, frames=170), Seat(12, "de_nuke", 1, 0, frames=170)]
    write_opencs2(root, seats)
    return seats


def test_the_tool_writes_the_manifest_of_a_stage(tmp_path, capsys):
    root = tmp_path / "opencs2"
    seats = _long_download(root)
    index, out = tmp_path / "media_index.jsonl", tmp_path / "stage1" / "train.jsonl"
    assert (
        load_tool("prepare_data").main(
            ["media-index", "--dataset-root", str(root), "--out", str(index), "--maps", *MAPS]
        )
        == 0
    )
    capsys.readouterr()
    base = ["--dataset-root", str(root), "--media-index", str(index), "--out", str(out)]
    assert _tool(*base) == 0
    assert capsys.readouterr().out == f"manifest: stage 1, 2 recordings, 4 windows -> {out}\n"
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["media_id"] for r in records] == [s.media_id for s in seats]
    # 81 video frames span 161 of 170 source frames: starts 0 and 8
    assert all(r["pixel_frames"] == 81 and r["start_frames"] == [0, 8] for r in records)
    assert all(r["stride"] == 8 and r["skip_frame"] == 2 and r["n_windows"] == 2 for r in records)

    only = tmp_path / "nuke.jsonl"
    assert _tool(*base[:-1], str(only), "--maps", "de_nuke") == 0
    assert [json.loads(line)["media_id"] for line in only.read_text().splitlines()] == [
        seats[1].media_id
    ]
    reserve = tmp_path / "reserve.txt"
    reserve.write_text(f"{seats[0].media_id}\n")
    left = tmp_path / "left.jsonl"
    assert _tool(*base[:-1], str(left), "--exclude-media", str(reserve)) == 0
    assert [json.loads(line)["media_id"] for line in left.read_text().splitlines()] == [
        seats[1].media_id
    ]
    capsys.readouterr()

    # stage 1_long needs 321 source frames: nothing is left, and the summary says why
    message = _refused(capsys, *base[:-1], str(tmp_path / "long.jsonl"), "--stage", "1_long")
    assert "no recording has a window of 161 video frames" in message
    assert not (tmp_path / "long.jsonl").exists()


def test_the_tool_lists_what_the_decode_scan_leaves_out(tmp_path, capsys):
    root = tmp_path / "opencs2"
    seats = _long_download(root)
    index, out, excluded = tmp_path / "index.jsonl", tmp_path / "m.jsonl", tmp_path / "left.txt"
    load_tool("prepare_data").main(
        ["media-index", "--dataset-root", str(root), "--out", str(index), "--maps", *MAPS]
    )
    video = root / seats[0].directory / "video.mp4"
    video.write_bytes(b"\0" * video.stat().st_size)
    capsys.readouterr()
    argv = ["--dataset-root", str(root), "--media-index", str(index), "--out", str(out)]
    assert _tool(*argv, "--decode-scan", "--excluded-out", str(excluded), "--workers", "2") == 0
    captured = capsys.readouterr()
    assert captured.out == f"manifest: stage 1, 1 recordings, 2 windows -> {out}\n"
    assert "left out 1 recordings" in captured.err and "decode_error: 1 (first: " in captured.err
    assert excluded.read_text() == f"{seats[0].media_id}\n"


def test_an_input_that_cannot_be_used_is_refused(opencs2, tmp_path, capsys):
    records = media_records(opencs2)
    with pytest.raises(ValueError, match="no video_path for 1 media, e.g. " + records[0].media_id):
        build_manifest(
            [dataclasses.replace(records[0], video_path=None), *records[1:]],
            opencs2,
            pixel_frames=PIXEL_FRAMES,
        )
    with pytest.raises(ValueError, match="stride and pixel_frames must be positive"):
        build_manifest(records, opencs2, pixel_frames=PIXEL_FRAMES, stride=0)
    index = tmp_path / "index.jsonl"
    base = [
        "--dataset-root",
        str(opencs2),
        "--media-index",
        str(index),
        "--out",
        str(tmp_path / "o"),
    ]
    assert "media index does not exist" in _refused(capsys, *base)
    write_jsonl(index, [{"media_id": "a"}])
    assert "lacks the field 'match_id'" in _refused(capsys, *base)
    write_jsonl(index, [])
    assert "no recording has a window" in _refused(capsys, *base)
    rows, _ = build_media_index(opencs2, MAPS)
    write_jsonl(index, rows)
    assert "has no media of ['de_inferno']" in _refused(capsys, *base, "--maps", "de_inferno")
    (tmp_path / "reserve.txt").write_text("a\na\n")
    reserve = str(tmp_path / "reserve.txt")
    assert "must list distinct media ids" in _refused(capsys, *base, "--exclude-media", reserve)
    # a tick table that changed since the index was written
    write_ticks(
        opencs2 / SEATS[0].directory / "ticks.parquet", dataclasses.replace(SEATS[0], rows=70)
    )
    assert "ticks_mismatch: 1 " in _refused(capsys, *base, "--stage", "1")


def test_the_tool_applies_the_stage_1_long_rules(tmp_path, capsys):
    root = tmp_path / "opencs2"
    # 330 frames hold 161-frame windows from 0 and 8; the player dies after the tick row 650, which
    # the second window owns
    seat = Seat(11, "de_dust2", 1, 0, frames=330, alive_rows=650)
    write_opencs2(root, (seat,))
    index = tmp_path / "media_index.jsonl"
    argv = ["--dataset-root", str(root), "--out", str(index), "--maps", "de_dust2"]
    load_tool("prepare_data").main(["media-index", *argv])
    capsys.readouterr()
    base = ["--dataset-root", str(root), "--media-index", str(index)]
    for stage, starts in (("1", list(range(0, 330 - 161 + 1, 8))), ("1_long", [0])):
        out = tmp_path / f"{stage}.jsonl"
        assert _tool(*base, "--out", str(out), "--stage", stage) == 0
        (record,) = [json.loads(line) for line in out.read_text().splitlines()]
        assert record["start_frames"] == starts and record["n_windows"] == len(starts)
    assert "left out 1 windows by alive" in capsys.readouterr().err
