"""The round index, the media index and the tick tables."""

import dataclasses
import json

import numpy as np
import pytest

from tests.data.support import media_record, tick_table
from worldcast.data import recordings
from worldcast.data.recordings import (
    MediaIndex,
    RoundIndexRow,
    load_round_index_row,
    player_rows,
    read_player_ticks,
    read_round_index,
    verify_ticks_file,
)

ROW = {
    "media_id": "1-de_x-r01-p03",
    "start_frame": 80,
    "match_id": 1,
    "round": 1,
    "map_name": "de_x",
    "latent_key": "win_000080",
}
MEDIA_ROW = {
    "media_id": "1-de_x-r01-p03",
    "match_id": 1,
    "map_name": "de_x",
    "round": 1,
    "player_slot": 3,
    "fps": 32.0,
    "video_frames": 3200,
    "ticks_path": "1/r01/p03/ticks.parquet",
    "ticks_rows": 6400,
    "ticks_file_size": 3,
}
#: sha256 of the three bytes ``abc``.
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_a_round_index_row_names_its_round_and_its_clients():
    alone = RoundIndexRow.from_row(ROW)
    assert alone.clients == () and alone.other_clients() == () and alone.round_seconds is None
    clients = ["1-de_x-r01-p00", "1-de_x-r01-p03"]
    row = RoundIndexRow.from_row(
        {**ROW, "player_slot": 3, "group_media": clients, "round_seconds": 87}
    )
    assert row.clients == tuple(clients) and row.other_clients() == ("1-de_x-r01-p00",)
    assert (row.media_id, row.start_frame, row.player_slot, row.round_seconds) == (
        "1-de_x-r01-p03",
        80,
        3,
        87,
    )
    with pytest.raises(ValueError, match="latent_key 'win_000000' does not encode start_frame 80"):
        RoundIndexRow.from_row({**ROW, "latent_key": "win_000000"})
    with pytest.raises(ValueError, match="negative start_frame"):
        RoundIndexRow.from_row({**ROW, "start_frame": -8})


def test_the_round_index_is_read_in_file_order(tmp_path):
    path = tmp_path / "round_index.jsonl"
    rows = [
        {**ROW, "media_id": f"1-de_x-r01-p0{slot}", "latent_key": "win_000080"} for slot in (3, 0)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n")
    assert [row.media_id for row in read_round_index(path)] == ["1-de_x-r01-p03", "1-de_x-r01-p00"]
    assert load_round_index_row(path, 1).media_id == "1-de_x-r01-p00"
    with pytest.raises(IndexError, match="row 2 outside the 2-row index"):
        load_round_index_row(path, 2)
    with pytest.raises(FileNotFoundError, match="round index does not exist"):
        read_round_index(tmp_path / "absent.jsonl")


def test_the_media_index_groups_a_rounds_recordings_by_slot(tmp_path):
    rows = [
        {**MEDIA_ROW, "capture_start_tick": 640},
        {**MEDIA_ROW, "media_id": "1-de_x-r01-p00", "player_slot": 0, "capture_start_tick": 640},
        {**MEDIA_ROW, "media_id": "1-de_x-r02-p03", "round": 2, "ticks_sha256": ABC_SHA256},
    ]
    path = tmp_path / "media_index.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    index = MediaIndex.load(path)
    media = index.media("1-de_x-r01-p03")
    assert (media.round_key, media.player_slot, media.source_frames) == ((1, "de_x", 1), 3, 3200)
    assert (media.fps, media.ticks_rows, media.ticks_sha256) == (32.0, 6400, None)
    assert index.media("1-de_x-r02-p03").ticks_sha256 == ABC_SHA256
    assert {slot: m.media_id for slot, m in index.slots((1, "de_x", 1)).items()} == {
        3: "1-de_x-r01-p03",
        0: "1-de_x-r01-p00",
    }
    with pytest.raises(KeyError, match="not in the media index"):
        index.media("1-de_x-r09-p00")
    path.write_text("".join(json.dumps(row) + "\n" for row in [rows[0], rows[0]]))
    with pytest.raises(ValueError, match="duplicate media_id"):
        MediaIndex.load(path)
    rows[1]["capture_start_tick"] = 641
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="misaligned capture start ticks"):
        MediaIndex.load(path).slots((1, "de_x", 1))


def test_a_tick_file_is_checked_by_size_and_content(tmp_path):
    media = recordings.MediaRecord.from_row(MEDIA_ROW)
    path = tmp_path / "ticks.parquet"
    path.write_bytes(b"abc")
    assert verify_ticks_file(media, path) == ABC_SHA256  # no recorded digest: the file's own
    recorded = dataclasses.replace(media, ticks_sha256=ABC_SHA256)
    assert verify_ticks_file(recorded, path) == ABC_SHA256
    path.write_bytes(b"abd")  # the same size, other bytes
    with pytest.raises(RuntimeError, match="does not match the media index sha256"):
        verify_ticks_file(recorded, path)
    path.write_bytes(b"abcd")
    with pytest.raises(RuntimeError, match="has 4 bytes, the media index says 3"):
        verify_ticks_file(recorded, path)
    path.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="malformed ticks_sha256"):
        verify_ticks_file(dataclasses.replace(media, ticks_sha256=ABC_SHA256.upper()), path)
    with pytest.raises(FileNotFoundError):
        verify_ticks_file(media, tmp_path / "absent.parquet")


def _columns(n: int = 64) -> dict[str, list]:
    """The columns of a tick parquet: a player rising 4 u per tick from tick 20 to 29."""
    z = np.zeros(n)
    z[20:30] = np.arange(1, 11) * 4.0 - 0.0976 * np.arange(1, 11) ** 2
    return {
        "tick": list(range(100, 100 + n)),
        "t": [i / 64.0 for i in range(n)],
        "x": [float(i) for i in range(n)],
        "y": [0.0] * n,
        "z": z.tolist(),
        "yaw": [0.0] * n,
        "pitch": [0.0] * n,
        "is_alive": [True] * n,
        "active": [["forward"] for _ in range(n)],
        "delta_pitch": [0.0] * n,
        "delta_yaw": [0.25] * n,
        "input_weapon": ["weapon_ak47"] * n,
        "team_num": [3] * n,
    }


def _read(monkeypatch, tmp_path, columns, **kwargs):
    """``read_player_ticks`` on a tick file whose parquet content is ``columns``."""
    (tmp_path / "ticks.parquet").write_bytes(b"abc")
    media = recordings.MediaRecord.from_row(
        {**MEDIA_ROW, "ticks_path": "ticks.parquet", "ticks_rows": len(columns["t"])}
    )
    monkeypatch.setattr(recordings, "read_tick_columns", lambda path, names, rows: columns)
    return read_player_ticks(media, tmp_path, **kwargs)


def test_a_tick_table_holds_the_columns_and_the_recalled_jump(monkeypatch, tmp_path):
    table = _read(monkeypatch, tmp_path, _columns())
    assert len(table) == 64 and table.t.dtype == np.float64 and table.team_num.dtype == np.int64
    assert table.x[5] == 5.0 and table.delta_yaw[5] == 0.25 and table.is_alive.all()
    jumps = ["jump" in row for row in table.active]
    assert any(jumps) and all(row[0] == "forward" for row in table.active)
    recorded = _read(monkeypatch, tmp_path, _columns(), jump_recall=False)
    assert recorded.active == _columns()["active"]  # the recorded buttons, as the state model reads


@pytest.mark.parametrize(
    "change, message",
    [
        ({"t": [i / 64.0 for i in range(1, 65)]}, "time base does not start at 0"),
        ({"t": [i / 32.0 for i in range(64)]}, "tick and time columns disagree"),
        ({"tick": [100] * 64}, "tick numbers are not strictly increasing"),
        ({"is_alive": [None] + [True] * 63}, "null is_alive"),
    ],
)
def test_a_tick_table_with_a_broken_time_base_is_refused(monkeypatch, tmp_path, change, message):
    with pytest.raises(RuntimeError, match=message):
        _read(monkeypatch, tmp_path, {**_columns(), **change})


def test_player_rows_are_the_last_tick_at_or_before_a_source_frame():
    walker, dead = tick_table(seconds=1.0), tick_table(seconds=1.0, dies_at=0.1, y=500.0)
    assert walker.states([10]).dtype == np.float32
    # source frame 5 of a 32 fps recording is 10 / 64 s: tick row 10
    rows = player_rows({0: walker, 3: dead}, 5, 32.0)
    assert rows.shape == (10, 6) and rows.dtype == np.float32
    assert rows[0].tolist() == [39.0625, 0.0, 0.0, 4.6875, 0.0, 1.0]
    assert rows[3].tolist() == [39.0625, 500.0, 0.0, 4.6875, 0.0, 0.0]  # dead since 0.1 s
    assert not rows[[1, 2, 4, 5, 6, 7, 8, 9]].any()  # slots without a recording
    # past the last tick a player keeps its last row
    assert player_rows({0: walker}, 400, 32.0)[0, 0] == pytest.approx(250.0 * 63 / 64)
    assert media_record(walker).source_frames == 32
