"""``tools/build_demo_library.py``: a manifest in, the path of ``library.json`` out; the log says
how many seats each round start offers and what the first-frame rule and the scope rule (a seat
whose loadout holds a weapon that zooms) left out; a manifest that leaves no round start ends with
a message and status 2."""

import json
import logging

import pytest
import yaml

import tests.engine.inference.support as sw
from tests.demo.support import write_labels, write_video
from tests.tools.support import load_tool


def test_the_tool_prints_the_path_of_the_library_it_builds(tmp_path, capsys):
    write_video(tmp_path / "p00.mp4", frames=16)
    seat = dict(slot=0, video="p00.mp4", team="T", spawn=[1, 2, 3, 0, 0])
    start = dict(id="r", map="de_nuke", label="Round", video_start=0, seats=[seat])
    (tmp_path / "library.yaml").write_text(yaml.safe_dump(dict(round_starts=[start])))
    out = tmp_path / "library"
    tool = load_tool("build_demo_library")
    assert tool.main(["--manifest", str(tmp_path / "library.yaml"), "--out-dir", str(out)]) == 0
    library = out.resolve() / "library.json"
    assert capsys.readouterr().out.strip() == str(library)
    assert [r["id"] for r in json.loads(library.read_text())["round_starts"]] == ["r"]


def _manifest(tmp_path, monkeypatch, labels: dict[int, dict]) -> list[str]:
    """The tool's arguments for a manifest of one round start of the synthetic round, offering
    the seats ``labels`` (``{slot: visible_at}``: whom each sees at the round start, source frame
    8, by its GT labels)."""
    world = sw.make_world(tmp_path / "world")
    monkeypatch.setattr(
        "worldcast.demo.build_library.read_player_ticks",
        lambda media, root: world["tables"][media.media_id],
    )
    write_video(tmp_path / "p05.mp4", frames=48)
    for slot, visible_at in labels.items():
        write_labels(tmp_path / "vis", slot, visible_at)
    seats = [dict(slot=slot, video="p05.mp4") for slot in labels]
    start = dict(id="nuke-r03", map=sw.MAP, label="Round 3", video_start=4)
    start.update(match_id=sw.MATCH, round=sw.ROUND, seats=seats)
    manifest = dict(round_starts=[start], visibility_label_root="vis")
    manifest.update(media_index=world["media_index"], dataset_root=world["dataset_root"])
    (tmp_path / "library.yaml").write_text(yaml.safe_dump(manifest))
    return ["--manifest", str(tmp_path / "library.yaml"), "--out-dir", str(tmp_path / "out")]


def test_the_tool_logs_the_seats_each_round_start_offers_and_those_it_leaves_out(
    tmp_path, monkeypatch, caplog
):
    args = _manifest(tmp_path, monkeypatch, {5: {8: (2, 4)}, 6: {}})
    with caplog.at_level(logging.INFO):
        assert load_tool("build_demo_library").main(args) == 0
    offered = json.loads((tmp_path / "out" / "library.json").read_text())["round_starts"][0][
        "seats"
    ]
    assert [seat["slot"] for seat in offered] == [6]
    assert [record.getMessage() for record in caplog.records] == [
        "nuke-r03, seat 5 left out: players 2, 4 may be seen in its first frame",
        "nuke-r03 offers 1 of 2 seats",
    ]


def test_the_tool_leaves_out_a_seat_whose_loadout_holds_a_weapon_that_zooms(
    tmp_path, monkeypatch, caplog
):
    """Nobody is in sight in either first frame; the manifest gives seat 5 an AWP."""
    args = _manifest(tmp_path, monkeypatch, {5: {}, 6: {}})
    keys = yaml.safe_load((tmp_path / "library.yaml").read_text())
    keys["round_starts"][0]["seats"][0]["loadout"] = ["awp", "glock", "knife", "hegrenade"]
    (tmp_path / "library.yaml").write_text(yaml.safe_dump(keys))
    with caplog.at_level(logging.INFO):
        assert load_tool("build_demo_library").main(args) == 0
    offered = json.loads((tmp_path / "out" / "library.json").read_text())["round_starts"][0][
        "seats"
    ]
    assert [seat["slot"] for seat in offered] == [6]
    assert [record.getMessage() for record in caplog.records] == [
        "nuke-r03, seat 5 left out: its loadout holds awp, which zooms (live play does not track"
        " the scope)",
        "nuke-r03 offers 1 of 2 seats",
    ]


def test_the_tool_stops_with_a_message_when_every_round_start_is_left_out(
    tmp_path, monkeypatch, caplog, capsys
):
    """The only seat of the round start sees players 2 and 4 at the round start."""
    args = _manifest(tmp_path, monkeypatch, {5: {8: (2, 4)}})
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as stopped:
        load_tool("build_demo_library").main(args)
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "no round start offers a seat whose first frame shows nobody and whose loadout" in error
    assert [record.getMessage() for record in caplog.records] == [
        "nuke-r03, seat 5 left out: players 2, 4 may be seen in its first frame",
        "nuke-r03 left out: none of its 1 seats qualifies",
    ]
    assert not (tmp_path / "out" / "library.json").exists()
