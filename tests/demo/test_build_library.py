"""A library built from a manifest: seats from the tick tables or the manifest, clips and previews;
a seat of a recorded round offered only if the GT visibility labels show no other player visible,
or in view with an unknown label, in its first frame, and no seat whose loadout holds a weapon
that zooms (live play does not track the scope); what was left out logged, and a manifest that
leaves nothing refused."""

import dataclasses
import logging
from pathlib import Path

import pytest
import yaml

import tests.engine.inference.support as sw
from tests.demo.support import write_labels, write_video
from worldcast.demo import build_library as builder
from worldcast.demo.library import Library

#: The source frame the round start of the manifests below is at: ``video_start = 4`` at 16 fps.
START = 8


def test_a_seat_comes_from_its_tick_table(tmp_path):
    world = sw.make_world(tmp_path)
    ticks = world["tables"][sw.media_id(7)]
    seat = builder.seat_from_ticks(ticks, source_fps=32.0, start_frame=64)  # two seconds in
    i = 2 * 64
    pose = [ticks.x[i], ticks.y[i], ticks.z[i], ticks.yaw[i], ticks.pitch[i]]
    assert seat["team"] == "CT" and seat["spawn"] == pytest.approx(pose, abs=0.005)
    # the weapon held all round is the primary; the others are the team's defaults
    assert seat["loadout"] == ["ak47", "usp_silencer", "knife", "smokegrenade"]


def _manifest(tmp_path, monkeypatch, round_starts: list[dict], **keys) -> Path:
    """The manifest of ``round_starts`` on the synthetic round (every one of its players recorded,
    4 frames into the clips), a video of 48 frames, and ``keys`` among its own."""
    world = sw.make_world(tmp_path / "world")
    monkeypatch.setattr(
        builder, "read_player_ticks", lambda media, root: world["tables"][media.media_id]
    )
    write_video(tmp_path / "p05.mp4", frames=48)
    manifest = dict(clip_seconds=1, round_starts=round_starts, **keys)
    manifest.update(media_index=world["media_index"], dataset_root=world["dataset_root"])
    (tmp_path / "library.yaml").write_text(yaml.safe_dump(manifest))
    return tmp_path / "library.yaml"


def _start(id: str, *seats: dict, **fields) -> dict:
    start = dict(id=id, map=sw.MAP, label="Round 3", video_start=4)
    return dict(start, match_id=sw.MATCH, round=sw.ROUND, seats=list(seats), **fields)


def test_a_library_is_built_from_a_manifest_and_loaded(tmp_path, monkeypatch):
    recorded = dict(slot=5, video="p05.mp4")
    given = dict(slot=6, video="p05.mp4", team="T", spawn=[500, 500, 0, 90, 0])
    manifest = _manifest(
        tmp_path,
        monkeypatch,
        [_start("nuke-r03", recorded, given)],
        visibility_label_root="vis",
    )
    for slot in (5, 6):  # nobody in sight in either first frame
        write_labels(tmp_path / "vis", slot, {})

    library = Library.load(str(builder.build_library(manifest, tmp_path / "out")))
    round_start = library.get("nuke-r03")
    assert (round_start.start_frame, round_start.match_id, round_start.round) == (8, sw.MATCH, 3)
    assert round_start.cover == 5 and round_start.seat(5).team == "CT"  # from its tick table
    assert round_start.seat(6).spawn == (500.0, 500.0, 0.0, 90.0, 0.0)  # from the manifest
    assert len(list(library.path(round_start.seat(5).clip).glob("*.jpg"))) == 16
    assert library.path(round_start.seat(5).preview).is_file()


def test_the_nav_mesh_an_earlier_manifest_names_is_ignored(tmp_path):
    write_video(tmp_path / "p00.mp4", frames=16)
    seat = dict(slot=0, video="p00.mp4", team="CT", spawn=[0, 0, 0, 0, 0])
    start = dict(id="r", map="de_nuke", label="Round", video_start=0, nav="meshes/nuke.npz")
    manifest = dict(round_starts=[dict(start, seats=[seat])])
    (tmp_path / "library.yaml").write_text(yaml.safe_dump(manifest))
    out = tmp_path / "out"
    library = Library.load(str(builder.build_library(tmp_path / "library.yaml", out)))
    assert library.get("r").cover == 0
    assert sorted(path.name for path in out.iterdir()) == ["library.json", "r"]


def test_a_seat_without_a_recording_needs_its_team_and_spawn(tmp_path):
    write_video(tmp_path / "p00.mp4", frames=16)
    start = dict(id="r", map="de_nuke", label="Round", video_start=0)
    manifest = dict(round_starts=[dict(start, seats=[dict(slot=0, video="p00.mp4")])])
    (tmp_path / "library.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="team and spawn"):
        builder.build_library(tmp_path / "library.yaml", tmp_path / "out")


def test_a_seat_without_a_recording_has_no_labels_and_is_offered(tmp_path):
    write_video(tmp_path / "p00.mp4", frames=16)
    seat = dict(slot=0, video="p00.mp4", team="T", spawn=[1, 2, 3, 0, 0])
    start = dict(id="r", map="de_nuke", label="Round", video_start=0, seats=[seat])
    (tmp_path / "library.yaml").write_text(yaml.safe_dump(dict(round_starts=[start])))
    library = Library.load(str(builder.build_library(tmp_path / "library.yaml", tmp_path / "out")))
    assert [seat.slot for seat in library.get("r").seats] == [0]


def test_a_seat_is_offered_only_if_no_other_player_may_be_seen_in_its_first_frame(
    tmp_path, monkeypatch, caplog
):
    """Seat 5 sees players around its first frame but at none of it, and one behind the camera at
    it; seat 6 sees players 2 and 4 in it, seat 7 only a player in view whose label is unknown,
    seat 8 only players it sees occluded. The cover of a round start is its first seat that
    remains; a round start whose seats all see someone is left out."""
    seats = [dict(slot=slot, video="p05.mp4") for slot in (5, 6, 7, 8)]
    manifest = _manifest(
        tmp_path,
        monkeypatch,
        [_start("nuke-r03", *seats, cover=6), _start("nuke-r03-b", seats[1])],
        visibility_label_root="vis",
    )
    write_labels(
        tmp_path / "vis", 5, {START - 1: (2,), START + 1: (3, 4)}, behind_at={START: (0, 9)}
    )
    write_labels(tmp_path / "vis", 6, {START: (2, 4), START + 16: (1,)})
    write_labels(tmp_path / "vis", 7, {}, unknown_at={START: (0,)})
    write_labels(tmp_path / "vis", 8, {})

    with caplog.at_level(logging.INFO, logger=builder.__name__):
        out = builder.build_library(manifest, tmp_path / "out")
    library = Library.load(str(out))
    assert list(library.round_starts) == ["nuke-r03"]
    kept = library.get("nuke-r03")
    assert [seat.slot for seat in kept.seats] == [5, 8] and kept.cover == 5
    assert not (tmp_path / "out" / "nuke-r03" / "p06").exists()  # no clip of a seat left out
    assert [record.getMessage() for record in caplog.records] == [
        "nuke-r03, seat 6 left out: players 2, 4 may be seen in its first frame",
        "nuke-r03, seat 7 left out: players 0 may be seen in its first frame",
        "nuke-r03 offers 2 of 4 seats",
        "nuke-r03: seat 6 is not offered; seat 5 is its cover",
        "nuke-r03-b, seat 6 left out: players 2, 4 may be seen in its first frame",
        "nuke-r03-b left out: none of its 1 seats qualifies",
    ]


def test_a_seat_whose_loadout_holds_a_weapon_that_zooms_is_left_out(tmp_path, monkeypatch, caplog):
    """Seat 5 holds an AWP all round (its tick table makes it the primary), seat 6's manifest
    loadout holds an AUG spelled ``AUG``, seat 7's a weapon that does not zoom (an M4A4 and a
    grenade), seat 8 has the tick table's AK-47. Nobody is in sight in any first frame. The seats
    that zoom are left out with their weapons logged by the model's names, and get no clip; a
    round start whose only seat zooms (a SCAR-20 and an AWP, spelled ``weapon_scar20`` and
    ``AWP``: a worker reads both as the weapons they are) is left out, and a manifest with nothing
    else is refused."""
    seats = [
        dict(slot=5, video="p05.mp4"),
        dict(slot=6, video="p05.mp4", loadout=["ak47", "glock", "knife", "AUG"]),
        dict(slot=7, video="p05.mp4", loadout=["m4a1", "glock", "knife", "hegrenade"]),
        dict(slot=8, video="p05.mp4"),
    ]
    zooming = dict(slot=6, video="p05.mp4", loadout=["weapon_scar20", "glock", "knife", "AWP"])
    manifest = _manifest(
        tmp_path,
        monkeypatch,
        [_start("nuke-r03", *seats), _start("nuke-r03-b", zooming)],
        visibility_label_root="vis",
    )
    for slot in (5, 6, 7, 8):
        write_labels(tmp_path / "vis", slot, {})
    tables = builder.read_player_ticks

    def awp_for_seat_5(media, root):
        ticks = tables(media, root)
        if media.media_id != sw.media_id(5):
            return ticks
        return dataclasses.replace(ticks, input_weapon=["weapon_awp"] * len(ticks))

    monkeypatch.setattr(builder, "read_player_ticks", awp_for_seat_5)

    with caplog.at_level(logging.INFO, logger=builder.__name__):
        out = builder.build_library(manifest, tmp_path / "out")
    library = Library.load(str(out))
    assert list(library.round_starts) == ["nuke-r03"]
    kept = library.get("nuke-r03")
    assert [seat.slot for seat in kept.seats] == [7, 8] and kept.cover == 7
    assert kept.seat(7).loadout == ("m4a1", "glock", "knife", "hegrenade")
    assert kept.seat(8).loadout == ("ak47", "usp_silencer", "knife", "smokegrenade")
    assert not (tmp_path / "out" / "nuke-r03" / "p05").exists()  # no clip of a seat left out
    assert not (tmp_path / "out" / "nuke-r03-b").exists()
    zoom = "left out: its loadout holds %s, which zooms (live play does not track the scope)"
    assert [record.getMessage() for record in caplog.records] == [
        "nuke-r03, seat 5 " + zoom % "awp",
        "nuke-r03, seat 6 " + zoom % "aug",
        "nuke-r03 offers 2 of 4 seats",
        "nuke-r03-b, seat 6 " + zoom % "scar20, awp",
        "nuke-r03-b left out: none of its 1 seats qualifies",
    ]


def test_a_manifest_that_leaves_no_round_start_is_refused_and_writes_no_library(
    tmp_path, monkeypatch
):
    manifest = _manifest(
        tmp_path,
        monkeypatch,
        [_start("nuke-r03", dict(slot=5, video="p05.mp4"))],
        visibility_label_root="vis",
    )
    write_labels(tmp_path / "vis", 5, {START: (2,)})
    with pytest.raises(ValueError, match="no round start offers a seat whose first frame shows"):
        builder.build_library(manifest, tmp_path / "out")
    assert not (tmp_path / "out" / "library.json").exists()


def test_a_seat_of_a_recorded_round_needs_the_labels_to_be_offered(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, monkeypatch, [_start("nuke-r03", dict(slot=5, video="p05.mp4"))])
    with pytest.raises(
        ValueError, match="offered only if no other player may be seen in its first"
    ):
        builder.build_library(manifest, tmp_path / "out")
    keys = yaml.safe_load(manifest.read_text()) | {"visibility_label_root": "vis"}
    manifest.write_text(yaml.safe_dump(keys))
    with pytest.raises(FileNotFoundError, match="no visibility labels for"):
        builder.build_library(manifest, tmp_path / "out")


def test_a_clip_that_starts_past_its_video_is_refused(tmp_path):
    write_video(tmp_path / "p00.mp4", frames=16)
    with pytest.raises(ValueError, match="no frame from frame 64 on"):
        builder.extract_clip(tmp_path / "p00.mp4", tmp_path / "clip", start=64, frames=16)
