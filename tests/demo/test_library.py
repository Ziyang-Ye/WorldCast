"""The library of round starts: the synthetic arena, a ``library.json`` and its files."""

import json
from dataclasses import asdict

import pytest

import tests.engine.inference.support as sw
from tests.demo.support import write_labels
from worldcast.demo.library import (
    Library,
    RoundStart,
    players_in_first_frame,
    synthetic_library,
    zooming_weapons,
)

#: The source frame of the round start the labels below are read at.
START = 8


def test_the_synthetic_arena_needs_no_files():
    library = Library.load(None)
    arena = library.get("arena")
    assert library.public() == synthetic_library().public() and len(arena.seats) == 4
    assert arena.seat(2).team == "CT" and "clip" not in arena.seat(2).public()
    assert arena.public()["map_label"] == "Synthetic arena" and arena.match_id is None
    with pytest.raises(KeyError):
        arena.seat(7)
    with pytest.raises(KeyError):
        library.get("nowhere")
    with pytest.raises(FileNotFoundError):
        library.path("maps/arena.jpg")


def test_a_round_start_needs_a_seat():
    record = dict(id="r", map="de_nuke", label="Round", start_frame=0, seats=[])
    with pytest.raises(ValueError, match="no seats"):
        RoundStart.from_json(record)


def test_a_library_file_of_an_earlier_build_with_radar_images_loads_without_them(tmp_path):
    record = dict(
        id="nuke-r03",
        map="de_nuke",
        label="Round 3",
        start_frame=8,
        radar=dict(image="maps/de_nuke.webp", x0=-2000.0, y0=1500.0, cx=24, cy=24, scale=0.4),
        seats=[dict(slot=5, team="CT", spawn=[1, 2, 3, 4, 5], preview="nuke-r03/p05.jpg")],
    )
    path = tmp_path / "library.json"
    path.write_text(json.dumps({"round_starts": [record]}))
    round_start = Library.load(str(path)).get("nuke-r03")
    assert round_start.public() == {
        "id": "nuke-r03",
        "map_label": "Nuke",
        "label": "Round 3",
        "note": "",
        "cover": 5,
        "seats": [
            {
                "slot": 5,
                "team": "CT",
                "loadout": ("m4a1_silencer", "usp_silencer", "knife", "smokegrenade"),
                "preview": "nuke-r03/p05.jpg",
            }
        ],
    }
    saved = json.loads(Library.load(str(path)).save().read_text())
    assert "radar" not in saved["round_starts"][0]


def test_a_round_start_covers_with_one_of_its_seats():
    arena = synthetic_library().get("arena")
    record = json.loads(json.dumps(asdict(arena)))
    assert RoundStart.from_json(record).cover == 0
    assert RoundStart.from_json(dict(record, cover=3)).cover == 3
    with pytest.raises(ValueError, match=r"'arena': its cover, seat 7, is not a seat"):
        RoundStart.from_json(dict(record, cover=7))


def test_a_library_file_round_trips_and_keeps_to_its_directory(tmp_path):
    arena = synthetic_library().get("arena")
    saved = Library({arena.id: arena}, tmp_path).save()
    assert saved == tmp_path / "library.json"
    record = json.loads(saved.read_text())["round_starts"][0]
    assert record["seats"][2] == {
        "slot": 2,
        "team": "CT",
        "spawn": [600.0, -300.0, 0.0, 150.0, 0.0],
        "loadout": ["m4a1_silencer", "usp_silencer", "knife", "smokegrenade"],
        "preview": None,
        "clip": None,
    }
    library = Library.load(str(saved))
    assert library.get("arena") == arena and library.path("p00.jpg") == tmp_path / "p00.jpg"
    with pytest.raises(FileNotFoundError):
        library.path("../library.yaml")
    saved.write_text('{"round_starts": []}')
    with pytest.raises(ValueError, match="no round starts"):
        Library.load(str(saved))


def test_the_players_in_the_first_frame_are_those_visible_or_in_view_with_an_unknown_label(
    tmp_path,
):
    # at the round start: 2 and 4 visible, 6 in view with its label unknown, 7 behind the camera
    # (unknown, out of view), 3 and 1 visible a frame before and after, the rest occluded in view
    write_labels(
        tmp_path,
        5,
        {START: (2, 4), START - 1: (3,), START + 1: (1,)},
        unknown_at={START: (6,)},
        behind_at={START: (7,)},
    )
    media = sw.media_id(5)
    assert players_in_first_frame(tmp_path, media, 5, START) == [2, 4, 6]
    assert players_in_first_frame(tmp_path, media, 5, START + 2) == []  # nobody the frame after
    assert players_in_first_frame(tmp_path, media, 5, START - 1) == [3]
    # the slot given is the seat's own: it never counts
    assert players_in_first_frame(tmp_path, media, 4, START) == [2, 6]
    with pytest.raises(FileNotFoundError, match="no visibility labels for"):
        players_in_first_frame(tmp_path, sw.media_id(6), 6, START)


def test_the_weapons_that_zoom_are_found_by_the_names_the_model_reads():
    """Live play marks the scope unknown while a weapon of ``SCOPE_ZOOM_FOV`` is held, as the worker
    reads the weapon: ``AWP`` and ``weapon_awp`` are the AWP. A loadout without one holds none."""
    assert zooming_weapons(["ak47", "glock", "knife", "hegrenade"]) == []
    assert zooming_weapons(["AWP", "weapon_ssg08", " Aug ", "knife", "weapon_scar20"]) == [
        "awp",
        "ssg08",
        "aug",
        "scar20",
    ]
    assert zooming_weapons(["sg556", "g3sg1", "m4a1", "<none>", "not a weapon"]) == [
        "sg556",
        "g3sg1",
    ]
    assert zooming_weapons(()) == []
