"""The map picker: the library's seats on a map's overview, and a click as the nearest seat.

Run: ``python -m pytest tests/demo/test_map_picker.py -q`` (needs ``pip install -e ".[test]"``).
"""

import json

import numpy as np
import pytest

from worldcast.demo.library import Library, RoundStart, Seat, synthetic_library
from worldcast.demo.map_picker import Floor, MapPicker, Overview

TWO_FLOORS = Overview(
    origin=(-1000.0, 1000.0),
    scale=2.0,
    size=(1000, 1000),
    floors=(Floor("upper", "o/up.png", (-500.0, 1e9)), Floor("lower", "o/low.png", (-1e9, -500.0))),
)


def seat(slot: int, x: float, y: float, z: float = 0.0, yaw: float = 0.0) -> Seat:
    return Seat(slot=slot, team="T", spawn=(x, y, z, yaw, 0.0))


def round_start(name: str, *seats: Seat, map: str = "de_nuke") -> RoundStart:
    return RoundStart(id=name, map=map, label=name, start_frame=0, seats=seats)


def library(*round_starts: RoundStart, root=None) -> Library:
    return Library({r.id: r for r in round_starts}, root)


def click_on(picker: MapPicker, name: str, x: float, y: float) -> tuple[float, float]:
    """The image position of a world point of a map."""
    u, v = picker.maps[name].overview.to_unit(np.array(x), np.array(y))
    return float(u), float(v)


def test_a_world_point_and_its_image_position_map_back_and_forth():
    overview = TWO_FLOORS
    assert overview.to_unit(np.array(-1000.0), np.array(1000.0)) == (0.0, 0.0)  # top left
    u, v = overview.to_unit(np.array(1000.0), np.array(-1000.0))
    assert (float(u), float(v)) == (1.0, 1.0)  # bottom right: 2 units per pixel, 1000 pixels
    x, y = overview.to_world(0.25, 0.75)
    assert (x, y) == (-500.0, -500.0)
    back = overview.to_unit(np.array(x), np.array(y))
    assert (float(back[0]), float(back[1])) == (0.25, 0.75)


def test_a_plain_panel_holds_every_seat():
    picker = MapPicker(synthetic_library())
    (arena,) = picker.public()
    assert arena["map"] == "arena" and arena["size"] == [1024, 1024]
    assert arena["floors"] == [{"name": "", "image": None}]
    assert len(arena["seats"]) == 4
    assert all(0.0 < u < 1.0 and 0.0 < v < 1.0 for u, v, _, _ in arena["seats"])


def test_the_nearest_seat_is_the_one_under_the_click():
    picker = MapPicker(synthetic_library())  # the seats of the arena: (+-600, +-300)
    for slot, (x, y) in enumerate([(-600, -300), (-600, 300), (600, -300), (600, 300)]):
        u, v = click_on(picker, "arena", x * 0.9, y * 0.9)
        assert picker.nearest("arena", u, v) == ("arena", slot)


def test_the_seats_of_every_round_start_of_a_map_compete():
    a = round_start("a", seat(0, 0, 0), seat(1, 100, 0))
    b = round_start("b", seat(0, 500, 500), seat(3, 600, 500))
    picker = MapPicker(library(a, b))
    assert picker.nearest("de_nuke", *click_on(picker, "de_nuke", 590, 490)) == ("b", 3)
    assert picker.nearest("de_nuke", *click_on(picker, "de_nuke", 90, 10)) == ("a", 1)
    assert picker.nearest("de_nuke", *click_on(picker, "de_nuke", 480, 520)) == ("b", 0)


def test_a_click_on_a_floor_picks_among_its_seats():
    start = round_start("a", seat(0, 0, 0, z=0.0), seat(1, 300, 0, z=-800.0))
    picker = MapPicker(library(start), {"de_nuke": TWO_FLOORS})
    u, v = click_on(picker, "de_nuke", 10, 0)  # beside the upper seat, above the lower
    assert picker.nearest("de_nuke", u, v) == ("a", 0)
    assert picker.nearest("de_nuke", u, v, floor=0) == ("a", 0)
    assert picker.nearest("de_nuke", u, v, floor=1) == ("a", 1)
    assert [floor for *_, floor in picker.public()[0]["seats"]] == [0, 1]


def test_a_map_or_a_floor_without_seats_is_not_found():
    picker = MapPicker(library(round_start("a", seat(0, 0, 0))), {"de_nuke": TWO_FLOORS})
    with pytest.raises(LookupError, match="no map"):
        picker.nearest("de_dust2", 0.5, 0.5)
    with pytest.raises(LookupError, match="floor 1"):
        picker.nearest("de_nuke", 0.5, 0.5, floor=1)


def test_the_overviews_file_places_the_images_and_other_maps_get_a_panel(tmp_path):
    records = {
        "de_nuke": {
            "origin": [-1000, 1000],
            "scale": 2.0,
            "size": [1000, 1000],
            "floors": [
                {"name": "upper", "image": "overviews/up.png", "z": [-500, 1e9]},
                {"name": "lower", "image": "overviews/low.png", "z": [-1e9, -500]},
            ],
        },
        "de_dust2": {
            "origin": [0, 0],
            "scale": 4.0,
            "size": [512, 512],
            "image": "overviews/d.png",
        },
    }
    (tmp_path / "overviews.json").write_text(json.dumps({"overviews": records}))
    starts = [
        round_start("n", seat(0, 0, 0), seat(1, 0, 0, z=-800.0)),
        round_start("d", seat(0, 100, -100), map="de_dust2"),
        round_start("m", seat(0, 5, 5), map="de_mirage"),
    ]
    maps = {m["map"]: m for m in MapPicker.load(library(*starts, root=tmp_path)).public()}
    assert [floor["image"] for floor in maps["de_nuke"]["floors"]] == [
        "/library/overviews/up.png",
        "/library/overviews/low.png",
    ]
    assert maps["de_dust2"]["floors"] == [{"name": "", "image": "/library/overviews/d.png"}]
    assert maps["de_dust2"]["seats"] == [[0.04883, 0.04883, 0.0, 0]]  # (100, -100) / (4 * 512)
    assert maps["de_mirage"]["floors"] == [{"name": "", "image": None}]
    assert [seat_[3] for seat_ in maps["de_nuke"]["seats"]] == [0, 1]


def test_a_library_without_an_overviews_file_gets_panels(tmp_path):
    picker = MapPicker.load(library(round_start("a", seat(0, 0, 0)), root=tmp_path))
    assert picker.public()[0]["floors"] == [{"name": "", "image": None}]


def test_an_overview_needs_a_positive_scale():
    with pytest.raises(ValueError, match="scale"):
        Overview.from_json({"origin": [0, 0], "scale": 0, "size": [8, 8]})
