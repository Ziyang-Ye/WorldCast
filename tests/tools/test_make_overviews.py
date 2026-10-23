"""``tools/make_overviews.py``: a floor plan drawn from a navigation mesh, placed on the world by
``overviews.json``, one image per floor of a map that has two.

Run: ``python -m pytest tests/tools/test_make_overviews.py -q`` (needs ``pip install -e ".[train]"``
for trimesh and ``.[demo]`` for Pillow).
"""

import json

import numpy as np
import pytest

from tests.tools.support import load_tool
from worldcast.demo.library import Library, RoundStart, Seat
from worldcast.demo.map_picker import MapPicker, Overview

trimesh = pytest.importorskip("trimesh")
Image = pytest.importorskip("PIL.Image")


def navigation_file(path, floors):
    """A navigation .glb whose ``navmesh_hull_0`` holds a square (side 400) at each height of
    ``floors`` ((x, y) of its low corner, height)."""
    vertices, faces = [], []
    for x, y, z in floors:
        base = len(vertices)
        vertices += [(x, y, z), (x + 400, y, z), (x + 400, y + 400, z), (x, y + 400, z)]
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    mesh = trimesh.Trimesh(np.array(vertices, float), np.array(faces))
    mesh.name = "navmesh_hull_0"
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name=mesh.name, node_name=mesh.name)
    scene.export(str(path))
    return path


def test_a_map_of_one_floor_gets_one_image_placed_on_the_world(tmp_path):
    mesh = navigation_file(tmp_path / "nav.glb", [(0, 0, 0)])
    out = tmp_path / "library"
    assert load_tool("make_overviews").main(["--out-dir", str(out), "--mesh", f"de_x={mesh}"]) == 0
    record = json.loads((out / "overviews.json").read_text())["overviews"]["de_x"]
    assert [floor["image"] for floor in record["floors"]] == ["overviews/de_x.png"]
    image = Image.open(out / "overviews" / "de_x.png")
    assert list(image.size) == record["size"] and image.mode == "RGBA"
    overview = Overview.from_json(record)
    # the square's centre, (200, 200), is the floor plan's fill; a corner of the image is empty
    u, v = overview.to_unit(np.array(200.0), np.array(200.0))
    assert image.getpixel((int(u * image.width), int(v * image.height)))[3] == 255
    assert image.getpixel((2, 2))[3] == 0
    # and the square's corners land where the image says: its left edge, 120 units in
    left, _ = overview.to_unit(np.array(0.0), np.array(200.0))
    assert left * image.width == pytest.approx(120 / record["scale"], abs=2)


def test_a_map_with_two_floors_gets_an_image_for_each_on_one_grid(tmp_path):
    mesh = navigation_file(tmp_path / "nav.glb", [(0, 0, 0), (600, 0, -800)])
    out = tmp_path / "library"
    load_tool("make_overviews").main(
        ["--out-dir", str(out), "--mesh", f"de_x={mesh}", "--split", "de_x=-500"]
    )
    record = json.loads((out / "overviews.json").read_text())["overviews"]["de_x"]
    assert [(floor["name"], floor["image"]) for floor in record["floors"]] == [
        ("Upper", "overviews/de_x_upper.png"),
        ("Lower", "overviews/de_x_lower.png"),
    ]
    upper, lower = (Image.open(out / floor["image"]) for floor in record["floors"])
    assert upper.size == lower.size == tuple(record["size"])
    overview = Overview.from_json(record)

    def filled(image, x, y):
        u, v = overview.to_unit(np.array(float(x)), np.array(float(y)))
        return image.getpixel((int(u * image.width), int(v * image.height)))[3] == 255

    assert filled(upper, 200, 200) and not filled(lower, 200, 200)
    assert filled(lower, 800, 200) and not filled(upper, 800, 200)


def test_the_overviews_serve_the_map_picker_of_a_library(tmp_path):
    mesh = navigation_file(tmp_path / "nav.glb", [(0, 0, 0), (600, 0, -800)])
    load_tool("make_overviews").main(
        ["--out-dir", str(tmp_path), "--mesh", f"de_x={mesh}", "--split", "de_x=-500"]
    )
    seats = (Seat(0, "T", (100, 100, 0, 0, 0)), Seat(1, "T", (700, 100, -800, 0, 0)))
    library = Library({"a": RoundStart("a", "de_x", "A", 0, seats)}, tmp_path)
    (shown,) = MapPicker.load(library).public()
    assert [floor["image"] for floor in shown["floors"]] == [
        "/library/overviews/de_x_upper.png",
        "/library/overviews/de_x_lower.png",
    ]
    assert [seat[3] for seat in shown["seats"]] == [0, 1]
    assert all(0 < seat[0] < 1 and 0 < seat[1] < 1 for seat in shown["seats"])


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--mesh", "de_x"], "name=value"),
        ([], "at least one --mesh"),
        (["--mesh", "de_x=a.glb", "--split", "de_y=-5"], "without a --mesh: de_y"),
    ],
)
def test_the_arguments_are_checked(tmp_path, capsys, arguments, message):
    with pytest.raises(SystemExit) as stopped:
        load_tool("make_overviews").main(["--out-dir", str(tmp_path), *arguments])
    assert stopped.value.code == 2 and message in capsys.readouterr().err


def test_a_file_without_the_floors_mesh_is_refused(tmp_path):
    box = trimesh.creation.box()
    box.name = "something_else"
    scene = trimesh.Scene()
    scene.add_geometry(box, geom_name=box.name, node_name=box.name)
    scene.export(str(tmp_path / "nav.glb"))
    with pytest.raises(ValueError, match="navmesh_hull_0"):
        load_tool("make_overviews").make_overviews({"de_x": tmp_path / "nav.glb"}, tmp_path)
