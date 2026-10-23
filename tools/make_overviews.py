#!/usr/bin/env python3
"""Draw the maps' top-down overviews for the demo's lobby, and place them on the world.

The overview of a map is the floor plan the lobby starts rooms from, and ``overviews.json`` says
where it sits on the world (:mod:`worldcast.demo.map_picker`). It is drawn from the map's
navigation mesh, which the user exports from their own copy of the game's map files (the
``navmesh_hull_0`` mesh of ``de_<map>_navigation.glb``, in engine units): every triangle is filled,
the gaps the mesh leaves between its pieces are closed, and the floor plan gets a darker outline.
The image holds one floor per file; a map with a floor above another (``--split de_nuke=-520``: the
heights from -520 up are the upper floor) gets one image per floor, drawn on the same grid. No
Valve file is distributed with WorldCast, and this tool writes nothing but the images it draws.

Writes ``<out-dir>/overviews/<map>[_<floor>].png`` and ``<out-dir>/overviews.json``; pass the
directory of a demo library as ``--out-dir`` and the lobby finds them. It needs trimesh
(``pip install -e ".[train]"``, or ``pip install trimesh``) and Pillow (``.[demo]``).

Example::

    python tools/make_overviews.py --out-dir demo-library --split de_nuke=-520 \\
        --mesh de_mirage=meshes/de_mirage_navigation.glb \\
        --mesh de_nuke=meshes/de_nuke_navigation.glb
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: The mesh of a navigation file that holds the floors (its ladders are left out).
MESH_NAME = "navmesh_hull_0"
#: Engine units per pixel, and the margin around the mesh, engine units.
UNITS_PER_PIXEL, MARGIN = 4220 / 1120, 120.0
#: Pixels the gaps between triangles are closed over, and the outline's width.
CLOSE, OUTLINE = 5, 13
#: The floor plan's fill and outline, RGBA: a light plan with a darker edge, on nothing.
FILL, EDGE = (222, 226, 224, 255), (99, 110, 115, 255)
#: What a floor is called when a map has two (the first holds the heights up from the split).
FLOORS = ("Upper", "Lower")


def parse_pairs(items: list[str], what: str, kind=str) -> dict:
    """``--mesh a=b`` pairs as a dict; ``ValueError`` for an item that is not ``name=value``."""
    pairs = {}
    for item in items:
        name, _, value = item.partition("=")
        if not name or not value:
            raise ValueError(f"{what} takes name=value, not {item!r}")
        pairs[name] = kind(value)
    return pairs


def triangles(path: str | Path) -> np.ndarray:
    """The triangles ``[n, 3, 3]`` (x, y, z; engine units) of a navigation file's floors."""
    import trimesh

    scene = trimesh.load(str(path), force="scene")
    if MESH_NAME not in scene.geometry:
        raise ValueError(f"{path} has no {MESH_NAME} (it holds {sorted(scene.geometry)})")
    mesh = scene.geometry[MESH_NAME]
    return np.asarray(mesh.vertices)[np.asarray(mesh.faces)]


def grid(tris: np.ndarray) -> tuple[tuple[float, float], tuple[int, int]]:
    """The grid every floor of a map is drawn on: the world x, y of the top-left corner and the
    width and height in pixels."""
    low, high = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    width = int(np.ceil((high[0] - low[0] + 2 * MARGIN) / UNITS_PER_PIXEL))
    height = int(np.ceil((high[1] - low[1] + 2 * MARGIN) / UNITS_PER_PIXEL))
    centre = (low + high) / 2
    origin = (
        float(centre[0] - width * UNITS_PER_PIXEL / 2),
        float(centre[1] + height * UNITS_PER_PIXEL / 2),
    )
    return origin, (width, height)


def draw(tris: np.ndarray, origin: tuple[float, float], size: tuple[int, int]):
    """One floor's plan as an RGBA image on a grid."""
    from PIL import Image, ImageDraw, ImageFilter

    def pixel(vertex):
        return (
            (vertex[0] - origin[0]) / UNITS_PER_PIXEL,
            (origin[1] - vertex[1]) / UNITS_PER_PIXEL,
        )

    mask = Image.new("L", size)
    pen = ImageDraw.Draw(mask)
    for triangle in tris:
        pen.polygon([pixel(vertex) for vertex in triangle], fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(CLOSE)).filter(ImageFilter.MinFilter(CLOSE))
    edge = mask.filter(ImageFilter.MaxFilter(OUTLINE))
    image = Image.new("RGBA", size, (255, 255, 255, 0))
    image.paste(EDGE, (0, 0, *size), edge)
    image.paste(FILL, (0, 0, *size), mask)
    return image


def floors(
    tris: np.ndarray, split: float | None
) -> list[tuple[str, np.ndarray, tuple[float, float]]]:
    """The floors of a map: each one's name, its triangles and the heights it holds (from, up to).
    A triangle belongs to the floor its centre is on."""
    if split is None:
        return [("", tris, (-np.inf, np.inf))]
    height = tris[:, :, 2].mean(axis=1)
    return [
        (FLOORS[0], tris[height >= split], (split, np.inf)),
        (FLOORS[1], tris[height < split], (-np.inf, split)),
    ]


def make_overviews(
    meshes: dict[str, str | Path], out_dir: str | Path, split: dict[str, float] | None = None
) -> dict:
    """Draw the overviews of the maps and write ``overviews.json``; returns its content."""
    out = Path(out_dir)
    (out / "overviews").mkdir(parents=True, exist_ok=True)
    records = {}
    for name, path in meshes.items():
        tris = triangles(path)
        origin, size = grid(tris)
        drawn = []
        for floor, part, (low, high) in floors(tris, (split or {}).get(name)):
            if not len(part):
                raise ValueError(f"{name}: no floor lies at the heights {low}..{high}")
            file = f"overviews/{name}{'_' + floor.lower() if floor else ''}.png"
            draw(part, origin, size).save(out / file)
            drawn.append({"name": floor, "image": file, "z": [_finite(low), _finite(high)]})
        records[name] = {
            "origin": list(origin),
            "scale": UNITS_PER_PIXEL,
            "size": list(size),
            "floors": drawn,
        }
    content = {"overviews": records}
    (out / "overviews.json").write_text(json.dumps(content, indent=1), encoding="utf-8")
    return content


def _finite(value: float) -> float:
    """A bound as JSON holds it: infinity as 1e9 (the picker's heights are engine units)."""
    return float(np.clip(value, -1e9, 1e9))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True, help="a demo library's directory")
    parser.add_argument(
        "--mesh", action="append", default=[], metavar="MAP=FILE", help="a map's navigation .glb"
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="MAP=Z",
        help="a map with two floors: the height they are told apart at",
    )
    args = parser.parse_args(argv)
    try:
        meshes = parse_pairs(args.mesh, "--mesh")
        splits = parse_pairs(args.split, "--split", float)
    except ValueError as error:
        parser.error(str(error))
    if not meshes:
        parser.error("name at least one --mesh MAP=FILE")
    if unknown := sorted(set(splits) - set(meshes)):
        parser.error(f"--split names maps without a --mesh: {', '.join(unknown)}")
    content = make_overviews(meshes, args.out_dir, splits)
    for name, record in content["overviews"].items():
        names = ", ".join(floor["name"] or "one floor" for floor in record["floors"])
        print(f"{name}: {record['size'][0]} x {record['size'][1]} px ({names})")
    print(f"wrote {Path(args.out_dir) / 'overviews.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
