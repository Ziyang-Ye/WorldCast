"""Picking a round start from a map: the library's seats as points on each map's overview, and the
seat nearest to a click.

A library offers many round starts per map, each with the spawns of its seats (x, y, z in engine
units). The page draws every seat as a dot on the overview of its map; a click is turned into the
point of the world under it and answered with the nearest seat, whose round start the room begins
from.

``overviews.json`` beside ``library.json`` says where each map's image sits on the world::

    {"overviews": {"de_nuke": {
        "origin": [-3453.0, 2887.0], "scale": 7.0, "size": [1024, 1024],
        "floors": [
            {"name": "Upper", "image": "overviews/de_nuke_upper.png", "z": [-520.0, 1e9]},
            {"name": "Lower", "image": "overviews/de_nuke_lower.png", "z": [-1e9, -520.0]}]}}}

``origin`` is the world x, y of the image's top-left corner and ``scale`` the engine units per
pixel (a radar's ``pos_x``, ``pos_y`` and ``scale``). A map with one image may write
``"image": "..."`` in place of ``floors``. A map without an entry gets a plain panel fitted to
its seats, so the demo needs no image.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .library import MAP_LABELS, Library

__all__ = ["Floor", "MapPicker", "Overview"]

#: The side of a plain panel, pixels, and its margin around the seats, engine units.
PANEL_SIZE, PANEL_MARGIN = 1024, 256.0
#: The file beside ``library.json`` that places the overview images.
OVERVIEWS_FILE = "overviews.json"
#: Decimals of a seat's position on the page: a ten-thousandth of the image.
POSITION_DECIMALS = 5


@dataclass(frozen=True)
class Floor:
    """One floor of a map: its image and the heights it holds.

    Attributes:
        name (str): its name on the page (empty for a map of one floor).
        image (str | None): the image, relative to the library; ``None``: a plain panel.
        z (tuple[float, float]): the heights from ``z[0]`` up to, not including, ``z[1]``, engine
            units.
    """

    name: str
    image: str | None
    z: tuple[float, float] = (-math.inf, math.inf)


@dataclass(frozen=True)
class Overview:
    """Where a map's image sits on the world, and its floors.

    Attributes:
        origin (tuple[float, float]): world x, y of the image's top-left corner, engine units.
        scale (float): engine units per pixel.
        size (tuple[int, int]): the image's width and height, pixels.
        floors (tuple[Floor, ...]): its floors; the first that holds a height is that height's.
    """

    origin: tuple[float, float]
    scale: float
    size: tuple[int, int]
    floors: tuple[Floor, ...]

    @classmethod
    def from_json(cls, record: dict) -> "Overview":
        """The overview of an ``overviews.json`` record; ``ValueError`` if its scale is not
        positive."""
        scale = float(record["scale"])
        if not scale > 0:
            raise ValueError(f"an overview's scale is engine units per pixel, not {scale}")
        floors = record.get("floors") or [{"name": "", "image": record.get("image")}]
        return cls(
            origin=(float(record["origin"][0]), float(record["origin"][1])),
            scale=scale,
            size=(int(record["size"][0]), int(record["size"][1])),
            floors=tuple(
                Floor(
                    name=floor.get("name", ""),
                    image=floor.get("image"),
                    z=tuple(float(v) for v in floor.get("z", (-math.inf, math.inf))),
                )
                for floor in floors
            ),
        )

    @classmethod
    def fitted(cls, xy: np.ndarray) -> "Overview":
        """A plain square panel that holds the points ``xy`` (``[n, 2]``) with a margin."""
        low, high = xy.min(axis=0) - PANEL_MARGIN, xy.max(axis=0) + PANEL_MARGIN
        side = float((high - low).max())
        centre = (low + high) / 2
        return cls(
            origin=(float(centre[0] - side / 2), float(centre[1] + side / 2)),
            scale=side / PANEL_SIZE,
            size=(PANEL_SIZE, PANEL_SIZE),
            floors=(Floor("", None),),
        )

    def to_unit(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The image positions of world points, as fractions of the image's width and height."""
        u = (x - self.origin[0]) / (self.scale * self.size[0])
        v = (self.origin[1] - y) / (self.scale * self.size[1])
        return u, v

    def to_world(self, u: float, v: float) -> tuple[float, float]:
        """The world x, y under the image position ``(u, v)``, fractions of its width and height."""
        return (
            self.origin[0] + u * self.scale * self.size[0],
            self.origin[1] - v * self.scale * self.size[1],
        )

    def floor_of(self, z: np.ndarray) -> np.ndarray:
        """The index of the floor that holds each height; the first floor for one none holds."""
        found = np.zeros(z.shape, dtype=int)
        for index in reversed(range(len(self.floors))):
            low, high = self.floors[index].z
            found = np.where((z >= low) & (z < high), index, found)
        return found


@dataclass
class _MapSeats:
    """The seats of one map: their round start and slot, position, heading and floor."""

    overview: Overview
    round_starts: list[str]
    slots: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    floor: np.ndarray


class MapPicker:
    """The seats of a library on the overviews of their maps.

    Args:
        library (Library): the round starts; every seat of every one is a point.
        overviews (Mapping[str, Overview] | None): the overview of a map by its name; a map
            without one gets a plain panel.
    """

    def __init__(self, library: Library, overviews: Mapping[str, Overview] | None = None) -> None:
        overviews = overviews or {}
        spawns: dict[str, list[tuple[str, int, tuple[float, ...]]]] = {}
        for round_start in library.round_starts.values():
            for seat in round_start.seats:
                spawns.setdefault(round_start.map, []).append(
                    (round_start.id, seat.slot, seat.spawn)
                )
        self.maps: dict[str, _MapSeats] = {}
        for name, seats in spawns.items():
            spawn = np.array([s for _, _, s in seats], dtype=float)
            overview = overviews.get(name) or Overview.fitted(spawn[:, :2])
            self.maps[name] = _MapSeats(
                overview=overview,
                round_starts=[round_start for round_start, _, _ in seats],
                slots=np.array([slot for _, slot, _ in seats]),
                xy=spawn[:, :2],
                yaw=spawn[:, 3],
                floor=overview.floor_of(spawn[:, 2]),
            )

    @classmethod
    def load(cls, library: Library) -> "MapPicker":
        """The picker of a library, with the overviews of the ``overviews.json`` beside it, if
        there is one."""
        overviews = {}
        if library.root is not None and (library.root / OVERVIEWS_FILE).is_file():
            text = (library.root / OVERVIEWS_FILE).read_text(encoding="utf-8")
            records = json.loads(text)["overviews"]
            overviews = {name: Overview.from_json(record) for name, record in records.items()}
        return cls(library, overviews)

    def public(self) -> list[dict]:
        """What the page reads: per map its label, image size and floors (``image``: a URL under
        ``/library/``, or ``None``), and its seats as ``[u, v, yaw, floor]`` (the position as
        fractions of the image, the heading in degrees)."""
        maps = []
        for name, seats in self.maps.items():
            u, v = seats.overview.to_unit(seats.xy[:, 0], seats.xy[:, 1])
            maps.append(
                {
                    "map": name,
                    "label": MAP_LABELS.get(name, name),
                    "size": list(seats.overview.size),
                    "floors": [
                        {
                            "name": floor.name,
                            "image": None if floor.image is None else f"/library/{floor.image}",
                        }
                        for floor in seats.overview.floors
                    ],
                    "seats": [
                        [
                            round(float(a), POSITION_DECIMALS),
                            round(float(b), POSITION_DECIMALS),
                            round(float(yaw), 1),
                            int(floor),
                        ]
                        for a, b, yaw, floor in zip(u, v, seats.yaw, seats.floor)
                    ],
                }
            )
        return maps

    def nearest(self, name: str, u: float, v: float, floor: int | None = None) -> tuple[str, int]:
        """The seat nearest to an image position.

        Args:
            name (str): the map.
            u (float), v (float): the position in the map's image, as fractions of its width and
                height.
            floor (int | None): only the seats of that floor; ``None``: all.

        Returns:
            tuple[str, int]: the id of the seat's round start and its slot.

        Raises:
            LookupError: the library has no such map, or no seat on that floor.
        """
        if name not in self.maps:
            raise LookupError(f"no map {name!r}")
        seats = self.maps[name]
        candidates = np.arange(len(seats.slots))
        if floor is not None:
            candidates = candidates[seats.floor == floor]
        if not len(candidates):
            raise LookupError(f"map {name!r} has no seat on floor {floor}")
        x, y = seats.overview.to_world(u, v)
        distance = np.hypot(seats.xy[candidates, 0] - x, seats.xy[candidates, 1] - y)
        index = candidates[int(np.argmin(distance))]
        return seats.round_starts[index], int(seats.slots[index])
