"""The round starts a room can begin from: maps, rounds and their seats (``library.json``).

``library.json`` is ``{"round_starts": [...]}``, each round start the fields of :class:`RoundStart`
(its ``seats`` those of :class:`Seat`). Image paths are relative to the library file and
served to browsers under ``/library/``. Without a library the demo offers one synthetic arena
that needs no files. ``tools/build_demo_library.py`` builds a library from rendered videos
(:mod:`worldcast.demo.build_library`), which offers a seat of a recorded round only if the
first frame of its client shows no other player (:func:`players_in_first_frame`): live play
shows only the players a human plays, and every window of a block attends that frame.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from worldcast.data.labels import frustum_rows, visibility_rows

__all__ = [
    "DEFAULT_LOADOUT",
    "PLAYER_COLORS",
    "Library",
    "RoundStart",
    "Seat",
    "players_in_first_frame",
    "synthetic_library",
    "zooming_weapons",
]

#: Display names of the paper's maps.
MAP_LABELS = {
    "de_ancient": "Ancient",
    "de_dust2": "Dust II",
    "de_mirage": "Mirage",
    "de_nuke": "Nuke",
    "arena": "Synthetic arena",
}
#: Weapon keys 1-4: primary, secondary, knife, grenade.
DEFAULT_LOADOUT = {
    "T": ("ak47", "glock", "knife", "hegrenade"),
    "CT": ("m4a1_silencer", "usp_silencer", "knife", "smokegrenade"),
}
#: The colour of each player slot on the page and in the mock client's view (the project page's
#: client colours, then Apple system colours).
PLAYER_COLORS = (
    "#E67F27",
    "#3D94E8",
    "#3FB576",
    "#BF5AF2",
    "#FF375F",
    "#64D2FF",
    "#FFD60A",
    "#AC8E68",
    "#5E5CE6",
    "#30D158",
)


@dataclass(frozen=True)
class Seat:
    """A seat a person can take: one player of a round start, where it spawns and what it holds.

    Attributes:
        slot (int): the player slot of the recorded round (0-9).
        team (str): ``T`` or ``CT``.
        spawn (tuple[float, ...]): x, y, z (feet, engine units), yaw and pitch (degrees) at the
            start.
        loadout (tuple[str, ...]): the weapons of keys 1-4.
        preview (str | None): the player's first frame, an image of the library.
        clip (str | None): the mock client's frames, a directory of JPEGs of the library.
    """

    slot: int
    team: str
    spawn: tuple[float, float, float, float, float]
    loadout: tuple[str, ...] = ()
    preview: str | None = None
    clip: str | None = None

    @classmethod
    def from_json(cls, record: dict) -> "Seat":
        """The seat of a ``library.json`` record; without a ``loadout``, its team's default."""
        return cls(
            slot=int(record["slot"]),
            team=record["team"],
            spawn=tuple(float(v) for v in record["spawn"]),
            loadout=tuple(record.get("loadout") or DEFAULT_LOADOUT[record["team"]]),
            preview=record.get("preview"),
            clip=record.get("clip"),
        )

    def public(self) -> dict:
        """What the page reads of a seat."""
        return {k: getattr(self, k) for k in ("slot", "team", "loadout", "preview")}


@dataclass(frozen=True)
class RoundStart:
    """A round start a room can begin from.

    Attributes:
        id (str): its name in the library.
        map (str): the map, e.g. ``de_nuke``.
        label (str): its title in the lobby.
        start_frame (int): the source frame (32 fps) the room starts at.
        seats (tuple[Seat, ...]): the players a person can take.
        note (str): a line under the title.
        cover (int): the slot of one of its seats, whose first frame stands for the round in the
            lobby.
        match_id (int | None), round (int | None): the recorded round, which the WorldCast client
            reads; ``None``: none (the mock client alone).
    """

    id: str
    map: str
    label: str
    start_frame: int
    seats: tuple[Seat, ...]
    note: str = ""
    cover: int = 0
    match_id: int | None = None
    round: int | None = None

    @classmethod
    def from_json(cls, record: dict) -> "RoundStart":
        """The round start of a ``library.json`` record; ``ValueError`` if it has no seat, or its
        cover is not one of them."""
        seats = tuple(Seat.from_json(seat) for seat in record["seats"])
        if not seats:
            raise ValueError(f"round start {record['id']!r} has no seats")
        cover = int(record.get("cover", seats[0].slot))
        if cover not in {seat.slot for seat in seats}:
            raise ValueError(
                f"round start {record['id']!r}: its cover, seat {cover}, is not a seat"
            )
        return cls(
            id=record["id"],
            map=record["map"],
            label=record["label"],
            start_frame=int(record["start_frame"]),
            seats=seats,
            note=record.get("note", ""),
            cover=cover,
            match_id=record.get("match_id"),
            round=record.get("round"),
        )

    @property
    def map_label(self) -> str:
        """The map's display name."""
        return MAP_LABELS.get(self.map, self.map)

    def seat(self, slot: int) -> Seat:
        """The seat of player ``slot``; ``KeyError`` if the round start has none."""
        for seat in self.seats:
            if seat.slot == int(slot):
                return seat
        raise KeyError(f"round start {self.id} has no seat {slot}")

    def public(self) -> dict:
        """What the page reads of a round start."""
        return {
            "id": self.id,
            "map_label": self.map_label,
            "label": self.label,
            "note": self.note,
            "cover": self.cover,
            "seats": [seat.public() for seat in self.seats],
        }


@dataclass
class Library:
    """The round starts by id, and the directory their files are relative to."""

    round_starts: dict[str, RoundStart] = field(default_factory=dict)
    root: Path | None = None

    def get(self, round_start: str) -> RoundStart:
        """The round start of that id; ``KeyError`` if the library has none."""
        if round_start not in self.round_starts:
            raise KeyError(f"unknown round start {round_start!r}")
        return self.round_starts[round_start]

    def path(self, relative: str) -> Path:
        """A file of the library; ``FileNotFoundError`` for a path outside its directory."""
        if self.root is None:
            raise FileNotFoundError(relative)
        target = (self.root / relative).resolve()
        if self.root.resolve() not in target.parents:
            raise FileNotFoundError(relative)
        return target

    def public(self) -> list[dict]:
        """What the page reads of the library: every round start's :meth:`RoundStart.public`."""
        return [round_start.public() for round_start in self.round_starts.values()]

    @classmethod
    def load(cls, path: str | None) -> "Library":
        """The library of a ``library.json``; ``None``: the synthetic arena."""
        if not path:
            return synthetic_library()
        records = json.loads(Path(path).read_text(encoding="utf-8"))["round_starts"]
        round_starts = {record["id"]: RoundStart.from_json(record) for record in records}
        if not round_starts:
            raise ValueError(f"{path}: the library has no round starts")
        return cls(round_starts, Path(path).parent)

    def save(self) -> Path:
        """Write ``library.json`` into the library's directory; returns its path."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "library.json"
        records = [asdict(round_start) for round_start in self.round_starts.values()]
        path.write_text(json.dumps({"round_starts": records}, indent=1), encoding="utf-8")
        return path


def players_in_first_frame(
    label_root: str | Path, media_id: str, slot: int, start_frame: int
) -> list[int]:
    """The other players the first frame of a seat's client may show, by the dataset's GT
    visibility labels at the round start.

    A player counts if its label says it is visible, or if it is inside the view frustum while its
    label is unknown (it may well be in the frame). A player behind the camera, or whose label
    says it is occluded, does not. A dead player is in no frustum: a corpse in the first frame is
    not detected (a round start at the start of the round has none).

    Args:
        label_root (str | Path): the visibility labels (``paths.visibility_label_root``).
        media_id (str): the seat's recording.
        slot (int): the seat's player, which never counts.
        start_frame (int): the round start, a source frame (video frame 0 of the client's window).

    Returns:
        list[int]: the slots of the players, in order.
    """
    frame = np.array([start_frame])
    visible, valid = visibility_rows(label_root, media_id, frame)
    seen = (visible[:, 0] > 0.5) | (frustum_rows(label_root, media_id, frame)[:, 0] & ~valid[:, 0])
    seen[int(slot)] = False
    return np.flatnonzero(seen).tolist()


def zooming_weapons(loadout: Iterable[str]) -> list[str]:
    """The weapons of a loadout that zoom, by their names in the model's weapon table.

    Live play does not track the scope: while a weapon that zooms
    (:data:`~worldcast.data.camera.SCOPE_ZOOM_FOV`) is held, the client marks the scope unknown, and
    the generator draws a black scope view from the first frame. The names are normalized as the
    client's weapon ids are (``AWP`` and ``weapon_awp`` are ``awp``).

    Args:
        loadout (Iterable[str]): the weapons of keys 1-4, as a manifest, a tick table or a library
            spells them.

    Returns:
        list[str]: the normalized names of those that zoom, in order.
    """
    # imported here: worldcast.data.camera loads torch, which the coordinator (it imports this
    # module) does not
    from worldcast.data import normalize_weapon_name
    from worldcast.data.camera import SCOPE_ZOOM_FOV

    names = [normalize_weapon_name(weapon) for weapon in loadout]
    return [name for name in names if name in SCOPE_ZOOM_FOV]


def synthetic_library() -> Library:
    """One synthetic arena with four seats, two per side: the mock client needs no files for it."""
    spawns = [
        (-600.0, -300.0, 0.0, 30.0, 0.0),
        (-600.0, 300.0, 0.0, -30.0, 0.0),
        (600.0, -300.0, 0.0, 150.0, 0.0),
        (600.0, 300.0, 0.0, -150.0, 0.0),
    ]
    teams = ("T", "T", "CT", "CT")
    seats = tuple(
        Seat(slot=i, team=team, spawn=spawn, loadout=DEFAULT_LOADOUT[team])
        for i, (team, spawn) in enumerate(zip(teams, spawns))
    )
    arena = RoundStart(
        id="arena",
        map="arena",
        label="Grid world",
        start_frame=0,
        seats=seats,
        note="No assets needed: the mock client renders a grid world.",
    )
    return Library({arena.id: arena})
