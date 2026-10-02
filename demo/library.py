"""The library of recorded round starts a room can begin from: maps, rounds and their seats (``library.json``).

Built by ``tools/build_demo_library.py``. Image paths are relative to the library file and served to browsers under
``/library/``. Without a library the demo offers one synthetic arena that needs no files.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Display names of the paper's maps.
MAP_LABELS = {
    "de_ancient": "Ancient",
    "de_dust2": "Dust II",
    "de_mirage": "Mirage",
    "de_nuke": "Nuke",
    "arena": "Synthetic arena",
}
#: ``<match>-<map>-r<round>-p<slot>``, the OpenCS2 media id of one player's recording of one round.
MEDIA_ID = re.compile(r"^(\d+)-(\w+)-r(\d+)-p(\d+)$")
#: Weapon keys 1-4: primary, secondary, knife, grenade.
DEFAULT_LOADOUT = {
    "T": ("ak47", "glock", "knife", "hegrenade"),
    "CT": ("m4a1_silencer", "usp_silencer", "knife", "smokegrenade"),
}


@dataclass(frozen=True)
class Seat:
    """One recorded player of a round start: where it spawns and what the engine needs to start its client."""

    seat: int
    media_id: str
    team: str  # "T" | "CT"
    spawn: tuple[float, float, float, float, float]  # x, y, z (engine units), yaw, pitch (degrees)
    loadout: tuple[str, ...] = ()
    preview: str | None = None  # first frame, relative to the library
    clip: str | None = None  # mock engine: directory of rendered JPEG frames

    def public(self) -> dict:
        d = asdict(self)
        d.pop("clip")
        return d


@dataclass(frozen=True)
class Radar:
    """A top-down map image and its transform: ``px = cx + scale * (x - x0)``, ``py = cy - scale * (y - y0)``."""

    image: str
    x0: float
    y0: float
    cx: float
    cy: float
    scale: float


@dataclass(frozen=True)
class RoundStart:
    id: str
    map: str
    label: str
    start_frame: int
    seats: tuple[Seat, ...]
    radar: Radar | None = None
    note: str = ""
    cover: int = 0  # the seat whose first frame stands for the round in the lobby
    match_id: int = 0  # the recorded round (0: none, e.g. the synthetic arena)
    round: int = 0

    def seat(self, index: int) -> Seat:
        for s in self.seats:
            if s.seat == int(index):
                return s
        raise KeyError(f"round {self.id} has no seat {index}")

    def public(self) -> dict:
        return {
            "id": self.id,
            "map": self.map,
            "map_label": MAP_LABELS.get(self.map, self.map),
            "label": self.label,
            "note": self.note,
            "start_frame": self.start_frame,
            "seats": [s.public() for s in self.seats],
            "radar": asdict(self.radar) if self.radar else None,
            "cover": self.cover,
        }


@dataclass
class Library:
    rounds: dict[str, RoundStart] = field(default_factory=dict)
    root: Path | None = None  # directory the relative paths resolve against

    def get(self, round_id: str) -> RoundStart:
        if round_id not in self.rounds:
            raise KeyError(f"unknown round start {round_id!r}")
        return self.rounds[round_id]

    def path(self, relative: str) -> Path:
        if self.root is None:
            raise FileNotFoundError(relative)
        target = (self.root / relative).resolve()
        if self.root.resolve() not in target.parents:
            raise FileNotFoundError(relative)
        return target

    def public(self) -> list[dict]:
        return [r.public() for r in self.rounds.values()]

    @classmethod
    def load(cls, path: str | None) -> "Library":
        if not path:
            return synthetic_library()
        file = Path(path)
        data = json.loads(file.read_text(encoding="utf-8"))
        rounds = {}
        for r in data["rounds"]:
            seats = tuple(
                Seat(
                    seat=int(s["seat"]),
                    media_id=s["media_id"],
                    team=s["team"],
                    spawn=tuple(float(v) for v in s["spawn"]),
                    loadout=tuple(s.get("loadout") or DEFAULT_LOADOUT[s["team"]]),
                    preview=s.get("preview"),
                    clip=s.get("clip"),
                )
                for s in r["seats"]
            )
            radar = Radar(**r["radar"]) if r.get("radar") else None
            recorded = MEDIA_ID.match(seats[0].media_id)
            rounds[r["id"]] = RoundStart(
                id=r["id"],
                map=r["map"],
                label=r["label"],
                start_frame=int(r["start_frame"]),
                seats=seats,
                radar=radar,
                note=r.get("note", ""),
                cover=int(r.get("cover", seats[0].seat)),
                match_id=int(recorded[1]) if recorded else 0,
                round=int(recorded[3]) if recorded else 0,
            )
        if not rounds:
            raise ValueError(f"{path}: the library has no round starts")
        return cls(rounds=rounds, root=file.parent)


def synthetic_library() -> Library:
    """One synthetic arena with four seats (two per side): runs with the mock engine and no files at all."""
    spawns = [
        (-600.0, -300.0, 0.0, 30.0, 0.0),
        (-600.0, 300.0, 0.0, -30.0, 0.0),
        (600.0, -300.0, 0.0, 150.0, 0.0),
        (600.0, 300.0, 0.0, -150.0, 0.0),
    ]
    seats = tuple(
        Seat(
            seat=i,
            media_id=f"arena-p{i:02d}",
            team="T" if i < 2 else "CT",
            spawn=spawn,
            loadout=DEFAULT_LOADOUT["T" if i < 2 else "CT"],
        )
        for i, spawn in enumerate(spawns)
    )
    arena = RoundStart(
        id="arena",
        map="arena",
        label="Grid world",
        start_frame=0,
        seats=seats,
        note="No assets needed: the mock engine renders a grid world.",
    )
    return Library(rounds={arena.id: arena})
