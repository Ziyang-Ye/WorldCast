"""The round index: one row per client (one player's window of one round).

JSON lines (config ``paths.round_index``); a client runs row ``run.index_row``. Formats:
docs/data.md.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .latents import window_key


@dataclass(frozen=True)
class RoundIndexRow:
    """The fields of one round-index row that inference reads.

    Attributes:
        media_id (str): the client's own recording, ``<match>-<map>-r<round>-p<slot>``.
        start_frame (int): first source frame (32 fps) of the window.
        match_id (int): the match.
        round (int): the round.
        map_name (str): the map.
        latent_key (str): the latent-cache member of the window's first latent,
            ``win_<start_frame:06d>``.
        player_slot (int | None): the client's player slot.
        group_media (tuple[str, ...]): the round's clients (this one included), run in lock-step.
        group_slots (tuple[int, ...]): their player slots, same order.
    """

    media_id: str
    start_frame: int
    match_id: int
    round: int
    map_name: str
    latent_key: str
    player_slot: int | None = None
    group_media: tuple[str, ...] = ()
    group_slots: tuple[int, ...] = ()

    @classmethod
    def from_row(cls, row: dict) -> "RoundIndexRow":
        """Build from one parsed JSON line. ``latent_key`` is taken verbatim and checked."""
        start_frame = int(row["start_frame"])
        if start_frame < 0:
            raise ValueError(f"index row {row.get('media_id')!r} has a negative start_frame")
        latent_key = str(row["latent_key"])
        if latent_key != window_key(start_frame):
            raise ValueError(
                f"index row {row.get('media_id')!r}: latent_key {latent_key!r} does not encode "
                f"start_frame {start_frame}"
            )
        return cls(
            media_id=str(row["media_id"]),
            start_frame=start_frame,
            match_id=int(row["match_id"]),
            round=int(row["round"]),
            map_name=str(row["map_name"]),
            latent_key=latent_key,
            player_slot=None if row.get("player_slot") is None else int(row["player_slot"]),
            group_media=tuple(str(m) for m in (row.get("group_media") or ())),
            group_slots=tuple(int(s) for s in (row.get("group_slots") or ())),
        )

    def lockstep_peers(self) -> tuple[str, ...]:
        """The other rendered clients of the round: ``group_media`` without this one."""
        return tuple(m for m in self.group_media if m != self.media_id)


def read_round_index(path: str | Path) -> list[RoundIndexRow]:
    """All rows of a round-index JSONL file, in file order."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"round index does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(RoundIndexRow.from_row(json.loads(line)))
    return rows


def load_round_index_row(path: str | Path, index_row: int) -> RoundIndexRow:
    """Row ``index_row`` (0-based) of the round index; ``IndexError`` if out of range."""
    rows = read_round_index(path)
    if not 0 <= int(index_row) < len(rows):
        raise IndexError(f"row {index_row} outside the {len(rows)}-row index {path}")
    return rows[int(index_row)]
