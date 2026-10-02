"""The media index: one row per recorded player-round (one "media"), mapping ids to files.

JSON lines (config ``paths.media_index``); paths in a row are relative to ``paths.dataset_root``.
Formats: docs/data.md.
"""

import json
from dataclasses import dataclass
from pathlib import Path

#: (match_id, map_name, round): identifies one round of one match.
RoundKey = tuple[int, str, int]


@dataclass(frozen=True)
class MediaRecord:
    """One media-index row (the fields the release reads).

    Attributes:
        media_id (str): ``<match>-<map>-r<round>-p<slot>``.
        match_id (int): the match.
        map_name (str): the map.
        round (int): the round.
        player_slot (int): the player's slot in the round (0-9).
        fps (float): frame rate of the recorded video (32.0).
        video_frames (int): frames in the video.
        ticks_path (str): the tick parquet, relative to the dataset root.
        ticks_rows (int): rows the tick parquet must hold.
        ticks_file_size (int): its size, bytes.
        ticks_sha256 (str | None): its hex digest, or None (the digest is then computed from the
            file).
        video_path (str | None): the recorded video, relative to the dataset root (training only).
        capture_start_tick (int | None): first engine tick of the capture, shared by a round's
            players.
    """

    media_id: str
    match_id: int
    map_name: str
    round: int
    player_slot: int
    fps: float
    video_frames: int
    ticks_path: str
    ticks_rows: int
    ticks_file_size: int
    ticks_sha256: str | None = None
    video_path: str | None = None
    capture_start_tick: int | None = None

    @property
    def round_key(self) -> RoundKey:
        """``(match_id, map_name, round)``."""
        return (self.match_id, self.map_name, self.round)

    @classmethod
    def from_row(cls, row: dict) -> "MediaRecord":
        """Build from one parsed JSON line; missing required fields raise ``KeyError``."""
        return cls(
            media_id=str(row["media_id"]),
            match_id=int(row["match_id"]),
            map_name=str(row["map_name"]),
            round=int(row["round"]),
            player_slot=int(row["player_slot"]),
            fps=float(row["fps"]),
            video_frames=int(row["video_frames"]),
            ticks_path=str(row["ticks_path"]),
            ticks_rows=int(row["ticks_rows"]),
            ticks_file_size=int(row["ticks_file_size"]),
            ticks_sha256=None if row.get("ticks_sha256") is None else str(row["ticks_sha256"]),
            video_path=None if row.get("video_path") is None else str(row["video_path"]),
            capture_start_tick=(
                None if row.get("capture_start_tick") is None else int(row["capture_start_tick"])
            ),
        )


class MediaIndex:
    """All media rows, by id (``by_id``) and by round (``round_slots``: ``{player_slot: record}``).

    A slot missing from a round (about 4% of rounds lack one player's media) is an absent player.
    """

    def __init__(self, records: dict[str, MediaRecord]) -> None:
        self.by_id: dict[str, MediaRecord] = dict(records)
        self.round_slots: dict[RoundKey, dict[int, MediaRecord]] = {}
        for record in self.by_id.values():
            self.round_slots.setdefault(record.round_key, {})[record.player_slot] = record

    @classmethod
    def load(cls, path: str | Path) -> "MediaIndex":
        """Read a media-index JSONL file. Duplicate ``media_id`` rows raise ``ValueError``."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"media index does not exist: {path}")
        records: dict[str, MediaRecord] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = MediaRecord.from_row(json.loads(line))
                if record.media_id in records:
                    raise ValueError(f"duplicate media_id in media index: {record.media_id}")
                records[record.media_id] = record
        return cls(records)

    def media(self, media_id: str) -> MediaRecord:
        """The row of ``media_id``; ``KeyError`` if the index does not list it."""
        record = self.by_id.get(str(media_id))
        if record is None:
            raise KeyError(f"media {media_id!r} is not in the media index")
        return record

    def slots(self, round_key: RoundKey) -> dict[int, MediaRecord]:
        """``{player_slot: record}`` of one round.

        Raises:
            ValueError: the round's rows disagree on ``capture_start_tick`` (the recordings would
                not share a time base).
        """
        slots = self.round_slots[round_key]
        capture_starts = {
            r.capture_start_tick for r in slots.values() if r.capture_start_tick is not None
        }
        if len(capture_starts) > 1:
            raise ValueError(f"round {round_key} has misaligned capture start ticks")
        return slots
