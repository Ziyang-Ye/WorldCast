"""Recordings: the round index, the media index and the per-player tick tables.

The round index (config ``paths.round_index``) has one row per client, one player's window of one
round; a client runs row ``run.index_row``. The media index (``paths.media_index``) has one row per
recorded player-round (one "media"), mapping its ids to its files, relative to
``paths.dataset_root``. A tick table is one player's OpenCS2 ``ticks.parquet``: its size must equal
the media row's ``ticks_file_size`` and, when the row records ``ticks_sha256``, its sha256 must
equal it. The digest keys the draws of :mod:`~worldcast.data.jump_recall`, so a re-exported
parquet with other bytes would change the controls; the mtime is not checked (a plain copy changes
it). Formats: docs/inference.md, "Data".
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from worldcast.utils.files import sha256_file

from .controls import count_until
from .game import TICK_RATE
from .jump_recall import apply_jump_recall
from .latents import window_key

__all__ = [
    "ALIVE_INDEX",
    "NUM_PLAYERS",
    "TICK_COLUMNS",
    "MediaIndex",
    "MediaRecord",
    "RoundIndexRow",
    "RoundKey",
    "TickTable",
    "load_round_index_row",
    "player_rows",
    "read_jsonl",
    "read_player_ticks",
    "read_round_index",
    "read_tick_columns",
    "verify_ticks_file",
]

#: (match_id, map_name, round): identifies one round of one match.
RoundKey = tuple[int, str, int]

#: Player slots per round.
NUM_PLAYERS = 10
#: Column of ``alive`` in a player's state row ``[x, y, z, yaw, pitch, alive]``
#: (:func:`player_rows`; ``player_states`` of a batch).
ALIVE_INDEX = 5
#: Parquet columns read (the files hold more; see docs/inference.md, "Data").
TICK_COLUMNS = (
    "tick",
    "t",
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "is_alive",
    "active",
    "delta_pitch",
    "delta_yaw",
    "input_weapon",
    "team_num",
)


def read_jsonl(path: str | Path, what: str) -> Iterator[tuple[int, dict]]:
    """``(line number from 1, record)`` of every non-blank line of the JSONL file ``path`` (``what``
    names it in the ``FileNotFoundError``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


# -------------------------------------------------------------------------------------- media index
@dataclass(frozen=True)
class MediaRecord:
    """One media-index row (the fields the release reads).

    Attributes:
        media_id (str): ``<match>-<map>-r<round>-p<slot>``.
        match_id (int): the match.
        map_name (str): the map.
        round (int): the round.
        player_slot (int): the player's slot in the round (0-9).
        fps (float): source frames per second of the recorded video (32.0).
        source_frames (int): source frames in the recorded video (the row's ``video_frames``).
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
    source_frames: int
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
            source_frames=int(row["video_frames"]),
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
        records: dict[str, MediaRecord] = {}
        for _, row in read_jsonl(path, "media index"):
            record = MediaRecord.from_row(row)
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


# -------------------------------------------------------------------------------------- round index
@dataclass(frozen=True)
class RoundIndexRow:
    """The fields of one round-index row that inference reads.

    Attributes:
        media_id (str): the client's own recording, ``<match>-<map>-r<round>-p<slot>``.
        start_frame (int): first source frame (32 fps) of the window.
        match_id (int): the match.
        round (int): the round.
        map_name (str): the map.
        player_slot (int | None): the client's player slot.
        clients (tuple[str, ...]): the media ids of the round's clients (this one included), which
            run in lockstep (the row's ``group_media``).
        round_seconds (int | None): the round's recorded length, s, when the index has it.
    """

    media_id: str
    start_frame: int
    match_id: int
    round: int
    map_name: str
    player_slot: int | None = None
    clients: tuple[str, ...] = ()
    round_seconds: int | None = None

    @classmethod
    def from_row(cls, row: dict) -> "RoundIndexRow":
        """Build from one parsed JSON line; a ``latent_key`` that is not the latent-cache member
        of ``start_frame`` raises ``ValueError``."""
        start_frame = int(row["start_frame"])
        if start_frame < 0:
            raise ValueError(f"index row {row.get('media_id')!r} has a negative start_frame")
        if str(row.get("latent_key", window_key(start_frame))) != window_key(start_frame):
            raise ValueError(
                f"index row {row.get('media_id')!r}: latent_key {row['latent_key']!r} does not"
                f" encode start_frame {start_frame}"
            )
        return cls(
            media_id=str(row["media_id"]),
            start_frame=start_frame,
            match_id=int(row["match_id"]),
            round=int(row["round"]),
            map_name=str(row["map_name"]),
            player_slot=None if row.get("player_slot") is None else int(row["player_slot"]),
            clients=tuple(str(m) for m in (row.get("group_media") or ())),
            round_seconds=None if row.get("round_seconds") is None else int(row["round_seconds"]),
        )

    def other_clients(self) -> tuple[str, ...]:
        """The other clients of the round: ``clients`` without this one."""
        return tuple(m for m in self.clients if m != self.media_id)


def read_round_index(path: str | Path) -> list[RoundIndexRow]:
    """All rows of a round-index JSONL file, in file order."""
    return [RoundIndexRow.from_row(row) for _, row in read_jsonl(path, "round index")]


def load_round_index_row(path: str | Path, index_row: int) -> RoundIndexRow:
    """Row ``index_row`` (0-based) of the round index; ``IndexError`` if out of range."""
    rows = read_round_index(path)
    if not 0 <= int(index_row) < len(rows):
        raise IndexError(f"row {index_row} outside the {len(rows)}-row index {path}")
    return rows[int(index_row)]


# -------------------------------------------------------------------------------------- tick tables
@dataclass(frozen=True)
class TickTable:
    """One player's ticks, ``n`` rows, with the columns of docs/inference.md ("Data").

    Float columns are float64 arrays holding the parquet's float32 values exactly; ``team_num``
    is int64, ``is_alive`` bool. ``active`` holds the names of the held buttons, with jump recall
    keyed by the file's digest unless the table was read without it.
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    is_alive: np.ndarray
    active: list[list | None]
    delta_pitch: np.ndarray
    delta_yaw: np.ndarray
    input_weapon: list[str | None]
    team_num: np.ndarray

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def states(self, rows: np.ndarray | Sequence[int]) -> np.ndarray:
        """``[len(rows), 6]`` float32 player states ``x, y, z, yaw, pitch, alive`` (u, degrees,
        {0, 1}) at the tick rows ``rows``."""
        columns = (self.x, self.y, self.z, self.yaw, self.pitch, self.is_alive)
        return np.stack([np.asarray(column, dtype=np.float32)[rows] for column in columns], axis=-1)


def verify_ticks_file(media: MediaRecord, ticks_path: Path) -> str:
    """Check size (and sha256 when recorded) of a tick parquet; return its sha256 (lower-case hex).

    Raises:
        FileNotFoundError: the file is missing.
        RuntimeError: a size or digest mismatch, or a recorded digest that is not 64 lower-case
            hex digits.
    """
    ticks_path = Path(ticks_path)
    if not ticks_path.is_file():
        raise FileNotFoundError(f"tick table of media {media.media_id} not found: {ticks_path}")
    size = ticks_path.stat().st_size
    if size != media.ticks_file_size:
        raise RuntimeError(
            f"tick table of media {media.media_id} has {size} bytes, the media index says "
            f"{media.ticks_file_size}: {ticks_path}"
        )
    digest = sha256_file(ticks_path)
    expected = media.ticks_sha256
    if expected is None:
        return digest
    if len(expected) != 64 or not all(c in "0123456789abcdef" for c in expected):
        raise RuntimeError(f"media {media.media_id} carries a malformed ticks_sha256 {expected!r}")
    if digest != expected:
        raise RuntimeError(
            f"tick table of media {media.media_id} does not match the media index sha256 "
            f"(file {digest}, index {expected}): {ticks_path}"
        )
    return digest


def read_tick_columns(ticks_path: str | Path, columns: Sequence[str], rows: int) -> dict[str, list]:
    """The named columns of a tick parquet as lists (needs pyarrow).

    Raises:
        RuntimeError: the file does not hold ``rows`` rows.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(str(ticks_path), columns=list(columns)).to_pydict()
    held = len(table[columns[0]])
    if held != int(rows):
        raise RuntimeError(f"tick table has {held} rows, expected {int(rows)}: {ticks_path}")
    return table


def read_player_ticks(
    media: MediaRecord, dataset_root: str | Path, *, jump_recall: bool = True
) -> TickTable:
    """Read and verify the tick table ``dataset_root / media.ticks_path`` (needs pyarrow).

    Args:
        media (MediaRecord): the player's media row.
        dataset_root (str | Path): the recordings.
        jump_recall (bool): repair the ``jump`` button (:mod:`~worldcast.data.jump_recall`), as the
            generator was trained from stage 2 on; the state model reads the recorded buttons.

    Returns:
        TickTable: the player's ticks.

    Raises:
        RuntimeError: the row count differs from the index, the table is empty, ticks or times are
            not strictly increasing, the time base does not start at 0, or
            ``t != (tick - tick[0]) / 64`` (tolerance 1e-6 s).
    """
    ticks_path = Path(dataset_root) / media.ticks_path
    digest = verify_ticks_file(media, ticks_path)
    table = read_tick_columns(ticks_path, TICK_COLUMNS, media.ticks_rows)
    if not table["t"]:
        raise RuntimeError(f"tick table is empty: {ticks_path}")
    tick = np.asarray(table["tick"], dtype=np.int64)
    timestamps = np.asarray(table["t"], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise RuntimeError(f"tick timestamps are not strictly increasing in {ticks_path}")
    if np.any(np.diff(tick) <= 0):
        raise RuntimeError(f"tick numbers are not strictly increasing in {ticks_path}")
    if abs(float(timestamps[0])) > 1e-6:
        raise RuntimeError(f"tick time base does not start at 0 in {ticks_path}")
    expected_timestamps = (tick - tick[0]).astype(np.float64) / TICK_RATE
    if not np.allclose(timestamps, expected_timestamps, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"tick and time columns disagree in {ticks_path}")
    if any(value is None for value in table["is_alive"]):
        raise RuntimeError(f"tick table has null is_alive values: {ticks_path}")

    def floats(name: str) -> np.ndarray:
        return np.asarray(table[name], dtype=np.float64)

    return TickTable(
        t=timestamps,
        x=floats("x"),
        y=floats("y"),
        z=floats("z"),
        yaw=floats("yaw"),
        pitch=floats("pitch"),
        is_alive=np.asarray(table["is_alive"], dtype=np.bool_),
        active=(
            apply_jump_recall(table["active"], table["z"], key=digest)
            if jump_recall
            else list(table["active"])
        ),
        delta_pitch=floats("delta_pitch"),
        delta_yaw=floats("delta_yaw"),
        input_weapon=list(table["input_weapon"]),
        team_num=np.asarray(table["team_num"], dtype=np.int64),
    )


def player_rows(tick_tables: Mapping[int, TickTable], frame: int, source_fps: float) -> np.ndarray:
    """``[P, 6]`` float32 ``x, y, z, yaw, pitch, alive`` of every player at source frame ``frame``
    (the last tick at or before ``frame / source_fps``; zeros for absent players)."""
    out = np.zeros((NUM_PLAYERS, 6), np.float32)
    for slot, table in tick_tables.items():
        row = int(count_until(table.t, frame / source_fps)) - 1
        if 0 <= row < len(table):
            out[int(slot)] = table.states([row])[0]
    return out
