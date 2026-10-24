"""A tiny OpenCS2 download on disk: tick tables, videos and the index that lists them."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Video frames of a recording: 1.25 s at 32 fps.
FRAMES = 40
#: The tick tables have two ticks per video frame (64 Hz against 32 fps).
TICKS_PER_FRAME = 2
#: Pixel size of the test videos.
VIDEO_SIZE = (32, 48)
#: The file's media path prefix in the index, as the dataset writes it.
HUB = "hf://datasets/blanchon/opencs2_dataset@main/"


@dataclass(frozen=True)
class Seat:
    """One player-round to write.

    Attributes:
        match_id (int): the match.
        map_name (str): the map.
        round (int): the round.
        slot (int): the player slot.
        frames (int): video frames of its recording.
        rows (int | None): rows of its tick table; ``2 * frames`` when None.
        missing (tuple[int, ...]): rows of that table to leave out (a gap in the ticks).
        buttons (tuple[tuple[int, str], ...]): ``(row, button)`` pairs held in addition to
            ``forward``.
        alive_rows (int | None): the player is alive in the rows before this one; None: always.
    """

    match_id: int
    map_name: str
    round: int
    slot: int
    frames: int = FRAMES
    rows: int | None = None
    missing: tuple[int, ...] = ()
    buttons: tuple[tuple[int, str], ...] = ()
    alive_rows: int | None = None

    @property
    def media_id(self) -> str:
        return f"{self.match_id}-{self.map_name}-r{self.round:02d}-p{self.slot:02d}"

    @property
    def first_tick(self) -> int:
        """The round's capture start, shared by its players."""
        return 5000 + 100 * self.round

    @property
    def directory(self) -> str:
        return (
            f"rounds/match_id={self.match_id}/map_name={self.map_name}/round={self.round:02d}"
            f"/player={self.slot:02d}"
        )


#: Two maps, three rounds, six recordings.
SEATS = (
    Seat(11, "de_dust2", 1, 0),
    Seat(11, "de_dust2", 1, 1),
    Seat(11, "de_dust2", 1, 2),
    Seat(11, "de_dust2", 2, 0),
    Seat(12, "de_nuke", 1, 0),
    Seat(12, "de_nuke", 1, 1),
)


def write_ticks(path: Path, seat: Seat) -> None:
    """The seat's OpenCS2 tick table: the columns the release reads and some it does not."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = seat.rows if seat.rows is not None else TICKS_PER_FRAME * seat.frames
    kept = [row for row in range(rows) if row not in seat.missing]
    ticks = np.asarray(kept, np.int64) + seat.first_tick
    extra = dict(seat.buttons)
    n = len(kept)
    zeros = np.zeros(n, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "tick": pa.array(ticks, pa.int32()),
                "t": pa.array((ticks - seat.first_tick) / 64.0, pa.float32()),
                "x": pa.array(np.asarray(kept, np.float32) * 4.0, pa.float32()),
                "y": pa.array(zeros, pa.float32()),
                "z": pa.array(zeros, pa.float32()),
                "yaw": pa.array(np.asarray(kept, np.float32) * 0.5, pa.float32()),
                "pitch": pa.array(zeros, pa.float32()),
                "is_alive": pa.array(
                    [seat.alive_rows is None or row < seat.alive_rows for row in kept]
                ),
                "active": pa.array(
                    [["forward"] + ([extra[row]] if row in extra else []) for row in kept],
                    pa.list_(pa.string()),
                ),
                "delta_pitch": pa.array(zeros, pa.float32()),
                "delta_yaw": pa.array(np.full(n, 0.25, np.float32), pa.float32()),
                "input_weapon": pa.array(["weapon_ak47"] * n),
                "team_num": pa.array([2] * n, pa.int8()),
                "health": pa.array([100] * n, pa.int16()),
            }
        ),
        path,
    )


def write_video(path: Path, frames: int) -> None:
    """A ``frames``-frame 32 fps H.264 recording of a brightening grey that wraps round to black."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=32, codec="libx264", quality=9, macro_block_size=1)
    for frame in range(frames):
        writer.append_data(np.full((*VIDEO_SIZE, 3), (5 * frame + 20) % 256, np.uint8))
    writer.close()


def write_opencs2(
    root: Path,
    seats: Sequence[Seat] = SEATS,
    *,
    index: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Write the seats' files under ``root`` and ``index/pov_rounds.parquet``, in reverse order.

    The index rows carry what the dataset's index holds, measured from the files; ``index`` names
    values to put in a row instead, by media id (an index that is wrong about its files).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for seat in seats:
        directory = root / seat.directory
        write_ticks(directory / "ticks.parquet", seat)
        write_video(directory / "video.mp4", seat.frames)
        row = {
            "media_id": seat.media_id,
            "match_id": seat.match_id,
            "map_name": seat.map_name,
            "round": seat.round,
            "player_slot": seat.slot,
            "player_side": "T",
            "capture_start_tick": seat.first_tick,
            "video_frames": seat.frames,
            "video": {"bytes": None, "path": f"{HUB}{seat.directory}/video.mp4"},
            "media_bytes": (directory / "video.mp4").stat().st_size,
            "sidecar_bytes": 1,
            "ticks_parquet_path": f"{HUB}{seat.directory}/ticks.parquet",
        }
        rows.append({**row, **(index or {}).get(seat.media_id, {})})
    video = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema(
        [
            ("media_id", pa.string()),
            ("match_id", pa.int32()),
            ("map_name", pa.string()),
            ("round", pa.int16()),
            ("player_slot", pa.int8()),
            ("player_side", pa.string()),
            ("capture_start_tick", pa.int32()),
            ("video_frames", pa.int32()),
            ("video", video),
            ("media_bytes", pa.int64()),
            ("sidecar_bytes", pa.int64()),
            ("ticks_parquet_path", pa.string()),
        ]
    )
    (root / "index").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows[::-1], schema=schema), root / "index/pov_rounds.parquet"
    )
