"""The media index: one row per player-round of OpenCS2.

Reads ``index/pov_rounds.parquet`` of an OpenCS2 download and, for every player-round of the chosen
maps, checks its video and its tick table and writes the media index (docs/inference.md, "Data"):
``media_id, match_id, map_name, round, player_slot, fps, video_frames, video_path, ticks_path,
ticks_rows, ticks_file_size, ticks_sha256, capture_start_tick``, the paths relative to the
download, one JSON object per line ordered by map, match, round and player slot.

A player-round is left out, and counted by reason in the summary, when its video is missing or
has another size than the index says, when the video does not hold the index's frame count or its
first frame does not decode, or when its tick table is missing, unreadable, empty or does not
start at the capture's first tick.

Examples::

    python tools/prepare_data.py media-index --dataset-root data/opencs2 \\
        --out data/media_index.jsonl
    python tools/prepare_data.py media-index --dataset-root data/opencs2 \\
        --out data/media_index.jsonl --maps de_dust2 de_nuke --workers 16
"""

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from pathlib import Path
from typing import Any

from worldcast.data.latents import SOURCE_FPS
from worldcast.utils.files import sha256_file, write_jsonl

__all__ = [
    "DEFAULT_WORKERS",
    "MAPS",
    "MediaRefused",
    "add_arguments",
    "add_workers_argument",
    "build_media_index",
    "check_all",
    "first_line",
    "media_row",
    "read_pov_rounds",
    "refusal_summary",
    "run",
]

#: The maps of the paper's training data, in the order of its bucket files (file ``i`` holds
#: map ``i``).
MAPS = ("de_dust2", "de_mirage", "de_nuke", "de_ancient")
#: OpenCS2's index of player-rounds, relative to the dataset root.
POV_INDEX = "index/pov_rounds.parquet"
#: The columns of :data:`POV_INDEX` that the step reads.
POV_COLUMNS = (
    "media_id",
    "match_id",
    "map_name",
    "round",
    "player_slot",
    "capture_start_tick",
    "video_frames",
    "video",
    "media_bytes",
    "ticks_parquet_path",
)
#: Threads of a scan, unless ``--workers`` sets them.
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
#: A scan reports its progress on stderr after this many recordings.
PROGRESS_EVERY = 5000


class MediaRefused(ValueError):
    """A recording that a step leaves out; ``reason`` names the line of the step's summary that
    counts it."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def first_line(error: Exception) -> str:
    """The first line of an error's message (a reader's own can run over several)."""
    return (str(error).splitlines() or [type(error).__name__])[0]


def add_workers_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--workers``."""
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="threads that check recordings in parallel (default %(default)s); the output does"
        " not depend on it",
    )


def check_all(check: Callable[[Any], Any], items: Sequence, workers: int, what: str) -> list:
    """``check(item)`` for every item, in the order of ``items``: its result, or the
    :class:`MediaRefused` it raised, so that one bad recording does not stop the scan.

    Args:
        check (Callable[[Any], Any]): the check of one item.
        items (Sequence): what to check.
        workers (int): threads; the results do not depend on it.
        what (str): names the scan in its progress lines.
    """
    finished = count(1)

    def attempt(item: Any) -> Any:
        try:
            result = check(item)
        except MediaRefused as refused:
            result = refused
        number = next(finished)
        if number % PROGRESS_EVERY == 0:
            print(f"{what}: {number} of {len(items)}", file=sys.stderr, flush=True)
        return result

    if workers <= 1:
        return [attempt(item) for item in items]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(attempt, items))


def refusal_summary(refused: Sequence[tuple[str, MediaRefused]]) -> list[str]:
    """One line per reason of the refused ``(media_id, refusal)`` pairs: how many, and the first
    one's message."""
    by_reason: dict[str, list[MediaRefused]] = {}
    for _, refusal in refused:
        by_reason.setdefault(refusal.reason, []).append(refusal)
    return [
        f"  {reason}: {len(group)} (first: {group[0]})"
        for reason, group in sorted(by_reason.items())
    ]


def _relative_path(hf_path: str, media_id: str) -> str:
    """The path below the dataset root of a media path of the index (``hf://.../rounds/...``)."""
    head, marker, tail = str(hf_path).partition("/rounds/")
    if not marker or ".." in tail.split("/"):
        raise ValueError(f"OpenCS2 index: media {media_id} has the path {hf_path!r}")
    return f"rounds/{tail}"


def read_pov_rounds(root: str | Path, maps: Sequence[str]) -> list[dict]:
    """The player-rounds of ``maps`` in ``<root>/index/pov_rounds.parquet`` (needs pyarrow).

    Args:
        root (str | Path): the OpenCS2 download.
        maps (Sequence[str]): the maps to keep.

    Returns:
        list[dict]: per player-round ``media_id, match_id, map_name, round, player_slot,
        capture_start_tick, video_frames, media_bytes`` and ``video_path`` and ``ticks_path``
        (relative to ``root``), ordered by map, match, round and player slot.

    Raises:
        FileNotFoundError: the index is missing.
        ValueError: the index cannot be read, lacks a column, repeats a media id, or has none of a
            requested map.
    """
    import pyarrow.parquet as pq

    path = Path(root) / POV_INDEX
    if not path.is_file():
        raise FileNotFoundError(f"OpenCS2 index does not exist: {path}")
    try:
        names = pq.read_schema(path).names
        missing = [name for name in POV_COLUMNS if name not in names]
        if missing:
            raise ValueError(f"it lacks the columns {missing}")
        table = pq.read_table(path, columns=list(POV_COLUMNS)).to_pylist()
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the OpenCS2 index {path}: {first_line(error)}") from error
    present = {row["map_name"] for row in table}
    absent = sorted(set(maps) - present)
    if absent:
        raise ValueError(
            f"the OpenCS2 index {path} has no media of {absent} (maps: {sorted(present)})"
        )
    rows, seen = [], set()
    for row in table:
        if row["map_name"] not in maps:
            continue
        media_id = str(row["media_id"])
        if media_id in seen:
            raise ValueError(f"the OpenCS2 index {path} repeats media {media_id}")
        seen.add(media_id)
        rows.append(
            {
                "media_id": media_id,
                "match_id": int(row["match_id"]),
                "map_name": str(row["map_name"]),
                "round": int(row["round"]),
                "player_slot": int(row["player_slot"]),
                "capture_start_tick": int(row["capture_start_tick"]),
                "video_frames": int(row["video_frames"]),
                "media_bytes": int(row["media_bytes"]),
                "video_path": _relative_path(row["video"]["path"], media_id),
                "ticks_path": _relative_path(row["ticks_parquet_path"], media_id),
            }
        )
    rows.sort(key=lambda r: (r["map_name"], r["match_id"], r["round"], r["player_slot"]))
    return rows


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this step's options to its subcommand's parser."""
    parser.add_argument(
        "--dataset-root",
        required=True,
        metavar="DIR",
        help="the OpenCS2 download: index/pov_rounds.parquet and rounds/",
    )
    parser.add_argument("--out", required=True, metavar="FILE", help="the media index to write")
    parser.add_argument(
        "--maps",
        nargs="+",
        default=list(MAPS),
        metavar="MAP",
        help=f"the maps to index (default: the paper's, {' '.join(MAPS)})",
    )
    add_workers_argument(parser)


def _tick_table_facts(path: Path, media_id: str) -> tuple[int, int]:
    """``(rows, first tick)`` of a tick table."""
    import pyarrow.parquet as pq

    try:
        ticks = pq.read_table(path, columns=["tick"])
    except (OSError, ValueError) as error:
        raise MediaRefused(
            "ticks_unreadable",
            f"tick table of media {media_id} cannot be read ({path}): {first_line(error)}",
        ) from error
    if ticks.num_rows == 0:
        raise MediaRefused("ticks_empty", f"tick table of media {media_id} is empty: {path}")
    return ticks.num_rows, int(ticks.column("tick")[0].as_py())


def media_row(pov: dict, root: Path) -> dict:
    """The media-index row of one player-round of :func:`read_pov_rounds`, its files checked.

    Raises:
        MediaRefused: the video or the tick table fails a check (listed in the module docstring).
    """
    from worldcast.data.video import load_video_window

    media_id = pov["media_id"]
    video, ticks = root / pov["video_path"], root / pov["ticks_path"]
    if not video.is_file():
        raise MediaRefused("video_missing", f"video of media {media_id} not found: {video}")
    size = video.stat().st_size
    if size != pov["media_bytes"]:
        raise MediaRefused(
            "video_size",
            f"video of media {media_id} has {size} bytes, the OpenCS2 index says"
            f" {pov['media_bytes']}: {video}",
        )
    if not ticks.is_file():
        raise MediaRefused("ticks_missing", f"tick table of media {media_id} not found: {ticks}")
    rows, first_tick = _tick_table_facts(ticks, media_id)
    if first_tick != pov["capture_start_tick"]:
        raise MediaRefused(
            "ticks_start",
            f"tick table of media {media_id} starts at tick {first_tick}, the OpenCS2 index says"
            f" the capture starts at {pov['capture_start_tick']}: {ticks}",
        )
    try:
        load_video_window(video, 0, video_frames=1, source_frames=pov["video_frames"])
    except Exception as error:  # a decoder's own error type (decord) is not a RuntimeError
        raise MediaRefused("video_frames", f"media {media_id}: {first_line(error)}") from error
    return {
        "media_id": media_id,
        "match_id": pov["match_id"],
        "map_name": pov["map_name"],
        "round": pov["round"],
        "player_slot": pov["player_slot"],
        "fps": SOURCE_FPS,
        "video_frames": pov["video_frames"],
        "video_path": pov["video_path"],
        "ticks_path": pov["ticks_path"],
        "ticks_rows": rows,
        "ticks_file_size": ticks.stat().st_size,
        "ticks_sha256": sha256_file(ticks),
        "capture_start_tick": first_tick,
    }


def build_media_index(
    root: str | Path, maps: list[str], *, workers: int = DEFAULT_WORKERS
) -> tuple[list[dict], list[tuple[str, MediaRefused]]]:
    """The media-index rows of the player-rounds of ``maps`` in an OpenCS2 download.

    Args:
        root (str | Path): the download.
        maps (list[str]): the maps to index.
        workers (int): threads checking recordings; the rows do not depend on it.

    Returns:
        tuple[list[dict], list[tuple[str, MediaRefused]]]: the rows, ordered by map, match, round
        and player slot, and the left-out ``(media_id, refusal)`` pairs in the same order.

    Raises:
        FileNotFoundError: the download has no index.
        ValueError: the index cannot be read or has none of a map, or no player-round passes its
            checks.
    """
    root = Path(root)
    povs = read_pov_rounds(root, maps)
    results = check_all(lambda pov: media_row(pov, root), povs, workers, "media-index")
    rows = [result for result in results if not isinstance(result, MediaRefused)]
    refused = [
        (pov["media_id"], result)
        for pov, result in zip(povs, results)
        if isinstance(result, MediaRefused)
    ]
    if not rows:
        reasons = "\n".join(refusal_summary(refused))
        raise ValueError(f"no player-round of {maps} under {root} passes its checks:\n{reasons}")
    return rows, refused


def run(args: argparse.Namespace) -> int:
    """Write the media index; returns the exit status (an input that cannot be used raises
    ``OSError`` or ``ValueError``, which ``tools/prepare_data.py`` ends with status 2)."""
    rows, refused = build_media_index(args.dataset_root, args.maps, workers=args.workers)
    write_jsonl(args.out, rows)
    print(f"media-index: {len(rows)} rows -> {args.out}")
    if refused:
        print(f"media-index: left out {len(refused)} player-rounds:", file=sys.stderr)
        print("\n".join(refusal_summary(refused)), file=sys.stderr)
    return 0
