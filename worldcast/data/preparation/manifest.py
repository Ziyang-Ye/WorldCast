"""The raw-video manifest of stages 1 and 1_long.

Reads the media index (``media-index``) and, for every recording, its tick table and its video, and
writes the windows stage 1 reads (docs/training.md, "Data formats"): JSON lines, one record per
recording, ordered by map, match, round and player slot. A window has ``pixel_frames`` video
frames, every second source frame (81 in stage 1, 161 in stage 1_long, ``--stage``), and starts at
a multiple of ``--stride`` source frames (8, one latent frame) inside the video.

A window is kept when the reader of the stage can serve it: every video frame owns a tick, the
last tick of a frame is within 1.5 ticks of it and no gap between ticks of the window is longer.
As in the paper's manifests, a window is left out when its ticks hold a button that OpenCS2 does not
list, the keyboard turns ``turn_left`` and ``turn_right`` among them, and stage 1_long keeps only
the windows in which the player is alive at every tick. A recording is left out, and counted by
reason in the summary, when its tick table is missing or differs from the media index (size,
sha256, rows), when its video is missing or shorter than a window, when its tick times or turns are
not valid, or when no window is left. ``--decode-scan`` also leaves out the videos that do not
decode end to end with ffmpeg or do not hold the media index's frame count.

Examples::

    python tools/prepare_data.py manifest --dataset-root data/opencs2 \\
        --media-index data/media_index.jsonl --out data/stage1/train.jsonl
    python tools/prepare_data.py manifest --dataset-root data/opencs2 \\
        --media-index data/media_index.jsonl --stage 1_long --out data/stage1_long/train.jsonl
    python tools/prepare_data.py manifest --dataset-root data/opencs2 \\
        --media-index data/media_index.jsonl --out data/stage1/train.jsonl \\
        --exclude-media configs/data/eval_reserve.txt --decode-scan \\
        --excluded-out data/excluded.txt
"""

import argparse
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from worldcast.config.training import STAGES
from worldcast.data.controls import (
    FRAME_TIME_SLACK,
    OPENCS2_BUTTONS,
    TICK_GATE_SECONDS,
    AlignmentError,
    count_until,
    frame_tick_ranges,
)
from worldcast.data.latents import (
    SOURCE_FRAMES_PER_LATENT,
    SOURCE_FRAMES_PER_VIDEO_FRAME,
    source_span,
    video_frame_count,
)
from worldcast.data.recordings import MediaIndex, MediaRecord, read_tick_columns, verify_ticks_file
from worldcast.utils.files import write_jsonl

from .media_index import (
    DEFAULT_WORKERS,
    MediaRefused,
    add_workers_argument,
    check_all,
    first_line,
    refusal_summary,
)

__all__ = [
    "RAW_VIDEO_STAGES",
    "STAGE_REQUIRES_ALIVE",
    "Manifest",
    "add_arguments",
    "build_manifest",
    "manifest_record",
    "run",
    "stage_pixel_frames",
]

#: The stages that train on raw video.
RAW_VIDEO_STAGES = tuple(name for name, stage in STAGES.items() if not stage.player_state_field)
#: Whether a stage's manifest keeps only the windows in which the player is alive at every tick.
STAGE_REQUIRES_ALIVE = {"1": False, "1_long": True}
#: The tick columns the controls of a window are read from.
CONTROL_COLUMNS = ("t", "active", "delta_pitch", "delta_yaw", "input_weapon")
#: ffmpeg may take this long to decode one video, seconds.
DECODE_TIMEOUT_SECONDS = 300


def stage_pixel_frames(stage: str) -> int:
    """The video frames of a window of a stage that trains on raw video: 81 in stage 1, 161 in
    stage 1_long."""
    if stage not in RAW_VIDEO_STAGES:
        raise ValueError(f"stage must be one of {RAW_VIDEO_STAGES}, got {stage!r}")
    return video_frame_count(STAGES[stage].latent_frames)


@dataclass(frozen=True)
class Manifest:
    """The result of :func:`build_manifest`.

    Attributes:
        records (list[dict]): the manifest records, ordered by map, match, round and player slot.
        refused (list[tuple[str, MediaRefused]]): the recordings left out, ``(media_id,
            refusal)``, in the same order.
        dropped_windows (dict[str, int]): the candidate windows of the kept recordings that are
            left out: ``ticks`` (the stage's reader would refuse their ticks), ``buttons`` (a
            button OpenCS2 does not list) and ``alive`` (``require_alive``).
    """

    records: list[dict]
    refused: list[tuple[str, MediaRefused]]
    dropped_windows: dict[str, int]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this step's options to its subcommand's parser."""
    parser.add_argument(
        "--dataset-root",
        required=True,
        metavar="DIR",
        help="the OpenCS2 download; the paths of the media index are relative to it",
    )
    parser.add_argument(
        "--media-index", required=True, metavar="FILE", help="the media index (media-index)"
    )
    parser.add_argument("--out", required=True, metavar="FILE", help="the manifest to write")
    parser.add_argument(
        "--stage",
        choices=RAW_VIDEO_STAGES,
        default="1",
        help="the stage that reads the manifest: windows of 81 (stage 1, 5 s) or 161 (stage 1_long,"
        " 10 s, the player alive throughout) video frames (default: %(default)s)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=SOURCE_FRAMES_PER_LATENT,
        help="source frames between the starts of a recording's windows (default: %(default)s,"
        " one latent frame)",
    )
    parser.add_argument(
        "--maps", nargs="+", metavar="MAP", help="only these maps (default: every map of the index)"
    )
    parser.add_argument(
        "--exclude-media",
        metavar="FILE",
        help="media ids to leave out, one per line: the evaluation reserve"
        " (data.exclude_media_manifest)",
    )
    parser.add_argument(
        "--decode-scan",
        action="store_true",
        help="decode every video end to end with ffmpeg and leave out the ones that fail",
    )
    parser.add_argument(
        "--excluded-out",
        metavar="FILE",
        help="write the ids of the recordings left out by a check, one per line (the format of"
        " --exclude-media)",
    )
    add_workers_argument(parser)


def _unlisted_button_prefix(active: Sequence) -> np.ndarray:
    """``[n + 1]`` int64: how many of the first ticks hold a button that OpenCS2 does not list."""
    listed = set(OPENCS2_BUTTONS)
    unlisted = [any(str(button) not in listed for button in row or ()) for row in active]
    return np.concatenate(([0], np.cumsum(unlisted, dtype=np.int64)))


def _decoded_frames(progress: str) -> int:
    """The last frame count of ffmpeg's ``-progress`` output."""
    frames = 0
    for line in progress.splitlines():
        key, _, value = line.partition("=")
        if key == "frame" and value.strip().isdigit():
            frames = int(value)
    return frames


def _check_decodes(media: MediaRecord, video: Path, ffmpeg: str) -> None:
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-max_error_rate",
        "0",
        "-threads",
        "1",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-progress",
        "pipe:1",
        "-nostats",
        "-f",
        "null",
        os.devnull,
    ]
    try:
        done = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=DECODE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise MediaRefused(
            "decode_timeout", f"decoding {video} took longer than {DECODE_TIMEOUT_SECONDS} s"
        ) from error
    if done.returncode != 0:
        detail = done.stderr.strip()[-300:] or f"ffmpeg exits with {done.returncode}"
        raise MediaRefused("decode_error", f"{video} does not decode: {detail}")
    decoded = _decoded_frames(done.stdout)
    if decoded != media.source_frames:
        raise MediaRefused(
            "decode_frames",
            f"{video} decodes to {decoded} frames, the media index says {media.source_frames}",
        )


def _read_ticks(media: MediaRecord, ticks_path: Path, columns: Sequence[str]) -> dict:
    """The tick columns ``columns`` of a recording, checked: the times must be finite and strictly
    increasing and the turns finite."""
    try:
        ticks = read_tick_columns(ticks_path, columns, media.ticks_rows)
    except RuntimeError as error:
        raise MediaRefused("ticks_rows", str(error)) from error
    except (OSError, ValueError) as error:
        raise MediaRefused(
            "ticks_unreadable", f"cannot read the tick table {ticks_path}: {first_line(error)}"
        ) from error
    timestamps = np.asarray(ticks["t"], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise MediaRefused(
            "ticks_time", f"tick times are not finite and strictly increasing: {ticks_path}"
        )
    turns = np.asarray([ticks["delta_pitch"], ticks["delta_yaw"]], dtype=np.float32)
    if not np.all(np.isfinite(turns)):
        raise MediaRefused("ticks_turn", f"tick turns are not finite: {ticks_path}")
    return ticks


def _window_starts(
    media: MediaRecord, ticks: dict, *, pixel_frames: int, stride: int, require_alive: bool
) -> tuple[list[int], dict[str, int]]:
    """The start frames of the windows of a recording that the manifest keeps, and how many
    candidates it leaves out, by ``ticks``, ``buttons`` and ``alive``."""
    timestamps = np.asarray(ticks["t"], dtype=np.float64)
    unlisted = _unlisted_button_prefix(ticks["active"])
    dead = None
    if require_alive:
        not_alive = [alive is not True for alive in ticks["is_alive"]]
        dead = np.concatenate(([0], np.cumsum(not_alive, dtype=np.int64)))
    starts, dropped = [], Counter()
    for start in range(0, media.source_frames - source_span(pixel_frames) + 1, stride):
        try:
            begin, end = frame_tick_ranges(
                timestamps=timestamps,
                start_frame=start,
                num_frames=pixel_frames,
                source_fps=media.fps,
                stride=SOURCE_FRAMES_PER_VIDEO_FRAME,
                max_tick_gap_seconds=TICK_GATE_SECONDS,
            )
        except AlignmentError:
            dropped["ticks"] += 1
            continue
        # as the paper's manifests: a keyboard turn bind leaves a window out too
        if unlisted[end[-1]] != unlisted[begin[0]]:
            dropped["buttons"] += 1
            continue
        if dead is not None:
            first = np.searchsorted(timestamps, start / media.fps - FRAME_TIME_SLACK, side="left")
            last = count_until(timestamps, (start + source_span(pixel_frames) - 1) / media.fps)
            if last <= first or dead[last] != dead[first]:
                dropped["alive"] += 1
                continue
        starts.append(start)
    return starts, dict(dropped)


def manifest_record(
    media: MediaRecord,
    root: Path,
    *,
    pixel_frames: int,
    stride: int,
    require_alive: bool = False,
    ffmpeg: str | None = None,
) -> tuple[dict, dict[str, int]]:
    """The manifest record of one recording, and how many of its candidate windows are left out.

    Args:
        media (MediaRecord): its media-index row; ``video_path`` must be set.
        root (Path): the OpenCS2 download.
        pixel_frames (int): video frames of a window.
        stride (int): source frames between window starts.
        require_alive (bool): keep only the windows in which the player is alive at every tick.
        ffmpeg (str | None): the ffmpeg executable of the decode scan; None: no scan.

    Returns:
        tuple[dict, dict[str, int]]: the record, and the candidate windows left out by ``ticks``
        (the reader would refuse their ticks), ``buttons`` (a button OpenCS2 does not list) and
        ``alive``.

    Raises:
        MediaRefused: the recording is left out (the module docstring lists the reasons).
    """
    ticks_path, video = root / media.ticks_path, root / str(media.video_path)
    try:
        digest = verify_ticks_file(media, ticks_path)
    except FileNotFoundError as error:
        raise MediaRefused("ticks_missing", str(error)) from error
    except RuntimeError as error:
        raise MediaRefused("ticks_mismatch", str(error)) from error
    if not video.is_file():
        raise MediaRefused("video_missing", f"video of media {media.media_id} not found: {video}")
    span = source_span(pixel_frames)
    if media.source_frames < span:
        raise MediaRefused(
            "short_video",
            f"video of media {media.media_id} has {media.source_frames} frames, a window needs"
            f" {span}",
        )
    columns = (*CONTROL_COLUMNS, "is_alive") if require_alive else CONTROL_COLUMNS
    ticks = _read_ticks(media, ticks_path, columns)
    starts, dropped = _window_starts(
        media, ticks, pixel_frames=pixel_frames, stride=stride, require_alive=require_alive
    )
    if not starts:
        raise MediaRefused("no_window", f"no window of media {media.media_id} can be served")
    if ffmpeg is not None:
        _check_decodes(media, video, ffmpeg)
    record = {
        "media_id": media.media_id,
        "match_id": media.match_id,
        "map_name": media.map_name,
        "round": media.round,
        "player_slot": media.player_slot,
        "video_path": media.video_path,
        "ticks_path": media.ticks_path,
        "video_frames": media.source_frames,
        "video_file_size": video.stat().st_size,
        "ticks_file_size": media.ticks_file_size,
        "ticks_sha256": digest,
        "ticks_rows": media.ticks_rows,
        "fps": media.fps,
        "pixel_frames": pixel_frames,
        "skip_frame": SOURCE_FRAMES_PER_VIDEO_FRAME,
        "stride": stride,
        "max_tick_gap_seconds": TICK_GATE_SECONDS,
        "start_frames": starts,
        "n_windows": len(starts),
    }
    return record, dropped


def build_manifest(
    records: Sequence[MediaRecord],
    root: str | Path,
    *,
    pixel_frames: int,
    stride: int = SOURCE_FRAMES_PER_LATENT,
    require_alive: bool = False,
    decode_scan: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> Manifest:
    """The raw-video manifest of recordings.

    Args:
        records (Sequence[MediaRecord]): the recordings (media-index rows with ``video_path``).
        root (str | Path): the OpenCS2 download.
        pixel_frames (int): video frames of a window (:func:`stage_pixel_frames`).
        stride (int): source frames between window starts.
        require_alive (bool): keep only the windows in which the player is alive at every tick.
        decode_scan (bool): decode every video end to end with ffmpeg.
        workers (int): threads checking recordings; the manifest does not depend on it.

    Returns:
        Manifest: the records of the recordings that have a window, ordered by map, match, round
        and player slot.

    Raises:
        ValueError: a record has no ``video_path``, ``stride`` or ``pixel_frames`` is not
            positive, or no recording has a window.
    """
    if stride < 1 or pixel_frames < 1:
        raise ValueError(
            f"stride and pixel_frames must be positive, got {stride} and {pixel_frames}"
        )
    unset = [media.media_id for media in records if media.video_path is None]
    if unset:
        raise ValueError(
            f"the media index has no video_path for {len(unset)} media, e.g. {unset[0]}"
        )
    ffmpeg = None
    if decode_scan:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    root = Path(root)
    ordered = sorted(records, key=lambda m: (m.map_name, m.match_id, m.round, m.player_slot))
    results = check_all(
        lambda media: manifest_record(
            media,
            root,
            pixel_frames=pixel_frames,
            stride=stride,
            require_alive=require_alive,
            ffmpeg=ffmpeg,
        ),
        ordered,
        workers,
        "manifest",
    )
    kept = [result for result in results if not isinstance(result, MediaRefused)]
    refused = [
        (media.media_id, result)
        for media, result in zip(ordered, results)
        if isinstance(result, MediaRefused)
    ]
    if not kept:
        reasons = "\n".join(refusal_summary(refused))
        raise ValueError(f"no recording has a window of {pixel_frames} video frames:\n{reasons}")
    dropped = Counter()
    for _, counts in kept:
        dropped.update(counts)
    return Manifest([record for record, _ in kept], refused, dict(sorted(dropped.items())))


def _read_media_index(path: str) -> MediaIndex:
    try:
        return MediaIndex.load(path)
    except KeyError as error:
        raise ValueError(f"the media index {path} lacks the field {error}") from error


def run(args: argparse.Namespace) -> int:
    """Write the manifest; returns the exit status (an input that cannot be used raises
    ``OSError`` or ``ValueError``, which ``tools/prepare_data.py`` ends with status 2)."""
    from worldcast.data.training import read_media_exclusion

    records = list(_read_media_index(args.media_index).by_id.values())
    if args.maps:
        absent = sorted(set(args.maps) - {media.map_name for media in records})
        if absent:
            raise ValueError(f"the media index {args.media_index} has no media of {absent}")
        records = [media for media in records if media.map_name in args.maps]
    if args.exclude_media:
        excluded = read_media_exclusion(args.exclude_media)
        records = [media for media in records if media.media_id not in excluded]
    manifest = build_manifest(
        records,
        args.dataset_root,
        pixel_frames=stage_pixel_frames(args.stage),
        stride=args.stride,
        require_alive=STAGE_REQUIRES_ALIVE[args.stage],
        decode_scan=args.decode_scan,
        workers=args.workers,
    )
    write_jsonl(args.out, manifest.records)
    if args.excluded_out:
        ids = "".join(f"{media_id}\n" for media_id, _ in manifest.refused)
        Path(args.excluded_out).write_text(ids, encoding="utf-8")
    windows = sum(record["n_windows"] for record in manifest.records)
    print(
        f"manifest: stage {args.stage}, {len(manifest.records)} recordings, {windows} windows"
        f" -> {args.out}"
    )
    if manifest.refused:
        print(f"manifest: left out {len(manifest.refused)} recordings:", file=sys.stderr)
        print("\n".join(refusal_summary(manifest.refused)), file=sys.stderr)
    for kind, count in manifest.dropped_windows.items():
        print(f"manifest: left out {count} windows by {kind}", file=sys.stderr)
    return 0
