"""The flash and scope labels, computed from the recorded video.

Reads the media index (``data.media_index``), the recordings' videos and tick tables (under
``data.dataset_root``; the tick tables need pyarrow) and writes per recording one file in each of
``flashlabels/`` and ``scopelabels/`` under ``data.observer_signal_label_root``: one sample per
source frame (32 fps), in the format :mod:`worldcast.data.labels` reads (docs/inference.md, "Data").
The video is decoded to 64 x 36 gray, every frame. Flash: the screen's mean luminance, which the
reader compares with the file's ``hot_threshold``. Scope: the overlay on screen (a dark vignette
around a lit centre), the scope button and the weapon in hand from the tick table, and the zoom
``level`` they make: a press steps a scoped weapon's level, and the overlay on screen gates it.

A recording whose video runs at another fps than the media index says, or decodes to another frame
count, is reported and gets no file: the reader takes a scope file for the recording's frames only
when their counts agree. A file that exists and matches (the frame count, the thresholds) is
skipped, so a run can be repeated or resumed. ``--compare`` computes the labels and reports, per
array, whether the files under another label root hold the same.

Examples::

    python tools/prepare_data.py observer-labels --config train_paths.yaml --workers 16
    python tools/prepare_data.py observer-labels --config train_paths.yaml \\
        --media-id 2393226-de_mirage-r16-p00
    python tools/prepare_data.py observer-labels --videos clips --out-dir data/obslabels
    python tools/prepare_data.py observer-labels --config train_paths.yaml --compare paper/obslabels
"""

import argparse
import multiprocessing
import subprocess
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np

from worldcast.config.loader import add_config_args, build, load, parse_overrides, require_set
from worldcast.config.training import DataConfig
from worldcast.data.controls import (
    IGNORED_BUTTONS,
    OPENCS2_BUTTONS,
    AlignmentError,
    encode_weapon,
    frame_tick_ranges,
)
from worldcast.data.labels import flash_label_path, scope_label_path
from worldcast.data.latents import SOURCE_FPS
from worldcast.data.recordings import MediaIndex, MediaRecord, TickTable, read_player_ticks
from worldcast.utils.files import atomic_write

__all__ = [
    "CENTER_MIN",
    "CORNER_ON",
    "GRAY_SIZE",
    "HOT_THRESHOLD",
    "add_arguments",
    "compare_arrays",
    "decode_gray",
    "flash_labels",
    "flash_luminance",
    "run",
    "scope_labels",
    "scope_overlay",
    "video_fps",
]

#: The decoded frames' ``(width, height)``, pixels.
GRAY_SIZE = (64, 36)
#: A screen brighter than this is flash-white (luminance in [0, 1]).
HOT_THRESHOLD = 0.85
#: The scope is on screen where the four corner patches are darker than ``CORNER_ON`` and the
#: centre patch brighter than ``CENTER_MIN`` (mean gray in [0, 1]).
CORNER_ON = 0.04
CENTER_MIN = 0.08
#: The corner patches are 8 x 8 pixels; the centre patch spans rows 14-21 and columns 28-35.
_CORNER = 8
_CENTER_ROWS = slice(14, 22)
_CENTER_COLUMNS = slice(28, 36)
#: ffmpeg's decoder threads: a damaged video is concealed differently with another count (two, as
#: the paper's labels were made).
_DECODER_THREADS = 2
#: A video's fps may differ from the expected one by this much (ffmpeg reports it rounded).
_FPS_TOLERANCE = 1e-3
#: Largest gap between ticks that a source frame may span, s.
_MAX_TICK_GAP_SECONDS = 1.0
#: The last source frames of the tick table's span that the labels leave without controls.
_TICK_MARGIN = 2
#: A tick table that covers no more source frames than this gives no controls.
_MIN_COVERED_FRAMES = 8
#: The arrays of a flash file and of a scope file, in the order they are stored.
_FLASH_KEYS = ("lum", "stride", "hot_threshold", "n_hot")
_SCOPE_KEYS = (
    "corner_max",
    "center",
    "scoped_vis",
    "attack2",
    "weapon_id",
    "level",
    "ncov",
    "thresholds",
)
#: The per-frame arrays of a scope file.
_SCOPE_SERIES = _SCOPE_KEYS[:6]
#: Weapons whose scope zooms twice (levels 1 and 2), and once (level 1).
_TWO_LEVEL_WEAPONS = ("awp", "ssg08", "scar20", "g3sg1")
_ONE_LEVEL_WEAPONS = ("aug", "sg556")
#: How often the progress line is printed, recordings.
_PROGRESS_EVERY = 1000


# -------------------------------------------------------------------------------------------- video
def video_fps(path: str | Path) -> float:
    """Frames per second of the first video stream of ``path`` (0.0 when ffmpeg cannot tell).

    Args:
        path (str | Path): the video.

    Raises:
        FileNotFoundError: the video is missing.
        ValueError: ffmpeg cannot read the video's header.
    """
    import imageio_ffmpeg

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {path}")
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt="gray", bits_per_pixel=8)
    try:
        return float(next(frames)["fps"])
    except OSError as error:
        raise ValueError(f"cannot read the header of {path}: {error}") from error
    finally:
        frames.close()


def decode_gray(path: str | Path) -> np.ndarray:
    """Every frame of ``path`` scaled to :data:`GRAY_SIZE` in gray, decoded by the ffmpeg of
    ``imageio-ffmpeg``.

    Args:
        path (str | Path): the video.

    Returns:
        np.ndarray: ``[n, 36, 64]`` uint8, one row per decoded frame.

    Raises:
        FileNotFoundError: the video is missing.
        ValueError: ffmpeg cannot decode the video, or it holds no frame.
    """
    import imageio_ffmpeg

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {path}")
    width, height = GRAY_SIZE
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        *("-v", "error", "-threads", str(_DECODER_THREADS), "-i", str(path)),
        *("-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "gray", "-"),
    ]
    done = subprocess.run(command, capture_output=True)
    if done.returncode != 0:
        message = done.stderr.decode(errors="replace").strip().splitlines()
        raise ValueError(f"ffmpeg cannot decode {path}: {message[-1] if message else 'no message'}")
    frames = np.frombuffer(done.stdout, np.uint8)
    count = len(frames) // (width * height)
    if count == 0:
        raise ValueError(f"{path} holds no video frame")
    return frames[: count * width * height].reshape(count, height, width)


def flash_luminance(gray: np.ndarray) -> np.ndarray:
    """The screen's mean luminance per frame.

    Args:
        gray (np.ndarray): ``[n, 36, 64]`` uint8 frames (:func:`decode_gray`).

    Returns:
        np.ndarray: ``[n]`` float32 in [0, 1]; the labels store it as float16.
    """
    return (gray.reshape(len(gray), -1).mean(1) / 255.0).astype(np.float32)


def scope_overlay(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The scope overlay of each frame: the brightest of the four corner patches, the centre
    patch, and whether the scope is on screen.

    Args:
        gray (np.ndarray): ``[n, 36, 64]`` uint8 frames (:func:`decode_gray`).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``corner_max`` and ``center`` ``[n]`` float32
        mean gray in [0, 1], and ``scoped`` ``[n]`` bool: ``corner_max < CORNER_ON`` and
        ``center > CENTER_MIN``.
    """
    frames = gray.astype(np.float32) / 255.0
    corners = np.stack(
        [
            frames[:, :_CORNER, :_CORNER].mean((1, 2)),
            frames[:, :_CORNER, -_CORNER:].mean((1, 2)),
            frames[:, -_CORNER:, :_CORNER].mean((1, 2)),
            frames[:, -_CORNER:, -_CORNER:].mean((1, 2)),
        ],
        1,
    )
    corner_max = corners.max(1)
    center = frames[:, _CENTER_ROWS, _CENTER_COLUMNS].mean((1, 2))
    return corner_max, center, (corner_max < CORNER_ON) & (center > CENTER_MIN)


# ------------------------------------------------------------------------------------------- labels
def flash_labels(gray: np.ndarray) -> dict[str, np.ndarray]:
    """The flash label of a recording: its screen luminance per source frame.

    Args:
        gray (np.ndarray): ``[n, 36, 64]`` uint8 frames, one per source frame
            (:func:`decode_gray`).

    Returns:
        dict[str, np.ndarray]: the arrays of a flash file: ``lum`` ``[n]`` float16, ``stride`` (1)
        int16, ``hot_threshold`` float32 and ``n_hot`` int32, the samples above it.
    """
    lum = flash_luminance(gray)
    return {
        "lum": lum.astype(np.float16),
        "stride": np.array(1, np.int16),
        "hot_threshold": np.array(HOT_THRESHOLD, np.float32),
        "n_hot": np.array(int((lum > HOT_THRESHOLD).sum()), np.int32),
    }


def scope_labels(gray: np.ndarray, ticks: TickTable, source_fps: float) -> dict[str, np.ndarray]:
    """The scope label of a recording: the overlay on screen, the scope button, the weapon in hand
    and the zoom level, per source frame.

    A source frame owns the ticks in ``(t - 1 / source_fps, t]``; the button is held if any is
    holding it, the weapon is the one its last tick names (``<none>`` if it names none). Frames
    beyond the tick table's span (``ncov``) have no button and weapon ``-1``.

    Args:
        gray (np.ndarray): ``[n, 36, 64]`` uint8 frames, one per source frame.
        ticks (TickTable): the player's ticks, read with their recorded buttons.
        source_fps (float): source frames per second.

    Returns:
        dict[str, np.ndarray]: the arrays of a scope file: ``corner_max`` and ``center`` ``[n]``
        float16, ``scoped_vis``, ``attack2`` ``[n]`` uint8, ``weapon_id`` ``[n]`` int16, ``level``
        ``[n]`` int8 (0 unscoped, 1, 2), ``ncov`` int32 and ``thresholds`` ``[2]`` float32.

    Raises:
        AlignmentError: a source frame has no tick, or two ticks of the span are further apart than
            a tick gap allows, or a tick holds a button outside the OpenCS2 enum.
    """
    frames = len(gray)
    corner_max, center, scoped = scope_overlay(gray)
    covered = int((ticks.t[-1] - ticks.t[0]) * source_fps) - _TICK_MARGIN
    covered = max(0, min(frames, covered))
    attack2 = np.zeros(frames, bool)
    weapon = np.full(frames, -1, np.int16)
    if covered > _MIN_COVERED_FRAMES:
        attack2[:covered], weapon[:covered] = _frame_controls(ticks, covered, source_fps)
    return {
        "corner_max": corner_max.astype(np.float16),
        "center": center.astype(np.float16),
        "scoped_vis": scoped.astype(np.uint8),
        "attack2": attack2.astype(np.uint8),
        "weapon_id": weapon,
        "level": _zoom_levels(scoped, attack2, weapon),
        "ncov": np.array(covered, np.int32),
        "thresholds": np.array([CORNER_ON, CENTER_MIN], np.float32),
    }


def _frame_controls(
    ticks: TickTable, frames: int, source_fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """``attack2`` ``[frames]`` bool and ``weapon`` ``[frames]`` int16 of the first ``frames``
    source frames."""
    begin, end = frame_tick_ranges(
        timestamps=ticks.t,
        start_frame=0,
        num_frames=frames,
        source_fps=source_fps,
        stride=1,
        max_tick_gap_seconds=_MAX_TICK_GAP_SECONDS,
    )
    first, last = int(begin[0]), int(end[-1])
    unknown, held = set(), np.zeros(last - first, bool)
    for index, row in enumerate(ticks.active[first:last]):
        for button in row or ():
            name = "<null>" if button is None else str(button)
            if name == "attack2":
                held[index] = True
            elif name not in OPENCS2_BUTTONS and name not in IGNORED_BUTTONS:
                unknown.add(name)
    if unknown:
        raise AlignmentError(f"unknown OpenCS2 buttons: {sorted(unknown)}")
    cumulative = np.concatenate([[0], np.cumsum(held)])
    attack2 = cumulative[end - first] > cumulative[begin - first]
    named = np.array([encode_weapon(name) for name in ticks.input_weapon])
    return attack2, named[end - 1].astype(np.int16)


def _zoom_levels(scoped: np.ndarray, attack2: np.ndarray, weapon: np.ndarray) -> np.ndarray:
    """The zoom level per source frame: a press of the scope button steps the level of a scoped
    weapon (0, 1, 2, 0 for a two-level weapon, 0, 1, 0 for a one-level one), any other weapon is
    unscoped, and a level counts only where the overlay is on screen."""
    two = {encode_weapon(name) for name in _TWO_LEVEL_WEAPONS}
    one = {encode_weapon(name) for name in _ONE_LEVEL_WEAPONS}
    frames = len(scoped)
    on = scoped.tolist()
    pressed = np.concatenate([[False], attack2[1:] & ~attack2[:-1]]).tolist()
    level = np.zeros(frames, np.int8)
    zoom = 0
    for i, held in enumerate(weapon.tolist()):
        if held in two:
            if pressed[i]:
                zoom = 0 if zoom >= 2 else zoom + 1
        elif held in one:
            if pressed[i]:
                zoom = 0 if zoom >= 1 else 1
        else:
            zoom = 0
        # the engine drops the scope without a press (a shot's bolt cycle, a weapon switch): the
        # overlay is the truth for on and off, the level only counts while it is on
        if (
            zoom > 0
            and not on[i]
            and i + 1 < frames
            and not on[i + 1]
            and (i == 0 or not on[i - 1])
        ):
            zoom = 0
        level[i] = zoom if on[i] else 0
    return level


def compare_arrays(
    built: Mapping[str, np.ndarray], stored: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """The largest absolute difference of each array of two label files.

    Args:
        built (Mapping[str, np.ndarray]): the arrays of one file.
        stored (Mapping[str, np.ndarray]): the arrays of the other.

    Returns:
        dict[str, float]: per array of either file: 0.0 where equal, else the largest absolute
        difference, or ``inf`` where the array is missing from one file or the shapes differ.
    """
    diffs = {}
    for key in dict.fromkeys([*built, *stored]):
        if key not in built or key not in stored or np.shape(built[key]) != np.shape(stored[key]):
            diffs[key] = float("inf")
        elif np.array_equal(built[key], stored[key]):
            diffs[key] = 0.0
        else:
            delta = np.abs(np.asarray(built[key], np.float64) - np.asarray(stored[key], np.float64))
            diffs[key] = float(delta.max())
    return diffs


# -------------------------------------------------------------------------------------------- files
def _read(path: Path) -> dict[str, np.ndarray] | None:
    """The arrays of a label file, None if it is missing or unreadable."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None


def _write(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a label file in one step, so that a file that exists is complete."""
    atomic_write(path, lambda handle: np.savez_compressed(handle, **arrays))


def _flash_is_current(stored: Mapping[str, np.ndarray] | None, frames: int | None) -> bool:
    return (
        stored is not None
        and set(stored) == set(_FLASH_KEYS)
        and stored["lum"].ndim == 1
        and (frames is None or len(stored["lum"]) == frames)
        and np.array_equal(stored["stride"], 1)
        and np.array_equal(stored["hot_threshold"], np.float32(HOT_THRESHOLD))
    )


def _scope_is_current(stored: Mapping[str, np.ndarray] | None, frames: int) -> bool:
    return (
        stored is not None
        and set(stored) == set(_SCOPE_KEYS)
        and {stored[key].shape for key in _SCOPE_SERIES} == {(frames,)}
        and np.array_equal(stored["thresholds"], np.array([CORNER_ON, CENTER_MIN], np.float32))
    )


# ----------------------------------------------------------------------------------- the recordings
@dataclass(frozen=True)
class _Job:
    media_id: str
    video: Path | None
    record: MediaRecord | None


@dataclass(frozen=True)
class _Settings:
    dataset_root: Path | None
    out: Path | None
    compare: Path | None


@dataclass(frozen=True)
class _Result:
    media_id: str
    status: str  # built, skipped, failed or compared
    detail: str = ""
    diffs: Mapping[str, float] = field(default_factory=dict)


def _label_media(job: _Job, settings: _Settings) -> _Result:
    """Build, skip or compare the labels of one recording.

    Each file stands on its own inputs, flash on the video, scope on the video and the tick table:
    a problem of one is a ``failed`` result and the other is still made; a problem of the video
    fails both.
    """
    root = settings.compare or settings.out
    paths = {
        "flash": flash_label_path(root, job.media_id),
        "scope": scope_label_path(root, job.media_id),
    }
    kinds = ("flash",) if job.record is None else ("flash", "scope")
    if settings.compare is None:
        kinds = _outdated(job, paths)
        if not kinds:
            return _Result(job.media_id, "skipped")
    try:
        gray = _decode(job)
    except (OSError, ValueError, RuntimeError) as error:
        return _Result(job.media_id, "failed", str(error))
    built, problems = {}, []
    for kind in kinds:
        try:
            if kind == "flash":
                built[kind] = flash_labels(gray)
            else:
                ticks = read_player_ticks(job.record, settings.dataset_root, jump_recall=False)
                built[kind] = scope_labels(gray, ticks, job.record.fps)
        except (OSError, ValueError, RuntimeError) as error:
            problems.append(f"{kind}: {error}")
    diffs = {}
    for kind, arrays in built.items():
        if settings.compare is None:
            _write(paths[kind], arrays)
            continue
        stored = _read(paths[kind])
        if stored is None:
            problems.append(f"{kind}: no label file {paths[kind]}")
            continue
        for key, diff in compare_arrays(arrays, stored).items():
            diffs[f"{kind}/{key}"] = diff
    if problems:
        return _Result(job.media_id, "failed", "; ".join(problems), diffs)
    return _Result(job.media_id, "built" if settings.compare is None else "compared", "", diffs)


def _outdated(job: _Job, paths: Mapping[str, Path]) -> tuple[str, ...]:
    """The label files of a recording that are missing or do not match it."""
    frames = None if job.record is None else job.record.source_frames
    outdated = []
    if not _flash_is_current(_read(paths["flash"]), frames):
        outdated.append("flash")
    if job.record is not None and not _scope_is_current(_read(paths["scope"]), frames):
        outdated.append("scope")
    return tuple(outdated)


def _decode(job: _Job) -> np.ndarray:
    """The recording's frames in gray, after its video's fps and frame count are checked against
    the media index (or, without one, the recordings' fps)."""
    if job.video is None:
        raise ValueError("the media index has no video_path")
    expected = SOURCE_FPS if job.record is None else job.record.fps
    fps = video_fps(job.video)
    if abs(fps - expected) > _FPS_TOLERANCE:
        raise ValueError(f"the video runs at {fps:g} fps, expected {expected:g}")
    gray = decode_gray(job.video)
    if job.record is not None and len(gray) != job.record.source_frames:
        raise ValueError(
            f"the video decodes to {len(gray)} frames, the media index says"
            f" {job.record.source_frames}"
        )
    return gray


def _plan(args: argparse.Namespace) -> tuple[list[_Job], _Settings]:
    """The recordings to label and where their labels go, from the options and the config's
    ``data`` section; a ``ValueError`` or ``OSError`` says what is wrong with them."""
    data = load(
        args.config,
        parse_overrides(args.overrides),
        lambda raw: build(DataConfig, raw.get("data", {}), "data"),
    )
    if args.out_dir is None and args.compare is None:
        require_set(data, "data", ["observer_signal_label_root"])
    out = args.out_dir or (None if args.compare else Path(data.observer_signal_label_root))
    if args.videos is not None:
        if not args.videos.is_dir():
            raise ValueError(f"--videos is not a directory: {args.videos}")
        jobs = {path.stem: _Job(path.stem, path, None) for path in args.videos.glob("*.mp4")}
        dataset_root = None
    else:
        require_set(data, "data", ["media_index", "dataset_root"])
        dataset_root = Path(data.dataset_root)
        jobs = {}
        for media_id, record in MediaIndex.load(data.media_index).by_id.items():
            video = None if record.video_path is None else dataset_root / record.video_path
            jobs[media_id] = _Job(media_id, video, record)
    if args.media_id is not None:
        absent = sorted(set(args.media_id) - set(jobs))
        if absent:
            raise ValueError(f"not among the recordings: {', '.join(absent)}")
        jobs = {media_id: jobs[media_id] for media_id in args.media_id}
    if not jobs:
        raise ValueError("no recordings to label")
    return [jobs[media_id] for media_id in sorted(jobs)], _Settings(dataset_root, out, args.compare)


def _results(jobs: Sequence[_Job], settings: _Settings, workers: int) -> Iterator[_Result]:
    """The results in the order of ``jobs``, one process per worker."""
    label = partial(_label_media, settings=settings)
    if workers == 1:
        yield from map(label, jobs)
        return
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        yield from pool.map(label, jobs)
    finally:
        pool.shutdown(cancel_futures=True)


# ---------------------------------------------------------------------------------------------- CLI
def _workers(text: str) -> int:
    count = int(text)
    if count < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {count}")
    return count


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this step's options to its subcommand's parser."""
    add_config_args(parser)
    parser.add_argument(
        "--videos",
        type=Path,
        help="a directory of <media_id>.mp4 in place of the media index: the flash labels only"
        " (the scope labels need the tick tables)",
    )
    sink = parser.add_mutually_exclusive_group()
    sink.add_argument(
        "--out-dir",
        type=Path,
        help="the label root to write, flashlabels/ and scopelabels/ under it"
        " (default data.observer_signal_label_root)",
    )
    sink.add_argument(
        "--compare",
        type=Path,
        help="a label root to compare the computed labels with, instead of writing: per array,"
        " equal or the largest absolute difference",
    )
    parser.add_argument(
        "--media-id", nargs="+", action="extend", metavar="ID", help="only these recordings"
    )
    parser.add_argument(
        "--workers", type=_workers, default=1, help="processes, one recording each at a time"
    )


def run(args: argparse.Namespace) -> int:
    """Label the recordings, or compare the labels with another root's; returns the exit status: 1
    if a recording failed or an array differs (options or a config that are refused raise
    ``OSError`` or ``ValueError``, which ``tools/prepare_data.py`` ends with status 2)."""
    jobs, settings = _plan(args)
    counts, compared, differing = Counter(), Counter(), Counter()
    worst: dict[str, float] = {}
    try:
        for done, result in enumerate(_results(jobs, settings, args.workers), start=1):
            counts[result.status] += 1
            if result.status == "failed":
                print(f"FAILED {result.media_id}: {result.detail}", flush=True)
            for key, diff in result.diffs.items():
                compared[key] += 1
                differing[key] += diff > 0
                worst[key] = max(worst.get(key, 0.0), diff)
            if done % _PROGRESS_EVERY == 0:
                print(f"{done}/{len(jobs)} recordings", flush=True)
    except ModuleNotFoundError as error:  # pyarrow, the tick tables' reader
        raise SystemExit(
            f"error: {error}: install the package with its dependencies, pip install -e ."
        ) from error
    for key, largest in worst.items():
        verdict = f"differs in {differing[key]}" if largest else "equal in all"
        note = f", largest absolute difference {largest:.6g}" if largest else ""
        print(f"{key}: {verdict} of {compared[key]} recordings{note}")
    print(", ".join(f"{count} {status}" for status, count in sorted(counts.items())))
    return int(counts["failed"] > 0 or any(worst.values()))
