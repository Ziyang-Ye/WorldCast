"""The training windows: the bucket windows over the latent cache and the raw-video windows.

* :class:`BucketWindows` (stages 2-4): the 41-latent client windows of the bucket index over the
  latent cache, with every player's tick table, the GT visibility labels and the observer signals;
  with memory frames, every served window carries an accepted target block and its memory frames
  (:mod:`worldcast.data.memory_selection`);
* :class:`RawVideoWindows` (stage 1): video windows of a raw-video manifest, decoded from the
  videos.

:mod:`worldcast.data.stream` draws them.
"""

import bisect
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .controls import AlignmentError, align_ticks_to_video_frames
from .game import TICK_RATE
from .latent_cache import load_window_latents
from .latents import SOURCE_FRAMES_PER_VIDEO_FRAME, source_span, window_key
from .memory_selection import (
    MemoryFrameSource,
    draw_target,
    memory_frame_item,
    select_memory_frames,
)
from .recordings import (
    MediaIndex,
    MediaRecord,
    RoundIndexRow,
    read_jsonl,
    read_tick_columns,
    verify_ticks_file,
)
from .video import load_video_window
from .window import DataPaths, WindowRefused, WindowSpec, load_client_window

__all__ = [
    "BUCKET_NAMES",
    "BUCKET_WEIGHTS",
    "DENSITY_EXPONENT",
    "RAW_VIDEO_MAX_TICK_GAP_SECONDS",
    "RETRY_ATTEMPTS",
    "RETRY_ATTEMPTS_MEMORY",
    "RETRY_STRIDE",
    "ZERO_PAIR_SHARE",
    "BucketWindow",
    "BucketWindows",
    "RawVideoWindows",
    "apply_zero_pair_floor",
    "read_bucket_index",
    "read_media_exclusion",
]

#: The bucket files of the training split, ``train_<name>.jsonl``, in dataset-index order. A file is
#: a slot: the paper's index holds one map in each of the first four (de_dust2, de_mirage, de_nuke,
#: de_ancient) and nothing in the fifth; the names do not mean a range of any score.
BUCKET_NAMES = ("q00-01", "q01-03", "q03-05", "q05-07", "q07-10")
#: The sampling weight of each file's windows, as the paper's stages 2-4 drew them: solved for the
#: paper's four files so that, after the zero-pair floor, each map is drawn a quarter of the time.
#: ``data.bucket_weights`` sets them for other files.
BUCKET_WEIGHTS = (0.687417402, 0.86527334, 1.0, 0.92999783, 1.0)
#: A window's weight is its bucket's times ``(1 + visible pairs) ** DENSITY_EXPONENT``.
DENSITY_EXPONENT = 0.5
#: The windows without visible pairs are boosted to at least this share of the total weight.
ZERO_PAIR_SHARE = 0.15
#: A window that cannot be served is replaced by ``(index + attempt * RETRY_STRIDE) % n``.
RETRY_STRIDE = 7919
#: Attempts per item: without memory frames, and with (most windows have no acceptable target).
RETRY_ATTEMPTS = 8
RETRY_ATTEMPTS_MEMORY = 64
#: Largest tolerated tick gap and frame-endpoint lag of the raw-video windows of stage 1, seconds.
RAW_VIDEO_MAX_TICK_GAP_SECONDS = 1.5 / TICK_RATE


# ==================================================================== the bucket index and weights
def _visible_pairs(row: Mapping, where: str) -> int:
    """The row's visible player pairs; a row without labels is refused, never read as 0."""
    if not row.get("vis_available"):
        raise ValueError(f"{where} has no visibility labels (vis_available)")
    count = row.get("vis_pair_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{where} has an invalid vis_pair_count {count!r}")
    return count


def apply_zero_pair_floor(weights: Sequence[float], pair_counts: Sequence[int]) -> list[float]:
    """Scale the weights of the windows without visible pairs by one factor so that they hold
    :data:`ZERO_PAIR_SHARE` of the total; their share is never lowered."""
    zero_mass = sum(w for w, c in zip(weights, pair_counts) if c == 0)
    total = sum(weights)
    if zero_mass == 0 or zero_mass / total >= ZERO_PAIR_SHARE:
        return [float(w) for w in weights]
    # boost * Z / (boost * Z + P) == share, P = the weight of the windows with visible pairs
    populated = total - zero_mass
    boost = (ZERO_PAIR_SHARE / (1.0 - ZERO_PAIR_SHARE)) * (populated / zero_mass)
    return [float(w * boost) if c == 0 else float(w) for w, c in zip(weights, pair_counts)]


def read_media_exclusion(path: str | Path) -> frozenset:
    """Media ids excluded from training (the held-out media): one per line."""
    media = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not media or len(set(media)) != len(media):
        raise ValueError(f"{path} must list distinct media ids")
    return frozenset(media)


@dataclass(frozen=True)
class BucketWindow:
    """One window of the bucket index: ``media_id``'s 41-latent window from source frame
    ``start_frame``, its round (``match_id``, ``round``) and its sampling ``weight``."""

    media_id: str
    start_frame: int
    match_id: int
    round: int
    weight: float


def read_bucket_index(
    bucket_dir: str | Path,
    exclusion: frozenset = frozenset(),
    bucket_weights: Sequence[float] | None = None,
) -> list[BucketWindow]:
    """The windows of ``<bucket_dir>/train_<bucket>.jsonl``, bucket by bucket, line by line.

    Each line holds ``media_id``, ``start_frame``, ``match_id``, ``round``, optionally
    ``latent_key``, and ``vis_available`` and ``vis_pair_count``. A window's weight is its bucket's
    times ``(1 + vis_pair_count) ** 0.5``, the zero-pair windows floored to 15 % of the total.
    Excluded media are dropped first; each must occur.

    Args:
        bucket_dir (str | Path): the directory of the five bucket files.
        exclusion (frozenset): media ids to leave out (:func:`read_media_exclusion`).
        bucket_weights (Sequence[float] | None): the weight of each file, in the order of
            :data:`BUCKET_NAMES`; ``None``: :data:`BUCKET_WEIGHTS`.

    Returns:
        list[BucketWindow]: in dataset index order.
    """
    rows, weights, pairs, seen, excluded = [], [], [], set(), set()
    file_weights = BUCKET_WEIGHTS if bucket_weights is None else bucket_weights
    for name, bucket_weight in zip(BUCKET_NAMES, file_weights, strict=True):
        path = Path(bucket_dir) / f"train_{name}.jsonl"
        for line_number, record in read_jsonl(path, "bucket file"):
            media_id, start_frame = str(record["media_id"]), int(record["start_frame"])
            if (media_id, start_frame) in seen:
                raise ValueError(f"{path}:{line_number} repeats window {media_id}@{start_frame}")
            seen.add((media_id, start_frame))
            if media_id in exclusion:
                excluded.add(media_id)
                continue
            count = _visible_pairs(record, f"{path}:{line_number}")
            if str(record.get("latent_key", window_key(start_frame))) != window_key(start_frame):
                raise ValueError(f"{path}:{line_number}: latent_key must be win_<start_frame>")
            rows.append((media_id, start_frame, int(record["match_id"]), int(record["round"])))
            weights.append(float(bucket_weight) * (1.0 + count) ** DENSITY_EXPONENT)
            pairs.append(count)
    if not rows:
        raise ValueError(f"no training windows under {bucket_dir}")
    missing = sorted(exclusion - excluded)
    if missing:
        raise ValueError(f"{len(missing)} excluded media are not in the buckets: {missing[:3]}")
    weights = apply_zero_pair_floor(weights, pairs)
    return [BucketWindow(*row, weight) for row, weight in zip(rows, weights)]


# =========================================================================== bucket windows (2-4)
class BucketWindows(Dataset):
    """The training windows of the bucket index over the latent cache (stages 2-4).

    ``__getitem__(index)`` returns the item (no batch axis): ``latents`` ``[41, 48, 24, 42]``
    float32, the keys of :meth:`worldcast.data.window.WindowItem.batch_dict` over the window's 161
    video frames (``metadata`` also holds ``dataset_index``, the index served) and, with memory
    frames, the keys of :func:`worldcast.data.memory_selection.memory_frame_item`.

    Args:
        windows (Sequence[BucketWindow]): :func:`read_bucket_index`.
        media_index (MediaIndex): every recording.
        paths (DataPaths): the data artefacts.
        spec (WindowSpec): the windows' sampling (41 latent frames, the stage's camera encoding).
        memory_frames (MemoryFrameSource | None): the source of the memory frames; every served
            window then has an accepted target block.
    """

    def __init__(
        self,
        windows: Sequence[BucketWindow],
        *,
        media_index: MediaIndex,
        paths: DataPaths,
        spec: WindowSpec,
        memory_frames: MemoryFrameSource | None = None,
    ) -> None:
        self.windows = list(windows)
        self.media_index = media_index
        self.paths = paths
        self.spec = spec
        self.memory_frames = memory_frames
        for w in self.windows:
            media = media_index.media(w.media_id)
            if (w.match_id, w.round) != (media.match_id, media.round):
                raise ValueError(f"window {w.media_id}@{w.start_frame} disagrees with its round")

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def sample_weights(self) -> list[float]:
        """The sampling weight of every window, in dataset-index order."""
        return [w.weight for w in self.windows]

    def __getitem__(self, index: int) -> dict:
        """The item of window ``index``, or of the first servable probe after it."""
        attempts = RETRY_ATTEMPTS if self.memory_frames is None else RETRY_ATTEMPTS_MEMORY
        for attempt in range(attempts):
            try:
                return self.load_window((index + attempt * RETRY_STRIDE) % len(self))
            except (AlignmentError, WindowRefused):
                pass
        raise RuntimeError(f"{attempts} windows from index {index} could not be served")

    def load_window(self, index: int) -> dict:
        """One attempt at window ``index``; raises when it cannot be served (the client's ticks end
        early, or, with memory frames, no target block is acceptable)."""
        w = self.windows[index]
        media = self.media_index.media(w.media_id)
        row = RoundIndexRow(
            media_id=media.media_id,
            start_frame=int(w.start_frame),
            match_id=media.match_id,
            round=media.round,
            map_name=media.map_name,
            player_slot=media.player_slot,
        )
        # a recording without observer-signal labels reads as unknown, as trained
        window = load_client_window(
            row,
            self.media_index,
            self.paths,
            self.spec,
            missing_signals_ok=True,
        )
        item = window.item.batch_dict()
        item["metadata"]["dataset_index"] = int(index)
        item["latents"] = load_window_latents(
            self.paths.latent_cache_root, media.media_id, row.start_frame
        )
        if self.memory_frames is not None:
            selection = select_memory_frames(self.memory_frames, window)
            target = draw_target(selection, dataset_index=index, start_frame=row.start_frame)
            if target is None:
                raise WindowRefused(f"{w.media_id}@{row.start_frame}: no acceptable target block")
            item.update(memory_frame_item(self.memory_frames, window, selection, target))
        return item


# ============================================================================= raw video (stage 1)
#: The fields of a raw-video manifest record (docs/data.md). ``video_frames`` counts the
#: recording's source frames and ``pixel_frames`` the window's video frames. A record may also name
#: ``skip_frame``, the source frames per video frame, which the paper fixes at 2.
_MANIFEST_FIELDS = (
    "media_id",
    "match_id",
    "map_name",
    "round",
    "player_slot",
    "video_path",
    "ticks_path",
    "video_frames",
    "video_file_size",
    "ticks_file_size",
    "ticks_sha256",
    "ticks_rows",
    "fps",
    "pixel_frames",
    "stride",
    "max_tick_gap_seconds",
    "start_frames",
    "n_windows",
)


class RawVideoWindows(Dataset):
    """The stage-1 windows of a raw-video manifest, decoded from the videos.

    The manifest is JSONL, one record per media (docs/data.md): the fields of
    :data:`_MANIFEST_FIELDS`, and ``sample_weight`` per record or ``window_sample_weights`` per
    window (all records or none). Relative paths resolve against ``dataset_root``.

    The item: ``buttons`` ``[T, 11]``, ``view_deltas`` ``[T, 2]``,
    ``weapon`` ``[T]``, ``frames`` ``[3, T, 384, 672]`` float32 in [-1, 1] and
    ``metadata`` (``media_id``, ``start_frame``, ``dataset_index``).

    Args:
        manifest_path (str | Path): the manifest.
        spec (WindowSpec): the windows' sampling (21 or 41 latent frames, the camera encoding).
        dataset_root (str | Path | None): the root of relative paths.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        spec: WindowSpec,
        dataset_root: str | Path | None = None,
    ) -> None:
        self.spec = spec
        self.dataset_root = None if dataset_root is None else Path(dataset_root)
        self._verified_ticks: set[str] = set()
        self.records = [record for _, record in read_jsonl(manifest_path, "raw-video manifest")]
        for i, record in enumerate(self.records):
            self._check(record, i)
        self.cumulative = list(itertools.accumulate(int(r["n_windows"]) for r in self.records))
        weighted = [("sample_weight" in r) + ("window_sample_weights" in r) for r in self.records]
        self.sample_weights = None
        if any(weighted):
            if set(weighted) != {1}:
                raise ValueError("give every record sample_weight or window_sample_weights")
            self.sample_weights = [
                float(w)
                for r in self.records
                for w in r.get("window_sample_weights") or [r["sample_weight"]] * r["n_windows"]
            ]
            if len(self.sample_weights) != len(self):
                raise ValueError("window_sample_weights must hold one weight per window")

    def _check(self, record: dict, i: int) -> None:
        missing = sorted(set(_MANIFEST_FIELDS).difference(record))
        if missing:
            raise ValueError(f"manifest record {i} misses {missing}")
        sampling = {
            "pixel_frames": self.spec.video_frames,
            "skip_frame": SOURCE_FRAMES_PER_VIDEO_FRAME,
            "max_tick_gap_seconds": self.spec.max_tick_gap_seconds,
        }
        for key, expected in sampling.items():
            if not math.isclose(float(record.get(key, expected)), expected, abs_tol=1e-12):
                raise ValueError(
                    f"manifest record {i} was built with {key} = {record[key]}, the stage reads"
                    f" windows with {expected}"
                )
        starts, stride = record["start_frames"], int(record["stride"])
        if (
            not starts
            or int(record["n_windows"]) != len(starts)
            or any(a >= b for a, b in zip(starts, starts[1:]))
            or any(s % stride for s in starts)
            or starts[0] < 0
            or starts[-1] + source_span(self.spec.video_frames) > int(record["video_frames"])
        ):
            raise ValueError(f"manifest record {i} has invalid start_frames")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _path(self, value: str) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() or self.dataset_root is None else self.dataset_root / path

    def _ticks(self, record: Mapping) -> dict:
        """The record's tick columns, its file checked by size and sha256 on first use."""
        path = self._path(record["ticks_path"])
        if str(path) not in self._verified_ticks:
            verify_ticks_file(MediaRecord.from_row(record), path)
            self._verified_ticks.add(str(path))
        columns = ("t", "active", "delta_pitch", "delta_yaw", "input_weapon")
        return read_tick_columns(path, columns, int(record["ticks_rows"]))

    def __getitem__(self, index: int) -> dict:
        record_index = bisect.bisect_right(self.cumulative, index)
        window_index = index - (self.cumulative[record_index - 1] if record_index else 0)
        record = self.records[record_index]
        start_frame = int(record["start_frames"][window_index])
        video_path = self._path(record["video_path"])
        if not video_path.is_file() or video_path.stat().st_size != int(record["video_file_size"]):
            raise RuntimeError(f"{video_path} is missing or changed since the manifest was built")
        # the recorded buttons, without the jump recall of the tick tables: as stage 1 was trained
        ticks = self._ticks(record)
        controls = align_ticks_to_video_frames(
            timestamps=ticks["t"],
            held_buttons=ticks["active"],
            delta_pitch=ticks["delta_pitch"],
            delta_yaw=ticks["delta_yaw"],
            input_weapons=ticks["input_weapon"],
            start_frame=start_frame,
            num_frames=self.spec.video_frames,
            source_fps=float(record["fps"]),
            max_tick_gap_seconds=self.spec.max_tick_gap_seconds,
            camera_encoding=self.spec.camera_encoding,
        )
        return {
            "buttons": torch.from_numpy(controls.buttons),
            "view_deltas": torch.from_numpy(controls.view_deltas),
            "weapon": torch.from_numpy(controls.weapon),
            "metadata": {
                "media_id": record["media_id"],
                "start_frame": start_frame,
                "dataset_index": index,
            },
            "frames": load_video_window(
                video_path,
                start_frame,
                video_frames=self.spec.video_frames,
                source_frames=int(record["video_frames"]),
            ),
        }
