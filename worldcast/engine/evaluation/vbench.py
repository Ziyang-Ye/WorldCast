"""VBench over long rollouts: the per-clip scores aggregated into the paper's bins.

VBench 0.1.5 scores each ten-second clip in an environment of its own (``custom_input`` mode, local
weights, the seven :data:`DIMENSIONS`); this module reads its per-clip results, one JSON file per
shard, ``{"dims": [...], "clips": {name: {dimension: value}}}``. A dimension's value over a set of
clips is the mean over the clips that have it. Dynamic degree is a per-clip flag (moving or not)
read as 1 or 0, so its mean is the fraction of clips classified as moving.

* one-hour rollouts (Tables 4b and 13): clips ``<rollout>__s<SSSSS>.mp4`` starting at second
  ``SSSSS``, in 20-minute bins;
* whole rounds (Table 13): clips ``<media>__start<NNNNNN>.mp4``, the ten-second windows from source
  frame ``NNNNNN`` (32 fps), in the ten-second bins of the first 60 s;
* sensitivity (Table 14): clips ``orig__<clip>`` and ``<damage>__<clip>``; per damage, the percent
  change of the mean over the damaged clips against the mean over their originals.
"""

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np

from worldcast.data.latents import WINDOW_SOURCE_FRAMES

__all__ = [
    "DAMAGES",
    "DIMENSIONS",
    "HOUR_BIN_SECONDS",
    "ORIGINAL",
    "ROUND_BINS",
    "ROUND_BIN_FRAMES",
    "hour_bins",
    "read_shards",
    "round_bins",
    "sensitivity",
]

#: The VBench dimensions of the paper's tables, in the order of Table 13.
DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "temporal_flickering",
    "dynamic_degree",
    "imaging_quality",
    "aesthetic_quality",
)
#: Seconds per bin of a one-hour rollout: 20 minutes.
HOUR_BIN_SECONDS = 1200
#: Source frames (32 fps) per bin of a whole round: ten seconds, a window cut from the round
#: (``tools/evaluate_paper.py cut``).
ROUND_BIN_FRAMES = WINDOW_SOURCE_FRAMES - 1
#: Bins of a whole round: its first 60 s, which every client of the whole rounds reaches.
ROUND_BINS = 6
#: Name prefix of an undamaged clip of the sensitivity test.
ORIGINAL = "orig"
#: The damages of Table 14 by their clip-name prefix, in the table's order.
DAMAGES = {
    "blur1": "Blur, sigma=0.5 px",
    "blur3": "Blur, sigma=2 px",
    "jpeg1": "JPEG, quality 30",
    "jpeg3": "JPEG, quality 12",
    "noise3": "Noise, sigma=10",
    "exposure3": "Gamma 1.5",
    "freeze3": "Every 8th frame held",
}


def read_shards(paths: Iterable[str | Path]) -> dict[str, dict[str, float]]:
    """The per-clip scores of VBench shard files, in the order of the files and of their clips.

    A flag (dynamic degree) reads as 1.0 or 0.0; a dimension VBench did not score (``null``) is
    left out. A clip in several files gets the dimensions of all of them.
    """
    scores: dict[str, dict[str, float]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            clips = json.load(handle)["clips"]
        for name, values in clips.items():
            scored = {dim: float(value) for dim, value in values.items() if value is not None}
            scores.setdefault(name, {}).update(scored)
    return scores


def _means(clips: list[Mapping[str, float]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {"clips": len(clips)}
    for dim in DIMENSIONS:
        values = [clip[dim] for clip in clips if dim in clip]
        if values:
            out[dim] = float(np.mean(values))
    return out


def _binned(
    scores: Mapping[str, Mapping[str, float]], bin_of: Callable[[str], int]
) -> dict[int, dict[str, float | int]]:
    bins: dict[int, list[Mapping[str, float]]] = {}
    for name, values in scores.items():
        bins.setdefault(bin_of(name), []).append(values)
    return {index: _means(bins[index]) for index in sorted(bins)}


def hour_bins(scores: Mapping[str, Mapping[str, float]]) -> dict[int, dict[str, float | int]]:
    """The 20-minute bins of one-hour rollouts.

    Args:
        scores (Mapping): per-clip scores (:func:`read_shards`) of clips
            ``<rollout>__s<SSSSS>.mp4``; bin ``b`` holds those starting in minutes
            ``20 b .. 20 (b + 1)``.

    Returns:
        dict[int, dict[str, float | int]]: per bin, ``clips`` and the mean of each of the
        :data:`DIMENSIONS` the clips have, in the clips' order.
    """
    return _binned(
        scores, lambda name: int(Path(name).stem.rsplit("__s", 1)[1]) // HOUR_BIN_SECONDS
    )


def round_bins(scores: Mapping[str, Mapping[str, float]]) -> dict[int, dict[str, float | int]]:
    """The ten-second bins of the first 60 s of whole rounds.

    Args:
        scores (Mapping): per-clip scores (:func:`read_shards`) of the windows
            ``<media>__start<NNNNNN>.mp4``; bin ``k < 6`` holds those from source frame
            ``320 k``, later windows are left out.

    Returns:
        dict[int, dict[str, float | int]]: per bin, ``clips`` and the mean of each of the
        :data:`DIMENSIONS` the clips have, in the clips' order.
    """
    bins = _binned(
        scores, lambda name: int(Path(name).stem.rsplit("__start", 1)[1]) // ROUND_BIN_FRAMES
    )
    return {index: means for index, means in bins.items() if index < ROUND_BINS}


def sensitivity(scores: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float | int]]:
    """The percent change of each dimension under each damage.

    A damage's clips are those ``<damage>__<clip>`` whose original ``orig__<clip>`` is scored. Per
    dimension, over the clips both have it for, the change is ``100 (mean damaged - mean
    original) / mean original``: a ratio of means, not a mean of per-clip changes.

    Args:
        scores (Mapping): per-clip scores (:func:`read_shards`) of the originals and the damaged
            clips.

    Returns:
        dict[str, dict[str, float | int]]: per damage prefix, in sorted order, ``clips`` and
        the change of each of the :data:`DIMENSIONS` whose original mean is not 0.
    """
    prefix = f"{ORIGINAL}__"
    originals = {
        name[len(prefix) :]: values for name, values in scores.items() if name.startswith(prefix)
    }
    damaged: dict[str, dict[str, Mapping[str, float]]] = {}
    for name, values in scores.items():
        damage, _, clip = name.partition("__")
        if damage != ORIGINAL and clip in originals:
            damaged.setdefault(damage, {})[clip] = values
    out: dict[str, dict[str, float | int]] = {}
    for damage in sorted(damaged):
        clips = sorted(damaged[damage])
        row: dict[str, float | int] = {"clips": len(clips)}
        for dim in DIMENSIONS:
            both = [c for c in clips if dim in originals[c] and dim in damaged[damage][c]]
            original = np.array([originals[c][dim] for c in both])
            changed = np.array([damaged[damage][c][dim] for c in both])
            if both and original.mean() != 0:
                row[dim] = float(100 * (changed.mean() - original.mean()) / original.mean())
        out[damage] = row
    return out
