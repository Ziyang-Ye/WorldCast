"""The bucket index of stages 2-4, the evaluation reserve and the validation index.

``split`` draws the held-out matches from the media index. ``windows`` writes the bucket index, the
windows of the other matches with their visible pairs from the GT visibility labels, one file per
map. ``reserve`` picks the evaluation reserve out of the bucket index, ``validation`` the validation
windows out of the windows of the held-out matches, and ``weights`` solves the sampling weight of
each bucket file. Every choice is a seeded or hashed order, so a rerun gives the same files.

Examples::

    python tools/prepare_data.py buckets split --media-index data/media_index.jsonl --seed 1 \\
        --out data/held_out_matches.txt
    python tools/prepare_data.py buckets windows --media-index data/media_index.jsonl \\
        --held-out data/held_out_matches.txt --visibility-label-root data/vislabels \\
        --out-dir data/buckets
    python tools/prepare_data.py buckets reserve --bucket-dir data/buckets --maps de_dust2 \\
        --target-capacity 2806 --out data/reserve.txt
    python tools/prepare_data.py buckets validation --media-index data/media_index.jsonl \\
        --held-out data/held_out_matches.txt --visibility-label-root data/vislabels \\
        --out data/validation_index.jsonl
    python tools/prepare_data.py buckets weights --bucket-dir data/buckets

``weights`` and ``validation`` print, beside the JSON summary, the lines of a config that use the
files: ``data.bucket_weights`` for the bucket files and ``validation.index_sha256`` for the
validation index.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from worldcast.data.controls import (
    FRAME_TIME_SLACK,
    TICK_GATE_SECONDS,
    AlignmentError,
    count_until,
    frame_tick_ranges,
)
from worldcast.data.labels import visibility_rows
from worldcast.data.latents import (
    SOURCE_FRAMES_PER_VIDEO_FRAME,
    WINDOW_SOURCE_FRAMES,
    WINDOW_VIDEO_FRAMES,
    window_key,
)
from worldcast.data.recordings import MediaIndex, MediaRecord, read_jsonl, read_tick_columns
from worldcast.utils.files import atomic_write, sha256_file, write_jsonl

from .media_index import MAPS

#: The share of all windows held out, as the paper's split: a tenth.
HELD_OUT_FRACTION = 0.1

#: Source frames between neighbouring grid windows: a training window's video frame count, so
#: neighbours share half their span.
GRID_STRIDE = WINDOW_VIDEO_FRAMES
#: A midpoint window starts this many source frames after a grid window.
MIDPOINT_OFFSET = GRID_STRIDE // 2

#: A grid window counts toward a reserve's capacity from this many visible pairs.
RESERVE_VIS_MIN = 100
#: The reserve takes at most this many rounds of a match.
RESERVE_MAX_ROUNDS_PER_MATCH = 4
#: Seed of the paper's reserve.
RESERVE_SEED = "opencs2-dust2-final-evalreserve-v3-20260819"

#: The validation windows of a map are drawn from those with at least this quantile of its visible
#: pairs.
VALIDATION_QUANTILE = 0.75
#: Windows per map of the paper's validation index, and its seed.
VALIDATION_PER_MAP = 16
VALIDATION_SEED = "maps4-final-eval-v1-20260828"

#: Iterations of :func:`solve_slot_weights`; the fixed point is reached after about six.
SOLVER_ITERATIONS = 100


def _rank(*parts: object) -> str:
    """sha256 of the parts joined by NUL: the seeded order of a draw."""
    return hashlib.sha256("\0".join(str(part) for part in parts).encode()).hexdigest()


# ================================================================================== the match split
def grid_window_counts(records: Iterable[MediaRecord]) -> dict[int, dict[str, int]]:
    """``{match_id: {map_name: windows}}``: the windows of every recording that fit its video
    (:func:`window_starts` without ticks)."""
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        counts[record.match_id][record.map_name] += len(window_starts(record.source_frames))
    return {match: dict(per_map) for match, per_map in counts.items()}


def draw_held_out(
    windows: Mapping[int, Mapping[str, int]],
    *,
    fraction: float,
    seed: str,
    exclude: Collection[int] = (),
    keep: Collection[int] = (),
) -> list[int]:
    """The held-out matches: whole matches, each map holding ``fraction / maps`` of all windows.

    The matches of ``keep`` are held out. The others, but those of ``exclude``, are taken in the
    order of ``sha256(seed NUL match_id)``, each if some map of it is still below its share, until
    every map holds its share.

    Args:
        windows (Mapping[int, Mapping[str, int]]): :func:`grid_window_counts`.
        fraction (float): the share of all windows held out, in (0, 1).
        seed (str): orders the draw.
        exclude (Collection[int]): matches that are never drawn (those an earlier stage trained on).
        keep (Collection[int]): matches that are held out.

    Returns:
        list[int]: the held-out matches, ascending.

    Raises:
        ValueError: ``fraction`` is outside (0, 1), a match of ``keep`` is not in ``windows``, or
            the matches that may be drawn cannot give every map its share.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"the held-out fraction must be in (0, 1), got {fraction}")
    unknown = sorted(set(keep) - set(windows))
    if unknown:
        raise ValueError(f"{len(unknown)} kept matches are not in the media index: {unknown[:3]}")
    maps = sorted({name for per_map in windows.values() for name in per_map})
    share = fraction * sum(sum(per_map.values()) for per_map in windows.values()) / len(maps)
    held, excluded = set(keep), set(exclude)
    got = {name: sum(windows[match].get(name, 0) for match in held) for name in maps}
    pool = sorted(
        (match for match in windows if match not in held and match not in excluded),
        key=lambda match: (_rank(seed, match), match),
    )
    for match in pool:
        if all(got[name] >= share for name in maps):
            break
        if any(got[name] < share for name in windows[match]):
            held.add(match)
            for name, count in windows[match].items():
                got[name] += count
    short = [name for name in maps if got[name] < share]
    if short:
        raise ValueError(
            f"the matches that may be held out cannot give {', '.join(short)} its share of"
            f" {share:.0f} windows"
        )
    return sorted(held)


def count_split(
    records: Iterable[MediaRecord], held_out: Collection[int]
) -> dict[str, dict[str, int]]:
    """Matches, match-maps (a match's games on one map), rounds and recordings of the two sides.

    Raises:
        ValueError: a held-out match is not in the media index.
    """
    records = list(records)
    unknown = sorted(set(held_out) - {record.match_id for record in records})
    if unknown:
        raise ValueError(
            f"{len(unknown)} held-out matches are not in the media index: {unknown[:3]}"
        )
    sides: dict[str, list[MediaRecord]] = {"train": [], "held_out": []}
    for record in records:
        sides["held_out" if record.match_id in held_out else "train"].append(record)
    return {
        side: {
            "matches": len({record.match_id for record in rows}),
            "match_maps": len({(record.match_id, record.map_name) for record in rows}),
            "rounds": len({record.round_key for record in rows}),
            "media": len(rows),
        }
        for side, rows in sides.items()
    }


# =============================================================================== windows of a media
def window_starts(source_frames: int, gate: Callable[[int], bool] | None = None) -> list[int]:
    """The start frames of a recording's windows, ascending.

    The grid windows start at multiples of :data:`GRID_STRIDE`, the midpoints
    :data:`MIDPOINT_OFFSET` after them; a window must fit in the ``source_frames`` of the video.
    Without ``gate`` every window that fits is kept. With it, the grid windows it passes are kept
    and a midpoint is added between each two neighbouring ones: its ticks lie within theirs, so
    their gate covers it.

    Args:
        source_frames (int): frames of the recorded video (32 fps).
        gate (Callable[[int], bool] | None): whether a grid window, by its start frame, is kept
            (:func:`tick_gate`).
    """
    grid = range(0, source_frames - WINDOW_SOURCE_FRAMES + 1, GRID_STRIDE)
    if gate is None:
        midpoints = range(MIDPOINT_OFFSET, source_frames - WINDOW_SOURCE_FRAMES + 1, GRID_STRIDE)
        return sorted([*grid, *midpoints])
    kept = [start for start in grid if gate(start)]
    midpoints = [
        left + MIDPOINT_OFFSET for left, right in zip(kept, kept[1:]) if right - left == GRID_STRIDE
    ]
    return sorted([*kept, *midpoints])


def tick_gate(
    timestamps: np.ndarray,
    is_alive: Sequence[object],
    source_fps: float,
    max_tick_gap_seconds: float = TICK_GATE_SECONDS,
) -> Callable[[int], bool]:
    """A player's ticks as the check of a window: whether the window can be served from them.

    A window passes if its controls align to the ticks within ``max_tick_gap_seconds``
    (:func:`worldcast.data.controls.frame_tick_ranges`) and every tick of its span is of a living
    player.

    Args:
        timestamps (np.ndarray): ``[n]`` float64 tick times, seconds, strictly increasing.
        is_alive (Sequence[object]): ``[n]`` whether the player lives at each tick; ``None`` is
            dead.
        source_fps (float): source frames per second of the video.
        max_tick_gap_seconds (float): largest tolerated tick gap and frame-endpoint lag.

    Returns:
        Callable[[int], bool]: the check, by a window's start frame.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    dead = [value is None or not bool(value) for value in is_alive]
    dead_before = np.concatenate(([0], np.cumsum(dead)))

    def passes(start_frame: int) -> bool:
        try:
            frame_tick_ranges(
                timestamps=timestamps,
                start_frame=start_frame,
                num_frames=WINDOW_VIDEO_FRAMES,
                source_fps=source_fps,
                stride=SOURCE_FRAMES_PER_VIDEO_FRAME,
                max_tick_gap_seconds=max_tick_gap_seconds,
            )
        except AlignmentError:
            return False
        first = int(
            np.searchsorted(timestamps, start_frame / source_fps - FRAME_TIME_SLACK, side="left")
        )
        last = int(count_until(timestamps, (start_frame + WINDOW_SOURCE_FRAMES - 1) / source_fps))
        return last > first and dead_before[last] == dead_before[first]

    return passes


def window_pair_counts(visible: np.ndarray, own_slot: int, starts: Iterable[int]) -> dict[int, int]:
    """The visible pairs of each window: the other players' visible labels over its video frames.

    Args:
        visible (np.ndarray): ``[P, F]`` the recording's visibility labels at source frames
            ``0 .. F - 1`` (:func:`worldcast.data.labels.visibility_rows`), non-zero where a player
            is visible.
        own_slot (int): the recording player's slot, whose row is left out.
        starts (Iterable[int]): start frames; each window must lie within ``F``.

    Returns:
        dict[int, int]: start frame -> the number of (player, video frame) pairs seen.
    """
    others = np.arange(visible.shape[0]) != own_slot
    return {
        int(start): int(np.count_nonzero(visible[others, start : start + WINDOW_SOURCE_FRAMES : 2]))
        for start in starts
    }


def window_row(record: MediaRecord, start_frame: int, pair_count: int) -> dict[str, Any]:
    """The bucket row of a window: the fields docs/training.md lists, and ``map_name``."""
    return {
        "latent_key": window_key(start_frame),
        "map_name": record.map_name,
        "match_id": record.match_id,
        "media_id": record.media_id,
        "round": record.round,
        "start_frame": int(start_frame),
        "vis_available": True,
        "vis_pair_count": int(pair_count),
    }


# ====================================================================================== the reserve
def _span_disjoint(starts: Iterable[int]) -> int:
    """Windows that can be picked from ``starts`` without two sharing a frame: greedily from the
    first, each at least ``2 * GRID_STRIDE`` after the last."""
    kept, last = 0, None
    for start in sorted(starts):
        if last is None or start - last >= 2 * GRID_STRIDE:
            kept, last = kept + 1, start
    return kept


def select_reserve(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_capacity: int,
    seed: str = RESERVE_SEED,
) -> list[str]:
    """The evaluation reserve: whole rounds, every recording of each, taken breadth first.

    A recording's capacity is the number of its grid windows of :data:`RESERVE_VIS_MIN` visible
    pairs that no two share a frame (``2 * GRID_STRIDE`` apart); a round's, the sum over its
    recordings, and a round of none is never taken. The matches, in the order of
    ``sha256(seed NUL "match" NUL match_id)``, give a round each (the first of its rounds in the
    order of ``sha256(seed NUL "round" NUL match_id NUL round)``), then a second, up to
    :data:`RESERVE_MAX_ROUNDS_PER_MATCH`, until the capacity reaches ``target_capacity``.

    Args:
        rows (Iterable[Mapping]): the windows of the recordings that may be reserved
            (``media_id``, ``match_id``, ``map_name``, ``round``, ``start_frame``,
            ``vis_pair_count``).
        target_capacity (int): the capacity to reach.
        seed (str): orders the draw.

    Returns:
        list[str]: the media ids of the reserved rounds, ascending.

    Raises:
        ValueError: the rounds cannot reach ``target_capacity``.
    """
    rounds_of: dict[str, tuple[int, str, int]] = {}
    dense: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        media_id, start = str(row["media_id"]), int(row["start_frame"])
        rounds_of[media_id] = (int(row["match_id"]), str(row["map_name"]), int(row["round"]))
        if start % GRID_STRIDE == 0 and int(row["vis_pair_count"]) >= RESERVE_VIS_MIN:
            dense[media_id].append(start)
    media_of: dict[tuple[int, str, int], list[str]] = defaultdict(list)
    for media_id, key in rounds_of.items():
        media_of[key].append(media_id)
    capacity = {
        key: sum(_span_disjoint(dense[media_id]) for media_id in media_ids)
        for key, media_ids in media_of.items()
    }
    by_match: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    for key, value in capacity.items():
        if value:
            by_match[key[0]].append(key)
    for rounds in by_match.values():
        rounds.sort(key=lambda key: (_rank(seed, "round", key[0], key[2]), key))
    matches = sorted(by_match, key=lambda match: (_rank(seed, "match", match), match))

    taken: list[tuple[int, str, int]] = []
    reached = 0
    for depth in range(RESERVE_MAX_ROUNDS_PER_MATCH):
        for match in matches:
            if reached >= target_capacity:
                break
            if depth < len(by_match[match]):
                taken.append(by_match[match][depth])
                reached += capacity[by_match[match][depth]]
    if reached < target_capacity:
        raise ValueError(
            f"{len(taken)} rounds, at most {RESERVE_MAX_ROUNDS_PER_MATCH} of a match, reach a"
            f" capacity of {reached}, not {target_capacity}"
        )
    return sorted(media_id for key in taken for media_id in media_of[key])


# =============================================================================== validation windows
def _quantile(values: Sequence[int], q: float) -> float:
    """The ``q`` quantile of ``values`` by linear interpolation between the order statistics."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def select_validation(
    rows: Iterable[Mapping[str, Any]],
    *,
    maps: Sequence[str],
    per_map: int = VALIDATION_PER_MAP,
    seed: str = VALIDATION_SEED,
) -> list[Mapping[str, Any]]:
    """The validation windows: per map, one window of each of the first ``per_map`` recordings.

    A map's candidates are its windows with at least the :data:`VALIDATION_QUANTILE` quantile of
    its visible pairs. They are taken in the order of ``sha256(seed NUL map_name NUL media_id NUL
    start_frame)``, a recording only once.

    Args:
        rows (Iterable[Mapping]): the windows of the held-out matches (the bucket row's fields).
        maps (Sequence[str]): the maps to draw from.
        per_map (int): windows per map.
        seed (str): orders the draw.

    Returns:
        list[Mapping]: the chosen rows by ``(map_name, media_id, start_frame)``.

    Raises:
        ValueError: a map has fewer than ``per_map`` recordings among its candidates.
    """
    by_map: dict[str, list[Mapping[str, Any]]] = {name: [] for name in maps}
    for row in rows:
        if row["map_name"] in by_map and row["vis_available"]:
            by_map[row["map_name"]].append(row)
    chosen = []
    for name, labelled in by_map.items():
        if not labelled:
            raise ValueError(f"no labelled windows of {name} among the held-out matches")
        bar = _quantile([int(row["vis_pair_count"]) for row in labelled], VALIDATION_QUANTILE)
        candidates = sorted(
            (row for row in labelled if int(row["vis_pair_count"]) >= bar),
            key=lambda row: (
                _rank(seed, name, row["media_id"], row["start_frame"]),
                row["media_id"],
                int(row["start_frame"]),
            ),
        )
        picked: dict[str, Mapping[str, Any]] = {}
        for row in candidates:
            picked.setdefault(str(row["media_id"]), row)
            if len(picked) == per_map:
                break
        if len(picked) < per_map:
            raise ValueError(
                f"{name}: {len(picked)} recordings among the {len(candidates)} windows of at least"
                f" {bar:g} visible pairs, {per_map} needed"
            )
        chosen.extend(picked.values())
    return sorted(
        chosen, key=lambda row: (row["map_name"], row["media_id"], int(row["start_frame"]))
    )


# ===================================================================================== slot weights
@dataclass
class SlotStats:
    """What a bucket file adds to the sampler's mass.

    Attributes:
        windows (int): its rows.
        zero_pair (int): rows without visible pairs.
        density (float): ``sum((1 + pairs) ** DENSITY_EXPONENT)`` over the rows with pairs.
    """

    windows: int = 0
    zero_pair: int = 0
    density: float = 0.0

    def add(self, pair_count: int) -> None:
        """Count a row of ``pair_count`` visible pairs."""
        from worldcast.data.training import DENSITY_EXPONENT

        self.windows += 1
        if pair_count:
            self.density += (1.0 + pair_count) ** DENSITY_EXPONENT
        else:
            self.zero_pair += 1


def _floor_boost(slots: Sequence[SlotStats], weights: Sequence[float]) -> float:
    """The factor :func:`worldcast.data.training.apply_zero_pair_floor` gives the zero-pair rows."""
    from worldcast.data.training import ZERO_PAIR_SHARE

    zero = sum(w * s.zero_pair for w, s in zip(weights, slots))
    dense = sum(w * s.density for w, s in zip(weights, slots))
    if zero == 0 or zero / (zero + dense) >= ZERO_PAIR_SHARE:
        return 1.0
    return ZERO_PAIR_SHARE / (1.0 - ZERO_PAIR_SHARE) * dense / zero


def draw_shares(slots: Sequence[SlotStats], weights: Sequence[float]) -> list[float]:
    """The share of the sampler's draws that each slot gets under ``weights``, floor included."""
    boost = _floor_boost(slots, weights)
    mass = [w * (s.density + boost * s.zero_pair) for w, s in zip(weights, slots)]
    return [m / sum(mass) for m in mass]


def solve_slot_weights(slots: Sequence[SlotStats]) -> list[float]:
    """The weights of the bucket files with which the sampler draws every map equally often.

    A slot is a map's file (:mod:`worldcast.data.training` weights a window by its file's weight,
    ``(1 + pairs) ** 0.5`` and the zero-pair floor). The largest weight is 1; an empty slot's is 1
    and selects nothing.

    Args:
        slots (Sequence[SlotStats]): the statistics of each file.

    Returns:
        list[float]: the weight of each file.
    """
    if not any(s.windows for s in slots):
        raise ValueError("no windows to weight")
    weights = [1.0 if s.windows else 0.0 for s in slots]
    for _ in range(SOLVER_ITERATIONS):
        boost = _floor_boost(slots, weights)
        raw = [1.0 / (s.density + boost * s.zero_pair) if s.windows else 0.0 for s in slots]
        weights = [r / max(raw) for r in raw]
    return [w if s.windows else 1.0 for w, s in zip(weights, slots)]


# ========================================================================================= files
def _read_lines(path: str, what: str) -> list[str]:
    """The non-blank lines of a text file."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{what} does not exist: {path}")
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _read_matches(path: str, what: str) -> list[int]:
    """The match ids of a text file, one per line, each once."""
    try:
        matches = [int(line) for line in _read_lines(path, what)]
    except ValueError:
        raise ValueError(f"{what} {path} must hold one match id per line") from None
    if len(set(matches)) != len(matches):
        raise ValueError(f"{what} {path} lists a match twice")
    return matches


def _load_records(path: str) -> list[MediaRecord]:
    return list(MediaIndex.load(path).by_id.values())


def _excluded_media(path: str | None, records: Collection[MediaRecord]) -> frozenset[str]:
    """The media ids of ``path``, each of which must be in the media index."""
    if path is None:
        return frozenset()
    excluded = frozenset(_read_lines(path, "excluded media"))
    unknown = sorted(excluded - {record.media_id for record in records})
    if unknown:
        raise ValueError(f"{len(unknown)} excluded media are not in the media index: {unknown[:3]}")
    return excluded


def _bucket_names() -> tuple[str, ...]:
    from worldcast.data.training import BUCKET_NAMES

    return BUCKET_NAMES


def _bucket_rows(bucket_dir: str) -> Iterator[tuple[int, Path, int, dict]]:
    """``(slot, file, line number, row)`` of the five bucket files, in the loader's order."""
    for slot, name in enumerate(_bucket_names()):
        path = Path(bucket_dir) / f"train_{name}.jsonl"
        for line_number, row in read_jsonl(path, "bucket file"):
            yield slot, path, line_number, row


def _write_lines(path: Path, lines: Iterable[object]) -> None:
    atomic_write(path, lambda handle: handle.write("".join(f"{line}\n" for line in lines).encode()))


def _report(summary: Mapping[str, Any]) -> None:
    print(json.dumps(summary, indent=2))


def _paste(section: str, key: str, value: str) -> None:
    """The YAML lines that set ``<section>.<key>`` for training and evaluation, on stderr beside
    the JSON summary of stdout."""
    print(f"{section}:\n  {key}: {value}", file=sys.stderr, flush=True)


# ===================================================================================== windows
def _gate(record: MediaRecord, dataset_root: str) -> Callable[[int], bool]:
    """The recording's ticks as the check of its windows."""
    ticks = Path(dataset_root) / record.ticks_path
    table = read_tick_columns(ticks, ("t", "is_alive"), record.ticks_rows)
    return tick_gate(np.asarray(table["t"], dtype=np.float64), table["is_alive"], record.fps)


def _pair_counts(label_root: str, record: MediaRecord, starts: list[int]) -> dict[int, int] | None:
    """The visible pairs of the windows ``starts`` of a recording, or None without its labels."""
    frames = np.arange(starts[-1] + WINDOW_SOURCE_FRAMES)
    try:
        visible, _ = visibility_rows(label_root, record.media_id, frames)
    except FileNotFoundError:
        return None
    return window_pair_counts(visible, record.player_slot, starts)


def _map_rows(
    records: list[MediaRecord],
    args: argparse.Namespace,
    gated: bool,
    excluded: Collection[str],
    left_out: Counter,
) -> Iterator[dict[str, Any]]:
    """The rows of the windows of one map's recordings, by media id and start frame.

    A recording that is excluded, has no window or has no labels adds to ``left_out`` instead.
    """
    for record in sorted(records, key=lambda record: record.media_id):
        if record.media_id in excluded:
            left_out["excluded"] += 1
            continue
        starts = window_starts(
            record.source_frames, _gate(record, args.dataset_root) if gated else None
        )
        if not starts:
            left_out["without a window"] += 1
            continue
        pairs = _pair_counts(args.visibility_label_root, record, starts)
        if pairs is None:
            left_out["without labels"] += 1
            continue
        for start in starts:
            yield window_row(record, start, pairs[start])


def _gated_maps(args: argparse.Namespace, records: Collection[MediaRecord]) -> set[str]:
    """The maps whose windows are checked against their ticks."""
    unknown = sorted(set(args.maps) - {record.map_name for record in records})
    if unknown:
        raise ValueError(f"the media index has no recordings of {', '.join(unknown)}")
    if args.tick_gate_maps and args.dataset_root is None:
        raise ValueError("--tick-gate-maps needs --dataset-root")
    if args.dataset_root is None:
        return set()
    gated = set(args.tick_gate_maps or args.maps)
    if not gated <= set(args.maps):
        raise ValueError("--tick-gate-maps must be among --maps")
    return gated


def _counted(rows: Iterable[dict[str, Any]], stats: SlotStats) -> Iterator[dict[str, Any]]:
    """The rows, each counted into ``stats`` as it passes."""
    for row in rows:
        stats.add(row["vis_pair_count"])
        yield row


def _slot_weights(stats: list[SlotStats]) -> dict[str, Any]:
    """The weights that give every map equal draws, and the draws the loader's weights give."""
    from worldcast.data.training import BUCKET_WEIGHTS

    weights = [round(weight, 9) for weight in solve_slot_weights(stats)]
    return {
        "slot_weights": weights,
        "matches_loader_weights": weights == list(BUCKET_WEIGHTS),
        "draw_share_with_loader_weights": [
            round(share, 6) for share in draw_shares(stats, BUCKET_WEIGHTS)
        ],
    }


# ========================================================================================= steps
def _run_split(args: argparse.Namespace) -> int:
    records = _load_records(args.media_index)
    windows = grid_window_counts(records)
    exclude = (
        _read_matches(args.exclude_matches, "excluded matches") if args.exclude_matches else ()
    )
    keep = _read_matches(args.keep_matches, "kept matches") if args.keep_matches else ()
    held_out = draw_held_out(
        windows, fraction=args.fraction, seed=args.seed, exclude=exclude, keep=keep
    )
    path = Path(args.out)
    _write_lines(path, held_out)
    held_out_windows: Counter = Counter()
    for match in held_out:
        held_out_windows.update(windows[match])
    _report(
        {
            "split": count_split(records, held_out),
            "held_out_windows": dict(sorted(held_out_windows.items())),
            "sha256": sha256_file(path),
        }
    )
    return 0


def _run_windows(args: argparse.Namespace) -> int:
    records = _load_records(args.media_index)
    held_out = set(_read_matches(args.held_out, "held-out matches"))
    split = count_split(records, held_out)
    gated, excluded = _gated_maps(args, records), _excluded_media(args.exclude_media, records)
    names = _bucket_names()
    if len(args.maps) > len(names):
        raise ValueError(f"at most {len(names)} maps fit the bucket files")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    left_out: Counter = Counter()
    stats = [SlotStats() for _ in names]
    files = {}
    for slot, name in enumerate(names):
        path = out_dir / f"train_{name}.jsonl"
        map_name = args.maps[slot] if slot < len(args.maps) else None
        rows: Iterable[dict[str, Any]] = ()
        if map_name is not None:
            mine = [r for r in records if r.map_name == map_name and r.match_id not in held_out]
            rows = _map_rows(mine, args, map_name in gated, excluded, left_out)
        write_jsonl(path, _counted(rows, stats[slot]))
        files[path.name] = {
            "map": map_name,
            "rows": stats[slot].windows,
            "sha256": sha256_file(path),
        }
        print(f"{path}: {stats[slot].windows} windows", file=sys.stderr, flush=True)
    if not any(s.windows for s in stats):
        raise ValueError(f"no windows to write; left out: {dict(left_out)}")
    _report(
        {"split": split, "left_out_media": dict(left_out), "files": files, **_slot_weights(stats)}
    )
    return 0


def _run_reserve(args: argparse.Namespace) -> int:
    rows = (row for _, _, _, row in _bucket_rows(args.bucket_dir))
    if args.maps:
        rows = (row for row in rows if row["map_name"] in args.maps)
    media = select_reserve(rows, target_capacity=args.target_capacity, seed=args.seed)
    path = Path(args.out)
    _write_lines(path, media)
    _report({"media": len(media), "sha256": sha256_file(path)})
    return 0


def _run_validation(args: argparse.Namespace) -> int:
    records = _load_records(args.media_index)
    held_out = set(_read_matches(args.held_out, "held-out matches"))
    split = count_split(records, held_out)
    gated, excluded = _gated_maps(args, records), _excluded_media(args.exclude_media, records)
    left_out: Counter = Counter()
    rows = []
    for map_name in args.maps:
        mine = [r for r in records if r.map_name == map_name and r.match_id in held_out]
        windows = list(_map_rows(mine, args, map_name in gated, excluded, left_out))
        print(f"{map_name}: {len(windows)} windows", file=sys.stderr, flush=True)
        rows.extend(windows)
    chosen = select_validation(rows, maps=args.maps, per_map=args.per_map, seed=args.seed)
    path = Path(args.out)
    write_jsonl(path, chosen)
    digest = sha256_file(path)
    _report(
        {
            "split": split,
            "windows": len(rows),
            "left_out_media": dict(left_out),
            "rows": len(chosen),
            "sha256": digest,
        }
    )
    _paste("validation", "index_sha256", digest)
    return 0


def _run_weights(args: argparse.Namespace) -> int:
    stats = [SlotStats() for _ in _bucket_names()]
    for slot, path, line_number, row in _bucket_rows(args.bucket_dir):
        if not row.get("vis_available"):
            raise ValueError(f"{path}:{line_number} has no visibility labels (vis_available)")
        stats[slot].add(int(row["vis_pair_count"]))
    summary = {"rows": [s.windows for s in stats], **_slot_weights(stats)}
    _report(summary)
    _paste("data", "bucket_weights", json.dumps(summary["slot_weights"]))
    return 0


# ============================================================================================ CLI
def _add_window_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of the steps that enumerate windows from the media index and the labels."""
    parser.add_argument("--media-index", required=True, help="the media index (JSONL)")
    parser.add_argument(
        "--held-out", required=True, help="the held-out matches, one match id per line"
    )
    parser.add_argument(
        "--visibility-label-root",
        required=True,
        help="the GT visibility labels, <media_id>.npz (data.visibility_label_root)",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=list(MAPS),
        help="the maps, in the order of the bucket files (default: the paper's four)",
    )
    parser.add_argument(
        "--exclude-media",
        help="media ids to leave out (recordings that do not decode), one per line",
    )
    parser.add_argument(
        "--dataset-root",
        help="the OpenCS2 recordings: with it a window must pass the tick tables (no tick gap"
        " above 1.5 / 64 s, the player alive), and midpoints lie between kept neighbours",
    )
    parser.add_argument(
        "--tick-gate-maps",
        nargs="+",
        help="the maps checked against their ticks (default: every map; needs --dataset-root)",
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this step's options to its subcommand's parser."""
    commands = parser.add_subparsers(dest="bucket_command", required=True)

    def command(name: str, run: Callable[[argparse.Namespace], int], summary: str):
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.set_defaults(bucket_run=run, parser=sub)
        return sub

    sub = command(
        "split", _run_split, "Draw the held-out matches: whole matches, each map holding its share."
    )
    sub.add_argument("--media-index", required=True, help="the media index (JSONL)")
    sub.add_argument("--out", required=True, help="the held-out matches, one match id per line")
    sub.add_argument("--seed", required=True, help="orders the draw")
    sub.add_argument(
        "--fraction", type=float, default=HELD_OUT_FRACTION, help="the share of all windows"
    )
    sub.add_argument("--exclude-matches", help="matches never drawn, one match id per line")
    sub.add_argument("--keep-matches", help="matches that stay held out, one match id per line")

    sub = command(
        "windows", _run_windows, "Write the bucket index of the matches that are not held out."
    )
    _add_window_arguments(sub)
    sub.add_argument(
        "--out-dir", required=True, help="receives train_q00-01.jsonl ... train_q07-10.jsonl"
    )

    sub = command("reserve", _run_reserve, "Pick the evaluation reserve out of the bucket index.")
    sub.add_argument("--bucket-dir", required=True, help="the bucket index (data.bucket_dir)")
    sub.add_argument("--out", required=True, help="the reserved media ids, one per line")
    sub.add_argument(
        "--target-capacity",
        type=int,
        required=True,
        help="windows of the reserve with 100 visible pairs that no two share a frame",
    )
    sub.add_argument("--maps", nargs="+", help="reserve from these maps only (default: all)")
    sub.add_argument("--seed", default=RESERVE_SEED, help="orders the draw (default: the paper's)")

    sub = command(
        "validation", _run_validation, "Write the validation index out of the held-out windows."
    )
    _add_window_arguments(sub)
    sub.add_argument("--out", required=True, help="the validation index (JSONL)")
    sub.add_argument("--per-map", type=int, default=VALIDATION_PER_MAP, help="windows per map")
    sub.add_argument(
        "--seed", default=VALIDATION_SEED, help="orders the draw (default: the paper's)"
    )

    sub = command("weights", _run_weights, "Solve the sampling weight of each bucket file.")
    sub.add_argument("--bucket-dir", required=True, help="the bucket index (data.bucket_dir)")


def run(args: argparse.Namespace) -> int:
    """Run this step's subcommand; returns the exit status."""
    return args.bucket_run(args)
