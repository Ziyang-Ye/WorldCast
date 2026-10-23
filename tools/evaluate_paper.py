#!/usr/bin/env python3
"""Compute the paper's evaluation metrics (docs/training.md, "The paper's metrics").

Steps, each with ``--help`` after its name:

    cut          round-long videos -> ten-second windows (CPU)
    detect       the player detector on the recorded and generated windows -> a detection dump (GPU)
    entity       Rendered and Unmatched of every view from detection dumps (CPU)
    entity-ci    their intervals and paired differences (CPU)
    depth        depth agreement rows of one arm (GPU)
    depth-cells  the cells of Tables 19, 10 and 4a from the rows of arms (CPU)
    match        Mat. Pix., Inliers and RotErr of every pair of one arm (GPU)
    match-cells  the cells of Table 4a from the pairs of arms (CPU)
    fvd          FVD of ten-second windows or of whole rounds (GPU)
    vbench       VBench's per-clip scores in the paper's bins (CPU)

An arm is one of the compared videos' sources (a model, or the recorded video), by the name a step
gives it. An evaluation directory holds ``generation/<stem>.mp4`` and ``gt/<stem>.mp4`` (the
recorded video, the VAE reconstruction of the recording), one pair per view, ``<stem>`` being
``<media_id>__start<NNNNNN>`` (its first source frame, 32 fps). A session directory is what
``tools/run_session.py`` writes, ``<round>/<media_id>/latents.npy`` with ``video.mp4`` beside it
from ``tools/decode.py``. The weights come from ``paths.detector``, ``paths.gim_dkm``,
``paths.loftr`` and ``paths.i3d`` (``tools/download_weights.py --with-metrics``), the data from the
``paths`` of a data config: ``data/evaluation/eval/windows.yaml`` or ``rounds.yaml`` of
``tools/download_eval_data.py``; ``fvd --rounds``, ``match --recorded`` and ``depth --rounds
--recorded`` read the recordings' video or latent cache and take one of your own that points at
OpenCS2.

Examples::

    python tools/evaluate_paper.py cut --videos runs/table3/*/*/video.mp4 \\
        --out-dir runs/table3_windows/generation
    python tools/evaluate_paper.py detect --config weights/paths.yaml \\
        --config data/evaluation/eval/windows.yaml --eval-dir runs/windows/worldcast \\
        --arm worldcast=runs/windows/worldcast --shard 0/8 --out runs/entity/dets-0.json.gz
    python tools/evaluate_paper.py entity --config data/evaluation/eval/windows.yaml \\
        --dets runs/entity/dets-*.json.gz --out runs/entity/table1
    python tools/evaluate_paper.py entity-ci --scores runs/entity/table1.json \\
        --difference worldcast,frame_concat --out runs/entity/table1_ci.json
    python tools/evaluate_paper.py vbench --whole-rounds runs/vbench/shard-*.json \\
        --out runs/vbench/rounds.json
"""

import argparse
import json
import re
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from worldcast.config.inference import PathsConfig, load_config
from worldcast.config.loader import add_config_args, parse_overrides, usage_errors
from worldcast.data.latents import (
    SOURCE_FPS,
    SOURCE_FRAMES_PER_LATENT,
    SOURCE_FRAMES_PER_VIDEO_FRAME,
    VIDEO_FRAMES_PER_LATENT,
    WINDOW_SOURCE_FRAMES,
    WINDOW_VIDEO_FRAMES,
)
from worldcast.data.recordings import (
    MediaIndex,
    MediaRecord,
    read_jsonl,
    read_player_ticks,
    read_round_index,
)
from worldcast.engine.evaluation.boxes import table_boxes, tick_states
from worldcast.engine.evaluation.depth_agreement import (
    BETWEEN_CLIENTS,
    CHANCE,
    OWN_NEXT_FRAME,
    SCENE_SPLIT_SECONDS,
    VIEW_SPLIT_SECONDS,
    covisibility_mask,
    recorded_ticks,
    segment_ratios,
)
from worldcast.engine.evaluation.detector import (
    DETECT_CONF,
    DETECT_IMGSZ,
    at_confidence,
    read_detections,
)
from worldcast.engine.evaluation.entity import (
    MATCH_RADIUS,
    MIN_SIDE,
    gt_visibility_gate,
    pool_by_crowding,
    score_view,
    summarize_entity,
    visibility_gate,
    window_of,
)
from worldcast.engine.evaluation.matching import GIM_FIELDS, LOFTR_FIELDS
from worldcast.engine.evaluation.stats import (
    cluster_bootstrap,
    mean_over_units,
    paired_difference,
    unit_interval,
)
from worldcast.engine.evaluation.vbench import DAMAGES, hour_bins, read_shards, round_bins
from worldcast.engine.evaluation.vbench import sensitivity as vbench_sensitivity
from worldcast.engine.inference.session import round_name, rounds_of
from worldcast.hub import METRIC_WEIGHTS

#: Source frames between the starts of consecutive windows of a round: ten seconds, so neighbours
#: share their boundary frame.
WINDOW_STRIDE = WINDOW_SOURCE_FRAMES - 1
#: The repository.
ROOT = Path(__file__).resolve().parents[1]
#: The columns of a tick table that the player boxes read: the time and the player state.
BOX_COLUMNS = ("t", "x", "y", "z", "yaw", "pitch", "is_alive")
#: The kinds of depth agreement rows, by their name in the cells.
DEPTH_KINDS = {
    "between_clients": BETWEEN_CLIENTS,
    "own_next_frame": OWN_NEXT_FRAME,
    "chance": CHANCE,
}
#: The per-pair results of :mod:`worldcast.engine.evaluation.matching`, by the matcher whose keys
#: they belong to.
MATCH_FIELDS = {"gim": GIM_FIELDS, "loftr": LOFTR_FIELDS}


# ------------------------------------------------------------------------------------- helpers
def _stem(media_id: str, start_frame: int) -> str:
    return f"{media_id}__start{int(start_frame):06d}"


def _parse_stem(stem: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)__start(\d+)", stem)
    if not match:
        raise ValueError(f"{stem}: not <media_id>__start<NNNNNN>")
    return match.group(1), int(match.group(2))


def _shard(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)/(\d+)", text)
    if not match or not int(match.group(1)) < int(match.group(2)):
        raise argparse.ArgumentTypeError(f"{text!r}: expected I/N with 0 <= I < N")
    return int(match.group(1)), int(match.group(2))


def _rollout_start(text: str) -> str | int:
    if text == "window":
        return text
    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"{text!r}: expected window or a source frame")
    return int(text)


def _pair(text: str) -> tuple[str, str]:
    first, _, second = text.partition(",")
    if not first or not second or "," in second:
        raise argparse.ArgumentTypeError(f"{text!r}: expected A,B")
    return first, second


def _named(parser: argparse.ArgumentParser, items: Sequence[str], flag: str) -> dict[str, str]:
    """``NAME=VALUE`` arguments by name; a malformed or repeated name is a usage error."""
    named: dict[str, str] = {}
    for item in items:
        name, _, value = item.partition("=")
        if not name or not value:
            parser.error(f"{flag} {item!r}: expected NAME=PATH")
        if name in named:
            parser.error(f"{flag}: {name} named twice")
        named[name] = value
    return named


def _check_differences(
    parser: argparse.ArgumentParser,
    pairs: Sequence[tuple[str, str]],
    arms: Sequence[str],
    flag: str,
) -> None:
    """A usage error unless every arm of ``--difference`` is one of ``arms``, given by ``flag``."""
    for pair in pairs:
        unknown = [arm for arm in pair if arm not in arms]
        if unknown:
            parser.error(
                f"--difference {','.join(pair)}: no arm {unknown[0]} in {flag} ({', '.join(arms)})"
            )


def _paths(parser: argparse.ArgumentParser, args: argparse.Namespace, *names: str) -> PathsConfig:
    """The ``paths`` of ``--config`` and ``--set``, with ``names`` set."""
    with usage_errors(parser):
        paths = load_config(args.config, parse_overrides(args.overrides)).paths
        paths.require(*names)
    return paths


def _floats(values: Iterable[float]) -> list[float | None]:
    """JSON numbers, ``None`` for NaN."""
    return [None if np.isnan(v) else float(v) for v in values]


def _write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")


def _round_ticks(index: MediaIndex, dataset_root: str, record: MediaRecord) -> dict:
    """Slot -> tick times and states of every player of ``record``'s round: each tick table in the
    round's directory (``player=<pp>/ticks.parquet``), that of a player the media index lacks (no
    recorded video) too; the tables the index lists are checked against it."""
    import pyarrow.parquet as pq

    ticks = {
        slot: tick_states(read_player_ticks(other, dataset_root, jump_recall=False))
        for slot, other in index.slots(record.round_key).items()
    }
    directory = (Path(dataset_root) / record.ticks_path).parents[1]
    for path in sorted(directory.glob("player=*/ticks.parquet")):
        slot = int(path.parent.name.partition("=")[2])
        if slot not in ticks:
            table = pq.read_table(str(path), columns=list(BOX_COLUMNS)).to_pydict()
            states = [np.asarray(table[name], dtype=np.float64) for name in BOX_COLUMNS[1:]]
            ticks[slot] = (np.asarray(table["t"], dtype=np.float64), np.stack(states, axis=1))
    return ticks


def _partners(path: str) -> dict[str, set[int]]:
    """View -> the slots of its partner, the window's other view player (a window-views index)."""
    partners = {}
    for _, row in read_jsonl(path, "window views"):
        view = f"{row['media_id']}@{int(row['start_frame'])}"
        partners[view] = {int(s) for s in row["view_player_slots"]} - {int(row["player_slot"])}
    return partners


class _Rows(NamedTuple):
    """A file of :func:`_depth`: the rows of one arm and its units."""

    rows: np.ndarray
    unit: str
    units: list[str]
    clients: list[list[str]]
    latent_frames: np.ndarray
    start_frames: np.ndarray


def _load_rows(path: str | Path) -> _Rows:
    with np.load(path, allow_pickle=False) as data:
        clients = [[str(m) for m in row if str(m)] for row in data["clients"]]
        return _Rows(
            rows=data["rows"],
            unit=str(data["unit"]),
            units=[str(u) for u in data["units"]],
            clients=clients,
            latent_frames=data["latent_frames"],
            start_frames=data["start_frames"],
        )


def _mean(table: np.ndarray) -> dict[str, list[float | None] | list[int]]:
    """A table cell from per-unit values ``[U, S]``: the mean over the units with a value, and how
    many have one."""
    return {"value": _floats(mean_over_units(table)), "units": np.isfinite(table).sum(0).tolist()}


def _cell(table: np.ndarray) -> dict[str, list[float | None] | list[int]]:
    """:func:`_mean` with its unit-bootstrap interval."""
    low, high = unit_interval(table)
    return {**_mean(table), "low": _floats(low), "high": _floats(high)}


def _print_cell(name: str, cell: dict, segments: Sequence[str]) -> None:
    parts = []
    for i, segment in enumerate(segments):
        value, low, high = (cell[key][i] for key in ("value", "low", "high"))
        if value is None:
            parts.append(f"{segment} -")
        else:
            parts.append(f"{segment} {value:.4g} [{low:.4g}, {high:.4g}] ({cell['units'][i]})")
    print(f"{name}: " + "; ".join(parts))


# ------------------------------------------------------------------------------------------ cut
def _windows(frames: Iterable[np.ndarray]) -> Iterator[list[np.ndarray]]:
    """Consecutive windows of :data:`~worldcast.data.latents.WINDOW_VIDEO_FRAMES` frames, each
    starting on the last frame of the one before; an incomplete last window is dropped."""
    window: list[np.ndarray] = []
    for frame in frames:
        window.append(frame)
        if len(window) == WINDOW_VIDEO_FRAMES:
            yield window
            window = [frame]


def _cut(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    import imageio.v2 as imageio

    from worldcast.engine.inference.decode import write_mp4

    videos = []
    for video in map(Path, args.videos):
        match = re.search(r"-s(\d+)$", video.parent.parent.name)
        if not video.is_file() or not match:
            parser.error(f"{video}: not a <round>/<media_id>/video.mp4 of a session directory")
        videos.append((video, video.parent.name, int(match.group(1))))
    out = Path(args.out_dir)
    for video, media_id, start in videos:
        reader = imageio.get_reader(str(video), format="ffmpeg")
        try:
            count = 0
            for k, frames in enumerate(_windows(reader)):
                write_mp4(frames, out / f"{_stem(media_id, start + WINDOW_STRIDE * k)}.mp4")
                count += 1
        finally:
            reader.close()
        print(f"{video}: {count} windows -> {out}", flush=True)
    return 0


# --------------------------------------------------------------------------------------- detect
def _detect(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from worldcast.engine.evaluation.detector import detect_window, load_detector, write_detections
    from worldcast.utils.files import sha256_file

    paths = _paths(parser, args, "detector", "media_index", "dataset_root")
    recorded = Path(args.eval_dir) / "gt"
    arms = _named(parser, args.arm, "--arm")
    if "real" in arms:
        parser.error("--arm: 'real' names the recorded video")
    stems = sorted(path.stem for path in recorded.glob("*.mp4"))
    if not stems:
        parser.error(f"no recorded videos in {recorded}")
    first, count = args.shard
    stems = stems[first::count]
    for stem in stems:  # every video in place, and every arm's recording the same
        digest = sha256_file(recorded / f"{stem}.mp4")
        for name, eval_dir in arms.items():
            if not (Path(eval_dir) / "generation" / f"{stem}.mp4").is_file():
                parser.error(f"{name}: no generation/{stem}.mp4 in {eval_dir}")
            own = Path(eval_dir) / "gt" / f"{stem}.mp4"
            if own.is_file() and sha256_file(own) != digest:
                parser.error(f"{name}: {own} is not the recorded video {recorded / stem}.mp4")
    index = MediaIndex.load(paths.media_index)
    model = load_detector(paths.detector, args.device)
    ticks: dict = {}
    windows = []
    started = time.time()
    for i, stem in enumerate(stems):
        media_id, start = _parse_stem(stem)
        record = index.media(media_id)
        if record.round_key not in ticks:
            ticks[record.round_key] = _round_ticks(index, paths.dataset_root, record)
        generated = {name: Path(d) / "generation" / f"{stem}.mp4" for name, d in arms.items()}
        windows.append(
            detect_window(
                model,
                recorded / f"{stem}.mp4",
                generated,
                ticks[record.round_key],
                media_id=media_id,
                map_name=record.map_name,
                start_frame=start,
                observer_slot=record.player_slot,
            )
        )
        print(f"{i + 1}/{len(stems)} {stem} ({time.time() - started:.0f} s)", flush=True)
    stamp = {
        "detector_sha256": METRIC_WEIGHTS["detector"].sha256,
        "conf": DETECT_CONF,
        "imgsz": DETECT_IMGSZ,
        "arms": sorted(arms),
        "shard": f"{first}/{count}",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_detections(args.out, windows, stamp)
    print(f"{len(windows)} views -> {args.out}")
    return 0


# --------------------------------------------------------------------------------------- entity
def _predicted_view(path: Path, window: dict, start: int) -> tuple[list, np.ndarray]:
    """The boxes and the gate of a view under the player states given to the generator."""
    with np.load(path) as table:
        states, labels = table["states"], table["labels"]
        offset = start - int(table["start_frame"])
    if offset < 0 or offset % SOURCE_FRAMES_PER_VIDEO_FRAME:
        raise ValueError(f"{path}: the view at {start} is not a frame of the table")
    first, count = offset // SOURCE_FRAMES_PER_VIDEO_FRAME, int(window["n_frames"])
    boxes = table_boxes(states, first, count, window["obs_slot"], window["width"], window["height"])
    return boxes, visibility_gate(labels, first, count)


def _entity(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    windows: dict[str, dict] = {}
    for path in args.dets:
        for window in read_detections(path)["windows"]:
            if window["media_id"] in windows:
                parser.error(f"{window['media_id']} is in two of --dets")
            windows[window["media_id"]] = window
    if not windows:
        parser.error("--dets: no views")
    held = {key[len("dets_") :] for w in windows.values() for key in w if key.startswith("dets_")}
    arms = args.arm or sorted(held - {"real"})
    for arm in arms:
        lacking = [view for view, w in windows.items() if f"dets_{arm}" not in w]
        if lacking:
            parser.error(f"--arm {arm}: not in {len(lacking)} views, e.g. {lacking[0]}")
    label_root = None
    if args.predicted_states is None:
        label_root = _paths(parser, args, "visibility_label_root").visibility_label_root
    partners = None
    if args.scope == "partner":
        with usage_errors(parser):
            partners = _partners(args.views)
        lacking = [view for view in windows if view not in partners]
        if lacking:
            parser.error(f"--views: no row of {len(lacking)} views, e.g. {lacking[0]}")
    counts: dict[str, list] = {arm: [] for arm in arms}
    rates: dict[str, dict[str, dict[str, float]]] = {"rendered": {}, "unmatched": {}}
    per_view = ["arm\tview\tn_pairs\thits\tn_dets\tn_unmatched"]
    per_frame = ["arm\tview\tframe\tin_view\tpairs\thits"]
    dropped = 0
    for view in sorted(windows):
        window = windows[view]
        if args.conf is not None:
            window = at_confidence(window, args.conf)
        media_id, start = view.rsplit("@", 1)
        start, count = int(start), int(window["n_frames"])
        if args.predicted_states is not None:
            table = Path(args.predicted_states) / f"{_stem(media_id, start)}.npz"
            if not table.is_file():
                dropped += 1
                continue
            boxes, gate = _predicted_view(table, window, start)
        else:
            rollout_start = start if args.rollout_start == "window" else args.rollout_start
            with usage_errors(parser):
                gate = gt_visibility_gate(label_root, media_id, start, count, rollout_start)
            boxes = None
        targets = None if partners is None else partners[view]
        for arm in arms:
            v = score_view(
                window,
                arm,
                gate,
                boxes,
                radius=args.radius,
                min_side=args.min_side,
                targets=targets,
            )
            counts[arm].append(v)
            if v.n_pairs or v.n_dets:
                per_view.append(
                    f"{arm}\t{view}\t{v.n_pairs}\t{v.hits}\t{v.n_dets}\t{v.n_unmatched}"
                )
            for measure in rates:
                value = getattr(v, measure)
                if value is not None:
                    rates[measure].setdefault(arm, {})[view] = value
            if args.perframe:
                for f in range(count):
                    per_frame.append(
                        f"{arm}\t{view}\t{f}\t{v.in_view[f]}\t{v.frame_pairs[f]}\t{v.frame_hits[f]}"
                    )
    out: dict[str, Any] = {
        "settings": {
            "conf": DETECT_CONF if args.conf is None else args.conf,
            "radius": args.radius,
            "min_side": args.min_side,
            "states": "given" if args.predicted_states is not None else "recorded",
            "rollout_start": args.rollout_start,
            "scope": args.scope,
        },
        "views": len(windows) - dropped,
        "dropped_views": dropped,
        "summary": {},
        **{measure: {arm: rates[measure].get(arm, {}) for arm in arms} for measure in rates},
    }
    if dropped:
        print(f"{dropped} of {len(windows)} views have no table of player states: left out")
    for arm in arms:
        summary = summarize_entity(counts[arm])
        out["summary"][arm] = {
            "rendered": None if np.isnan(summary.rendered) else summary.rendered,
            "rendered_views": summary.rendered_views,
            "unmatched": None if np.isnan(summary.unmatched) else summary.unmatched,
            "unmatched_views": summary.unmatched_views,
        }
        print(
            f"{arm}: Rendered {summary.rendered:.3f} ({summary.rendered_views}),"
            f" Unmatched {summary.unmatched:.3f} ({summary.unmatched_views})"
        )
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.parent / f"{prefix.name}_perview.tsv").write_text("\n".join(per_view) + "\n")
    if args.perframe:
        out["crowding"] = {}
        for arm in arms:
            frames = (
                np.concatenate([getattr(v, name) for v in counts[arm]] or [np.zeros(0, int)])
                for name in ("in_view", "frame_pairs", "frame_hits")
            )
            out["crowding"][arm] = {
                str(n): {
                    "frames": pooled.frames,
                    "pairs": pooled.pairs,
                    "hits": pooled.hits,
                    "rendered": pooled.rendered if pooled.pairs else None,
                }
                for n, pooled in pool_by_crowding(*frames).items()
            }
        (prefix.parent / f"{prefix.name}_perframe.tsv").write_text("\n".join(per_frame) + "\n")
    _write_json(prefix.parent / f"{prefix.name}.json", out)
    return 0


def _entity_ci(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    rates: dict[str, dict[str, dict[str, float]]] = {"rendered": {}, "unmatched": {}}
    for path in args.scores:
        scores = json.loads(Path(path).read_text())
        for measure, arms in rates.items():
            for arm, values in scores[measure].items():
                if arm in arms:
                    parser.error(f"{arm} is in two of --scores")
                arms[arm] = values
    names = sorted(rates["rendered"])
    _check_differences(parser, args.difference, names, "--scores")
    statistic = np.median if args.statistic == "median" else np.mean
    out: dict[str, Any] = {"statistic": args.statistic, "arms": {}, "differences": {}}
    for arm in names:
        out["arms"][arm] = {}
        for measure, arms in rates.items():
            views = sorted(arms[arm])
            if not views:
                out["arms"][arm][measure] = None
                continue
            values = [arms[arm][view] for view in views]
            low, high = cluster_bootstrap(values, [window_of(v) for v in views], args.statistic)
            point = float(statistic(values))
            out["arms"][arm][measure] = {
                "value": point,
                "low": low,
                "high": high,
                "views": len(views),
            }
            print(f"{arm} {measure}: {point:.3f} [{low:.3f}, {high:.3f}] ({len(views)} views)")
    for first, second in args.difference:
        name = f"{first} - {second}"
        out["differences"][name] = {}
        for measure, arms in rates.items():
            if not set(arms[first]) & set(arms[second]):
                out["differences"][name][measure] = None
                print(f"{name} {measure}: no view scored by both")
                continue
            difference = paired_difference(arms[first], arms[second], window_of, args.statistic)
            out["differences"][name][measure] = {
                "value": difference.point,
                "low": difference.low,
                "high": difference.high,
                "views": difference.views,
                "windows": difference.clusters,
            }
            print(
                f"{name} {measure}: {difference.point:+.3f} [{difference.low:+.3f},"
                f" {difference.high:+.3f}] ({difference.views} views)"
            )
    _write_json(args.out, out)
    return 0


# ---------------------------------------------------------------------------------------- depth
def _depth_units(rounds: str | None, windows: str | None) -> list[tuple[str, int, list[str]]]:
    """The units of ``depth``, ``(name, start frame, clients)``: the rounds of a round index or the
    view pairs of ten-second windows, each with its clients sorted, in the order of their names
    (rounds; windows by round, then start). The rows number the units in this order, as the
    paper's did, and the intervals resample them by their number."""
    units = []
    if rounds:
        rows = read_round_index(rounds)
        for members in rounds_of(rows):
            first, clients = rows[members[0]], sorted(rows[i].media_id for i in members)
            units.append((round_name(first), first.start_frame, clients))
    else:
        for _, row in read_jsonl(windows, "windows"):
            clients = sorted(str(m) for m in row["view_media_ids"])
            units.append((str(row["group_id"]), int(row["start_frame"]), clients))
    return sorted(units)


def _encode_window(vae: Any, video: Path) -> np.ndarray:
    """A ten-second window's mp4 encoded with the Wan2.2 VAE, ``[41, 48, 24, 42]`` float16."""
    from worldcast.data.preparation.latent_cache import encode_windows
    from worldcast.engine.evaluation.fvd import read_window

    frames = read_window(video)
    if len(frames) != WINDOW_VIDEO_FRAMES:
        raise ValueError(f"{video}: {len(frames)} frames, a window has {WINDOW_VIDEO_FRAMES}")
    pixels = ((frames - 0.5) * 2.0).permute(1, 0, 2, 3)[None]
    return encode_windows(vae, pixels)[0, 0]


def _depth(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    import torch

    from worldcast.data.latent_cache import cached_window_starts, load_window_latents
    from worldcast.engine.evaluation.depth_agreement import (
        block_log_depth,
        latent_depth,
        pair_rows,
        recorded_cameras,
        recorded_depth,
        recorded_windows,
    )
    from worldcast.modeling.depth_head import load_depth_predictor

    if args.rounds and (not args.session or args.videos):
        parser.error("--rounds takes --session (the clients' latent frames), not --videos")
    if args.windows and (not args.videos or args.session):
        parser.error("--windows takes --videos (an evaluation directory), not --session")
    required = ["depth_head", "depth_readout", "media_index", "dataset_root"]
    if args.windows:
        required.append("wan22_root")
    elif args.recorded:
        required.append("latent_cache_root")
    paths = _paths(parser, args, *required)
    index = MediaIndex.load(paths.media_index)

    def log_depth(latents: np.ndarray) -> np.ndarray:
        return predictor.log_depth(latents, paper_numerics=True)

    def cameras_of(media_id: str, start_frame: int, count: int):
        record = index.media(media_id)
        table = read_player_ticks(record, paths.dataset_root, jump_recall=False)
        return recorded_cameras(table, start_frame, count, record.fps)

    units = _depth_units(args.rounds, args.windows)
    if args.rounds and args.recorded and any(start for _, start, _ in units):
        parser.error("--recorded reads a round from its first frame: every start_frame is 0")
    recorded = {}
    if args.rounds and args.recorded:
        for _, _, clients in units:
            for media_id in clients:
                recorded[media_id] = recorded_windows(index.media(media_id).source_frames)
        cached = {
            m: set(cached_window_starts(paths.latent_cache_root, m).tolist()) for m in recorded
        }
        missing = [
            {"media_id": media_id, "start_frame": start}
            for media_id, starts in recorded.items()
            for start in starts
            if start not in cached[media_id]
        ]
        if missing:
            listed = Path(args.out).with_suffix(".windows.jsonl")
            listed.parent.mkdir(parents=True, exist_ok=True)
            listed.write_text("".join(json.dumps(window) + "\n" for window in missing))
            parser.error(
                f"the latent cache lacks {len(missing)} of the windows the recorded depth reads,"
                f" e.g. {missing[0]['media_id']} from source frame {missing[0]['start_frame']};"
                f" they are listed in {listed}, for tools/prepare_data.py latent-cache --windows"
            )
    predictor = load_depth_predictor(paths.depth_head, paths.depth_readout, device=args.device)
    vae = None
    if args.windows:
        from worldcast.modeling.wan22.vae import VAE_CHECKPOINT_NAME, load_wan22_vae

        device = torch.device(args.device)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32  # bf16: as the paper
        vae = load_wan22_vae(
            Path(paths.wan22_root) / VAE_CHECKPOINT_NAME, device=device, dtype=dtype
        )
    out_rows, counts = [], []
    started = time.time()
    for unit, (name, start, clients) in enumerate(units):
        depths, cameras = [], []
        for media_id in clients:
            if args.rounds:
                session = Path(args.session) / name / media_id / "latents.npy"
                latents = np.load(session, mmap_mode="r" if args.recorded else None)
                if args.recorded:
                    windows = {
                        first: log_depth(
                            load_window_latents(paths.latent_cache_root, media_id, first)
                        )
                        for first in recorded[media_id]
                    }
                    depths.append(recorded_depth(windows, len(latents)))
                else:
                    depths.append(latent_depth(block_log_depth(log_depth, latents)))
            else:
                folder = "gt" if args.recorded else "generation"
                video = Path(args.videos) / folder / f"{_stem(media_id, start)}.mp4"
                depths.append(latent_depth(log_depth(_encode_window(vae, video))))
            cameras.append(cameras_of(media_id, start, len(depths[-1])))
        counts.append([len(depth) for depth in depths])
        out_rows.append(
            pair_rows(depths, cameras, unit=unit, most_distant_chance=bool(args.windows))
        )
        print(f"{unit + 1}/{len(units)} {name} ({time.time() - started:.0f} s)", flush=True)
    width = max(len(clients) for _, _, clients in units)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        rows=np.concatenate(out_rows),
        unit=np.array("round" if args.rounds else "window"),
        units=np.array([name for name, _, _ in units]),
        clients=np.array([clients + [""] * (width - len(clients)) for _, _, clients in units]),
        latent_frames=np.array([c + [0] * (width - len(c)) for c in counts], dtype=np.int64),
        start_frames=np.array([start for _, start, _ in units], dtype=np.int64),
    )
    print(f"{len(units)} units -> {args.out}")
    return 0


def _depth_cells(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    arms = {name: _load_rows(path) for name, path in _named(parser, args.rows, "--rows").items()}
    first = next(iter(arms.values()))
    for name, arm in arms.items():
        if (arm.unit, arm.units) != (first.unit, first.units):
            parser.error(f"--rows {name}: other units than {next(iter(arms))}")
    _check_differences(parser, args.difference, list(arms), "--rows")
    mask = None if args.mask is None else _load_rows(args.mask).rows
    if first.unit == "round":
        split, segments = SCENE_SPLIT_SECONDS, ("first 30 s", "after 30 s")
    else:
        split, segments = VIEW_SPLIT_SECONDS, ("first 5 s", "last 5 s")
    units = len(first.units)
    tables: dict[str, dict[str, np.ndarray]] = {}
    for name, arm in arms.items():
        rows = arm.rows
        try:
            covisible = None if mask is None else covisibility_mask(rows, mask)
        except ValueError as error:
            parser.error(f"--rows {name} and --mask: {error}")
        tables[name] = {}
        for kind_name, kind in DEPTH_KINDS.items():
            # the mask selects pairs of clients; a client's own next latent frame is not one
            kept = rows if covisible is None or kind == OWN_NEXT_FRAME else rows[covisible]
            tables[name][kind_name] = segment_ratios(kept, kind, units, split)
    out: dict[str, Any] = {
        "unit": first.unit,
        "segments": list(segments),
        "masked": args.mask is not None,
        "arms": {},
        "differences": {},
    }
    for name, kinds in tables.items():
        out["arms"][name] = {kind: _cell(table) for kind, table in kinds.items()}
        for kind, cell in out["arms"][name].items():
            _print_cell(f"{name} {kind}", cell, segments)
    for a, b in args.difference:
        cells = {kind: _cell(tables[a][kind] - tables[b][kind]) for kind in DEPTH_KINDS}
        out["differences"][f"{a} - {b}"] = cells
        for kind, cell in cells.items():
            _print_cell(f"{a} - {b} {kind}", cell, segments)
    _write_json(args.out, out)
    return 0


# ---------------------------------------------------------------------------------------- match
def _match(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from worldcast.engine.evaluation.matching import (
        GIM_STRIDE,
        LOFTR_STRIDE,
        camera_rotations,
        load_gim,
        load_loftr,
        match_keys,
        needed_frames,
        read_frames,
        score_matched_pixels,
        score_relative_poses,
    )

    paths = _paths(parser, args, "gim_dkm", "loftr", "media_index", "dataset_root")
    recorded = _load_rows(args.recorded_rows)
    if recorded.unit != "round" or recorded.start_frames.any():
        parser.error("--recorded-rows: the rows of whole rounds, each from its first frame")
    index = MediaIndex.load(paths.media_index)
    gim, loftr = load_gim(paths.gim_dkm, args.device), load_loftr(paths.loftr, args.device)
    # the recording's chance pairs give the self-check; the LoFTR batches follow the keys' order
    kinds = (BETWEEN_CLIENTS, CHANCE) if args.recorded else (BETWEEN_CLIENTS,)
    keys: dict[str, list[np.ndarray]] = {"gim": [], "loftr": []}
    scores: dict[str, list[np.ndarray]] = {
        f: [] for fields in MATCH_FIELDS.values() for f in fields
    }
    started = time.time()
    for unit, name in enumerate(recorded.units):
        clients = recorded.clients[unit]
        counts = recorded.latent_frames[unit][: len(clients)]
        rows = recorded.rows[recorded.rows[:, 0] == unit]
        unit_keys = {
            "gim": match_keys(rows, counts, GIM_STRIDE, kinds),
            "loftr": match_keys(rows, counts, LOFTR_STRIDE, kinds),
        }
        frames, rotations = {}, {}
        for client, latent in needed_frames(np.concatenate(list(unit_keys.values()))).items():
            record = index.media(clients[client])
            if args.recorded:
                video = Path(paths.dataset_root) / str(record.video_path)
                frames[client] = read_frames(video, latent, SOURCE_FRAMES_PER_LATENT, args.device)
            else:
                video = Path(args.session) / name / clients[client] / "video.mp4"
                frames[client] = read_frames(video, latent, VIDEO_FRAMES_PER_LATENT, args.device)
            table = read_player_ticks(record, paths.dataset_root, jump_recall=False)
            states = table.states(recorded_ticks(table, 0, counts[client], record.fps))
            rotations[client] = camera_rotations(states[:, 3], states[:, 4])
        found = score_matched_pixels(gim, unit_keys["gim"], frames)
        found |= score_relative_poses(loftr, unit_keys["loftr"], frames, rotations)
        for matcher, fields in MATCH_FIELDS.items():
            column = np.full((len(unit_keys[matcher]), 1), unit, dtype=np.int64)
            keys[matcher].append(np.hstack([column, unit_keys[matcher]]))
            for field in fields:
                scores[field].append(found[field])
        print(
            f"{unit + 1}/{len(recorded.units)} {name} ({time.time() - started:.0f} s)", flush=True
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        units=np.array(recorded.units),
        gim_keys=np.concatenate(keys["gim"]),
        loftr_keys=np.concatenate(keys["loftr"]),
        **{field: np.concatenate(values) for field, values in scores.items()},
    )
    print(f"{len(recorded.units)} rounds -> {args.out}")
    return 0


def _match_cells(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from worldcast.engine.evaluation.matching import (
        STATISTICS,
        covisible,
        round_statistics,
        self_check,
    )

    recorded = _load_rows(args.recorded_rows)
    arms = {}
    for name, path in _named(parser, args.pairs, "--pairs").items():
        with np.load(path, allow_pickle=False) as data:
            arms[name] = {key: data[key] for key in data.files}
        if [str(u) for u in arms[name]["units"]] != recorded.units:
            parser.error(f"--pairs {name}: other rounds than --recorded-rows")
    _check_differences(parser, args.difference, list(arms), "--pairs")
    if args.self_check and args.self_check not in arms:
        parser.error(f"--self-check: no arm {args.self_check} in --pairs ({', '.join(arms)})")
    units = range(len(recorded.units))
    matcher_of = {field: matcher for matcher, fields in MATCH_FIELDS.items() for field in fields}

    def per_round(arm: dict, field: str, measure) -> np.ndarray:
        """``[R, 2]`` one value per round and segment (or same instant / 10 s apart)."""
        keys = arm[f"{matcher_of[field]}_keys"]
        out = []
        for unit in units:
            mine = keys[:, 0] == unit
            seen = covisible(keys[mine, 1:], recorded.rows[recorded.rows[:, 0] == unit])
            out.append(measure(keys[mine, 1:], arm[field][mine], seen, STATISTICS[field]))
        return np.array(out, dtype=np.float64)

    tables = {
        name: {f: per_round(arm, f, round_statistics) for f in STATISTICS}
        for name, arm in arms.items()
    }
    segments = ("first 30 s", "after 30 s")
    out: dict[str, Any] = {"segments": list(segments), "arms": {}, "differences": {}}
    for name, fields in tables.items():
        out["arms"][name] = {}
        for field, table in fields.items():
            cell = out["arms"][name][field] = _mean(table)
            values = " / ".join("-" if v is None else f"{v:.4g}" for v in cell["value"])
            print(f"{name} {field}: {values} ({' / '.join(map(str, cell['units']))} rounds)")
    for a, b in args.difference:
        cells = {field: _cell(tables[a][field] - tables[b][field]) for field in STATISTICS}
        out["differences"][f"{a} - {b}"] = cells
        for field, cell in cells.items():
            _print_cell(f"{a} - {b} {field}", cell, segments)
    if args.self_check:
        out["self_check"] = {}
        for field in STATISTICS:
            cell = out["self_check"][field] = _mean(
                per_round(arms[args.self_check], field, self_check)
            )
            same, apart = ("-" if v is None else f"{v:.4g}" for v in cell["value"])
            print(f"self-check {field}: same instant {same}, 10 s apart {apart}")
    _write_json(args.out, out)
    return 0


# ------------------------------------------------------------------------------------------ fvd
def _fvd(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    import torch

    from worldcast.engine.evaluation.fvd import (
        frechet_distance,
        fvd_from_dir,
        load_i3d,
        round_features,
    )

    if args.rounds and (not args.sessions or args.real_cache) or args.eval_dir and args.sessions:
        parser.error(
            "--eval-dir takes [--real-cache] (ten-second windows), --rounds takes"
            " --sessions NAME=DIR (whole rounds)"
        )
    sessions = _named(parser, args.sessions, "--sessions")
    paths = _paths(parser, args, "i3d", *(("media_index", "dataset_root") if args.rounds else ()))
    model = load_i3d(paths.i3d, args.device)
    out: dict[str, Any] = {"i3d_sha256": METRIC_WEIGHTS["i3d"].sha256}
    if args.rounds:
        rows = read_round_index(args.rounds)
        if any(row.start_frame for row in rows):
            parser.error("--rounds: whole rounds are scored from their first frame (start_frame 0)")
        index = MediaIndex.load(paths.media_index)
        out["rounds"] = {}
        for name, session in sessions.items():
            fake, real = [], []
            for row in rows:
                record = index.media(row.media_id)
                video = Path(session) / round_name(row) / row.media_id / "video.mp4"
                recording = Path(paths.dataset_root) / str(record.video_path)
                generated, recorded = round_features(
                    model, video, recording, source_frames=record.source_frames
                )
                fake.append(generated)
                real.append(recorded)
            fake_set, real_set = torch.cat(fake), torch.cat(real)
            value = frechet_distance(real_set, fake_set)
            out["rounds"][name] = {"fvd": value, "clips": len(fake_set)}
            print(f"{name}: FVD {value:.3f} over {len(fake_set)} ten-second segments")
    else:
        result = fvd_from_dir(args.eval_dir, model, real_cache=args.real_cache)
        out["windows"] = {**result, "frames_per_clip": WINDOW_VIDEO_FRAMES}
        print(f"{args.eval_dir}: FVD {result['fvd']:.3f} over {result['clips']} windows")
    _write_json(args.out, out)
    return 0


# --------------------------------------------------------------------------------------- vbench
def _print_bins(title: str, bins: dict, label) -> None:
    from worldcast.engine.evaluation.vbench import DIMENSIONS

    print(title)
    for key, row in bins.items():
        values = " ".join(f"{dim}={row[dim]:.4f}" for dim in DIMENSIONS if dim in row)
        print(f"  {label(key)} ({row['clips']} clips): {values}")


def _vbench(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from worldcast.engine.evaluation.vbench import HOUR_BIN_SECONDS, ROUND_BIN_FRAMES

    if not (args.hour or args.whole_rounds or args.sensitivity):
        parser.error("give --hour, --whole-rounds or --sensitivity")
    out: dict[str, Any] = {}
    if args.hour:
        bins = hour_bins(read_shards(args.hour))
        out["hour"] = {str(k): row for k, row in bins.items()}
        minutes = HOUR_BIN_SECONDS // 60
        _print_bins("one-hour rollouts", bins, lambda k: f"{minutes * k}-{minutes * (k + 1)} min")
    if args.whole_rounds:
        bins = round_bins(read_shards(args.whole_rounds))
        out["whole_rounds"] = {str(k): row for k, row in bins.items()}
        seconds = round(ROUND_BIN_FRAMES / SOURCE_FPS)
        _print_bins("whole rounds", bins, lambda k: f"{seconds * k}-{seconds * (k + 1)} s")
    if args.sensitivity:
        changes = vbench_sensitivity(read_shards(args.sensitivity))
        out["sensitivity"] = {
            damage: {"damage": DAMAGES.get(damage, damage), **row}
            for damage, row in changes.items()
        }
        print("sensitivity, percent change of the mean")
        for damage, row in changes.items():
            values = " ".join(f"{k}={v:+.1f}%" for k, v in row.items() if k != "clips")
            print(f"  {DAMAGES.get(damage, damage)} ({row['clips']} clips): {values}")
    _write_json(args.out, out)
    return 0


# ----------------------------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    """Run one step; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    steps = parser.add_subparsers(dest="step", required=True, metavar="step")

    def step(name: str, run, summary: str, *, config: bool = False) -> argparse.ArgumentParser:
        sub = steps.add_parser(name, help=summary, description=summary)
        sub.set_defaults(run=run, parser=sub)
        if config:
            add_config_args(sub)
        return sub

    cut = step("cut", _cut, "Cut round-long videos into ten-second windows.")
    cut.add_argument(
        "--videos", nargs="+", required=True, help="<round>/<media_id>/video.mp4 of sessions"
    )
    cut.add_argument("--out-dir", required=True, help="receives <media_id>__start<NNNNNN>.mp4")

    detect = step("detect", _detect, "Run the player detector on the views' videos.", config=True)
    detect.add_argument(
        "--eval-dir", required=True, help="the evaluation directory whose gt/ holds the views"
    )
    detect.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="an arm: the generation/ of an evaluation directory (repeat)",
    )
    detect.add_argument("--shard", type=_shard, default=(0, 1), help="I/N: every N-th view from I")
    detect.add_argument("--device", default="cuda", help="torch device")
    detect.add_argument("--out", required=True, help="the detection dump, .json.gz")

    entity = step(
        "entity", _entity, "Score Rendered and Unmatched from detection dumps.", config=True
    )
    entity.add_argument("--dets", nargs="+", required=True, help="detection dumps (shards)")
    entity.add_argument("--arm", action="append", help="an arm to score (default: every one)")
    entity.add_argument(
        "--rollout-start",
        type=_rollout_start,
        default="window",
        metavar="window|FRAME",
        help="where the rollout of a view began: at the view (ten-second windows) or at this"
        " source frame (windows of whole rounds: the sessions' first frame, 0 for the paper's)",
    )
    entity.add_argument(
        "--scope",
        choices=("all", "partner"),
        default="all",
        help="the players whose pairs Rendered scores: every one, or the partner alone (the"
        " window's other view player, from --views)",
    )
    entity.add_argument(
        "--views",
        default=str(ROOT / "configs/eval/window_views.jsonl"),
        help="the window views of --scope partner",
    )
    entity.add_argument(
        "--predicted-states",
        metavar="DIR",
        help="<stem>.npz per view, the player states given to the generator (states, labels:"
        " visible and valid, start_frame); default: the recorded states and"
        " paths.visibility_label_root",
    )
    entity.add_argument("--conf", type=float, help="drop detections below this confidence")
    entity.add_argument("--radius", type=float, default=MATCH_RADIUS, help="body widths")
    entity.add_argument("--min-side", type=float, default=MIN_SIDE, help="smallest box, px")
    entity.add_argument(
        "--perframe", action="store_true", help="also every frame's counts and their crowding"
    )
    entity.add_argument("--out", required=True, help="prefix of <out>.json and the TSV files")

    entity_ci = step(
        "entity-ci", _entity_ci, "Bootstrap intervals and paired differences of entity scores."
    )
    entity_ci.add_argument("--scores", nargs="+", required=True, help="<out>.json of entity")
    entity_ci.add_argument(
        "--difference", action="append", type=_pair, default=[], metavar="A,B", help="A - B"
    )
    entity_ci.add_argument(
        "--statistic",
        choices=("median", "mean"),
        default="median",
        help="the summary over views: median, or mean as Table 12",
    )
    entity_ci.add_argument("--out", required=True, help="the result JSON")

    depth = step("depth", _depth, "Depth agreement rows of one arm.", config=True)
    units = depth.add_mutually_exclusive_group(required=True)
    units.add_argument("--rounds", help="a round index: whole rounds (Tables 19, 4a)")
    units.add_argument("--windows", help="the view pairs of ten-second windows (Table 10)")
    depth.add_argument("--session", help="the session directory of --rounds")
    depth.add_argument(
        "--videos", help="the evaluation directory of --windows: its generation/ (or gt/) videos"
    )
    depth.add_argument(
        "--recorded",
        action="store_true",
        help="the recorded video instead: the gt/ videos of --windows, or for --rounds the"
        " recordings' windows of the latent cache every 161 source frames from 0, then from 80",
    )
    depth.add_argument("--device", default="cuda", help="torch device")
    depth.add_argument("--out", required=True, help="the rows, .npz")

    depth_cells = step("depth-cells", _depth_cells, "Table cells of depth agreement rows.")
    depth_cells.add_argument(
        "--rows", action="append", required=True, metavar="NAME=FILE", help="an arm (repeat)"
    )
    depth_cells.add_argument(
        "--mask", help="the recorded video's rows: keep the pairs co-visible in it (Table 19)"
    )
    depth_cells.add_argument(
        "--difference", action="append", type=_pair, default=[], metavar="A,B", help="A - B"
    )
    depth_cells.add_argument("--out", required=True, help="the result JSON")

    match = step("match", _match, "Mat. Pix., Inliers and RotErr of one arm.", config=True)
    match.add_argument(
        "--recorded-rows", required=True, help="depth rows of the recorded video (the pairs)"
    )
    arm = match.add_mutually_exclusive_group(required=True)
    arm.add_argument("--session", help="the session directory whose videos are compared")
    arm.add_argument("--recorded", action="store_true", help="the recordings instead")
    match.add_argument("--device", default="cuda", help="torch device")
    match.add_argument("--out", required=True, help="the per-pair results, .npz")

    match_cells = step("match-cells", _match_cells, "Table cells of the per-pair results.")
    match_cells.add_argument(
        "--pairs", action="append", required=True, metavar="NAME=FILE", help="an arm (repeat)"
    )
    match_cells.add_argument("--recorded-rows", required=True, help="as given to match")
    match_cells.add_argument(
        "--difference", action="append", type=_pair, default=[], metavar="A,B", help="A - B"
    )
    match_cells.add_argument("--self-check", metavar="NAME", help="the recorded arm of --pairs")
    match_cells.add_argument("--out", required=True, help="the result JSON")

    fvd = step("fvd", _fvd, "FVD of ten-second windows or of whole rounds.", config=True)
    clips = fvd.add_mutually_exclusive_group(required=True)
    clips.add_argument("--eval-dir", help="an evaluation directory of ten-second windows")
    clips.add_argument("--rounds", help="a round index: whole rounds (Table 17)")
    fvd.add_argument("--real-cache", help="an .npz of the recorded clips' features to reuse")
    fvd.add_argument(
        "--sessions",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="an arm of --rounds: its session directory (repeat)",
    )
    fvd.add_argument("--device", default="cuda", help="torch device")
    fvd.add_argument("--out", required=True, help="the result JSON")

    vbench = step("vbench", _vbench, "Aggregate VBench's per-clip scores into the paper's bins.")
    vbench.add_argument("--hour", nargs="+", help="shard files of one-hour rollouts")
    vbench.add_argument("--whole-rounds", nargs="+", help="shard files of whole-round windows")
    vbench.add_argument("--sensitivity", nargs="+", help="shard files of the sensitivity test")
    vbench.add_argument("--out", required=True, help="the result JSON")

    args = parser.parse_args(argv)
    return args.run(args, args.parser)


if __name__ == "__main__":
    sys.exit(main())
