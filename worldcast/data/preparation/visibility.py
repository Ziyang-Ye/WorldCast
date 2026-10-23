"""The GT visibility labels: line-of-sight tests of every player pair.

For each recording of the media index (or those named by --media-id) it tests every other player
of the round against the recording player's view, frame by frame, on the map's mesh, and writes
<out-dir>/<media_id>.npz: [10, T] arrays per player slot over the recording's T video frames
(docs/inference.md, "Data"). A file that is already there from the same inputs is kept, so an
interrupted run resumes; --workers runs one process per worker, and the files do not depend on it.

Example::

    python tools/prepare_data.py visibility --media-index data/media_index.jsonl \\
        --dataset-root data/opencs2 --out-dir data/vislabels \\
        --visual-meshes de_nuke=meshes/de_nuke_visual_complete.glb --workers 8
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from worldcast.data.game import TICK_RATE
from worldcast.data.recordings import (
    MediaIndex,
    MediaRecord,
    RoundKey,
    read_tick_columns,
    verify_ticks_file,
)
from worldcast.utils.files import atomic_write, sha256_file

from ._visibility_geometry import (
    Occluders,
    load_collision_occluders,
    load_visual_occluders,
)
from ._visibility_labels import SCHEMA, PlayerTicks, decode_meta, encode_meta, observer_labels
from ._visibility_scope import scoped_frames

__all__ = ["ROUND_TAIL_TICKS", "VisibilityJob", "add_arguments", "derive_round_events", "run"]

#: The recordings of a round run this long past its end, ticks (0.5 s).
ROUND_TAIL_TICKS = 32
#: What a label file says about itself (``_meta.semantics``).
SEMANTICS = (
    "Binary visibility from the map's opaque triangles and exact radius-16 capsules of the other"
    " players; mask, blend and effect triangles are diagnostic. Pair _binary_visible with"
    " _binary_eval_valid: false outside that mask is unknown, not occluded. Frames seen through a"
    " scope, and their neighbours, are not evaluated."
)
_TICK_COLUMNS = ("tick", "t", "x", "y", "z", "yaw", "pitch", "is_alive", "active", "team_num")
_TEAMS = (2, 3)


@dataclass(frozen=True)
class VisibilityJob:
    """What :func:`run` is asked to do (the command line's options).

    Attributes:
        media_index (str): the media index, JSONL.
        dataset_root (str): the recordings: tick tables and, for ``scope == "video"``, videos.
        out_dir (str): where the label files go.
        meshes (dict[str, str]): ``{map_name: collision mesh}`` (``data.collision_meshes``), used
            for a map without a visual mesh.
        visual_meshes (dict[str, str]): ``{map_name: visual GLB}``.
        pov_index (str | None): OpenCS2's ``index/pov_rounds.parquet``: each player's
            ``death_tick`` and each round's ``round_end_tick``; None derives them from the ticks.
        scope (str): ``"video"`` finds the scoped frames in the recorded videos, ``"none"`` takes
            every frame as unscoped.
        capsule_unknown (bool): a player that only another player's capsule hides is unknown, not
            occluded.
        media_ids (tuple[str, ...]): the recordings to label; empty: every one.
        workers (int): processes.
        overwrite (bool): redo files that are already there.
    """

    media_index: str
    dataset_root: str
    out_dir: str
    meshes: dict[str, str] = field(default_factory=dict)
    visual_meshes: dict[str, str] = field(default_factory=dict)
    pov_index: str | None = None
    scope: str = "video"
    capsule_unknown: bool = False
    media_ids: tuple[str, ...] = ()
    workers: int = 1
    overwrite: bool = False


def _map_path(text: str) -> tuple[str, str]:
    name, separator, path = text.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"expected MAP=PATH, got {text!r}")
    return name, path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this step's options to its subcommand's parser."""
    parser.add_argument("--media-index", required=True, help="media index (JSONL)")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="the recordings: rounds/match_id=<m>/.../ticks.parquet",
    )
    parser.add_argument("--out-dir", required=True, help="writes <media_id>.npz here")
    parser.add_argument(
        "--visual-meshes",
        action="append",
        type=_map_path,
        default=[],
        metavar="MAP=PATH",
        help="a map's visual GLB (Source2Viewer-CLI export of maps/<map>.vmap_c with"
        " --gltf_export_materials and --gltf_export_extras); repeat per map",
    )
    parser.add_argument(
        "--meshes",
        action="append",
        type=_map_path,
        default=[],
        metavar="MAP=PATH",
        help="a map's collision mesh (data.collision_meshes), for a map without --visual-meshes:"
        " every triangle is then opaque, so the labels differ from the visual mesh's",
    )
    parser.add_argument(
        "--pov-index",
        help="OpenCS2 index/pov_rounds.parquet, for each player's death_tick and each round's"
        " round_end_tick (default: derived from the tick tables: a player's first dead tick, or the"
        f" tick after its table when that ends before the round; the round ends {ROUND_TAIL_TICKS}"
        " ticks before the last recorded tick)",
    )
    parser.add_argument(
        "--scope",
        choices=("video", "none"),
        default="video",
        help="video: leave out the frames seen through a scope, found in the recorded videos;"
        " none: take every frame as unscoped (default: video)",
    )
    parser.add_argument(
        "--capsule-unknown",
        action="store_true",
        help="a player that only another player's capsule hides, while the map leaves a probe"
        " clear, is unknown instead of occluded (as in the paper's de_dust2 labels)",
    )
    parser.add_argument(
        "--media-id",
        nargs="+",
        action="extend",
        default=[],
        dest="media_ids",
        metavar="ID",
        help="label only these recordings (default: every recording of the index)",
    )
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default: 1)")
    parser.add_argument(
        "--overwrite", action="store_true", help="redo the files that are already there"
    )


# ------------------------------------------------------------------------------------ the round
def derive_round_events(
    ticks: dict[int, np.ndarray], alive: dict[int, np.ndarray]
) -> tuple[int, dict[int, int | None]]:
    """The round's end tick and each player's death tick, from the tick tables alone.

    The recordings run :data:`ROUND_TAIL_TICKS` past the round's end, so the round ends that many
    ticks before the last recorded tick. A player is dead from its first not-alive row; one whose
    table ends before the round ends died when it did, one tick after its last row.

    Args:
        ticks (dict[int, np.ndarray]): by slot, ``[n]`` the ticks of the player's table.
        alive (dict[int, np.ndarray]): by slot, ``[n]`` bool, its ``is_alive``.

    Returns:
        tuple[int, dict[int, int | None]]: the round's end tick and each slot's death tick (None:
        alive at the end).
    """
    round_end = max(int(t[-1]) for t in ticks.values()) - ROUND_TAIL_TICKS
    death: dict[int, int | None] = {}
    for slot, tick in ticks.items():
        if not alive[slot].all():
            death[slot] = int(tick[np.argmax(~alive[slot])])
        else:
            death[slot] = int(tick[-1]) + 1 if int(tick[-1]) < round_end else None
    return round_end, death


def _read_pov_events(path: str, media_ids: set[str]) -> dict[str, tuple[int | None, int]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["media_id", "death_tick", "round_end_tick"]).to_pydict()
    events = {}
    for media_id, death, end in zip(
        table["media_id"], table["death_tick"], table["round_end_tick"], strict=True
    ):
        if media_id in media_ids:
            events[str(media_id)] = (None if death is None else int(death), int(end))
    missing = sorted(media_ids - set(events))
    if missing:
        raise ValueError(f"{path} has no row for {len(missing)} recordings, first {missing[:3]}")
    return events


def _read_ticks(record: MediaRecord, dataset_root: Path) -> tuple[dict, str]:
    path = dataset_root / record.ticks_path
    digest = verify_ticks_file(record, path)
    table = read_tick_columns(path, _TICK_COLUMNS, record.ticks_rows)
    tick = np.asarray(table["tick"], dtype=np.int64)
    if np.any(np.diff(tick) <= 0):
        raise RuntimeError(f"ticks are not strictly increasing in {path}")
    seconds = (tick - tick[0]) / TICK_RATE
    if not np.allclose(np.asarray(table["t"], dtype=np.float64), seconds, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"tick and time columns disagree in {path}")
    if record.capture_start_tick is not None and int(tick[0]) != record.capture_start_tick:
        raise RuntimeError(f"the first tick of {path} is not the media row's capture_start_tick")
    team = np.asarray(table["team_num"], dtype=np.int64)
    if not np.isin(team, _TEAMS).all():
        raise RuntimeError(f"team_num is not 2 or 3 in {path}")
    alive = np.asarray(table["is_alive"], dtype=bool)
    columns = [
        np.asarray(table[name], dtype=np.float64) for name in ("x", "y", "z", "yaw", "pitch")
    ]
    return {
        "tick": tick,
        "states": np.stack(columns, axis=1),
        "alive": alive,
        "crouched": np.asarray(["duck" in set(held or ()) for held in table["active"]], dtype=bool),
        "team": int(team[0]),
    }, digest


@dataclass(frozen=True)
class _Round:
    roster: dict[int, PlayerTicks]
    round_end_tick: int
    inputs: dict


def _read_round(
    records: dict[int, MediaRecord],
    dataset_root: Path,
    events: dict[str, tuple[int | None, int]] | None,
) -> _Round:
    tables, digests = {}, {}
    for slot, record in records.items():
        tables[slot], digests[record.media_id] = _read_ticks(record, dataset_root)
    if events is None:
        round_end, death = derive_round_events(
            {slot: t["tick"] for slot, t in tables.items()},
            {slot: t["alive"] for slot, t in tables.items()},
        )
    else:
        ends = {events[r.media_id][1] for r in records.values()}
        if len(ends) != 1:
            raise ValueError(
                f"the round of {next(iter(records.values())).media_id} has {ends} ends"
            )
        round_end = ends.pop()
        death = {slot: events[r.media_id][0] for slot, r in records.items()}
    roster = {slot: PlayerTicks(death_tick=death[slot], **t) for slot, t in tables.items()}
    inputs = {
        "ticks_sha256": digests,
        "round_end_tick": round_end,
        "death_tick": {str(slot): death[slot] for slot in sorted(death)},
    }
    return _Round(roster, round_end, inputs)


# ------------------------------------------------------------------------------------ one process
class _Maps:
    """A process's occluders per map, loaded on first use, with the identity of their files."""

    def __init__(self, job: VisibilityJob) -> None:
        self.job = job
        self.occluders: dict[str, Occluders] = {}
        self.identities: dict[str, dict] = {}

    def kind_and_path(self, map_name: str) -> tuple[str, str]:
        if map_name in self.job.visual_meshes:
            return "visual", self.job.visual_meshes[map_name]
        if map_name in self.job.meshes:
            return "collision", self.job.meshes[map_name]
        raise ValueError(
            f"no mesh for {map_name}: pass --visual-meshes {map_name}=<GLB> or"
            f" --meshes {map_name}=<GLB>"
        )

    def identity(self, map_name: str) -> dict:
        if map_name not in self.identities:
            kind, path = self.kind_and_path(map_name)
            self.identities[map_name] = {"mesh": kind, "mesh_sha256": sha256_file(path)}
        return self.identities[map_name]

    def get(self, map_name: str) -> Occluders:
        if map_name not in self.occluders:
            kind, path = self.kind_and_path(map_name)
            loader = load_visual_occluders if kind == "visual" else load_collision_occluders
            self.occluders[map_name] = loader(path)
        return self.occluders[map_name]


def _settings(job: VisibilityJob, mesh: dict) -> dict:
    return {
        **mesh,
        "scope": job.scope,
        "capsule_unknown": job.capsule_unknown,
        "round_events": "index" if job.pov_index else "ticks",
    }


def _label_one(
    job: VisibilityJob, maps: _Maps, record: MediaRecord, found: _Round, path: Path
) -> bool:
    """Label one recording into ``path``; False when a file from the same inputs is there."""
    meta = {
        "schema": SCHEMA,
        "media_id": record.media_id,
        "match_id": record.match_id,
        "map_name": record.map_name,
        "round": record.round,
        "observer_slot": record.player_slot,
        "video_frames": record.source_frames,
        "fps": record.fps,
        "settings": _settings(job, maps.identity(record.map_name)),
        "inputs": found.inputs,
    }
    if path.is_file() and not job.overwrite:
        with np.load(path) as archive:
            existing = decode_meta(archive["_meta"])
        if all(existing.get(key) == meta[key] for key in meta):
            return False
        raise ValueError(f"{path} was made from other inputs or settings; pass --overwrite")
    frames = record.source_frames
    if job.scope == "video":
        if record.video_path is None:
            raise ValueError(f"{record.media_id} has no video_path: pass --scope none")
        scoped = scoped_frames(Path(job.dataset_root) / record.video_path, frames)
    else:
        scoped = np.zeros(frames, dtype=bool)
    occluders = maps.get(record.map_name)
    arrays, moved = observer_labels(
        record.player_slot,
        found.roster,
        fps=record.fps,
        frames=frames,
        round_end_tick=found.round_end_tick,
        scoped=scoped,
        occluders=occluders,
        capsule_unknown=job.capsule_unknown,
    )
    meta.update(
        mesh=occluders.summary,
        capture_start_tick=int(found.roster[record.player_slot].tick[0]),
        round_end_tick=found.round_end_tick,
        scoped_frames=int(scoped.sum()),
        capsule_unknown_cells=moved,
        semantics=SEMANTICS,
    )
    arrays["_meta"] = encode_meta(meta)
    atomic_write(path, lambda handle: np.savez_compressed(handle, **arrays))
    return True


def _run_shard(job: VisibilityJob, rounds: list[tuple[RoundKey, list[str]]], worker: int) -> dict:
    index = MediaIndex.load(job.media_index)
    maps = _Maps(job)
    dataset_root = Path(job.dataset_root)
    out_dir = Path(job.out_dir)
    events = None
    if job.pov_index:
        needed = {r.media_id for key, _ in rounds for r in index.slots(key).values()}
        events = _read_pov_events(job.pov_index, needed)
    counts = {"labelled": 0, "kept": 0}
    for number, (key, media_ids) in enumerate(rounds, start=1):
        records = index.slots(key)
        found = _read_round(records, dataset_root, events)
        for media_id in media_ids:
            made = _label_one(job, maps, index.media(media_id), found, out_dir / f"{media_id}.npz")
            counts["labelled" if made else "kept"] += 1
        print(f"[worker {worker}] round {number}/{len(rounds)} {counts}", flush=True)
    return counts


def run(args: argparse.Namespace) -> int:
    """Run this step; returns the exit status."""
    job = VisibilityJob(
        media_index=args.media_index,
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        meshes=dict(args.meshes),
        visual_meshes=dict(args.visual_meshes),
        pov_index=args.pov_index,
        scope=args.scope,
        capsule_unknown=args.capsule_unknown,
        media_ids=tuple(args.media_ids),
        workers=args.workers,
        overwrite=args.overwrite,
    )
    if job.workers < 1:
        raise ValueError("--workers must be at least 1")
    index = MediaIndex.load(job.media_index)
    observers = list(job.media_ids) or sorted(index.by_id)
    by_round: dict[RoundKey, list[str]] = {}
    for media_id in observers:
        try:
            by_round.setdefault(index.media(media_id).round_key, []).append(media_id)
        except KeyError as error:
            raise ValueError(str(error.args[0])) from None
    maps = _Maps(job)
    for map_name in sorted({key[1] for key in by_round}):
        maps.kind_and_path(map_name)
    Path(job.out_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"labelling {len(observers)} recordings of {len(by_round)} rounds, {job.workers} workers",
        flush=True,
    )
    # a worker takes a run of rounds ordered by map, so that it loads few of the maps' meshes
    ordered = sorted(by_round, key=lambda key: (key[1], key[0], key[2]))
    size = -(-len(ordered) // job.workers)
    shards = [
        [(key, sorted(by_round[key])) for key in ordered[i * size : (i + 1) * size]]
        for i in range(job.workers)
    ]
    if job.workers == 1:
        results = [_run_shard(job, shards[0], 0)]
    else:
        with ProcessPoolExecutor(
            max_workers=job.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(_run_shard, job, shard, i) for i, shard in enumerate(shards)]
            results = [future.result() for future in futures]
    total = {name: sum(r[name] for r in results) for name in ("labelled", "kept")}
    print(f"visibility labels in {job.out_dir}: {total['labelled']} written, {total['kept']} kept")
    return 0
