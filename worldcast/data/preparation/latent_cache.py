"""The latent cache: every window encoded with the Wan2.2 VAE.

Writes ``<media_id>.npz`` with a member ``win_<start:06d>`` per window, float16
``[1, 41, 48, 24, 42]`` (docs/training.md, "Data formats"): the VAE's latents of the frames
``start, start + 2, ..., start + 320`` of the recording, resized bilinearly to 384 x 672 and mapped
to ``(x / 255 - 0.5) * 2``. Reads the media index and the videos (``data.media_index``,
``data.dataset_root``), the Wan2.2 snapshot (``model.wan22_root``) and the windows: the bucket files
of ``data.bucket_dir``, the ``--windows`` files (JSONL, a ``media_id`` and a ``start_frame`` per
line) or every ``--stride``-th start frame of every video. The archives go to ``--out-dir``
(default ``data.latent_cache_root``).

A run skips the windows its archives hold, so it can be stopped and started again, and writes each
archive to a temporary file that it renames into place. One process encodes on one GPU;
``--shard I/N`` gives it the I-th of N parts of the recordings.

Examples::

    python tools/prepare_data.py latent-cache --config train_paths.yaml
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python tools/prepare_data.py latent-cache \\
            --config train_paths.yaml --shard $i/4 &
    done; wait
"""

import argparse
import os
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch

from worldcast.config.loader import add_config_args, load, merge, parse_overrides, require_set
from worldcast.config.training import TrainConfig
from worldcast.modeling.wan22.vae import VAE_CHECKPOINT_NAME, Wan22VAE, load_wan22_vae

from ..latent_cache import cached_window_starts
from ..latents import WINDOW_SOURCE_FRAMES, WINDOW_VIDEO_FRAMES, window_key
from ..recordings import MediaIndex, MediaRecord, read_jsonl
from ..training import BUCKET_NAMES
from ..video import load_video_window

__all__ = [
    "add_arguments",
    "bucket_files",
    "cache_media",
    "check_windows",
    "encode_windows",
    "read_windows",
    "run",
    "select_media",
    "tile_windows",
]

#: dtype of the VAE's weights and of the pixels it encodes on CUDA: bf16, as the paper's cache
#: (float32 gives other latents).
CACHE_DTYPE = torch.bfloat16

#: ``{media_id: sorted start frames}``, the recordings in ascending order.
Windows = dict[str, list[int]]


# ============================================================================== the windows
def bucket_files(bucket_dir: str | Path) -> list[Path]:
    """The five bucket files ``train_<bucket>.jsonl`` of ``bucket_dir``."""
    return [Path(bucket_dir) / f"train_{name}.jsonl" for name in BUCKET_NAMES]


def read_windows(paths: Iterable[str | Path]) -> Windows:
    """The windows of JSONL files with a ``media_id`` and a ``start_frame`` per line (the bucket
    files, the validation index); a window named twice counts once."""
    starts: dict[str, set[int]] = {}
    for path in paths:
        for line_number, row in read_jsonl(path, "window file"):
            try:
                media_id, start_frame = str(row["media_id"]), int(row["start_frame"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"{path}:{line_number}: a window has a media_id and a start_frame"
                ) from None
            starts.setdefault(media_id, set()).add(start_frame)
    return {media_id: sorted(starts[media_id]) for media_id in sorted(starts)}


def tile_windows(records: Iterable[MediaRecord], stride: int) -> Windows:
    """The windows of every recording from source frame 0 on, ``stride`` source frames apart, as
    long as they fit in it; a recording shorter than a window has none."""
    windows: Windows = {}
    for record in sorted(records, key=lambda record: record.media_id):
        last_start = record.source_frames - WINDOW_SOURCE_FRAMES
        if last_start >= 0:
            windows[record.media_id] = list(range(0, last_start + 1, stride))
    return windows


def select_media(windows: Windows, media_ids: Sequence[str]) -> Windows:
    """The windows of the recordings ``media_ids``; each must have some."""
    missing = sorted(set(media_ids) - set(windows))
    if missing:
        raise ValueError(f"no windows of {', '.join(missing)}")
    return {media_id: windows[media_id] for media_id in sorted(set(media_ids))}


def check_windows(windows: Windows, index: MediaIndex) -> None:
    """Raise ``ValueError`` unless every window lies in a recording of the media index that has a
    video."""
    unknown = sorted(set(windows) - set(index.by_id))
    if unknown:
        raise ValueError(f"{len(unknown)} recordings are not in the media index: {unknown[:3]}")
    for media_id, starts in windows.items():
        record = index.by_id[media_id]
        if record.video_path is None:
            raise ValueError(f"{media_id} has no video_path in the media index")
        outside = [s for s in starts if s < 0 or s + WINDOW_SOURCE_FRAMES > record.source_frames]
        if outside:
            raise ValueError(
                f"{media_id}: the window at {outside[0]} is not inside the recording's"
                f" {record.source_frames} frames"
            )


# ================================================================================ the encoder


def encode_windows(vae: Wan22VAE, pixels: torch.Tensor) -> np.ndarray:
    """The cache members of windows' frames: the VAE's normalised latents, as float16.

    Args:
        vae (Wan22VAE): the VAE (:func:`worldcast.modeling.wan22.vae.load_wan22_vae`).
        pixels (torch.Tensor): ``[b, 3, 161, 384, 672]`` float32 in [-1, 1].

    Returns:
        np.ndarray: ``[b, 1, 41, 48, 24, 42]`` float16.
    """
    weight = next(vae.parameters())
    with torch.inference_mode():
        # pixels in the VAE's dtype and its default scale, the VAE's own latent normalisation
        latents = vae.encode(pixels.to(device=weight.device, dtype=weight.dtype))
    return latents.permute(0, 2, 1, 3, 4).float().cpu().numpy().astype(np.float16)[:, None]


def cache_media(
    vae: Wan22VAE,
    record: MediaRecord,
    starts: list[int],
    *,
    dataset_root: str | Path,
    out_dir: str | Path,
    batch_size: int = 1,
) -> None:
    """Encode the windows ``starts`` of a recording and add them to its archive.

    The archive ``<out_dir>/<media_id>.npz`` is written whole to a temporary file and renamed into
    place; its other members are kept.

    Args:
        vae (Wan22VAE): the VAE.
        record (MediaRecord): the recording, with a ``video_path`` under ``dataset_root``.
        starts (list[int]): the windows' first source frames.
        dataset_root (str | Path): the recordings.
        out_dir (str | Path): the latent cache.
        batch_size (int): windows per VAE call.

    Raises:
        RuntimeError: the video cannot be decoded, or does not have the index's frame count.
    """
    video = Path(dataset_root) / record.video_path
    members = {}
    for first in range(0, len(starts), batch_size):
        chunk = starts[first : first + batch_size]
        pixels = torch.stack(
            [
                load_video_window(
                    video,
                    start,
                    video_frames=WINDOW_VIDEO_FRAMES,
                    source_frames=record.source_frames,
                )
                for start in chunk
            ]
        )
        for start, latents in zip(chunk, encode_windows(vae, pixels)):
            members[window_key(start)] = latents
    _add_members(Path(out_dir) / f"{record.media_id}.npz", members)


def _add_members(archive: Path, new: dict[str, np.ndarray]) -> None:
    members = {}
    if archive.is_file():
        with np.load(archive) as old:
            members = {name: old[name] for name in old.files}
    members.update(new)
    part = archive.with_name(archive.name + ".part")
    try:
        with part.open("wb") as handle:
            np.savez(handle, **dict(sorted(members.items())))
            handle.flush()
            os.fsync(handle.fileno())
        part.replace(archive)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _load_vae(wan22_root: str, device: torch.device) -> Wan22VAE:
    dtype = CACHE_DTYPE if device.type == "cuda" else torch.float32
    return load_wan22_vae(Path(wan22_root) / VAE_CHECKPOINT_NAME, device=device, dtype=dtype)


def _positive(text: str) -> int:
    if not text.isdecimal() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return int(text)


def _shard(text: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected I/N, got {text!r}") from None
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError(f"expected 0 <= I < N, got {text!r}")
    return index, count


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this step's options to its subcommand's parser."""
    add_config_args(parser)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--windows",
        nargs="+",
        metavar="JSONL",
        help="the windows: JSONL files with a media_id and a start_frame per line"
        " (default: the bucket files of data.bucket_dir)",
    )
    source.add_argument(
        "--stride",
        type=_positive,
        metavar="N",
        help="the windows: starting at every N-th source frame of every video, as far as a whole"
        " window fits",
    )
    parser.add_argument(
        "--media-id", nargs="+", action="extend", metavar="ID", help="only these recordings"
    )
    parser.add_argument(
        "--out-dir", metavar="DIR", help="where the archives go (default: data.latent_cache_root)"
    )
    parser.add_argument(
        "--shard",
        type=_shard,
        default=(0, 1),
        metavar="I/N",
        help="this process encodes the I-th of N parts of the recordings (I from 0)",
    )
    parser.add_argument(
        "--batch-size", type=_positive, default=1, metavar="N", help="windows per VAE call"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device, e.g. cuda or cuda:1 (cpu encodes in float32)",
    )


def _config(args: argparse.Namespace) -> TrainConfig:
    def from_dict(data: dict) -> TrainConfig:
        # the step reads paths only: any stage fills the rest; a file or --set may name its own
        return TrainConfig.from_dict(merge({"run": {"stage": "2"}}, data))

    return load(args.config, parse_overrides(args.overrides), from_dict)


def _windows(args: argparse.Namespace, cfg: TrainConfig, index: MediaIndex) -> Windows:
    if args.stride:
        windows = tile_windows(index.by_id.values(), args.stride)
    else:
        if args.windows is None:
            require_set(cfg.data, "data", ["bucket_dir"])
        windows = read_windows(args.windows or bucket_files(cfg.data.bucket_dir))
    if args.media_id:
        windows = select_media(windows, args.media_id)
    check_windows(windows, index)
    return windows


def _pending(windows: Windows, out_dir: Path, shard: int, shards: int) -> Windows:
    """The windows of the shard's recordings that their archives do not hold yet."""
    pending: Windows = {}
    for media_id in list(windows)[shard::shards]:
        have = set(cached_window_starts(out_dir, media_id).tolist())
        missing = [start for start in windows[media_id] if start not in have]
        if missing:
            pending[media_id] = missing
    return pending


def run(args: argparse.Namespace) -> int:
    """Run this step; returns the exit status: 1 if a recording could not be decoded."""
    cfg = _config(args)
    require_set(cfg.model, "model", ["wan22_root"])
    require_set(cfg.data, "data", ["media_index", "dataset_root"])
    if args.out_dir is None:
        require_set(cfg.data, "data", ["latent_cache_root"])
    out_dir = Path(args.out_dir or cfg.data.latent_cache_root)
    index = MediaIndex.load(cfg.data.media_index)
    windows = _windows(args, cfg, index)
    shard, shards = args.shard
    prefix = f"[shard {shard}/{shards}]"
    todo = _pending(windows, out_dir, shard, shards)
    count = sum(len(starts) for starts in todo.values())
    print(f"{prefix} {count} windows of {len(todo)} recordings to encode -> {out_dir}", flush=True)
    if not todo:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    vae = _load_vae(cfg.model.wan22_root, torch.device(args.device))
    failed = []
    for position, (media_id, starts) in enumerate(todo.items(), start=1):
        began = time.perf_counter()
        try:
            cache_media(
                vae,
                index.by_id[media_id],
                starts,
                dataset_root=cfg.data.dataset_root,
                out_dir=out_dir,
                batch_size=args.batch_size,
            )
        except RuntimeError as error:
            failed.append(media_id)
            print(f"{prefix} {position}/{len(todo)} {media_id}: not encoded: {error}", flush=True)
            continue
        seconds = (time.perf_counter() - began) / len(starts)
        print(
            f"{prefix} {position}/{len(todo)} {media_id}: {len(starts)} windows"
            f" ({seconds:.1f} s per window)",
            flush=True,
        )
    print(f"{prefix} done: {len(todo) - len(failed)} of {len(todo)} recordings", flush=True)
    if failed:
        print(f"{prefix} not encoded: {' '.join(failed)}", file=sys.stderr, flush=True)
    return 1 if failed else 0
