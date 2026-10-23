"""The latent-cache step: the windows it encodes, what it writes, resuming and sharding."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from tests.tools.support import load_tool
from worldcast.config.training import WINDOW_LATENT_FRAMES
from worldcast.data.latent_cache import (
    cached_window_starts,
    load_block_latents,
    load_first_latent,
    load_window_latents,
)
from worldcast.data.latents import WINDOW_SOURCE_FRAMES, WINDOW_VIDEO_FRAMES, window_key
from worldcast.data.preparation import latent_cache
from worldcast.data.preparation.latent_cache import (
    bucket_files,
    check_windows,
    read_windows,
    tile_windows,
)
from worldcast.data.recordings import MediaIndex, MediaRecord

#: Source frames of the ``video`` fixture's recording and the grey level of each.
FRAMES = 340
LEVELS = [10 + 20 * (i % 12) for i in range(FRAMES)]
#: Bytes of a window's member: its float16 values and the 128-byte header of the ``.npy``.
MEMBER_BYTES = 41 * 48 * 24 * 42 * 2 + 128


class StubVAE(torch.nn.Module):
    """The Wan2.2 VAE's call signature and shapes without its weights: ``encode`` takes pixels
    ``[b, 3, 1 + 4 k, H, W]`` and returns ``[b, 48, 1 + k, 24, 42]``, channel ``c`` of a latent
    frame being ``c + 1`` times the mean of its video frames (the first alone, then four) over each
    cell of the 24 x 42 grid. It keeps what it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.loads: list[tuple] = []
        self.shapes: list[tuple[int, ...]] = []
        self.dtypes: list[torch.dtype] = []
        self.frame_levels: list[torch.Tensor] = []
        self.value_range: list[tuple[float, float]] = []

    def encode(self, x: torch.Tensor, scale=None) -> torch.Tensor:
        b, _, t, _, _ = x.shape
        self.shapes.append(tuple(x.shape))
        self.dtypes.append(x.dtype)
        self.frame_levels.append(x.float().mean(dim=(1, 3, 4)))
        self.value_range.append((float(x.min()), float(x.max())))
        cells = F.adaptive_avg_pool2d(x.float().mean(dim=1), (24, 42))
        blocks = cells[:, 1:].reshape(b, (t - 1) // 4, 4, 24, 42).mean(dim=2)
        latents = torch.cat([cells[:, :1], blocks], dim=1)[:, None]
        return (latents * torch.arange(1, 49).view(1, 48, 1, 1, 1)).to(x.dtype)


def media_row(media_id: str, frames: int) -> dict:
    return {
        "media_id": media_id,
        "match_id": 1,
        "map_name": "de_test",
        "round": 1,
        "player_slot": 0,
        "fps": 32.0,
        "video_frames": frames,
        "ticks_path": f"{media_id}.parquet",
        "ticks_rows": 1,
        "ticks_file_size": 1,
        "video_path": f"{media_id}.mp4",
    }


def record(media_id: str, frames: int = FRAMES) -> MediaRecord:
    return MediaRecord.from_row(media_row(media_id, frames))


def make_world(tmp_path: Path, media: dict[str, int], buckets: dict[str, list[int]] | None = None):
    """The config of a tiny dataset: a media index of ``media`` (id -> source frames) and bucket
    files that hold the windows ``buckets`` (id -> start frames; the last file is empty).

    Returns:
        Path: the config file.
    """
    index = tmp_path / "media_index.jsonl"
    index.write_text("".join(json.dumps(media_row(m, n)) + "\n" for m, n in sorted(media.items())))
    bucket_dir = tmp_path / "buckets"
    bucket_dir.mkdir()
    rows: list[list[dict]] = [[] for _ in range(5)]
    for media_id, starts in (buckets or {}).items():
        for i, start in enumerate(starts):
            rows[i % 4].append({"media_id": media_id, "start_frame": start, "round": 1})
    for name, lines in zip(("q00-01", "q01-03", "q03-05", "q05-07", "q07-10"), rows):
        (bucket_dir / f"train_{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in lines)
        )
    config = tmp_path / "paths.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"wan22_root": str(tmp_path / "wan22")},
                "data": {
                    "dataset_root": str(tmp_path / "videos"),
                    "media_index": str(index),
                    "bucket_dir": str(bucket_dir),
                    "latent_cache_root": str(tmp_path / "cache"),
                },
            }
        )
    )
    return config


def prepare(*argv) -> int:
    return load_tool("prepare_data").main(["latent-cache", *map(str, argv)])


def members_of(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


@pytest.fixture
def stub(monkeypatch):
    """The stub VAE in place of the Wan2.2 one; ``stub.loads`` lists the loads."""
    vae = StubVAE()
    monkeypatch.setattr(latent_cache, "_load_vae", lambda *a: vae.loads.append(a) or vae)
    return vae


@pytest.fixture
def decoded(monkeypatch):
    """A decoder that serves a flat frame of ``start / 1000`` per window on a small grid and
    refuses ``bad.mp4``; the list holds its calls ``(video, start, video_frames, source_frames)``.
    """
    calls = []

    def decode(video_path, start_frame, *, video_frames, source_frames):
        calls.append((Path(video_path), start_frame, video_frames, source_frames))
        if Path(video_path).name == "bad.mp4":
            raise RuntimeError(f"{video_path}: cannot decode")
        return torch.full((3, video_frames, 24, 42), start_frame / 1000)

    monkeypatch.setattr(latent_cache, "load_video_window", decode)
    return calls


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """The directory of ``m0.mp4``: ``FRAMES`` frames of 48 x 84, frame ``i`` a flat grey of
    ``LEVELS[i]``."""
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio

    root = tmp_path_factory.mktemp("recordings")
    writer = imageio.get_writer(
        str(root / "m0.mp4"), fps=32, codec="libx264", quality=9, macro_block_size=1
    )
    for level in LEVELS:
        writer.append_data(np.full((48, 84, 3), level, np.uint8))
    writer.close()
    return root


# ------------------------------------------------------------------------------ what is encoded
def test_a_window_is_every_second_frame_from_its_start_resized_and_scaled(video, stub, tmp_path):
    starts = [0, 3]
    latent_cache.cache_media(stub, record("m0"), starts, dataset_root=video, out_dir=tmp_path)
    assert stub.shapes == [(1, 3, 161, 384, 672)] * 2 and stub.dtypes == [torch.float32] * 2
    members = members_of(tmp_path / "m0.npz")
    for window, start in enumerate(starts):
        levels = (stub.frame_levels[window][0] / 2 + 0.5) * 255
        source = [LEVELS[start + 2 * k] for k in range(WINDOW_VIDEO_FRAMES)]
        assert levels.tolist() == pytest.approx(source, abs=5)
        assert -1.0 <= stub.value_range[window][0] and stub.value_range[window][1] <= 1.0
        scaled = [(level / 255 - 0.5) * 2 for level in source]
        for f in (0, 1, 7, 40):
            frames = [0] if f == 0 else range(4 * f - 3, 4 * f + 1)
            expected = sum(scaled[i] for i in frames) / len(frames)
            latents = members[window_key(start)][0, f]
            assert latents[0, 0, 0] == pytest.approx(expected, abs=0.05)
            assert latents[47, 5, 9] == pytest.approx(48 * expected, abs=2.4)


def test_the_windows_asked_for_are_the_windows_decoded(stub, decoded, tmp_path):
    config = make_world(tmp_path, {"a": 800, "b": 700}, {"a": [322, 0, 161], "b": [80]})
    assert prepare("--config", config) == 0
    videos = tmp_path / "videos"
    assert decoded == [
        (videos / "a.mp4", 0, 161, 800),
        (videos / "a.mp4", 161, 161, 800),
        (videos / "a.mp4", 322, 161, 800),
        (videos / "b.mp4", 80, 161, 700),
    ]


def test_the_vae_loads_in_bf16_on_cuda_and_float32_elsewhere(monkeypatch):
    seen = []
    monkeypatch.setattr(
        latent_cache, "load_wan22_vae", lambda path, *, device, dtype: seen.append((path, dtype))
    )
    latent_cache._load_vae("weights/wan22", torch.device("cuda"))
    latent_cache._load_vae("weights/wan22", torch.device("cpu"))
    vae = Path("weights/wan22/Wan2.2_VAE.pth")
    assert seen == [(vae, torch.bfloat16), (vae, torch.float32)]


def test_the_real_vae_is_called_as_the_stub_is(tiny_vae):
    torch.manual_seed(0)
    pixels = torch.rand(1, 3, WINDOW_VIDEO_FRAMES, 384, 672) * 2 - 1
    encode, latents = tiny_vae.encode, []
    tiny_vae.encode = lambda x, scale=None: latents.append(encode(x, scale)) or latents[-1]
    members = latent_cache.encode_windows(tiny_vae, pixels)
    assert members.shape == (1, 1, WINDOW_LATENT_FRAMES, 48, 24, 42) and members.dtype == np.float16
    assert latents[0].shape == (1, 48, WINDOW_LATENT_FRAMES, 24, 42)
    assert np.array_equal(
        members[0, 0], latents[0][0].permute(1, 0, 2, 3).numpy().astype(np.float16)
    )


# ---------------------------------------------------------------------------------- what is written
def test_members_are_named_by_start_and_hold_float16_windows(stub, decoded, tmp_path):
    config = make_world(tmp_path, {"a": 800}, {"a": [161, 0]})
    assert prepare("--config", config, "--device", "cpu") == 0
    path = tmp_path / "cache" / "a.npz"
    members = members_of(path)
    assert list(members) == ["win_000000", "win_000161"]
    for start in (0, 161):
        window = members[window_key(start)]
        assert window.shape == (1, 41, 48, 24, 42) and window.dtype == np.float16
        channels = (np.arange(1, 49) * start / 1000).reshape(1, 1, 48, 1, 1)
        assert np.allclose(window, np.broadcast_to(channels, window.shape), rtol=2e-3)
    assert 2 * MEMBER_BYTES <= path.stat().st_size < 2 * MEMBER_BYTES + 1000
    assert [p.name for p in path.parent.iterdir()] == ["a.npz"]


def test_what_is_written_reads_back_through_the_latent_cache(stub, decoded, tmp_path):
    config = make_world(tmp_path, {"a": 800}, {"a": [0, 161]})
    assert prepare("--config", config) == 0
    root = tmp_path / "cache"
    assert cached_window_starts(root, "a").tolist() == [0, 161]
    window = load_window_latents(root, "a", 161)
    assert window.shape == (41, 48, 24, 42) and window.dtype == torch.float32
    assert torch.equal(
        window, torch.from_numpy(members_of(root / "a.npz")["win_000161"][0]).float()
    )
    assert torch.equal(load_first_latent(root, "a", 161), window[:1])
    assert torch.equal(load_block_latents(root, "a", 161, 5), window[5:9])


def test_the_batch_size_does_not_change_the_cache(stub, decoded, tmp_path):
    config = make_world(tmp_path, {"a": 1000}, {"a": [0, 161, 322, 483, 644]})
    assert prepare("--config", config, "--out-dir", tmp_path / "one") == 0
    assert prepare("--config", config, "--out-dir", tmp_path / "three", "--batch-size", 3) == 0
    assert [shape[0] for shape in stub.shapes] == [1] * 5 + [3, 2]
    one, three = members_of(tmp_path / "one" / "a.npz"), members_of(tmp_path / "three" / "a.npz")
    assert one.keys() == three.keys() and all(np.array_equal(one[k], three[k]) for k in one)


def test_an_archive_is_written_whole_or_not_at_all(stub, decoded, tmp_path, monkeypatch):
    root = tmp_path / "cache"
    root.mkdir()
    options = dict(dataset_root=tmp_path, out_dir=root)
    latent_cache.cache_media(stub, record("a", 800), [0], **options)
    before = (root / "a.npz").read_bytes()

    def killed(handle, **arrays):
        handle.write(b"PK half an archive")
        raise OSError("no space left on device")

    with monkeypatch.context() as broken:
        broken.setattr(np, "savez", killed)
        for media_id in ("a", "new"):
            with pytest.raises(OSError, match="no space"):
                latent_cache.cache_media(stub, record(media_id, 800), [161], **options)
    assert [p.name for p in root.iterdir()] == ["a.npz"]
    assert (root / "a.npz").read_bytes() == before

    (root / "b.npz.part").write_bytes(b"left by a killed run")
    assert cached_window_starts(root, "b").tolist() == []
    latent_cache.cache_media(stub, record("b", 800), [0], **options)
    assert sorted(p.name for p in root.iterdir()) == ["a.npz", "b.npz"]
    assert cached_window_starts(root, "b").tolist() == [0]


# ---------------------------------------------------------------------------------------- resuming
def test_a_second_run_encodes_nothing_and_a_longer_one_only_what_is_new(stub, decoded, tmp_path):
    config = make_world(tmp_path, {"a": 800}, {"a": [0, 161]})
    root = tmp_path / "cache"
    assert prepare("--config", config) == 0
    first = members_of(root / "a.npz")
    stub.loads.clear()
    decoded.clear()

    assert prepare("--config", config) == 0
    assert stub.loads == [] and decoded == []

    longer = tmp_path / "longer.jsonl"
    longer.write_text(
        "".join(f'{{"media_id": "a", "start_frame": {s}}}\n' for s in (0, 80, 161, 322))
    )
    assert prepare("--config", config, "--windows", longer) == 0
    assert [call[1] for call in decoded] == [80, 322]
    members = members_of(root / "a.npz")
    assert list(members) == ["win_000000", "win_000080", "win_000161", "win_000322"]
    assert all(np.array_equal(members[key], first[key]) for key in first)


def test_a_recording_that_cannot_be_decoded_is_reported_and_the_rest_go_on(
    stub, decoded, tmp_path, capsys
):
    media = {"a": 800, "bad": 800, "c": 800}
    config = make_world(tmp_path, media, {m: [0] for m in media})
    assert prepare("--config", config) == 1
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["a.npz", "c.npz"]
    assert "not encoded: bad" in capsys.readouterr().err
    assert prepare("--config", config) == 1 and len(decoded) == 4


# ---------------------------------------------------------------------------------------- sharding
def test_the_shards_take_every_window_once(stub, decoded, tmp_path):
    media = {f"m{i}": 1000 for i in range(7)}
    windows = {m: [0, 161, 322, 483][: 1 + i % 4] for i, m in enumerate(media)}
    config = make_world(tmp_path, media, windows)
    taken = []
    for shard in range(3):
        decoded.clear()
        assert prepare("--config", config, "--shard", f"{shard}/3") == 0
        taken.append([(call[0].stem, call[1]) for call in decoded])
    everything = sorted((m, s) for m, starts in windows.items() for s in starts)
    assert sorted(w for part in taken for w in part) == everything
    assert [sorted({m for m, _ in part}) for part in taken] == [
        ["m0", "m3", "m6"],
        ["m1", "m4"],
        ["m2", "m5"],
    ]
    for media_id, starts in windows.items():
        assert cached_window_starts(tmp_path / "cache", media_id).tolist() == starts


# ----------------------------------------------------------------------------------- the windows
def test_every_video_is_tiled_by_the_stride(stub, decoded, tmp_path):
    media = {"long": 700, "just": WINDOW_SOURCE_FRAMES, "short": WINDOW_SOURCE_FRAMES - 1}
    records = [record(m, n) for m, n in media.items()]
    assert tile_windows(records, 161) == {"just": [0], "long": [0, 161, 322]}
    assert tile_windows(records, 80)["long"] == [0, 80, 160, 240, 320]
    config = make_world(tmp_path, media)
    assert prepare("--config", config, "--stride", 161, "--media-id", "long") == 0
    assert cached_window_starts(tmp_path / "cache", "long").tolist() == [0, 161, 322]
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["long.npz"]


def test_the_bucket_files_hold_each_window_once(tmp_path):
    config = make_world(tmp_path, {"a": 800, "b": 800}, {"a": [161, 0, 322, 483, 644], "b": [80]})
    bucket_dir = Path(yaml.safe_load(config.read_text())["data"]["bucket_dir"])
    with (bucket_dir / "train_q07-10.jsonl").open("a") as last:
        last.write('{"media_id": "a", "start_frame": 161, "latent_key": "win_000161"}\n')
    assert read_windows(bucket_files(bucket_dir)) == {"a": [0, 161, 322, 483, 644], "b": [80]}


@pytest.mark.parametrize(
    "window, message",
    [
        ('{"media_id": "ghost", "start_frame": 0}', "not in the media index"),
        ('{"media_id": "a", "start_frame": 480}', "the window at 480 is not inside"),
        ('{"media_id": "a", "start_frame": -1}', "the window at -1 is not inside"),
        ('{"media_id": "a"}', "a window has a media_id and a start_frame"),
        ("[1, 2]", "a window has a media_id and a start_frame"),
    ],
)
def test_windows_that_cannot_be_encoded_are_refused_before_any_work(
    stub, tmp_path, capsys, window, message
):
    config = make_world(tmp_path, {"a": 800})
    path = tmp_path / "windows.jsonl"
    path.write_text(window + "\n")
    with pytest.raises(SystemExit) as refused:
        prepare("--config", config, "--windows", path)
    assert refused.value.code == 2 and message in capsys.readouterr().err
    assert stub.loads == [] and not (tmp_path / "cache").exists()


def test_a_recording_without_a_video_is_refused():
    index = MediaIndex({"a": dataclasses.replace(record("a"), video_path=None)})
    with pytest.raises(ValueError, match="a has no video_path"):
        check_windows({"a": [0]}, index)


# --------------------------------------------------------------------------------- config, options
def test_the_paths_come_from_the_config_and_no_stage_is_needed(stub, decoded, tmp_path, capsys):
    config = make_world(tmp_path, {"a": 800}, {"a": [0]})
    assert prepare("--config", config, "--set", f"data.latent_cache_root={tmp_path / 'set'}") == 0
    assert (tmp_path / "set" / "a.npz").is_file()
    assert prepare("--config", config, "--out-dir", tmp_path / "out") == 0
    assert (tmp_path / "out" / "a.npz").is_file() and not (tmp_path / "cache").exists()
    assert stub.loads[0] == (str(tmp_path / "wan22"), torch.device("cuda"))
    assert f"-> {tmp_path / 'out'}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--set", "model.wan22_root=null"], "model.wan22_root"),
        (["--set", "data.media_index=null"], "data.media_index"),
        (["--set", "data.bucket_dir=null"], "data.bucket_dir"),
        (["--set", "data.latent_cache_root=null"], "data.latent_cache_root"),
        (["--set", "data.nonsense=1"], "unknown config keys"),
        (["--shard", "3/3"], "expected 0 <= I < N"),
        (["--shard", "one"], "expected I/N"),
        (["--stride", "0"], "expected a positive integer"),
        (["--stride", "many"], "expected a positive integer"),
        (["--batch-size", "0"], "expected a positive integer"),
        (["--stride", "161", "--windows", "x.jsonl"], "not allowed with argument"),
        (["--media-id", "ghost"], "no windows of ghost"),
    ],
)
def test_a_step_that_cannot_run_says_why(stub, tmp_path, capsys, argv, message):
    config = make_world(tmp_path, {"a": 800}, {"a": [0]})
    with pytest.raises(SystemExit) as refused:
        prepare("--config", config, *argv)
    assert refused.value.code == 2 and message in capsys.readouterr().err
    assert stub.loads == []
