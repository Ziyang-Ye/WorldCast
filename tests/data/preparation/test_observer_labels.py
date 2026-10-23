"""``tools/prepare_data.py observer-labels``: the flash and scope labels of synthetic recordings.

The recording is a 200-frame 128 x 72 video at 32 fps: flat grey with a white-out ramp over known
frames (flash), and, where the scope is on, black with a lit centre. Its tick table presses the
scope button and switches the weapon at known frames, so every array of the label files has a
known value.
"""

import dataclasses
import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

from tests.data.support import tick_table
from tests.tools.support import load_tool
from worldcast.data.controls import AlignmentError, encode_weapon
from worldcast.data.game import TICK_RATE
from worldcast.data.labels import (
    OBSERVER_SIGNAL_KEYS,
    flash_label_path,
    load_flash_curve,
    load_observer_signals,
    load_scope_curve,
    scope_label_path,
)
from worldcast.data.preparation import observer_labels
from worldcast.data.preparation.observer_labels import (
    compare_arrays,
    flash_labels,
    scope_labels,
)
from worldcast.data.recordings import TickTable

FRAMES = 200
MEDIA = "1-de_x-r01-p00"
TICKS = 2 * FRAMES + 5  # 64 Hz ticks: a source frame owns two, and ``ncov`` reaches the last frame

#: Frames of the white-out: ``HOT`` is white; the steps of the ramp around it are not hot.
HOT = range(6, 14)
RAMP = {4: 160, 5: 210, 14: 210, 15: 160}
#: A black screen: dark corners, but no lit centre.
BLACK = range(112, 114)
#: Frames where the scope's overlay is on screen, and where the scope button is held (a press is
#: the first frame of each run).
OVERLAY = [range(21, 40), range(42, 110), range(120, 130), range(140, 170), range(176, 200)]
HELD = [range(f, f + 3) for f in (20, 60, 100, 119, 150, 175, 190, 195)]
#: The weapon in hand: a two-level scope, a rifle without one, a one-level scope.
WEAPONS = ["weapon_awp"] * 150 + ["weapon_ak47"] * 20 + ["weapon_aug"] * 30


def frames_in(ranges) -> np.ndarray:
    mask = np.zeros(FRAMES, bool)
    for frames in ranges:
        mask[list(frames)] = True
    return mask


def expected_levels() -> np.ndarray:
    """The zoom level the presses make, gated by the overlay: the awp's first press (frame 20)
    zooms once, its second (60) twice, its third (100) drops it while the overlay stays; the
    overlay's gap at 40-41 is too short to drop it; the press at 119 zooms once and the overlay's
    gap at 130-139 drops it without a press, so the overlay at 140-149 shows no level; the ak47 has
    none; the aug zooms once at 175, is off at 190 and zooms once again at 195."""
    level = np.zeros(FRAMES, np.int8)
    level[21:40], level[42:60], level[60:100], level[120:130] = 1, 1, 2, 1
    level[176:190], level[195:200] = 1, 1
    return level


def write_video(path, frames: np.ndarray) -> None:
    import imageio.v2 as imageio

    writer = imageio.get_writer(str(path), fps=32, codec="libx264", quality=9, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def recording_frames() -> np.ndarray:
    """``[FRAMES, 72, 128, 3]`` uint8: flat grey (100), the white-out ramp, the scope's overlay."""
    frames = np.full((FRAMES, 72, 128, 3), 100, np.uint8)
    for frame in HOT:
        frames[frame] = 255
    for frame in BLACK:
        frames[frame] = 0
    for frame, level in RAMP.items():
        frames[frame] = level
    for frame in np.flatnonzero(frames_in(OVERLAY)):
        frames[frame] = 0
        frames[frame, 24:48, 48:80] = 150
    return frames


def recording_ticks() -> TickTable:
    """The player's ticks: a source frame ``f`` owns the ticks ``2f - 1`` and ``2f``."""
    active = [["forward"] for _ in range(TICKS)]
    for frame in np.flatnonzero(frames_in(HELD)):
        for tick in (2 * frame - 1, 2 * frame):
            if tick >= 0:
                active[tick] = ["forward", "attack2"]
    weapons = [WEAPONS[min(-(-tick // 2), FRAMES - 1)] for tick in range(TICKS)]
    return dataclasses.replace(tick_table(TICKS / TICK_RATE), active=active, input_weapon=weapons)


def without_ticks(ticks: TickTable, drop: slice) -> TickTable:
    keep = np.ones(len(ticks), bool)
    keep[drop] = False
    columns = {}
    for field in dataclasses.fields(ticks):
        column = getattr(ticks, field.name)
        columns[field.name] = (
            [value for value, kept in zip(column, keep) if kept]
            if isinstance(column, list)
            else column[keep]
        )
    return TickTable(**columns)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """The directory of ``m0.mp4``, the recording."""
    pytest.importorskip("imageio_ffmpeg")
    root = tmp_path_factory.mktemp("dataset")
    write_video(root / "m0.mp4", recording_frames())
    return root


@pytest.fixture(scope="module")
def gray(dataset):
    return observer_labels.decode_gray(dataset / "m0.mp4")


@pytest.fixture
def media_index(tmp_path, monkeypatch):
    """The media index of the recording, its tick table served from memory (no pyarrow needed)."""
    ticks = recording_ticks()

    def read(media, dataset_root, *, jump_recall=True):
        assert media.media_id == MEDIA and jump_recall is False
        return ticks

    monkeypatch.setattr(observer_labels, "read_player_ticks", read)
    return write_index(tmp_path / "media_index.jsonl")


def write_index(path, **row):
    fields = dict(
        media_id=MEDIA,
        match_id=1,
        map_name="de_x",
        round=1,
        player_slot=0,
        fps=32.0,
        video_frames=FRAMES,
        ticks_path="m0.parquet",
        ticks_rows=TICKS,
        ticks_file_size=1,
        video_path="m0.mp4",
    )
    fields.update(row)
    path.write_text(json.dumps(fields) + "\n")
    return path


def label(*arguments) -> int:
    return load_tool("prepare_data").main(["observer-labels", *map(str, arguments)])


def paths_file(dataset, media_index, out=None) -> str:
    """A paths file like ``train_paths.yaml``, beside the media index: its ``data`` section names
    the media index, the dataset root and, if given, the label root."""
    data = {"media_index": str(media_index), "dataset_root": str(dataset)}
    if out is not None:
        data["observer_signal_label_root"] = str(out)
    path = media_index.with_name("paths.yaml")
    path.write_text(yaml.safe_dump({"data": data, "model": {"wan22_root": "elsewhere"}}))
    return str(path)


def build(dataset, media_index, out, *more) -> int:
    return label("--config", paths_file(dataset, media_index, out), *more)


def arrays_of(path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def damage(path, **changes) -> None:
    np.savez(path, **{**arrays_of(path), **changes})


# ------------------------------------------------------------------------------------ the curves
def test_flash_is_the_screens_luminance_and_its_white_out_is_hot(gray):
    arrays = flash_labels(gray)
    assert list(arrays) == ["lum", "stride", "hot_threshold", "n_hot"]
    lum, hot = arrays["lum"], arrays["hot_threshold"]
    assert lum.shape == (FRAMES,) and lum.dtype == np.float16
    assert arrays["stride"].dtype == np.int16 and arrays["stride"] == 1
    assert hot.dtype == np.float32 and hot == np.float32(0.85)
    # the grey and the ramp's steps are the frames' levels; only the white frames are hot
    levels = recording_frames()[:, 0, 0, 0] / 255.0
    plain = ~frames_in(OVERLAY)
    np.testing.assert_allclose(lum[plain], levels[plain], atol=2 / 255)
    assert np.flatnonzero(np.float32(lum) > hot).tolist() == list(HOT)
    assert arrays["n_hot"].dtype == np.int32 and arrays["n_hot"] == len(HOT)


def test_scope_overlay_is_where_the_vignette_is(gray):
    corner_max, center, scoped = observer_labels.scope_overlay(gray)
    overlay = frames_in(OVERLAY)
    assert scoped.tolist() == overlay.tolist()
    # a vignette is black in the corners around a lit centre; the grey frames are lit everywhere
    # and a black screen has no lit centre
    black = frames_in([BLACK])
    assert corner_max[overlay].max() < observer_labels.CORNER_ON
    assert corner_max[~overlay & ~black].min() > observer_labels.CORNER_ON
    assert corner_max[black].max() < observer_labels.CORNER_ON
    assert center[overlay].min() > observer_labels.CENTER_MIN > center[black].max()


def test_scope_label_has_the_buttons_weapons_and_levels_of_the_ticks(gray):
    arrays = scope_labels(gray, recording_ticks(), 32.0)
    assert list(arrays) == [
        "corner_max",
        "center",
        "scoped_vis",
        "attack2",
        "weapon_id",
        "level",
        "ncov",
        "thresholds",
    ]
    assert {key: value.dtype for key, value in arrays.items()} == {
        "corner_max": np.float16,
        "center": np.float16,
        "scoped_vis": np.uint8,
        "attack2": np.uint8,
        "weapon_id": np.int16,
        "level": np.int8,
        "ncov": np.int32,
        "thresholds": np.float32,
    }
    assert arrays["scoped_vis"].tolist() == frames_in(OVERLAY).astype(int).tolist()
    assert arrays["attack2"].tolist() == frames_in(HELD).astype(int).tolist()
    assert arrays["weapon_id"].tolist() == [encode_weapon(w) for w in WEAPONS]
    assert arrays["level"].tolist() == expected_levels().tolist()
    assert arrays["ncov"] == FRAMES
    np.testing.assert_array_equal(arrays["thresholds"], np.float32([0.04, 0.08]))


def test_frames_beyond_the_tick_table_have_no_controls(gray):
    held = tick_table(2.0)  # 128 ticks span 63 source frames, two of them left out
    held = dataclasses.replace(held, active=[["attack2"]] * len(held))
    arrays = scope_labels(gray, held, 32.0)
    assert arrays["ncov"] == 61
    assert arrays["attack2"][:61].all() and not arrays["attack2"][61:].any()
    assert (arrays["weapon_id"][:61] == encode_weapon("ak47")).all()
    assert (arrays["weapon_id"][61:] == -1).all()
    # the overlay on screen without a weapon that zooms is no level
    assert arrays["scoped_vis"][61:].any() and not arrays["level"][61:].any()
    short = scope_labels(gray, tick_table(0.1), 32.0)
    assert short["ncov"] == 0 and (short["weapon_id"] == -1).all() and not short["attack2"].any()


def test_a_frames_weapon_is_the_one_its_last_tick_names(gray):
    names = [None] * TICKS
    names[60], names[100], names[101], names[140] = "weapon_awp", "nan", " ", "weapon_ak47"
    arrays = scope_labels(gray, dataclasses.replace(recording_ticks(), input_weapon=names), 32.0)
    # frame f holds the weapon of its last tick, 2f; a tick that names none is no weapon
    expected = np.zeros(FRAMES, np.int16)
    expected[30], expected[70] = encode_weapon("awp"), encode_weapon("ak47")
    assert arrays["weapon_id"].tolist() == expected.tolist()


def test_ticks_that_cannot_be_aligned_are_an_error(gray):
    ticks = recording_ticks()
    active = list(ticks.active)
    active[100], active[101] = ["forward", "teleport"], [None]
    with pytest.raises(AlignmentError, match=r"unknown OpenCS2 buttons: \['<null>', 'teleport'\]"):
        scope_labels(gray, dataclasses.replace(ticks, active=active), 32.0)
    # the keyboard's turn binds are known buttons
    active[100], active[101] = ["turn_left"], ["turn_right"]
    scope_labels(gray, dataclasses.replace(ticks, active=active), 32.0)
    # a frame owns no tick where two are missing
    with pytest.raises(ValueError, match="no tick rows align to frame 51"):
        scope_labels(gray, without_ticks(ticks, slice(101, 103)), 32.0)


def test_compare_arrays_gives_the_largest_difference_of_each_array():
    built = {"a": np.arange(4, dtype=np.float16), "b": np.zeros(3), "c": np.array(1)}
    nudged = np.arange(4, dtype=np.float16) + np.float16([0, 0, 0.5, 0.25])
    assert compare_arrays(built, {"a": nudged, "b": np.zeros(4)}) == {
        "a": 0.5,
        "b": float("inf"),
        "c": float("inf"),
    }
    assert compare_arrays(built, built) == {"a": 0.0, "b": 0.0, "c": 0.0}


# ------------------------------------------------------------------------------------- the files
def test_the_files_are_what_the_label_readers_take(dataset, media_index, tmp_path, capsys):
    out = tmp_path / "labels"
    assert build(dataset, media_index, out) == 0
    assert "1 built" in capsys.readouterr().out
    flash, scope = flash_label_path(out, MEDIA), scope_label_path(out, MEDIA)
    assert list(arrays_of(flash)) == ["lum", "stride", "hot_threshold", "n_hot"]
    assert list(arrays_of(scope))[-2:] == ["ncov", "thresholds"]
    curve = load_flash_curve(flash)
    assert curve.stride == 1 and curve.first_frame == 0 and curve.hot_threshold == np.float32(0.85)
    assert len(curve.lum) == FRAMES
    # the scope file's layout is the recording's frames: the reader takes it only on their count
    assert load_scope_curve(scope, source_frames=FRAMES).stride == 1
    assert load_scope_curve(scope, source_frames=FRAMES - 1).stride is None

    latents = 1 + (FRAMES - 1) // 8
    signals = load_observer_signals(
        out, MEDIA, source_frames=FRAMES, start_frame=0, latent_frames=latents
    )
    assert sorted(signals) == sorted(OBSERVER_SIGNAL_KEYS)
    # latent 0 holds source frame 0, latent k the source frames 8k - 6, 8k - 4, 8k - 2 and 8k
    sources = [[0]] + [[8 * k - 6, 8 * k - 4, 8 * k - 2, 8 * k] for k in range(1, latents)]
    flag = [int(any(frame in HOT for frame in frames)) for frames in sources]
    assert signals["obs_flash_flag"].tolist() == flag and sum(flag) == 2
    last = np.arange(latents) * 8
    assert signals["obs_scope_on"].tolist() == frames_in(OVERLAY)[last].astype(int).tolist()
    assert signals["obs_scope_level"].tolist() == expected_levels()[last].tolist()
    assert signals["obs_scope_level"].max() == 2
    assert signals["obs_flash_valid"].all() and signals["obs_scope_valid"].all()


def test_a_run_is_deterministic_and_resumable(dataset, media_index, tmp_path, capsys):
    first, second = tmp_path / "first", tmp_path / "second"
    assert build(dataset, media_index, first) == 0 and build(dataset, media_index, second) == 0
    flash, scope = flash_label_path(first, MEDIA), scope_label_path(first, MEDIA)
    for path, other in (
        (flash, flash_label_path(second, MEDIA)),
        (scope, scope_label_path(second, MEDIA)),
    ):
        assert all(v == 0.0 for v in compare_arrays(arrays_of(path), arrays_of(other)).values())
    capsys.readouterr()

    stamps = [path.stat().st_mtime_ns for path in (flash, scope)]
    assert build(dataset, media_index, first) == 0
    assert "1 skipped" in capsys.readouterr().out
    assert [path.stat().st_mtime_ns for path in (flash, scope)] == stamps
    assert not list(first.rglob("*.tmp"))

    # a file that does not match is rebuilt, and only that one
    scope.write_bytes(b"not an archive")
    assert build(dataset, media_index, first) == 0
    assert "1 built" in capsys.readouterr().out
    assert flash.stat().st_mtime_ns == stamps[0]
    assert load_scope_curve(scope, source_frames=FRAMES).stride == 1
    expected = arrays_of(flash)
    for name, value in (("hot_threshold", np.float32(0.9)), ("lum", np.zeros(5, np.float16))):
        damage(flash, **{name: value})
        assert build(dataset, media_index, first) == 0
        assert "1 built" in capsys.readouterr().out
        assert all(v == 0.0 for v in compare_arrays(arrays_of(flash), expected).values())
    damage(scope, thresholds=np.float32([0.05, 0.08]))
    assert build(dataset, media_index, first) == 0
    assert "1 built" in capsys.readouterr().out
    assert arrays_of(scope)["thresholds"].tolist() == np.float32([0.04, 0.08]).tolist()


def test_a_directory_of_videos_gives_the_flash_labels_in_any_number_of_processes(
    dataset, tmp_path, capsys
):
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("b", "a", "c"):
        (videos / f"{name}.mp4").write_bytes((dataset / "m0.mp4").read_bytes())
    one, two = tmp_path / "one", tmp_path / "two"
    assert label("--videos", videos, "--out-dir", one) == 0
    assert label("--videos", videos, "--out-dir", two, "--workers", 2) == 0
    assert capsys.readouterr().out.count("3 built") == 2
    assert not (one / "scopelabels").exists()
    for name in ("a", "b", "c"):
        built, other = arrays_of(flash_label_path(one, name)), arrays_of(
            flash_label_path(two, name)
        )
        assert all(v == 0.0 for v in compare_arrays(built, other).values())
        assert np.flatnonzero(np.float32(built["lum"]) > built["hot_threshold"]).tolist() == list(
            HOT
        )
    assert label("--videos", videos, "--out-dir", one, "--media-id", "b", "a") == 0
    assert "2 skipped" in capsys.readouterr().out
    assert label("--videos", videos, "--compare", two) == 0
    report = capsys.readouterr().out
    assert "flash/lum: equal in all of 3 recordings" in report and "scope" not in report


def test_a_recording_that_disagrees_with_the_media_index_is_reported_and_gets_no_file(
    dataset, media_index, tmp_path, capsys
):
    out = tmp_path / "labels"
    (tmp_path / "broken.mp4").write_bytes(b"not a video")
    for row, message in (
        ({"fps": 16.0}, "the video runs at 32 fps, expected 16"),
        (
            {"video_frames": FRAMES - 1},
            f"the video decodes to {FRAMES} frames, the media index says {FRAMES - 1}",
        ),
        ({"video_path": None}, "the media index has no video_path"),
        ({"video_path": "absent.mp4"}, "video not found: "),
    ):
        write_index(media_index, **row)
        assert build(dataset, media_index, out) == 1
        reported = capsys.readouterr().out
        assert f"FAILED {MEDIA}: {message}" in reported and "1 failed" in reported
        assert not out.exists()
    write_index(media_index, video_path="broken.mp4")
    assert build(tmp_path, media_index, out) == 1
    assert f"FAILED {MEDIA}: cannot read the header" in capsys.readouterr().out
    assert not out.exists()


def test_the_decoder_scales_every_frame_with_the_threads_the_labels_were_made_with(
    dataset, monkeypatch
):
    commands = []
    run = subprocess.run

    def spy(command, **kwargs):
        commands.append(command)
        return run(command, **kwargs)

    monkeypatch.setattr(observer_labels.subprocess, "run", spy)
    assert observer_labels.decode_gray(dataset / "m0.mp4").shape == (FRAMES, 36, 64)
    (command,) = commands
    assert command[command.index("-threads") + 1] == "2"
    assert command[command.index("-vf") + 1] == "scale=64:36"
    assert command[command.index("-pix_fmt") + 1] == "gray"


def test_a_video_ffmpeg_cannot_read_is_a_value_error(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="cannot read the header"):
        observer_labels.video_fps(broken)
    with pytest.raises(ValueError, match="ffmpeg cannot decode"):
        observer_labels.decode_gray(broken)
    for read in (observer_labels.video_fps, observer_labels.decode_gray):
        with pytest.raises(FileNotFoundError, match="video not found"):
            read(tmp_path / "absent.mp4")


def test_a_tick_table_the_reader_refuses_fails_the_scope_label_only(
    dataset, media_index, tmp_path, monkeypatch, capsys
):
    ticks, calls = recording_ticks(), []

    def read(media, dataset_root, *, jump_recall=True):
        calls.append(media.media_id)
        if len(calls) == 1:
            raise RuntimeError(f"tick table of media {media.media_id} has 1 bytes")
        return ticks

    monkeypatch.setattr(observer_labels, "read_player_ticks", read)
    out = tmp_path / "labels"
    assert build(dataset, media_index, out) == 1
    message = f"FAILED {MEDIA}: scope: tick table of media {MEDIA} has 1 bytes"
    assert message in capsys.readouterr().out
    # the flash label stands on the video alone; the next run makes the scope label
    flash, scope = flash_label_path(out, MEDIA), scope_label_path(out, MEDIA)
    assert flash.is_file() and not scope.exists()
    stamp = flash.stat().st_mtime_ns
    assert build(dataset, media_index, out) == 0
    assert "1 built" in capsys.readouterr().out
    assert scope.is_file() and flash.stat().st_mtime_ns == stamp


def test_compare_reports_each_array_against_another_label_root(
    dataset, media_index, tmp_path, capsys
):
    out = tmp_path / "labels"
    assert build(dataset, media_index, out) == 0
    capsys.readouterr()
    args = ("--config", paths_file(dataset, media_index), "--compare", out)
    assert label(*args) == 0
    report = capsys.readouterr().out
    assert "flash/lum: equal in all of 1 recordings" in report
    assert "scope/level: equal in all of 1 recordings" in report and "1 compared" in report

    corner_max = arrays_of(scope_label_path(out, MEDIA))["corner_max"]
    damage(
        scope_label_path(out, MEDIA),
        corner_max=corner_max + np.float16(0.25),
        weapon_id=np.zeros(FRAMES - 1, np.int16),
    )
    assert label(*args) == 1
    report = capsys.readouterr().out
    assert (
        "scope/corner_max: differs in 1 of 1 recordings, largest absolute difference 0.25" in report
    )
    assert (
        "scope/weapon_id: differs in 1 of 1 recordings, largest absolute difference inf" in report
    )
    assert "flash/lum: equal in all" in report and "scope/level: equal in all" in report

    flash_label_path(out, MEDIA).unlink()
    assert label(*args) == 1
    assert f"FAILED {MEDIA}: flash: no label file" in capsys.readouterr().out


def test_the_paths_come_from_the_config_the_set_option_or_the_options(
    dataset, media_index, tmp_path, capsys
):
    # the label root of the config, or --out over it, or --set over the dataset root
    paths = paths_file(dataset, media_index, tmp_path / "from_config")
    assert label("--config", paths) == 0
    assert flash_label_path(tmp_path / "from_config", MEDIA).is_file()
    assert label("--config", paths, "--out-dir", tmp_path / "from_out") == 0
    assert flash_label_path(tmp_path / "from_out", MEDIA).is_file()
    (tmp_path / "moved").mkdir()
    (tmp_path / "moved" / "m0.mp4").symlink_to(dataset / "m0.mp4")
    arguments = ("--config", paths, "--set", f"data.dataset_root={tmp_path / 'moved'}")
    assert label(*arguments, "--out-dir", tmp_path / "from_set") == 0
    assert scope_label_path(tmp_path / "from_set", MEDIA).is_file()
    capsys.readouterr()


def test_a_config_or_an_option_that_cannot_run_is_a_usage_error(
    dataset, media_index, tmp_path, capsys
):
    out = tmp_path / "labels"
    paths = paths_file(dataset, media_index)
    videos = tmp_path / "videos"
    videos.mkdir()
    for arguments, message in (
        (("--config", paths), "not set: data.observer_signal_label_root"),
        (("--out-dir", out), "not set: data.media_index, data.dataset_root"),
        (
            ("--config", paths, "--out-dir", out, "--set", "data.bogus=1"),
            "unknown config keys in data",
        ),
        (("--config", tmp_path / "absent.yaml", "--out-dir", out), "absent.yaml"),
        (("--config", paths, "--out-dir", out, "--set", "data.media_index=absent"), "absent"),
        (
            ("--config", paths, "--out-dir", out, "--media-id", "x", "y"),
            "not among the recordings: x, y",
        ),
        (("--videos", tmp_path / "absent", "--out-dir", out), "--videos is not a directory"),
        (("--videos", videos, "--out-dir", out), "no recordings to label"),
    ):
        with pytest.raises(SystemExit) as stop:
            label(*arguments)
        assert stop.value.code == 2
        error = capsys.readouterr().err
        assert " observer-labels: error: " in error and message in error
    assert not out.exists()
    for arguments in (
        ("--config", paths, "--out-dir", out, "--compare", out),
        ("--config", paths, "--out-dir", out, "--workers", "0"),
        ("--media-index", media_index, "--out-dir", out),
    ):
        with pytest.raises(SystemExit) as stop:
            label(*arguments)
        assert stop.value.code == 2
    assert "must be at least 1, got 0" in capsys.readouterr().err


def test_the_tick_tables_reader_is_named_when_it_is_missing(dataset, tmp_path, monkeypatch):
    def read_tick_columns(*args):
        raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")

    monkeypatch.setattr(
        sys.modules["worldcast.data.recordings"], "read_tick_columns", read_tick_columns
    )
    root = tmp_path / "root"
    root.mkdir()
    (root / "m0.mp4").symlink_to(dataset / "m0.mp4")
    (root / "m0.parquet").write_bytes(b"x")  # the size the index row names
    index = write_index(tmp_path / "media_index.jsonl")
    with pytest.raises(SystemExit, match="No module named 'pyarrow': install the package"):
        build(root, index, tmp_path / "labels")
