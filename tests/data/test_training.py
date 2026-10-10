"""The training windows: the bucket index and its weights, the bucket windows over the latent
cache and the raw-video windows of stage 1."""

import hashlib
import json

import numpy as np
import pytest
import torch

from tests.data.support import RECORDING_LEVELS
from tests.data.synthetic_round import CLIENT, TEAMMATE
from worldcast.data import training as D
from worldcast.data.controls import CONTROL_BUTTONS
from worldcast.data.memory_selection import BLOCK_CAUSAL, MemoryFrameSource
from worldcast.data.window import WindowSpec


# ------------------------------------------------------------------------------------ bucket index
def _row(media: int, start: int, pairs: int | None, **extra) -> dict:
    row = {"media_id": f"m{media}", "start_frame": start, "match_id": media, "round": 1, **extra}
    return row if pairs is None else {**row, "vis_available": True, "vis_pair_count": pairs}


def _write_buckets(root, rows: dict[str, list[dict]]) -> None:
    for name in D.BUCKET_NAMES:
        lines = "".join(json.dumps(row) + "\n" for row in rows.get(name, []))
        (root / f"train_{name}.jsonl").write_text(lines)


def test_a_windows_weight_is_its_buckets_times_the_density_of_its_visible_pairs(tmp_path):
    assert D.BUCKET_NAMES == ("q00-01", "q01-03", "q03-05", "q05-07", "q07-10")
    _write_buckets(
        tmp_path,
        {
            "q00-01": [_row(1, 0, 3, latent_key="win_000000")],
            "q03-05": [_row(2, 8, 0), _row(2, 16, 8)],
        },
    )
    windows = D.read_bucket_index(tmp_path)
    assert [(w.media_id, w.start_frame, w.match_id, w.round) for w in windows] == [
        ("m1", 0, 1, 1),
        ("m2", 8, 2, 1),
        ("m2", 16, 2, 1),
    ]
    # bucket weight x (1 + pairs) ** 0.5; the window without pairs holds 18 % of the total
    assert [w.weight for w in windows] == pytest.approx([0.687417402 * 2.0, 1.0, 3.0])


def test_the_bucket_weights_of_a_config_set_the_draw_shares_of_the_files(tmp_path):
    _write_buckets(
        tmp_path,
        {
            "q00-01": [_row(1, 0, 3)],
            "q01-03": [_row(2, 0, 0)],
            "q03-05": [_row(3, 0, 8)],
        },
    )
    papers = D.read_bucket_index(tmp_path)
    explicit = D.read_bucket_index(tmp_path, bucket_weights=D.BUCKET_WEIGHTS)
    assert [w.weight for w in explicit] == [w.weight for w in papers]

    # by hand: file weight x sqrt(1 + pairs) = 2 x 2 = 4, 0.5 x 1 and 1 x 3; the window without
    # pairs holds 0.5 / 7.5 = 6.7 % < 15 %, so it is boosted to 15 % of the total: its weight
    # becomes 0.15 / 0.85 x 7 = 1.235294 and the total 8.235294
    windows = D.read_bucket_index(tmp_path, bucket_weights=[2.0, 0.5, 1.0, 1.0, 1.0])
    total = sum(w.weight for w in windows)
    shares = [w.weight / total for w in windows]
    assert shares == pytest.approx([0.85 * 4 / 7, 0.15, 0.85 * 3 / 7])
    # the paper's weights: 0.687417402 x 2, 0.86527334 x 1 and 1 x 3; the window without pairs
    # holds 0.86527334 / 5.240108144 = 16.5 % already, so there is no boost
    default = [w.weight / sum(v.weight for v in papers) for w in papers]
    assert default == pytest.approx(
        [1.374834804 / 5.240108144, 0.86527334 / 5.240108144, 3 / 5.240108144]
    )

    with pytest.raises(ValueError, match="argument 2 is shorter"):
        D.read_bucket_index(tmp_path, bucket_weights=[1.0, 1.0, 1.0])


def test_the_windows_without_visible_pairs_hold_at_least_15_percent():
    weights, counts = [1.0, 33.0, 33.0, 33.0], [0, 5, 9, 17]
    # one factor on the zero-pair weights: 0.15 / 0.85 x 99 / 1
    assert D.apply_zero_pair_floor(weights, counts) == pytest.approx([17.4705882, 33.0, 33.0, 33.0])
    assert D.apply_zero_pair_floor([30.0, 70.0], [0, 4]) == [30.0, 70.0]  # never lowered
    assert D.apply_zero_pair_floor([1.0, 2.0], [3, 4]) == [1.0, 2.0]


def test_the_bucket_index_drops_the_held_out_media(tmp_path):
    _write_buckets(tmp_path, {"q01-03": [_row(1, 0, 1), _row(2, 0, 1)], "q07-10": [_row(3, 0, 1)]})
    held_out = tmp_path / "held_out.txt"
    held_out.write_text("m2\n\n")
    exclusion = D.read_media_exclusion(held_out)
    assert exclusion == frozenset({"m2"})
    assert [w.media_id for w in D.read_bucket_index(tmp_path, exclusion)] == ["m1", "m3"]
    with pytest.raises(ValueError, match="1 excluded media are not in the buckets"):
        D.read_bucket_index(tmp_path, frozenset({"m9"}))
    held_out.write_text("m2\nm2\n")
    with pytest.raises(ValueError, match="distinct media ids"):
        D.read_media_exclusion(held_out)


@pytest.mark.parametrize(
    "row, message",
    [
        (_row(1, 0, None), "no visibility labels"),
        (_row(1, 0, None, vis_available=False, vis_pair_count=3), "no visibility labels"),
        (_row(1, 0, None, vis_available=True), "invalid vis_pair_count None"),
        (_row(1, 0, -1), "invalid vis_pair_count -1"),
        (_row(1, 0, True), "invalid vis_pair_count True"),
        (_row(1, 8, 1, latent_key="win_000000"), "latent_key must be win_<start_frame>"),
    ],
)
def test_a_bucket_row_is_refused_never_defaulted(tmp_path, row, message):
    _write_buckets(tmp_path, {"q05-07": [row]})
    with pytest.raises(ValueError, match=message):
        D.read_bucket_index(tmp_path)


def test_the_bucket_index_refuses_a_repeated_window_and_an_empty_index(tmp_path):
    _write_buckets(tmp_path, {"q01-03": [_row(1, 0, 1)], "q05-07": [_row(1, 0, 2)]})
    with pytest.raises(ValueError, match=r"train_q05-07.jsonl:1 repeats window m1@0"):
        D.read_bucket_index(tmp_path)
    _write_buckets(tmp_path, {})
    with pytest.raises(ValueError, match="no training windows"):
        D.read_bucket_index(tmp_path)


# ---------------------------------------------------------------------- bucket windows (stages 2-4)
def test_a_bucket_window_is_the_clients_window_with_its_cached_latents(recorded_round):
    windows = [D.BucketWindow(CLIENT, 0, 1, 1, 2.0), D.BucketWindow(TEAMMATE, 0, 1, 1, 3.0)]
    dataset = D.BucketWindows(
        windows,
        media_index=recorded_round.media_index,
        paths=recorded_round.paths,
        spec=WindowSpec(41),
    )
    assert len(dataset) == 2 and dataset.sample_weights == [2.0, 3.0]
    item = dataset[1]
    assert item["metadata"] == {"media_id": TEAMMATE, "start_frame": 0, "dataset_index": 1}
    assert item["latents"].shape == (41, 48, 24, 42) and item["latents"].dtype == torch.float32
    with np.load(recorded_round.paths.latent_cache_root / f"{TEAMMATE}.npz") as cache:
        assert np.array_equal(item["latents"].numpy(), cache["win_000000"][0].astype(np.float32))
    assert item["client_slot"].item() == 1 and item["player_states"].shape == (10, 161, 6)
    assert not any(key.startswith(("window_", "memory_frames_")) for key in item)
    with pytest.raises(ValueError, match="disagrees with its round"):
        D.BucketWindows(
            [D.BucketWindow(CLIENT, 0, 1, 2, 1.0)],
            media_index=recorded_round.media_index,
            paths=recorded_round.paths,
            spec=WindowSpec(41),
        )


def test_with_memory_frames_a_served_window_has_an_accepted_target_block(recorded_round):
    source = MemoryFrameSource(
        dataset_root=recorded_round.paths.dataset_root,
        latent_cache_root=recorded_round.paths.latent_cache_root,
        observer_signal_label_root=recorded_round.paths.observer_signal_label_root,
        meshes=recorded_round.meshes,
        spec=WindowSpec(41),
        config=BLOCK_CAUSAL,
    )
    windows = [D.BucketWindow(TEAMMATE, 0, 1, 1, 1.0), D.BucketWindow(CLIENT, 0, 1, 1, 1.0)]
    dataset = D.BucketWindows(
        windows,
        media_index=recorded_round.media_index,
        paths=recorded_round.paths,
        spec=WindowSpec(41),
        memory_frames=source,
    )
    item = dataset.load_window(1)
    assert int(item["window_target_start"]) == 29 and item["window_memory_mask"].sum() == 132
    assert item["memory_frames_latents"].shape == (4, 48, 24, 42)
    assert item["window_c2w"].shape == (41, 4, 4) and item["window_tans"].shape == (41, 2)
    # the teammate's own window has no acceptable target (its only teammate looks at seen walls):
    # the loader serves the next window it can, here index (0 + 7919) % 2
    with pytest.raises(D.WindowRefused, match="no acceptable target block"):
        dataset.load_window(0)
    assert dataset[0]["metadata"]["dataset_index"] == 1


# ----------------------------------------------------------------------------- raw video (stage 1)
N_TICKS = 33
TICKS_BYTES = b"the bytes of the tick file"


def _manifest_record(recording, **changes) -> dict:
    """The manifest record of the ``recording`` fixture: two windows of five video frames."""
    (recording / "ticks.parquet").write_bytes(TICKS_BYTES)
    record = {
        "media_id": "m0",
        "match_id": 1,
        "map_name": "de_test",
        "round": 1,
        "player_slot": 0,
        "video_path": "m0.mp4",
        "ticks_path": "ticks.parquet",
        "video_frames": 12,
        "video_file_size": (recording / "m0.mp4").stat().st_size,
        "ticks_file_size": len(TICKS_BYTES),
        "ticks_sha256": hashlib.sha256(TICKS_BYTES).hexdigest(),
        "ticks_rows": N_TICKS,
        "fps": 32.0,
        "pixel_frames": 5,
        "stride": 2,
        "max_tick_gap_seconds": 1.5 / 64.0,
        "start_frames": [0, 2],
        "n_windows": 2,
    }
    return {**record, **changes}


def _raw_video_windows(tmp_path, recording, monkeypatch, *records) -> D.RawVideoWindows:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    # the tick parquet's columns: ``attack`` at tick 6 only, the weapon changing at tick 7
    columns = {
        "t": [tick / 64.0 for tick in range(N_TICKS)],
        "active": [["attack"] if tick == 6 else [] for tick in range(N_TICKS)],
        "delta_pitch": [0.0] * N_TICKS,
        "delta_yaw": [0.5] * N_TICKS,
        "input_weapon": ["weapon_ak47"] * 7 + ["weapon_awp"] * (N_TICKS - 7),
    }
    monkeypatch.setattr(D, "read_tick_columns", lambda path, names, rows: columns)
    spec = WindowSpec(
        2, max_tick_gap_seconds=D.RAW_VIDEO_MAX_TICK_GAP_SECONDS, camera_encoding="clip"
    )
    return D.RawVideoWindows(manifest, spec=spec, dataset_root=recording)


def test_a_stage1_window_is_the_video_and_the_controls_from_its_start_frame(
    tmp_path, recording, monkeypatch
):
    # a manifest as docs/training.md describes it: without skip_frame
    dataset = _raw_video_windows(tmp_path, recording, monkeypatch, _manifest_record(recording))
    assert len(dataset) == 2 and dataset.sample_weights is None
    item = dataset[1]  # the window from source frame 2: source frames 2, 4, 6, 8, 10
    assert item["metadata"] == {"media_id": "m0", "start_frame": 2, "dataset_index": 1}
    assert item["frames"].shape == (3, 5, 384, 672) and item["frames"].dtype == torch.float32
    levels = (item["frames"].mean(dim=(0, 2, 3)) / 2.0 + 0.5) * 255.0
    assert levels.tolist() == pytest.approx([RECORDING_LEVELS[k] for k in (2, 4, 6, 8, 10)], abs=4)
    # tick 6 is at source frame 3: the video frame at source frame 4 owns it
    attack = CONTROL_BUTTONS.index("attack")
    assert item["buttons"][:, attack].tolist() == [0, 1, 0, 0, 0]
    assert dataset[0]["buttons"][:, attack].tolist() == [0, 0, 1, 0, 0]
    assert item["view_deltas"][:, 1].tolist() == pytest.approx([0.175] * 5)
    assert item["weapon"].tolist() == [2, 4, 4, 4, 4]  # ak47, then awp


def test_a_manifest_built_for_another_sampling_is_refused_by_the_field(
    tmp_path, recording, monkeypatch
):
    def windows(**changes):
        record = _manifest_record(recording, **changes)
        return _raw_video_windows(tmp_path, recording, monkeypatch, record)

    assert len(windows(skip_frame=2)) == 2
    with pytest.raises(
        ValueError, match="built with skip_frame = 1, the stage reads windows with 2"
    ):
        windows(skip_frame=1)
    with pytest.raises(ValueError, match="built with pixel_frames = 81"):
        windows(pixel_frames=81)
    with pytest.raises(ValueError, match="built with max_tick_gap_seconds = 0.1"):
        windows(max_tick_gap_seconds=0.1)
    with pytest.raises(ValueError, match="has invalid start_frames"):
        windows(start_frames=[0, 4], n_windows=2)  # source frames 4 .. 12 run past the recording
    record = _manifest_record(recording)
    del record["ticks_sha256"]
    with pytest.raises(ValueError, match=r"misses \['ticks_sha256'\]"):
        _raw_video_windows(tmp_path, recording, monkeypatch, record)


def test_a_manifests_weights_are_per_record_or_per_window(tmp_path, recording, monkeypatch):
    first = _manifest_record(recording, sample_weight=2.0)
    second = _manifest_record(recording, media_id="m1", window_sample_weights=[1.0, 3.0])
    dataset = _raw_video_windows(tmp_path, recording, monkeypatch, first, second)
    assert dataset.sample_weights == [2.0, 2.0, 1.0, 3.0]
    with pytest.raises(
        ValueError, match="give every record sample_weight or window_sample_weights"
    ):
        _raw_video_windows(tmp_path, recording, monkeypatch, first, _manifest_record(recording))


def test_a_changed_recording_is_refused(tmp_path, recording, monkeypatch):
    record = _manifest_record(recording)
    dataset = _raw_video_windows(tmp_path, recording, monkeypatch, record)
    (recording / "ticks.parquet").write_bytes(TICKS_BYTES[::-1])  # the same size, other bytes
    with pytest.raises(RuntimeError, match="does not match the media index sha256"):
        dataset[0]
    (recording / "ticks.parquet").write_bytes(TICKS_BYTES)
    changed = _manifest_record(recording, video_file_size=record["video_file_size"] + 1)
    with pytest.raises(RuntimeError, match="missing or changed since the manifest was built"):
        _raw_video_windows(tmp_path, recording, monkeypatch, changed)[0]
