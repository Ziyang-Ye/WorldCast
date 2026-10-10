"""The labels of a recording: the GT visibility labels and the observer signals."""

import numpy as np
import pytest

from tests.data.support import write_observer_labels
from worldcast.data.labels import (
    OBSERVER_SIGNAL_KEYS,
    FlashCurve,
    ScopeCurve,
    load_flash_curve,
    load_observer_signal_curves,
    load_observer_signals,
    load_scope_curve,
    observer_signal_rows,
    scope_label_path,
    visibility_rows,
)

MEDIA = "1-de_x-r01-p03"


def test_visibility_rows_are_the_labels_at_the_windows_source_frames(tmp_path):
    visible = np.zeros((10, 6), bool)
    valid = np.zeros((10, 6), bool)
    visible[2] = [1, 1, 0, 0, 1, 1]  # player 2 as the recording's player sees it
    valid[2] = [1, 0, 1, 1, 1, 1]  # ... and where that label is defined
    np.savez(tmp_path / f"{MEDIA}.npz", _binary_visible=visible, _binary_eval_valid=valid)
    got_visible, got_valid = visibility_rows(tmp_path, MEDIA, np.array([-2, 0, 1, 2, 4, 5, 6]))
    assert got_visible.shape == (10, 7) and got_visible.dtype == np.float32
    assert got_valid.dtype == np.bool_
    # outside the recording a frame is neither visible nor valid; visible only where valid
    assert got_visible[2].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    assert got_valid[2].tolist() == [False, True, False, True, True, True, False]
    assert not got_visible[[0, 1, 3]].any() and not got_valid[[0, 1, 3]].any()
    with pytest.raises(FileNotFoundError, match="no visibility labels for absent"):
        visibility_rows(tmp_path, "absent", np.arange(3))
    np.savez(tmp_path / "nine.npz", _binary_visible=visible[:9], _binary_eval_valid=valid[:9])
    with pytest.raises(ValueError, match="hold 9 players, expected 10"):
        visibility_rows(tmp_path, "nine", np.arange(3))


def _curves(first_frame: int = 0, stride: int | None = 1) -> tuple[FlashCurve, ScopeCurve]:
    """Flash-white at source frame 6 only; scoped at source frames 8 (level 2) and 16 (level 5)."""
    lum = np.full(40, 0.5, np.float16)
    lum[6] = 0.9
    scoped, level = np.zeros(40, np.uint8), np.zeros(40, np.int8)
    scoped[[8, 16]], level[[8, 16]] = 1, [2, 5]
    return (
        FlashCurve(lum, hot_threshold=0.85, stride=stride, first_frame=first_frame),
        ScopeCurve(scoped, level, stride=stride, first_frame=first_frame),
    )


def test_flash_is_any_of_a_latent_frames_video_frames_and_scope_its_last():
    signals = observer_signal_rows(*_curves(), start_frame=0, latent_frames=4)
    assert tuple(signals) == OBSERVER_SIGNAL_KEYS
    assert all(value.dtype == np.int64 and value.shape == (4,) for value in signals.values())
    # latent frame 1 = video frames 1 .. 4 = source frames 2, 4, 6, 8; its last one is 8
    assert signals["obs_flash_flag"].tolist() == [0, 1, 0, 0]
    assert signals["obs_scope_on"].tolist() == [0, 1, 1, 0]
    assert signals["obs_scope_level"].tolist() == [0, 2, 2, 0]  # levels above 2 read as 2
    assert signals["obs_flash_valid"].tolist() == signals["obs_scope_valid"].tolist() == [1] * 4
    # from source frame 6 on, latent frame 0 is the flashed frame itself
    late = observer_signal_rows(*_curves(), start_frame=6, latent_frames=2)
    assert late["obs_flash_flag"].tolist() == [1, 0] and late["obs_scope_on"].tolist() == [0, 0]


def test_a_latent_frame_the_curves_cannot_be_sampled_at_is_not_valid():
    # the window's source frames are even; curves sampled at the odd ones cannot serve them
    off_lattice = observer_signal_rows(*_curves(first_frame=1, stride=2), 0, 4)
    unknown = observer_signal_rows(*_curves(stride=None), 0, 4)
    for signals in (off_lattice, unknown):
        assert not any(value.any() for value in signals.values())
    # a window that runs past the curves' 40 samples: latent frame 5 ends at source frame 40
    past = observer_signal_rows(*_curves(), start_frame=0, latent_frames=6)
    assert past["obs_flash_valid"].tolist() == past["obs_scope_valid"].tolist() == [1] * 5 + [0]


def test_the_label_files_of_a_recording(tmp_path):
    lum = np.full(40, 0.5)
    lum[6] = 0.9
    scoped, level = np.zeros(40), np.zeros(40)
    scoped[8], level[8] = 1, 1
    write_observer_labels(tmp_path, MEDIA, 40, lum=lum, scoped_vis=scoped, level=level)
    flash, scope = load_observer_signal_curves(tmp_path, MEDIA, source_frames=40)
    assert (flash.stride, flash.first_frame, scope.stride, scope.first_frame) == (1, 0, 1, 0)
    assert flash.hot_threshold == pytest.approx(0.85) and flash.lum.dtype == np.float16
    signals = load_observer_signals(
        tmp_path, MEDIA, source_frames=40, start_frame=0, latent_frames=3
    )
    assert signals["obs_flash_flag"].tolist() == [0, 1, 0]
    assert signals["obs_scope_level"].tolist() == [0, 1, 0]
    # a scope file without a stride is aligned only when it holds one sample per source frame
    assert load_scope_curve(scope_label_path(tmp_path, MEDIA), source_frames=41).stride is None
    other = load_observer_signals(tmp_path, MEDIA, source_frames=41, start_frame=0, latent_frames=3)
    assert other["obs_scope_valid"].tolist() == [0, 0, 0] and other["obs_flash_valid"].all()
    np.savez(tmp_path / "bad.npz", lum=np.zeros(4, np.float16), hot_threshold=0.85, stride=0)
    with pytest.raises(ValueError, match="label stride must be positive"):
        load_flash_curve(tmp_path / "bad.npz")


def test_a_missing_label_file_is_an_error_unless_asked(tmp_path):
    window = dict(source_frames=40, start_frame=0, latent_frames=3)
    with pytest.raises(FileNotFoundError, match="flash label file missing"):
        load_observer_signals(tmp_path, MEDIA, **window)
    unknown = load_observer_signals(tmp_path, MEDIA, **window, missing_ok=True)
    assert tuple(unknown) == OBSERVER_SIGNAL_KEYS
    assert not any(value.any() for value in unknown.values())
