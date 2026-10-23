"""The detection pass with a stand-in detector and decord: the weights it takes, channel order,
batches, the NMS guard, the recording's confirmation, the kept frames and the dump."""

import gzip
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from worldcast.data.latents import WINDOW_VIDEO_FRAMES
from worldcast.engine.evaluation import detector
from worldcast.engine.evaluation.boxes import player_box
from worldcast.engine.evaluation.detector import (
    DETECT_BATCH,
    DETECT_IMGSZ,
    NMS_ATTEMPTS,
    NMSTruncationError,
    at_confidence,
    confirm_recording,
    detect_video,
    detect_window,
    detections_of,
    read_detections,
    write_detections,
)
from worldcast.hub import METRIC_WEIGHTS

WIDTH, HEIGHT = 64, 36
#: Three views of the paper's ten-second windows with their stored detections.
FIXTURE = Path(__file__).parent / "fixtures" / "entity" / "views.json.gz"


class _Boxes:
    def __init__(self, rows: list[list[float]]) -> None:
        data = torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)
        self.xyxy, self.conf = data[:, :4], data[:, 4]

    def __len__(self) -> int:
        return len(self.conf)


class StubDetector:
    """Detects one person per frame whose box encodes the frame's first pixel as it arrives: the
    blue value of an RGB frame is its first channel in BGR order. Logs a truncated NMS on its first
    ``truncated`` calls."""

    def __init__(self, truncated: int = 0) -> None:
        self.calls: list[dict] = []
        self.truncated = truncated

    def predict(self, frames, conf, imgsz, classes, verbose):
        self.calls.append(dict(frames=len(frames), conf=conf, imgsz=imgsz, classes=classes))
        assert not verbose and all(frame.flags["C_CONTIGUOUS"] for frame in frames)
        if len(self.calls) <= self.truncated:
            logging.getLogger("ultralytics").warning("NMS time limit 2.800s exceeded")
        out = []
        for frame in frames:
            first = float(frame[0, 0, 0])
            rows = [] if first == 0 else [[first, 0.0, first + 10.0, 20.0, 0.5]]
            out.append(types.SimpleNamespace(boxes=_Boxes(rows)))
        return out


def rgb_frames(count: int) -> np.ndarray:
    """``[count, H, W, 3]`` frames: frame ``i`` is ``(R, G, B) = (100, 101, i % 50)``."""
    frames = np.zeros((count, HEIGHT, WIDTH, 3), dtype=np.uint8)
    frames[..., 0], frames[..., 1] = 100, 101
    frames[..., 2] = (np.arange(count) % 50)[:, None, None]
    return frames


@pytest.fixture
def videos(monkeypatch) -> dict[str, np.ndarray]:
    """A stand-in ``decord`` reading the videos registered in the returned dict by path."""
    store: dict[str, np.ndarray] = {}

    class VideoReader:
        def __init__(self, path: str, num_threads: int) -> None:
            self.frames = store[path]

        def __len__(self) -> int:
            return len(self.frames)

        def get_batch(self, index: list[int]):
            return types.SimpleNamespace(asnumpy=lambda: self.frames[index])

    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=VideoReader))
    monkeypatch.setattr(detector.time, "sleep", lambda seconds: None)
    return store


def test_the_detector_takes_only_the_papers_weights(tmp_path):
    path = tmp_path / "yolo11x_cs2_player.pt"
    path.write_bytes(b"not the detector")
    with pytest.raises(ValueError, match=METRIC_WEIGHTS["detector"].sha256):
        detector.load_detector(path, "cpu")


def test_frames_reach_the_detector_as_bgr_in_batches(videos):
    videos["a.mp4"] = rgb_frames(WINDOW_VIDEO_FRAMES + 9)
    model = StubDetector()
    result = detect_video(model, "a.mp4")
    assert (result.width, result.height, len(result.frames)) == (WIDTH, HEIGHT, WINDOW_VIDEO_FRAMES)
    assert [call["frames"] for call in model.calls] == [DETECT_BATCH] * 10 + [1]
    assert model.calls[0] == dict(frames=DETECT_BATCH, conf=0.05, imgsz=DETECT_IMGSZ, classes=[0])
    assert result.frames[0] == []  # blue 0: no person
    assert result.frames[7] == [[7.0, 0.0, 17.0, 20.0, 0.5]]  # blue, not red (100)


def test_a_truncated_nms_runs_the_batch_again(videos):
    videos["a.mp4"] = rgb_frames(20)
    model = StubDetector(truncated=2)
    detect_video(model, "a.mp4")
    assert [call["frames"] for call in model.calls] == [DETECT_BATCH, DETECT_BATCH, DETECT_BATCH, 4]
    with pytest.raises(NMSTruncationError):
        detect_video(StubDetector(truncated=NMS_ATTEMPTS), "a.mp4")
    model = StubDetector(truncated=NMS_ATTEMPTS - 1)
    detect_video(model, "a.mp4")
    assert len(model.calls) == NMS_ATTEMPTS + 1


def test_confirmation_by_iou_or_the_smaller_area():
    box = [[[0, 10.0, 0.0, 50.0, 100.0, 1.0]]]  # 40 x 100

    def confirmed(det: list[float]) -> bool:
        return bool(confirm_recording(box, [[det]], 1, 640, 360)[0, 0])

    assert confirmed([30.0, 0.0, 70.0, 100.0])  # IoU 1/3, intersection over the smaller area 1/2
    assert not confirmed([35.0, 0.0, 75.0, 100.0])  # 0.23 and 0.38
    assert confirmed([44.5, 10.0, 54.5, 50.0])  # 0.55 of the 10 x 40 detection
    assert not confirmed([45.0, 10.0, 55.0, 50.0])  # 0.50
    outside = [[[0, 700.0, 0.0, 740.0, 100.0, 1.0]]]
    assert not confirm_recording(outside, [[[700.0, 0.0, 740.0, 100.0]]], 1, 640, 360)[0, 0]


def test_gaps_of_at_most_two_frames_are_closed_where_the_player_has_a_box():
    pattern = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    boxed = [i != 10 for i in range(len(pattern))]  # no box in frame 10
    box = [0, 10.0, 0.0, 50.0, 100.0, 1.0]
    boxes = [[box] if boxed[i] else [] for i in range(len(pattern))]
    dets = [[box[1:5]] if hit else [] for hit in pattern]
    closed = confirm_recording(boxes, dets, 2, 640, 360)
    assert closed[0].astype(int).tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1]
    assert not closed[1].any()


def test_the_paper_views_confirm_as_dumped():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)
    windows = [item["window"] for item in fixture["gt_views"]]
    for window in windows + [fixture["predicted_view"]["window"]]:
        for key in [k for k in window if k.startswith("dets_")]:
            arm = key[len("dets_") :]
            confirmed = confirm_recording(
                window["boxes"],
                detections_of(window, arm),
                window["n_slots"],
                window["width"],
                window["height"],
            )
            assert confirmed.astype(int).tolist() == window[f"confirmed_{arm}"], arm
        assert window["confirmed"] == window["confirmed_real"]


def write_ticks(slot_states: dict[int, list[np.ndarray]]) -> dict:
    """Per slot: one tick per source frame of a window from source frame 100."""
    return {
        slot: (np.arange(100, 100 + len(rows)) / 32.0, np.stack(rows))
        for slot, rows in slot_states.items()
    }


def test_a_window_keeps_the_frames_whose_observer_is_known_and_alive(videos):
    count = 20
    videos["rec.mp4"] = rgb_frames(count)
    videos["gen.mp4"] = rgb_frames(count)[:, :, :, ::-1].copy()  # a different video
    alive = [1.0] * 2 * count
    alive[2 * 3] = 0.0  # the observer is dead at video frame 3
    observer = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, a]) for a in alive[: 2 * 15]]  # ticks end
    peer = [np.array([200.0, 0.0, 0.0, 0.0, 0.0, 1.0])] * (2 * count)
    ticks = write_ticks({0: observer, 2: peer})
    window = detect_window(
        StubDetector(),
        "rec.mp4",
        {"gen": "gen.mp4"},
        ticks,
        media_id="m-p00",
        map_name="de_dust2",
        start_frame=100,
        observer_slot=0,
    )
    kept = [str(i) for i in range(15) if i != 3]  # the observer's ticks cover video frames 0-14
    assert sorted(window["dets_real"], key=int) == kept == sorted(window["dets_gen"], key=int)
    assert window["dets_real"]["7"] == [[7.0, 0.0, 17.0, 20.0, 0.5]]
    assert (window["media_id"], window["n_slots"], window["n_frames"]) == ("m-p00@100", 3, count)
    expected = list(player_box(observer[0], peer[0], WIDTH, HEIGHT))
    assert window["boxes"][0] == [[2] + expected]
    assert window["boxes"][3] == [] and window["boxes"][15] == []
    assert window["projected"][2] == [int(i in range(15) and i != 3) for i in range(count)]
    assert window["projected"][1] == [0] * count
    assert window["confirmed"] == window["confirmed_real"]
    videos["short.mp4"] = rgb_frames(count - 1)
    with pytest.raises(ValueError, match="differs"):
        detect_window(
            StubDetector(),
            "rec.mp4",
            {"gen": "short.mp4"},
            ticks,
            media_id="m-p00",
            map_name="de_dust2",
            start_frame=100,
            observer_slot=0,
        )


def dumped_view(media_id: str = "m@0") -> dict:
    box = [0, 10.0, 0.0, 50.0, 100.0, 1.0]
    return {
        "media_id": media_id,
        "n_frames": 3,
        "n_slots": 1,
        "width": 640,
        "height": 360,
        "boxes": [[box], [box], [box]],
        "projected": [[1, 1, 1]],
        "confirmed": [[1, 1, 1]],
        "dets_real": {"0": [[10.0, 0.0, 50.0, 100.0, 0.9]], "2": [[10.0, 0.0, 50.0, 100.0, 0.2]]},
        "confirmed_real": [[1, 1, 1]],
        "dets_gen": {"1": [[10.0, 0.0, 50.0, 100.0, 0.3]]},
        "confirmed_gen": [[0, 1, 0]],
    }


def test_a_dump_round_trips_sorted_by_view(tmp_path):
    path = tmp_path / "dets.json.gz"
    write_detections(path, [dumped_view("b@0"), dumped_view("a@0")], {"conf": 0.05})
    dump = read_detections(path)
    assert [w["media_id"] for w in dump["windows"]] == ["a@0", "b@0"]
    assert dump["stamp"] == {"conf": 0.05}
    assert detections_of(dump["windows"][0], "gen") == [[], [[10.0, 0.0, 50.0, 100.0, 0.3]], []]


def test_a_higher_confidence_filters_the_dump_and_confirms_again():
    view = dumped_view()
    raised = at_confidence(view, 0.25)
    assert raised["dets_real"] == {"0": [[10.0, 0.0, 50.0, 100.0, 0.9]], "2": []}
    assert raised["confirmed"] == raised["confirmed_real"] == [[1, 0, 0]]  # no gap to close
    assert raised["confirmed_gen"] == [[0, 1, 0]]
    assert at_confidence(view, 0.3)["dets_gen"] == view["dets_gen"]  # kept at the threshold
    assert view["confirmed"] == [[1, 1, 1]]  # the view itself is unchanged
