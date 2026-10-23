"""The detection pass of the entity criterion: the fine-tuned YOLO11x on the recorded and the
generated videos of a view, and the recording's confirmation of each player box.

A video is decoded with decord (RGB; at most the 161 frames of a ten-second window) and handed to
the detector 16 frames per call as BGR arrays, the channel order ultralytics assumes for numpy
frames. The detector letterboxes them to 1280 px and keeps persons above confidence 0.05 (App.
"Evaluation protocols and long horizons in detail", Data), with ultralytics' other defaults (NMS IoU
0.7, at most 300 detections, float32). Ultralytics cuts an NMS that exceeds its time limit short and
only logs it; such a call is repeated, at most six calls per batch. The recorded video is the VAE
reconstruction of the recording, encoded like the generated videos.

A player box (:mod:`~worldcast.engine.evaluation.boxes`) is confirmed in a video when a detection
overlaps the box clipped to the frame with an IoU of at least 0.30, or covers at least 0.55 of the
smaller of the two; gaps of at most two frames between confirmed frames are closed.

The dump of a set of views (``dets.json.gz``; :func:`detect_window`, :func:`write_detections`)
holds per view ``media_id`` (``<media>@<start_frame>``), ``map_name``, ``start_frame``,
``obs_slot``, ``width``, ``height``, ``n_frames``, ``n_slots``; ``boxes``, the box list of the
recorded players; ``projected`` and ``confirmed``, ``[n_slots][n_frames]`` 0/1, which recorded
players have a box and which of them the recorded video confirms; and per video ``dets_<arm>``,
``{frame: [[x0, y0, x1, y1, conf], ...]}``, and ``confirmed_<arm>``, the name ``real`` being the
recorded video's. Detections are kept for the frames in which the observer is known and alive, and
only for those. The criterion at a higher confidence (Table 12) filters a dump
(:func:`at_confidence`) instead of running the detector again.
"""

import gzip
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldcast.data.latents import SOURCE_FRAMES_PER_VIDEO_FRAME, WINDOW_VIDEO_FRAMES
from worldcast.data.recordings import ALIVE_INDEX
from worldcast.hub import check_metric_weight

from .boxes import clip_box, frame_states, view_boxes

__all__ = [
    "CONFIRM_GAP",
    "CONFIRM_IOMIN",
    "CONFIRM_IOU",
    "DETECT_BATCH",
    "DETECT_CONF",
    "DETECT_IMGSZ",
    "NMS_ATTEMPTS",
    "NMS_RETRY_PAUSE",
    "NMSTruncationError",
    "VideoDetections",
    "at_confidence",
    "confirm_recording",
    "detect_video",
    "detect_window",
    "detections_of",
    "load_detector",
    "read_detections",
    "write_detections",
]

#: Confidence threshold of the detector.
DETECT_CONF = 0.05
#: Detector input size, px: the frames are letterboxed to it.
DETECT_IMGSZ = 1280
#: Frames per detector call.
DETECT_BATCH = 16
#: Detector calls per batch while ultralytics reports a truncated NMS.
NMS_ATTEMPTS = 6
#: Pause before a batch is run again after a truncated NMS, s.
NMS_RETRY_PAUSE = 3.0
#: Confirmation: IoU of a detection with the box clipped to the frame.
CONFIRM_IOU = 0.30
#: Confirmation: intersection over the smaller of the two areas.
CONFIRM_IOMIN = 0.55
#: Confirmation: gaps of at most this many frames between confirmed frames are closed.
CONFIRM_GAP = 2


class NMSTruncationError(RuntimeError):
    """The detector truncated its NMS on every call of a batch."""


class _NmsWatch(logging.Handler):
    """Notes ultralytics' warning that an NMS hit its time limit (the call returns normally)."""

    def __init__(self) -> None:
        super().__init__()
        self.hit = False

    def emit(self, record: logging.LogRecord) -> None:
        if "NMS time limit" in record.getMessage():
            self.hit = True


_NMS_WATCH = _NmsWatch()


@dataclass(frozen=True)
class VideoDetections:
    """The detections of one video.

    Attributes:
        width (int): frame width, px.
        height (int): frame height, px.
        frames (list[list[list[float]]]): per decoded frame (at most the
            :data:`~worldcast.data.latents.WINDOW_VIDEO_FRAMES` of a ten-second window), the
            detections ``[x0, y0, x1, y1, conf]`` in pixels of the frame.
    """

    width: int
    height: int
    frames: list[list[list[float]]]


def load_detector(path: str | Path, device: torch.device | str) -> Any:
    """The fine-tuned YOLO11x (ultralytics) from its weights file (checked against its sha256 in
    :data:`~worldcast.hub.METRIC_WEIGHTS`), on ``device``."""
    check_metric_weight("detector", path)
    from ultralytics import YOLO

    model = YOLO(str(path))
    model.to(device)
    return model


def _predict(model: Any, frames: list[np.ndarray]) -> list:
    logger = logging.getLogger("ultralytics")
    if _NMS_WATCH not in logger.handlers:
        logger.addHandler(_NMS_WATCH)
    for attempt in range(NMS_ATTEMPTS):
        _NMS_WATCH.hit = False
        results = model.predict(
            frames, conf=DETECT_CONF, imgsz=DETECT_IMGSZ, classes=[0], verbose=False
        )
        if not _NMS_WATCH.hit:
            return results
        if attempt + 1 < NMS_ATTEMPTS:
            time.sleep(NMS_RETRY_PAUSE)
    raise NMSTruncationError(f"the detector truncated its NMS on {NMS_ATTEMPTS} calls of a batch")


def _detections(result: Any) -> list[list[float]]:
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    return [xyxy[k].tolist() + [float(conf[k])] for k in range(len(xyxy))]


def detect_video(model: Any, path: str | Path) -> VideoDetections:
    """Detect the persons in every frame of a video (at most the
    :data:`~worldcast.data.latents.WINDOW_VIDEO_FRAMES` of a ten-second window).

    Args:
        model (Any): the detector (:func:`load_detector`).
        path (str | Path): the mp4.

    Returns:
        VideoDetections: the frame size and the detections per frame.
    """
    import decord

    reader = decord.VideoReader(str(path), num_threads=2)
    count = min(len(reader), WINDOW_VIDEO_FRAMES)
    if count == 0:
        raise ValueError(f"{path}: no frames")
    frames: list[list[list[float]]] = []
    for lo in range(0, count, DETECT_BATCH):
        pixels = reader.get_batch(list(range(lo, min(count, lo + DETECT_BATCH)))).asnumpy()
        bgr = [np.ascontiguousarray(frame[..., ::-1]) for frame in pixels]
        frames.extend(_detections(result) for result in _predict(model, bgr))
    return VideoDetections(width=int(pixels.shape[2]), height=int(pixels.shape[1]), frames=frames)


def _matches(box: Sequence[float], det: Sequence[float], width: int, height: int) -> bool:
    clipped = clip_box(box, width, height)
    if clipped is None:
        return False
    ix = max(0.0, min(clipped[2], det[2]) - max(clipped[0], det[0]))
    iy = max(0.0, min(clipped[3], det[3]) - max(clipped[1], det[1]))
    inter = ix * iy
    if inter <= 0:
        return False
    area_box = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
    area_det = (det[2] - det[0]) * (det[3] - det[1])
    if inter / (area_box + area_det - inter) >= CONFIRM_IOU:
        return True
    return inter / max(1e-6, min(area_box, area_det)) >= CONFIRM_IOMIN


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    for row in out:
        on = np.flatnonzero(row)
        for lo, hi in zip(on[:-1], on[1:]):
            if hi - lo - 1 <= max_gap:
                row[lo + 1 : hi] = True
    return out


def confirm_recording(
    boxes: Sequence[Sequence[Sequence[float]]],
    detections: Sequence[Sequence[Sequence[float]]],
    n_slots: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Which player boxes a video confirms.

    Args:
        boxes (Sequence): the box list of the view
            (:func:`~worldcast.engine.evaluation.boxes.view_boxes`), ``T`` frames.
        detections (Sequence): per frame (at least ``T``) the video's detections ``[x0, y0, x1,
            y1, ...]``.
        n_slots (int): rows of the result (every slot of a box is below it).
        width (int): frame width, px.
        height (int): frame height, px.

    Returns:
        np.ndarray: ``[n_slots, T]`` bool: a detection matches the box, with gaps of at most
        :data:`CONFIRM_GAP` frames closed, and the player has a box.
    """
    projected = np.zeros((int(n_slots), len(boxes)), dtype=bool)
    confirmed = np.zeros_like(projected)
    for i, frame in enumerate(boxes):
        for box in frame:
            slot = int(box[0])
            projected[slot, i] = True
            confirmed[slot, i] = any(_matches(box[1:5], d, width, height) for d in detections[i])
    return _close_gaps(confirmed, CONFIRM_GAP) & projected


def detect_window(
    model: Any,
    recording: str | Path,
    generated: Mapping[str, str | Path],
    ticks: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    media_id: str,
    map_name: str,
    start_frame: int,
    observer_slot: int,
) -> dict[str, Any]:
    """Detect one view in its recorded and generated videos: its entry of the dump.

    Args:
        model (Any): the detector (:func:`load_detector`).
        recording (str | Path): the recorded video of the view (VAE reconstruction).
        generated (Mapping[str, str | Path]): arm -> generated video of the view.
        ticks (Mapping[int, tuple[np.ndarray, np.ndarray]]): slot -> the player's tick times
            ``[n]`` (s) and states ``[n, 6]``, every player of the round.
        media_id (str): the observer's recording.
        map_name (str): the map.
        start_frame (int): the view's first source frame.
        observer_slot (int): the observer's slot.

    Returns:
        dict[str, Any]: the view's entry (module docstring).
    """
    if "real" in generated:
        raise ValueError("'real' names the recorded video, not a generated arm")
    others = [int(slot) for slot in ticks if int(slot) != int(observer_slot)]
    if int(observer_slot) not in ticks or not others:
        raise ValueError(
            f"{media_id}: the ticks hold no observer {observer_slot} or no other player"
        )
    videos = {"real": detect_video(model, recording)}
    count, width, height = len(videos["real"].frames), videos["real"].width, videos["real"].height
    for name, path in generated.items():
        video = detect_video(model, path)
        if (len(video.frames), video.width, video.height) != (count, width, height):
            raise ValueError(f"{media_id}: the video of {name} differs from the recording in size")
        videos[name] = video
    source = int(start_frame) + SOURCE_FRAMES_PER_VIDEO_FRAME * np.arange(count, dtype=np.float64)
    states = {int(slot): frame_states(times, rows, source) for slot, (times, rows) in ticks.items()}
    observer = states[int(observer_slot)]
    boxes = view_boxes(observer, states, int(observer_slot), width, height)
    n_slots = max(others) + 1
    projected = np.zeros((n_slots, count), dtype=np.int8)
    for i, frame in enumerate(boxes):
        for box in frame:
            projected[int(box[0]), i] = 1
    kept = [i for i in range(count) if observer[i, ALIVE_INDEX] >= 0.5]
    confirmed = {
        name: confirm_recording(boxes, video.frames, n_slots, width, height)
        for name, video in videos.items()
    }
    window: dict[str, Any] = {
        "media_id": f"{media_id}@{int(start_frame)}",
        "map_name": map_name,
        "start_frame": int(start_frame),
        "obs_slot": int(observer_slot),
        "width": width,
        "height": height,
        "n_frames": count,
        "n_slots": n_slots,
        "boxes": boxes,
        "projected": projected.tolist(),
        "confirmed": confirmed["real"].astype(np.int8).tolist(),
    }
    for name, video in videos.items():
        window[f"dets_{name}"] = {str(i): video.frames[i] for i in kept}
        window[f"confirmed_{name}"] = confirmed[name].astype(np.int8).tolist()
    return window


def detections_of(window: Mapping[str, Any], arm: str) -> list[list[list[float]]]:
    """The detections of one video of a dumped view per frame (none where none are kept)."""
    dets = window[f"dets_{arm}"]
    return [dets.get(str(i), []) for i in range(int(window["n_frames"]))]


def at_confidence(window: Mapping[str, Any], conf: float) -> dict[str, Any]:
    """A dumped view at a higher confidence threshold: every video's detections below ``conf``
    dropped and its confirmation of the recorded players recomputed from the rest.

    Args:
        window (Mapping[str, Any]): a view of the dump (:func:`read_detections`).
        conf (float): confidence threshold; a detection at ``conf`` is kept.

    Returns:
        dict[str, Any]: the view at ``conf`` (a new dict; ``window`` is not changed).
    """
    out = dict(window)
    for key in [k for k in window if k.startswith("dets_")]:
        arm = key[len("dets_") :]
        out[key] = {f: [d for d in dets if d[4] >= conf] for f, dets in window[key].items()}
        confirmed = confirm_recording(
            window["boxes"],
            detections_of(out, arm),
            window["n_slots"],
            window["width"],
            window["height"],
        )
        out[f"confirmed_{arm}"] = confirmed.astype(np.int8).tolist()
    out["confirmed"] = out["confirmed_real"]
    return out


def write_detections(
    path: str | Path, windows: Sequence[Mapping[str, Any]], stamp: Mapping[str, Any]
) -> None:
    """Write the dump: the views sorted by ``media_id``, and ``stamp`` (detector, settings)."""
    payload = {"windows": sorted(windows, key=lambda w: w["media_id"]), "stamp": dict(stamp)}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def read_detections(path: str | Path) -> dict[str, Any]:
    """The dump at ``path``: ``windows`` and ``stamp``."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)
