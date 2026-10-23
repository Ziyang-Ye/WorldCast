"""The entity criterion: Rendered and Unmatched of a view, and their summaries (Sec. 4, Metrics;
App. "Evaluation protocols and long horizons in detail", The entity criterion).

A view is one player's view of a ten-second window; a whole round is split into consecutive
ten-second windows. It is scored from the detection dump
(:mod:`~worldcast.engine.evaluation.detector`) against the player state given to the generator: its
boxes (the recorded boxes with GT states, else
:func:`~worldcast.engine.evaluation.boxes.table_boxes` of the given states) and its visibility
gate (:func:`visibility_gate`, :func:`gt_visibility_gate`).

* A detection's position is its box centre; a player's is the centre of its box clipped to the
  frame, and its body width the clipped box's width.
* Rendered. The candidates of a frame are the players with a box that are visible in the gate and
  confirmed in the recorded video; they are matched one to one to the generated video's detections
  of the frame, a detection within one body width (:func:`assign_one_to_one`). A candidate is a
  scored pair if its clipped box covers at least ``24^2`` px and the recorded video's own
  detections hit it under the same rule (one to one among all confirmed players,
  :func:`recording_hits`); the size and the recorded video's own hit select after the matching.
  The rate is the hit share of the view's scored pairs. Scored on the partner alone (App.
  "Training stages and scaling": the other player of the window's pair of views), the other
  players' pairs are dropped after the matching too.
* Unmatched. A generated detection with a short side of at least 24 px is unmatched if no expected
  centre lies within ``max(w_expected, w_detection)`` of its centre (unclipped widths, not one to
  one); expected are the recorded video's detections and the boxes of the players visible in the
  gate, each with a short side of at least 24 px.
* The gate is the given visibility folded onto latent frames, as the generator receives it: a
  frame is visible if any video frame of its latent frame is, counted from the rollout's first
  frame (the window's for ten-second windows, the session's for whole rounds).
* A summary is the median over the views with at least 8 scored pairs (Rendered) or 8 counted
  detections (Unmatched); Table 16 pools the scored pairs of the frames by crowding instead.

The radius and the smallest box are settings of :func:`score_view` for the criterion's other
settings (Table 12); the detector's confidence is set on the dump
(:func:`~worldcast.engine.evaluation.detector.at_confidence`).
"""

import math
import statistics
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from worldcast.data.labels import visibility_rows
from worldcast.data.latents import SOURCE_FRAMES_PER_VIDEO_FRAME, video_frames_of

from .boxes import clip_box, clipped_side
from .detector import detections_of

__all__ = [
    "MATCH_RADIUS",
    "MIN_DETECTIONS",
    "MIN_PAIRS",
    "MIN_SIDE",
    "CrowdingBin",
    "EntitySummary",
    "ViewCounts",
    "assign_one_to_one",
    "gt_visibility_gate",
    "pool_by_crowding",
    "recording_hits",
    "score_view",
    "summarize_entity",
    "visibility_gate",
    "window_of",
]

#: A detection matches a player within this many body widths.
MATCH_RADIUS = 1.0
#: Smallest box, px: of a scored pair (``sqrt`` of the clipped area), and of a counted detection and
#: an expected box of Unmatched (short side).
MIN_SIDE = 24.0
#: A view enters the Rendered median with at least this many scored pairs.
MIN_PAIRS = 8
#: A view enters the Unmatched median with at least this many counted detections.
MIN_DETECTIONS = 8
#: Cost of a forbidden assignment: the matching first maximizes the number of allowed pairs.
_FORBIDDEN = 1.0e6


@dataclass(frozen=True)
class ViewCounts:
    """The counts of one view.

    Attributes:
        n_pairs (int): scored pairs.
        hits (int): scored pairs the generated video renders.
        n_dets (int): counted detections of the generated video.
        n_unmatched (int): counted detections that are unmatched.
        in_view (np.ndarray): ``[T]`` int, players visible in the gate with a clipped box of at
            least the smallest side squared, per frame (the crowding of the frame).
        frame_pairs (np.ndarray): ``[T]`` int, scored pairs per frame.
        frame_hits (np.ndarray): ``[T]`` int, rendered scored pairs per frame.
    """

    n_pairs: int
    hits: int
    n_dets: int
    n_unmatched: int
    in_view: np.ndarray
    frame_pairs: np.ndarray
    frame_hits: np.ndarray

    @property
    def rendered(self) -> float | None:
        """The view's Rendered, ``None`` below :data:`MIN_PAIRS` scored pairs."""
        return self.hits / self.n_pairs if self.n_pairs >= MIN_PAIRS else None

    @property
    def unmatched(self) -> float | None:
        """The view's Unmatched, ``None`` below :data:`MIN_DETECTIONS` counted detections."""
        return self.n_unmatched / self.n_dets if self.n_dets >= MIN_DETECTIONS else None


@dataclass(frozen=True)
class EntitySummary:
    """Rendered and Unmatched of a set of views: medians and the number of views each takes.

    Attributes:
        rendered (float): median Rendered (``nan`` without a view).
        rendered_views (int): views with at least :data:`MIN_PAIRS` scored pairs.
        unmatched (float): median Unmatched (``nan`` without a view).
        unmatched_views (int): views with at least :data:`MIN_DETECTIONS` counted detections.
    """

    rendered: float
    rendered_views: int
    unmatched: float
    unmatched_views: int


@dataclass(frozen=True)
class CrowdingBin:
    """The frames with one number of players in view, pooled.

    Attributes:
        frames (int): frames.
        pairs (int): their scored pairs.
        hits (int): their rendered scored pairs.
    """

    frames: int
    pairs: int
    hits: int

    @property
    def rendered(self) -> float:
        """Rendered of the pooled pairs (``nan`` without a pair)."""
        return self.hits / self.pairs if self.pairs else math.nan


def assign_one_to_one(cost: np.ndarray, radius: float) -> list[tuple[int, int]]:
    """Match players to detections one to one within ``radius``: as many pairs as possible, then
    the least total cost.

    Args:
        cost (np.ndarray): ``[players, detections]`` distance in body widths (``inf``: no match).
        radius (float): largest allowed cost.

    Returns:
        list[tuple[int, int]]: the matched ``(player, detection)`` index pairs.
    """
    if cost.size == 0:
        return []
    allowed = np.isfinite(cost) & (cost <= float(radius))
    if not allowed.any():
        return []
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(np.where(allowed, cost, _FORBIDDEN))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if allowed[r, c]]


def _cost(
    boxes: list[list[float]], dets: list[Sequence[float]], width: int, height: int
) -> np.ndarray:
    cost = np.full((len(boxes), len(dets)), np.inf)
    if not boxes or not dets:
        return cost
    centres = np.array([(0.5 * (d[0] + d[2]), 0.5 * (d[1] + d[3])) for d in dets], dtype=np.float64)
    for e, box in enumerate(boxes):
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        cx, cy = 0.5 * (clipped[0] + clipped[2]), 0.5 * (clipped[1] + clipped[3])
        cost[e] = np.hypot(centres[:, 0] - cx, centres[:, 1] - cy) / (clipped[2] - clipped[0])
    return cost


def _hits(
    boxes: Sequence[Sequence[Sequence[float]]],
    candidates: np.ndarray,
    dets: Sequence[Sequence[Sequence[float]]],
    width: int,
    height: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame, the candidates matched one to one to the detections: ``hit``, the clipped side
    and ``inside`` (a candidate), each ``[slots, T]``."""
    hit = np.zeros(candidates.shape, dtype=bool)
    side = np.zeros(candidates.shape)
    inside = np.zeros(candidates.shape, dtype=bool)
    for i in range(candidates.shape[1]):
        chosen = [
            (int(b[0]), [float(v) for v in b[1:5]])
            for b in boxes[i]
            if 0 <= int(b[0]) < candidates.shape[0] and candidates[int(b[0]), i]
        ]
        if not chosen:
            continue
        for slot, box in chosen:
            inside[slot, i] = True
            side[slot, i] = clipped_side(box, width, height)
        cost = _cost([box for _, box in chosen], [d[:4] for d in dets[i]], width, height)
        for e, _ in assign_one_to_one(cost, radius):
            hit[chosen[e][0], i] = True
    return hit, side, inside


def _short_side(box: Sequence[float]) -> float:
    return min(box[2] - box[0], box[3] - box[1])


def _centre(box: Sequence[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _body_width(box: Sequence[float]) -> float:
    return max(box[2] - box[0], 1e-6)


def _unmatched(
    boxes: Sequence[Sequence[Sequence[float]]],
    gate: np.ndarray,
    dets: Sequence[Sequence[Sequence[float]]],
    recorded: Sequence[Sequence[Sequence[float]]],
    radius: float,
    min_side: float,
) -> tuple[int, int]:
    """``(counted, unmatched)`` detections of a view."""
    counted = unmatched = 0
    for f in range(gate.shape[1]):
        expected = [d[:4] for d in recorded[f] if _short_side(d) >= min_side]
        for b in boxes[f]:
            slot = int(b[0])
            if 0 <= slot < gate.shape[0] and gate[slot, f]:
                box = [float(v) for v in b[1:5]]
                if _short_side(box) >= min_side:
                    expected.append(box)
        centres = [_centre(b) for b in expected]
        widths = [_body_width(b) for b in expected]
        frame = [d[:4] for d in dets[f] if _short_side(d) >= min_side]
        counted += len(frame)
        for d in frame:
            cx, cy = _centre(d)
            if not any(
                np.hypot(cx - x, cy - y) <= radius * max(w, _body_width(d))
                for (x, y), w in zip(centres, widths)
            ):
                unmatched += 1
    return counted, unmatched


def _slots(mask: np.ndarray, n_slots: int) -> np.ndarray:
    out = np.zeros((n_slots, mask.shape[1]), dtype=bool)
    rows = min(n_slots, mask.shape[0])
    out[:rows] = mask[:rows]
    return out


def recording_hits(window: Mapping[str, Any], *, radius: float = MATCH_RADIUS) -> np.ndarray:
    """The recorded video's own hits of a dumped view: the confirmed recorded players that its
    detections hit, one to one among all confirmed players per frame.

    Args:
        window (Mapping[str, Any]): a view of the dump
            (:func:`~worldcast.engine.evaluation.detector.read_detections`).
        radius (float): match radius, body widths.

    Returns:
        np.ndarray: ``[n_slots, n_frames]`` bool.
    """
    confirmed = np.asarray(window["confirmed"]) > 0
    hit, _, _ = _hits(
        window["boxes"],
        confirmed,
        detections_of(window, "real"),
        window["width"],
        window["height"],
        radius,
    )
    return hit


def visibility_gate(labels: np.ndarray, first: int, count: int) -> np.ndarray:
    """The visibility gate of a view: the given visibility folded onto latent frames.

    Frame ``r`` of the rollout lies in latent frame ``0`` if ``r = 0``, else ``(r + 3) // 4``; it is
    visible if any of that latent frame's video frames is.

    Args:
        labels (np.ndarray): ``[P, R]`` bool, each player's visibility given to the generator per
            video frame of the rollout, from its first frame; frames past ``R`` are not visible.
        first (int): the rollout frame of the view's first frame.
        count (int): frames of the view.

    Returns:
        np.ndarray: ``[P, count]`` bool.
    """
    labels = np.asarray(labels, dtype=bool)
    rows = int(first) + np.arange(int(count))
    latents = np.where(rows <= 0, 0, (rows + 3) // 4)
    out = np.zeros((labels.shape[0], int(count)), dtype=bool)
    for latent in np.unique(latents):
        frames = [r for r in video_frames_of(int(latent)) if r < labels.shape[1]]
        if frames:
            out[:, latents == latent] = labels[:, frames].any(axis=1)[:, None]
    return out


def gt_visibility_gate(
    label_root: str | Path, media_id: str, start_frame: int, count: int, rollout_start: int
) -> np.ndarray:
    """The visibility gate of a view under GT states: the observer's map visibility labels
    (visible and valid; :func:`~worldcast.data.labels.visibility_rows`) at the rollout's video
    frames, folded (:func:`visibility_gate`).

    Args:
        label_root (str | Path): holds ``<media_id>.npz``.
        media_id (str): the observer's recording.
        start_frame (int): the view's first source frame.
        count (int): frames of the view.
        rollout_start (int): the rollout's first source frame: ``start_frame`` for a ten-second
            window, the session's first frame for a window of a whole round (0 for the paper's).

    Returns:
        np.ndarray: ``[P, count]`` bool.
    """
    offset = int(start_frame) - int(rollout_start)
    if offset < 0 or offset % SOURCE_FRAMES_PER_VIDEO_FRAME:
        raise ValueError(
            f"the view at {start_frame} is not a frame of the rollout at {rollout_start}"
        )
    first = offset // SOURCE_FRAMES_PER_VIDEO_FRAME
    last = int(first) + int(count) - 1
    rows = np.arange(video_frames_of((last + 3) // 4)[-1] + 1)
    visible, _ = visibility_rows(
        label_root, media_id, int(rollout_start) + SOURCE_FRAMES_PER_VIDEO_FRAME * rows
    )
    return visibility_gate(visible > 0, first, count)


def score_view(
    window: Mapping[str, Any],
    arm: str,
    gate: np.ndarray,
    boxes: Sequence[Sequence[Sequence[float]]] | None = None,
    *,
    radius: float = MATCH_RADIUS,
    min_side: float = MIN_SIDE,
    targets: Collection[int] | None = None,
) -> ViewCounts:
    """The counts of one generated video of a dumped view.

    Args:
        window (Mapping[str, Any]): a view of the dump
            (:func:`~worldcast.engine.evaluation.detector.read_detections`).
        arm (str): the generated video (``dets_<arm>``); ``real`` scores the recorded video.
        gate (np.ndarray): ``[P, n_frames]`` bool, the visibility gate (:func:`visibility_gate`),
            rows by slot.
        boxes (Sequence | None): the box list of the given player states; ``None``: the recorded
            boxes (GT states).
        radius (float): match radius, body widths, of Rendered, of the recorded video's own hits
            and of Unmatched.
        min_side (float): smallest box, px, of a scored pair, a counted detection and an expected
            box.
        targets (Collection[int] | None): the slots whose scored pairs count, kept after the
            matching of every candidate (the partner alone: the window's other view player);
            ``None``: every slot. Unmatched and the crowding count every player either way.

    Returns:
        ViewCounts: the view's counts.
    """
    width, height = window["width"], window["height"]
    n_slots = int(window["n_slots"])
    gate = _slots(np.asarray(gate, dtype=bool), n_slots)
    if gate.shape[1] != int(window["n_frames"]):
        raise ValueError(f"the gate has {gate.shape[1]} frames, the view {window['n_frames']}")
    boxes = window["boxes"] if boxes is None else boxes
    confirmed = np.asarray(window["confirmed"]) > 0
    dets = detections_of(window, arm)
    hit, side, inside = _hits(boxes, gate & confirmed, dets, width, height, radius)
    pairs = inside & (side >= min_side) & recording_hits(window, radius=radius)
    if targets is not None:
        pairs &= np.isin(np.arange(n_slots), list(targets))[:, None]
    rendered = hit & pairs
    in_view = np.zeros(gate.shape[1], dtype=np.int64)
    for i in range(gate.shape[1]):
        in_view[i] = sum(
            1
            for b in boxes[i]
            if 0 <= int(b[0]) < n_slots
            and gate[int(b[0]), i]
            and clipped_side([float(v) for v in b[1:5]], width, height) >= min_side
        )
    n_dets, n_unmatched = _unmatched(
        boxes, gate, dets, detections_of(window, "real"), radius, min_side
    )
    return ViewCounts(
        n_pairs=int(pairs.sum()),
        hits=int(rendered.sum()),
        n_dets=n_dets,
        n_unmatched=n_unmatched,
        in_view=in_view,
        frame_pairs=pairs.sum(axis=0),
        frame_hits=rendered.sum(axis=0),
    )


def summarize_entity(views: Iterable[ViewCounts]) -> EntitySummary:
    """Median Rendered and Unmatched over the views that reach their minimum counts."""
    views = list(views)
    rendered = [v.rendered for v in views if v.rendered is not None]
    unmatched = [v.unmatched for v in views if v.unmatched is not None]
    return EntitySummary(
        rendered=statistics.median(rendered) if rendered else math.nan,
        rendered_views=len(rendered),
        unmatched=statistics.median(unmatched) if unmatched else math.nan,
        unmatched_views=len(unmatched),
    )


def pool_by_crowding(
    in_view: np.ndarray, pairs: np.ndarray, hits: np.ndarray
) -> dict[int, CrowdingBin]:
    """Pool the scored pairs of the frames by the number of players in view (frames with none
    left out).

    Args:
        in_view (np.ndarray): ``[N]`` players in view per frame (:attr:`ViewCounts.in_view`).
        pairs (np.ndarray): ``[N]`` scored pairs per frame.
        hits (np.ndarray): ``[N]`` rendered scored pairs per frame.

    Returns:
        dict[int, CrowdingBin]: players in view -> the pooled frames, ascending.
    """
    in_view, pairs, hits = (np.asarray(x, dtype=np.int64) for x in (in_view, pairs, hits))
    return {
        int(n): CrowdingBin(
            frames=int((in_view == n).sum()),
            pairs=int(pairs[in_view == n].sum()),
            hits=int(hits[in_view == n].sum()),
        )
        for n in np.unique(in_view[in_view >= 1])
    }


def window_of(view: str) -> str:
    """The ten-second window of a view ``<match>-<map>-r<round>-p<slot>@<start>``:
    ``<match>-<map>-r<round>@<start>``, shared by the window's views (the bootstrap's cluster)."""
    media, start = view.rsplit("@", 1)
    return f"{media.rsplit('-', 1)[0]}@{int(start)}"
