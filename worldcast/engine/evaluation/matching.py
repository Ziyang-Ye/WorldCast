"""Agreement between clients by feature matching: Mat. Pix., Inliers and RotErr (Table 4a; App.
"Scene state in detail", Agreement on the scene).

Two clients' frames are compared at 384 x 672 (a recording's frames resized, bilinear and
antialiased):

* Mat. Pix. GIM's dense matcher (DKM with the ``gim_dkm_100h`` weights;
  :mod:`worldcast.third_party.gim_dkm`) matches the two frames padded with black rows to 4:3 (504 x
  672), as GIM's demo runs it. Mat. Pix. is the number of cells of the first frame whose certainty
  exceeds 0.05, on DKM's 1152 x 1536 grid over the padded frame; GIM gives black pixels, the
  padding among them, no certainty.
* Inliers and RotErr. LoFTR (kornia's ``outdoor``) matches the two grey frames; its matches with a
  confidence above 0.5 give an essential matrix (OpenCV's MAGSAC++, 1 px, confidence 0.999) at the
  frames' intrinsics (:func:`camera_intrinsics`). Inliers is the matrix's inlier count, 0 with
  fewer than 8 matches or no matrix; RotErr is the angle between the rotation ``recoverPose``
  recovers from it and the relative rotation of the two recorded cameras
  (:func:`camera_rotations`), none without a matrix.

A pair is an ordered pair of clients ``(a, b)`` at a latent frame ``k``, a between-clients row of
the recorded video's pair rows (:func:`~worldcast.engine.evaluation.depth_agreement.pair_rows`:
both players alive, the overlap gate passed): for Mat. Pix. at ``k % 32 == 0`` (every 8 s), for
the others at ``k % 4 == 0`` (every second); :func:`match_keys`. A pair is scored when it is
co-visible in the recording, in its own direction: its recorded between-clients row has an
agreeing pixel (:func:`covisible`). Per round and segment (the first 30 s, after 30 s), Mat. Pix.
and Inliers are the mean over the scored pairs and RotErr the median over those with an estimate
(:func:`round_statistics`); a table cell is the mean over the rounds, each weighted equally
(:func:`~worldcast.engine.evaluation.stats.mean_over_units`), and a difference between arms, taken
per round, has the 95% round-cluster bootstrap interval
(:func:`~worldcast.engine.evaluation.stats.unit_interval`).

On the recorded video the same pairs with the first view taken 10 s away (the chance rows) score
far worse (:func:`self_check`). With the screen mask over the weapon and hands
(:func:`in_screen_mask`), a cell counts only if it and its match lie outside the mask, and LoFTR
matches with an end inside it are dropped before the essential matrix.

The paper's values come from decord-decoded frames on NVIDIA H20 GPUs with torch's default TF32
settings (cuDNN convolutions in TF32, matmuls in float32) and LoFTR run on 8 consecutive pairs at a
time; another GPU, TF32 setting or batching changes single matches.

Optional dependencies, imported when used: decord, OpenCV, kornia (LoFTR), torchvision and einops
(GIM's matcher; unpickling its checkpoint needs yacs).
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
import torch.nn.functional as F

from worldcast.data.camera import half_angle_tangents
from worldcast.data.latents import FPS, FRAME_SIZE, VIDEO_FRAMES_PER_LATENT
from worldcast.hub import check_metric_weight

from .depth_agreement import (
    BETWEEN_CLIENTS,
    CHANCE,
    ROW_COLUMNS,
    SCENE_SPLIT_SECONDS,
    chance_frames,
    frame_rotations,
)

__all__ = [
    "CERTAINTY_THRESHOLD",
    "GIM_FIELDS",
    "GIM_SIZE",
    "GIM_STRIDE",
    "KEY_COLUMNS",
    "LOFTR_BATCH",
    "LOFTR_CONFIDENCE",
    "LOFTR_FIELDS",
    "LOFTR_STRIDE",
    "LUMA",
    "MAGSAC_CONFIDENCE",
    "MAGSAC_THRESHOLD",
    "MASK_LOWER_LEFT",
    "MASK_LOWER_RIGHT",
    "MIN_MATCHES",
    "STATISTICS",
    "camera_intrinsics",
    "camera_rotations",
    "covisible",
    "grey",
    "in_screen_mask",
    "load_gim",
    "load_loftr",
    "loftr_matches",
    "match_keys",
    "matched_pixels",
    "needed_frames",
    "read_frames",
    "relative_pose",
    "rotation_error",
    "round_statistics",
    "score_matched_pixels",
    "score_relative_poses",
    "self_check",
]

#: ``(height, width)`` GIM's matcher runs at; a frame is padded to its 4:3 aspect first.
GIM_SIZE = (672, 896)
#: A cell is matched (Mat. Pix.) when its certainty exceeds this.
CERTAINTY_THRESHOLD = 0.05
#: Mat. Pix. is scored at the latent frames ``k`` with ``k % GIM_STRIDE == 0`` (every 8 s).
GIM_STRIDE = 32
#: Inliers and RotErr are scored at ``k % LOFTR_STRIDE == 0`` (every second).
LOFTR_STRIDE = 4
#: LoFTR's matches with a confidence above this are kept.
LOFTR_CONFIDENCE = 0.5
#: LoFTR runs on this many consecutive pairs at a time.
LOFTR_BATCH = 8
#: Fewest matches an essential matrix is estimated from.
MIN_MATCHES = 8
#: MAGSAC++ inlier threshold, px.
MAGSAC_THRESHOLD = 1.0
#: MAGSAC++ confidence.
MAGSAC_CONFIDENCE = 0.999
#: Weights of R, G and B in LoFTR's grey frames (ITU-R BT.601 luma).
LUMA = (0.299, 0.587, 0.114)
#: The screen mask covers the lower right, ``u >= 0.5`` and ``v >= 0.45`` (fractions of the frame's
#: width and height) ...
MASK_LOWER_RIGHT = (0.50, 0.45)
#: ... and the lower-left corner, ``u <= 0.3`` and ``v >= 0.7``.
MASK_LOWER_LEFT = (0.30, 0.70)
#: The int64 columns of a key: the row kind, the two clients, the latent frame, and the latent
#: frames of the two clients' compared frames (a chance key's first view is 10 s away).
KEY_COLUMNS = ("kind", "a", "b", "frame", "frame_a", "frame_b")
#: The per-pair results of GIM's matcher: Mat. Pix., and with the screen mask.
GIM_FIELDS = ("matched_pixels", "matched_pixels_masked")
#: The per-pair results of LoFTR: Inliers and RotErr, each also with the screen mask.
LOFTR_FIELDS = ("inliers", "inliers_masked", "rotation_error", "rotation_error_masked")
#: The per-pair results and their per-round statistic: the mean over the scored pairs (Mat. Pix.,
#: Inliers), or the median over those with an estimate (RotErr).
STATISTICS = MappingProxyType(
    {
        field: "median" if field.startswith("rotation_error") else "mean"
        for field in (*GIM_FIELDS, *LOFTR_FIELDS)
    }
)

_AGREE = ROW_COLUMNS.index("agree")
#: Camera axes (forward, right, up) -> OpenCV's (right, down, forward).
_TO_OPENCV = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def camera_intrinsics() -> np.ndarray:
    """The intrinsic matrix of a 384 x 672 frame, ``[3, 3]`` float64.

    The focal lengths follow the recordings' field of view (HFOV 106.26, ``tan_v = tan_h 9 / 16``):
    ``fx = 336 / tan_h = 252.0009``, ``fy = 192 / tan_v = 256.0010``; the principal point is the
    frame's centre.
    """
    tan_h, tan_v = half_angle_tangents()
    height, width = FRAME_SIZE
    return np.array(
        [
            [width / 2 / tan_h, 0.0, width / 2],
            [0.0, height / 2 / tan_v, height / 2],
            [0.0, 0.0, 1.0],
        ]
    )


def camera_rotations(yaw_degrees: np.ndarray, pitch_degrees: np.ndarray) -> np.ndarray:
    """World-to-camera rotations of recorded views, in OpenCV's camera axes (x right, y down, z
    forward).

    The inverse of :func:`~worldcast.engine.evaluation.depth_agreement.frame_rotations` (fitted to
    the float32 token rays, as the paper's RotErr was computed) with its axes reordered.

    Args:
        yaw_degrees (np.ndarray): ``[N]`` recorded yaw, degrees.
        pitch_degrees (np.ndarray): ``[N]`` recorded pitch, degrees (> 0 looks down).

    Returns:
        np.ndarray: ``[N, 3, 3]`` float64.
    """
    fitted = frame_rotations(yaw_degrees, pitch_degrees)
    return _TO_OPENCV @ np.linalg.inv(fitted).swapaxes(-1, -2)


def rotation_error(estimated: np.ndarray, relative: np.ndarray) -> float:
    """The angle between two ``[3, 3]`` rotations, degrees."""
    cosine = (np.trace(estimated.T @ relative) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def relative_pose(
    points_a: np.ndarray, points_b: np.ndarray, rotation_a: np.ndarray, rotation_b: np.ndarray
) -> tuple[int, float]:
    """Inliers and RotErr of one pair from its matches.

    Args:
        points_a (np.ndarray): ``[m, 2]`` float32 positions of the matches in frame ``a``, px (pixel
            centres at ``+ 0.5``).
        points_b (np.ndarray): ``[m, 2]`` float32 their positions in frame ``b``.
        rotation_a (np.ndarray): ``[3, 3]`` the world-to-camera rotation of ``a``
            (:func:`camera_rotations`).
        rotation_b (np.ndarray): ``[3, 3]`` that of ``b``.

    Returns:
        tuple[int, float]: the essential matrix's inliers (0 without a matrix) and RotErr, degrees
        (NaN without a matrix).
    """
    import cv2

    if len(points_a) < MIN_MATCHES:
        return 0, math.nan
    intrinsics = camera_intrinsics()
    essential, inliers = cv2.findEssentialMat(
        points_a,
        points_b,
        intrinsics,
        method=cv2.USAC_MAGSAC,
        prob=MAGSAC_CONFIDENCE,
        threshold=MAGSAC_THRESHOLD,
    )
    if essential is None or essential.shape != (3, 3):
        return 0, math.nan
    # recoverPose tests cheirality on a copy of the inlier mask; the count is MAGSAC++'s
    _, recovered, _, _ = cv2.recoverPose(
        essential, points_a, points_b, intrinsics, mask=inliers.copy()
    )
    return int(inliers.sum()), rotation_error(recovered, rotation_b @ rotation_a.T)


def in_screen_mask(
    u: np.ndarray | torch.Tensor, v: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Whether positions lie in the screen mask over the weapon and hands.

    Args:
        u (np.ndarray | torch.Tensor): horizontal positions, fractions of the frame's width.
        v (np.ndarray | torch.Tensor): vertical positions, fractions of its height (0 at the top).

    Returns:
        np.ndarray | torch.Tensor: bool, of the inputs' kind and shape.
    """
    right, below = MASK_LOWER_RIGHT
    left, corner = MASK_LOWER_LEFT
    return ((u >= right) & (v >= below)) | ((u <= left) & (v >= corner))


def match_keys(
    rows: np.ndarray,
    latent_frames: Sequence[int],
    stride: int,
    kinds: Sequence[int] = (BETWEEN_CLIENTS,),
) -> np.ndarray:
    """The pairs a matcher compares in one round, from the recorded video's pair rows.

    Args:
        rows (np.ndarray): ``[n, 10]`` the recorded video's pair rows of the round
            (:data:`~worldcast.engine.evaluation.depth_agreement.ROW_COLUMNS`).
        latent_frames (Sequence[int]): latent frames ``N`` of each client, by client index.
        stride (int): :data:`GIM_STRIDE` or :data:`LOFTR_STRIDE`; the rows at ``k % stride == 0``.
        kinds (Sequence[int]): the row kinds kept: between clients, and the chance rows for
            :func:`self_check`.

    Returns:
        np.ndarray: ``[m, 6]`` int64 keys (:data:`KEY_COLUMNS`) in the rows' order. The second
        client's frame is at ``k``, the first's too, or for a chance key at
        :func:`~worldcast.engine.evaluation.depth_agreement.chance_frames` of ``min(N_a, N_b)``.
    """
    rows = np.asarray(rows, dtype=np.int64)
    rows = rows[(rows[:, 3] % stride == 0) & np.isin(rows[:, 4], kinds)]
    kind, a, b, k = rows[:, 4], rows[:, 1], rows[:, 2], rows[:, 3]
    counts = np.asarray(latent_frames, dtype=np.int64)
    frame_a = np.where(kind == CHANCE, chance_frames(k, np.minimum(counts[a], counts[b])), k)
    return np.stack([kind, a, b, k, frame_a, k], 1)


def needed_frames(keys: np.ndarray) -> dict[int, list[int]]:
    """Each client's latent frames that ``keys`` compare, ascending."""
    keys = np.asarray(keys)
    out: dict[int, set[int]] = {}
    for _, a, b, _, frame_a, frame_b in keys.tolist():
        out.setdefault(a, set()).add(frame_a)
        out.setdefault(b, set()).add(frame_b)
    return {client: sorted(frames) for client, frames in sorted(out.items())}


def read_frames(
    path: str | Path,
    latent_frames: Sequence[int],
    frames_per_latent: int,
    device: torch.device | str,
) -> dict[int, torch.Tensor]:
    """A client's frames at latent frames, decoded with decord.

    Latent frame ``k`` is video frame ``frames_per_latent * k``: 4 in a generated video
    (``VIDEO_FRAMES_PER_LATENT``), 8 in a recording (``SOURCE_FRAMES_PER_LATENT``), whose round
    starts at its first frame.

    Args:
        path (str | Path): the video.
        latent_frames (Sequence[int]): the latent frames.
        frames_per_latent (int): video frames per latent frame.
        device (torch.device | str): where the frames go.

    Returns:
        dict[int, torch.Tensor]: latent frame -> ``[3, 384, 672]`` float32 RGB in ``[0, 1]``.
    """
    import decord

    reader = decord.VideoReader(str(path), num_threads=4)
    indices = [frames_per_latent * int(k) for k in latent_frames]
    if max(indices) >= len(reader):
        raise ValueError(f"{path}: {len(reader)} frames, frame {max(indices)} asked for")
    pixels = torch.from_numpy(reader.get_batch(indices).asnumpy())
    frames = pixels.to(device).permute(0, 3, 1, 2).float() / 255.0
    if tuple(frames.shape[-2:]) != FRAME_SIZE:
        frames = F.interpolate(
            frames, FRAME_SIZE, mode="bilinear", antialias=True, align_corners=False
        ).clamp(0, 1)
    return {int(k): frame for k, frame in zip(latent_frames, frames)}


def load_gim(path: str | Path, device: torch.device | str) -> torch.nn.Module:
    """GIM's dense matcher with the weights at ``path`` (checked against their sha256 in
    :data:`~worldcast.hub.METRIC_WEIGHTS`), on ``device``, in eval mode."""
    check_metric_weight("gim_dkm", path)
    from worldcast.third_party.gim_dkm.model import DKMv3

    model = DKMv3(*GIM_SIZE)
    # the checkpoint pickles its training config (yacs) beside the weights; the digest is checked
    state = torch.load(str(path), map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(
        {
            key.removeprefix("model."): value
            for key, value in state.items()
            if "encoder.net.fc" not in key
        }
    )
    return model.eval().to(device)


def matched_pixels(
    model: torch.nn.Module, query: torch.Tensor, support: torch.Tensor
) -> tuple[int, int]:
    """Mat. Pix. of two frames, and of the screen-masked frames.

    Args:
        model (torch.nn.Module): :func:`load_gim`.
        query (torch.Tensor): ``[3, 384, 672]`` RGB in ``[0, 1]``, the first client's frame, on the
            model's device.
        support (torch.Tensor): ``[3, 384, 672]`` the second client's frame.

    Returns:
        tuple[int, int]: the matched cells of the query frame, and those of them that lie, with
        their match, outside the screen mask.
    """
    from worldcast.third_party.gim_dkm.padding import get_padding_size

    images, sizes = [], []
    for frame in (query[None], support[None]):
        width, height, left, right, top, bottom = get_padding_size(frame, *GIM_SIZE)
        images.append(F.pad(frame, (left, right, top, bottom)))
        sizes.append((width, height, left, top, width + left + right, height + top + bottom))
    warp, certainty = model.match(*images)
    cells = certainty.shape[1] // 2
    matched = certainty[:, :cells] > CERTAINTY_THRESHOLD
    warp = warp[:, :cells]
    outside = torch.ones_like(matched)
    for (width, height, left, top, padded_width, padded_height), x, y in zip(sizes, (0, 2), (1, 3)):
        u = ((warp[..., x] + 1) / 2 * padded_width - left) / width
        v = ((warp[..., y] + 1) / 2 * padded_height - top) / height
        outside &= ~in_screen_mask(u, v)
    return int(matched.sum()), int((matched & outside).sum())


def load_loftr(path: str | Path, device: torch.device | str) -> torch.nn.Module:
    """kornia's LoFTR with its ``outdoor`` weights at ``path`` (checked against their sha256 in
    :data:`~worldcast.hub.METRIC_WEIGHTS`), on ``device``, in eval mode."""
    check_metric_weight("loftr", path)
    from kornia.feature import LoFTR

    model = LoFTR(pretrained=None)
    model.load_state_dict(
        torch.load(str(path), map_location="cpu", weights_only=False)["state_dict"]
    )
    return model.to(device).eval()


def grey(frames: torch.Tensor) -> torch.Tensor:
    """``[n, 3, H, W]`` RGB frames -> ``[n, 1, H, W]`` grey (:data:`LUMA`), in their dtype."""
    return LUMA[0] * frames[:, 0:1] + LUMA[1] * frames[:, 1:2] + LUMA[2] * frames[:, 2:3]


def loftr_matches(
    model: torch.nn.Module, first: torch.Tensor, second: torch.Tensor
) -> list[tuple[np.ndarray, np.ndarray]]:
    """LoFTR's matches with a confidence above :data:`LOFTR_CONFIDENCE`, of a batch of frame pairs.

    Args:
        model (torch.nn.Module): :func:`load_loftr`.
        first (torch.Tensor): ``[B, 1, 384, 672]`` the first frames, grey (:func:`grey`).
        second (torch.Tensor): ``[B, 1, 384, 672]`` the second frames.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: per pair, ``[m, 2]`` float32 positions of the matches
        in each frame, px (pixel centres at ``+ 0.5``).
    """
    with torch.no_grad():
        out = model({"image0": first, "image1": second})
    keep = out["confidence"] > LOFTR_CONFIDENCE
    points_a = out["keypoints0"][keep].float().cpu().numpy()
    points_b = out["keypoints1"][keep].float().cpu().numpy()
    pair = out["batch_indexes"][keep].long().cpu().numpy()
    return [(points_a[pair == i] + 0.5, points_b[pair == i] + 0.5) for i in range(len(first))]


def score_matched_pixels(
    model: torch.nn.Module, keys: np.ndarray, frames: Mapping[int, Mapping[int, torch.Tensor]]
) -> dict[str, np.ndarray]:
    """Mat. Pix. of every key.

    Args:
        model (torch.nn.Module): :func:`load_gim`.
        keys (np.ndarray): ``[n, 6]`` keys (:func:`match_keys` at :data:`GIM_STRIDE`).
        frames (Mapping[int, Mapping[int, torch.Tensor]]): client -> latent frame -> its frame
            (:func:`read_frames`).

    Returns:
        dict[str, np.ndarray]: ``matched_pixels`` and ``matched_pixels_masked``, ``[n]`` int64.
    """
    out = {name: np.zeros(len(keys), np.int64) for name in GIM_FIELDS}
    for i, (_, a, b, _, frame_a, frame_b) in enumerate(np.asarray(keys).tolist()):
        counts = matched_pixels(model, frames[a][frame_a], frames[b][frame_b])
        out["matched_pixels"][i], out["matched_pixels_masked"][i] = counts
    return out


def score_relative_poses(
    model: torch.nn.Module,
    keys: np.ndarray,
    frames: Mapping[int, Mapping[int, torch.Tensor]],
    rotations: Mapping[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Inliers and RotErr of every key, LoFTR on :data:`LOFTR_BATCH` consecutive keys at a time.

    Args:
        model (torch.nn.Module): :func:`load_loftr`.
        keys (np.ndarray): ``[n, 6]`` keys (:func:`match_keys` at :data:`LOFTR_STRIDE`), in their
            order.
        frames (Mapping[int, Mapping[int, torch.Tensor]]): client -> latent frame -> its frame
            (:func:`read_frames`).
        rotations (Mapping[int, np.ndarray]): client -> ``[N, 3, 3]`` its recorded cameras'
            :func:`camera_rotations`.

    Returns:
        dict[str, np.ndarray]: ``inliers`` and ``inliers_masked``, ``[n]`` int64;
        ``rotation_error`` and ``rotation_error_masked``, ``[n]`` float32 degrees, NaN without an
        estimate.
    """
    keys = np.asarray(keys)
    height, width = FRAME_SIZE
    out = {name: np.zeros(len(keys), np.int64) for name in ("inliers", "inliers_masked")}
    for name in ("rotation_error", "rotation_error_masked"):
        out[name] = np.full(len(keys), np.nan, np.float32)
    for start in range(0, len(keys), LOFTR_BATCH):
        batch = keys[start : start + LOFTR_BATCH].tolist()
        first = grey(torch.stack([frames[a][frame_a] for _, a, _, _, frame_a, _ in batch]))
        second = grey(torch.stack([frames[b][frame_b] for _, _, b, _, _, frame_b in batch]))
        for i, (key, (points_a, points_b)) in enumerate(
            zip(batch, loftr_matches(model, first, second)), start
        ):
            _, a, b, _, frame_a, frame_b = key
            pair = (rotations[a][frame_a], rotations[b][frame_b])
            out["inliers"][i], out["rotation_error"][i] = relative_pose(points_a, points_b, *pair)
            masked = in_screen_mask(points_a[:, 0] / width, points_a[:, 1] / height)
            masked |= in_screen_mask(points_b[:, 0] / width, points_b[:, 1] / height)
            out["inliers_masked"][i], out["rotation_error_masked"][i] = relative_pose(
                points_a[~masked], points_b[~masked], *pair
            )
    return out


def covisible(keys: np.ndarray, recorded_rows: np.ndarray) -> np.ndarray:
    """Whether each key's pair is co-visible in the recording at its latent frame.

    The recorded video's between-clients row of ``(a, b, k)`` has an agreeing pixel; ``(a, b)`` and
    ``(b, a)`` are judged apart, and a chance key goes with its same-instant pair.

    Args:
        keys (np.ndarray): ``[n, 6]`` keys (:data:`KEY_COLUMNS`).
        recorded_rows (np.ndarray): ``[r, 10]`` the recorded video's pair rows of the round.

    Returns:
        np.ndarray: ``[n]`` bool.
    """
    rows = np.asarray(recorded_rows, dtype=np.int64)
    seen = rows[(rows[:, 4] == BETWEEN_CLIENTS) & (rows[:, _AGREE] > 0), 1:4]
    pairs = np.asarray(keys, dtype=np.int64)[:, 1:4]
    shape = tuple(np.maximum(seen.max(0, initial=0), pairs.max(0, initial=0)) + 1)
    return np.isin(np.ravel_multi_index(pairs.T, shape), np.ravel_multi_index(seen.T, shape))


def _statistic(values: np.ndarray, statistic: str) -> float:
    if statistic == "mean":
        return float(values.mean()) if len(values) else math.nan
    if statistic == "median":
        return float(np.nanmedian(values)) if np.isfinite(values).any() else math.nan
    raise ValueError(f"statistic {statistic!r}: expected 'mean' or 'median'")


def round_statistics(
    keys: np.ndarray, values: np.ndarray, covisible_keys: np.ndarray, statistic: str
) -> np.ndarray:
    """One round's statistic of a per-pair result over its scored pairs, per segment.

    The scored pairs are the co-visible between-clients keys; the segments are the first 30 s and
    after 30 s (latent frame ``k`` is at ``k / 4`` s).

    Args:
        keys (np.ndarray): ``[n, 6]`` keys (:data:`KEY_COLUMNS`).
        values (np.ndarray): ``[n]`` the per-pair result (:data:`STATISTICS`).
        covisible_keys (np.ndarray): ``[n]`` bool, :func:`covisible`.
        statistic (str): ``mean``, or ``median`` over the pairs with an estimate.

    Returns:
        np.ndarray: ``[2]`` float64; NaN for a segment without a scored pair (with an estimate).
    """
    keys = np.asarray(keys)
    values = np.asarray(values, dtype=np.float64)
    scored = np.asarray(covisible_keys, dtype=bool) & (keys[:, 0] == BETWEEN_CLIENTS)
    later = keys[:, 3] * (VIDEO_FRAMES_PER_LATENT / FPS) >= SCENE_SPLIT_SECONDS
    return np.array([_statistic(values[scored & (later == part)], statistic) for part in (0, 1)])


def self_check(
    keys: np.ndarray, values: np.ndarray, covisible_keys: np.ndarray, statistic: str
) -> tuple[float, float]:
    """One round's statistic on the recorded video at the same instant and with the first view 10 s
    away, over the whole round.

    Both are taken over the co-visible pairs and latent frames that have both a between-clients and
    a chance key.

    Args:
        keys (np.ndarray): ``[n, 6]`` the recorded video's keys, both kinds (:func:`match_keys`).
        values (np.ndarray): ``[n]`` the per-pair result (:data:`STATISTICS`).
        covisible_keys (np.ndarray): ``[n]`` bool, :func:`covisible`.
        statistic (str): ``mean``, or ``median`` over the pairs with an estimate.

    Returns:
        tuple[float, float]: the statistic at the same instant and 10 s apart.
    """
    keys = np.asarray(keys)
    values = np.asarray(values, dtype=np.float64)
    index: dict[int, dict[tuple[int, int, int], int]] = {BETWEEN_CLIENTS: {}, CHANCE: {}}
    for i, (kind, a, b, k) in enumerate(keys[:, :4].tolist()):
        if kind in index and covisible_keys[i]:
            index[kind][(a, b, k)] = i
    common = sorted(index[BETWEEN_CLIENTS].keys() & index[CHANCE].keys())
    return tuple(
        _statistic(values[[index[kind][pair] for pair in common]], statistic)
        for kind in (BETWEEN_CLIENTS, CHANCE)
    )
