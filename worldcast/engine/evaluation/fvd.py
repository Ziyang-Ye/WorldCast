"""FVD: the Fréchet distance between the I3D features of generated and recorded clips.

A clip goes into the Kinetics-400 I3D detector whole, at 0-255, with the detector's own rescale to
[-1, 1] and its bilinear resize of every frame to 224 x 224 (no antialiasing; the aspect ratio is
not kept); its feature is the detector's 400 logits before the softmax, averaged over time. The
distance is between the Gaussian fits of the two sets, in float64; it is unpaired, the stems only
pair the reads. Two clip sets:

* the ten-second windows (Tables 1, 2, 5 and 9): every frame of a window, all 161, the recorded
  first frame included; the real clip is the window's ``gt/`` mp4 (the VAE reconstruction);
* the whole rounds (Table 17): each complete ten-second segment ``k`` of a client's round, the
  generated frames ``1 + 160 k .. 160 (k + 1)``; the real clip is the recording over the same video
  frames, every second source frame from the round's start.

Values compare only at the same clip count: the distance is biased at small sample sizes. The
paper's values come from decord-decoded clips on NVIDIA H20 GPUs with torch's default TF32
convolutions; another decoder, GPU or TF32 setting changes the features in their last digits.

Optional dependencies, imported when used: decord (the mp4s), OpenCV (the recordings), scipy.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from worldcast.data.latents import FRAME_SIZE, SOURCE_FRAMES_PER_VIDEO_FRAME, WINDOW_VIDEO_FRAMES
from worldcast.hub import check_metric_weight
from worldcast.utils.files import sha256_file

__all__ = [
    "FEATURE_DIM",
    "MAX_IMAGINARY",
    "MIN_CLIPS",
    "SEGMENT_FRAMES",
    "SQRTM_EPS",
    "frechet_distance",
    "fvd_from_dir",
    "i3d_features",
    "load_i3d",
    "read_window",
    "round_features",
]

#: Width of an I3D feature: the 400 Kinetics logits.
FEATURE_DIM = 400
#: Generated frames of a ten-second segment of a whole round; the recorded first frame is in none.
SEGMENT_FRAMES = WINDOW_VIDEO_FRAMES - 1
#: Fewest windows an evaluation directory is scored on.
MIN_CLIPS = 64
#: Largest imaginary part of the covariance square root dropped as rounding.
MAX_IMAGINARY = 1e-3
#: Added to the diagonal of both covariances when the square root of their product is not finite.
SQRTM_EPS = 1e-6


def load_i3d(path: str | Path, device: torch.device | str) -> torch.jit.ScriptModule:
    """The I3D detector at ``path`` (TorchScript of the Kinetics-400 I3D, ``forward(x, rescale,
    resize, return_features)``; checked against its sha256 in
    :data:`~worldcast.hub.METRIC_WEIGHTS`) on ``device``, in eval mode."""
    check_metric_weight("i3d", path)
    return torch.jit.load(str(path), map_location=device).eval()


def i3d_features(model: torch.nn.Module, clip: torch.Tensor) -> torch.Tensor:
    """The I3D feature of one clip.

    Args:
        model (torch.nn.Module): the detector (:func:`load_i3d`), on its device.
        clip (Tensor): ``[T, 3, H, W]`` frames in ``[0, 1]``, ``T >= 2``.

    Returns:
        Tensor: ``[400]`` float32 on the CPU.
    """
    if clip.ndim != 4 or clip.shape[1] != 3 or clip.shape[0] < 2:
        raise ValueError(f"expected a [T >= 2, 3, H, W] clip, got {tuple(clip.shape)}")
    device = next(model.parameters()).device
    x = clip.float().clamp(0, 1).to(device).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    with torch.no_grad():
        # the detector's rescale maps 0-255 to [-1, 1]: it takes x * 255, not x
        feature = model(x * 255.0, rescale=True, resize=True, return_features=True)
    return feature.float().cpu()[0]


def frechet_distance(real: np.ndarray | torch.Tensor, fake: np.ndarray | torch.Tensor) -> float:
    """The Fréchet distance between the Gaussian fits of two feature sets.

    In float64: the means, the covariances with ``N - 1`` (``np.cov``) and scipy's ``sqrtm`` of
    their product. When that root is not finite it is taken of the product of the covariances
    with :data:`SQRTM_EPS` on their diagonals; an imaginary part up to :data:`MAX_IMAGINARY` is
    dropped.

    Args:
        real (np.ndarray | Tensor): ``[N, D]`` features of the real clips, ``N >= 2``.
        fake (np.ndarray | Tensor): ``[M, D]`` features of the generated clips, ``M >= 2``.

    Returns:
        float: ``|mu_r - mu_f|^2 + tr(C_r + C_f - 2 (C_r C_f)^(1/2))``, at least 0.
    """
    from scipy import linalg

    real, fake = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1]:
        raise ValueError(f"expected [N, D] and [M, D] features, got {real.shape} and {fake.shape}")
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("the Fréchet distance needs at least two real and two generated clips")
    if not (np.isfinite(real).all() and np.isfinite(fake).all()):
        raise ValueError("the features are not all finite")
    mu_r, mu_f = real.mean(axis=0), fake.mean(axis=0)
    cov_r, cov_f = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    covmean = linalg.sqrtm(cov_r @ cov_f)
    if not np.isfinite(covmean).all():
        eye = np.eye(len(cov_r))
        covmean = linalg.sqrtm((cov_r + SQRTM_EPS * eye) @ (cov_f + SQRTM_EPS * eye))
    if np.iscomplexobj(covmean):
        imaginary = float(np.abs(covmean.imag).max())
        if imaginary > MAX_IMAGINARY:
            raise FloatingPointError(
                f"the covariance square root has an imaginary part {imaginary}"
            )
        covmean = covmean.real
    delta = mu_r - mu_f
    value = float(delta @ delta + np.trace(cov_r + cov_f - 2.0 * covmean))
    if not np.isfinite(value):
        raise FloatingPointError("the Fréchet distance is not finite")
    return max(0.0, value)


def _decode(reader, indices: Sequence[int]) -> torch.Tensor:
    frames = torch.from_numpy(reader.get_batch(list(indices)).asnumpy())
    return frames.permute(0, 3, 1, 2).float() / 255.0


def read_window(path: str | Path) -> torch.Tensor:
    """The frames of a window's mp4 that I3D sees: its first
    :data:`~worldcast.data.latents.WINDOW_VIDEO_FRAMES` (all of them, the recorded first frame
    included), decoded with decord, ``[T, 3, H, W]`` float32 ``uint8 / 255``."""
    import decord

    reader = decord.VideoReader(str(path), num_threads=2)
    return _decode(reader, range(min(len(reader), WINDOW_VIDEO_FRAMES)))


def _read_cache(path: Path | None) -> dict[str, tuple[np.ndarray, tuple[int, ...]]]:
    if path is None or not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as cache:
        return {
            str(digest): (feature, tuple(int(n) for n in shape))
            for digest, feature, shape in zip(cache["sha256"], cache["features"], cache["shapes"])
        }


def _write_cache(path: Path, entries: dict[str, tuple[np.ndarray, tuple[int, ...]]]) -> None:
    digests = sorted(entries)
    part = path.with_name(f".{path.name}.part")
    with part.open("wb") as handle:
        np.savez(
            handle,
            sha256=np.array(digests),
            features=np.stack([entries[d][0] for d in digests]).astype(np.float32),
            shapes=np.array([entries[d][1] for d in digests], dtype=np.int64),
        )
    os.replace(part, path)


def fvd_from_dir(
    eval_dir: str | Path, model: torch.nn.Module, *, real_cache: str | Path | None = None
) -> dict[str, float | int]:
    """The FVD of an evaluation directory's ten-second windows.

    The directory holds ``generation/<stem>.mp4`` and ``gt/<stem>.mp4`` for each window; every
    generated stem is scored, the real set is the ``gt/`` clips of the same stems, and a window's
    two clips must have one shape.

    Args:
        eval_dir (str | Path): the evaluation directory.
        model (torch.nn.Module): the detector (:func:`load_i3d`).
        real_cache (str | Path | None): an ``.npz`` of real-clip features by the sha256 of their
            file, read and extended with the clips it lacks; it holds one machine's features.
            ``None``: no cache.

    Returns:
        dict[str, float | int]: ``fvd`` and ``clips``, the windows scored.
    """
    eval_dir = Path(eval_dir)
    stems = sorted(path.name for path in (eval_dir / "generation").glob("*.mp4"))
    if len(stems) < MIN_CLIPS:
        raise ValueError(
            f"{eval_dir}: {len(stems)} generated windows, fewer than {MIN_CLIPS}: the distance is"
            " biased at small sample sizes"
        )
    cache_path = None if real_cache is None else Path(real_cache)
    cache = _read_cache(cache_path)
    added = False
    fake, real = [], []
    for stem in stems:
        generated = read_window(eval_dir / "generation" / stem)
        truth_path = eval_dir / "gt" / stem
        digest = sha256_file(truth_path)
        if digest not in cache:
            truth = read_window(truth_path)
            cache[digest] = (i3d_features(model, truth).numpy(), tuple(truth.shape))
            added = True
        feature, shape = cache[digest]
        if tuple(generated.shape) != shape:
            raise ValueError(f"{stem}: generated {tuple(generated.shape)}, gt {shape}")
        fake.append(i3d_features(model, generated).numpy())
        real.append(feature)
    if cache_path is not None and added:
        _write_cache(cache_path, cache)
    return {"fvd": frechet_distance(np.stack(real), np.stack(fake)), "clips": len(stems)}


def _read_recording(path: str | Path, count: int, source_frames: int) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(path))
    total = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if total != source_frames:
        raise RuntimeError(f"{path}: {total} frames, expected {source_frames}")
    frames, index = [], 0
    while len(frames) < count and capture.grab():
        if index % SOURCE_FRAMES_PER_VIDEO_FRAME == 0:
            _, frame = capture.retrieve()
            if frame.shape[:2] != FRAME_SIZE:
                frame = cv2.resize(frame, FRAME_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()
    if len(frames) != count:
        raise RuntimeError(f"{path}: read {len(frames)} of {count} video frames")
    return np.stack(frames)


def round_features(
    model: torch.nn.Module, video: str | Path, recording: str | Path, *, source_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The I3D features of the complete ten-second segments of a client's whole round.

    Segment ``k`` is the generated frames ``1 + 160 k .. 160 (k + 1)`` (decord); its real clip is
    the same video frames of the recording, source frame ``2 i`` for video frame ``i`` from the
    round's start, decoded with OpenCV and resized (``INTER_LINEAR``) to 384 x 672.

    Args:
        model (torch.nn.Module): the detector (:func:`load_i3d`).
        video (str | Path): the client's generated round, from its recorded first frame.
        recording (str | Path): the client's recording of the round.
        source_frames (int): the recording's frame count, checked against the decoder's.

    Returns:
        tuple[Tensor, Tensor]: ``[K, 400]`` features of the generated and of the real segments.
    """
    import decord

    reader = decord.VideoReader(str(video), num_threads=2)
    segments = (len(reader) - 1) // SEGMENT_FRAMES
    if not segments:
        return torch.zeros(0, FEATURE_DIM), torch.zeros(0, FEATURE_DIM)
    truth = _read_recording(recording, 1 + SEGMENT_FRAMES * segments, source_frames)
    fake, real = [], []
    for k in range(segments):
        indices = range(1 + SEGMENT_FRAMES * k, 1 + SEGMENT_FRAMES * (k + 1))
        fake.append(i3d_features(model, _decode(reader, indices)))
        frames = torch.from_numpy(truth[indices.start : indices.stop])
        real.append(i3d_features(model, frames.permute(0, 3, 1, 2).float() / 255.0))
    return torch.stack(fake), torch.stack(real)
