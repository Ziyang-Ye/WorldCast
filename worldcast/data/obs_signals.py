"""Observer signals (flash blindness, scope zoom) per latent frame, from labels of the video.

The label files (``flashlabels/``, ``scopelabels/`` under config ``paths.obs_signal_label_root``)
are described in docs/data.md; a missing file is an error. The signals feed the generator's
observer-signal branch.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Item / batch keys of the five per-latent signals, in the branch's argument order.
OBS_SIGNAL_KEYS = (
    "obs_flash_flag",
    "obs_flash_valid",
    "obs_scope_on",
    "obs_scope_level",
    "obs_scope_valid",
)
#: Scope zoom levels: 0 unscoped, 1 first zoom, 2 second zoom.
SCOPE_LEVELS = 3
#: The per-frame fields of a scope label file in its full-video layout.
_SCOPE_DENSE_FIELDS = ("scoped_vis", "level", "corner_max", "center", "attack2", "weapon_id")


@dataclass(frozen=True)
class FlashCurve:
    """Flash label of one recording.

    Attributes:
        lum (np.ndarray): ``[L]`` float16 screen luminance per sample, as stored.
        hot_threshold (float): a sample above it is flash-white.
        stride (int | None): source frames per sample (None: unknown alignment).
        first_frame (int): source frame of sample 0.
    """

    lum: np.ndarray
    hot_threshold: float
    stride: int | None
    first_frame: int


@dataclass(frozen=True)
class ScopeCurve:
    """Scope label of one recording.

    Attributes:
        scoped_vis (np.ndarray): ``[L]``, > 0 where the scope overlay is on screen.
        level (np.ndarray): ``[L]`` zoom level.
        stride (int | None): source frames per sample (None: unknown alignment).
        first_frame (int): source frame of sample 0.
    """

    scoped_vis: np.ndarray
    level: np.ndarray
    stride: int | None
    first_frame: int


@dataclass(frozen=True)
class ObsSignals:
    """Per-latent observer signals of one window, each ``[N]`` int64.

    Attributes:
        flash_flag (np.ndarray): 1 if any of the latent frame's pixel frames is flash-white.
        flash_valid (np.ndarray): 0 where the flash label cannot be sampled.
        scope_on (np.ndarray): 1 if the scope overlay is on at the latent frame's last pixel frame.
        scope_level (np.ndarray): its zoom level.
        scope_valid (np.ndarray): 0 where the scope label cannot be sampled.
    """

    flash_flag: np.ndarray
    flash_valid: np.ndarray
    scope_on: np.ndarray
    scope_level: np.ndarray
    scope_valid: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """``{key: [N] int64}`` keyed by :data:`OBS_SIGNAL_KEYS`."""
        values = (
            self.flash_flag,
            self.flash_valid,
            self.scope_on,
            self.scope_level,
            self.scope_valid,
        )
        return dict(zip(OBS_SIGNAL_KEYS, values))


def flash_label_path(label_root: str | Path, media_id: str) -> Path:
    """``<label_root>/flashlabels/<media_id>.npz``."""
    return Path(label_root) / "flashlabels" / f"{media_id}.npz"


def scope_label_path(label_root: str | Path, media_id: str) -> Path:
    """``<label_root>/scopelabels/<media_id>.npz``."""
    return Path(label_root) / "scopelabels" / f"{media_id}.npz"


def _check_stride(stride: int | None) -> int | None:
    if stride is None:
        return None
    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"label stride must be positive, got {stride}")
    return stride


def load_flash_curve(path: str | Path) -> FlashCurve:
    """Read a flash label file; ``FileNotFoundError`` if it does not exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"flash label file missing: {path}")
    with np.load(path, allow_pickle=False) as z:
        return FlashCurve(
            lum=z["lum"].copy(),
            hot_threshold=float(z["hot_threshold"]),
            stride=_check_stride(int(z["stride"]) if "stride" in z.files else None),
            first_frame=int(z["first_frame"]) if "first_frame" in z.files else 0,
        )


def load_scope_curve(path: str | Path, *, video_frames: int) -> ScopeCurve:
    """Read a scope label file; ``FileNotFoundError`` if it does not exist.

    Without an explicit ``stride`` the file gets stride 1 only in the full-video layout: ``ncov``
    and ``thresholds`` present and all six per-frame fields 1-D with exactly ``video_frames``
    samples. Otherwise the alignment stays unknown.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"scope label file missing: {path}")
    count = int(video_frames)
    with np.load(path, allow_pickle=False) as z:
        stride = int(z["stride"]) if "stride" in z.files else None
        if (
            stride is None
            and count > 0
            and "ncov" in z.files
            and "thresholds" in z.files
            and all(
                k in z.files and z[k].ndim == 1 and len(z[k]) == count for k in _SCOPE_DENSE_FIELDS
            )
        ):
            stride = 1
        return ScopeCurve(
            scoped_vis=z["scoped_vis"].copy(),
            level=z["level"].copy(),
            stride=_check_stride(stride),
            first_frame=int(z["first_frame"]) if "first_frame" in z.files else 0,
        )


def _sample_indices(
    stride: int | None, first_frame: int, raw_frames, size: int
) -> np.ndarray | None:
    """Curve indices of source frames ``raw_frames``, or None if any is off-lattice or outside."""
    if stride is None:
        return None
    delta = np.asarray(raw_frames, dtype=np.int64) - int(first_frame)
    if np.any(delta < 0) or np.any(delta % stride):
        return None
    ids = delta // stride
    return ids if bool(np.all(ids < size)) else None


def obs_signal_rows(
    flash: FlashCurve,
    scope: ScopeCurve,
    start_frame: int,
    latent_frames: int,
    *,
    skip_frame: int = 2,
) -> ObsSignals:
    """Sample both curves at the source frames of each latent frame of a window.

    Latent frame ``k`` ends at source frame ``hi = start_frame + 4 * skip_frame * k``. Flash: OR of
    ``lum > hot_threshold`` over its pixel frames (``hi - 3 skip .. hi`` step ``skip``; only ``hi``
    for latent frame 0). Scope: the sample at ``hi``.
    """
    step, base, frames = int(skip_frame), int(start_frame), int(latent_frames)
    out = {k: np.zeros(frames, dtype=np.int64) for k in OBS_SIGNAL_KEYS}

    lum, hot = np.asarray(flash.lum, dtype=np.float32), float(flash.hot_threshold)
    for k in range(frames):
        hi = base + 4 * step * k
        raw = [hi] if k == 0 else np.arange(hi - 3 * step, hi + 1, step)
        ids = _sample_indices(flash.stride, flash.first_frame, raw, len(lum))
        if ids is not None:
            out["obs_flash_flag"][k] = int((lum[ids] > hot).any())
            out["obs_flash_valid"][k] = 1

    vis, lev = np.asarray(scope.scoped_vis), np.asarray(scope.level)
    for k in range(frames):
        ids = _sample_indices(
            scope.stride, scope.first_frame, [base + 4 * step * k], min(len(vis), len(lev))
        )
        if ids is not None:
            idx = int(ids[0])
            out["obs_scope_on"][k] = int(vis[idx] > 0)
            out["obs_scope_level"][k] = int(max(0, min(SCOPE_LEVELS - 1, int(lev[idx]))))
            out["obs_scope_valid"][k] = 1
    return ObsSignals(*(out[k] for k in OBS_SIGNAL_KEYS))


def load_obs_signals(
    label_root: str | Path,
    media_id: str,
    *,
    video_frames: int,
    start_frame: int,
    latent_frames: int,
    skip_frame: int = 2,
) -> ObsSignals:
    """Per-latent observer signals of the window of ``media_id`` starting at ``start_frame``.

    ``video_frames`` (the media row's frame count) identifies the scope file's layout.
    """
    flash = load_flash_curve(flash_label_path(label_root, media_id))
    scope = load_scope_curve(scope_label_path(label_root, media_id), video_frames=video_frames)
    return obs_signal_rows(flash, scope, start_frame, latent_frames, skip_frame=skip_frame)
