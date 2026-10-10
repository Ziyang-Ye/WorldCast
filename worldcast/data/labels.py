"""Labels of a recording: the GT visibility of the other players, and the observer signals.

Visibility: ``<visibility_label_root>/<media_id>.npz`` (config ``paths.visibility_label_root``),
per source frame and three-valued: visible, occluded (valid and not visible) and unknown (not
valid); the recording player's own row is never valid. :mod:`worldcast.player_state.visibility`
folds them to latent frames. The file also says which players are inside the player's view frustum,
whether their label is known or not.

Observer signals (flash blindness, scope zoom), per latent frame, sampled from labels of the
recorded video: ``flashlabels/`` and ``scopelabels/`` under config
``paths.observer_signal_label_root``; at inference a missing file is an error. They feed the
generator's observer-signal embedding (:mod:`worldcast.modeling.observer_signals`). Formats:
docs/inference.md, "Data".
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .latents import SOURCE_FRAMES_PER_VIDEO_FRAME, source_frame, video_frames_of
from .recordings import NUM_PLAYERS

__all__ = [
    "OBSERVER_SIGNAL_KEYS",
    "SCOPE_LEVELS",
    "FlashCurve",
    "ScopeCurve",
    "flash_label_path",
    "load_flash_curve",
    "load_observer_signal_curves",
    "load_observer_signals",
    "load_scope_curve",
    "observer_signal_rows",
    "scope_label_path",
    "visibility_rows",
]

#: Item / batch keys of the five per-latent observer signals, in the order the generator takes
#: them: ``obs_flash_flag`` (1 if any of the latent frame's video frames is flash-white),
#: ``obs_scope_on`` and ``obs_scope_level`` (the scope overlay and its zoom level at the latent
#: frame's last video frame), and per label ``obs_*_valid`` (0 where it cannot be sampled).
OBSERVER_SIGNAL_KEYS = (
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


# --------------------------------------------------------------------------------------- visibility
def _label_frames(
    label_root: str | Path, media_id: str, source_frames: np.ndarray, *names: str
) -> list[np.ndarray]:
    """The label arrays ``names`` of a recording's file, each ``[P, T]`` bool at ``source_frames``:
    a frame outside the recording is False.

    Raises:
        FileNotFoundError: the recording has no label file.
        ValueError: an array does not hold :data:`~worldcast.data.recordings.NUM_PLAYERS` players.
    """
    path = Path(label_root) / f"{media_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"no visibility labels for {media_id}: {path}")
    with np.load(path) as data:
        arrays = [data[name] for name in names]
    index = np.asarray(source_frames, dtype=np.int64)
    out = []
    for array in arrays:
        if array.shape[0] != NUM_PLAYERS:
            raise ValueError(
                f"visibility labels of {media_id} hold {array.shape[0]} players, expected"
                f" {NUM_PLAYERS}"
            )
        inside = (index >= 0) & (index < array.shape[1])
        out.append(array[:, np.clip(index, 0, max(array.shape[1] - 1, 0))] & inside[None, :])
    return out


def visibility_rows(
    label_root: str | Path, media_id: str, source_frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The GT visibility labels of a recording at a window's video frames: which players its
    player sees.

    The labels were computed from the recorded positions and the map assets. A frame outside the
    recording is neither visible nor valid.

    Args:
        label_root (str | Path): holds ``<media_id>.npz``.
        media_id (str): the recording.
        source_frames (np.ndarray): ``[T]`` source frame of each video frame
            (:meth:`worldcast.data.window.WindowSpec.source_frames`).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``visible`` ``[P, T]`` float32 in {0, 1} (always within
        ``valid``) and ``valid`` ``[P, T]`` bool; outside ``valid`` the label is unknown, not "not
        visible".
    """
    visible, valid = _label_frames(
        label_root, media_id, source_frames, "_binary_visible", "_binary_eval_valid"
    )
    return (visible & valid).astype(np.float32), valid.astype(np.bool_)


# --------------------------------------------------------------------------------- observer signals
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


def load_scope_curve(path: str | Path, *, source_frames: int) -> ScopeCurve:
    """Read a scope label file; ``FileNotFoundError`` if it does not exist.

    Without an explicit ``stride`` the file gets stride 1 only in the full-video layout: ``ncov``
    and ``thresholds`` present and all six per-frame fields 1-D with exactly ``source_frames``
    samples (the recording's frame count). Otherwise the alignment stays unknown.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"scope label file missing: {path}")
    count = int(source_frames)
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
    stride: int | None, first_frame: int, source_frames: np.ndarray | list[int], size: int
) -> np.ndarray | None:
    """Curve indices of ``source_frames``, or None if any is off-lattice or outside."""
    if stride is None:
        return None
    delta = np.asarray(source_frames, dtype=np.int64) - int(first_frame)
    if np.any(delta < 0) or np.any(delta % stride):
        return None
    ids = delta // stride
    return ids if bool(np.all(ids < size)) else None


def observer_signal_rows(
    flash: FlashCurve, scope: ScopeCurve, start_frame: int, latent_frames: int
) -> dict[str, np.ndarray]:
    """The observer signals of a window: both curves sampled at the source frames of each latent
    frame.

    Flash: OR of ``lum > hot_threshold`` over the latent frame's video frames (four; one for latent
    frame 0). Scope: the sample at its last video frame. A latent frame with a source frame off the
    curve's samples or outside them is not valid, and its signals are 0.

    Returns:
        dict[str, np.ndarray]: :data:`OBSERVER_SIGNAL_KEYS` -> ``[latent_frames]`` int64.
    """
    frames = int(latent_frames)
    out = {key: np.zeros(frames, dtype=np.int64) for key in OBSERVER_SIGNAL_KEYS}

    lum, hot = np.asarray(flash.lum, dtype=np.float32), float(flash.hot_threshold)
    for k in range(frames):
        sources = int(start_frame) + SOURCE_FRAMES_PER_VIDEO_FRAME * np.asarray(video_frames_of(k))
        samples = _sample_indices(flash.stride, flash.first_frame, sources, len(lum))
        if samples is not None:
            out["obs_flash_flag"][k] = int((lum[samples] > hot).any())
            out["obs_flash_valid"][k] = 1

    overlay, level = np.asarray(scope.scoped_vis), np.asarray(scope.level)
    for k in range(frames):
        last = source_frame(start_frame, k)
        samples = _sample_indices(
            scope.stride, scope.first_frame, [last], min(len(overlay), len(level))
        )
        if samples is not None:
            sample = int(samples[0])
            out["obs_scope_on"][k] = int(overlay[sample] > 0)
            out["obs_scope_level"][k] = int(max(0, min(SCOPE_LEVELS - 1, int(level[sample]))))
            out["obs_scope_valid"][k] = 1
    return out


def load_observer_signal_curves(
    label_root: str | Path, media_id: str, *, source_frames: int, missing_ok: bool = False
) -> tuple[FlashCurve, ScopeCurve]:
    """The flash and scope curves of one recording.

    Args:
        label_root (str | Path): holds ``flashlabels/`` and ``scopelabels/``.
        media_id (str): the recording.
        source_frames (int): the recording's frame count (it identifies the scope file's layout).
        missing_ok (bool): a missing file reads as a curve of unknown alignment, whose latent
            frames are never valid (as trained), instead of raising ``FileNotFoundError``.
    """
    flash_path = flash_label_path(label_root, media_id)
    scope_path = scope_label_path(label_root, media_id)
    if missing_ok and not flash_path.is_file():
        flash = FlashCurve(
            np.zeros(0, np.float16), hot_threshold=np.nan, stride=None, first_frame=0
        )
    else:
        flash = load_flash_curve(flash_path)
    if missing_ok and not scope_path.is_file():
        scope = ScopeCurve(np.zeros(0, np.uint8), np.zeros(0, np.int8), stride=None, first_frame=0)
    else:
        scope = load_scope_curve(scope_path, source_frames=int(source_frames))
    return flash, scope


def load_observer_signals(
    label_root: str | Path,
    media_id: str,
    *,
    source_frames: int,
    start_frame: int,
    latent_frames: int,
    missing_ok: bool = False,
) -> dict[str, np.ndarray]:
    """The observer signals of the window of ``media_id`` starting at ``start_frame``:
    :data:`OBSERVER_SIGNAL_KEYS` -> ``[latent_frames]`` int64
    (:func:`load_observer_signal_curves`, :func:`observer_signal_rows`)."""
    flash, scope = load_observer_signal_curves(
        label_root, media_id, source_frames=source_frames, missing_ok=missing_ok
    )
    return observer_signal_rows(flash, scope, start_frame, latent_frames)
