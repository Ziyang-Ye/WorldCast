"""Ground-truth visibility labels of the other players, per source frame of one recording.

``<visibility_label_root>/<media_id>.npz`` (config ``paths.visibility_label_root``; docs/data.md).
The labels are three-valued: visible, occluded (valid and not visible) and unknown (not valid); the
recording player's own row is never valid. :mod:`worldcast.player_state.visibility` folds them to
latent frames.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FIELDS = {
    "visible": "_binary_visible",
    "eval_valid": "_binary_eval_valid",
    "in_frustum": "_in_frustum",
    "offscreen": "_binary_offscreen",
}


@dataclass(frozen=True)
class VisibilityWindow:
    """Labels of all players at the requested frames, each ``[P, F]`` bool.

    Frames outside the recording are neither visible nor valid.

    Attributes:
        visible (np.ndarray): engine-confirmed line of sight (always within ``valid``).
        valid (np.ndarray): the label is defined.
        in_frustum (np.ndarray): the player projects into the view.
        offscreen (np.ndarray): the player projects outside the view.
    """

    visible: np.ndarray
    valid: np.ndarray
    in_frustum: np.ndarray
    offscreen: np.ndarray


class VisibilityLabels:
    """Reader of one visibility-label directory (one ``<media_id>.npz`` per recording)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"visibility label root not found: {self.root}")
        if next(self.root.glob("*.npz"), None) is None:
            raise FileNotFoundError(f"visibility label root holds no *.npz files: {self.root}")

    def path(self, media_id: str) -> Path:
        """``<root>/<media_id>.npz``."""
        return self.root / f"{media_id}.npz"

    def load(self, media_id: str) -> dict[str, np.ndarray]:
        """The four label arrays of one media (``visible``, ``eval_valid``, ``in_frustum``,
        ``offscreen``), ``[P, video_frames]`` bool each."""
        path = self.path(media_id)
        if not path.is_file():
            raise FileNotFoundError(f"no visibility labels for {media_id}: {path}")
        with np.load(path) as data:
            return {name: data[key] for name, key in _FIELDS.items()}

    def window(self, media_id: str, frame_indices: np.ndarray) -> VisibilityWindow:
        """Labels at source frames ``frame_indices`` (``[F]`` int)."""
        entry = self.load(media_id)
        frames = entry["visible"].shape[1]
        idx = np.asarray(frame_indices, dtype=np.int64)
        inside = (idx >= 0) & (idx < frames)
        safe = np.clip(idx, 0, max(frames - 1, 0))
        visible, valid, frustum, offscreen = (
            entry[name][:, safe] & inside[None, :] for name in _FIELDS
        )
        return VisibilityWindow(
            visible=visible & valid, valid=valid, in_frustum=frustum, offscreen=offscreen
        )


def observer_visibility_rows(
    labels: VisibilityLabels, media_id: str, source_frames: np.ndarray, *, num_players: int
) -> tuple[np.ndarray, np.ndarray]:
    """The recording player's view of every player at the window's pixel frames.

    Args:
        labels (VisibilityLabels): the label directory.
        media_id (str): the recording.
        source_frames (np.ndarray): ``[T]`` source frame of each pixel frame
            (:meth:`worldcast.data.player_frames.WindowSpec.source_frames`).
        num_players (int): players of the round (10).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``visible`` ``[P, T]`` float32 in {0, 1} and ``valid``
        ``[P, T]`` bool; outside ``valid`` the label is unknown, not "not visible".
    """
    window = labels.window(media_id, source_frames)
    if window.visible.shape[0] != int(num_players):
        raise ValueError(
            f"visibility labels of {media_id} hold {window.visible.shape[0]} players, expected"
            f" {num_players}"
        )
    return window.visible.astype(np.float32), window.valid.astype(np.bool_)
