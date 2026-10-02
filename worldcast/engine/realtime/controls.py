"""A block's controls, and how live input becomes them.

The generator reads 16 rows of controls per block (one per video frame): 11 held buttons, the mu-law encoded turn
``[pitch, yaw]`` of the frame (``noclip``, :func:`worldcast.data.actions.quantize_camera_delta`) and the weapon id.
A block's 16 frames are generated at once, so their controls must be known before the 4-step ladder starts.
:class:`LiveControls` samples the input at that moment and fills the block by holding it: the buttons held now are
held for the 16 frames; the turn made since the previous block is applied over the first latent frame (four
frames) or, with ``mouse="hold"``, the current turn rate is held for the whole block.
"""

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from worldcast.data.actions import PAPER_ACTION_BUTTONS, encode_weapon, quantize_camera_delta

__all__ = ["FRAMES_PER_BLOCK", "BlockControls", "LiveControls"]

#: Video frames per 4-latent block.
FRAMES_PER_BLOCK = 16


@dataclass
class BlockControls:
    """The observer's controls for one block: ``buttons`` ``[16, 11]`` 0/1 in the config's button order,
    ``camera`` ``[16, 2]`` encoded turn (pitch, yaw), ``weapon`` ``[16]`` weapon ids in [0, 52)."""

    buttons: np.ndarray
    camera: np.ndarray
    weapon: np.ndarray

    def __post_init__(self) -> None:
        self.buttons = np.asarray(self.buttons, dtype=np.float32)
        self.camera = np.asarray(self.camera, dtype=np.float32)
        self.weapon = np.asarray(self.weapon, dtype=np.int64)
        n = int(self.buttons.shape[0])
        if self.buttons.ndim != 2 or self.camera.shape != (n, 2) or self.weapon.shape != (n,):
            raise ValueError(
                f"controls must be buttons [n, B], camera [n, 2], weapon [n]; got "
                f"{self.buttons.shape}, {self.camera.shape}, {self.weapon.shape}"
            )
        if self.weapon.size and (self.weapon.min() < 0 or self.weapon.max() >= 52):
            raise ValueError("weapon ids must lie in [0, 52)")

    @classmethod
    def from_raw(
        cls,
        buttons: np.ndarray,
        turn_degrees: np.ndarray,
        weapon: Sequence,
        *,
        camera_encoding: str = "noclip",
    ) -> "BlockControls":
        """From per-frame held buttons, per-frame turns ``[n, 2]`` (pitch, yaw, degrees) and weapon names or ids."""
        clip = camera_encoding == "clip"
        camera = np.stack(
            [quantize_camera_delta(t, clip=clip) for t in np.asarray(turn_degrees, np.float32)]
        )
        ids = [w if isinstance(w, (int, np.integer)) else encode_weapon(w) for w in weapon]
        return cls(buttons=buttons, camera=camera, weapon=np.asarray(ids, dtype=np.int64))


class LiveControls:
    """Thread-safe live input; :meth:`sample` turns it into the next block's :class:`BlockControls`.

    Args:
        button_names: the config's button order (``data.action_buttons``).
        weapon: the initial weapon id.
        mouse: ``accumulate`` (the turn since the last sample, over the first ``spread`` frames) or ``hold``
            (the last turn rate, held for every frame).
        spread: frames the accumulated turn is spread over (4 = one latent frame).
    """

    def __init__(
        self,
        button_names: Sequence[str] = PAPER_ACTION_BUTTONS,
        *,
        weapon: int = 0,
        mouse: str = "accumulate",
        spread: int = 4,
        camera_encoding: str = "noclip",
    ) -> None:
        if mouse not in ("accumulate", "hold"):
            raise ValueError("mouse must be 'accumulate' or 'hold'")
        self.names = tuple(button_names)
        self.index: dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self.mouse, self.spread, self.camera_encoding = mouse, int(spread), camera_encoding
        self._lock = threading.Lock()
        self._held = np.zeros(len(self.names), dtype=np.float32)
        self._pressed_since = np.zeros(len(self.names), dtype=np.float32)
        self._turn = np.zeros(2, dtype=np.float64)  # unapplied (pitch, yaw), degrees
        self._rate = np.zeros(2, dtype=np.float64)  # degrees per frame, for mouse='hold'
        self._weapon = int(weapon)
        self._last_sample = time.monotonic()

    def press(self, name: str, down: bool = True) -> None:
        with self._lock:
            i = self.index[name]
            self._held[i] = 1.0 if down else 0.0
            if down:
                self._pressed_since[i] = 1.0  # a tap between two samples still reaches a block

    def release(self, name: str) -> None:
        self.press(name, down=False)

    def turn(
        self, pitch_degrees: float, yaw_degrees: float, *, frame_seconds: float = 1.0 / 16
    ) -> None:
        with self._lock:
            self._turn += (float(pitch_degrees), float(yaw_degrees))
            now = time.monotonic()
            dt = max(now - self._last_sample, frame_seconds)
            self._rate = self._turn * frame_seconds / dt

    def set_weapon(self, weapon_id: int) -> None:
        with self._lock:
            self._weapon = int(weapon_id)

    def sample(self, frames: int = FRAMES_PER_BLOCK) -> BlockControls:
        """The next block's controls from the input as it is now; resets the accumulated turn and taps."""
        with self._lock:
            buttons = np.repeat(np.maximum(self._held, self._pressed_since)[None], frames, axis=0)
            self._pressed_since[:] = 0.0
            turns = np.zeros((frames, 2), dtype=np.float32)
            if self.mouse == "hold":
                turns[:] = self._rate.astype(np.float32)
            else:
                n = max(1, min(self.spread, frames))
                turns[:n] = (self._turn / n).astype(np.float32)
            self._turn[:] = 0.0
            self._last_sample = time.monotonic()
            weapon = np.full(frames, self._weapon, dtype=np.int64)
        return BlockControls.from_raw(buttons, turns, weapon, camera_encoding=self.camera_encoding)
