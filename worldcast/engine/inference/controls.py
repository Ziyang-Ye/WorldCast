"""A block's controls: the 16 rows (one per video frame) the generator reads for the block.

A block's 16 frames are denoised together, so all their controls must be known before its 4-step
denoising starts. A player's live input (:class:`LiveBlockControls`) also carries the turn its
camera integrates and when each frame was taken.
"""

from dataclasses import dataclass

import numpy as np

from worldcast.data.controls import CONTROL_BUTTONS, check_weapon_ids
from worldcast.data.latents import VIDEO_FRAMES_PER_BLOCK

__all__ = ["BlockControls", "LiveBlockControls"]


@dataclass
class BlockControls:
    """The client's controls for one block.

    Attributes:
        buttons (np.ndarray): ``[16, 11]`` float32, 0/1 held buttons in the order of
            :data:`worldcast.data.controls.CONTROL_BUTTONS`.
        view_deltas (np.ndarray): ``[16, 2]`` float32, the view controls: the pitch and yaw
            deltas of each frame as the generator reads them, mu-law encoded
            (:func:`worldcast.data.controls.quantize_camera_delta`, ``noclip``).
        weapon (np.ndarray): ``[16]`` int64, the held weapon's id in the 52-way vocabulary
            (:data:`worldcast.data.controls.OPENCS2_WEAPONS`).
    """

    buttons: np.ndarray
    view_deltas: np.ndarray
    weapon: np.ndarray

    def __post_init__(self) -> None:
        self.buttons = np.asarray(self.buttons, dtype=np.float32)
        self.view_deltas = np.asarray(self.view_deltas, dtype=np.float32)
        self.weapon = np.asarray(self.weapon, dtype=np.int64)
        n = int(self.buttons.shape[0])
        shapes = self.buttons.shape, self.view_deltas.shape, self.weapon.shape
        if self.buttons.ndim != 2 or shapes[1:] != ((n, 2), (n,)):
            raise ValueError(
                f"controls must be buttons [n, B], view_deltas [n, 2], weapon [n]; got {shapes}"
            )
        check_weapon_ids(self.weapon)

    def conditions(self) -> dict[str, np.ndarray]:
        """The block's rows of the generator's control conditions, under
        :data:`~worldcast.modeling.controls.CONTROL_KEYS` (spelled out here: a live client's
        controls load no torch)."""
        return {
            "buttons": self.buttons,
            "view_deltas": self.view_deltas,
            "weapon": self.weapon,
        }

    def check_block(self) -> None:
        """Raise unless these are the controls of one WorldCast block: 16 rows of 11 buttons."""
        expected = (VIDEO_FRAMES_PER_BLOCK, len(CONTROL_BUTTONS))
        if self.buttons.shape != expected:
            raise ValueError(
                f"a block's controls are {expected[0]} rows (one per video frame) of"
                f" {expected[1]} buttons; got {self.buttons.shape}"
            )


@dataclass
class LiveBlockControls(BlockControls):
    """The client's controls for one block from a player's live input
    (``player_state.source: live``): the rows of :class:`BlockControls`, with the turn and the
    time of each frame.

    Attributes:
        turn (np.ndarray): ``[16, 2]`` float32, degrees turned during each frame (pitch, yaw),
            raw. ``view_deltas`` is its mu-law code unless the input path fed the generator another
            yaw (the demo's ``hybrid`` mapping, ``worldcast/demo/inputs.py``); the client's camera
            integrates this turn.
        times (np.ndarray): ``[16]`` float64, the session time of each frame: s on the clock all
            clients of the session share (the live demo's room clock).
    """

    turn: np.ndarray
    times: np.ndarray

    def __post_init__(self) -> None:
        super().__post_init__()
        self.turn = np.asarray(self.turn, dtype=np.float32)
        self.times = np.asarray(self.times, dtype=np.float64)
        n = int(self.buttons.shape[0])
        if self.turn.shape != (n, 2) or self.times.shape != (n,):
            raise ValueError(
                f"live controls of {n} rows have turn [{n}, 2] and times [{n}]; got"
                f" {self.turn.shape} and {self.times.shape}"
            )
