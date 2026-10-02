"""Browser input ticks -> the per-frame controls of the next block, in the paper model's action encoding.

The browser samples keyboard and mouse once per display frame and sends each sample at once (a tick: held buttons,
degrees turned since the previous tick, weapon). The worker keeps the ticks that arrived since its previous block
and, when the engine is ready for the next block, folds them into ``F`` frames (:meth:`InputBuffer.cut`):

* ``latest`` (default): every frame holds the buttons held at the cut, frame 0 also every button tapped since the
  previous cut, and the turn is spread evenly over the block. An input shows from the block's first frame on.
* ``realtime``: each tick lands on the frame of its own 1/16 s (the most recent 1/16 s is the block's last frame),
  as in the recorded data. Timing inside a block is kept, at the price of about one more block of latency.
"""

import threading
from dataclasses import dataclass

import numpy as np

from demo.engine import BlockActions
from worldcast.data.actions import PAPER_ACTION_BUTTONS, quantize_camera_delta

NUM_BUTTONS = len(PAPER_ACTION_BUTTONS)
#: Engine ticks (64 Hz) per 16 fps frame: the substeps of the peer material.
SUBSTEPS = 4
#: Divisor of the substep turn deltas (config ``peer_camera_delta_scale``).
SUBSTEP_TURN_SCALE = 5.0
PITCH_LIMIT = 89.0


def buttons_from_mask(mask: int) -> np.ndarray:
    """``[11]`` float32 {0, 1} from a bitmask over ``PAPER_ACTION_BUTTONS`` (bit i = button i)."""
    return np.array([(int(mask) >> i) & 1 for i in range(NUM_BUTTONS)], dtype=np.float32)


@dataclass
class InputTick:
    seq: int
    arrival: float  # room seconds when the worker received it
    buttons: int  # bitmask over PAPER_ACTION_BUTTONS
    dpitch: float  # degrees turned since the previous tick (pitch > 0 looks down)
    dyaw: float  # (yaw > 0 turns left)
    weapon: int


class InputBuffer:
    """The ticks received since the previous block, and the state they leave behind (thread-safe: ticks arrive on
    the network thread, the engine thread cuts)."""

    def __init__(self, *, fps: float, mapping: str, weapon: int) -> None:
        self.fps, self.mapping = float(fps), str(mapping)
        self.ticks: list[InputTick] = []
        self.held = 0
        self.weapon = int(weapon)
        self.last_seq = -1
        self._lock = threading.Lock()

    def add(self, tick: InputTick) -> None:
        with self._lock:
            self.ticks.append(tick)

    def cut(self, *, now: float, block: int, first_frame: int, frames: int) -> BlockActions:
        """Fold the pending ticks into the ``frames`` frames of ``block``; its last frame ends at room time ``now``."""
        with self._lock:
            ticks, self.ticks = self.ticks, []
        f = int(frames)
        times = now - (f - 1 - np.arange(f)) / self.fps
        masks = np.full(f, self.held, np.int64)
        turn = np.zeros((f, 2), np.float32)
        weapon = np.full(f, self.weapon, np.int64)
        seq = np.full(f, self.last_seq, np.int64)
        if self.mapping == "realtime":
            frame_of = np.clip(
                np.searchsorted(times, [t.arrival for t in ticks], side="left"), 0, f - 1
            )
            for t, j in zip(ticks, frame_of):
                masks[j] |= t.buttons  # a tap inside the frame counts for the frame
                masks[j + 1 :] = t.buttons
                turn[j] += (t.dpitch, t.dyaw)
                weapon[j:], seq[j:] = t.weapon, t.seq
        elif ticks:
            masks[:] = ticks[-1].buttons
            for t in ticks:
                masks[0] |= t.buttons
                turn += np.array((t.dpitch, t.dyaw), np.float32) / f
            weapon[:], seq[:] = ticks[-1].weapon, ticks[-1].seq
        if ticks:
            self.held, self.weapon, self.last_seq = (
                int(ticks[-1].buttons),
                int(ticks[-1].weapon),
                int(ticks[-1].seq),
            )
        buttons = np.stack([buttons_from_mask(m) for m in masks])
        substeps = np.concatenate(
            [
                np.repeat(buttons[:, None], SUBSTEPS, 1),
                np.repeat(turn[:, None] / (SUBSTEPS * SUBSTEP_TURN_SCALE), SUBSTEPS, 1),
            ],
            -1,
        )
        camera = np.stack(
            [quantize_camera_delta(row, clip=False) for row in turn]
        )  # noclip, as deployed
        return BlockActions(
            block=int(block),
            first_frame=int(first_frame),
            times=times,
            buttons=buttons,
            turn=turn,
            camera=camera.astype(np.float32),
            weapon=weapon,
            substeps=substeps.astype(np.float32),
            substep_valid=np.ones((f, SUBSTEPS), np.bool_),
            input_seq=seq,
        )
