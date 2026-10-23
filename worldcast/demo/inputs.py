"""From a browser's input samples to a block's controls.

The browser samples keyboard, mouse, gamepad and touch pad at its display's frames, at most 60 a
second, and sends each sample at once (held buttons, degrees turned since the previous sample,
weapon); a button or the weapon changing is sent at once besides. A worker keeps the samples that
arrived since its previous block and, when the client is ready for the next block, folds them into
the block's controls (:meth:`InputBuffer.cut`):

* ``latest``: every frame holds the buttons held at the cut, frame 0 also every button tapped
  since the previous cut, and the turn is spread evenly over the block. An input shows from the
  block's first frame on.
* ``front``: as ``latest``, but the turn is done in the block's first :data:`FRONT_FRAMES`
  frames, or in as many more as keep each frame's turn, as the generator is fed it, within its
  trained range (:data:`~worldcast.data.controls.CAMERA_MAX_DEGREES`): a flick is done at the
  block's start instead of at its end. What the block's frames cannot hold within the range (a
  flick of more than 320 degrees, or a refresh's near the block's end) is carried to the next
  cut.
* ``hybrid``: chosen for each block by the turn it holds, the larger of its pitch and
  yaw: ``front`` for a block that turns less than :data:`HYBRID_DEGREES`, ``latest`` for one that
  turns that much or more, so that a small correction acts and a flick or a sweep turns smoothly.
  The block's turn is all in its own block: none is held back for a later one. The generator is fed
  :data:`HYBRID_YAW_GAIN` times the block's yaw; the turn, the camera and
  :attr:`~worldcast.demo.clients.LiveControls.turned` are the turn the player made.
* ``realtime``: each sample lands on the frame shown while it was taken (the most recent frame
  period is the block's last frame), as in the recorded data. Timing inside a block is kept, at the
  price of about one more block of latency.

The inspect button (``look_at_weapon``) reaches the generator only in the frames that hold a knife
(:data:`INSPECT_WEAPONS`): a cut clears it from every other row, whatever the page sends.

Every sample's turn is folded into some frame. The controls say how far the player has turned in
the session through each frame (:attr:`~worldcast.demo.clients.LiveControls.turned`), which the
page compares with the turn it has sent to show each frame turned by the turn it does not contain
yet (docs/demo.md, "Latency").

The page sends samples while it plays. When none has arrived for
``release_after_s`` (a hidden page, a lost link), a cut releases the held buttons instead of holding
them for every block after.

With ``play.refresh`` a worker cuts the block on screen again when the input changes
(:meth:`InputBuffer.changed`, :meth:`InputBuffer.recut`): the rows already on their way to the
browser keep their controls, the later ones take the samples since the last cut, and the block's
turn is the turn of every sample folded into it. With ``front`` (the demo's mapping with refresh)
the turn a recut gives the rows from its switch on is done in their first latent frame, its first
:data:`FRONT_FRAMES` rows, or in as many more as keep each row within the trained range, so that a
turn shows a refresh after it is made instead of being spread over the rest of the block.
"""

import math
import threading
from dataclasses import dataclass

import numpy as np

from worldcast.data.controls import (
    CAMERA_MAX_DEGREES,
    CONTROL_BUTTONS,
    OPENCS2_WEAPONS,
    buttons_from_bits,
    encode_turn,
)
from worldcast.data.latents import VIDEO_FRAMES_PER_LATENT

from .clients import LiveControls

__all__ = [
    "FRONT_FRAMES",
    "GATHER_DEGREES",
    "GATHER_S",
    "HYBRID_DEGREES",
    "HYBRID_FRONT_FRAMES",
    "HYBRID_YAW_GAIN",
    "INSPECT_WEAPONS",
    "InputBuffer",
    "InputSample",
]

#: ``front`` does a block's turn, and a refresh's from its switch, in this many frames at least:
#: two, so that a small turn is done a frame after the switch, in two steps rather than one.
FRONT_FRAMES = 2
#: ``hybrid`` does a block that turns little in its first latent frame, this many frames (as
#: measured, :data:`HYBRID_DEGREES`).
HYBRID_FRONT_FRAMES = VIDEO_FRAMES_PER_LATENT
#: With ``play.refresh``, a refresh asked for by a turn that starts or turns back waits until
#: ``gather_s`` (``play.gather_ms``, by default GATHER_S) after the turn's first sample arrived, as
#: a flick of the mouse comes in, unless the turn has reached GATHER_DEGREES (a large flick goes at
#: once): more of a flick goes into the refresh, rather than its first sample into one and the rest
#: into the next, a refresh later (:meth:`InputBuffer.gathering`).
GATHER_S = 0.02
GATHER_DEGREES = 20.0

#: ``hybrid`` cuts a block that turns less than this many degrees (the larger of its pitch and yaw)
#: as ``front`` does and one that turns this much or more as ``latest`` does. Measured on an H200:
#: ``latest`` turns the view not at all under 14-16 degrees a block, ``front`` follows up to 20
#: (0.7-0.9 of the turn) and falls at 40 (0.71).
HYBRID_DEGREES = 20.0

#: ``hybrid`` feeds the generator this times a block's yaw, which the view it generates turns
#: short of (the camera and the page keep the turn made). Single H200 readings of the view's turn
#: over the player's yaw with it: 0.72-0.98 at 5-20 degrees a block and 0.72-0.92 at 30-160,
#: against 0.71-0.84 and 0.70-0.85 without it; the same controls read up to 0.14 apart in two jobs.
#: The gain takes the rows of a block of 291 to 320 degrees of yaw past the generator's trained 20
#: degrees a frame, which the turn made stays within; above 320 the rows pass it with no gain too.
HYBRID_YAW_GAIN = 1.1

#: The weapon ids with which the inspect button (``look_at_weapon``) reaches the generator: the
#: knives, every one of :data:`~worldcast.data.controls.OPENCS2_WEAPONS` (the names of a knife start
#: with ``knife``, but for the bayonet; the aliases of a knife are read as one of them). The
#: generator learned the inspect of a gun badly.
INSPECT_WEAPONS = frozenset(
    i for i, name in enumerate(OPENCS2_WEAPONS) if name.startswith("knife") or name == "bayonet"
)
_INSPECT_BIT = 1 << CONTROL_BUTTONS.index("look_at_weapon")


@dataclass
class InputSample:
    """One input sample of the browser, sent at its display's frames, at most 60 a second, and
    when a button or the weapon changes."""

    seq: int
    arrival: float  # room seconds when the worker received it
    buttons: int  # bitmask over CONTROL_BUTTONS
    dpitch: float  # degrees turned since the previous sample (pitch > 0 looks down)
    dyaw: float  # (yaw > 0 turns left)
    weapon: int


@dataclass
class _Rows:
    """A block's rows as they are cut: room times, button masks (int64 bits), the turn as made
    (float64 degrees), weapon ids and the newest input sample of each."""

    times: np.ndarray
    masks: np.ndarray
    turn: np.ndarray
    weapon: np.ndarray
    seq: np.ndarray


@dataclass
class _Cut:
    """The block cut last: its controls, its rows as cut, and the turn the cuts had folded in
    before it."""

    controls: LiveControls
    rows: _Rows
    before: np.ndarray


def _inspect_with_knives(masks: np.ndarray, weapon: np.ndarray) -> np.ndarray:
    """``masks`` ``[F]`` with the inspect bit cleared in the rows whose weapon ``[F]`` is no
    knife."""
    knife = np.isin(weapon, list(INSPECT_WEAPONS))
    return np.where(knife, masks, masks & ~_INSPECT_BIT)


def _direction(sample: InputSample) -> np.ndarray:
    """``[2]`` the sign of a sample's turn (pitch, yaw): -1, 0 or 1."""
    return np.sign([sample.dpitch, sample.dyaw])


def _turn_of(samples: list[InputSample]) -> np.ndarray:
    """``[2]`` float64 the turn of ``samples`` (pitch, yaw), degrees."""
    if not samples:
        return np.zeros(2)
    return np.array([(s.dpitch, s.dyaw) for s in samples]).sum(axis=0)


def _front_rows(
    turn: np.ndarray, gains: np.ndarray, frames: int, least: int
) -> tuple[int, np.ndarray]:
    """How ``front`` does a turn ``[2]`` (as made; ``gains`` ``[2]``, what the generator is fed of
    it): over ``least`` frames, or as many more as keep each frame's turn as fed within the trained
    range on both axes, at most ``frames``, which then take the share of it they hold within the
    range. Returns the frames and the turn they take, ``[2]``."""
    needed = float(np.abs(turn * gains).max()) / CAMERA_MAX_DEGREES
    spread = min(int(frames), max(least, math.ceil(needed)))
    return spread, (turn if needed <= spread else turn * (spread / needed))


class InputBuffer:
    """The samples received since the previous block, and the state they leave behind.

    Thread-safe: samples arrive on the network thread, the client thread cuts.

    Args:
        fps (float): the frames shown per second (``play.fps``).
        mapping (str): ``latest``, ``front``, ``hybrid`` or ``realtime`` (the module docstring).
        weapon (int): the weapon id held before the first sample.
        release_after_s (float): a cut later than this after the newest sample releases the held
            buttons (``play.release_after_s``).
        yaw_gain (float): the generator is fed this times the block's yaw, with any mapping (and
            with ``hybrid``'s :data:`HYBRID_YAW_GAIN` on top); the turn, the camera and the page's
            count stay the turn made (``play.yaw_gain``).
        pitch_gain (float): likewise for the pitch (``play.pitch_gain``).
        gather_s (float): how long a refresh for a turn that starts waits for more of it
            (``play.gather_ms``, :meth:`gathering`), s.

    Attributes:
        received (np.ndarray): ``[2]`` float64, the turn of every sample received in the session
            (pitch, yaw), degrees.
        turned (np.ndarray): ``[2]`` float64, the turn the cuts have folded into the controls so
            far: :attr:`received` but for the samples that wait for the next cut and
            :attr:`carry`.
        carry (np.ndarray): ``[2]`` float64, the turn cut that its block's frames did not hold
            within the generator's trained range (``front``), which the next cut takes, degrees.
        direction (np.ndarray): ``[2]`` the sign (-1, 0 or 1) of the turn of the newest sample the
            cuts have folded in (pitch, yaw): the turn rate's, for :meth:`changed`.
    """

    def __init__(
        self,
        *,
        fps: float,
        mapping: str,
        weapon: int,
        release_after_s: float,
        yaw_gain: float = 1.0,
        pitch_gain: float = 1.0,
        gather_s: float = GATHER_S,
    ) -> None:
        self.fps, self.mapping = float(fps), str(mapping)
        self.gather_s = float(gather_s)
        self.yaw_gain, self.pitch_gain = float(yaw_gain), float(pitch_gain)
        hybrid = HYBRID_YAW_GAIN if self.mapping == "hybrid" else 1.0
        self.gains = np.array([self.pitch_gain, self.yaw_gain * hybrid])  # fed per turn made
        self._least = HYBRID_FRONT_FRAMES if self.mapping == "hybrid" else FRONT_FRAMES
        self.release_after_s = float(release_after_s)
        self.samples: list[InputSample] = []
        self.held, self.weapon, self.last_seq = 0, int(weapon), -1
        self.last_arrival = -math.inf  # when the newest sample arrived, room seconds
        self.received, self.turned = np.zeros(2), np.zeros(2)
        self.carry = np.zeros(2)
        self.direction = np.zeros(2)
        self._lock = threading.Lock()
        self._cut: _Cut | None = None

    def add(self, sample: InputSample) -> None:
        """Queue a sample for the next cut."""
        with self._lock:
            self.samples.append(sample)
            self.received += (sample.dpitch, sample.dyaw)

    def cut(self, *, now: float, frames: int) -> LiveControls:
        """Fold the pending samples into a block's controls.

        Args:
            now (float): room seconds of the block's last frame.
            frames (int): frames of the block.

        Returns:
            LiveControls: one row per frame; without a pending sample, what the last one left held,
            or nothing once it is older than ``release_after_s``.
        """
        samples = self._take(now)
        f = int(frames)
        rows = _Rows(
            times=now - (f - 1 - np.arange(f)) / self.fps,
            masks=np.full(f, self.held, np.int64),
            turn=np.zeros((f, 2)),
            weapon=np.full(f, self.weapon, np.int64),
            seq=np.full(f, self.last_seq, np.int64),
        )
        self._fold(samples, rows, 0, taps=0, turn=_turn_of(samples))
        return self._controls(rows, 0, self.turned.copy(), None)

    def changed(self, threshold_deg: float) -> bool:
        """Whether the samples since the last cut or recut ask for the block to be cut again (a
        refresh): a button pressed or released or the weapon changed against what the last cut left
        held (the inspect button only with a knife), a turn of at least ``threshold_deg`` degrees on
        either axis, or a turn rate whose sign changed on either axis: the newest sample turns
        another way than the newest one the cuts folded in (:attr:`direction`), a turn that starts,
        stops or turns back."""
        with self._lock:
            samples = list(self.samples)
        if not samples:
            return False
        if self._buttons_changed(samples):
            return True
        if (_direction(samples[-1]) != self.direction).any():
            return True
        return float(np.abs(_turn_of(samples)).max()) >= float(threshold_deg)

    def gathering(self, now: float) -> float:
        """How long a refresh that :meth:`changed` asks for is to wait for more of a turn, s (0:
        none).

        A turn that starts or turns back (its first sample turns another way, on either axis, than
        the newest one the cuts folded in, :attr:`direction`) waits until ``gather_s`` after that
        first sample arrived, unless it has turned :data:`GATHER_DEGREES` on either axis by
        now. A button or the weapon changing, and a turn that goes on or stops, wait for nothing.

        Args:
            now (float): room seconds now.
        """
        with self._lock:
            samples = list(self.samples)
        if not samples or self._buttons_changed(samples):
            return 0.0
        starts = [
            i
            for i, s in enumerate(samples)
            if ((_direction(s) != 0) & (_direction(s) != self.direction)).any()
        ]
        if not starts:
            return 0.0
        first = starts[0]
        if float(np.abs(_turn_of(samples[first:])).max()) >= GATHER_DEGREES:
            return 0.0
        return max(0.0, samples[first].arrival + self.gather_s - float(now))

    def _buttons_changed(self, samples: list[InputSample]) -> bool:
        """Whether a sample holds other buttons than the last cut left held (the inspect button
        only with a knife), or another weapon."""
        held = _inspect_with_knives(np.int64(self.held), np.int64(self.weapon))
        for s in samples:
            if s.weapon != self.weapon:
                return True
            if _inspect_with_knives(np.int64(s.buttons), np.int64(s.weapon)) != held:
                return True
        return False

    def recut(self, *, now: float, frames: int, from_row: int) -> LiveControls:
        """Fold the samples since the last cut into the block cut last, from row ``from_row`` on
        (a refresh of the block on screen).

        Rows before ``from_row`` keep the controls of the last cut (they are on their way to the
        browser). From it on the rows hold the buttons held now, with every button tapped since
        the last cut in row ``from_row``, the newest weapon and input sample, and the turn the
        last cut left for them plus the turn of the new samples, spread over them as the mapping
        spreads a block's turn (``realtime``: each new sample on its frame, the rows before
        ``from_row`` left alone). The block keeps its times, and its turn is the turn of every
        sample folded into it, so :attr:`turned` stays the turn the cuts have folded in.

        Args:
            now (float): room seconds now.
            frames (int): frames of the block.
            from_row (int): the first row to change; one past the block's last changes its last
                (the samples are folded in all the same).

        Returns:
            LiveControls: the block's controls again.

        Raises:
            RuntimeError: no block was cut.
            ValueError: ``frames`` is not the block's.
        """
        if self._cut is None:
            raise RuntimeError("a recut changes the block cut last; no block was cut")
        cut, f = self._cut, int(frames)
        if f != cut.controls.num_frames:
            raise ValueError(f"the block cut last has {cut.controls.num_frames} rows, not {f}")
        r = min(max(int(from_row), 0), f - 1)
        samples = self._take(now)
        rows = _Rows(
            times=cut.rows.times,
            masks=cut.rows.masks.copy(),
            turn=cut.rows.turn.copy(),
            weapon=cut.rows.weapon.copy(),
            seq=cut.rows.seq.copy(),
        )
        kept = rows.turn[r:].sum(axis=0)
        taps = rows.masks[0] if r == 0 else 0  # the taps the cut put in the block's first row
        self._fold(samples, rows, r, taps=taps, turn=kept + _turn_of(samples) if samples else kept)
        return self._controls(rows, r, cut.before, cut.controls)

    def _take(self, now: float) -> list[InputSample]:
        """The samples since the last cut, taken; with none for ``release_after_s`` at room
        seconds ``now``, the held buttons are released."""
        with self._lock:
            samples, self.samples = self.samples, []
        if not samples and now - self.last_arrival > self.release_after_s:
            self.held = 0
        return samples

    def _fold(
        self, samples: list[InputSample], rows: _Rows, row: int, *, taps: int, turn: np.ndarray
    ) -> None:
        """Fold ``samples`` into ``rows`` from ``row`` on, in place, as the mapping does, and keep
        what the newest leaves held: ``realtime`` puts each on its frame; the others put the
        buttons held now in every row, with ``taps`` and every button pressed since in row
        ``row``, the newest weapon and sample, and spread ``turn`` (``[2]``, as made) with
        :attr:`carry` over the rows."""
        f = len(rows.times)
        if self.mapping == "realtime":
            frame_of = np.clip(
                np.searchsorted(rows.times, [s.arrival for s in samples], side="left"), row, f - 1
            )
            for s, j in zip(samples, frame_of):
                rows.masks[j] |= s.buttons  # a tap inside the frame counts for the frame
                rows.masks[j + 1 :] = s.buttons
                rows.turn[j] += (s.dpitch, s.dyaw)
                rows.weapon[j:], rows.seq[j:] = s.weapon, s.seq
        else:
            rows.masks[row:] = samples[-1].buttons if samples else self.held
            rows.masks[row] |= taps
            for s in samples:
                rows.masks[row] |= s.buttons
            if samples:
                rows.weapon[row:], rows.seq[row:] = samples[-1].weapon, samples[-1].seq
            if self.carry.any():
                turn = turn + self.carry
            concentrate = self.mapping == "front" or (
                self.mapping == "hybrid" and float(np.abs(turn).max()) < HYBRID_DEGREES
            )
            left = f - row
            spread, placed = (
                _front_rows(turn, self.gains, left, self._least) if concentrate else (left, turn)
            )
            rows.turn[row:] = 0.0
            rows.turn[row : row + spread] = placed / spread
            self.carry = turn - placed
        if samples:
            last = samples[-1]
            self.held, self.weapon, self.last_seq = int(last.buttons), int(last.weapon), last.seq
            self.last_arrival = last.arrival
            self.direction = _direction(last)

    def _controls(
        self, rows: _Rows, row: int, before: np.ndarray, kept: LiveControls | None
    ) -> LiveControls:
        """The controls of a block whose ``rows`` were folded from ``row`` on, the rows before it
        those of ``kept``, remembered as the block cut last; ``before`` is the turn the cuts had
        folded in before the block."""
        rows.masks = _inspect_with_knives(rows.masks, rows.weapon)
        turned = (before if row == 0 else kept.turned[row - 1]) + np.cumsum(rows.turn[row:], 0)
        view_turn = rows.turn[row:] * self.gains if (self.gains != 1.0).any() else rows.turn[row:]
        view_deltas = np.stack([encode_turn(turn) for turn in view_turn])
        if row:
            turned = np.concatenate([kept.turned[:row], turned])
            view_deltas = np.concatenate([kept.view_deltas[:row], view_deltas])
        self.turned = turned[-1]
        controls = LiveControls(
            buttons=buttons_from_bits(rows.masks),
            view_deltas=view_deltas,
            weapon=rows.weapon,
            times=rows.times,
            turn=rows.turn,
            input_seq=rows.seq,
            turned=turned,
        )
        self._cut = _Cut(controls, rows, before)
        return controls
