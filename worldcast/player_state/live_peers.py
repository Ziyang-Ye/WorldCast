"""The other players under live controls: from the rows their own cameras used.

In live play every client publishes, per block, the rows its camera follows at the block's video
frames, with each frame's session time and the block's controls (the ``live`` part). A client
draws another live player, a peer, at its own video frames' session times, never by block index
(:class:`LivePeer`): between received rows it interpolates them; past the newest it extends them
with the physics prior of the peer's last keys, held for :data:`PEER_HOLD_ROWS` rows and then
released, frozen after :data:`PEER_HORIZON_ROWS`, integrated only as far as the times drawn need;
after :data:`PEER_HIDE_S` of silence the peer is hidden. A newer part that moves or turns the peer
is blended in over the block, however far it turns, so that the peer moves on the client's view
instead of vanishing for a block; only a block whose rows jump farther than the peer runs while it
is drawn past its newest row (beyond :data:`PEER_SNAP_U`) is hidden instead of drawn. A late,
lost, duplicated or reordered part never raises.

Only a player a human plays is in the world: a player is drawn from its first part on, and nothing
is drawn for a seat nobody plays. Every player alive at the round's first frame is alive on every
video frame (:func:`keep_alive`), so no player is ever drawn as a corpse and a human can take any
seat.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from worldcast.data.controls import CAMERA_DELTA_SCALE, CONTROL_BUTTONS, SUBSTEPS_PER_VIDEO_FRAME
from worldcast.data.game import PITCH_MAX_DEGREES, PITCH_MIN_DEGREES
from worldcast.data.latents import FPS, VIDEO_FRAMES_PER_BLOCK, VIDEO_FRAMES_PER_LATENT
from worldcast.data.recordings import ALIVE_INDEX

from .extrapolation import JUMP_COLUMN, PhysicsPrior, PriorState, physics_prior_displacement

__all__ = [
    "PEER_BLEND",
    "PEER_HIDE_S",
    "PEER_HOLD_ROWS",
    "PEER_HORIZON_ROWS",
    "PEER_KEPT_ROWS",
    "PEER_SNAP_U",
    "WARM_ROWS",
    "LivePeer",
    "PeerBlock",
    "PeerRows",
    "block_substeps",
    "hold_others",
    "keep_alive",
    "packed_substeps",
    "prior_displacements",
]

#: Rows the physics prior runs over before the rows it extends: started at rest that far back, its
#: velocity, key hold and flight are the continuous integral's to a fraction of a unit.
WARM_ROWS = 32
#: Rows past a peer's newest row that hold its last keys (but jump, a tap); later rows hold no key,
#: and the prior coasts to rest.
PEER_HOLD_ROWS = 8
#: Rows past a peer's newest row after which it stands frozen.
PEER_HORIZON_ROWS = 32
#: Session time past a peer's newest row after which it is hidden, s. A longer gap between two
#: received rows ends the track before it: the rows before the gap are dropped.
PEER_HIDE_S = 3.0
#: A newer part that moves a peer at the block's first video frame by more than this in xy, u, is
#: not blended: its rows jumped, and the block is hidden. A smaller offset, and any turn, is
#: blended. A peer is blended only when the block before drew it at its last video frame, at most
#: :data:`PEER_HIDE_S` past its newest row: the newer rows, extended, have it run at most that long
#: and a video frame from that row, and the extension the block before was drawn from adds its
#: held keys and coasting (about half a second of running): less than 4 s of running at the
#: physics prior's top speed in all, which this is a little over.
PEER_SNAP_U = 1024.0
#: The share of a newer part's offset the block's four latent frames keep; the video frames
#: between them keep a linear share, from all of it at the video frame before the block.
PEER_BLEND = (0.75, 0.5, 0.25, 0.0)
#: Received rows kept per peer: the prior's warm rows and one block more.
PEER_KEPT_ROWS = WARM_ROWS + VIDEO_FRAMES_PER_BLOCK

#: The weight of the offset at each video frame of a block (:data:`PEER_BLEND`).
_BLEND_WEIGHTS = np.interp(
    np.arange(VIDEO_FRAMES_PER_BLOCK),
    np.arange(-1, VIDEO_FRAMES_PER_BLOCK, VIDEO_FRAMES_PER_LATENT),
    (1.0, *PEER_BLEND),
)


def _wrap(degrees: np.ndarray) -> np.ndarray:
    """:func:`~worldcast.player_state.states.wrap_degrees` of float64 degrees, in numpy (the
    remainder is exact in either)."""
    return np.remainder(np.asarray(degrees, np.float64) + 180.0, 360.0) - 180.0


def _extension_buttons(newest: np.ndarray) -> np.ndarray:
    """``[PEER_HORIZON_ROWS + 1, 11]`` the buttons from a peer's newest row on: that row's, then
    held for :data:`PEER_HOLD_ROWS` rows without jump, then none."""
    buttons = np.zeros((PEER_HORIZON_ROWS + 1, len(CONTROL_BUTTONS)), np.float32)
    buttons[: PEER_HOLD_ROWS + 1] = newest
    buttons[1 : PEER_HOLD_ROWS + 1, JUMP_COLUMN] = 0.0
    return buttons


def block_substeps(buttons: np.ndarray, turn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A player's ordered substeps of video frames from the buttons held and the turn of each.

    Each frame's buttons are held on its four substeps and its turn is split evenly over them,
    divided by :data:`~worldcast.data.controls.CAMERA_DELTA_SCALE`: the layout of
    :func:`~worldcast.data.controls.align_ticks_to_ordered_substeps` with one tick per substep,
    which the physics prior and the camera integrals read.

    Args:
        buttons (np.ndarray): ``[n, 11]`` 0/1 held buttons in the order of
            :data:`~worldcast.data.controls.CONTROL_BUTTONS`.
        turn (np.ndarray): ``[n, 2]`` degrees turned during each frame (pitch, yaw).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``values`` ``[n, 4, 13]`` float32 (the buttons, then the
        pitch and yaw turns divided by 5) and ``valid`` ``[n, 4]`` bool, all True.
    """
    buttons = np.asarray(buttons, np.float32)
    turn = np.asarray(turn, np.float32)
    n = len(buttons)
    if buttons.shape != (n, len(CONTROL_BUTTONS)) or turn.shape != (n, 2):
        raise ValueError(
            f"buttons are [n, {len(CONTROL_BUTTONS)}] and turn [n, 2]; got {buttons.shape} and"
            f" {turn.shape}"
        )
    values = np.empty((n, SUBSTEPS_PER_VIDEO_FRAME, len(CONTROL_BUTTONS) + 2), np.float32)
    values[..., :-2] = buttons[:, None]
    values[..., -2:] = (turn / np.float32(SUBSTEPS_PER_VIDEO_FRAME * CAMERA_DELTA_SCALE))[:, None]
    return values, np.ones((n, SUBSTEPS_PER_VIDEO_FRAME), bool)


def packed_substeps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """:func:`~worldcast.player_state.states.pack_substeps` in numpy: the substeps ``[..., Q, A]``
    (:func:`block_substeps`) zeroed where invalid, with their valid flags ``[..., Q]`` appended:
    ``[..., Q, A + 1]`` float32."""
    flag = np.asarray(valid, bool)[..., None].astype(np.float32)
    return np.concatenate([np.asarray(values, np.float32) * flag, flag], -1)


def prior_displacements(
    inputs: Sequence[tuple[torch.Tensor, torch.Tensor, PriorState | None]], *, prior: PhysicsPrior
) -> list[tuple[torch.Tensor, PriorState]]:
    """:func:`~worldcast.player_state.extrapolation.physics_prior_displacement` of several players
    in one call, which costs about what the longest one costs alone.

    A shorter input is padded in front with invalid substeps, which keep the integrator as it is:
    each player's result is the one its input gives alone.

    Args:
        inputs (Sequence[tuple[torch.Tensor, torch.Tensor, PriorState | None]]): per player, the
            packed substeps ``[T_i, Q, A + 1]`` (valid flag last), the yaw ``[T_i]`` (degrees) and
            the integrator before them (``[1]`` each; None: at rest).
        prior (PhysicsPrior): the fitted prior.

    Returns:
        list[tuple[torch.Tensor, PriorState]]: per player, the displacement during each of its
        video frames ``[T_i, 3]`` float32 (u) and the integrator after them.
    """
    longest = max(int(controls.shape[0]) for controls, _, _ in inputs)
    controls = np.zeros((len(inputs), longest, *inputs[0][0].shape[1:]), np.float32)
    yaw = np.zeros((len(inputs), longest), np.float32)
    states = [PriorState.at_rest(1) if state is None else state for _, _, state in inputs]
    for i, (rows, degrees, _) in enumerate(inputs):
        controls[i, longest - int(rows.shape[0]) :] = np.asarray(rows)
        yaw[i, longest - int(rows.shape[0]) :] = np.asarray(degrees)
    before = PriorState(
        *(
            torch.from_numpy(np.concatenate([np.asarray(getattr(state, name)) for state in states]))
            for name in ("velocity", "hold", "flight")
        )
    )
    moved, after = physics_prior_displacement(
        torch.from_numpy(controls), torch.from_numpy(yaw), prior=prior, state=before
    )
    moved, velocity, hold, flight = (
        np.asarray(value) for value in (moved, after.velocity, after.hold, after.flight)
    )
    return [
        (
            torch.from_numpy(moved[i, longest - int(rows.shape[0]) :]),
            PriorState(*(torch.from_numpy(value[i : i + 1]) for value in (velocity, hold, flight))),
        )
        for i, (rows, _, _) in enumerate(inputs)
    ]


def keep_alive(states: torch.Tensor) -> None:
    """Every player alive at video frame 0 is made alive on every video frame of ``states``, in
    place: no player is drawn as a corpse, and a human can take any seat without the player coming
    back to life (one life per round). A player dead or absent from frame 0 keeps its rows.

    Args:
        states (torch.Tensor): ``[1, P, T, 6]`` the round's player states, rewritten in place.
    """
    alive = states[0, :, 0, ALIVE_INDEX] > 0.5
    states[0, alive, :, ALIVE_INDEX] = 1.0


def hold_others(batch: dict[str, torch.Tensor], client_slot: int) -> None:
    """Every player but the client stands at its video frame 0 on every video frame of ``batch``,
    in place, holding nothing: position, angles, no control substep (flags cleared) and the weapon
    of frame 0. A player is in the world only from its first ``live`` part on, and until then
    nothing a block reads of it may depend on the round's recorded rows: a round that slides keeps
    the rows of its last block for the frames it vacates, so a recorded row would freeze where a
    longer round would follow the recording.

    Args:
        batch (dict[str, torch.Tensor]): the round batch (batch size 1): ``player_states``,
            ``player_control_substeps``, ``player_control_substep_valid`` and
            ``player_weapon_ids``, rewritten in place.
        client_slot (int): the client's slot, whose rows are the live loop's.
    """
    others = [p for p in range(int(batch["player_states"].shape[1])) if p != int(client_slot)]
    batch["player_states"][0, others, 1:, :ALIVE_INDEX] = batch["player_states"][
        0, others, :1, :ALIVE_INDEX
    ]
    batch["player_control_substeps"][0, others, 1:] = 0.0
    batch["player_control_substep_valid"][0, others, 1:] = False
    batch["player_weapon_ids"][0, others, 1:] = batch["player_weapon_ids"][0, others, :1]


@dataclass
class PeerRows:
    """The rows a peer published for one block (its ``live`` part), as a client reads them.

    Attributes:
        times (np.ndarray): ``[n]`` float64, ``n >= 1``, the increasing session time of each video
            frame, s.
        rows (np.ndarray): ``[n, 5]`` float64 x, y, z (u), yaw, pitch (degrees): the player state
            the peer's camera followed.
        buttons (np.ndarray): ``[n, 11]`` float32 0/1 held buttons in the order of
            :data:`~worldcast.data.controls.CONTROL_BUTTONS`.
        weapon (np.ndarray): ``[n]`` int64, the held weapon's id.
    """

    times: np.ndarray
    rows: np.ndarray
    buttons: np.ndarray
    weapon: np.ndarray

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, np.float64)
        self.rows = np.asarray(self.rows, np.float64)
        self.buttons = np.asarray(self.buttons, np.float32)
        self.weapon = np.asarray(self.weapon, np.int64)
        n = int(self.times.shape[0]) if self.times.ndim == 1 else -1
        shapes = self.times.shape, self.rows.shape, self.buttons.shape, self.weapon.shape
        if (
            shapes != ((n,), (n, 5), (n, len(CONTROL_BUTTONS)), (n,))
            or n < 1
            or (np.diff(self.times) <= 0).any()
        ):
            raise ValueError(
                f"a peer's rows are increasing times [n >= 1], rows [n, 5], buttons"
                f" [n, {len(CONTROL_BUTTONS)}] and weapon [n]; got {shapes}"
            )


@dataclass(frozen=True)
class PeerBlock:
    """A peer at the 16 video frames of a client's block, as the client writes it into its player
    state table.

    Attributes:
        rows (np.ndarray): ``[16, 5]`` float64 x, y, z (u), yaw, pitch (degrees); the pitch
            within ``[-89, 89]`` (:data:`~worldcast.data.game.PITCH_MAX_DEGREES`),
            where the 13-wide camera integral clamps.
        buttons (np.ndarray): ``[16, 11]`` float32 0/1 held buttons.
        weapon (np.ndarray): ``[16]`` int64, the held weapon's id.
        values (np.ndarray): ``[16, 4, 13]`` float32 ordered substeps (:func:`block_substeps`):
            the buttons, and as turns the differences of consecutive rows' yaw and pitch from the
            ``previous`` facing of :meth:`LivePeer.at` on, so that a camera integral that holds
            ``previous`` at the video frame before the block, the 6-wide and the 13-wide player
            state table's alike, reproduces the rows' facing (to float32 rounding).
        valid (np.ndarray): ``[16, 4]`` bool, all True.
        present (np.ndarray): ``[16]`` bool, whether the peer is drawn at the video frame: False
            past :data:`PEER_HIDE_S` of silence, before its first row, on a snapped block and while
            its player is done.
        snapped (bool): the rows jump at the block from what the client showed (an offset beyond
            :data:`PEER_SNAP_U` in xy), so the block is hidden instead of blended.
    """

    rows: np.ndarray
    buttons: np.ndarray
    weapon: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    present: np.ndarray
    snapped: bool


class _Extension:
    """The physics prior's extension past a peer's newest row: over its last :data:`WARM_ROWS`
    received rows from rest, then over :data:`PEER_HORIZON_ROWS` rows of its held keys
    (:func:`_extension_buttons`), integrated row by row only as far as asked. A later ask goes on
    from the integrator where the last one stopped, which gives the rows of one call over them."""

    def __init__(self, buttons: np.ndarray, rows: np.ndarray) -> None:
        self._warm = min(WARM_ROWS, len(rows))
        held = np.concatenate([buttons[-self._warm :], _extension_buttons(buttons[-1])[1:]])
        yaw = np.concatenate([rows[-self._warm :, 3], np.full(PEER_HORIZON_ROWS, rows[-1, 3])])
        values, valid = block_substeps(held, np.zeros((len(held), 2), np.float32))
        self._controls = packed_substeps(values, valid)
        self._yaw = yaw.astype(np.float32)
        self._state: PriorState | None = None  # the integrator after the rows integrated
        self._asked = 0
        self.moved = np.zeros((1, 3))  # [n + 1, 3] the displacement from the newest row, u

    def prior_input(self, rows: int) -> tuple[torch.Tensor, torch.Tensor, PriorState | None] | None:
        """The prior's input that integrates the extension up to ``rows`` rows past the newest
        (at most :data:`PEER_HORIZON_ROWS`), or None if it is integrated that far."""
        done = len(self.moved) - 1
        self._asked = min(int(rows), PEER_HORIZON_ROWS)
        if self._asked <= done:
            return None
        first = 0 if self._state is None else self._warm + done
        stop = self._warm + self._asked
        controls, yaw = self._controls[first:stop], self._yaw[first:stop]
        return torch.from_numpy(controls), torch.from_numpy(yaw), self._state

    def take(self, displacement: torch.Tensor, state: PriorState) -> None:
        """Take the prior's result over the latest :meth:`prior_input`: the displacement during
        each of its rows ``[n, 3]`` (u) and the integrator after them."""
        done = len(self.moved) - 1
        expected = self._asked - done + (self._warm if self._state is None else 0)
        if self._asked <= done or int(displacement.shape[0]) != expected:
            raise ValueError(
                f"a peer's extension takes the displacement of its latest prior input ({expected}"
                f" rows); got {int(displacement.shape[0])} rows"
            )
        after = np.asarray(displacement, np.float64)[expected - (self._asked - done) :]
        moved = np.cumsum(np.concatenate([self.moved[-1:], after]), 0)
        self.moved = np.concatenate([self.moved, moved[1:]])
        self._state = state


@dataclass(frozen=True)
class _Timeline:
    """A peer's received rows and the prior's extension past the newest: its rows at any session
    time the extension is integrated for (:meth:`rows_past`)."""

    times: np.ndarray
    rows: np.ndarray
    buttons: np.ndarray
    weapon: np.ndarray
    period: float  # session time between the peer's video frames, s
    extension: _Extension
    extension_buttons: np.ndarray  # [PEER_HORIZON_ROWS + 1, 11] (_extension_buttons)

    def rows_past(self, time: float) -> int:
        """The rows of the extension past the newest row that the session times up to ``time``
        need."""
        offset = (time - self.times[-1]) / self.period
        return int(np.clip(np.ceil(offset), 0, PEER_HORIZON_ROWS))

    def at(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(rows [n, 5], buttons [n, 11], weapon [n], present [n])`` at the session ``times``."""
        n = len(times)
        newest = len(self.times) - 1
        rows = np.repeat(self.rows[:1], n, 0)
        buttons = np.repeat(self.buttons[:1], n, 0)
        weapon = np.repeat(self.weapon[:1], n, 0)
        present = np.zeros(n, bool)
        at_or_before = np.searchsorted(self.times, times, side="right") - 1

        inside = (at_or_before >= 0) & (at_or_before < newest)
        i = at_or_before[inside]
        share = (times[inside] - self.times[i]) / (self.times[i + 1] - self.times[i])
        a, b = self.rows[i], self.rows[i + 1]
        mixed = a + (b - a) * share[:, None]
        mixed[:, 3] = _wrap(a[:, 3] + _wrap(b[:, 3] - a[:, 3]) * share)
        rows[inside], buttons[inside], weapon[inside] = mixed, self.buttons[i], self.weapon[i]
        present[inside] = True

        past = at_or_before == newest
        after = times[past] - self.times[newest]
        offset = after / self.period
        moved = self.extension.moved
        steps = np.arange(len(moved))
        rows[past] = self.rows[newest]
        rows[past, :3] += np.stack([np.interp(offset, steps, axis) for axis in moved.T], -1)
        buttons[past] = self.extension_buttons[
            np.minimum(np.floor(offset).astype(np.int64), PEER_HORIZON_ROWS)
        ]
        weapon[past] = self.weapon[newest]
        present[past] = after <= PEER_HIDE_S
        return rows, buttons, weapon, present


class LivePeer:
    """Another live player, drawn where its own camera was (live play, not the paper's).

    A client feeds it the rows the peer published (:meth:`update`) and asks for the peer at the
    session times of its own block's video frames (:meth:`at`): provisionally when it prepares the
    block, finally at the controls cut. Rows of a block are written once and never again, so a part
    that arrives later moves the peer from the next block on, blended:

    - a part is newer than another when its last session time is later. Parts overlap in session
      time when the peer's controls cuts come less than 15 video frames apart; a newer part's rows
      then replace an older part's from the newer part's first time on, so the rows never step back;
    - a time between two received rows interpolates them: x, y, z and pitch linearly, yaw along the
      shortest arc; buttons and weapon of the row at or before it;
    - a time past the newest row is the newest row moved by the physics prior over the peer's last
      :data:`WARM_ROWS` rows and an extension at the peer's frame period: its last keys held for
      :data:`PEER_HOLD_ROWS` rows (but jump), then none; no turn; frozen after
      :data:`PEER_HORIZON_ROWS`; hidden past :data:`PEER_HIDE_S`;
    - a time before the first row holds it, hidden;
    - where the previous final block showed the peer at its last video frame and newer rows move
      the peer by ``delta`` at the block's first video frame, against the rows that block was drawn
      from, the block keeps ``delta`` times :data:`PEER_BLEND` at its latent frames, however far
      it turns (yaw along the shortest arc): the peer moves and turns on the client's view instead
      of vanishing for a block. Beyond :data:`PEER_SNAP_U` in xy it snaps: the rows jump and the
      block is hidden;
    - where the previous final block hid the peer at its last video frame (silence, a snap, its
      player done), or the peer is new, nothing drawn is jumped from: the block is drawn at once,
      unblended. A player who takes the seat of one that left is therefore drawn where it is from
      its first block on, not slid there from where the leaver stood.

    The prior's extension is integrated once per newer row, and only as far as the times drawn
    need: up to the row of the block's last time (none while the received rows reach it), on from
    where it stopped when later times need more. :meth:`prior_inputs` and :meth:`extend` let a
    client integrate several players in one :func:`prior_displacements` call; :meth:`at`
    integrates what is missing alone otherwise.

    Args:
        prior (PhysicsPrior): the fitted prior.
    """

    def __init__(self, prior: PhysicsPrior) -> None:
        self.prior = prior
        self._parts: dict[float, PeerRows] = {}  # by last session time, the parts whose rows count
        self._times = np.zeros(0)
        self._rows = np.zeros((0, 5))
        self._buttons = np.zeros((0, len(CONTROL_BUTTONS)), np.float32)
        self._weapon = np.zeros(0, np.int64)
        self._extension: _Extension | None = None
        self._pending: list[_Extension] = []  # those whose prior input is handed out
        self._timeline: _Timeline | None = None
        self._drawn: _Timeline | None = None  # what the last final block drew at its last frame

    def update(self, parts: Iterable[PeerRows]) -> None:
        """Add the rows of received parts, in any order and any number of times.

        A part is known by its last session time: a copy adds nothing. A newer part's rows
        replace an older part's from its first time on (:class:`LivePeer`). A gap longer than
        :data:`PEER_HIDE_S` ends the track: the rows before it are dropped. The newest
        :data:`PEER_KEPT_ROWS` rows are kept.

        Args:
            parts (Iterable[PeerRows]): the rows of the parts.
        """
        held = dict(self._parts)
        for part in parts:
            held.setdefault(float(part.times[-1]), part)
        if len(held) == len(self._parts):
            return
        newest_first = sorted(held.values(), key=lambda part: float(part.times[-1]), reverse=True)
        kept, before = [], np.inf
        for part in newest_first:
            kept.append(part.times < before)
            before = min(before, float(part.times[0]))
        oldest_first = list(zip(reversed(newest_first), reversed(kept)))
        times = np.concatenate([part.times[keep] for part, keep in oldest_first])
        owner = np.concatenate(
            [np.full(int(keep.sum()), i) for i, (_, keep) in enumerate(oldest_first)]
        )
        gaps = np.flatnonzero(np.diff(times) > PEER_HIDE_S)
        first = max(int(gaps[-1]) + 1 if len(gaps) else 0, len(times) - PEER_KEPT_ROWS)
        self._parts = {
            float(oldest_first[i][0].times[-1]): oldest_first[i][0]
            for i in np.unique(owner[first:])
        }
        tail = [array[-WARM_ROWS:] for array in (self._times, self._rows, self._buttons)]
        self._times = times[first:]
        self._rows = np.concatenate([part.rows[keep] for part, keep in oldest_first])[first:]
        self._buttons = np.concatenate([part.buttons[keep] for part, keep in oldest_first])[first:]
        self._weapon = np.concatenate([part.weapon[keep] for part, keep in oldest_first])[first:]
        now = (self._times[-WARM_ROWS:], self._rows[-WARM_ROWS:], self._buttons[-WARM_ROWS:])
        if not all(np.array_equal(old, new) for old, new in zip(tail, now)):
            self._extension = None
        self._timeline = None

    def prior_inputs(
        self, times: np.ndarray
    ) -> list[tuple[torch.Tensor, torch.Tensor, PriorState | None]]:
        """The physics prior's inputs that drawing the peer at the session ``times`` ``[16]``
        (:meth:`at`) still needs: the extension past its newest row up to the row of the last
        time, and the extension of the rows the previous final block was drawn from up to the row
        of the first (the blend), where not integrated yet. Hand their results to :meth:`extend`.

        Each input is the packed substeps ``[T, 4, 14]``, the yaw ``[T]`` and the integrator before
        them (None: at rest).

        Raises:
            ValueError: no row has arrived.
        """
        timeline = self._current()
        asked = [(timeline.extension, timeline.rows_past(times[-1]))]
        if self._drawn is not None:
            drawn = self._drawn.extension, self._drawn.rows_past(times[0])
            if drawn[0] is timeline.extension:
                asked = [(drawn[0], max(asked[0][1], drawn[1]))]
            else:
                asked.append(drawn)
        inputs, self._pending = [], []
        for extension, rows in asked:
            given = extension.prior_input(rows)
            if given is not None:
                inputs.append(given)
                self._pending.append(extension)
        return inputs

    def extend(self, results: Sequence[tuple[torch.Tensor, PriorState]]) -> None:
        """Take the prior's results over the latest :meth:`prior_inputs`, in their order: the
        displacement during each row ``[T, 3]`` (u) and the integrator after them.

        Raises:
            ValueError: not one result per input, or a displacement not its input's length.
        """
        if len(results) != len(self._pending):
            raise ValueError(
                f"a peer takes one result per input of its latest prior_inputs"
                f" ({len(self._pending)}); got {len(results)}"
            )
        for extension, (displacement, state) in zip(self._pending, results):
            extension.take(displacement, state)
        self._pending = []

    def _current(self) -> _Timeline:
        if not len(self._times):
            raise ValueError("a peer is drawn from the rows it published; none has arrived")
        if self._extension is None:
            self._extension = _Extension(self._buttons, self._rows)
        if self._timeline is None:
            gaps = np.diff(self._times)
            self._timeline = _Timeline(
                times=self._times,
                rows=self._rows,
                buttons=self._buttons,
                weapon=self._weapon,
                period=float(np.median(gaps)) if len(gaps) else 1.0 / FPS,
                extension=self._extension,
                extension_buttons=_extension_buttons(self._buttons[-1]),
            )
        return self._timeline

    def at(
        self, times: np.ndarray, previous: np.ndarray, *, final: bool, hidden: bool = False
    ) -> PeerBlock:
        """The peer at the video frames of one of the client's blocks.

        Args:
            times (np.ndarray): ``[16]`` the session time of the block's video frames, s: the
                client's own (predicted before the controls cut), increasing, and after the
                previous final block's. A client's own cut times overlap like a peer's parts when
                its cuts come less than 15 video frames apart; it passes them made monotone, or
                the peer steps back.
            previous (np.ndarray): ``[2]`` the peer's yaw and pitch at the video frame before the
                block (the row of the latent frame before it), degrees, as the camera integral
                the client's player state field reads holds them there (the continuous columns of
                that latent frame), not the stored row: the block's first turn starts there, so
                the integral reproduces the rows from the block's first video frame on, whatever an
                earlier block or the round's recorded controls left between them.
            final (bool): the rows the block is denoised with: the next block is blended from
                them. A provisional call changes nothing.
            hidden (bool): the peer's player is done: the block draws nobody, and the next one
                blends from nothing.

        Returns:
            PeerBlock: the rows, the controls and whether the peer is drawn.

        Raises:
            ValueError: ``times`` is not ``[16]``, or no row has arrived.
        """
        times = np.asarray(times, np.float64)
        if times.shape != (VIDEO_FRAMES_PER_BLOCK,):
            raise ValueError(f"a block has {VIDEO_FRAMES_PER_BLOCK} times; got {times.shape}")
        inputs = self.prior_inputs(times)
        if inputs:
            self.extend(prior_displacements(inputs, prior=self.prior))
        timeline = self._current()
        rows, buttons, weapon, present = timeline.at(times)
        snapped = False
        if self._drawn is not None:
            delta = rows[0] - self._drawn.at(times[:1])[0][0]
            delta[3] = _wrap(delta[3])
            snapped = bool(np.hypot(delta[0], delta[1]) > PEER_SNAP_U)
            if not snapped:
                rows = rows - delta * _BLEND_WEIGHTS[:, None]
                rows[:, 3] = _wrap(rows[:, 3])
        rows[:, 4] = np.clip(rows[:, 4], PITCH_MIN_DEGREES, PITCH_MAX_DEGREES)
        facing = np.concatenate([np.asarray(previous, np.float64).reshape(1, 2), rows[:, 3:5]])
        turn = np.stack([np.diff(facing[:, 1]), _wrap(np.diff(facing[:, 0]))], -1)
        values, valid = block_substeps(buttons, turn)
        present = present & (not snapped) & (not hidden)
        if final:
            self._drawn = timeline if present[-1] else None
        return PeerBlock(
            rows=rows,
            buttons=buttons,
            weapon=weapon,
            values=values,
            valid=valid,
            present=present,
            snapped=snapped,
        )
