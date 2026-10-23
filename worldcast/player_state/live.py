"""Live play (not the paper's): a client's player states when its player's controls are live
(``player_state.source: live``, the web demo).

The client's own camera follows the live controls. Over block ``s`` it is the own position at
latent frame ``s - 1``, read off the generated video by the state model and guarded against jumps
(:class:`~worldcast.player_state.live_position.GuardedPosition`, not Eq. (4)), plus the physics
prior's displacement of the block's keys, stopped at the walls in its way (and, with
``live.wall_keys``, the keys cleared from the row it is stopped at); its view angles integrate the
live turns. A block's own rows are written when it is prepared, with the last
controls held, and again at its controls cut, right before its denoising; those are the rows it
is denoised with, and they are published as the block's ``live`` part. Only players a human plays
are in the world: another player is drawn at the rows its own camera used, from its first part on
(:class:`~worldcast.player_state.live_peers.LivePeer`), and the round's recorded players are
nowhere, in the first frame as in every other, for a seat whose first frame the dataset's labels
show empty (the demo's library offers only such seats, and its client refuses another: the first
six blocks read those labels of latent frame 0, and a window labels the first frame against a
world without the recorded players). :class:`LiveLoop` takes the closed loop's place in the block
loop (:class:`~worldcast.player_state.closed_loop.ClosedLoop`).
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import torch

from worldcast.data.camera import (
    NEAR_U,
    SCOPE_ZOOM_FOV,
    c2w_from_state_rows,
    camera_to_world,
    half_angle_tangents,
)
from worldcast.data.controls import (
    CAMERA_DELTA_SCALE,
    CONTROL_BUTTONS,
    DEFAULT_CAMERA_ENCODING,
    OPENCS2_WEAPONS,
    STATE_MODEL_SUBSTEPS,
    SUBSTEPS_PER_VIDEO_FRAME,
    buttons_from_bits,
    count_until,
    encode_turn,
    state_model_controls,
)
from worldcast.data.game import PITCH_MAX_DEGREES, PITCH_MIN_DEGREES, TICK_RATE
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.latents import (
    BLOCK,
    LATENT_GRID,
    PIXELS_PER_CELL,
    SOURCE_FRAMES_PER_LATENT,
    VIDEO_FRAMES_PER_BLOCK,
    VIDEO_FRAMES_PER_LATENT,
    last_video_frame,
    source_frame,
    video_frames_of,
)
from worldcast.data.memory_frames import MemoryBlock
from worldcast.data.recordings import ALIVE_INDEX, TickTable
from worldcast.modeling.depth_head import DepthFn
from worldcast.modeling.state_model import (
    DISPLACEMENT_UNIT_U,
    STATE_MODEL_WINDOW_LATENT_FRAMES,
    STATE_MODEL_WINDOW_STRIDE,
    StateModel,
)
from worldcast.sampling.window import KEEP_LATENTS, KEEP_VIDEO, roll

from . import predicted_visibility
from .closed_loop import ClosedLoop, StateReader, window_of
from .extrapolation import (
    MOVEMENT_COLUMNS,
    ClientCameras,
    Motion,
    PhysicsPrior,
    PositionTrack,
    PredictedStateTable,
    PriorState,
    physics_prior_displacement,
)
from .live_peers import (
    PEER_KEPT_ROWS,
    LivePeer,
    PeerRows,
    block_substeps,
    hold_others,
    keep_alive,
    packed_substeps,
    prior_displacements,
)
from .live_position import IDLE_STEP_U, MOVED_SHARE, GuardedPosition, Walkable, wall_clearance
from .predicted_visibility import MARGIN_U, PLAYER_LIFT_U, POSE_RADIUS_U, PredictedVisibility
from .states import (
    SUBSTEP_PITCH_INDEX,
    SUBSTEP_VALID_INDEX,
    SUBSTEP_YAW_INDEX,
    integrate_camera_angles,
    pack_substeps,
)

__all__ = [
    "CLEAR_BLOCKS",
    "CONTINUATION_DEGREES",
    "HOP_WEIGHTS",
    "LIVE_BATCH_KEYS",
    "LIVE_PARTS_READ",
    "LiveCameras",
    "LiveExchange",
    "LiveLoop",
    "LiveMotion",
    "LivePart",
    "LiveStateReader",
    "LiveVisibility",
    "WALL_PASSED_U",
    "ZBUFFERS_KEPT",
    "live_observer_signals",
]

#: The ``live`` parts of another player read per block: as many as its rows a client keeps.
LIVE_PARTS_READ = -(-PEER_KEPT_ROWS // VIDEO_FRAMES_PER_BLOCK)
#: The entries of the round batch that live play writes (:meth:`LiveLoop.build` takes them as its
#: own): every player's rows, controls and weapons, the field of view and the observer signals.
LIVE_BATCH_KEYS = (
    "player_states",
    "player_control_substeps",
    "player_control_substep_valid",
    "player_weapon_ids",
    "window_tans",
    *OBSERVER_SIGNAL_KEYS,
)
#: A block keeps the slowdown of a wall the block before ran into (blocked continuation) while it
#: holds the same movement keys and turns less than this from that block's last row, degrees.
CONTINUATION_DEGREES = 15.0
#: ``live.wall_keys: guard`` clears a player's movement keys for at most this many blocks in a row
#: (a block is a second); the block after them passes its keys, so that the generator shows whether
#: the wall is there (a stop the depth head makes where the picture has no wall ends in one probe:
#: the view moves and the motion head concedes). Three: a wall met takes a block to be found and a
#: block to show standing, and a player who pushes on for three seconds is asking.
CLEAR_BLOCKS = 3
#: ``live.hop: spread`` lets the camera of a block start where the block before it ended and run
#: into the position estimate over the block: the weight of the hop (the camera's end less the
#: estimate) left at each of the block's 16 video frames, linear from nearly all to none at the
#: last.
HOP_WEIGHTS = 1.0 - np.arange(1, VIDEO_FRAMES_PER_BLOCK + 1) / VIDEO_FRAMES_PER_BLOCK
#: The reprojections of a block's context a client keeps, each for one set of the target frames'
#: cameras: the prepared block's and those of the cuts after it (a cut can be made again).
ZBUFFERS_KEPT = 4
#: A wall a block stopped before is forgotten once a block starts, or the place head (followed,
#: inside its gate) reads the player, more than this past the stop (xy, u): the player is not at
#: it.
WALL_PASSED_U = 64.0


class LivePart(Protocol):
    """One block of a player's rows as another client reads them: its ``live`` part
    (:class:`~worldcast.engine.inference.messages.MessagePart`)."""

    f0: int
    times: np.ndarray  # [16] float64 session times, s
    rows: np.ndarray  # [16, 5] float32 x, y, z, yaw, pitch
    buttons: np.ndarray  # [16] uint16 bitmasks over CONTROL_BUTTONS
    turn: np.ndarray  # [16, 2] float32 degrees, pitch then yaw
    weapon: np.ndarray  # [16] uint8 weapon ids


class LiveExchange(Protocol):
    """Where the clients publish their rows in live play and read each other's: the shared world
    state over messages (:class:`~worldcast.engine.inference.messages.MessageWorldState`)."""

    def refresh(self) -> None:
        """Take in what the other clients have published since the last call."""
        ...

    def publish_live_state(
        self,
        f0: int,
        *,
        times: np.ndarray,
        rows: np.ndarray,
        buttons: np.ndarray,
        turn: np.ndarray,
        weapon: np.ndarray,
    ) -> None:
        """Publish this client's rows of block ``f0`` with the block's controls (its ``live``
        part)."""
        ...

    def live_part(self, media_id: str, f0: int) -> LivePart | None:
        """The ``live`` part that keys ``media_id``'s memory entry of block ``f0``, or None."""
        ...

    def recent_live(self, media_id: str, n: int = 2) -> list[LivePart]:
        """The newest ``n`` ``live`` parts of ``media_id`` by session time, oldest first."""
        ...

    def done_status(self, media_id: str) -> str | None:
        """None while ``media_id`` runs, else its done status."""
        ...


def live_observer_signals(weapon_ids: np.ndarray) -> dict[str, np.ndarray]:
    """The observer signals of latent frames under live controls.

    Nothing flashes and the scope is off; while a weapon that zooms is held
    (:data:`~worldcast.data.camera.SCOPE_ZOOM_FOV`) the scope is unknown instead, the learned
    "unknown": live play does not track the zoom, so the field of view stays unscoped.

    Args:
        weapon_ids (np.ndarray): ``[n]`` the weapon held at each latent frame's last video frame.

    Returns:
        dict[str, np.ndarray]: :data:`~worldcast.data.labels.OBSERVER_SIGNAL_KEYS` -> ``[n]``
        int64.
    """
    weapon_ids = np.asarray(weapon_ids, np.int64).reshape(-1)
    zooms = np.array([OPENCS2_WEAPONS[w] in SCOPE_ZOOM_FOV for w in weapon_ids], bool)
    off = np.zeros(len(weapon_ids), np.int64)
    return {
        "obs_flash_flag": off,
        "obs_flash_valid": off + 1,
        "obs_scope_on": off,
        "obs_scope_level": off,
        "obs_scope_valid": (~zooms).astype(np.int64),
    }


def _unscoped_tans(n: int) -> torch.Tensor:
    """``[n, 2]`` float32: the unscoped ``(tan(hfov / 2), tan(vfov / 2))``."""
    return torch.tensor(half_angle_tangents(), dtype=torch.float32).expand(n, 2).clone()


def _view_angles(
    yaw: float | np.ndarray, pitch: float | np.ndarray, packed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The yaw and pitch ``[..., n + 1]`` float32 of players facing ``yaw``, ``pitch`` ``[...]``
    (degrees) that turn as the packed substeps ``[..., n, 4, A + 1]`` of ``n`` rows say: the start,
    then the angles after each row; :func:`~worldcast.player_state.states.integrate_camera_angles`
    operation by operation, each player's alone (its sums are torch's, the rest numpy)."""
    packed = np.asarray(packed, np.float32)
    rows = np.concatenate([np.zeros_like(packed[..., :1, :, :]), packed], -3)
    valid = (rows[..., SUBSTEP_VALID_INDEX] > 0.5).astype(np.float32)
    turn_pitch, turn_yaw = (
        torch.from_numpy(rows[..., index] * valid).sum(-1).numpy() * CAMERA_DELTA_SCALE
        for index in (SUBSTEP_PITCH_INDEX, SUBSTEP_YAW_INDEX)
    )
    turn_pitch[..., 0] = turn_yaw[..., 0] = 0.0
    turned = np.asarray(yaw, np.float32)[..., None] + torch.from_numpy(turn_yaw).cumsum(-1).numpy()
    yaws = np.remainder(turned + np.float32(180.0), np.float32(360.0)) - np.float32(180.0)
    low, high = np.float32(PITCH_MIN_DEGREES), np.float32(PITCH_MAX_DEGREES)
    pitches = np.empty_like(yaws)
    current = np.minimum(np.maximum(np.asarray(pitch, np.float32), low), high)
    for row in range(turn_pitch.shape[-1]):
        current = pitches[..., row] = np.minimum(
            np.maximum(current + turn_pitch[..., row], low), high
        )
    return yaws, pitches


def _copied(value: Any) -> Any:
    """A copy of a piece of state that later writes cannot reach: arrays and tensors copied, dicts
    and lists one level deep, anything else as it is (replaced, never written into)."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _copied(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copied(item) for item in value]
    return value


def _saved(obj: Any, skip: Sequence[str] = ()) -> dict[str, Any]:
    """The attributes of ``obj`` but ``skip``, :func:`_copied`."""
    return {name: _copied(value) for name, value in vars(obj).items() if name not in skip}


def _restore(obj: Any, saved: Mapping[str, Any]) -> None:
    """Give ``obj`` the attributes of :func:`_saved` again; an array or tensor of the same shape
    and dtype is written in place, as others may hold it (the round batch's tables)."""
    for name, value in saved.items():
        current = getattr(obj, name, None)
        same = (
            type(current) is type(value)
            and getattr(current, "shape", None) == getattr(value, "shape", None)
            and getattr(current, "dtype", None) == getattr(value, "dtype", None)
        )
        if same and torch.is_tensor(value):
            current.copy_(value)
        elif same and isinstance(value, np.ndarray):
            np.copyto(current, value)
        else:
            setattr(obj, name, _copied(value))


def _apply_stops(moved: np.ndarray, stops: Sequence[tuple[np.ndarray, float]]) -> list[int | None]:
    """Cut the cumulative displacement ``moved`` ``[n, 3]`` of a block's rows (u) at ``stops``, in
    turn and in place: beyond its clearance along its direction a row's displacement stops along
    it. Returns the first row each stop cut, or None."""
    rows = []
    for direction, clearance in stops:
        beyond = np.maximum(moved[:, :2] @ direction - clearance, 0.0)
        cut = np.flatnonzero(beyond > 0.0)
        rows.append(int(cut[0]) if len(cut) else None)
        moved[:, :2] -= beyond[:, None] * direction[None]
    return rows


# ------------------------------------------------------------------------------------- the z-test
# The labels a live client draws at its cut are those of :func:`~worldcast.player_state.
# predicted_visibility.ztest_labels`, operation by operation, with a player's nine test points at
# once: each projection's einsum is torch's (one call per test point, as there), the rest numpy
# (comparisons, floors, gathers, a sort of four values and float64 arithmetic in the same order).
#: The eight more test points around a player, as ztest_labels adds them: the circle of
#: :data:`~worldcast.player_state.predicted_visibility.POSE_RADIUS_U`.
_POSE_OFFSETS = tuple(
    (POSE_RADIUS_U * math.cos(k * math.pi / 4.0), POSE_RADIUS_U * math.sin(k * math.pi / 4.0), 0.0)
    for k in range(8)
)


def _ztest(
    points: np.ndarray,
    c2w: np.ndarray,
    depth: np.ndarray,
    live: np.ndarray,
    tans: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """``(visible, valid)`` ``[R, P]`` of the players ``live`` ``[R, P]`` whose test points are
    ``points`` ``[R, P, 3]`` (u) and the eight around each, against the depth ``[R, H, W]`` (NaN:
    no surface) seen from ``c2w`` ``[R, 4, 4]`` (float64)."""
    h, w = LATENT_GRID
    cell = PIXELS_PER_CELL
    tan_h, tan_v = float(tans[0]), float(tans[1])
    tested = [points] + [points + np.array(offset) for offset in _POSE_OFFSETS]
    relative = np.stack(tested) - c2w[None, :, None, :3, 3]
    rotation = torch.from_numpy(c2w[:, :3, :3])
    cam = np.stack(
        [torch.einsum("rmk,rkj->rmj", torch.from_numpy(r), rotation).numpy() for r in relative]
    )
    z = cam[..., 2]
    zs = np.where(np.abs(z) > 1e-9, z, 1e-9)
    fc = (cam[..., 0] / (zs * tan_h) + 1.0) * (w / 2.0) * cell
    fr = (cam[..., 1] / (zs * tan_v) + 1.0) * (h / 2.0) * cell
    in_view = (
        (z > NEAR_U) & (fc >= 0.5) & (fc <= w * cell - 0.5) & (fr >= 0.5) & (fr <= h * cell - 0.5)
    )
    cx = np.clip(np.floor(fc), 0, w * cell - 1).astype(np.int64) // cell
    cy = np.clip(np.floor(fr), 0, h * cell - 1).astype(np.int64) // cell
    # the median of the 2 x 2 cells (cy-1..cy) x (cx-1..cx), clamped at 0, that hold a depth
    rows = np.stack([np.maximum(cy - 1, 0), cy], -1)
    cols = np.stack([np.maximum(cx - 1, 0), cx], -1)
    flat = (rows[..., :, None] * w + cols[..., None, :]).reshape(*cx.shape, 4)
    values = depth.reshape(len(depth), -1)[np.arange(len(depth))[None, :, None, None], flat]
    known = ~np.isnan(values)
    n = known.sum(-1)
    ordered = np.sort(np.where(known, values, np.inf), -1)
    lo, hi = np.maximum(n - 1, 0) // 2, np.minimum(n // 2, 3)
    median = (
        0.5
        * (
            np.take_along_axis(ordered, lo[..., None], -1)
            + np.take_along_axis(ordered, hi[..., None], -1)
        )[..., 0]
    )
    visible = (live & in_view & (n > 0) & (median - z > -MARGIN_U)).any(0)
    return visible, (live & ~(in_view[0] & (n[0] == 0) & live)) | visible


def _folded(visible: np.ndarray, first: int) -> np.ndarray:
    """:func:`~worldcast.player_state.visibility.fold_video_frames` of a block's labels ``[16, P]``
    from latent frame ``first`` on: ``[4, P]`` bool."""
    sizes = [len(video_frames_of(f)) for f in range(int(first), int(first) + BLOCK)]
    return np.stack([part.any(0) for part in np.split(visible, np.cumsum(sizes)[:-1])])


# ------------------------------------------------------------------------------------- the motion
class LiveMotion(Motion):
    """A player's :class:`~worldcast.player_state.extrapolation.Motion` written block by block from
    its live controls.

    A block's rows are written provisionally before its controls are cut, then finally with them;
    only final rows are history. Their view angles integrate their turns from the angles of the row
    before them (:func:`~worldcast.player_state.states.integrate_camera_angles`). Their
    displacement is the physics prior's
    (:func:`~worldcast.player_state.extrapolation.physics_prior_displacement`), continued from its
    integrator after the newest final row (at rest before the substeps of video frame 0): the
    integral of the round's keys. The prior's input and output pass through :meth:`prior_input`
    and :meth:`write`, so that a client integrates several players in one call
    (:func:`~worldcast.player_state.live_peers.prior_displacements`). Rows after the newest written
    hold its values.

    Attributes:
        prior_displacement (np.ndarray): ``[T, 3]`` float64 the prior's displacement from video
            frame 0, u. :attr:`displacement`, which the camera follows, is it stopped where a
            block met a wall in its way, with a wall's slowdown kept over a block (:meth:`write`).
        prior_state (PriorState): the prior's integrator after the newest final row.
        final_row (int): the newest final row.
        version (int): counts the writes.

    Args:
        start_state (np.ndarray): ``[6]`` the round-start player-state row.
        start_substeps (torch.Tensor): ``[4, A + 1]`` the packed substeps of video frame 0.
        video_frames (int): video frames ``T`` of the round.
        prior (PhysicsPrior): the physics prior.
        spread (bool): keep a :attr:`~worldcast.player_state.extrapolation.Motion.spread`
            (``live.hop: spread``, :meth:`LiveLoop.publish_position`).
    """

    #: The arrays :meth:`slide` moves: those with a video frame axis.
    SLID = ("yaw", "pitch", "displacement", "prior_displacement", "spread")

    def __init__(
        self,
        start_state: np.ndarray,
        start_substeps: torch.Tensor,
        video_frames: int,
        *,
        prior: PhysicsPrior,
        spread: bool = False,
    ) -> None:
        start = np.asarray(start_state, np.float64).reshape(6)
        initial = torch.from_numpy(start).float().view(1, 1, 6)
        yaw, pitch = integrate_camera_angles(initial, start_substeps.float()[None, None, None])
        frames = int(video_frames)
        super().__init__(
            yaw=np.full(frames, float(yaw[0, 0, 0])),
            pitch=np.full(frames, float(pitch[0, 0, 0])),
            displacement=np.zeros((frames, 3)),
            start_state=start,
            spread=np.zeros((frames, 3)) if spread else None,
        )
        self.prior_displacement = np.zeros((frames, 3))
        _, self.prior_state = physics_prior_displacement(
            start_substeps.float()[None, None], yaw[0, 0, :1][None], prior=prior
        )
        self.final_row, self.version = 0, 0
        self._pending: tuple[int, np.ndarray, np.ndarray] | None = None

    def slide(self) -> None:
        """The round has slid by a block: move the rows (the cumulative displacements keep their
        values: only differences of rows are read) and the newest final row with them."""
        for name in self.SLID:
            if getattr(self, name) is not None:
                roll(getattr(self, name), 0, VIDEO_FRAMES_PER_BLOCK, KEEP_VIDEO)
        self.final_row -= VIDEO_FRAMES_PER_BLOCK

    def prior_input(
        self, first_row: int, substeps: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, PriorState]:
        """Integrate the view angles of the rows ``first_row ..`` from their packed substeps; return
        the physics prior's input over them, whose displacement :meth:`write` takes.

        Args:
            first_row (int): the row after the newest final row.
            substeps (torch.Tensor | np.ndarray): ``[n, 4, A + 1]`` the rows' packed substeps.

        Returns:
            tuple[torch.Tensor, torch.Tensor, PriorState]: the packed substeps ``[n, 4, A + 1]``,
            the yaw ``[n]`` (degrees) of the rows and the integrator after the newest final row.

        Raises:
            ValueError: ``first_row`` does not follow the newest final row.
        """
        first = int(first_row)
        if first != self.final_row + 1:
            raise ValueError(f"rows from {first} after the final row {self.final_row}")
        substeps = np.asarray(substeps, np.float32)
        yaw, pitch = _view_angles(self.yaw[first - 1], self.pitch[first - 1], substeps)
        yaw, pitch = yaw[1:], pitch[1:]
        self._pending = (first, yaw.astype(np.float64), pitch.astype(np.float64))
        return torch.from_numpy(substeps), torch.from_numpy(yaw), self.prior_state

    def write(
        self,
        displacement: torch.Tensor,
        state: PriorState,
        *,
        final: bool,
        stops: Sequence[tuple[np.ndarray, float]] = (),
        kept: tuple[np.ndarray, float] | None = None,
    ) -> None:
        """Write the rows of the latest :meth:`prior_input` from the prior's result over its input.

        Args:
            displacement (torch.Tensor): ``[n, 3]`` the prior's displacement during each row, u.
            state (PriorState): the prior's integrator after the rows.
            final (bool): the rows' controls are the block's: they become history.
            stops (Sequence[tuple[np.ndarray, float]]): walls in the rows' way, in turn: an xy
                unit direction ``[2]`` and how far along it they may move from the row before them,
                u; beyond, their displacement along it stops (across it, it slides on).
            kept (tuple[np.ndarray, float] | None): a wall's slowdown kept over the rows (blocked
                continuation): the xy unit direction ``[2]`` the player was blocked along and the
                share of its displacement along it the camera keeps.

        Raises:
            ValueError: no :meth:`prior_input` is pending.
        """
        if self._pending is None:
            raise ValueError(
                "a write takes the displacement of the pending prior_input; none is pending"
            )
        first, yaw, pitch = self._pending
        self._pending = None
        moved = np.cumsum(np.asarray(displacement, np.float64), 0)
        span, later = slice(first, first + len(yaw)), slice(first + len(yaw), None)
        self.prior_displacement[span] = self.prior_displacement[first - 1] + moved
        _apply_stops(moved, stops)
        if kept is not None:
            direction, share = kept
            along = moved[:, :2] @ direction
            moved[:, :2] -= (1.0 - share) * along[:, None] * direction[None]
        self.displacement[span] = self.displacement[first - 1] + moved
        self.yaw[span], self.pitch[span] = yaw, pitch
        columns = (
            self.yaw,
            self.pitch,
            self.displacement,
            self.prior_displacement,
        )
        for column in columns:
            column[later] = column[span][-1]
        if final:
            self.prior_state = state
            self.final_row = span.stop - 1
        self.version += 1


class LiveCameras(ClientCameras):
    """The client's cameras (:class:`~worldcast.player_state.extrapolation.ClientCameras`) under
    live controls: a block's motion is written twice, so the cameras of :meth:`as_predicted` are
    kept per length of the position track and version of the motion."""

    def __init__(self, track: PositionTrack, motion: LiveMotion, latent_frames: int) -> None:
        super().__init__(track, motion, latent_frames)
        self._version = motion.version

    def as_predicted(self) -> torch.Tensor:
        """:meth:`ClientCameras.as_predicted` of the motion as it is now."""
        if self._version != self.motion.version:
            self._cache, self._version = None, self.motion.version
        return super().as_predicted()

    def with_block(self, cameras: torch.Tensor, s: int) -> torch.Tensor:
        """What :meth:`ClientCameras.for_block` gives for block ``s`` once its rows are written
        again, from ``cameras`` ``[N, 4, 4]``, what it gave before: the earlier latent frames' as
        they were (the block does not change them), the block's from the motion as it is now, the
        later latent frames holding its last."""
        s = int(s)
        return self._block_then_held(cameras.clone(), s, self._c2w(self.block_rows(s)))


# -------------------------------------------------------------------------------- the state model
class LiveStateReader(StateReader):
    """The state model in the block loop (:class:`~worldcast.player_state.closed_loop.
    StateReader`) under live controls.

    A window's controls (:func:`~worldcast.data.controls.state_model_controls`, unchanged) are those
    of a tick stream: the recorded ticks up to latent frame 0, the recorded first frame; then four
    ticks per final row, 1/64 s apart from the last recorded one, each with the row's buttons and a
    quarter of its turn (one tick per ordered substep); then idle ticks to the window's end. Every
    part of the model is causal, so the idle ticks change nothing at the latent frames read.

    A session has no end, and its round slides: the reader numbers the latent frames it reads by
    the session's (:attr:`offset`, the latent frames the round has slid by) and keeps those of the
    window it reads itself (at most 41, as the model's fp16 input), so every one of them must be
    read, in order, once. It drops the live ticks and the controls of the windows it will not
    read again; the controls of a window are the same bits whether it dropped them or not.

    Attributes:
        offset (int): the latent frames the round has slid by.

    Args: those of :class:`~worldcast.player_state.closed_loop.StateReader`.
    """

    def __init__(
        self,
        model: StateModel,
        ticks: TickTable,
        *,
        map_name: str,
        source_fps: float,
        start_frame: int,
        device: torch.device | str,
    ) -> None:
        super().__init__(
            model,
            ticks,
            map_name=map_name,
            source_fps=source_fps,
            start_frame=start_frame,
            device=device,
        )
        recorded = int(count_until(ticks.t, start_frame / float(source_fps)))
        self._recorded_t = np.asarray(ticks.t[:recorded], np.float64)
        self._anchor = float(self._recorded_t[-1])
        self._recorded_active = list(ticks.active[:recorded])
        self._recorded_turn = np.stack(
            [ticks.delta_pitch[:recorded], ticks.delta_yaw[:recorded]], -1
        )
        self._live_active: list[list[str]] = []
        self._live_turn = np.zeros((0, 2), np.float32)
        self._live_dropped = 0  # live ticks dropped from the front of the stream
        self._windows: dict[int, tuple[int, torch.Tensor]] = {}
        self._frames: dict[int, np.ndarray] = {}  # session latent frame -> its fp16 latent
        self.offset = 0

    def append(self, buttons: np.ndarray, turn: np.ndarray) -> None:
        """Add the ticks of final rows.

        Args:
            buttons (np.ndarray): ``[n, 11]`` 0/1 held buttons in the order of
                :data:`~worldcast.data.controls.CONTROL_BUTTONS`.
            turn (np.ndarray): ``[n, 2]`` degrees turned during each row (pitch, yaw).
        """
        for row in np.asarray(buttons):
            held = [CONTROL_BUTTONS[i] for i in np.flatnonzero(row > 0.5)]
            self._live_active.extend([held] * SUBSTEPS_PER_VIDEO_FRAME)
        quarter = np.asarray(turn, np.float32) / np.float32(SUBSTEPS_PER_VIDEO_FRAME)
        self._live_turn = np.concatenate(
            [self._live_turn, np.repeat(quarter, SUBSTEPS_PER_VIDEO_FRAME, 0)]
        )

    def read(
        self, latents: torch.Tensor, latent_frames: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`StateReader.read` at the round's ``latent_frames``, which the reader numbers as
        the session's; the window's earlier latent frames are the ones it was given before.

        Raises:
            KeyError: a latent frame of the window was never given.
        """
        frames = [int(f) + self.offset for f in latent_frames]
        for frame, latent in zip(frames, latent_frames):
            self._frames[frame] = np.asarray(latents[int(latent)], np.float32).astype(np.float16)
        result = super().read(latents, frames)
        self._drop_before(min(window_of(f)[0] for f in frames))
        return result

    def _window_input(self, latents: torch.Tensor, first: int, last: int) -> np.ndarray:
        shape = self._frames[first + last].shape
        v = np.zeros((STATE_MODEL_WINDOW_LATENT_FRAMES, *shape), np.float16)
        for position in range(last + 1):
            v[position] = self._frames[first + position]
        return v

    def _controls_of(self, w: int) -> torch.Tensor:
        live = self._live_dropped + len(self._live_active)
        if self._windows.get(w, (None,))[0] != live:
            first = STATE_MODEL_WINDOW_STRIDE * w
            last = source_frame(self.start_frame, first + STATE_MODEL_WINDOW_LATENT_FRAMES - 1)
            ticks = max(live, int(np.ceil((last / self.source_fps - self._anchor) * TICK_RATE)) + 1)
            idle = ticks - live
            turn = np.concatenate(
                [self._recorded_turn, self._live_turn, np.zeros((idle, 2), np.float32)]
            )
            # the ticks kept are those of the whole stream from the first one dropped on, at the
            # same times
            times = self._anchor + np.arange(self._live_dropped + 1, ticks + 1) / TICK_RATE
            controls = state_model_controls(
                timestamps=np.concatenate([self._recorded_t, times]),
                held_buttons=self._recorded_active + self._live_active + [[]] * idle,
                delta_pitch=turn[:, 0],
                delta_yaw=turn[:, 1],
                start_frame=source_frame(self.start_frame, first),
                source_fps=self.source_fps,
                latent_frames=STATE_MODEL_WINDOW_LATENT_FRAMES,
            )
            self._windows[w] = live, torch.from_numpy(controls)[None].to(self.device).float()
        return self._windows[w][1]

    def _drop_before(self, w: int) -> None:
        """Window ``w`` is the oldest the reader will read: drop the latent frames before its first,
        the controls of the windows before it, and the ticks before the one the controls of its
        first latent frame start from (the last at or before the latent frame before it) less
        one tick of margin, and the recorded ticks, which are all before them."""
        first = STATE_MODEL_WINDOW_STRIDE * w
        self._frames = {f: v for f, v in self._frames.items() if f >= first}
        self._windows = {k: v for k, v in self._windows.items() if k >= w}
        before = (
            source_frame(self.start_frame, first) - SOURCE_FRAMES_PER_LATENT
        ) / self.source_fps
        keep_from = int(np.floor((before - self._anchor) * TICK_RATE)) - 2
        drop = min(max(keep_from - self._live_dropped, 0), len(self._live_active))
        if drop:
            del self._live_active[:drop]
            self._live_turn = self._live_turn[drop:]
            self._live_dropped += drop
            self._recorded_t, self._recorded_active = self._recorded_t[:0], []
            self._recorded_turn = self._recorded_turn[:0]


# ------------------------------------------------------------------------------------ visibility
class LiveVisibility(PredictedVisibility):
    """The visibility labels a client computes for itself
    (:class:`~worldcast.player_state.predicted_visibility.PredictedVisibility`) under live
    controls.

    Its own camera angles are its own rows' in the player-state table (columns 3, 4), which
    :class:`LiveLoop` writes. ``live`` holds where each player is in the world: nobody, until
    :class:`LiveLoop` writes another live player where it is drawn. A block's target
    frames are labelled when it is prepared and tested again at its controls cut, against the
    surface of its context read then (:meth:`retest_target`; :meth:`prepare_target` for latents
    1-24), which is reprojected into the target frames' cameras on the client's device (the
    closed loop's labels reproject on the CPU); the reprojection of the context into the newest
    :data:`ZBUFFERS_KEPT` sets of cameras is kept, so that a cut whose cameras are the ones the
    block was prepared (or last cut) with reprojects nothing. The z-test is
    :func:`~worldcast.player_state.predicted_visibility.ztest_labels`'s, operation by operation
    (:func:`_ztest`). The labels a block is denoised with at its first step are kept
    (:meth:`drawn`). :meth:`surface` reads where :class:`LiveLoop` finds a wall ahead.

    Args:
        batch (Mapping[str, torch.Tensor]): as :class:`PredictedVisibility`.
        depth_fn (DepthFn): the depth head.
        device (torch.device | str): the client's device.
    """

    #: The arrays :meth:`slide` moves: those with a video frame axis; the table is the round
    #: batch's, which slides with it.
    SLID = ("cameras", "points", "live", "_yaw", "_pitch")

    def __init__(
        self, batch: Mapping[str, torch.Tensor], *, depth_fn: DepthFn, device: torch.device | str
    ) -> None:
        super().__init__(batch, depth_fn=depth_fn)
        self.live.zero_()
        self.device = torch.device(device)
        self._context: tuple[list[int], torch.Tensor] | None = None
        self._drawn: dict[int, np.ndarray] = {}
        # the context the kept reprojections are of, and each by the bytes of its cameras
        self._reprojected: tuple[torch.Tensor | None, dict[bytes, np.ndarray]] = (None, {})

    def slide(self) -> None:
        """The round has slid by a block: move the geometry of the video frames, and the labels
        drawn with the latent frames they belong to."""
        for name in self.SLID:
            roll(getattr(self, name), 0, VIDEO_FRAMES_PER_BLOCK, KEEP_VIDEO)
        self._drawn = {
            (f if f < KEEP_LATENTS else f - BLOCK): drawn
            for f, drawn in self._drawn.items()
            if f < KEEP_LATENTS or f - BLOCK >= KEEP_LATENTS
        }

    def update_frames(self, video_frames: Sequence[int] | np.ndarray) -> None:
        """:meth:`PredictedVisibility.update_frames`, with the client's angles of its rows."""
        v = np.asarray(video_frames, np.int64)
        states = self.states.numpy()[0]
        own = states[self.client_slot, v].astype(np.float64)
        yaw, pitch = self._yaw.numpy(), self._pitch.numpy()
        yaw[v], pitch[v] = own[:, 3], own[:, 4]
        self.cameras.numpy()[v] = camera_to_world(
            torch.from_numpy(own[:, :3]), torch.from_numpy(yaw[v]), torch.from_numpy(pitch[v])
        ).numpy()
        points = states[:, v, :3].transpose(1, 0, 2).astype(np.float64)
        points[..., 2] += PLAYER_LIFT_U
        self.points.numpy()[v] = points

    def target_labels(
        self,
        video_frames: Sequence[int],
        depth: Mapping[int, torch.Tensor],
        context_frames: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`PredictedVisibility.target_labels`, reprojected on the client's device; the
        context's surface is kept for :meth:`retest_target`."""
        self._remember(video_frames, depth, context_frames)
        return self._reprojected_labels(*self._context)

    def prepare_target(self, start: int, latents: torch.Tensor) -> None:
        """Read the surface of the context of the block at latent frame ``start`` (one of latents
        1-24) off ``latents`` ``[N, 48, 24, 42]`` before its controls are cut, for
        :meth:`target_visible`."""
        self._remember(*self._target_context(int(start), latents))

    def target_visible(self, start: int, latents: torch.Tensor) -> torch.Tensor:
        """:meth:`PredictedVisibility.target_visible` against the context read by
        :meth:`prepare_target` (read now if it has not been), kept as the block's :meth:`drawn`."""
        start = int(start)
        if self._context is None or self._context[0] != video_frames_of(start, BLOCK):
            self.prepare_target(start, latents)
        visible = _folded(self._labels(*self._context)[0], start)
        self._drawn.update((start + i, frame) for i, frame in enumerate(visible))
        return torch.from_numpy(visible)

    def _remember(
        self,
        video_frames: Sequence[int],
        depth: Mapping[int, torch.Tensor],
        context_frames: Sequence[int],
    ) -> None:
        """Keep the surface of ``context_frames`` on the client's device for the block of
        ``video_frames``."""
        surface = self._context_points(depth, context_frames).to(self.device)
        self._context = list(video_frames), surface

    def surface(self, latent: int, latents: torch.Tensor) -> np.ndarray:
        """``[M, 3]`` float64: the world points (u) of the surface the depth head sees at the last
        video frame of latent frame ``latent`` of ``latents`` ``[> latent, 48, 24, 42]``; it reads
        the latent frame with the one before it, as in a block's context."""
        latent = int(latent)
        first = max(latent - 1, 0)
        depth = self._depth(latents[first : latent + 1], first)
        return self._context_points(depth, [last_video_frame(latent)]).numpy()

    def retest_target(self, window: Mapping[str, torch.Tensor], s: int) -> dict:
        """The gathered window of block ``s`` with its last 16 video frames labelled again, with
        the cameras and test points as they are now, against the surface of the context they
        were labelled with when the window was gathered (:meth:`for_block`); kept as the block's
        :meth:`drawn`.

        Raises:
            ValueError: the last target frames labelled are not block ``s``'s.
        """
        frames, _ = self._context
        if frames != video_frames_of(s, BLOCK):
            raise ValueError(
                f"block {s} is tested again right after its for_block; the last frames labelled"
                f" are the video frames {frames[0]}-{frames[-1]}"
            )
        visible, valid = self._labels(*self._context)
        labels, known = (
            window["client_visibility"].numpy().copy(),
            window["client_visibility_valid"].numpy().copy(),
        )
        labels[0, :, -len(frames) :] = visible.T
        known[0, :, -len(frames) :] = valid.T
        self._drawn.update((int(s) + i, frame) for i, frame in enumerate(_folded(visible, s)))
        return dict(
            window,
            client_visibility=torch.from_numpy(labels),
            client_visibility_valid=torch.from_numpy(known),
        )

    def drawn(self, latent: int) -> np.ndarray | None:
        """``[P]`` bool: the players labelled visible at latent frame ``latent`` when its block was
        denoised (its first denoising step), or None before."""
        drawn = self._drawn.get(int(latent))
        return None if drawn is None else drawn.copy()

    def _reprojected_labels(
        self, video_frames: Sequence[int], context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visible, valid = self._labels(video_frames, context)
        return torch.from_numpy(visible), torch.from_numpy(valid)

    def _labels(
        self, video_frames: Sequence[int], context: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`PredictedVisibility._reprojected_labels` as numpy ``[R, P]`` bool, the
        reprojection kept per context and cameras."""
        idx = list(video_frames)
        cameras = self.cameras.numpy()[idx]
        kept_for, kept = self._reprojected
        if kept_for is not context:
            kept = {}
            self._reprojected = context, kept
        key = cameras.tobytes()
        depth = kept.pop(key, None)
        if depth is None:
            c2w = torch.from_numpy(cameras)
            depth = predicted_visibility._zbuffer(context, c2w.to(context.device), self.tans)
            depth = depth.cpu().numpy()
            if len(kept) >= ZBUFFERS_KEPT:
                del kept[next(iter(kept))]
        kept[key] = depth  # the newest last
        return _ztest(self.points.numpy()[idx], cameras, depth, self.live.numpy()[idx], self.tans)


# --------------------------------------------------------------------------------------- the loop
class LiveLoop(ClosedLoop):
    """One client's player states under live controls (live play, not the paper's), in the closed
    loop's place in the block loop (:class:`~worldcast.player_state.closed_loop.ClosedLoop`).

    Per block ``s``:

    - when it is prepared, :meth:`publish_position` reads the own position at the latent frames
      ``s - 4 .. s - 1`` off the generated video (:class:`LiveStateReader`,
      :class:`~worldcast.player_state.live_position.GuardedPosition`) and writes the block's own
      rows with the last controls held: the newest final row's buttons and weapon, and the newest
      block's turn row by row, so that a block that turns as the one before is prepared as it is
      cut, however its rows share the turn; :meth:`read_positions` writes the other live players
      at the block's predicted session times. No position is published;
    - at its controls cut, :meth:`take_controls` takes in what has arrived, writes the block's own
      rows from its controls and the other live players at its session times, once and finally,
      and publishes the own rows as the block's ``live`` part. A block that may be refreshed
      (``live.refresh``) is cut without publishing; :meth:`restore_cut` takes the loop back to
      before the cut (:meth:`save_cut`) for its cut again, and :meth:`publish_controls` publishes
      the cut that stays.

    The own rows of block ``s`` are the own position at latent frame ``s - 1`` plus the physics
    prior's displacement over the block (:class:`LiveMotion`), stopped at the walls in its way,
    and the own position steps with that camera, as far as the view followed it
    (:class:`~worldcast.player_state.live_position.GuardedPosition`). A block that does not leave
    the ground, as prepared and as cut, stops along its way where the depth head of latent frame
    ``s - 1`` (read when it is prepared) sees a wall ahead of the block's start
    (:func:`~worldcast.player_state.live_position.wall_clearance`), and before the wall a block
    last stopped before while the player stands or pushes on (right before a wall the depth head
    no longer sees its lower part). That wall is forgotten once the player moves another way or
    is past it (:data:`WALL_PASSED_U`); once the motion head of the state model says the player
    moved on where a wall stopped it
    (:attr:`~worldcast.player_state.live_position.GuardedPosition.moved_past_wall`), it is
    forgotten and the next block is not stopped. Every block stops at the edge of the map's
    walkable extent (:class:`~worldcast.player_state.live_position.Walkable`), as the own
    position does. If the motion head found that the view did not follow the camera on the newest
    latent frame of the block before, a block that holds the same movement keys and turns less
    than :data:`CONTINUATION_DEGREES` keeps the share of its way along it that the view followed,
    a quarter of it at the least, so that the view can show whether it moved on.
    Another player is in the world from its first ``live`` part on, whoever the client started
    with (:class:`~worldcast.player_state.live_peers.LivePeer`), and not before: a seat nobody
    plays is nowhere. A player whose client is done is hidden. No slot is written dead, so nobody
    is drawn as a corpse and a seat can be taken at any time.

    With ``live.wall_keys`` (:meth:`release_keys`) a block's movement keys are cleared from the
    row its camera is stopped at, before the block is cut, so that the generator, the state model
    and the block's ``live`` part read a player who stands there.

    :meth:`build` sets it up; the arguments are those of :class:`ClosedLoop` and:

    Attributes:
        offset (int): the latent frames the round has slid by (:meth:`slide`): the session's block
            is the round's plus it, and the ``live`` part of a block is published under it.

    Args:
        batch (dict[str, torch.Tensor]): the round batch (batch size 1), rewritten in place.
        world_state (LiveExchange): where the clients publish their rows.
        client (str): the client's media id.
        client_slot (int): its slot.
        round_media (Mapping[int, str]): media id of every slot of the round.
        prior (PhysicsPrior): the physics prior.
        walkable (Walkable): the walkable extent of the round's map.
        facing (tuple[np.ndarray, np.ndarray]): ``[P, T]`` the yaw and pitch, degrees, the camera
            integral of the round batch's controls holds at every video frame: where a player's
            first block starts turning.
        camera_encoding (str): the encoding of a turn in the view controls
            (:data:`~worldcast.data.controls.CAMERA_ENCODINGS`), the recordings' own.
        wall_keys (str): ``live.wall_keys`` (:meth:`release_keys`).
        dead_zone_u (float): ``live.dead_zone_u``
            (:class:`~worldcast.player_state.live_position.GuardedPosition`).
        hop (str): ``live.hop`` (:meth:`publish_position`).
    """

    #: The arrays :meth:`slide` moves: those with a video frame axis (the yaw and pitch of every
    #: player's recorded controls).
    SLID = ("_batch_facing",)

    def __init__(
        self,
        *,
        batch: dict[str, torch.Tensor],
        reader: LiveStateReader,
        cameras: LiveCameras,
        table: PredictedStateTable,
        visibility: LiveVisibility,
        world_state: LiveExchange,
        client: str,
        client_slot: int,
        round_media: Mapping[int, str],
        prior: PhysicsPrior,
        walkable: Walkable,
        facing: tuple[np.ndarray, np.ndarray],
        wall_keys: str = "held",
        camera_encoding: str = DEFAULT_CAMERA_ENCODING,
        dead_zone_u: float = 0.0,
        hop: str = "step",
    ) -> None:
        super().__init__(
            reader=reader,
            cameras=cameras,
            table=table,
            visibility=visibility,
            world_state=world_state,
            others=(),
        )
        self.batch, self.client, self.slot, self.prior = batch, str(client), int(client_slot), prior
        self.camera_encoding = camera_encoding
        self.offset = 0
        self.walkable, self.wall_keys = walkable, wall_keys
        self.dead_zone_u, self.hop = float(dead_zone_u), hop
        self.media_of = {int(s): str(m) for s, m in round_media.items() if int(s) != self.slot}
        self.peers: dict[int, LivePeer] = {}
        self._handed: dict[int, tuple[float, ...]] = {}  # the parts _discover last handed a peer
        self.position = self._guard(cameras.track.xyz[0], cameras.track.xyz[0])
        self._batch_facing = facing
        self._facing: dict[int, np.ndarray] = {}
        held = np.zeros(len(CONTROL_BUTTONS), np.float32)
        weapon = int(batch["player_weapon_ids"][0, self.slot, 0])
        self._last = held, np.zeros((VIDEO_FRAMES_PER_BLOCK, 2), np.float32), weapon
        self._blocked: tuple[np.ndarray, float, np.ndarray] | None = None
        # of the block prepared last: the surface the depth head saw at its latent frame s - 1, and
        # the place head's newest place term where it was accepted; the wall a block last stopped
        # before: where it stopped (xy) and the direction it ran in
        self._surface = np.zeros((0, 3))
        self._place: np.ndarray | None = None
        self._wall: tuple[np.ndarray, np.ndarray] | None = None
        self._moved_on = False
        # ``guard``: the movement keys of the blocks cleared in a row so far, how many, the row the
        # last block was cleared from, and whether the picture moved while they were
        self._clear_keys: tuple[bool, ...] | None = None
        self._clear_run, self._cleared_from, self._clear_off = 0, None, False
        self._times: np.ndarray | None = None
        self._until = -np.inf
        self._unpublished: tuple | None = None  # a cut taken without publishing

    @classmethod
    def build(
        cls,
        batch: dict[str, torch.Tensor],
        *,
        reader: LiveStateReader,
        client: str,
        client_slot: int,
        round_media: Mapping[int, str],
        latent_frames: int,
        start_frame: int,
        prior: PhysicsPrior,
        walkable: Walkable,
        depth_fn: DepthFn,
        world_state: LiveExchange,
        device: torch.device | str,
        wall_keys: str = "held",
        camera_encoding: str = DEFAULT_CAMERA_ENCODING,
        dead_zone_u: float = 0.0,
        hop: str = "step",
    ) -> "LiveLoop":
        """Set up the live player states of one client.

        The round batch is rewritten for live play: every player alive at the round start is alive
        on every video frame (:func:`~worldcast.player_state.live_peers.keep_alive`); the
        client's own controls after video frame 0 are cleared, its rows after it stand at the round
        start, and its latent frames after the first frame have the unscoped field of view. No
        other player is in the world until its first ``live`` part (:class:`LiveVisibility`), and
        every other player's rows stand at its round start, holding nothing, until then
        (:func:`~worldcast.player_state.live_peers.hold_others`): the round that slides never reads
        a recorded row.

        Args:
            batch (dict[str, torch.Tensor]): the whole round (batch size 1), rewritten in place.
            reader (LiveStateReader): the client's state model in the block loop.
            client (str): the client's media id.
            client_slot (int): its slot.
            round_media (Mapping[int, str]): media id of every slot of the round.
            latent_frames (int): latent frames of the client's window of the round.
            start_frame (int): source frame of latent frame 0.
            prior (PhysicsPrior): the physics prior.
            walkable (Walkable): the walkable extent of the round's map.
            depth_fn (DepthFn): the depth head.
            world_state (LiveExchange): where the clients publish their rows.
            device (torch.device | str): the client's device (:class:`LiveVisibility`).
            wall_keys (str): ``live.wall_keys`` (:meth:`release_keys`).
            camera_encoding (str): see :class:`LiveLoop`.
            dead_zone_u (float): see :class:`LiveLoop`.
            hop (str): see :class:`LiveLoop`.

        Returns:
            LiveLoop: the client's live player states.
        """
        batch.update((key, batch[key].clone()) for key in LIVE_BATCH_KEYS)
        states, slot = batch["player_states"], int(client_slot)
        keep_alive(states)
        states[0, slot, :, ALIVE_INDEX] = 1.0
        hold_others(batch, slot)
        packed = pack_substeps(
            batch["player_control_substeps"], batch["player_control_substep_valid"]
        )
        yaw, pitch = integrate_camera_angles(states[:, :, 0], packed)
        batch["player_control_substeps"][0, slot, 1:] = 0.0
        batch["player_control_substep_valid"][0, slot, 1:] = False
        batch["window_tans"][0, 1:] = _unscoped_tans(1)
        motion = LiveMotion(
            states[0, slot, 0].double().numpy(),
            packed[0, slot, 0],
            int(states.shape[2]),
            prior=prior,
            spread=hop == "spread",
        )
        track = PositionTrack(client, start_frame, motion.start_state[:3])
        table = PredictedStateTable(states, {slot: (track, motion)})
        table.advance(1)  # block 1 starts from the round start
        states[0, slot, 1:, 3], states[0, slot, 1:, 4] = motion.yaw[0], motion.pitch[0]
        return cls(
            batch=batch,
            reader=reader,
            cameras=LiveCameras(track, motion, latent_frames),
            table=table,
            visibility=LiveVisibility(batch, depth_fn=depth_fn, device=device),
            world_state=world_state,
            client=client,
            client_slot=slot,
            round_media=round_media,
            prior=prior,
            walkable=walkable,
            facing=(yaw[0].double().numpy(), pitch[0].double().numpy()),
            wall_keys=wall_keys,
            camera_encoding=camera_encoding,
            dead_zone_u=dead_zone_u,
            hop=hop,
        )

    @property
    def motion(self) -> LiveMotion:
        """The client's own :class:`LiveMotion`."""
        return self.cameras.motion

    @property
    def status(self) -> str:
        """The own position's estimator status (:attr:`GuardedPosition.status`)."""
        return self.position.status

    def _guard(self, p0: np.ndarray, a0: np.ndarray) -> GuardedPosition:
        """The own position estimator from ``p_0`` and the place head's ``A_0``: it runs no
        farther in a latent frame than the physics prior's top speed takes it, and stays on the
        map."""
        top_step = self.prior.top_speed * STATE_MODEL_SUBSTEPS / TICK_RATE
        return GuardedPosition(
            p0, a0, top_step_u=top_step, walkable=self.walkable, dead_zone_u=self.dead_zone_u
        )

    def slide(self) -> None:
        """The round has slid by a block (:func:`~worldcast.sampling.window.slide_window_entries`
        slides its batch, which the table and the visibility geometry read): move everything this
        loop indexes by frame, and count the block in :attr:`offset` and in the reader's."""
        self.motion.slide()
        self.cameras.track.slide()
        self.table.slide()
        self.visibility.slide()
        for name in self.SLID:
            for array in getattr(self, name):
                roll(array, 1, VIDEO_FRAMES_PER_BLOCK, KEEP_VIDEO)
        self.reader.offset += BLOCK
        self.offset += BLOCK

    def clamp_pitch(self, turn: np.ndarray) -> np.ndarray:
        """The next block's turn with each row's pitch turn cut where the own pitch would leave
        [-89, 89] from the newest final row on (pitch > 0 looks down): the effective turn. The
        camera integrals clamp row by row, the 6-wide table's does not; with the effective turn
        they agree.

        Args:
            turn (np.ndarray): ``[16, 2]`` degrees turned during each row (pitch, yaw).

        Returns:
            np.ndarray: ``[16, 2]`` float32; rows the clamp did not reach are unchanged.
        """
        turn = np.array(turn, np.float32)
        values, valid = block_substeps(np.zeros((len(turn), len(CONTROL_BUTTONS))), turn)
        _, pitch = _view_angles(
            0.0, self.motion.pitch[self.motion.final_row], packed_substeps(values, valid)
        )
        clamped = ((pitch[1:] >= PITCH_MAX_DEGREES) & (turn[:, 0] > 0)) | (
            (pitch[1:] <= PITCH_MIN_DEGREES) & (turn[:, 0] < 0)
        )
        turn[clamped, 0] = np.diff(pitch)[clamped]
        return turn

    # ------------------------------------------------------------------------------- the own rows
    def publish_position(self, s: int, latents: torch.Tensor) -> None:
        """Read the own position at the latent frames ``s - 4 .. s - 1`` off ``latents``
        ``[>= s, 48, 24, 42]`` and extend the own track with it (block 1 reads latent frame 0 as
        the place head's ``A_0`` and extends nothing), and the surface the depth head sees at
        latent frame ``s - 1`` (:meth:`LiveVisibility.surface`); then write block ``s``'s own
        rows provisionally, with the newest final row's buttons and weapon held and the newest
        block's turn, row by row. Nothing is published."""
        s = int(s)
        self._surface = self.visibility.surface(s - 1, latents)
        if s == 1:
            place, _ = self.reader.read(latents, [0])
            self.position = self._guard(self.cameras.track.xyz[0], place[0])
        else:
            frames = list(range(s - BLOCK, s))
            place, displacement = self.reader.read(latents, frames)
            steps, camera = (
                [track[last_video_frame(f)] - track[last_video_frame(f - 1)] for f in frames]
                for track in (self.motion.prior_displacement, self.motion.displacement)
            )
            xyz = [self.position.step(*args) for args in zip(place, displacement, steps, camera)]
            self.cameras.track.append(frames, np.stack(xyz))
            if self.hop == "spread" and s > BLOCK:
                self._spread_hop(s)
            self._picture_moved(displacement)
            progress, self._blocked = self.position.blocked_progress, None
            if progress is not None:
                along = steps[-1][:2] / np.linalg.norm(steps[-1][:2])
                self._blocked = along, progress, self._last[0][list(MOVEMENT_COLUMNS)]
            self._place = self.position.accepted_place
            self._moved_on = self.position.moved_past_wall
            if self._moved_on:
                self._wall = None
        buttons, turn, weapon = self._last
        self._write_own(
            s,
            np.tile(buttons, (VIDEO_FRAMES_PER_BLOCK, 1)),
            self.clamp_pitch(turn),
            np.full(VIDEO_FRAMES_PER_BLOCK, weapon),
            final=False,
        )

    def take_controls(
        self,
        s: int,
        buttons: np.ndarray,
        turn: np.ndarray,
        weapon: np.ndarray,
        times: np.ndarray,
        *,
        publish: bool = True,
    ) -> None:
        """Write block ``s`` at its controls cut, finally: take in what has arrived, write the own
        rows and every other live player at the block's session times, re-read the visibility
        geometry of its video frames and publish the own rows as the block's ``live`` part.

        Args:
            s (int): the block's first latent frame.
            buttons (np.ndarray): ``[16, 11]`` 0/1 held buttons.
            turn (np.ndarray): ``[16, 2]`` degrees turned during each video frame (pitch, yaw),
                the effective turn (:meth:`clamp_pitch`).
            weapon (np.ndarray): ``[16]`` the held weapon's id.
            times (np.ndarray): ``[16]`` the session time of each video frame, s, increasing.
            publish (bool): publish the ``live`` part, hand the rows' ticks to the state model
                and hold the block's controls for the next one; ``False``: the cut of a block that
                may be cut again (``live.refresh``), whose :meth:`publish_controls` does it.

        Raises:
            ValueError: ``times`` does not increase.
        """
        s, times = int(s), np.asarray(times, np.float64)
        if times.shape != (VIDEO_FRAMES_PER_BLOCK,) or (np.diff(times) <= 0).any():
            raise ValueError(
                f"a block's {VIDEO_FRAMES_PER_BLOCK} session times increase; got {times}"
            )
        buttons, turn = np.asarray(buttons, np.float32), np.asarray(turn, np.float32)
        weapon = np.asarray(weapon, np.int64)
        self.world_state.refresh()
        self._discover()
        # the times of two cuts less than a block apart overlap: the other players never step back
        drawn = np.maximum(times, self._until)
        self._write_own(s, buttons, turn, weapon, final=True, others=drawn)
        self._write_others(s, drawn, final=True)
        self._write_table(s)
        frames = video_frames_of(s, BLOCK)
        own = self.batch["player_states"].numpy()[0, self.slot, frames[0] : frames[-1] + 1, :5]
        cut = (s + self.offset, times, own.copy(), buttons, turn, weapon, float(drawn[-1]))
        if publish:
            self._publish_cut(*cut)
        else:
            self._unpublished = cut

    def publish_controls(self) -> None:
        """Publish the newest cut that :meth:`take_controls` took without publishing: the own rows
        as the block's ``live`` part, their ticks to the state model, the block's controls held for
        the next one.

        Raises:
            RuntimeError: no such cut.
        """
        if self._unpublished is None:
            raise RuntimeError("no cut was taken without publishing")
        cut, self._unpublished = self._unpublished, None
        self._publish_cut(*cut)

    def _publish_cut(
        self,
        session: int,
        times: np.ndarray,
        rows: np.ndarray,
        buttons: np.ndarray,
        turn: np.ndarray,
        weapon: np.ndarray,
        until: float,
    ) -> None:
        self.world_state.publish_live_state(
            session, times=times, rows=rows, buttons=buttons, turn=turn, weapon=weapon
        )
        self.reader.append(buttons, turn)
        self._last = buttons[-1], turn, int(weapon[-1])
        self._times, self._until = times, until

    def save_cut(self) -> dict[str, Any]:
        """What a controls cut changes (:meth:`release_keys`, :meth:`take_controls`), as it is
        before the cut, for :meth:`restore_cut`: the loop's own state, the own motion, the
        player-state table, the visibility geometry, the other players' and the rows the round
        batch holds of live play (:data:`LIVE_BATCH_KEYS`)."""
        return dict(
            loop=_saved(self, skip=("batch", "peers")),
            motion=_saved(self.motion, skip=("version",)),
            table=_saved(self.table),
            visibility=_saved(self.visibility),
            peers={slot: (peer, _saved(peer)) for slot, peer in self.peers.items()},
            batch={key: self.batch[key].clone() for key in LIVE_BATCH_KEYS},
        )

    def restore_cut(self, saved: Mapping[str, Any]) -> None:
        """Take the loop back to :meth:`save_cut`'s state, so that the block is cut again as it
        was the first time (a refresh, ``live.refresh``); ``saved`` stays as it is, for the next
        time. A player that appeared since is forgotten until the next cut takes in its parts
        again; the motion counts a new version, so that nothing kept of the cameras is reused."""
        _restore(self, saved["loop"])
        _restore(self.motion, saved["motion"])
        self.motion.version += 1
        _restore(self.table, saved["table"])
        _restore(self.visibility, saved["visibility"])
        self.peers = {}
        for slot, (peer, state) in saved["peers"].items():
            _restore(peer, state)
            self.peers[slot] = peer
        for key, value in saved["batch"].items():
            self.batch[key].copy_(value)

    def _write_own(
        self,
        s: int,
        buttons: np.ndarray,
        turn: np.ndarray,
        weapon: np.ndarray,
        *,
        final: bool,
        others: np.ndarray | None = None,
    ) -> None:
        """The own rows of block ``s`` from its controls: the motion (one prior call with what the
        other live players still need to be drawn at the session times ``others``; stopped at the
        walls in its way), the substeps, the weapon, the view angles and the observer signals."""
        frames = video_frames_of(s, BLOCK)
        values, valid = block_substeps(buttons, turn)
        displacement, state = self._integrate(
            self.motion.prior_input(frames[0], packed_substeps(values, valid)), others
        )
        stops, _ = self._stops(s, displacement, final=final)
        self.motion.write(
            displacement, state, final=final, stops=stops, kept=self._kept(buttons, turn)
        )
        rows, b = slice(frames[0], frames[-1] + 1), self.batch
        b["player_control_substeps"].numpy()[0, self.slot, rows] = values
        b["player_control_substep_valid"].numpy()[0, self.slot, rows] = valid
        b["player_weapon_ids"].numpy()[0, self.slot, rows] = weapon
        states = b["player_states"].numpy()
        states[0, self.slot, rows, 3] = self.motion.yaw[rows]
        states[0, self.slot, rows, 4] = self.motion.pitch[rows]
        signals = live_observer_signals(
            weapon[VIDEO_FRAMES_PER_LATENT - 1 :: VIDEO_FRAMES_PER_LATENT]
        )
        for key, value in signals.items():
            b[key].numpy()[0, s : s + BLOCK] = value

    def _stops(
        self, s: int, displacement: torch.Tensor, *, final: bool
    ) -> tuple[list[tuple[np.ndarray, float]], bool]:
        """The walls in the way of block ``s``'s rows, which move by the prior's ``displacement``
        ``[16, 3]`` from the block's start ``E_{s-1}`` (:meth:`LiveMotion.write`): the plane across
        the direction the own rows last stopped before a wall in, at that stop; and a wall the
        depth head saw ahead at latent frame ``s - 1``, for a block that moves. That plane is kept
        while the blocks stand or push on, moving within :data:`CONTINUATION_DEGREES` of that
        direction, until the block's start or the place head lies more than
        :data:`WALL_PASSED_U` past the stop, or the motion head says the view moved past a stop
        (that block is not stopped either). No wall stops a block that leaves the ground. At the
        cut the wall seen ahead is kept in place of the one before. Every block that moves stops at
        the edge of the walkable extent.

        Returns:
            tuple[list[tuple[np.ndarray, float]], bool]: the stops, in the order they cut in: the
            plane kept, the wall seen ahead, the edge; and whether the edge is among them, the last.
        """
        start = self.cameras.track.xyz[s - 1]
        moved = np.asarray(displacement, np.float64)
        way = moved[:, :2].sum(0)
        length = float(np.linalg.norm(way))
        wall = self._wall
        if wall is not None:
            stop, direction = wall
            on = float(way @ direction) >= length * np.cos(np.deg2rad(CONTINUATION_DEGREES))
            past = [start] if self._place is None else [start, self._place]
            passed = max(float((p[:2] - stop) @ direction) for p in past) > WALL_PASSED_U
            if passed or not (on or length < IDLE_STEP_U):
                wall = None
        stops = []
        if not (self._moved_on or (moved[:, 2] != 0.0).any()):
            if wall is not None:
                stop, direction = wall
                stops.append((direction, max(float((stop - start[:2]) @ direction), 0.0)))
            if length >= IDLE_STEP_U:
                direction = way / length
                clearance = wall_clearance(self._surface - start, direction)
                if np.isfinite(clearance):
                    stops.append((direction, clearance))
                    wall = start[:2] + clearance * direction, direction
        at_edge = False
        if length >= IDLE_STEP_U:
            edge = self.walkable.clearance(start[:2], way / length, length)
            if np.isfinite(edge):
                stops.append((way / length, edge))
                at_edge = True
        if final:
            self._wall = wall
        return stops, at_edge

    def release_keys(self, s: int, buttons: np.ndarray, turn: np.ndarray) -> np.ndarray:
        """The block's buttons as its player's camera can hold them: where the camera is stopped
        (:meth:`_stops`) the movement keys are cleared from the first row it is stopped at, so that
        the generator, the state model and the block's ``live`` part read a player who stands
        there. ``live.wall_keys``: ``held`` clears nothing, ``edge`` the stops at the edge of the
        walkable extent, ``walls`` every stop, ``guard`` the stops of a wall the depth head sees
        (the plane kept from one included): never the edge (a position at the edge is wrong by
        construction), for at most :data:`CLEAR_BLOCKS` blocks in a row (the block after them
        passes its keys: a probe), not after the picture moved while the keys were cleared
        (:meth:`_picture_moved`), and not once the player's movement keys change or are all
        released. A slowdown kept over the block (:meth:`_kept`) is no stop. The block is moved
        with the keys pressed to find the row, and again, by :meth:`take_controls`, with the keys
        it keeps.

        Args:
            s (int): the block's first latent frame.
            buttons (np.ndarray): ``[16, 11]`` 0/1 held buttons.
            turn (np.ndarray): ``[16, 2]`` degrees turned during each video frame (pitch, yaw),
                the effective turn (:meth:`clamp_pitch`).

        Returns:
            np.ndarray: ``buttons`` itself where no key is cleared, else a copy ``[16, 11]``.
        """
        movement = list(MOVEMENT_COLUMNS)
        guard = self.wall_keys == "guard"
        pressed = tuple(bool(k) for k in buttons[:, movement].any(axis=0))
        if guard and pressed != self._clear_keys:
            self._clear_keys, self._clear_run, self._clear_off = pressed, 0, False
        if self.wall_keys == "held" or not any(pressed):
            return buttons
        values, valid = block_substeps(buttons, turn)
        frames = video_frames_of(s, BLOCK)
        displacement, _ = self._integrate(
            self.motion.prior_input(frames[0], packed_substeps(values, valid)), None
        )
        stops, at_edge = self._stops(s, displacement, final=False)
        cut = _apply_stops(np.cumsum(np.asarray(displacement, np.float64), 0), stops)
        if self.wall_keys == "edge":
            cut = cut[-1:] if at_edge else []
        elif guard and at_edge:
            cut = cut[:-1]
        rows = [row for row in cut if row is not None]
        self._cleared_from = None
        if guard:
            if not rows or self._clear_off:
                self._clear_run = 0
                return buttons
            if self._clear_run >= CLEAR_BLOCKS:
                self._clear_run = 0
                return buttons
            self._clear_run += 1
            self._cleared_from = min(rows)
        if not rows:
            return buttons
        released = np.array(buttons)
        released[min(rows) :, movement] = 0.0
        return released

    def _spread_hop(self, s: int) -> None:
        """``live.hop: spread``: the camera of block ``s`` starts where the block before it ended
        and runs into the estimate over the block, so that the hop (the camera's end less the
        estimate, what the view did not follow of it and the place head's corrections) shows as a
        glide and not as a step at the block's start. The offset goes into the motion's
        :attr:`~worldcast.player_state.extrapolation.Motion.spread`, which the rays and the
        player state field read alike, and it is gone by the block's last video frame."""
        ended = self.cameras.block_rows(s - BLOCK)[-1, :3]
        hop = ended - self.cameras.track.xyz[s - 1]
        self.motion.spread[video_frames_of(s, BLOCK)] = HOP_WEIGHTS[:, None] * hop[None]

    def _picture_moved(self, displacement: np.ndarray) -> None:
        """``live.wall_keys: guard``: the picture did not stand for a player whose keys were cleared
        when the motion head reads, on the last two latent frames of that block, at least
        :data:`~worldcast.player_state.live_position.MOVED_SHARE` of the running step: the clearing
        is no use there (the generator walks without the keys' help), so it stops until the
        player's keys change."""
        cleared, self._cleared_from = self._cleared_from, None
        if cleared is None or cleared > VIDEO_FRAMES_PER_BLOCK // 2:
            return
        view = (
            np.linalg.norm(np.asarray(displacement, np.float64)[:, :2], axis=1)
            * DISPLACEMENT_UNIT_U
        )
        if float(np.median(view[-2:])) >= MOVED_SHARE * self.position.top_step_u:
            self._clear_off = True

    def _kept(self, buttons: np.ndarray, turn: np.ndarray) -> tuple[np.ndarray, float] | None:
        """The blocked continuation of a block's rows, or None."""
        if self._blocked is None:
            return None
        along, progress, keys = self._blocked
        if (buttons[:, list(MOVEMENT_COLUMNS)] != keys).any():
            return None
        if np.abs(np.cumsum(turn[:, 1])).max() >= CONTINUATION_DEGREES:
            return None
        return along, progress

    def _integrate(
        self, own: tuple[torch.Tensor, torch.Tensor, PriorState] | None, times: np.ndarray | None
    ) -> tuple[torch.Tensor, PriorState] | None:
        """One physics prior call: the own rows' input ``own`` (if given) and what every other
        live player's extension still needs for it to be drawn at the session ``times`` (if given,
        :meth:`LivePeer.prior_inputs`); returns the own result."""
        inputs = [] if own is None else [own]
        pending = []
        if times is not None:
            pending = [(peer, peer.prior_inputs(times)) for peer in self.peers.values()]
        given = inputs + [one for _, asked in pending for one in asked]
        if not given:
            return None
        results = prior_displacements(given, prior=self.prior)
        first = len(inputs)
        for peer, asked in pending:
            peer.extend(results[first : first + len(asked)])
            first += len(asked)
        return None if own is None else results[0]

    # ----------------------------------------------------------------------- the other players
    def read_positions(self, s: int) -> None:
        """Take in what has arrived and write block ``s``'s rows provisionally: every other live
        player at the block's predicted session times (the newest final block's period on), the
        own position and the visibility geometry of the block's video frames."""
        s = int(s)
        self.world_state.refresh()
        self._discover()
        times = None
        if self._times is not None:
            period = float(np.median(np.diff(self._times)))
            times = self._until + period * np.arange(1, VIDEO_FRAMES_PER_BLOCK + 1)
        self._integrate(None, times)
        self._write_others(s, times, final=False)
        self._write_table(s)

    def _discover(self) -> None:
        """Hand every player's newest ``live`` parts to its :class:`LivePeer`; a player's first
        part makes it one."""
        for slot, media in self.media_of.items():
            parts = self.world_state.recent_live(media, LIVE_PARTS_READ)
            if not parts:
                continue
            if slot not in self.peers:
                self.peers[slot] = LivePeer(self.prior)
            # a part is known by its last session time: those handed last time change nothing
            handed = tuple(float(part.times[-1]) for part in parts)
            if self._handed.get(slot) == handed:
                continue
            self._handed[slot] = handed
            self.peers[slot].update(
                PeerRows(part.times, part.rows, buttons_from_bits(part.buttons), part.weapon)
                for part in parts
            )

    def _write_others(self, s: int, times: np.ndarray | None, *, final: bool) -> None:
        """Every live player's rows of block ``s`` at ``times`` (none before the own block 1's cut):
        the other players are in the world only where these rows put them."""
        frames = video_frames_of(s, BLOCK)
        rows, b = slice(frames[0], frames[-1] + 1), self.batch
        states, substeps, valid, weapons = (
            b[key].numpy()
            for key in (
                "player_states",
                "player_control_substeps",
                "player_control_substep_valid",
                "player_weapon_ids",
            )
        )
        live = self.visibility.live.numpy()
        facing = []  # final: each player's facing before the block and its packed substeps
        for slot, media in self.media_of.items():
            peer = self.peers.get(slot)
            if peer is None or times is None:
                continue
            done = self.world_state.done_status(media) is not None
            before = last_video_frame(s - 1)
            batch_facing = np.array(
                [self._batch_facing[0][slot, before], self._batch_facing[1][slot, before]]
            )
            previous = self._facing.get(slot, batch_facing)
            block = peer.at(times, previous, final=final, hidden=done)
            states[0, slot, rows, :5] = block.rows
            substeps[0, slot, rows] = block.values
            valid[0, slot, rows] = block.valid
            weapons[0, slot, rows] = block.weapon
            live[rows, slot] = block.present
            if final:
                facing.append((slot, previous, packed_substeps(block.values, block.valid)))
        if facing:
            slots, previous, packed = zip(*facing)
            start = np.stack(previous)
            yaw, pitch = _view_angles(start[:, 0], start[:, 1], np.stack(packed))
            for slot, angles in zip(slots, np.stack([yaw[:, -1], pitch[:, -1]], -1)):
                self._facing[slot] = angles.astype(np.float64)

    def _write_table(self, s: int) -> None:
        """The own position over block ``s`` into the table, and the visibility geometry of the
        block's video frames: the own camera and every player's test point."""
        self.table.advance(s)
        self.visibility.update_frames(video_frames_of(s, BLOCK))

    # ------------------------------------------------------------------ the other players' entries
    def block_at(self, slot: int, window_start: int, f0: int) -> MemoryBlock | None:
        """Another player's block at a published key, keyed at the cameras of the rows it was
        denoised with (its ``live`` part); None while its part has not arrived. A block is never
        skipped for a death in the recording.

        Args:
            slot (int): the player's slot.
            window_start (int): the source frame its rollout starts at.
            f0 (int): the block's first latent frame in that rollout.
        """
        media = self.media_of[int(slot)]
        part = self.world_state.live_part(media, f0)
        if part is None:
            return None
        rows = np.asarray(part.rows, np.float32)[
            VIDEO_FRAMES_PER_LATENT - 1 :: VIDEO_FRAMES_PER_LATENT
        ]
        c2w = c2w_from_state_rows(np.column_stack([rows, np.ones(len(rows), np.float32)]))
        return MemoryBlock.at(media, slot, window_start, f0, c2w)

    def inputs_as_generated(
        self, inputs: dict[str, Any], block: MemoryBlock, part: LivePart
    ) -> dict[str, Any]:
        """The memory frames' inputs (:func:`~worldcast.data.memory_frames.memory_frame_inputs`)
        of a block as its player generated it, from its ``live`` part ``part``, the client's own
        blocks' included; with the observer signals of live play (:func:`live_observer_signals`)
        and the unscoped field of view. Its view controls encode the turn the player made, the
        part's: a gain the player's input path fed the generator on the yaw is not in them."""
        buttons, turn = buttons_from_bits(part.buttons), np.asarray(part.turn, np.float32)
        values, valid = block_substeps(buttons, turn)
        weapon = np.asarray(part.weapon, np.int64)
        states = np.column_stack([part.rows, np.ones(len(turn))]).astype(np.float32)
        signals = live_observer_signals(
            weapon[VIDEO_FRAMES_PER_LATENT - 1 :: VIDEO_FRAMES_PER_LATENT]
        )
        given = {
            "states": torch.from_numpy(states),
            "buttons": torch.from_numpy(buttons),
            "view_deltas": torch.from_numpy(
                np.stack([encode_turn(t, self.camera_encoding) for t in turn])
            ),
            "weapon_ids": torch.from_numpy(weapon),
            "control_substeps": torch.from_numpy(values),
            "control_substep_valid": torch.from_numpy(valid),
            **{key: torch.from_numpy(value) for key, value in signals.items()},
        }
        return dict(inputs, **given, tans=_unscoped_tans(BLOCK))

    # ---------------------------------------------------------------------------------- the page
    def in_world(self, latent: int) -> np.ndarray:
        """``[P]`` bool: the players in the world at latent frame ``latent``, once its block is
        written: another live player drawn there (False for the client itself and for every seat
        nobody plays)."""
        return self.visibility.live[last_video_frame(int(latent))].numpy().copy()
