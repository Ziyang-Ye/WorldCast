"""The own position under live controls: what the player's camera moved, as far as the state model's
motion head says the generated view followed it, corrected by the place head while the place head
can be trusted, so that the client's camera never jumps; the wall ahead of the player, read off the
depth head, and the edge of the map, where a block's camera stops.

When the controls are the player's and not the recording's, a client's own position is estimated
from what it generates. Eq. (4) (:class:`~worldcast.player_state.closed_loop.ComplementaryFilter`,
the paper's closed loop) moves its estimate by half of every place-head innovation, and on generated
video the place head is off by a cell now and then, or loses the player for good: the camera would
jump, and every player the player state field draws with it. :class:`GuardedPosition` is not
Eq. (4). The camera's displacement (the physics prior's, stopped at the walls) advances its
estimate where the motion head measured the view moving, and the place head only corrects it,
through a gate and a rate limit. A place head that disagrees with the estimate for seconds,
without jumping, while it agreed with it earlier (or the estimate is at the map's edge), is
followed to where it is. A standing player is never moved, no player is moved faster than it runs,
and a jump in place lands where it took off. :func:`wall_clearance` finds how far the player can
move before a wall the depth head sees, :class:`Walkable` before the map's edge.
"""

from collections import deque

import numpy as np

from worldcast.data.game import BODY_HEIGHT, PLAYER_WIDTH
from worldcast.data.latents import BLOCK
from worldcast.modeling.state_model import CELL_SIZE, DISPLACEMENT_UNIT_U

from .closed_loop import FUSION_WEIGHT

__all__ = [
    "BLOCK_LAG_U",
    "BODY_RADIUS_U",
    "CONTINUATION_FLOOR",
    "CREDENTIAL_FRAMES",
    "CREDENTIAL_NEAR",
    "DEGRADED_REJECTIONS",
    "DEGRADED_WINDOW",
    "EDGE_NEAR",
    "GATE_XY_U",
    "GATE_Z_U",
    "HEAD_FRAMES",
    "HEAD_MOVES",
    "IDLE_STEP_U",
    "MOVED_SHARE",
    "PASSED_SHARE",
    "PERSISTENT_FRAMES",
    "PERSISTENT_NEAR",
    "PROBE_MIN_U",
    "RATE_U",
    "RATE_XY_GAIN",
    "RATE_Z_GAIN",
    "REACQUIRE_RATE_U",
    "STEP_HEIGHT_U",
    "STILL_SHARE",
    "STUCK_MIN_STEP_U",
    "WALKABLE_MARGIN",
    "WALKABLE_STEP_U",
    "WALL_POINTS",
    "WALL_SLIVERS",
    "WALL_SLIVER_DEGREES",
    "WALL_SPREAD",
    "WALL_SPREAD_U",
    "GuardedPosition",
    "Walkable",
    "wall_clearance",
]

#: The gate, u: one cell of the place head in xy, and its height. An innovation outside it is
#: rejected; a first place estimate ``A_0`` outside it around ``p_0`` is replaced by ``p_0``.
GATE_XY_U = CELL_SIZE[0]
GATE_Z_U = 96.0
#: A correction is at most ``RATE_U`` plus a share of the latent frame's own xy step, u: the share
#: in xy, and the larger one in height (stairs, slopes and drops).
RATE_U = 4.0
RATE_XY_GAIN = 0.25
RATE_Z_GAIN = 0.75
#: A place head that disagrees with the estimate for seconds is followed at this rate per latent
#: frame (xy and height, u; a block's camera is off by four times it, two body widths); a standing
#: player back on the ground returns to its take-off height at the same rate.
REACQUIRE_RATE_U = 16.0
#: The place head's robust position is the median of its newest ``HEAD_FRAMES`` place terms (a
#: glitch of the head is a latent frame or two). It disagrees persistently when, in at least
#: ``PERSISTENT_NEAR`` of the last ``PERSISTENT_FRAMES`` latent frames, it is outside the gate of
#: the estimate and has not jumped by more than the gate: its own outputs are consistent for
#: seconds. A head that holds its place while the camera moved more than half a gate (it moved
#: less than ``HEAD_MOVES`` of the camera's net way) is stuck, not right. A persistent head is
#: followed if it is credible: it agreed with the estimate (inside the gate) on at least
#: ``CREDENTIAL_NEAR`` of the last ``CREDENTIAL_FRAMES`` latent frames, or the prior's step led
#: out of the map's extent on at least ``EDGE_NEAR`` of the last ``PERSISTENT_FRAMES`` (the
#: estimate is wrong by construction). A head that has not agreed lately and keeps its distance is
#: lost, as often as it is right: the estimate keeps its dead reckoning.
HEAD_FRAMES = 5
PERSISTENT_FRAMES = 16
PERSISTENT_NEAR = 14
HEAD_MOVES = 0.25
CREDENTIAL_FRAMES = 64
CREDENTIAL_NEAR = 16
EDGE_NEAR = 4
#: A prior step shorter than this in xy is standing still, u.
IDLE_STEP_U = 1.0
#: The estimate steps with the camera as far as the generated view followed it. The view did not
#: move along the camera's step when the motion head's displacement is below ``STILL_SHARE`` of it,
#: and moved from ``MOVED_SHARE`` on: in between the step is scaled linearly. It is read where the
#: prior's step is at least ``STUCK_MIN_STEP_U`` (xy, u; 0.65 of the running step: on recorded
#: video the head reads less than 0.4 of a 28-40 u step on 29% of the latent frames, of a 55 u
#: step on 2%; a slower gait follows the camera) and the camera's at least ``PROBE_MIN_U`` (what a
#: continuation at its floor still moves). The head's one latent frame of a still view is at times
#: its glitch, so the larger share of the newest two latent frames counts. The view moved past a
#: wall the depth head stopped the camera at when its displacement is from ``PASSED_SHARE`` of the
#: prior's step on in both of the newest two latent frames. A block that follows a latent frame the
#: view did not follow keeps at least ``CONTINUATION_FLOOR`` of the camera's step along the way, so
#: that the view can show it moved.
STUCK_MIN_STEP_U = 40.0
STILL_SHARE = 0.15
MOVED_SHARE = 0.4
PASSED_SHARE = 0.5
CONTINUATION_FLOOR = 0.25
PROBE_MIN_U = CONTINUATION_FLOOR * STUCK_MIN_STEP_U
#: A block's camera starts at the estimate, so what the view did not follow of the camera's way
#: shows as a hop at the next block's start: the estimate falls at most ``BLOCK_LAG_U`` behind the
#: camera within a block, u (two body widths).
BLOCK_LAG_U = 2.0 * PLAYER_WIDTH
#: A map's walkable extent is its cells' columns of the place head's cell table, and
#: ``WALKABLE_MARGIN`` columns around them (a cell is 256 u); the edge is searched for every
#: ``WALKABLE_STEP_U`` (u) along a block's way.
WALKABLE_MARGIN = 1
WALKABLE_STEP_U = 16.0
#: The place head is rejected most of the time with at least ``DEGRADED_REJECTIONS`` rejections in
#: the last ``DEGRADED_WINDOW`` latent frames.
DEGRADED_REJECTIONS = 12
DEGRADED_WINDOW = 16
#: The player's hull: half its width, u; what it steps up without a jump, u.
BODY_RADIUS_U = PLAYER_WIDTH / 2
STEP_HEIGHT_U = 18.0
#: A wall in the hull's way: in at least ``WALL_SLIVERS`` slivers of ``WALL_SLIVER_DEGREES`` of
#: directions, the surface points of the lower and of the upper half of the body's height (at
#: least ``WALL_POINTS`` each) lie at one distance, within ``WALL_SPREAD`` of it or
#: ``WALL_SPREAD_U``, and the slivers agree on it as closely. A slope or stairs put the upper half
#: farther (27 u at 45 degrees); the depth head is off by about a tenth on most patches.
WALL_SLIVER_DEGREES = 2.5
WALL_POINTS = 2
WALL_SPREAD = 0.1
WALL_SPREAD_U = 8.0
WALL_SLIVERS = 3


def _spread(distance: float) -> float:
    return max(WALL_SPREAD_U, WALL_SPREAD * distance)


def wall_clearance(surface: np.ndarray, direction: np.ndarray) -> float:
    """How far the player can move along ``direction`` before its hull meets a wall of
    ``surface``.

    The surface in the hull's way (within :data:`BODY_RADIUS_U` across the way, between
    :data:`STEP_HEIGHT_U` and the body's height above the feet) is cut into slivers of directions
    (:data:`WALL_SLIVER_DEGREES`). A sliver sees a wall where its lower and its upper half of the
    body's height lie at one distance (:data:`WALL_POINTS`, :data:`WALL_SPREAD`,
    :data:`WALL_SPREAD_U`): a vertical surface; a slope, stairs, the floor and what hangs overhead
    are no wall. The nearest distance :data:`WALL_SLIVERS` slivers agree on is the wall's.

    Args:
        surface (np.ndarray): ``[M, 3]`` surface points relative to the player's feet, u (x, y
            horizontal, z up).
        direction (np.ndarray): ``[2]`` the unit xy direction of the movement.

    Returns:
        float: the distance the player's origin can move, u, at least 0; ``inf`` with no wall in
        the way.
    """
    direction = np.asarray(direction, np.float64).reshape(2)
    points = np.asarray(surface, np.float64).reshape(-1, 3)
    along = points[:, :2] @ direction
    across = points[:, :2] @ np.array([-direction[1], direction[0]])
    height = points[:, 2]
    way = (
        (along > 0.0)
        & (np.abs(across) <= BODY_RADIUS_U)
        & (height > STEP_HEIGHT_U)
        & (height <= BODY_HEIGHT)
    )
    sliver = np.floor(np.degrees(np.arctan2(across[way], along[way])) / WALL_SLIVER_DEGREES)
    upper = height[way] > 0.5 * (STEP_HEIGHT_U + BODY_HEIGHT)
    along = along[way]
    walls = []
    for k in np.unique(sliver):
        low, high = along[(sliver == k) & ~upper], along[(sliver == k) & upper]
        if min(len(low), len(high)) < WALL_POINTS:
            continue
        near, far = sorted((float(np.median(low)), float(np.median(high))))
        if far - near <= _spread(near):
            walls.append(near)
    walls = np.sort(walls)
    for distance in walls:
        if np.sum(np.abs(walls - distance) <= _spread(distance)) >= WALL_SLIVERS:
            return max(float(distance) - BODY_RADIUS_U, 0.0)
    return float("inf")


class Walkable:
    """A map's walkable extent: where the place head puts players, which is where the recordings
    went. A position over no cell of the map, more than :data:`WALKABLE_MARGIN` columns from the
    nearest, is wrong by construction.

    Args:
        columns (np.ndarray): ``[n, 2]`` int lattice indices (x, y) of the cells' columns of one
            map (:meth:`~worldcast.modeling.state_model.PlaceHead.cell_columns`).
    """

    def __init__(self, columns: np.ndarray) -> None:
        columns = np.asarray(columns, np.int64).reshape(-1, 2)
        self._low = columns.min(0) - WALKABLE_MARGIN
        size = columns.max(0) - self._low + WALKABLE_MARGIN + 1
        self._mask = np.zeros(tuple(size), bool)
        for i, j in columns - self._low:
            self._mask[
                i - WALKABLE_MARGIN : i + WALKABLE_MARGIN + 1,
                j - WALKABLE_MARGIN : j + WALKABLE_MARGIN + 1,
            ] = True

    def contains(self, xy: np.ndarray) -> bool:
        """Whether the position ``xy`` ``[2]`` (u) is in the walkable extent."""
        i, j = np.floor(np.asarray(xy, np.float64) / CELL_SIZE[0]).astype(np.int64) - self._low
        return bool(
            0 <= i < self._mask.shape[0] and 0 <= j < self._mask.shape[1] and self._mask[i, j]
        )

    def clearance(self, start: np.ndarray, direction: np.ndarray, reach: float) -> float:
        """How far a player can walk from ``start`` along ``direction`` before it leaves the
        walkable extent, searched up to ``reach``.

        Args:
            start (np.ndarray): ``[2]`` the position (u).
            direction (np.ndarray): ``[2]`` the unit xy direction of the movement.
            reach (float): how far it moves at most, u.

        Returns:
            float: the distance, u, in steps of :data:`WALKABLE_STEP_U`; ``inf`` if it does not
            leave within ``reach``, or starts outside (the cell table says nothing of where a
            player outside may go).
        """
        start, direction = np.asarray(start, np.float64), np.asarray(direction, np.float64)
        if not self.contains(start):
            return float("inf")
        for k in range(1, int(np.ceil(reach / WALKABLE_STEP_U)) + 1):
            if not self.contains(start + direction * min(k * WALKABLE_STEP_U, reach)):
                return (k - 1) * WALKABLE_STEP_U
        return float("inf")


def _outside_gate(offset: np.ndarray) -> bool:
    return bool(np.linalg.norm(offset[:2]) > GATE_XY_U or abs(offset[2]) > GATE_Z_U)


class GuardedPosition:
    """A client's own position under live controls, latent frame by latent frame.

    Per latent frame ``f``, with ``d_f`` the physics prior's displacement over the frame, ``c_f``
    what the camera moved (``d_f`` stopped at a wall in the way and slowed by a continuation,
    :class:`~worldcast.player_state.live.LiveLoop`), ``Delta_f`` the motion head's displacement
    and ``a_f = A_f - A_0 + p_0`` the place term of Eq. (4):

    - the step is ``c_f`` scaled by whether the generated view followed it: where ``|d_f,xy|`` is
      at least :data:`STUCK_MIN_STEP_U` and ``|c_f,xy|`` at least :data:`PROBE_MIN_U`, the length
      of ``Delta_f`` is a share of ``|c_f,xy|`` (in any direction: in live play the motion head's
      displacement does not follow the keys' direction), and the step is whole from
      :data:`MOVED_SHARE` on, nothing below :data:`STILL_SHARE`, linear in between, by the larger
      share of the newest two latent frames (one is at times the head's glitch). A camera that
      pushes on where the view did not move does not carry the estimate along, but for the
      :data:`BLOCK_LAG_U` the estimate may fall behind it within a block (the camera of the next
      block starts at the estimate: the rest would be a hop). :attr:`blocked_progress` is what the
      camera of the next block keeps along the way after such a latent frame, at least
      :data:`CONTINUATION_FLOOR`: the view follows a camera that moves, so a block that keeps
      moving shows whether the view moved on. Where a wall stopped or a continuation slowed the
      camera and the view moved at least :data:`PASSED_SHARE` of ``|d_f,xy|`` in both of the
      newest two latent frames (:attr:`moved_past_wall`), the view went through: the step is
      ``d_f`` scaled by the newest such share. A camera that moved less than :data:`IDLE_STEP_U`
      (held at a wall by a stop) is no step, and standing still (``|d_f,xy|`` below
      :data:`IDLE_STEP_U`) the step keeps only its height (a jump);
    - the prediction is ``q = E_{f-1} + step``. Standing on the ground (the prior's height step is
      0), its height returns to the take-off height of a jump in place at
      :data:`REACQUIRE_RATE_U` per latent frame: every flight of the prior ends higher than it
      started;
    - the place head is followed unless at least :data:`DEGRADED_REJECTIONS` of the last
      :data:`DEGRADED_WINDOW` innovations ``a_f - q`` fell outside the gate (:data:`GATE_XY_U` in
      xy, :data:`GATE_Z_U` in height), and then again while it disagrees persistently and is
      credible (:data:`PERSISTENT_NEAR` of the last :data:`PERSISTENT_FRAMES` latent frames with
      its robust position, the median of its newest :data:`HEAD_FRAMES` place terms, outside the
      gate and not jumped by more than the gate; moving with the player, :data:`HEAD_MOVES`;
      agreeing with the estimate on :data:`CREDENTIAL_NEAR` of the last
      :data:`CREDENTIAL_FRAMES` latent frames, or the prior's step leading out of the map's extent
      on :data:`EDGE_NEAR` of the last :data:`PERSISTENT_FRAMES`);
    - followed and moving, ``q`` is corrected, in xy and in height each inside its own gate, by
      ``FUSION_WEIGHT (a_f - q)`` at most :data:`RATE_U` plus a share of the camera's xy step
      (:data:`RATE_XY_GAIN`; :data:`RATE_Z_GAIN` in height); outside its gate, the persistent
      head's robust position is followed at :data:`REACQUIRE_RATE_U` per latent frame; else
      nothing;
    - standing still nothing is corrected: a standing player's estimate is bit-constant on the
      ground;
    - the estimate moves at most ``top_step_u`` in xy per latent frame, the step and the
      correction together (no player runs faster), and no further than the edge of the walkable
      extent (:class:`Walkable`): a position outside the map is wrong by construction;
    - ``A_0`` outside the gate around ``p_0`` is replaced by ``p_0``.

    So ``|E_f - E_{f-1} - step|`` is at most ``max(REACQUIRE_RATE_U, RATE_U + RATE_XY_GAIN
    |c_xy|)`` in xy, and with :data:`RATE_Z_GAIN` in height, and the camera of a block ends at most
    :data:`BLOCK_LAG_U` plus four such corrections from the estimate: no hop of more than a few
    body widths. Positions are float64, u.

    With ``dead_zone_u`` above 0 the motion head is read for every step of at least
    :data:`IDLE_STEP_U`, a tap and a slow walk included (below :data:`STUCK_MIN_STEP_U` it is not
    otherwise): a displacement shorter than ``dead_zone_u`` counts as a view that stands, and the
    step is the camera's scaled by the share the view followed, as above, by the larger share of
    the newest two latent frames that were checked (a step after one that was not checked counts
    its own: a tap has no latent frame before it to give the doubt to).

    Args:
        p0 (np.ndarray): ``[3]`` the given initial position ``p_0`` (latent frame 0), u.
        a0 (np.ndarray): ``[3]`` the place head's estimate ``A_0`` at latent frame 0, u.
        top_step_u (float): the farthest a player runs in a latent frame, u: the physics prior's
            top speed over a latent frame.
        walkable (Walkable): the map's walkable extent.
        dead_zone_u (float): the motion head's displacement below which the view stands, u, for a
            step shorter than :data:`STUCK_MIN_STEP_U`; 0: such a step is not checked.
    """

    def __init__(
        self,
        p0: np.ndarray,
        a0: np.ndarray,
        *,
        top_step_u: float,
        walkable: Walkable,
        dead_zone_u: float = 0.0,
    ) -> None:
        self.p0 = np.asarray(p0, np.float64).reshape(3).copy()
        a0 = np.asarray(a0, np.float64).reshape(3)
        self.a0 = self.p0.copy() if _outside_gate(a0 - self.p0) else a0.copy()
        self.top_step_u, self.walkable = float(top_step_u), walkable
        self.dead_zone_u = float(dead_zone_u)
        self._position = self.p0.copy()
        # the height a standing player stands at; the prior starts at rest on the ground
        self._ground: float | None = float(self.p0[2])
        self._rejected: deque[bool] = deque(maxlen=DEGRADED_WINDOW)
        self._agreed: deque[bool] = deque(maxlen=CREDENTIAL_FRAMES)
        self._places: deque[np.ndarray] = deque(maxlen=HEAD_FRAMES)
        self._offsets: deque[np.ndarray] = deque(maxlen=HEAD_FRAMES)
        self._track: np.ndarray | None = None
        self._head_steps: deque[np.ndarray] = deque(maxlen=PERSISTENT_FRAMES)
        self._cameras: deque[np.ndarray] = deque(maxlen=PERSISTENT_FRAMES)
        self._apart: deque[bool] = deque(maxlen=PERSISTENT_FRAMES)
        self._edge: deque[bool] = deque(maxlen=PERSISTENT_FRAMES)
        self._persistent = False
        self._share, self._seen, self._checked = 1.0, 1.0, False
        self._frames = 0
        self._behind = np.zeros(2)
        self._reacquiring = False
        self._blocked: float | None = None
        self._moved_past_wall = False
        self._accepted_place: np.ndarray | None = None

    @property
    def followed(self) -> bool:
        """Whether the place head is followed: not rejected most of the time, or persistently
        disagreeing with a credential."""
        return sum(self._rejected) < DEGRADED_REJECTIONS or self._persistent

    @property
    def status(self) -> str:
        """``reacquiring`` while a persistently disagreeing place head is followed to where it
        is, ``degraded`` while the place head is not followed (the estimate steps with the
        camera), else ``guarded``.
        """
        if self._reacquiring:
            return "reacquiring"
        return "guarded" if self.followed else "degraded"

    @property
    def blocked_progress(self) -> float | None:
        """The share of its step along the way that the camera of the next block keeps after a
        newest latent frame the view did not follow (at least :data:`CONTINUATION_FLOOR`);
        ``None`` when the view moved."""
        return self._blocked

    @property
    def accepted_place(self) -> np.ndarray | None:
        """``[3]`` the place term ``a_f`` of the newest latent frame (u) if the place head is
        followed and ``a_f`` lay inside the gate; else None."""
        return None if self._accepted_place is None else self._accepted_place.copy()

    @property
    def moved_past_wall(self) -> bool:
        """Whether the motion head said the newest latent frame moved on where a wall in the way
        stopped its step: a wall that is not there."""
        return self._moved_past_wall

    def _persists(self) -> bool:
        """Whether the place head disagrees persistently with the estimate, with a credential,
        and moves with the player."""
        if len(self._apart) < PERSISTENT_FRAMES or sum(self._apart) < PERSISTENT_NEAR:
            return False
        if sum(self._agreed) < CREDENTIAL_NEAR and sum(self._edge) < EDGE_NEAR:
            return False
        way = float(np.linalg.norm(np.sum(self._cameras, axis=0)))
        moved = float(np.linalg.norm(np.sum(self._head_steps, axis=0)))
        return way < GATE_XY_U / 2 or moved >= HEAD_MOVES * way

    def step(
        self,
        place: np.ndarray,
        displacement: np.ndarray,
        prior_step: np.ndarray,
        camera_step: np.ndarray,
    ) -> np.ndarray:
        """The estimate ``E_f`` of the next latent frame (frames are fed in order from 1).

        Args:
            place (np.ndarray): ``[3]`` the place-head estimate ``A_f``, u.
            displacement (np.ndarray): ``[3]`` the motion-head displacement ``Delta_f``, units of
                64 u.
            prior_step (np.ndarray): ``[3]`` the physics prior's displacement ``d_f`` over the
                latent frame (video frames ``4f - 4 .. 4f``), u; its height is exactly 0 on the
                ground.
            camera_step (np.ndarray): ``[3]`` ``c_f``, what the camera moved over the frame:
                ``prior_step`` stopped at a wall in the way and slowed by a continuation
                (``prior_step`` itself where there is none), u.

        Returns:
            np.ndarray: ``[3]`` float64 ``E_f``, u.
        """
        a = np.asarray(place, np.float64).reshape(3) - self.a0 + self.p0
        measured = np.asarray(displacement, np.float64).reshape(3)[:2] * DISPLACEMENT_UNIT_U
        d = np.array(prior_step, np.float64).reshape(3)
        camera = np.array(camera_step, np.float64).reshape(3)
        step = camera.copy()
        length, by_camera = float(np.linalg.norm(d[:2])), float(np.linalg.norm(camera[:2]))
        view = float(np.linalg.norm(measured))
        standing = length < IDLE_STEP_U
        grounded = standing and d[2] == 0.0
        self._blocked, self._moved_past_wall = None, False
        share, seen, checked = 1.0, 1.0, False
        if standing:
            step[:2] = 0.0
        elif length >= STUCK_MIN_STEP_U:
            seen = view / length
            # the view moved on, for two latent frames, past a camera held back
            if (
                np.linalg.norm(d[:2] - camera[:2]) >= IDLE_STEP_U
                and min(seen, self._seen) >= PASSED_SHARE
            ):
                self._moved_past_wall = True
                step[:2] = d[:2] * min(seen, 1.0)
            elif by_camera >= PROBE_MIN_U:
                share, checked = view / by_camera, True
                # the view stood still, or moved on, for two latent frames: one is the head's glitch
                moved = (max(share, self._share) - STILL_SHARE) / (MOVED_SHARE - STILL_SHARE)
                moved = float(np.clip(moved, 0.0, 1.0))
                step[:2] *= moved
                if moved < 1.0:
                    self._blocked = CONTINUATION_FLOOR + (1.0 - CONTINUATION_FLOOR) * moved
        elif self.dead_zone_u > 0.0 and by_camera >= IDLE_STEP_U:
            share, checked = (view if view >= self.dead_zone_u else 0.0) / by_camera, True
            previous = self._share if self._checked else 0.0
            moved = (max(share, previous) - STILL_SHARE) / (MOVED_SHARE - STILL_SHARE)
            moved = float(np.clip(moved, 0.0, 1.0))
            step[:2] *= moved
            if moved < 1.0:
                self._blocked = CONTINUATION_FLOOR + (1.0 - CONTINUATION_FLOOR) * moved
        self._share, self._seen, self._checked = share, seen, checked
        if by_camera < IDLE_STEP_U and not self._moved_past_wall:
            step[:2] = 0.0  # a camera held at a wall by a stop stands still, to the bit
        way = camera[:2] * (not standing)
        self._frames += 1
        if self._frames % BLOCK == 1:
            self._behind = np.zeros(2)
        self._behind += way - step[:2]
        behind = float(np.linalg.norm(self._behind))
        if behind > BLOCK_LAG_U:
            catch = self._behind * (1.0 - BLOCK_LAG_U / behind)
            step[:2] += catch
            self._behind -= catch

        q = self._position + step
        landing = grounded and self._ground is not None and q[2] != self._ground
        if landing:
            q[2] = np.clip(self._ground, q[2] - REACQUIRE_RATE_U, q[2] + REACQUIRE_RATE_U)
        innovation = a - q
        inside_xy = float(np.linalg.norm(innovation[:2])) <= GATE_XY_U
        inside_z = abs(innovation[2]) <= GATE_Z_U
        accepted = inside_xy and inside_z
        self._rejected.append(not accepted)
        self._agreed.append(accepted)
        self._edge.append(
            not standing
            and self.walkable.clearance(self._position[:2], d[:2] / length, length) < length
        )
        self._places.append(a)
        self._offsets.append(innovation)
        track = np.median(np.stack(self._places), axis=0)[:2]
        offset = np.median(np.stack(self._offsets), axis=0)
        head = q + offset
        head_step = np.zeros(2) if self._track is None else track - self._track
        jumped = bool(np.linalg.norm(head_step) > GATE_XY_U)
        self._track = track
        self._head_steps.append(np.zeros(2) if jumped else head_step)
        self._cameras.append(way)
        self._apart.append(not jumped and _outside_gate(offset))
        self._persistent = self._persists()
        self._accepted_place = a.copy() if accepted and self.followed else None

        stride = by_camera * float(not standing)
        target, rate_xy, rate_z = a.copy(), 0.0, 0.0
        self._reacquiring = False
        if self.followed and not standing:
            if inside_xy:
                rate_xy = RATE_U + RATE_XY_GAIN * stride
            elif self._persistent:
                target[:2], rate_xy = head[:2], REACQUIRE_RATE_U
            if inside_z:
                rate_z = RATE_U + RATE_Z_GAIN * stride
            elif self._persistent:
                target[2], rate_z = head[2], REACQUIRE_RATE_U
            self._reacquiring = self._persistent and not accepted
        if landing:
            rate_z = 0.0

        correction = FUSION_WEIGHT * (target - q)
        pull = float(np.linalg.norm(correction[:2]))
        if pull > rate_xy:
            correction[:2] *= rate_xy / pull
        correction[2] = np.clip(correction[2], -rate_z, rate_z)
        estimate = q + correction
        run = estimate[:2] - self._position[:2]
        distance = float(np.linalg.norm(run))
        if distance > 0.0:
            reach = min(distance, self.top_step_u)
            reach = min(reach, self.walkable.clearance(self._position[:2], run / distance, reach))
            if reach < distance:
                estimate[:2] = self._position[:2] + run * (reach / distance)
        self._position = estimate
        if not standing:
            self._ground = None
        elif grounded and not landing:
            self._ground = float(self._position[2])
        return self._position.copy()
