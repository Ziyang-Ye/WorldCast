"""Extrapolation within a block (Sec. 3.4; App. "State model in detail", Within a block).

A block has no frames when it starts, so the cameras of its four latent frames and the other
players' positions over it are extrapolated from the controls. A player's position over block ``s``
(latent frames ``s .. s+3``) is the position it published at latent frame ``s - 1`` plus the physics
prior's displacement over the block: the movement keys integrated at 64 Hz with per-key walking
speeds, jump and gravity, and no collisions. Its view angles are the round-start angles plus the
integral of its own view controls.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from worldcast.data.camera import c2w_from_state_rows
from worldcast.data.controls import CONTROL_BUTTONS
from worldcast.data.game import GRAVITY_U_PER_S2
from worldcast.data.latents import (
    BLOCK,
    FPS,
    VIDEO_FRAMES_PER_BLOCK,
    block_starts,
    last_video_frames,
    video_frames_of,
)
from worldcast.data.recordings import ALIVE_INDEX
from worldcast.sampling.window import KEEP_LATENTS, KEEP_VIDEO, roll

from .states import integrate_camera_angles, pack_substeps

__all__ = [
    "FLIGHT_SUBSTEPS",
    "GAIT_COLUMNS",
    "JUMP_COLUMN",
    "JUMP_U_PER_S",
    "MOVEMENT_COLUMNS",
    "ClientCameras",
    "Motion",
    "PhysicsPrior",
    "PositionTrack",
    "PredictedStateTable",
    "PriorState",
    "extrapolate",
    "physics_prior_displacement",
    "physics_prior_track",
    "player_motion",
]

#: Jump impulse of the engine, u/s.
JUMP_U_PER_S = 260.0
#: Substeps a jump stays airborne.
FLIGHT_SUBSTEPS = 40
#: Columns of the movement keys (forward, back, left, right), of the gait keys (walk, duck) and of
#: jump in the substep buttons.
MOVEMENT_COLUMNS = tuple(
    CONTROL_BUTTONS.index(name) for name in ("forward", "back", "move_left", "move_right")
)
GAIT_COLUMNS = tuple(CONTROL_BUTTONS.index(name) for name in ("speed", "duck"))
JUMP_COLUMN = CONTROL_BUTTONS.index("jump")


@dataclass(frozen=True)
class PhysicsPrior:
    """The fitted prior (``configs/state_model/physics_prior.json``).

    Attributes:
        table (tuple[tuple[float, ...], ...]): target speed ``table[gait, hold]``, u/s; gait = 2
            walk + duck, hold = substeps the keys have been held, capped at the last column.
        alpha_up (float): velocity relaxation rate while speeding up.
        alpha_down (float): velocity relaxation rate while slowing down.
    """

    table: tuple[tuple[float, ...], ...]
    alpha_up: float
    alpha_down: float

    @classmethod
    def load(cls, path: str | Path) -> "PhysicsPrior":
        """Read the prior from its JSON file."""
        d = json.loads(Path(path).read_text())
        return cls(
            table=tuple(tuple(float(v) for v in row) for row in d["table"]),
            alpha_up=float(d["alpha_up"]),
            alpha_down=float(d["alpha_down"]),
        )

    @property
    def top_speed(self) -> float:
        """The fastest target speed of the table, u/s: how fast a player runs."""
        return max(max(row) for row in self.table)


def _movement(
    controls: np.ndarray, yaw: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The movement keys per substep: the wish direction in the world ``[B, T Q, 3]`` float32 (zero
    while no key is held), the gait ``[B, T Q]`` (2 walk + duck), the jump key and the valid flag.
    The yaw's cosine and sine are torch's; the rest is exact in any arithmetic (the wish's length
    is 0, 1 or the float32 square root of 2)."""
    batch, frames, per_frame = controls.shape[:3]
    valid = controls[..., -1] > 0.5
    move = (np.abs(controls[..., list(MOVEMENT_COLUMNS)]) > 0.5).astype(np.float32)
    forward = move[..., 0] - move[..., 1]
    strafe = move[..., 3] - move[..., 2]
    gait_held = np.abs(controls[..., list(GAIT_COLUMNS)]) > 0.5
    jumped = np.abs(controls[..., JUMP_COLUMN]) > 0.5
    magnitude = np.sqrt(forward * forward + strafe * strafe)
    moving = (magnitude > np.float32(1e-6)).astype(np.float32)
    magnitude = np.maximum(magnitude, np.float32(1e-6))
    f, r = forward / magnitude, strafe / magnitude
    psi = torch.deg2rad(yaw[:, :, None].float())
    cos, sin = torch.cos(psi).numpy(), torch.sin(psi).numpy()
    direction = np.stack(
        [(f * cos + r * sin) * moving, (f * sin - r * cos) * moving, np.zeros_like(f)], -1
    ).reshape(batch, frames * per_frame, 3)
    gait_index = (gait_held[..., 0] * 2 + gait_held[..., 1]).astype(np.int64).reshape(batch, -1)
    return direction, gait_index, jumped.reshape(batch, -1), valid.reshape(batch, -1)


@dataclass(frozen=True)
class PriorState:
    """The integrator of the physics prior between two substeps, per player
    (:func:`physics_prior_displacement`).

    Attributes:
        velocity (torch.Tensor): ``[B, 3]`` float32, u/s.
        hold (torch.Tensor): ``[B]`` long, the substeps a movement key has been held, capped at the
            speed table's last column.
        flight (torch.Tensor): ``[B]`` long, the substeps a jump stays airborne.
    """

    velocity: torch.Tensor
    hold: torch.Tensor
    flight: torch.Tensor

    @classmethod
    def at_rest(cls, batch: int) -> "PriorState":
        """``batch`` players standing, no key held."""
        return cls(
            torch.from_numpy(np.zeros((batch, 3), np.float32)),
            torch.from_numpy(np.zeros(batch, np.int64)),
            torch.from_numpy(np.zeros(batch, np.int64)),
        )


@lru_cache(maxsize=None)
def _jump_speeds(fall: float) -> tuple[float, ...]:
    """The vertical speed on the ``FLIGHT_SUBSTEPS`` airborne substeps of a jump, float32 values:
    the impulse, then less ``fall`` per substep, subtracted one by one in float32, the velocity's
    dtype, as the engine does."""
    speeds = [torch.tensor(JUMP_U_PER_S, dtype=torch.float32)]
    for _ in range(FLIGHT_SUBSTEPS - 1):
        speeds.append(speeds[-1] - fall)
    return tuple(float(speed) for speed in speeds)


def _flights(
    jumped: np.ndarray, live: np.ndarray, flight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where each player is airborne: ``[B, N]`` int64 the substeps left in the air as each valid
    substep starts (``FLIGHT_SUBSTEPS`` at a takeoff, 0 on the ground), and ``[B]`` those left
    after the last. A valid substep with jump held takes off on the ground; the flight counts
    valid substeps only."""
    started = np.zeros(live.shape, np.int64)
    left = np.array(flight, np.int64)
    for b in range(len(started)):
        valid = np.flatnonzero(live[b])
        takeoffs = valid[jumped[b, valid]]
        k, airborne = 0, int(left[b])
        while k < len(valid):
            if airborne == 0:
                later = takeoffs[takeoffs >= valid[k]]
                if not len(later):
                    break
                k, airborne = int(np.searchsorted(valid, later[0])), FLIGHT_SUBSTEPS
            take = min(airborne, len(valid) - k)
            started[b, valid[k : k + take]] = np.arange(airborne, airborne - take, -1)
            k, airborne = k + take, airborne - take
        left[b] = airborne
    return started, left


#: The relaxation of the planar velocity compares the keys' speed with the velocity's (``speed >=
#: |v|``, torch's float32 norm, within 2 units of float32 rounding, 2^-24, of the exact norm). Where
#: a float32 estimate of ``|v|^2`` lies farther than this share from ``speed^2`` the comparison is
#: read off it, otherwise off torch's norm itself.
_NORM_MARGIN = 2.0**-21
#: Keys' speeds below this (u/s, but 0) are compared on torch's norm alone: the float32 squares of
#: velocities about as slow underflow.
_NORM_FLOOR = 1e-15


def _relaxed_velocity(
    velocity: np.ndarray,
    target: np.ndarray,
    speed: np.ndarray,
    relaxes: np.ndarray,
    prior: PhysicsPrior,
) -> np.ndarray:
    """The planar velocity ``[B, N, 2]`` float32 after each substep: from ``velocity`` ``[B, 2]``
    it relaxes towards ``target`` ``[B, N, 2]`` on the substeps ``relaxes`` ``[B, N]``, at
    ``alpha_up`` while the keys' ``speed`` ``[B, N]`` (torch's norm of the target) is at least the
    velocity's own, else at ``alpha_down``; float32 arithmetic, operation by operation."""
    up, down = np.float32(prior.alpha_up), np.float32(prior.alpha_down)
    speed = np.ascontiguousarray(speed.T)
    wide = speed.astype(np.float64)
    usable = wide >= _NORM_FLOOR
    # a velocity whose squared estimate is at most ``below`` is surely no faster than the keys,
    # one beyond ``above`` surely faster (at a speed of 0: any velocity but 0)
    below = np.where(usable, (wide / (1.0 + _NORM_MARGIN)) ** 2, -1.0)
    above = np.where(usable, (wide / (1.0 - _NORM_MARGIN)) ** 2, np.where(wide == 0, 0, np.inf))
    every, some = relaxes.all(0), relaxes.any(0)
    target = np.ascontiguousarray(target.transpose(1, 2, 0))  # [N, 2, B]
    relaxes = np.ascontiguousarray(relaxes.T)
    out = np.empty(target.shape, np.float32)
    planar = np.array(velocity.T, np.float32)  # [2, B]
    norms: dict[int, float] = {}  # torch's norm of a velocity, by its bits
    for i in range(len(target)):
        if some[i]:
            squared = planar * planar
            squared = squared[0] + squared[1]
            speeding, slowing = squared <= below[i], squared > above[i]
            step = target[i] - planar
            if not (speeding | slowing).all():
                # a velocity of 0 is no faster than any keys; torch's norm decides the rest where
                # the rates move the velocity apart
                speeding = speeding | ~planar.any(0)
                open_ = ~(speeding | slowing)
                if open_.any():
                    open_ &= (
                        (planar + up * step).view(np.int32) != (planar + down * step).view(np.int32)
                    ).any(0)
                if open_.any():
                    speeding = speeding.copy()
                    rows = np.flatnonzero(open_)
                    speeding[rows] = speed[i, rows] >= _torch_norms(planar, rows, norms)
            relaxed = planar + np.where(speeding, up, down) * step
            planar = relaxed if every[i] else np.where(relaxes[i], relaxed, planar)
        out[i] = planar
    return out.transpose(2, 0, 1)


def _torch_norms(planar: np.ndarray, rows: np.ndarray, norms: dict[int, float]) -> np.ndarray:
    """Torch's float32 norm of the velocities ``planar`` ``[2, B]`` at ``rows``, each read once
    per value (a velocity at the keys' speed stays a few units of rounding short of it, the same
    bits substep after substep, and only torch's norm says which side of the keys' speed it is)."""
    velocities = np.ascontiguousarray(planar.T)
    bits = velocities.view(np.uint64)[:, 0].tolist()
    if any(bits[row] not in norms for row in rows):
        norm = torch.linalg.norm(torch.from_numpy(velocities), dim=-1, keepdim=True)
        norms.update(zip(bits, norm.numpy()[:, 0].tolist()))
    return np.array([norms[bits[row]] for row in rows], np.float32)


@torch.no_grad()
def physics_prior_displacement(
    controls: torch.Tensor,
    yaw: torch.Tensor,
    *,
    prior: PhysicsPrior,
    state: PriorState | None = None,
) -> tuple[torch.Tensor, PriorState]:
    """Integrate the movement keys at 64 Hz (App. "State model in detail", Within a block): the
    speed ramp towards the keys' walking speed, jump and gravity.

    How long a movement key has been held and when a jump flies depend on the keys alone, so they
    are found for every substep at once; the velocity then relaxes towards the keys' speed
    substep by substep. The arithmetic is float32 on the CPU, operation by operation: numpy's, with
    torch's cosine, sine and norms.

    Args:
        controls (torch.Tensor): ``[B, T, Q, A]`` packed substeps per video frame, valid flag last
            (an invalid substep freezes the integrator; the player moves on at its velocity).
        yaw (torch.Tensor): ``[B, T]`` degrees, held across a video frame's substeps.
        prior (PhysicsPrior): the fitted prior.
        state (PriorState | None): the integrator before the first substep, the state a previous
            call returned; at rest by default.

    Returns:
        tuple[torch.Tensor, PriorState]: the displacement during each video frame ``[B, T, 3]``
        float32, u, and the integrator after the last substep (on ``controls``' device).
    """
    batch, frames, per_frame = (int(v) for v in controls.shape[:3])
    substeps = frames * per_frame
    dt = 1.0 / (FPS * per_frame)
    device = controls.device
    direction, gait, jumped, live = _movement(_host(controls), torch.from_numpy(_host(yaw)))
    state = PriorState.at_rest(batch) if state is None else state
    velocity = _host(state.velocity).astype(np.float32)
    hold, flight = _host(state.hold), _host(state.flight)
    table = np.asarray(prior.table, np.float32)

    # the speed ramp keeps running across direction changes, resets when the player stops: the
    # valid substeps with a movement key held since the last valid one without
    wishing = np.abs(direction).sum(-1) > 1e-6
    count = np.cumsum(live & wishing, axis=1, dtype=np.int64)
    since = np.maximum.accumulate(np.where(live & ~wishing, count, -hold[:, None]), axis=1)
    held = np.minimum(count - since, table.shape[1] - 1)
    want = direction * table[gait, held][..., None]
    wanted = torch.linalg.norm(torch.from_numpy(want)[..., :2], dim=-1, keepdim=True).numpy()

    started, flight = _flights(jumped, live, flight)
    airborne = started > 0
    speeds = np.asarray(_jump_speeds(GRAVITY_U_PER_S2 * dt), np.float32)
    lifted = np.where(
        airborne,
        speeds[np.clip(FLIGHT_SUBSTEPS - started, 0, FLIGHT_SUBSTEPS - 1)],
        np.float32(0.0),
    )
    newest = np.maximum.accumulate(np.where(live, np.arange(substeps), -1), axis=1)
    vertical = np.where(
        newest >= 0, np.take_along_axis(lifted, np.maximum(newest, 0), 1), velocity[:, 2:3]
    )

    # on the ground the horizontal velocity relaxes towards the keys' speed, faster speeding up
    horizontal = _relaxed_velocity(
        velocity[:, :2], want[..., :2], wanted[..., 0], live & ~airborne, prior
    )
    velocities = np.concatenate([horizontal, vertical[..., None]], -1)

    during = (velocities * np.float32(dt)).reshape(batch, frames, per_frame, 3)
    moved = np.zeros_like(during[:, :, 0])
    for i in range(per_frame):
        moved = moved + during[:, :, i]
    after = PriorState(
        *(_on(device, value) for value in (velocities[:, -1].copy(), held[:, -1].copy(), flight))
    )
    return _on(device, moved), after


def _host(tensor: torch.Tensor) -> np.ndarray:
    """``tensor`` as a numpy array (a view of it on the CPU)."""
    return tensor.numpy() if tensor.device.type == "cpu" else tensor.cpu().numpy()


def _on(device: torch.device, array: np.ndarray) -> torch.Tensor:
    """``array`` as a tensor on ``device`` (a view of it on the CPU)."""
    tensor = torch.from_numpy(array)
    return tensor if device.type == "cpu" else tensor.to(device)


@torch.no_grad()
def physics_prior_track(
    controls: torch.Tensor, yaw: torch.Tensor, start: torch.Tensor, *, prior: PhysicsPrior
) -> torch.Tensor:
    """Integrate the movement keys at 64 Hz from rest (:func:`physics_prior_displacement`).

    Args:
        controls (torch.Tensor): ``[B, T, Q, A]`` packed substeps per video frame, valid flag last
            (an invalid substep freezes the integrator).
        yaw (torch.Tensor): ``[B, T]`` degrees, held across a video frame's substeps.
        start (torch.Tensor): ``[B, 1, 3]`` the position at video frame 0, u.
        prior (PhysicsPrior): the fitted prior.

    Returns:
        torch.Tensor: ``[B, T, 3]`` positions, u; video frame 0 adds no displacement (its keys
        precede the window).
    """
    moved, _ = physics_prior_displacement(controls, yaw, prior=prior)
    displacement = torch.cat([torch.zeros_like(moved[:, :1]), moved[:, 1:]], dim=1)
    return start + torch.cumsum(displacement, dim=1)


@dataclass
class Motion:
    """What a player's own controls say, per video frame of the round (all float64).

    Attributes:
        yaw (np.ndarray): ``[T]`` degrees.
        pitch (np.ndarray): ``[T]`` degrees.
        displacement (np.ndarray): ``[T, 3]`` the prior's displacement from video frame 0, u.
        start_state (np.ndarray): ``[6]`` the round-start player-state row.
        spread (np.ndarray | None): ``[T, 3]`` an offset (u) added to the positions extrapolated to
            a video frame, relative to its value at the video frame they start from (live play,
            ``live.hop: spread``); None: none.
    """

    yaw: np.ndarray
    pitch: np.ndarray
    displacement: np.ndarray
    start_state: np.ndarray
    spread: np.ndarray | None = None


def player_motion(batch: Mapping[str, torch.Tensor], slot: int, *, prior: PhysicsPrior) -> Motion:
    """Integrate the controls of the player at ``slot`` once over the round.

    Args:
        batch (Mapping[str, torch.Tensor]): the whole round of one client, batch size 1
            (``player_states`` ``[1, P, T, 6]`` and the control substeps).
        slot (int): the player.
        prior (PhysicsPrior): the fitted prior.
    """
    states = batch["player_states"]
    if int(states.shape[0]) != 1:
        raise ValueError(f"one client's round (batch size 1), got {int(states.shape[0])}")
    packed = pack_substeps(batch["player_control_substeps"], batch["player_control_substep_valid"])
    yaw, pitch = integrate_camera_angles(states[:, :, 0].float(), packed)
    displacement = physics_prior_track(
        packed[:, slot].float(), yaw[:, slot].float(), torch.zeros(1, 1, 3), prior=prior
    )[0]
    return Motion(
        yaw=yaw[0, slot].double().numpy(),
        pitch=pitch[0, slot].double().numpy(),
        displacement=displacement.double().numpy(),
        start_state=states[0, slot, 0].double().numpy(),
    )


class PositionTrack:
    """A player's published positions, one per latent frame of its window, growing block by block
    from latent frame 0, the round-start position.

    Args:
        media_id (str): the player's media id.
        start_frame (int): source frame of latent frame 0.
        p0 (np.ndarray): ``[3]`` the round-start position, u.
    """

    #: The arrays :meth:`slide` moves: those with a latent frame axis.
    SLID = ("xyz",)

    def __init__(self, media_id: str, start_frame: int, p0: np.ndarray) -> None:
        self.media_id, self.start_frame = str(media_id), int(start_frame)
        self.xyz = np.asarray(p0, np.float64).reshape(1, 3).copy()

    def __len__(self) -> int:
        return len(self.xyz)

    def append(self, latent_frames: Sequence[int], xyz: np.ndarray) -> None:
        """Add the positions ``xyz`` ``[n, 3]``, u, published at the next latent frames
        ``latent_frames`` (consecutive)."""
        latent_frames = [int(f) for f in latent_frames]
        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        expected = list(range(len(self), len(self) + len(latent_frames)))
        if latent_frames != expected or len(xyz) != len(latent_frames):
            raise ValueError(
                f"{self.media_id}: latent frames {latent_frames} after {len(self)} positions"
            )
        if not np.isfinite(xyz).all():
            raise ValueError(f"{self.media_id}: non-finite position")
        self.xyz = np.concatenate([self.xyz, xyz], 0)

    def slide(self) -> None:
        """Drop the positions of the block after the first :data:`KEEP_LATENTS` latent frames: the
        round has slid by a block (:func:`~worldcast.sampling.window.slide_window_entries`) and
        the latent frames after them moved down."""
        for name in self.SLID:
            values = getattr(self, name)
            setattr(
                self, name, np.concatenate([values[:KEEP_LATENTS], values[KEEP_LATENTS + BLOCK :]])
            )

    def latest_published(self, latent_frame: int) -> int | None:
        """The newest published latent frame at or before ``latent_frame``, or None when the
        player published none within a latent frame of it."""
        latest = min(int(latent_frame), len(self) - 1)
        return latest if 0 <= latest and int(latent_frame) - latest <= 1 else None

    def require_published(self, latent_frame: int) -> int:
        """:meth:`latest_published`; raises when the player published none within a latent frame."""
        latest = self.latest_published(latent_frame)
        if latest is None:
            raise ValueError(
                f"{self.media_id}: no position published at latent frame {int(latent_frame)}"
            )
        return latest


def extrapolate(
    track: PositionTrack, motion: Motion, published: np.ndarray, video_frames: np.ndarray
) -> np.ndarray:
    """The positions published at latent frames, moved to video frames by the prior.

    Args:
        track (PositionTrack): the player's published positions.
        motion (Motion): what its controls say.
        published (np.ndarray): ``[n]`` int, the published latent frame each position starts from.
        video_frames (np.ndarray): ``[n]`` int, the video frames to extrapolate to.

    Returns:
        np.ndarray: ``[n, 3]`` float64: the published position plus the prior's displacement from
        that latent frame's last video frame to the video frame.
    """
    published = np.asarray(published, dtype=np.int64)
    video_frames = np.asarray(video_frames, dtype=np.int64)
    starts = last_video_frames(0, len(track))[published]
    start = motion.displacement[starts]
    xyz = track.xyz[published] + (motion.displacement[video_frames] - start)
    if motion.spread is None:
        return xyz
    return xyz + (motion.spread[video_frames] - motion.spread[starts])


class ClientCameras:
    """The client's cameras from its position track (App. "State model in detail", Within a
    block).

    Latent frames before block ``s`` sit at their own published positions, block ``s`` is
    extrapolated from the position of latent frame ``s - 1``, later latent frames hold.

    Args:
        track (PositionTrack): the client's published positions.
        motion (Motion): the client's own controls.
        latent_frames (int): latent frames of the client's window of the round.
    """

    def __init__(self, track: PositionTrack, motion: Motion, latent_frames: int) -> None:
        self.track, self.motion, self.latent_frames = track, motion, int(latent_frames)
        self._cache: tuple[int, torch.Tensor] | None = None

    def _state_rows(self, first: int, count: int, published: np.ndarray) -> np.ndarray:
        """``[count, 6]`` player-state rows of the latent frames ``first ..``, each extrapolated
        from the latent frame ``published`` names for it."""
        last = last_video_frames(first, count)
        xyz = extrapolate(self.track, self.motion, published, last)
        yaw, pitch, alive = self.motion.yaw[last], self.motion.pitch[last], np.ones(count)
        return np.column_stack([xyz, yaw, pitch, alive])

    def _known_rows(self, count: int) -> np.ndarray:
        published = [self.track.require_published(f) for f in range(count)]
        return self._state_rows(0, count, np.asarray(published, dtype=np.int64))

    def block_rows(self, s: int) -> np.ndarray:
        """``[4, 6]`` float64 player-state rows of block ``s``, extrapolated from the position
        published at latent frame ``s - 1``."""
        published = self.track.require_published(s - 1)
        return self._state_rows(s, BLOCK, np.full(BLOCK, published))

    def _c2w(self, rows: np.ndarray) -> torch.Tensor:
        return c2w_from_state_rows(rows.astype(np.float32)).float()

    @staticmethod
    def _block_then_held(
        out: np.ndarray | torch.Tensor, s: int, block: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """``out`` (rows or cameras, one per latent frame) with block ``s``'s ``block`` written and
        every later latent frame holding its last."""
        out[s : s + BLOCK] = block
        out[s + BLOCK :] = out[s + BLOCK - 1]
        return out

    def for_block(self, s: int) -> torch.Tensor:
        """``[N, 4, 4]`` cameras known at the start of block ``s`` (its rays and retrieval)."""
        s = int(s)
        out = np.zeros((self.latent_frames, 6), dtype=np.float64)
        out[:s] = self._known_rows(s)
        return self._c2w(self._block_then_held(out, s, self.block_rows(s)))

    def as_predicted(self) -> torch.Tensor:
        """``[N, 4, 4]``: every latent frame as extrapolated when its block started, for the blocks
        whose start position is published; later frames hold the newest (the rays of the first six
        blocks and the cameras of their memory entries)."""
        if self._cache is not None and self._cache[0] == len(self.track):
            return self._cache[1]
        out = np.zeros((self.latent_frames, 6), dtype=np.float64)
        out[0] = self._known_rows(1)[0]
        last = 0
        for s in block_starts(self.latent_frames):
            if len(self.track) < s:
                break
            out[s : s + BLOCK] = self.block_rows(s)
            last = s + BLOCK - 1
        out[last + 1 :] = out[last]
        self._cache = (len(self.track), self._c2w(out))
        return self._cache[1]


class PredictedStateTable:
    """The round's player-state table under predicted states, rewritten block by block in place.

    Slots with a position track (the round's clients) take, on the video frames of block ``s``,
    their extrapolation from the position published at latent frame ``s - 1``; their later alive
    video frames hold the newest value. A client that published none (finished or late) holds its
    last value. Every other slot, and every column but ``x y z``, keeps the recording.

    Args:
        states (torch.Tensor): ``[1, P, T, 6]`` float32 recorded table, rewritten in place.
        slots (Mapping[int, tuple[PositionTrack, Motion]]): track and motion of each client's slot.
    """

    #: The arrays :meth:`slide` moves: those with a video frame axis; the table itself is the round
    #: batch's, which slides with it.
    SLID = ("alive",)

    def __init__(
        self, states: torch.Tensor, slots: Mapping[int, tuple[PositionTrack, Motion]]
    ) -> None:
        self.states = states
        # the alive column is never rewritten
        self.alive = states[0, :, :, ALIVE_INDEX].numpy() > 0.5
        self.slots = dict(slots)
        self._held = {
            p: np.asarray(track.xyz[0], np.float64) for p, (track, _) in self.slots.items()
        }

    def slide(self) -> None:
        """The round has slid by a block: move the alive flags with the table's rows."""
        for name in self.SLID:
            roll(getattr(self, name), 1, VIDEO_FRAMES_PER_BLOCK, KEEP_VIDEO)

    def track_of(self, media_id: str) -> PositionTrack | None:
        """The position track of ``media_id``, or None for a slot without one."""
        for track, _ in self.slots.values():
            if track.media_id == str(media_id):
                return track
        return None

    def advance(self, s: int) -> dict[int, np.ndarray]:
        """Write the video frames of block ``s`` (latent frames ``s .. s+3``); returns the video
        frames written, by slot."""
        s = int(s)
        frames = np.asarray(video_frames_of(s, BLOCK))
        frames = frames[frames < int(self.states.shape[2])]
        touched = {}
        for p, (track, motion) in self.slots.items():
            block = frames[self.alive[p, frames]]
            if not len(block):
                continue
            published = track.latest_published(s - 1)
            if published is None:  # nothing published at s - 1: the client holds its last position
                values = np.repeat(self._held[p][None], len(block), 0)
            else:
                values = extrapolate(track, motion, np.full(len(block), published), block)
            states = self.states.numpy()
            states[0, p, block, :3] = values
            self._held[p] = np.asarray(values[-1], np.float64)
            later = np.flatnonzero(self.alive[p])
            later = later[later > frames[-1]]
            if len(later):
                states[0, p, later, :3] = values[-1]
            touched[p] = np.concatenate([block, later])
        return touched
