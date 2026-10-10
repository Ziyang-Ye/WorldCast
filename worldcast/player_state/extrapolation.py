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
    block_starts,
    last_video_frames,
    video_frames_of,
)
from worldcast.data.recordings import ALIVE_INDEX

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


def _to_world(fru: torch.Tensor, yaw_degrees: torch.Tensor) -> torch.Tensor:
    psi = torch.deg2rad(yaw_degrees.float())
    cos, sin = torch.cos(psi), torch.sin(psi)
    f, r, u = fru[..., 0], fru[..., 1], fru[..., 2]
    return torch.stack([f * cos + r * sin, f * sin - r * cos, u], dim=-1)


def _movement(
    controls: torch.Tensor, yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The movement keys per substep: the wish direction in the world ``[B, T Q, 3]`` (zero while
    no key is held), the gait ``[B, T Q]`` (2 walk + duck), the jump key and the valid flag."""
    batch, frames, per_frame = (int(v) for v in controls.shape[:3])
    valid = controls[..., -1] > 0.5
    move = (controls[..., list(MOVEMENT_COLUMNS)].abs() > 0.5).float()
    forward = move[..., 0] - move[..., 1]
    strafe = move[..., 3] - move[..., 2]
    gait_held = controls[..., list(GAIT_COLUMNS)].abs() > 0.5
    jumped = controls[..., JUMP_COLUMN].abs() > 0.5
    wish = torch.stack([forward, strafe], dim=-1)
    magnitude = wish.norm(dim=-1, keepdim=True)
    unit = wish / magnitude.clamp_min(1e-6)
    moving = (magnitude.squeeze(-1) > 1e-6).unsqueeze(-1)
    direction = _to_world(
        torch.cat([unit, torch.zeros_like(unit[..., :1])], dim=-1), yaw[:, :, None]
    )
    direction = (direction * moving.to(direction.dtype)).reshape(batch, frames * per_frame, 3)
    gait_index = (gait_held[..., 0].long() * 2 + gait_held[..., 1].long()).reshape(batch, -1)
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
            torch.zeros(batch, 3),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
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
    jumped: torch.Tensor, live: torch.Tensor, flight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where each player is airborne: ``[B, N]`` long the substeps left in the air as each valid
    substep starts (``FLIGHT_SUBSTEPS`` at a takeoff, 0 on the ground), and ``[B]`` those left
    after the last. A valid substep with jump held takes off on the ground; the flight counts
    valid substeps only."""
    started = np.zeros(tuple(live.shape), np.int64)
    left = flight.cpu().numpy().copy()
    jumped, live = jumped.cpu().numpy(), live.cpu().numpy()
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
    return torch.from_numpy(started).to(flight.device), torch.from_numpy(left).to(flight.device)


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
    substep by substep.

    Args:
        controls (torch.Tensor): ``[B, T, Q, A]`` packed substeps per video frame, valid flag last
            (an invalid substep freezes the integrator; the player moves on at its velocity).
        yaw (torch.Tensor): ``[B, T]`` degrees, held across a video frame's substeps.
        prior (PhysicsPrior): the fitted prior.
        state (PriorState | None): the integrator before the first substep, the state a previous
            call returned; at rest by default.

    Returns:
        tuple[torch.Tensor, PriorState]: the displacement during each video frame ``[B, T, 3]``
        float32, u, and the integrator after the last substep.
    """
    batch, frames, per_frame = (int(v) for v in controls.shape[:3])
    substeps = frames * per_frame
    dt = 1.0 / (FPS * per_frame)
    direction, gait, jumped, live = _movement(controls, yaw)
    device, dtype = direction.device, direction.dtype
    state = PriorState.at_rest(batch) if state is None else state
    velocity = state.velocity.to(device, dtype)
    hold, flight = state.hold.to(device), state.flight.to(device)
    table = torch.as_tensor(prior.table, dtype=dtype, device=device)

    # the speed ramp keeps running across direction changes, resets when the player stops: the
    # valid substeps with a movement key held since the last valid one without
    wishing = direction.abs().sum(-1) > 1e-6
    count = torch.cumsum((live & wishing).long(), dim=1)
    since = torch.where(live & ~wishing, count, -hold[:, None].expand_as(count)).cummax(dim=1)
    held = (count - since.values).clamp_max(int(table.shape[1]) - 1)
    want = direction * table[gait, held].unsqueeze(-1)
    wanted = torch.linalg.norm(want[..., :2], dim=-1, keepdim=True)

    started, flight = _flights(jumped, live, flight)
    airborne = started > 0
    speeds = torch.tensor(_jump_speeds(GRAVITY_U_PER_S2 * dt), dtype=dtype, device=device)
    lifted = torch.where(
        airborne, speeds[(FLIGHT_SUBSTEPS - started).clamp(0, FLIGHT_SUBSTEPS - 1)], 0.0
    )
    steps = torch.arange(substeps, device=device).expand(batch, substeps)
    newest = torch.where(live, steps, -1).cummax(dim=1).values
    vertical = torch.where(
        newest >= 0,
        lifted.gather(1, newest.clamp_min(0)),
        velocity[:, 2:3].expand(batch, substeps),
    )

    # on the ground the horizontal velocity relaxes towards the keys' speed, faster speeding up
    relaxes = live & ~airborne
    every, some = relaxes.all(0).tolist(), relaxes.any(0).tolist()
    up, down = (
        torch.tensor(alpha, dtype=dtype, device=device)
        for alpha in (prior.alpha_up, prior.alpha_down)
    )
    horizontal, planar = [], velocity[:, :2]
    for target, speed, relaxing, all_relax, any_relax in zip(
        want[..., :2].unbind(1), wanted.unbind(1), relaxes[..., None].unbind(1), every, some
    ):
        if any_relax:
            alpha = torch.where(speed >= torch.linalg.norm(planar, dim=-1, keepdim=True), up, down)
            relaxed = planar + alpha * (target - planar)
            planar = relaxed if all_relax else torch.where(relaxing, relaxed, planar)
        horizontal.append(planar)
    velocities = torch.cat([torch.stack(horizontal, 1), vertical[..., None]], -1)

    during = (velocities * dt).view(batch, frames, per_frame, 3)
    moved = torch.zeros_like(during[:, :, 0])
    for i in range(per_frame):
        moved = moved + during[:, :, i]
    return moved, PriorState(velocities[:, -1].clone(), held[:, -1].clone(), flight)


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
    """

    yaw: np.ndarray
    pitch: np.ndarray
    displacement: np.ndarray
    start_state: np.ndarray


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
    return track.xyz[published] + (motion.displacement[video_frames] - start)


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
            self.states[0, p, torch.as_tensor(block), :3] = torch.from_numpy(values).to(
                torch.float32
            )
            self._held[p] = np.asarray(values[-1], np.float64)
            later = np.flatnonzero(self.alive[p])
            later = later[later > frames[-1]]
            if len(later):
                self.states[0, p, torch.as_tensor(later), :3] = torch.from_numpy(values[-1]).to(
                    torch.float32
                )
            touched[p] = np.concatenate([block, later])
        return touched
