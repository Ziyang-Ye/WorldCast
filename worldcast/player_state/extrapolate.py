"""Extrapolation from the controls (App. "Closed-loop deployment"): players in an undrawn block.

A player's position over block ``s`` (latent frames ``s .. s+3``) is its published position at knot
``s - 1`` plus the physics prior's displacement over the block: the movement keys integrated at
64 Hz with per-key walking speeds, jump and gravity, and no collisions. Its view angles are the
round-start angles plus the integral of its own view controls. A knot is one latent frame, 8
source frames.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .projection import c2w_from_state_rows
from .tables import integrate_camera_angles, pack_substeps

__all__ = [
    "BLOCK",
    "SOURCE_FRAMES_PER_LATENT",
    "PhysicsPrior",
    "physics_prior_track",
    "prior_channels",
    "Motion",
    "player_motion",
    "PositionTrack",
    "block_anchor_knot",
    "dead_reckoned_xyz",
    "OwnCameraPlan",
    "PredictedStateTable",
]

#: Latent frames per block.
BLOCK = 4
#: Pixel rows per latent frame, source frames (32 fps) per pixel row and per latent frame.
ROWS_PER_LATENT = 4
SOURCE_FRAMES_PER_ROW = 2
SOURCE_FRAMES_PER_LATENT = ROWS_PER_LATENT * SOURCE_FRAMES_PER_ROW
#: Pixel rows per second; the prior integrates at its substep rate (64 Hz, the engine tick).
PIXEL_HZ = 16.0
#: Jump impulse (u/s) and gravity (u/s^2) of the engine.
JUMP_U_PER_S = 260.0
GRAVITY_U_PER_S2 = 800.0
#: Substeps a jump stays airborne.
FLIGHT_SUBSTEPS = 40
MOVEMENT_BUTTONS = ("forward", "back", "move_left", "move_right")
GAIT_BUTTONS = ("speed", "duck")


@dataclass(frozen=True)
class PhysicsPrior:
    """The fitted prior (``configs/state_model/physics_prior_v1.json``).

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


def prior_channels(button_names: Sequence[str]) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """``(movement, gait, jump)`` button columns, by button name."""
    names = [str(n) for n in button_names]
    return (
        tuple(names.index(n) for n in MOVEMENT_BUTTONS),
        tuple(names.index(n) for n in GAIT_BUTTONS),
        names.index("jump"),
    )


def _to_world(fru: torch.Tensor, yaw_degrees: torch.Tensor) -> torch.Tensor:
    psi = torch.deg2rad(yaw_degrees.float())
    cos, sin = torch.cos(psi), torch.sin(psi)
    f, r, u = fru[..., 0], fru[..., 1], fru[..., 2]
    return torch.stack([f * cos + r * sin, f * sin - r * cos, u], dim=-1)


def physics_prior_track(
    action: torch.Tensor,
    yaw: torch.Tensor,
    anchor: torch.Tensor,
    *,
    channels: tuple[tuple[int, ...], tuple[int, ...], int],
    prior: PhysicsPrior,
) -> torch.Tensor:
    """Integrate the movement keys.

    Args:
        action (torch.Tensor): ``[B, T, Q, A]`` packed substeps, valid flag last (an invalid substep
            freezes the integrator).
        yaw (torch.Tensor): ``[B, T]`` degrees, held across a row's substeps.
        anchor (torch.Tensor): ``[B, 1, 3]`` start position, u.
        channels (tuple): :func:`prior_channels`.
        prior (PhysicsPrior): the fitted prior.

    Returns:
        torch.Tensor: ``[B, T, 3]`` positions, u; row 0 adds no displacement (its keys precede the
        window).
    """
    columns, gait, jump = channels
    batch, frames, per_row = (int(v) for v in action.shape[:3])
    dt = 1.0 / (PIXEL_HZ * per_row)
    fall = GRAVITY_U_PER_S2 * dt
    device = anchor.device
    with torch.no_grad():
        valid = action[..., -1] > 0.5
        move = (action[..., list(columns)].abs() > 0.5).float()
        forward = move[..., 0] - move[..., 1]
        strafe = move[..., 3] - move[..., 2]
        gait_held = action[..., list(gait)].abs() > 0.5
        jumped = action[..., int(jump)].abs() > 0.5
        wish = torch.stack([forward, strafe], dim=-1)
        magnitude = wish.norm(dim=-1, keepdim=True)
        unit = wish / magnitude.clamp_min(1e-6)
        moving = (magnitude.squeeze(-1) > 1e-6).unsqueeze(-1)
        direction = _to_world(
            torch.cat([unit, torch.zeros_like(unit[..., :1])], dim=-1), yaw[:, :, None]
        )
        direction = (direction * moving.to(direction.dtype)).reshape(batch, frames * per_row, 3)
        gait_index = (gait_held[..., 0].long() * 2 + gait_held[..., 1].long()).reshape(batch, -1)
        table = torch.as_tensor(prior.table, dtype=direction.dtype, device=device)
        hold_cap = table.shape[1] - 1
        jumped, valid = jumped.reshape(batch, -1), valid.reshape(batch, -1)

        velocity = torch.zeros(batch, 3, dtype=direction.dtype, device=device)
        hold = torch.zeros(batch, dtype=torch.long, device=device)
        flight = torch.zeros(batch, dtype=torch.long, device=device)
        rows = []
        for row in range(frames):
            moved = torch.zeros(batch, 3, dtype=direction.dtype, device=device)
            for sub in range(per_row):
                i = row * per_row + sub
                live = valid[:, i]
                # the speed ramp keeps running across direction changes, resets when the player
                # stops
                wishing = direction[:, i].abs().sum(-1) > 1e-6
                held = torch.where(wishing, (hold + 1).clamp_max(hold_cap), torch.zeros_like(hold))
                takeoff = jumped[:, i] & (flight == 0)
                started = torch.where(takeoff, torch.full_like(flight, FLIGHT_SUBSTEPS), flight)
                airborne = started > 0
                want = direction[:, i] * table[gait_index[:, i], held].unsqueeze(-1)
                current = torch.linalg.norm(velocity[:, :2], dim=-1)
                wanted = torch.linalg.norm(want[:, :2], dim=-1)
                alpha = torch.where(
                    wanted >= current,
                    torch.full_like(current, prior.alpha_up),
                    torch.full_like(current, prior.alpha_down),
                )
                grounded_v = torch.stack(
                    [
                        velocity[:, 0] + alpha * (want[:, 0] - velocity[:, 0]),
                        velocity[:, 1] + alpha * (want[:, 1] - velocity[:, 1]),
                        torch.zeros_like(velocity[:, 2]),
                    ],
                    dim=-1,
                )
                airborne_v = torch.stack(
                    [
                        velocity[:, 0],
                        velocity[:, 1],
                        torch.where(
                            takeoff,
                            torch.full_like(velocity[:, 2], JUMP_U_PER_S),
                            velocity[:, 2] - fall,
                        ),
                    ],
                    dim=-1,
                )
                stepped = torch.where(airborne.unsqueeze(-1), airborne_v, grounded_v)
                velocity = torch.where(live.unsqueeze(-1), stepped, velocity)
                hold = torch.where(live, held, hold)
                flight = torch.where(live, torch.where(airborne, started - 1, started), flight)
                moved = moved + velocity * dt
            rows.append(moved)
        displacement = torch.stack(rows, dim=1)
        displacement = torch.cat(
            [torch.zeros_like(displacement[:, :1]), displacement[:, 1:]], dim=1
        )
        return anchor + torch.cumsum(displacement, dim=1)


@dataclass
class Motion:
    """What a player's own controls say, per pixel row (all float64).

    Attributes:
        yaw (np.ndarray): ``[T]`` degrees.
        pitch (np.ndarray): ``[T]`` degrees.
        disp (np.ndarray): ``[T, 3]`` the prior's displacement from row 0, u.
        s0 (np.ndarray): ``[6]`` the round-start player-state row.
    """

    yaw: np.ndarray
    pitch: np.ndarray
    disp: np.ndarray
    s0: np.ndarray


def player_motion(
    batch: Mapping[str, torch.Tensor],
    slot: int,
    *,
    camera_delta_scale: float,
    channels: tuple[tuple[int, ...], tuple[int, ...], int],
    prior: PhysicsPrior,
) -> Motion:
    """Integrate seat ``slot``'s controls once over the round (``batch``: the whole round)."""
    states = batch["player_states"]
    packed = pack_substeps(batch["player_action_substeps"], batch["player_action_substep_valid"])
    yaw, pitch = integrate_camera_angles(
        states[:, :, 0].float(), packed, camera_delta_scale=float(camera_delta_scale)
    )
    disp = physics_prior_track(
        packed[:, slot].float(),
        yaw[:, slot].float(),
        torch.zeros(1, 1, 3),
        channels=channels,
        prior=prior,
    )[0]
    return Motion(
        yaw=yaw[0, slot].double().numpy(),
        pitch=pitch[0, slot].double().numpy(),
        disp=disp.double().numpy(),
        s0=states[0, slot, 0].double().numpy(),
    )


class PositionTrack:
    """A player's published positions, one knot per latent frame (knot ``k`` at source frame
    ``start + 8k``), growing block by block from knot 0, the round-start position.

    Args:
        media_id (str): the player's media id.
        start_frame (int): source frame of knot 0.
        s0 (np.ndarray): ``[3]`` the round-start position, u.
    """

    def __init__(self, media_id: str, start_frame: int, s0) -> None:
        self.media_id, self.start_frame = str(media_id), int(start_frame)
        self.frame = np.array([self.start_frame], dtype=np.int64)
        self.xyz = np.asarray(s0, np.float64).reshape(1, 3).copy()

    def append(self, knots: Sequence[int], xyz) -> None:
        """Add the next knots ``knots`` (consecutive) at positions ``xyz`` ``[n, 3]``, u."""
        knots = [int(k) for k in knots]
        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        if knots != list(range(len(self.frame), len(self.frame) + len(knots))) or len(xyz) != len(
            knots
        ):
            raise ValueError(f"{self.media_id}: knots {knots} after {len(self.frame)} knots")
        if not np.isfinite(xyz).all():
            raise ValueError(f"{self.media_id}: non-finite knot")
        self.frame = np.concatenate(
            [self.frame, self.start_frame + SOURCE_FRAMES_PER_LATENT * np.asarray(knots)]
        )
        self.xyz = np.concatenate([self.xyz, xyz], 0)

    def knot_at_or_before(self, frame: int) -> int:
        """Index of the newest knot at or before ``frame``; raises if older than a latent frame."""
        k = int(np.searchsorted(self.frame, int(frame), side="right")) - 1
        if k < 0 or int(frame) - int(self.frame[k]) > SOURCE_FRAMES_PER_LATENT:
            raise ValueError(f"{self.media_id}: no fresh knot at frame {int(frame)}")
        return k


def block_anchor_knot(track: PositionTrack, s: int) -> int:
    """The knot block ``s`` is extrapolated from: the newest at or before latent frame ``s - 1``."""
    return track.knot_at_or_before(track.start_frame + SOURCE_FRAMES_PER_LATENT * (int(s) - 1))


def dead_reckoned_xyz(track: PositionTrack, motion: Motion, anchor_knots, rows) -> np.ndarray:
    """``[len(rows), 3]``: each anchor knot's position plus the prior's displacement from the knot's
    pixel row to ``rows``."""
    rows = np.asarray(rows, dtype=np.int64)
    knot_rows = (
        track.frame[np.asarray(anchor_knots, dtype=np.int64)] - track.start_frame
    ) // SOURCE_FRAMES_PER_ROW
    return track.xyz[anchor_knots] + (motion.disp[rows] - motion.disp[knot_rows])


class OwnCameraPlan:
    """The client's own cameras from its position track.

    Latent frames before block ``s`` sit at their own knots, block ``s`` is extrapolated from knot
    ``s - 1``, later latent frames hold.

    Args:
        track (PositionTrack): the client's published positions.
        motion (Motion): the client's own controls.
        n_latents (int): latent frames of the rollout.
        eye_height (float): camera height above the feet, u.
    """

    def __init__(
        self, track: PositionTrack, motion: Motion, n_latents: int, *, eye_height: float
    ) -> None:
        self.track, self.motion = track, motion
        self.n_latents, self.eye_height = int(n_latents), float(eye_height)
        self._cache: tuple[int, torch.Tensor] | None = None

    def _state_rows(self, latents, anchor_knots) -> np.ndarray:
        latents = np.asarray(latents, dtype=np.int64)
        rows = ROWS_PER_LATENT * latents
        out = np.zeros((len(latents), 6), dtype=np.float64)
        out[:, :3] = dead_reckoned_xyz(self.track, self.motion, anchor_knots, rows)
        out[:, 3], out[:, 4], out[:, 5] = self.motion.yaw[rows], self.motion.pitch[rows], 1.0
        return out

    def _known_rows(self, latents) -> np.ndarray:
        knots = [
            self.track.knot_at_or_before(self.track.start_frame + SOURCE_FRAMES_PER_LATENT * int(l))
            for l in latents
        ]
        return self._state_rows(latents, np.asarray(knots, dtype=np.int64))

    def block_rows(self, s: int) -> np.ndarray:
        """``[4, 6]`` float64 player-state rows of block ``s``, extrapolated from knot ``s - 1``."""
        return self._state_rows(
            np.arange(s, s + BLOCK), np.full(BLOCK, block_anchor_knot(self.track, s))
        )

    def _c2w(self, rows: np.ndarray) -> torch.Tensor:
        return c2w_from_state_rows(rows.astype(np.float32), eye_height=self.eye_height).float()

    def for_block(self, s: int) -> torch.Tensor:
        """``[N, 4, 4]`` cameras known at the start of block ``s`` (its rays and retrieval)."""
        s, n = int(s), self.n_latents
        out = np.zeros((n, 6), dtype=np.float64)
        out[:s] = self._known_rows(np.arange(s))
        out[s : s + BLOCK] = self.block_rows(s)
        out[s + BLOCK :] = out[s + BLOCK - 1]
        return self._c2w(out)

    def as_predicted(self) -> torch.Tensor:
        """``[N, 4, 4]``: every latent frame as extrapolated when its block started, for the blocks
        whose anchor knot exists; later frames hold the newest (the plain prefix's rays and the keys
        of its memory entries)."""
        if self._cache is not None and self._cache[0] == len(self.track.frame):
            return self._cache[1]
        n = self.n_latents
        out = np.zeros((n, 6), dtype=np.float64)
        out[0] = self._known_rows([0])[0]
        last = 0
        for s in range(1, n - BLOCK + 1, BLOCK):
            if len(self.track.frame) < s:
                break
            out[s : s + BLOCK] = self.block_rows(s)
            last = s + BLOCK - 1
        out[last + 1 :] = out[last]
        self._cache = (len(self.track.frame), self._c2w(out))
        return self._cache[1]


class PredictedStateTable:
    """The round's player-state table under predicted states, rewritten block by block in place.

    Seats with a position track (the round's clients) take, on the pixel rows of block ``s``, their
    extrapolation from knot ``s - 1``; their later alive rows hold the newest value. A client that
    published no knot ``s - 1`` (finished or late) holds its last value. Every other seat, and every
    column but ``x y z``, keeps the recording.

    Args:
        states (torch.Tensor): ``[1, P, T, 6]`` float32 recorded table, rewritten in place.
        seats (Mapping[int, tuple[PositionTrack, Motion]]): track and motion of each client's seat.
    """

    def __init__(
        self, states: torch.Tensor, seats: Mapping[int, tuple[PositionTrack, Motion]]
    ) -> None:
        self.states = states
        self.alive = states[0, :, :, 5].numpy() > 0.5  # the alive column is never rewritten
        self.seats = dict(seats)
        self._held = {
            p: np.asarray(track.xyz[0], np.float64) for p, (track, _) in self.seats.items()
        }

    def track_of(self, media_id: str) -> PositionTrack | None:
        """The position track of ``media_id``, or None for a seat without one."""
        for track, _ in self.seats.values():
            if track.media_id == str(media_id):
                return track
        return None

    def advance(self, s: int) -> dict[int, np.ndarray]:
        """Write the rows of block ``s`` (latent frames ``s .. s+3``); returns them by slot."""
        s, n_rows = int(s), int(self.states.shape[2])
        rows = np.arange(ROWS_PER_LATENT * (s - 1) + 1, ROWS_PER_LATENT * (s + BLOCK - 1) + 1)
        rows = rows[rows < n_rows]
        touched = {}
        for p, (track, motion) in self.seats.items():
            block_rows = rows[self.alive[p, rows]]
            if not len(block_rows):
                continue
            try:
                anchor = block_anchor_knot(track, s)
                values = dead_reckoned_xyz(
                    track, motion, np.full(len(block_rows), anchor), block_rows
                )
            except ValueError:
                values = np.repeat(self._held[p][None], len(block_rows), 0)
            self.states[0, p, torch.as_tensor(block_rows), :3] = torch.from_numpy(values).to(
                torch.float32
            )
            self._held[p] = np.asarray(values[-1], np.float64)
            later = np.flatnonzero(self.alive[p])
            later = later[later > rows[-1]]
            if len(later):
                self.states[0, p, torch.as_tensor(later), :3] = torch.from_numpy(values[-1]).to(
                    torch.float32
                )
            touched[p] = np.concatenate([block_rows, later])
        return touched
