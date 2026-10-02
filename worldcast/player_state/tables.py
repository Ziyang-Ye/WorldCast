"""Player-state tables: the loader's pixel-rate tensors folded to latent rate (Sec. 3.2).

The loader (``worldcast.data``) gives, for all ``P`` (= 10) players of the round, one row per pixel
frame (16 fps), ``T = 1 + 4 (F - 1)`` rows for ``F`` latent frames; latent frame ``f`` is sampled at
pixel row ``4 f`` and owns the rows ``4f-3 .. 4f`` (row 0 for ``f = 0``), the causal-VAE mapping:

* ``player_states`` ``[B, P, T, 6]``: x, y, z (u), yaw, pitch (degrees), alive;
* ``player_action_substeps`` ``[B, P, T, Q, A]``: ``Q = 4`` ordered substeps per row, ``A = 13``:
  the 11 buttons, then the pitch and yaw deltas / 5;
* ``player_action_substep_valid`` ``[B, P, T, Q]`` bool;
* ``player_team_ids`` ``[B, P]``: engine team (2, 3; 0 for a seat without media);
* ``player_weapon_ids`` ``[B, P, T]``: 52-way weapon id (optional);
* ``observer_slot`` ``[B]``: the client's own seat.

The peer state table the field projects is ``[B, F, P, 6 | 13]`` float32. Columns 0:3 are x, y, z
(u); 3, 4 the engine yaw and pitch (degrees; a 6-wide table reads only row 0 of them, a 13-wide one
never); 5 alive. A gathered (13-wide) table adds 6, 7 the continuous yaw and pitch; 8 corpse; 9 time
since death (latent frames, capped at 32); 10:13 the frozen x, y, z of the last alive row. The
width is the switch: 6 for the contiguous plain prefix, 13 for every gathered window. The 7 columns
(:func:`continuous_row_state`) are only right on the whole round, so they are computed before the
window gather and carried through it (:data:`CONTINUOUS_COLUMNS_KEY`).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch

from worldcast.data.obs_signals import OBS_SIGNAL_KEYS

__all__ = [
    "STATE_BASE_DIM",
    "STATE_CONTINUOUS_DIM",
    "CONTINUOUS_WIDTH",
    "CONT_YAW_INDEX",
    "CONT_PITCH_INDEX",
    "CONT_CORPSE_INDEX",
    "CONT_TSD_INDEX",
    "CONT_FROZEN_XYZ",
    "CONTINUOUS_COLUMNS_KEY",
    "CONTINUOUS_ROWS_KEY",
    "SUBSTEP_PITCH_INDEX",
    "SUBSTEP_YAW_INDEX",
    "SUBSTEP_VALID_INDEX",
    "CAMERA_DELTA_SCALE",
    "PITCH_MIN_DEGREES",
    "PITCH_MAX_DEGREES",
    "DEATH_TAU",
    "DEATH_TSD_CAP",
    "PlayerStates",
    "DeathState",
    "wrap_degrees",
    "latent_frame_count",
    "pixel_to_latent_rows",
    "pack_substeps",
    "integrate_camera_angles",
    "continuous_row_state",
    "peer_conditions",
    "has_continuous_columns",
    "derive_death_state",
    "gathered_death_state",
    "death_state_for_table",
    "check_block_alive_agreement",
]

# ------------------------------------------------------------------------------------- table layout
STATE_BASE_DIM = 6
CONT_YAW_INDEX = 6
CONT_PITCH_INDEX = 7
CONT_CORPSE_INDEX = 8
CONT_TSD_INDEX = 9
CONT_FROZEN_XYZ = (10, 13)
STATE_CONTINUOUS_DIM = 13
CONTINUOUS_WIDTH = STATE_CONTINUOUS_DIM - STATE_BASE_DIM
#: Batch key of the continuous columns computed on the whole round (latent axis, before the gather).
CONTINUOUS_ROWS_KEY = "peer_continuous_rows"
#: Batch key of the same columns gathered with the rows by the window
#: (``worldcast.sampling.window``).
CONTINUOUS_COLUMNS_KEY = "peer_continuous_columns"

#: Packed substep layout (:func:`pack_substeps`): buttons, pitch delta, yaw delta, valid flag.
SUBSTEP_PITCH_INDEX = -3
SUBSTEP_YAW_INDEX = -2
SUBSTEP_VALID_INDEX = -1

#: Degrees per unit camera delta: the loader divides by it, the integral multiplies it back.
CAMERA_DELTA_SCALE = 5.0
PITCH_MIN_DEGREES = -89.0
PITCH_MAX_DEGREES = 89.0

#: Dying decays as ``exp(-tsd / DEATH_TAU)``, tsd in latent frames, saturating at ``DEATH_TSD_CAP``.
DEATH_TAU = 5.0
DEATH_TSD_CAP = 32.0


class DeathState(NamedTuple):
    """Death state of a table, per latent frame and player.

    Attributes:
        alive (torch.Tensor): ``[B, F, P]`` bool, table column 5 > 0.5.
        corpse (torch.Tensor): ``[B, F, P]`` bool, dead with a known death site (a player dead from
            the first row has none and never gets a corpse).
        dying (torch.Tensor): ``[B, F, P]`` table dtype, ``exp(-tsd / 5)`` on corpse frames, else 0.
        time_since_death (torch.Tensor): ``[B, F, P]`` table dtype, latent frames since the last
            alive row, capped at 32.
        frozen_states (torch.Tensor): ``[B, F, P, S]`` the table with x, y, z replaced by the
            position at the last alive row.
    """

    alive: torch.Tensor
    corpse: torch.Tensor
    dying: torch.Tensor
    time_since_death: torch.Tensor
    frozen_states: torch.Tensor


def wrap_degrees(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle in degrees to ``[-180, 180)``."""
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def latent_frame_count(pixel_frames: int) -> int:
    """``F`` for ``T = 1 + 4 (F - 1)`` pixel rows; raises if ``T`` is not of that form."""
    pixel_frames = int(pixel_frames)
    if (pixel_frames - 1) % 4:
        raise ValueError(f"{pixel_frames} pixel rows; expected 1 + 4(F - 1)")
    return 1 + (pixel_frames - 1) // 4


def pixel_to_latent_rows(pixel_frames: int, latent_frames: int) -> torch.Tensor:
    """``[F]`` long (CPU): the pixel row sampled for each latent frame, ``4 f``."""
    if pixel_frames != 1 + 4 * (latent_frames - 1):
        raise ValueError(
            f"pixel_frames={pixel_frames} does not match latent_frames={latent_frames}"
        )
    idx = torch.arange(latent_frames) * 4
    return idx.clamp_(max=pixel_frames - 1)


def pack_substeps(substeps: torch.Tensor, substep_valid: torch.Tensor) -> torch.Tensor:
    """Zero invalid substeps and append the valid flag, ``[..., Q, A]`` -> ``[..., Q, A + 1]``."""
    valid_flag = substep_valid.bool()
    packed = substeps.float() * valid_flag.unsqueeze(-1).float()
    return torch.cat([packed, valid_flag.unsqueeze(-1).float()], dim=-1)


# -------------------------------------------------------------------------------- continuous camera
def integrate_camera_angles(
    initial_state: torch.Tensor,
    substeps: torch.Tensor,
    *,
    camera_delta_scale: float = CAMERA_DELTA_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-rate yaw and pitch of every player, integrated from row 0 as the engine does.

    Row ``j`` covers ``(time(j-1), time(j)]``, so the angle at row ``j`` is the row-0 angle plus
    every valid delta of rows ``1..j``. Yaw wraps to ``[-180, 180)``; pitch is clamped to ``[-89,
    89]`` row by row, so a delta that would leave the range is discarded rather than banked.

    Args:
        initial_state (torch.Tensor): ``[B, P, 6]`` table row 0 (yaw, pitch in degrees at columns 3,
            4).
        substeps (torch.Tensor): ``[B, P, T, Q, A + 1]`` packed substeps (:func:`pack_substeps`);
            deltas in units of ``camera_delta_scale`` degrees.
        camera_delta_scale (float): degrees per unit delta.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(yaw, pitch)``, each ``[B, P, T]`` float32 degrees.
    """
    scale = float(camera_delta_scale)
    values = substeps.float()
    valid = values[..., SUBSTEP_VALID_INDEX] > 0.5
    delta_pitch = (values[..., SUBSTEP_PITCH_INDEX] * valid).sum(-1) * scale
    delta_yaw = (values[..., SUBSTEP_YAW_INDEX] * valid).sum(-1) * scale
    delta_pitch = torch.cat([torch.zeros_like(delta_pitch[:, :, :1]), delta_pitch[:, :, 1:]], dim=2)
    delta_yaw = torch.cat([torch.zeros_like(delta_yaw[:, :, :1]), delta_yaw[:, :, 1:]], dim=2)

    yaw = wrap_degrees(initial_state[..., 3:4].float() + delta_yaw.cumsum(dim=2))
    current = initial_state[..., 4].float().clamp(PITCH_MIN_DEGREES, PITCH_MAX_DEGREES)
    rows_out = []
    for row in range(int(delta_pitch.shape[2])):
        current = (current + delta_pitch[:, :, row]).clamp(PITCH_MIN_DEGREES, PITCH_MAX_DEGREES)
        rows_out.append(current)
    return yaw, torch.stack(rows_out, dim=2)


def continuous_row_state(
    batch: Mapping[str, torch.Tensor], *, camera_delta_scale: float = CAMERA_DELTA_SCALE
) -> torch.Tensor:
    """The 7 continuous columns of the whole round's table.

    They are functions of the full history, so they are computed before any window gather, which
    gathers them by latent index with the rows and never recomputes them.

    Args:
        batch (Mapping[str, torch.Tensor]): ``player_states``, ``player_action_substeps`` and
            ``player_action_substep_valid`` (module docstring).
        camera_delta_scale (float): degrees per unit delta.

    Returns:
        torch.Tensor: ``[B, F, P, 7]`` float32: continuous yaw, pitch (degrees,
        :func:`integrate_camera_angles` at the latent rows), corpse (0/1), time since death (latent
        frames), frozen x, y, z (u).
    """
    states = batch["player_states"]
    pixel_frames = int(states.shape[2])
    rows = pixel_to_latent_rows(pixel_frames, latent_frame_count(pixel_frames)).to(states.device)
    packed = pack_substeps(batch["player_action_substeps"], batch["player_action_substep_valid"])
    yaw_pixel, pitch_pixel = integrate_camera_angles(
        states[:, :, 0], packed, camera_delta_scale=camera_delta_scale
    )
    yaw = yaw_pixel.index_select(2, rows).permute(0, 2, 1)
    pitch = pitch_pixel.index_select(2, rows).permute(0, 2, 1)
    latent_states = states.float().index_select(2, rows).permute(0, 2, 1, 3)
    death = derive_death_state(latent_states)
    return torch.cat(
        [
            yaw.unsqueeze(-1),
            pitch.unsqueeze(-1),
            death.corpse.unsqueeze(-1).to(yaw.dtype),
            death.time_since_death.unsqueeze(-1).to(yaw.dtype),
            death.frozen_states[..., :3].to(yaw.dtype),
        ],
        dim=-1,
    ).contiguous()


# ----------------------------------------------------------------------------- latent player states
@dataclass(frozen=True)
class PlayerStates:
    """Every player's states and controls at latent rate, for one window (the client included).

    Attributes:
        actions (torch.Tensor): ``[B, F, P, 4 Q, A + 1]`` float32 packed substeps of the four pixel
            rows each latent frame owns (16 substeps per latent frame; the last column is the valid
            flag).
        xyz (torch.Tensor): ``[B, F, P, 3]`` float32 engine position, u.
        yaw (torch.Tensor): ``[B, F, P]`` float32 engine yaw, degrees.
        pitch (torch.Tensor): ``[B, F, P]`` float32 engine pitch, degrees.
        alive (torch.Tensor): ``[B, F, P]`` bool.
        observer_slot (torch.Tensor): ``[B]`` long, the client's own seat.
        team_ids (torch.Tensor): ``[B, P]`` engine team ids.
        latent_rows (torch.Tensor): ``[F]`` long, the pixel row sampled for each latent frame
            (``4 f``).
        weapon_ids (torch.Tensor): ``[B, F, P]`` long 52-way weapon id at the latent frame's pixel
            row.
        continuous (torch.Tensor | None): ``[B, F, P, 7]`` float32 gathered continuous columns
            (gathered window), or None (contiguous window).
    """

    actions: torch.Tensor
    xyz: torch.Tensor
    yaw: torch.Tensor
    pitch: torch.Tensor
    alive: torch.Tensor
    observer_slot: torch.Tensor
    team_ids: torch.Tensor
    latent_rows: torch.Tensor
    weapon_ids: torch.Tensor
    continuous: torch.Tensor | None = None

    @classmethod
    def from_batch(
        cls, batch: Mapping[str, torch.Tensor], *, device: torch.device | None = None
    ) -> "PlayerStates":
        """Fold one loader batch (the whole round or a gathered window) to latent rate.

        The latent count comes from the batch, so one call serves the whole round (plain prefix) and
        a gathered window (21 or 17 latent frames). A gathered batch carries its continuous columns
        under :data:`CONTINUOUS_COLUMNS_KEY` (``[B, F, P, 7]``); they are carried, never recomputed.

        Args:
            batch (Mapping[str, torch.Tensor]): the keys of the module docstring.
            device (torch.device | None): where to put the states (default: the batch's device).

        Returns:
            PlayerStates: float tensors float32.
        """
        required = (
            "player_states",
            "player_team_ids",
            "player_action_substeps",
            "player_action_substep_valid",
            "observer_slot",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"player-state batch is missing {missing}")
        states = batch["player_states"]
        team_ids = batch["player_team_ids"]
        observer_slot = batch["observer_slot"]
        if states.ndim != 4 or states.shape[-1] != STATE_BASE_DIM:
            raise ValueError("player_states must be [B, P, T, 6]")
        batch_size, players, pixel_frames = states.shape[:3]
        if team_ids.shape != (batch_size, players):
            raise ValueError("player_team_ids must be [B, P]")
        latent_frames = latent_frame_count(int(pixel_frames))

        source = states.device
        rows = pixel_to_latent_rows(int(pixel_frames), latent_frames).to(source)
        substeps = batch["player_action_substeps"]
        packed = pack_substeps(substeps, batch["player_action_substep_valid"])

        continuous = batch.get(CONTINUOUS_COLUMNS_KEY)
        if continuous is not None:
            expected = (batch_size, latent_frames, players, CONTINUOUS_WIDTH)
            if tuple(continuous.shape) != expected:
                raise ValueError(
                    f"{CONTINUOUS_COLUMNS_KEY} must be {list(expected)} (frame-major, latent"
                    f" rate), got {tuple(continuous.shape)}"
                )
            continuous = continuous.to(source).float()

        # latent frame f owns pixel rows 4f-3 .. 4f; latent frame 0 repeats row 0
        window = (rows[:, None] + torch.arange(-3, 1, device=source)[None, :]).clamp_min(0)
        actions = packed.index_select(2, window.reshape(-1)).reshape(
            batch_size, players, latent_frames, 4 * substeps.shape[3], packed.shape[-1]
        )
        actions = actions.permute(0, 2, 1, 3, 4).contiguous()

        latent_states = states.index_select(2, rows).permute(0, 2, 1, 3).contiguous()
        raw_weapons = batch.get("player_weapon_ids")
        if raw_weapons is None:
            weapon_ids = torch.zeros(latent_states.shape[:3], dtype=torch.long, device=source)
        else:
            weapon_ids = (
                raw_weapons.to(source).index_select(2, rows).permute(0, 2, 1).contiguous().long()
            )
        alive = latent_states[..., 5] > 0.5

        target = source if device is None else torch.device(device)
        return cls(
            actions=actions.to(target).float(),
            xyz=latent_states[..., :3].contiguous().to(target).float(),
            yaw=latent_states[..., 3].contiguous().to(target).float(),
            pitch=latent_states[..., 4].contiguous().to(target).float(),
            alive=alive.contiguous().to(target),
            observer_slot=observer_slot.contiguous().to(target),
            team_ids=team_ids.contiguous().to(target),
            latent_rows=rows.to(target),
            weapon_ids=weapon_ids.to(target),
            continuous=None if continuous is None else continuous.to(target).float(),
        )

    def table(self) -> torch.Tensor:
        """The peer state table, ``[B, F, P, 6]`` (contiguous) or ``[B, F, P, 13]`` (gathered)."""
        table = torch.cat(
            [
                self.xyz,
                self.yaw.unsqueeze(-1),
                self.pitch.unsqueeze(-1),
                self.alive.unsqueeze(-1).to(self.xyz.dtype),
            ],
            dim=-1,
        )
        if self.continuous is None:
            return table
        return torch.cat([table, self.continuous.to(table.dtype)], dim=-1)


def peer_conditions(
    states: PlayerStates,
    visible: torch.Tensor,
    *,
    obs_signals: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """The generator's player-state conditions; the field is built from them inside the generator.

    Args:
        states (PlayerStates): every player's latent-rate states.
        visible (torch.Tensor): ``[B, F, P]`` bool, the visibility gate per latent frame
            (``worldcast.player_state.visibility``; GT labels in Table 3).
        obs_signals (Mapping[str, torch.Tensor] | None): a mapping holding the five observer-signal
            rows ``[B, F]`` (the batch itself will do), or None to omit them.

    Returns:
        dict[str, torch.Tensor]: ``peer_states [B, F, P, 6|13]`` float32, ``peer_actions
        [B, F, P, 16, 14]`` float32, ``peer_observer_slot [B]``, ``peer_team_ids [B, P]``,
        ``peer_alive [B, F, P]`` float32, ``peer_visible [B, F, P]`` float32, ``peer_weapons
        [B, F, P]`` long, then the observer-signal keys as long.
    """
    conditions = {
        "peer_states": states.table(),
        "peer_actions": states.actions,
        "peer_observer_slot": states.observer_slot,
        "peer_team_ids": states.team_ids,
        "peer_alive": states.alive.to(torch.float32),
        "peer_visible": visible.to(torch.float32),
        "peer_weapons": states.weapon_ids,
    }
    if obs_signals is not None:
        signals = [obs_signals[key] for key in OBS_SIGNAL_KEYS]
        device = states.actions.device
        for key, value in zip(OBS_SIGNAL_KEYS, signals):
            conditions[key] = value.to(device=device, dtype=torch.long)
    return conditions


# -------------------------------------------------------------------------------------- death state
def has_continuous_columns(states: torch.Tensor) -> bool:
    """True for a 13-wide (gathered) table, False for a 6-wide one; raises for any other."""
    width = int(states.shape[-1])
    if width == STATE_BASE_DIM:
        return False
    if width == STATE_CONTINUOUS_DIM:
        return True
    raise ValueError(
        f"peer state table must be {STATE_BASE_DIM} columns wide (x y z yaw pitch alive) or "
        f"{STATE_CONTINUOUS_DIM} (plus the gathered continuous columns); got {width}"
    )


def derive_death_state(states: torch.Tensor) -> DeathState:
    """Death state of a contiguous table (one life per round).

    The position after death freezes at the last alive frame. A player dead from frame 0 has no
    known death site and gets no corpse. An alive rise along the frame axis (resurrection) is a data
    defect and raises.

    Args:
        states (torch.Tensor): ``[B, F, P, >= 6]`` table, alive at column 5.

    Returns:
        DeathState: ``dying``, ``time_since_death`` and ``frozen_states`` in the table's dtype.
    """
    alive = states[..., 5] > 0.5
    rises = alive[:, 1:] & ~alive[:, :-1]
    if bool(rises.any()):
        where = rises.nonzero()
        raise ValueError(
            f"alive rises along the frame axis (resurrection) at {where.shape[0]} (batch, frame,"
            f" player) sites, first at {where[0].tolist()} -- one life per round"
        )

    frames = alive.shape[1]
    idx = torch.arange(frames, device=alive.device).view(1, frames, 1)
    last_alive_idx = (
        torch.where(alive, idx.expand_as(alive), alive.new_tensor(-1, dtype=torch.long))
        .cummax(dim=1)
        .values
    )

    dead = ~alive
    corpse = dead & (last_alive_idx >= 0)
    time_since_death = (
        (idx - last_alive_idx).clamp(min=0).to(states.dtype) * dead.to(states.dtype)
    ).clamp(max=DEATH_TSD_CAP)
    dying = torch.exp(-time_since_death / DEATH_TAU) * corpse.to(states.dtype)

    gather_idx = (
        last_alive_idx.clamp(min=0).unsqueeze(-1).expand(*last_alive_idx.shape, states.shape[-1])
    )
    frozen_states = states.gather(1, gather_idx)
    return DeathState(
        alive=alive,
        corpse=corpse,
        dying=dying,
        time_since_death=time_since_death,
        frozen_states=frozen_states,
    )


def gathered_death_state(states: torch.Tensor) -> DeathState:
    """Death state of a gathered (13-wide) table, read from its continuous columns.

    Checks that the columns and the rows were gathered by the same indices: a corpse row must be
    dead, and on every alive row the frozen position must equal the row's own (NaN-tolerant).

    Args:
        states (torch.Tensor): ``[B, F, P, 13]``.

    Returns:
        DeathState: ``frozen_states`` is the table with x, y, z replaced by the frozen columns (the
        camera angles stay the row's own: the client keeps turning after a player dies).
    """
    if states.ndim != 4 or int(states.shape[-1]) != STATE_CONTINUOUS_DIM:
        raise ValueError(
            f"gathered_death_state needs [B, F, P, {STATE_CONTINUOUS_DIM}]; got"
            f" {tuple(states.shape)}"
        )
    alive = states[..., 5] > 0.5
    corpse = states[..., CONT_CORPSE_INDEX] > 0.5
    if bool((corpse & alive).any()):
        where = (corpse & alive).nonzero()
        raise ValueError(
            f"gathered death columns disagree with the rows: corpse set on {where.shape[0]} alive"
            f" sites, first at {where[0].tolist()}"
        )
    lo, hi = CONT_FROZEN_XYZ
    frozen_xyz = states[..., lo:hi]
    own_xyz = states[..., :3]
    same = (frozen_xyz == own_xyz) | (torch.isnan(frozen_xyz) & torch.isnan(own_xyz))
    moved = (~same).any(dim=-1) & alive
    if bool(moved.any()):
        where = moved.nonzero()
        raise ValueError(
            "gathered death columns disagree with the rows: the frozen position of"
            f" {where.shape[0]} alive sites is not the row's own, first at {where[0].tolist()}"
        )
    time_since_death = (
        states[..., CONT_TSD_INDEX].to(states.dtype).clamp(min=0.0, max=DEATH_TSD_CAP)
    )
    dying = torch.exp(-time_since_death / DEATH_TAU) * corpse.to(states.dtype)
    frozen_states = states.clone()
    frozen_states[..., :3] = frozen_xyz
    return DeathState(
        alive=alive,
        corpse=corpse,
        dying=dying,
        time_since_death=time_since_death,
        frozen_states=frozen_states,
    )


def death_state_for_table(states: torch.Tensor) -> DeathState:
    """The death state of a 13-wide (gathered) or a 6-wide (contiguous) table."""
    if has_continuous_columns(states):
        return gathered_death_state(states)
    return derive_death_state(states)


def check_block_alive_agreement(
    derived_alive: torch.Tensor,
    alive_window: torch.Tensor | None,
    *,
    frame_offset: int,
    num_frames: int,
) -> None:
    """Raise if the table's alive column and ``peer_alive`` disagree on the decoded frames.

    Args:
        derived_alive (torch.Tensor): ``[B, F, P]`` bool, from the table.
        alive_window (torch.Tensor | None): ``[B, F, P]`` ``peer_alive`` padded back to the window
            (zeros outside the decoded block), or None (nothing to check).
        frame_offset (int): first decoded frame.
        num_frames (int): decoded frames.
    """
    if alive_window is None:
        return
    lo = int(frame_offset)
    hi = lo + int(num_frames)
    disagree = (alive_window[:, lo:hi] > 0.5) != derived_alive[:, lo:hi]
    if bool(disagree.any()):
        where = disagree.nonzero()
        raise ValueError(
            f"peer_alive and the table's alive column disagree at {where.shape[0]} sites within"
            f" frames [{lo}, {hi}), first at {where[0].tolist()}"
        )
