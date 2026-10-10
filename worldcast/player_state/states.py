"""Player states: every player's record at latent rate and the player state table.

The loader (``worldcast.data``) gives, for all ``P`` (= 10) players of the round, one row per video
frame (16 fps), ``T = 1 + 4 (F - 1)`` rows for ``F`` latent frames; latent frame ``f`` is read at
its last video frame, row ``4 f``, and owns the rows ``4f-3 .. 4f`` (row 0 for ``f = 0``), the
causal-VAE mapping:

* ``player_states`` ``[B, P, T, 6]``: x, y, z (u), yaw, pitch (degrees), alive;
* ``player_control_substeps`` ``[B, P, T, Q, A]``: ``Q = 4`` ordered substeps per row, ``A = 13``:
  the 11 buttons, then the pitch and yaw turns / 5;
* ``player_control_substep_valid`` ``[B, P, T, Q]`` bool;
* ``player_team_ids`` ``[B, P]``: engine team (2, 3; 0 for a slot without media);
* ``player_weapon_ids`` ``[B, P, T]``: 52-way weapon id;
* ``client_slot`` ``[B]``: the client's slot.

The player state table the field projects is ``[B, F, P, 6 | 13]`` float32. Columns 0:3 are x, y, z
(u); 3, 4 the engine yaw and pitch (degrees; a 6-wide table reads only row 0 of them, a 13-wide one
never); 5 alive. A 13-wide table adds the continuous columns: 6, 7 the continuous yaw and pitch; 8
corpse; 9 time since death (latent frames, capped at 32); 10:13 the frozen x, y, z of the last
alive row. They depend on the whole round, so they are computed before a window is gathered
(:func:`continuous_row_state`) and carried through the gather (:data:`CONTINUOUS_COLUMNS_KEY`). The
width is the switch: 13 when the batch carries them (every gathered window at inference, the
memory windows of stages 3 and 4), 6 when it does not (a contiguous window: the first six blocks,
an ordinary training window; and the memory windows of stage 2s, which integrate on the gathered
rows, as trained). What the field writes of a player besides geometry is
:mod:`worldcast.player_state.attributes`.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from worldcast.data.controls import CAMERA_DELTA_SCALE
from worldcast.data.game import PITCH_MAX_DEGREES, PITCH_MIN_DEGREES
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.latents import (
    BLOCK,
    VIDEO_FRAMES_PER_LATENT,
    last_video_frames,
    latent_frame_count,
    video_frames_of,
)
from worldcast.data.recordings import ALIVE_INDEX
from worldcast.sampling.window import CONTINUOUS_COLUMNS_KEY

__all__ = [
    "CONT_CORPSE_INDEX",
    "CONT_FROZEN_XYZ",
    "CONT_PITCH_INDEX",
    "CONT_TSD_INDEX",
    "CONT_YAW_INDEX",
    "CONTINUOUS_WIDTH",
    "DEATH_TAU",
    "DEATH_TSD_CAP",
    "FIELD_CONDITION_KEYS",
    "STATE_BASE_DIM",
    "STATE_CONTINUOUS_DIM",
    "SUBSTEP_PITCH_INDEX",
    "SUBSTEP_VALID_INDEX",
    "SUBSTEP_YAW_INDEX",
    "DeathState",
    "PlayerState",
    "camera_turns",
    "check_block_alive_agreement",
    "continuous_row_state",
    "death_state_for_table",
    "derive_death_state",
    "gathered_death_state",
    "has_continuous_columns",
    "held_fraction",
    "integrate_camera_angles",
    "latent_frame_rows",
    "owned_rows",
    "pack_substeps",
    "player_state_conditions",
    "wrap_degrees",
]

# ------------------------------------------------------------------------------------- table layout
#: Columns of a table without and with the continuous columns.
STATE_BASE_DIM = 6
STATE_CONTINUOUS_DIM = 13
#: The continuous columns of a 13-wide table: yaw, pitch, corpse, time since death, frozen x, y, z.
CONT_YAW_INDEX = 6
CONT_PITCH_INDEX = 7
CONT_CORPSE_INDEX = 8
CONT_TSD_INDEX = 9
CONT_FROZEN_XYZ = (10, 13)
CONTINUOUS_WIDTH = STATE_CONTINUOUS_DIM - STATE_BASE_DIM

#: Packed substep layout (:func:`pack_substeps`): buttons, pitch turn, yaw turn, valid flag.
SUBSTEP_PITCH_INDEX = -3
SUBSTEP_YAW_INDEX = -2
SUBSTEP_VALID_INDEX = -1

#: Dying decays as ``exp(-tsd / DEATH_TAU)``, tsd in latent frames, saturating at ``DEATH_TSD_CAP``.
DEATH_TAU = 5.0
DEATH_TSD_CAP = 32.0
#: The generator's player-state conditions (:func:`player_state_conditions`): what the field
#: builder reads, the arguments of :func:`~worldcast.player_state.field.player_state_field` in its
#: order.
FIELD_CONDITION_KEYS = (
    "player_state_table",
    "player_controls",
    "client_slot",
    "player_team_ids",
    "player_alive",
    "player_visible",
    "player_weapons",
)


@dataclass(frozen=True)
class DeathState:
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


def latent_frame_rows(video_frames: int) -> torch.Tensor:
    """``[F]`` long (CPU): the row each latent frame is read at, its last video frame ``4 f``, in a
    table of ``T = 1 + 4 (F - 1)`` video frames."""
    return torch.from_numpy(last_video_frames(0, latent_frame_count(video_frames)))


def owned_rows(latent_frames: int) -> torch.Tensor:
    """``[F, 4]`` long (CPU): the rows of the video frames each latent frame owns
    (:func:`~worldcast.data.latents.video_frames_of`); latent frame 0, which owns row 0 alone,
    repeats it."""
    owned = [video_frames_of(f) for f in range(int(latent_frames))]
    return torch.tensor([rows * (VIDEO_FRAMES_PER_LATENT // len(rows)) for rows in owned])


def pack_substeps(substeps: torch.Tensor, substep_valid: torch.Tensor) -> torch.Tensor:
    """Zero invalid substeps and append the valid flag, ``[..., Q, A]`` -> ``[..., Q, A + 1]``."""
    valid_flag = substep_valid.bool()
    packed = substeps.float() * valid_flag.unsqueeze(-1).float()
    return torch.cat([packed, valid_flag.unsqueeze(-1).float()], dim=-1)


def held_fraction(controls: torch.Tensor, button_index: int) -> torch.Tensor:
    """Share of a latent frame's valid substeps during which a button is held.

    Args:
        controls (torch.Tensor): ``[B, F, P, Q, A + 1]`` packed substeps (16 per latent frame).
        button_index (int): button column.

    Returns:
        torch.Tensor: ``[B, F, P]`` in ``[0, 1]``, ``controls``' dtype (0 where no substep is
        valid).
    """
    valid = (controls[..., SUBSTEP_VALID_INDEX] > 0.5).to(controls.dtype)
    held = (controls[..., button_index] > 0.5).to(controls.dtype) * valid
    return held.sum(-1) / valid.sum(-1).clamp(min=1.0)


# -------------------------------------------------------------------------------- continuous camera
def camera_turns(substeps: torch.Tensor, *, frame_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The pitch and yaw turn of every row, degrees: the turns of its valid substeps summed, times
    :data:`~worldcast.data.controls.CAMERA_DELTA_SCALE`.

    Args:
        substeps (torch.Tensor): ``[..., Q, A + 1]`` packed substeps (:func:`pack_substeps`).
        frame_dim (int): the axis of the rows; the turn of row 0 along it, which precedes the
            window, is zero.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(pitch, yaw)``, each ``[...]`` in ``substeps``' dtype.
    """
    valid = (substeps[..., SUBSTEP_VALID_INDEX] > 0.5).to(substeps.dtype)
    turns = []
    for index in (SUBSTEP_PITCH_INDEX, SUBSTEP_YAW_INDEX):
        turn = (substeps[..., index] * valid).sum(-1) * CAMERA_DELTA_SCALE
        turn.select(frame_dim, 0).zero_()
        turns.append(turn)
    return turns[0], turns[1]


def integrate_camera_angles(
    initial_state: torch.Tensor, substeps: torch.Tensor, *, clamp_pitch_sum: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Yaw and pitch of every player at every row, integrated from row 0 as the engine does.

    Row ``j`` covers ``(time(j-1), time(j)]``, so the angle at row ``j`` is the row-0 angle plus
    the turns of rows ``1..j`` (:func:`camera_turns`). Yaw wraps to ``[-180, 180)``; pitch is
    clamped to ``[-89, 89]`` row by row, so a turn that would leave the range is discarded rather
    than banked.

    Args:
        initial_state (torch.Tensor): ``[B, P, 6]`` table row 0 (yaw, pitch in degrees at columns 3,
            4).
        substeps (torch.Tensor): ``[B, P, T, Q, A + 1]`` packed substeps (:func:`pack_substeps`).
        clamp_pitch_sum (bool): clamp the summed pitch instead, so that a turn beyond the range is
            banked: the pitch of the foreground weight, as trained.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(yaw, pitch)``, each ``[B, P, T]`` float32 degrees.
    """
    turn_pitch, turn_yaw = camera_turns(substeps.float(), frame_dim=2)
    yaw = _turned_yaw(initial_state[..., 3], turn_yaw)
    if clamp_pitch_sum:
        pitch = initial_state[..., 4:5].float() + turn_pitch.cumsum(dim=2)
        return yaw, pitch.clamp(PITCH_MIN_DEGREES, PITCH_MAX_DEGREES)
    current = initial_state[..., 4].float().clamp(PITCH_MIN_DEGREES, PITCH_MAX_DEGREES)
    return yaw, _clamped_pitch(current, turn_pitch)


def _turned_yaw(yaw: torch.Tensor, turn_yaw: torch.Tensor) -> torch.Tensor:
    """``[B, P, T]`` the yaw at every row: ``yaw`` ``[B, P]`` plus the turns ``[B, P, T]`` summed,
    wrapped."""
    return wrap_degrees(yaw[..., None].float() + turn_yaw.cumsum(dim=2))


def _clamped_pitch(current: torch.Tensor, turn_pitch: torch.Tensor) -> torch.Tensor:
    """``[B, P, T]`` the pitch after each row: ``current`` ``[B, P]`` plus each row's turn
    ``[B, P, T]``, clamped to ``[-89, 89]`` row by row."""
    rows_out = []
    for row in range(int(turn_pitch.shape[2])):
        current = (current + turn_pitch[:, :, row]).clamp(PITCH_MIN_DEGREES, PITCH_MAX_DEGREES)
        rows_out.append(current)
    return torch.stack(rows_out, dim=2)


def continuous_row_state(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """The 7 continuous columns of the whole round's table.

    They are functions of the full history, so they are computed before any window gather, which
    gathers them by latent frame with the rows and never recomputes them.

    Args:
        batch (Mapping[str, torch.Tensor]): ``player_states``, ``player_control_substeps`` and
            ``player_control_substep_valid`` (module docstring).

    Returns:
        torch.Tensor: ``[B, F, P, 7]`` float32: continuous yaw, pitch (degrees,
        :func:`integrate_camera_angles` at the latent frames' rows), corpse (0/1), time since death
        (latent frames), frozen x, y, z (u).
    """
    states = batch["player_states"]
    rows = latent_frame_rows(int(states.shape[2])).to(states.device)
    packed = pack_substeps(batch["player_control_substeps"], batch["player_control_substep_valid"])
    yaw_rows, pitch_rows = integrate_camera_angles(states[:, :, 0], packed)
    yaw = yaw_rows.index_select(2, rows).permute(0, 2, 1)
    pitch = pitch_rows.index_select(2, rows).permute(0, 2, 1)
    latent_states = states.float().index_select(2, rows).permute(0, 2, 1, 3)
    columns = yaw.new_zeros(*yaw.shape, CONTINUOUS_WIDTH)
    _write_columns(columns, yaw, pitch, derive_death_state(latent_states), slice(None))
    return columns


def _write_columns(
    columns: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    death: DeathState,
    frames: slice,
) -> None:
    """Fill the latent frames ``frames`` of ``columns`` ``[B, F, P, 7]`` from their yaw and pitch
    ``[B, f, P]`` and the death state of the latent frames up to them."""
    lo, hi = CONT_FROZEN_XYZ
    out = columns[:, frames]
    out[..., CONT_YAW_INDEX - STATE_BASE_DIM] = yaw
    out[..., CONT_PITCH_INDEX - STATE_BASE_DIM] = pitch
    out[..., CONT_CORPSE_INDEX - STATE_BASE_DIM] = death.corpse[:, frames].to(yaw.dtype)
    out[..., CONT_TSD_INDEX - STATE_BASE_DIM] = death.time_since_death[:, frames].to(yaw.dtype)
    out[..., lo - STATE_BASE_DIM : hi - STATE_BASE_DIM] = death.frozen_states[:, frames, ..., :3]


def memory_continuous_columns(
    continuous_columns: torch.Tensor, memory_states: torch.Tensor, client_slot: torch.Tensor
) -> torch.Tensor:
    """The continuous columns of a window's memory frames.

    Every other player keeps its columns of the first frame. The client's row holds the memory
    entry's source player: its engine yaw and pitch at each memory frame's last video frame, no
    corpse, no time since death, and its own position as the frozen x, y, z.

    Args:
        continuous_columns (torch.Tensor): ``[B, F, P, 7]`` the round's columns
            (:func:`continuous_row_state`).
        memory_states (torch.Tensor): ``[B, 16, 6]`` the source player's states at the memory
            frames' 16 video frames.
        client_slot (torch.Tensor): ``[B]`` long, the client's slot.

    Returns:
        torch.Tensor: ``[B, 4, P, 7]`` float32, for the window gather (the memory input of
        ``worldcast.sampling.window.ROUND_CONTINUOUS_COLUMNS_KEY``).
    """
    rows = continuous_columns[:, :1].float().expand(-1, BLOCK, -1, -1).clone()
    states = memory_states.to(torch.float32)[
        :, VIDEO_FRAMES_PER_LATENT - 1 :: VIDEO_FRAMES_PER_LATENT
    ]
    yaw, pitch, corpse, time_since_death = (
        index - STATE_BASE_DIM
        for index in (CONT_YAW_INDEX, CONT_PITCH_INDEX, CONT_CORPSE_INDEX, CONT_TSD_INDEX)
    )
    frozen = slice(*(index - STATE_BASE_DIM for index in CONT_FROZEN_XYZ))
    for b, client in enumerate(client_slot.tolist()):
        rows[b, :, client, yaw] = states[b, :, 3]
        rows[b, :, client, pitch] = states[b, :, 4]
        rows[b, :, client, corpse] = 0.0
        rows[b, :, client, time_since_death] = 0.0
        rows[b, :, client, frozen] = states[b, :, :3]
    return rows


# ----------------------------------------------------------------------------- latent player states
@dataclass(frozen=True)
class PlayerState:
    """Every player's states and controls at latent rate, for one window (the client included).

    Attributes:
        controls (torch.Tensor): ``[B, F, P, 4 Q, A + 1]`` float32 packed substeps of the four video
            frames each latent frame owns (16 substeps per latent frame; the last column is the
            valid flag).
        xyz (torch.Tensor): ``[B, F, P, 3]`` float32 engine position, u.
        yaw (torch.Tensor): ``[B, F, P]`` float32 engine yaw, degrees.
        pitch (torch.Tensor): ``[B, F, P]`` float32 engine pitch, degrees.
        alive (torch.Tensor): ``[B, F, P]`` bool.
        client_slot (torch.Tensor): ``[B]`` long, the client's slot.
        team_ids (torch.Tensor): ``[B, P]`` engine team ids.
        weapon_ids (torch.Tensor): ``[B, F, P]`` long 52-way weapon id at the latent frame's last
            video frame.
        continuous (torch.Tensor | None): ``[B, F, P, 7]`` float32 continuous columns (a gathered
            window that carries them), or None.
    """

    controls: torch.Tensor
    xyz: torch.Tensor
    yaw: torch.Tensor
    pitch: torch.Tensor
    alive: torch.Tensor
    client_slot: torch.Tensor
    team_ids: torch.Tensor
    weapon_ids: torch.Tensor
    continuous: torch.Tensor | None = None

    @classmethod
    def from_batch(
        cls, batch: Mapping[str, torch.Tensor], *, device: torch.device | None = None
    ) -> "PlayerState":
        """Fold one loader batch (the whole round or a gathered window) to latent rate.

        The number of latent frames comes from the batch, so one call serves the whole round (the
        first six blocks) and a gathered window (21 or 17 latent frames at inference). A gathered
        batch may carry its continuous columns under :data:`CONTINUOUS_COLUMNS_KEY` (``[B, F, P,
        7]``); they are carried, never recomputed.

        Args:
            batch (Mapping[str, torch.Tensor]): the keys of the module docstring.
            device (torch.device | None): where to put the states (default: the batch's device).

        Returns:
            PlayerState: float tensors float32.
        """
        required = (
            "player_states",
            "player_team_ids",
            "player_weapon_ids",
            "player_control_substeps",
            "player_control_substep_valid",
            "client_slot",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"player-state batch is missing {missing}")
        states = batch["player_states"]
        team_ids = batch["player_team_ids"]
        client_slot = batch["client_slot"]
        if states.ndim != 4 or states.shape[-1] != STATE_BASE_DIM:
            raise ValueError("player_states must be [B, P, T, 6]")
        batch_size, players, video_frames = states.shape[:3]
        if team_ids.shape != (batch_size, players):
            raise ValueError("player_team_ids must be [B, P]")
        latent_frames = latent_frame_count(int(video_frames))

        source = states.device
        rows = latent_frame_rows(int(video_frames)).to(source)
        substeps = batch["player_control_substeps"]
        packed = pack_substeps(substeps, batch["player_control_substep_valid"])

        continuous = batch.get(CONTINUOUS_COLUMNS_KEY)
        if continuous is not None:
            expected = (batch_size, latent_frames, players, CONTINUOUS_WIDTH)
            if tuple(continuous.shape) != expected:
                raise ValueError(
                    f"{CONTINUOUS_COLUMNS_KEY} must be {list(expected)} (frame-major, latent"
                    f" rate), got {tuple(continuous.shape)}"
                )
            continuous = continuous.to(source).float()

        owned = owned_rows(latent_frames).reshape(-1).to(source)
        controls = packed.index_select(2, owned).reshape(
            batch_size,
            players,
            latent_frames,
            VIDEO_FRAMES_PER_LATENT * substeps.shape[3],
            packed.shape[-1],
        )
        controls = controls.permute(0, 2, 1, 3, 4).contiguous()

        latent_states = states.index_select(2, rows).permute(0, 2, 1, 3).contiguous()
        weapon_ids = batch["player_weapon_ids"].to(source).index_select(2, rows)
        weapon_ids = weapon_ids.permute(0, 2, 1).contiguous().long()
        alive = latent_states[..., ALIVE_INDEX] > 0.5

        target = source if device is None else torch.device(device)
        return cls(
            controls=controls.to(target).float(),
            xyz=latent_states[..., :3].contiguous().to(target).float(),
            yaw=latent_states[..., 3].contiguous().to(target).float(),
            pitch=latent_states[..., 4].contiguous().to(target).float(),
            alive=alive.contiguous().to(target),
            client_slot=client_slot.contiguous().to(target),
            team_ids=team_ids.contiguous().to(target),
            weapon_ids=weapon_ids.to(target),
            continuous=None if continuous is None else continuous.to(target).float(),
        )

    def table(self) -> torch.Tensor:
        """The player state table, ``[B, F, P, 6]``, or ``[B, F, P, 13]`` with the continuous
        columns."""
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


def player_state_conditions(
    states: PlayerState,
    visible: torch.Tensor,
    *,
    observer_signals: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """The generator's player-state conditions; the field is built from them inside the generator.

    Args:
        states (PlayerState): every player's latent-rate states.
        visible (torch.Tensor): ``[B, F, P]`` bool, the visibility gate per latent frame
            (``worldcast.player_state.visibility``; GT labels in Table 3).
        observer_signals (Mapping[str, torch.Tensor] | None): a mapping holding the five
            observer-signal rows ``[B, F]`` (the batch itself will do), or None to omit them.

    Returns:
        dict[str, torch.Tensor]: the entries :data:`FIELD_CONDITION_KEYS`
        (``player_state_table [B, F, P, 6|13]`` float32, ``player_controls [B, F, P, 16, 14]``
        float32, ``client_slot [B]``, ``player_team_ids [B, P]``, ``player_alive [B, F, P]``
        float32, ``player_visible [B, F, P]`` float32, ``player_weapons [B, F, P]`` long), then
        the observer-signal keys as long.
    """
    values = (
        states.table(),
        states.controls,
        states.client_slot,
        states.team_ids,
        states.alive.to(torch.float32),
        visible.to(torch.float32),
        states.weapon_ids,
    )
    conditions = dict(zip(FIELD_CONDITION_KEYS, values))
    if observer_signals is not None:
        for key in OBSERVER_SIGNAL_KEYS:
            conditions[key] = observer_signals[key].to(states.controls.device, torch.long)
    return conditions


# -------------------------------------------------------------------------------------- death state
def has_continuous_columns(states: torch.Tensor) -> bool:
    """True for a 13-wide table, False for a 6-wide one; raises for any other."""
    width = int(states.shape[-1])
    if width == STATE_BASE_DIM:
        return False
    if width == STATE_CONTINUOUS_DIM:
        return True
    raise ValueError(
        f"player state table must be {STATE_BASE_DIM} columns wide (x y z yaw pitch alive) or "
        f"{STATE_CONTINUOUS_DIM} (plus the continuous columns); got {width}"
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
    alive = states[..., ALIVE_INDEX] > 0.5
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
    since = idx - last_alive_idx
    time_since_death = (since.clamp(min=0).to(states.dtype) * dead.to(states.dtype)).clamp(
        max=DEATH_TSD_CAP
    )
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
    """Death state of a 13-wide table, read from its continuous columns.

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
    alive = states[..., ALIVE_INDEX] > 0.5
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
    """The death state of a 13-wide table (read from its columns) or a 6-wide one (derived)."""
    if has_continuous_columns(states):
        return gathered_death_state(states)
    return derive_death_state(states)


def check_block_alive_agreement(
    derived_alive: torch.Tensor,
    alive_window: torch.Tensor,
    *,
    frame_offset: int,
    num_frames: int,
) -> None:
    """Raise if the table's alive column and ``player_alive`` disagree on the call's latent frames.

    Args:
        derived_alive (torch.Tensor): ``[B, F, P]`` bool, from the table.
        alive_window (torch.Tensor): ``[B, F, P]`` ``player_alive`` padded back to the window
            (zeros outside the call's latent frames).
        frame_offset (int): the call's first latent frame.
        num_frames (int): its latent frames.
    """
    lo = int(frame_offset)
    hi = lo + int(num_frames)
    disagree = (alive_window[:, lo:hi] > 0.5) != derived_alive[:, lo:hi]
    if bool(disagree.any()):
        where = disagree.nonzero()
        raise ValueError(
            f"player_alive and the table's alive column disagree at {where.shape[0]} sites within"
            f" frames [{lo}, {hi}), first at {where[0].tolist()}"
        )
