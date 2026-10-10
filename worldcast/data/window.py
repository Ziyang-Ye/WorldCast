"""One client's window of a recorded round: every player's frames and the conditioning item.

A window is a span of a recorded round: a ten-second training window, or the whole round a client
generates. A window of N latent frames has ``T = 1 + 4 (N - 1)`` video frames at 16 fps: video
frame ``k`` is source frame ``start_frame + 2 k`` of the 32 fps recording. A player's state at a
video frame is the last tick at or before that frame's time. A window that cannot be built raises;
nothing is substituted.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldcast.config.loader import require_set

from .controls import (
    CONTROL_BUTTONS,
    DEFAULT_CAMERA_ENCODING,
    SUBSTEPS_PER_VIDEO_FRAME,
    align_ticks_to_ordered_substeps,
    align_ticks_to_video_frames,
    count_until,
)
from .game import TICK_RATE
from .labels import load_observer_signals, visibility_rows
from .latents import SOURCE_FRAMES_PER_VIDEO_FRAME, video_frame_count
from .recordings import (
    NUM_PLAYERS,
    MediaIndex,
    MediaRecord,
    RoundIndexRow,
    TickTable,
    read_player_ticks,
)

__all__ = [
    "COVERAGE_SLACK_FRAMES",
    "MAX_TICK_GAP_SECONDS",
    "ClientWindow",
    "DataPaths",
    "PlayerFrames",
    "WindowItem",
    "WindowRefused",
    "WindowSpec",
    "build_window_item",
    "client_record",
    "collate_windows",
    "covered_frames",
    "frame_endpoint_rows",
    "load_client_window",
]

#: Largest tolerated tick gap and frame-endpoint lag of a window, seconds.
MAX_TICK_GAP_SECONDS = 2.5 / TICK_RATE
#: The client's ticks may end this many video frames before the window end (the round ended inside
#: the last block; those frames read as dead).
COVERAGE_SLACK_FRAMES = 4


class WindowRefused(RuntimeError):
    """A window that cannot be served: the client's ticks end before it does."""


# ------------------------------------------------------------------------------------ player frames
@dataclass(frozen=True)
class WindowSpec:
    """How a client window samples the recordings.

    Attributes:
        latent_frames (int): N, the window's latent frames (41 for a ten-second window).
        max_tick_gap_seconds (float): largest tolerated tick gap and frame-endpoint lag, seconds.
        camera_encoding (str): the encoding of the client's turns
            (:data:`~worldcast.data.controls.CAMERA_ENCODINGS`): ``noclip`` for the released
            four-step generator, a training stage's ``Stage.camera_encoding`` otherwise.
    """

    latent_frames: int
    max_tick_gap_seconds: float = MAX_TICK_GAP_SECONDS
    camera_encoding: str = DEFAULT_CAMERA_ENCODING

    @property
    def video_frames(self) -> int:
        """T = 1 + 4 (N - 1)."""
        return video_frame_count(self.latent_frames)

    def source_frames(self, start_frame: int) -> np.ndarray:
        """``[T]`` int64 source frame of every video frame."""
        return int(start_frame) + SOURCE_FRAMES_PER_VIDEO_FRAME * np.arange(
            self.video_frames, dtype=np.int64
        )


def frame_endpoint_rows(
    timestamps: np.ndarray, source_fps: float, start_frame: int, spec: WindowSpec
) -> tuple[np.ndarray, int]:
    """Tick row read at each video frame, and how many leading video frames the ticks cover.

    Args:
        timestamps (np.ndarray): ``[n]`` float64 tick times, seconds.
        source_fps (float): source frames per second.
        start_frame (int): source frame of video frame 0.
        spec (WindowSpec): the window.

    Returns:
        tuple[np.ndarray, int]: ``rows`` ``[T]`` int64, the last tick at or before each video frame
        (:func:`~worldcast.data.controls.count_until`; clipped to ``[0, n - 1]``), and ``covered``,
        the video frames at or before ``t[-1] + max_tick_gap_seconds`` (at most T).
    """
    frame_times = spec.source_frames(start_frame).astype(np.float64) / source_fps
    rows = np.clip(count_until(timestamps, frame_times) - 1, 0, len(timestamps) - 1)
    covered = int(count_until(frame_times, timestamps[-1] + spec.max_tick_gap_seconds))
    return rows, min(spec.video_frames, covered)


def covered_frames(table: TickTable, media: MediaRecord, start_frame: int, spec: WindowSpec) -> int:
    """Video frames of the window (from ``start_frame``) the player's ticks cover (<= T)."""
    return frame_endpoint_rows(table.t, float(media.fps), start_frame, spec)[1]


@dataclass(frozen=True)
class PlayerFrames:
    """One player's states and controls at the ``T`` video frames of a window.

    Video frames past ``covered`` hold the last tick's pose with ``alive = 0`` and zero controls.

    Attributes:
        states (np.ndarray): ``[T, 6]`` float32 ``x, y, z, yaw, pitch, alive`` (u, degrees, {0, 1}).
        buttons (np.ndarray): ``[T, 11]`` float32 held buttons.
        view_deltas (np.ndarray): ``[T, 2]`` float32 mu-law ``pitch, yaw`` turn.
        weapon_ids (np.ndarray): ``[T]`` int64 weapon id.
        control_substeps (np.ndarray): ``[T, 4, 13]`` float32 ordered substeps: the buttons, then
            the pitch and yaw turns.
        control_substep_valid (np.ndarray): ``[T, 4]`` bool.
        team_id (int): engine team of the first tick (0 for a slot without media).
        covered (int): leading video frames within ``max_tick_gap_seconds`` of the last tick.
    """

    states: np.ndarray
    buttons: np.ndarray
    view_deltas: np.ndarray
    weapon_ids: np.ndarray
    control_substeps: np.ndarray
    control_substep_valid: np.ndarray
    team_id: int
    covered: int

    @classmethod
    def from_ticks(
        cls, table: TickTable, media: MediaRecord, start_frame: int, spec: WindowSpec
    ) -> "PlayerFrames":
        """A present player's frames for the window starting at source frame ``start_frame``."""
        source_fps = float(media.fps)
        if source_fps <= 0:
            raise ValueError(f"fps {media.fps} of {media.media_id} is not positive")
        ticks_per_frame = TICK_RATE / source_fps
        if abs(ticks_per_frame - round(ticks_per_frame)) > 1e-9:
            raise ValueError(f"tick rate {TICK_RATE} is not a multiple of fps {media.fps}")
        frames = cls.absent(spec)
        rows, covered = frame_endpoint_rows(table.t, source_fps, start_frame, spec)
        states = table.states(rows)
        states[covered:, -1] = 0.0  # past the ticks the player reads as dead
        if covered > 0:
            ticks = dict(
                timestamps=table.t,
                held_buttons=table.active,
                delta_pitch=table.delta_pitch,
                delta_yaw=table.delta_yaw,
                start_frame=int(start_frame),
                num_frames=int(covered),
                source_fps=source_fps,
                max_tick_gap_seconds=spec.max_tick_gap_seconds,
            )
            controls = align_ticks_to_video_frames(
                **ticks, input_weapons=table.input_weapon, camera_encoding=spec.camera_encoding
            )
            frames.buttons[:covered] = controls.buttons
            frames.view_deltas[:covered] = controls.view_deltas
            frames.weapon_ids[:covered] = controls.weapon
            substeps = align_ticks_to_ordered_substeps(**ticks)
            frames.control_substeps[:covered] = substeps.values
            frames.control_substep_valid[:covered] = substeps.valid
        team_id = int(table.team_num[0])
        if not 0 <= team_id < 4:
            raise ValueError(f"team_num {team_id} outside the 4-entry team vocabulary")
        return cls(
            states=states,
            buttons=frames.buttons,
            view_deltas=frames.view_deltas,
            weapon_ids=frames.weapon_ids,
            control_substeps=frames.control_substeps,
            control_substep_valid=frames.control_substep_valid,
            team_id=team_id,
            covered=int(covered),
        )

    @classmethod
    def absent(cls, spec: WindowSpec) -> "PlayerFrames":
        """The frames of a slot with no media in the round: all zeros (dead, silent, team 0)."""
        t, width, substeps = spec.video_frames, len(CONTROL_BUTTONS), SUBSTEPS_PER_VIDEO_FRAME
        return cls(
            states=np.zeros((t, 6), dtype=np.float32),
            buttons=np.zeros((t, width), dtype=np.float32),
            view_deltas=np.zeros((t, 2), dtype=np.float32),
            weapon_ids=np.zeros((t,), dtype=np.int64),
            control_substeps=np.zeros((t, substeps, width + 2), dtype=np.float32),
            control_substep_valid=np.zeros((t, substeps), dtype=np.bool_),
            team_id=0,
            covered=0,
        )


# ----------------------------------------------------------------------------------------- the item
@dataclass(frozen=True)
class DataPaths:
    """The data artefacts of a client (docs/inference.md, "Data").

    ``observer_signal_label_root`` holds ``flashlabels/`` and ``scopelabels/``.
    """

    dataset_root: Path
    media_index: Path
    latent_cache_root: Path
    visibility_label_root: Path
    observer_signal_label_root: Path

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """The paths' names, as the configs spell them."""
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_config(cls, section: Any, where: str) -> "DataPaths":
        """The paths of the config section ``where`` that names them alike (``paths`` of an
        inference config, ``data`` of a training config); raises ``ValueError`` for an unset
        one."""
        require_set(section, where, cls.names())
        return cls(**{name: Path(getattr(section, name)) for name in cls.names()})


@dataclass
class WindowItem:
    """The conditioning tensors of one client window, on the CPU.

    N latent frames, T = 1 + 4 (N - 1) video frames, P = 10 players.

    Attributes:
        media_id (str): the client's recording.
        start_frame (int): the window's first source frame.
        client_slot (int): the client's slot.
        buttons (torch.Tensor): ``[T, 11]`` float32, the client's buttons.
        view_deltas (torch.Tensor): ``[T, 2]`` float32, the client's mu-law turn.
        weapon (torch.Tensor): ``[T]`` int64, the client's weapon id.
        player_states (torch.Tensor): ``[P, T, 6]`` float32 ``x, y, z, yaw, pitch, alive``.
        player_weapon_ids (torch.Tensor): ``[P, T]`` int64.
        player_control_substeps (torch.Tensor): ``[P, T, 4, 13]`` float32.
        player_control_substep_valid (torch.Tensor): ``[P, T, 4]`` bool.
        player_team_ids (torch.Tensor): ``[P]`` int64.
        signals (dict[str, np.ndarray]): the client's observer signals
            (:data:`~worldcast.data.labels.OBSERVER_SIGNAL_KEYS`), ``[N]`` int64 each.
        client_visibility (torch.Tensor): ``[P, T]`` float32, the client's GT visibility labels.
        client_visibility_valid (torch.Tensor): ``[P, T]`` bool, where the labels are defined.
    """

    media_id: str
    start_frame: int
    client_slot: int
    buttons: torch.Tensor
    view_deltas: torch.Tensor
    weapon: torch.Tensor
    player_states: torch.Tensor
    player_weapon_ids: torch.Tensor
    player_control_substeps: torch.Tensor
    player_control_substep_valid: torch.Tensor
    player_team_ids: torch.Tensor
    signals: dict[str, np.ndarray]
    client_visibility: torch.Tensor
    client_visibility_valid: torch.Tensor

    def batch_dict(self) -> dict:
        """The item as a batch dict without the batch axis; ``metadata`` names the window
        (``media_id``, ``start_frame``)."""
        out = {
            "buttons": self.buttons,
            "view_deltas": self.view_deltas,
            "weapon": self.weapon,
            "player_states": self.player_states,
            "player_weapon_ids": self.player_weapon_ids,
            "player_control_substeps": self.player_control_substeps,
            "player_control_substep_valid": self.player_control_substep_valid,
            "player_team_ids": self.player_team_ids,
            "client_slot": torch.tensor(self.client_slot, dtype=torch.long),
            "client_visibility": self.client_visibility,
            "client_visibility_valid": self.client_visibility_valid,
            "metadata": {"media_id": self.media_id, "start_frame": self.start_frame},
        }
        out.update({key: torch.from_numpy(value) for key, value in self.signals.items()})
        return out


def collate_windows(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Batch items: tensors are stacked on a new leading axis, other values become lists."""
    return {
        key: (
            torch.stack([item[key] for item in items])
            if isinstance(items[0][key], torch.Tensor)
            else [item[key] for item in items]
        )
        for key in items[0]
    }


@dataclass
class ClientWindow:
    """Everything the recordings give one client.

    Attributes:
        row (RoundIndexRow): the client's round-index row.
        spec (WindowSpec): the window, at the length the client runs.
        media (MediaRecord): the client's media row.
        round_slots (dict[int, MediaRecord]): the round's media rows by player slot.
        tick_tables (dict[int, TickTable]): tick table of every present slot.
        player_frames (dict[int, PlayerFrames]): frames of every slot (absent slots hold zeros).
        item (WindowItem): the conditioning item.
    """

    row: RoundIndexRow
    spec: WindowSpec
    media: MediaRecord
    round_slots: dict[int, MediaRecord]
    tick_tables: dict[int, TickTable]
    player_frames: dict[int, PlayerFrames]
    item: WindowItem

    @property
    def client_frames(self) -> PlayerFrames:
        """The client's frames."""
        return self.player_frames[self.media.player_slot]


def client_record(row: RoundIndexRow, media_index: MediaIndex) -> MediaRecord:
    """The client's media row, checked against its index row (match, map, round, slot)."""
    media = media_index.media(row.media_id)
    if (media.match_id, media.map_name, media.round) != (row.match_id, row.map_name, row.round):
        raise ValueError(
            f"index row {row.media_id} disagrees with its media row on match/map/round"
        )
    if row.player_slot is not None and row.player_slot != media.player_slot:
        raise ValueError(
            f"index row {row.media_id} names slot {row.player_slot}, media row {media.player_slot}"
        )
    return media


def build_window_item(
    *,
    row: RoundIndexRow,
    media: MediaRecord,
    player_frames: dict[int, PlayerFrames],
    client_visibility: np.ndarray,
    client_visibility_valid: np.ndarray,
    signals: Mapping[str, np.ndarray],
) -> WindowItem:
    """Stack the frames of every slot ``0 .. 9`` into the item tensors."""
    slots = range(NUM_PLAYERS)
    client = player_frames[media.player_slot]

    def stack(name: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([getattr(player_frames[s], name) for s in slots]))

    return WindowItem(
        media_id=media.media_id,
        start_frame=int(row.start_frame),
        client_slot=int(media.player_slot),
        buttons=torch.from_numpy(client.buttons.copy()),
        view_deltas=torch.from_numpy(client.view_deltas.copy()),
        weapon=torch.from_numpy(client.weapon_ids.copy()),
        player_states=stack("states"),
        player_weapon_ids=stack("weapon_ids"),
        player_control_substeps=stack("control_substeps"),
        player_control_substep_valid=stack("control_substep_valid"),
        player_team_ids=torch.tensor([player_frames[s].team_id for s in slots], dtype=torch.long),
        signals=signals,
        client_visibility=torch.from_numpy(np.asarray(client_visibility, dtype=np.float32)),
        client_visibility_valid=torch.from_numpy(
            np.asarray(client_visibility_valid, dtype=np.bool_)
        ),
    )


def load_client_window(
    row: RoundIndexRow,
    media_index: MediaIndex,
    paths: DataPaths,
    spec: WindowSpec,
    *,
    missing_signals_ok: bool = False,
) -> ClientWindow:
    """Load one client's window: every player's ticks and frames, the labels and the item.

    Args:
        row (RoundIndexRow): the client's round-index row.
        media_index (MediaIndex): the media index.
        paths (DataPaths): the data artefacts.
        spec (WindowSpec): the window at the length N the client runs.
        missing_signals_ok (bool): a missing observer-signal file reads as unknown instead of
            raising (:func:`~worldcast.data.labels.load_observer_signal_curves`).

    Returns:
        ClientWindow: the client's data.

    Raises:
        WindowRefused: the client's ticks end more than :data:`COVERAGE_SLACK_FRAMES` video frames
            before the window does.
    """
    media = client_record(row, media_index)
    round_slots = media_index.slots(media.round_key)
    start_frame = int(row.start_frame)
    tick_tables = {
        slot: read_player_ticks(slot_media, paths.dataset_root)
        for slot, slot_media in sorted(round_slots.items())
    }
    player_frames = {
        slot: (
            PlayerFrames.from_ticks(tick_tables[slot], round_slots[slot], start_frame, spec)
            if slot in round_slots
            else PlayerFrames.absent(spec)
        )
        for slot in range(NUM_PLAYERS)
    }
    covered = player_frames[media.player_slot].covered
    if spec.video_frames - covered > COVERAGE_SLACK_FRAMES:
        raise WindowRefused(
            f"client {media.media_id} does not cover the whole window from {start_frame} "
            f"({covered}/{spec.video_frames} video frames)"
        )
    visible, valid = visibility_rows(
        paths.visibility_label_root, media.media_id, spec.source_frames(start_frame)
    )
    signals = load_observer_signals(
        paths.observer_signal_label_root,
        media.media_id,
        source_frames=media.source_frames,
        start_frame=start_frame,
        latent_frames=spec.latent_frames,
        missing_ok=missing_signals_ok,
    )
    item = build_window_item(
        row=row,
        media=media,
        player_frames=player_frames,
        client_visibility=visible,
        client_visibility_valid=valid,
        signals=signals,
    )
    return ClientWindow(
        row=row,
        spec=spec,
        media=media,
        round_slots=dict(round_slots),
        tick_tables=tick_tables,
        player_frames=player_frames,
        item=item,
    )
