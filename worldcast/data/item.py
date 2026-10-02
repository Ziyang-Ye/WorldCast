"""One client's window: every player's ticks and frames, the conditioning item, the sink latent.

A window that cannot be built raises; nothing is substituted. A missing observer-signal file is an
error.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .index import RoundIndexRow
from .latents import load_first_latent
from .media import MediaIndex, MediaRecord
from .obs_signals import ObsSignals, load_obs_signals
from .player_frames import PlayerFrames, WindowSpec
from .ticks import TickTable, read_player_ticks
from .visibility_labels import VisibilityLabels, observer_visibility_rows

#: The fixed text prompt of every window (config ``model.fixed_prompt``).
FIXED_PROMPT = "first-person Counter-Strike 2 gameplay"
#: Player slots per round.
NUM_PLAYERS = 10
#: The client's own ticks may end this many pixel frames before the window end (the round ended
#: inside the last block; those frames read as dead).
VIEW_COVERAGE_SLACK_FRAMES = 4


@dataclass(frozen=True)
class DataPaths:
    """The data artefacts of a client (config ``paths.<name>``; docs/data.md).

    ``obs_signal_label_root`` holds ``flashlabels/`` and ``scopelabels/``.
    """

    dataset_root: Path
    media_index: Path
    latent_cache_root: Path
    visibility_label_root: Path
    obs_signal_label_root: Path


@dataclass(frozen=True)
class WindowIdentity:
    """Which window an item is. ``fps``: frame rate of the pixel frames (16.0)."""

    media_id: str
    match_id: int
    map_name: str
    round: int
    start_frame: int
    latent_key: str
    fps: float
    skip_frame: int


@dataclass
class WindowItem:
    """The conditioning tensors of one client window, on the CPU.

    N latent frames, T = 1 + 4 (N - 1) pixel frames, P = 10 players, B = 11 buttons, S = 4 substeps.

    Attributes:
        identity (WindowIdentity): which window.
        prompt (str): the text prompt.
        observer_slot (int): the client's own seat.
        button_condition (torch.Tensor): ``[T, B]`` float32, the client's buttons.
        camera_condition (torch.Tensor): ``[T, 2]`` float32, the client's mu-law turn.
        weapon_condition (torch.Tensor): ``[T]`` int64, the client's weapon id.
        player_states (torch.Tensor): ``[P, T, 6]`` float32 ``x, y, z, yaw, pitch, alive``.
        player_buttons (torch.Tensor): ``[P, T, B]`` float32.
        player_camera (torch.Tensor): ``[P, T, 2]`` float32 absolute ``pitch, yaw``, degrees.
        player_weapon_ids (torch.Tensor): ``[P, T]`` int64.
        player_action_substeps (torch.Tensor): ``[P, T, S, B + 2]`` float32.
        player_action_substep_valid (torch.Tensor): ``[P, T, S]`` bool.
        player_team_ids (torch.Tensor): ``[P]`` int64.
        obs (ObsSignals): per-latent observer signals, ``[N]`` each.
        observer_visibility (torch.Tensor): ``[P, T]`` float32, the client's GT visibility labels.
        observer_visibility_valid (torch.Tensor): ``[P, T]`` bool, where the labels are defined.
    """

    identity: WindowIdentity
    prompt: str
    observer_slot: int
    button_condition: torch.Tensor
    camera_condition: torch.Tensor
    weapon_condition: torch.Tensor
    player_states: torch.Tensor
    player_buttons: torch.Tensor
    player_camera: torch.Tensor
    player_weapon_ids: torch.Tensor
    player_action_substeps: torch.Tensor
    player_action_substep_valid: torch.Tensor
    player_team_ids: torch.Tensor
    obs: ObsSignals
    observer_visibility: torch.Tensor
    observer_visibility_valid: torch.Tensor

    def batch_dict(self) -> dict:
        """The item as a batch dict without the batch axis; ``metadata`` holds the identity."""
        identity = self.identity
        out = {
            "prompts": self.prompt,
            "button_condition": self.button_condition,
            "camera_condition": self.camera_condition,
            "weapon_condition": self.weapon_condition,
            "player_states": self.player_states,
            "player_buttons": self.player_buttons,
            "player_camera": self.player_camera,
            "player_weapon_ids": self.player_weapon_ids,
            "player_action_substeps": self.player_action_substeps,
            "player_action_substep_valid": self.player_action_substep_valid,
            "player_team_ids": self.player_team_ids,
            "observer_slot": torch.tensor(self.observer_slot, dtype=torch.long),
            "observer_visibility": self.observer_visibility,
            "observer_visibility_valid": self.observer_visibility_valid,
            "metadata": {
                "media_id": identity.media_id,
                "match_id": identity.match_id,
                "map_name": identity.map_name,
                "round": identity.round,
                "start_frame": identity.start_frame,
                "latent_key": identity.latent_key,
                "fps": identity.fps,
                "skip_frame": identity.skip_frame,
            },
        }
        out.update({key: torch.from_numpy(value) for key, value in self.obs.as_dict().items()})
        return out


@dataclass
class ClientWindow:
    """Everything the data path gives one client.

    Attributes:
        row (RoundIndexRow): the client's round-index row.
        spec (WindowSpec): the window, at the length the client runs.
        observer (MediaRecord): the client's own media row.
        round_slots (dict[int, MediaRecord]): the round's media rows by player slot.
        tick_tables (dict[int, TickTable]): tick table of every present slot.
        player_frames (dict[int, PlayerFrames]): frames of every slot (absent slots hold zeros); the
            scene state and the pool reuse them for published peer blocks.
        item (WindowItem): the conditioning item.
        first_latent (torch.Tensor): ``[1, 48, 24, 42]`` float32 latent 0 of the cached window (the
            sink).
    """

    row: RoundIndexRow
    spec: WindowSpec
    observer: MediaRecord
    round_slots: dict[int, MediaRecord]
    tick_tables: dict[int, TickTable]
    player_frames: dict[int, PlayerFrames]
    item: WindowItem
    first_latent: torch.Tensor


def observer_record(row: RoundIndexRow, media_index: MediaIndex) -> MediaRecord:
    """The client's own media row, checked against its index row (match, map, round, slot)."""
    observer = media_index.media(row.media_id)
    if (observer.match_id, observer.map_name, observer.round) != (
        row.match_id,
        row.map_name,
        row.round,
    ):
        raise ValueError(
            f"index row {row.media_id} disagrees with its media row on match/map/round"
        )
    if row.player_slot is not None and row.player_slot != observer.player_slot:
        raise ValueError(
            f"index row {row.media_id} names slot {row.player_slot}, media row"
            f" {observer.player_slot}"
        )
    return observer


def build_window_item(
    *,
    row: RoundIndexRow,
    observer: MediaRecord,
    player_frames: dict[int, PlayerFrames],
    observer_visible: np.ndarray,
    observer_visible_valid: np.ndarray,
    obs: ObsSignals,
    spec: WindowSpec,
    num_players: int = NUM_PLAYERS,
    prompt: str = FIXED_PROMPT,
) -> WindowItem:
    """Stack the frames of every slot ``0 .. num_players - 1`` into the item tensors.

    Raises:
        RuntimeError: the client's own ticks end more than :data:`VIEW_COVERAGE_SLACK_FRAMES` pixel
            frames before the window end.
    """
    slots = range(int(num_players))
    missing = [s for s in slots if s not in player_frames]
    if missing:
        raise ValueError(f"player frames missing for slots {missing}")
    own = player_frames[observer.player_slot]
    if spec.pixel_frames - int(own.covered) > VIEW_COVERAGE_SLACK_FRAMES:
        raise RuntimeError(
            f"view player {observer.media_id} does not cover the whole window "
            f"({own.covered}/{spec.pixel_frames} frames)"
        )

    def stack(name: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([getattr(player_frames[s], name) for s in slots]))

    identity = WindowIdentity(
        media_id=observer.media_id,
        match_id=observer.match_id,
        map_name=observer.map_name,
        round=observer.round,
        start_frame=int(row.start_frame),
        latent_key=row.latent_key,
        fps=float(observer.fps) / spec.skip_frame,
        skip_frame=spec.skip_frame,
    )
    return WindowItem(
        identity=identity,
        prompt=str(prompt),
        observer_slot=int(observer.player_slot),
        button_condition=torch.from_numpy(own.buttons.copy()),
        camera_condition=torch.from_numpy(own.camera_quantized.copy()),
        weapon_condition=torch.from_numpy(own.weapon_ids.copy()),
        player_states=stack("states"),
        player_buttons=stack("buttons"),
        player_camera=stack("camera_abs"),
        player_weapon_ids=stack("weapon_ids"),
        player_action_substeps=stack("action_substeps"),
        player_action_substep_valid=stack("action_substep_valid"),
        player_team_ids=torch.tensor([player_frames[s].team_id for s in slots], dtype=torch.long),
        obs=obs,
        observer_visibility=torch.from_numpy(np.asarray(observer_visible, dtype=np.float32)),
        observer_visibility_valid=torch.from_numpy(
            np.asarray(observer_visible_valid, dtype=np.bool_)
        ),
    )


def load_client_window(
    row: RoundIndexRow,
    media_index: MediaIndex,
    paths: DataPaths,
    spec: WindowSpec,
    *,
    prompt: str = FIXED_PROMPT,
) -> ClientWindow:
    """Load one client's window: every player's ticks and frames, the item and the sink latent.

    Args:
        row (RoundIndexRow): the client's round-index row.
        media_index (MediaIndex): the media index.
        paths (DataPaths): the data artefacts.
        spec (WindowSpec): the window at the length N the client runs, already clipped to its own
            coverage (:func:`worldcast.data.player_frames.covered_frames`).
        prompt (str): the text prompt.

    Returns:
        ClientWindow: the client's data.
    """
    observer = observer_record(row, media_index)
    round_slots = media_index.slots(observer.round_key)
    start_frame = int(row.start_frame)
    tick_tables = {
        slot: read_player_ticks(media, paths.dataset_root)
        for slot, media in sorted(round_slots.items())
    }
    player_frames = {
        slot: (
            PlayerFrames.from_ticks(tick_tables[slot], round_slots[slot], start_frame, spec)
            if slot in round_slots
            else PlayerFrames.absent(spec)
        )
        for slot in range(NUM_PLAYERS)
    }
    visible, valid = observer_visibility_rows(
        VisibilityLabels(paths.visibility_label_root),
        observer.media_id,
        spec.source_frames(start_frame),
        num_players=NUM_PLAYERS,
    )
    obs = load_obs_signals(
        paths.obs_signal_label_root,
        observer.media_id,
        video_frames=observer.video_frames,
        start_frame=start_frame,
        latent_frames=spec.latent_frames,
        skip_frame=spec.skip_frame,
    )
    item = build_window_item(
        row=row,
        observer=observer,
        player_frames=player_frames,
        observer_visible=visible,
        observer_visible_valid=valid,
        obs=obs,
        spec=spec,
        prompt=prompt,
    )
    return ClientWindow(
        row=row,
        spec=spec,
        observer=observer,
        round_slots=dict(round_slots),
        tick_tables=tick_tables,
        player_frames=player_frames,
        item=item,
        first_latent=load_first_latent(paths.latent_cache_root, observer.media_id, start_frame),
    )
