"""One player's states and controls at every pixel frame of a window.

A window of N latent frames has ``T = 1 + 4 (N - 1)`` pixel frames at 16 fps: pixel frame ``k`` is
source frame ``start_frame + skip_frame * k`` of the 32 fps recording. Latent frame 0 holds pixel
frame 0, latent frame ``f >= 1`` pixel frames ``4f - 3 .. 4f``. A player's state at a pixel frame is
the last tick at or before that frame's time.
"""

from dataclasses import dataclass

import numpy as np

from .actions import (
    PAPER_ACTION_BUTTONS,
    PAPER_CAMERA_ENCODING,
    align_ticks_to_ordered_substeps,
    align_ticks_to_video_frames,
)
from .media import MediaRecord
from .ticks import TICK_RATE, TickTable


@dataclass(frozen=True)
class WindowSpec:
    """How a client window samples the recordings; the defaults are the paper's.

    Attributes:
        latent_frames (int): N, the window's latent frames (441 for the paper's 110 s clients).
        skip_frame (int): source frames per pixel frame (config ``data.skip_frame``).
        max_tick_gap_seconds (float): largest tolerated tick gap and frame-endpoint lag, seconds
            (2.5 / 64; config ``data.max_tick_gap_seconds``).
        button_names (tuple[str, ...]): the button channels (config ``data.action_buttons``).
        num_substeps (int): substeps per pixel frame of the players' controls
            (config ``model.player_field.action_substeps``).
        camera_delta_scale (float): degrees per unit substep camera delta
            (config ``model.player_field.camera_delta_scale``).
        camera_encoding (str): config ``data.camera_encoding``.
    """

    latent_frames: int
    skip_frame: int = 2
    max_tick_gap_seconds: float = 0.0390625
    button_names: tuple[str, ...] = PAPER_ACTION_BUTTONS
    num_substeps: int = 4
    camera_delta_scale: float = 5.0
    camera_encoding: str = PAPER_CAMERA_ENCODING

    @property
    def pixel_frames(self) -> int:
        """T = 1 + 4 (N - 1)."""
        return 1 + 4 * (int(self.latent_frames) - 1)

    def source_frames(self, start_frame: int) -> np.ndarray:
        """``[T]`` int64 source-frame index of every pixel frame."""
        return int(start_frame) + int(self.skip_frame) * np.arange(
            self.pixel_frames, dtype=np.int64
        )


def frame_endpoint_rows(
    timestamps: np.ndarray, video_fps: float, start_frame: int, spec: WindowSpec
) -> tuple[np.ndarray, int]:
    """Tick row sampled at each pixel frame, and how many leading frames the ticks cover.

    Args:
        timestamps (np.ndarray): ``[n]`` float64 tick times, seconds.
        video_fps (float): source frame rate.
        start_frame (int): source frame of pixel frame 0.
        spec (WindowSpec): the window.

    Returns:
        tuple[np.ndarray, int]: ``rows`` ``[T]`` int64, the last tick with ``t <= frame time +
        1e-7`` (clipped to ``[0, n - 1]``), and ``covered``, the frames with time ``<= t[-1] +
        max_tick_gap_seconds + 1e-7`` (at most T).
    """
    frame_ids = start_frame + spec.skip_frame * np.arange(spec.pixel_frames, dtype=np.int64)
    frame_times = frame_ids.astype(np.float64) / video_fps
    end = np.searchsorted(timestamps, frame_times + 1e-7, side="right")
    rows = np.clip(end - 1, 0, len(timestamps) - 1)
    covered = int(
        np.searchsorted(
            frame_times, timestamps[-1] + spec.max_tick_gap_seconds + 1e-7, side="right"
        )
    )
    return rows, min(spec.pixel_frames, covered)


def covered_frames(table: TickTable, media: MediaRecord, start_frame: int, spec: WindowSpec) -> int:
    """Pixel frames of the window (from ``start_frame``) the player's ticks cover (<= T)."""
    return frame_endpoint_rows(table.t, float(media.fps), start_frame, spec)[1]


@dataclass(frozen=True)
class PlayerFrames:
    """One player's states and controls at the ``T`` pixel frames of a window.

    Frames past ``covered`` hold the last tick's pose with ``alive = 0`` and zero controls.

    Attributes:
        states (np.ndarray): ``[T, 6]`` float32 ``x, y, z, yaw, pitch, alive`` (u, degrees, {0, 1}).
        camera_abs (np.ndarray): ``[T, 2]`` float32 absolute ``pitch, yaw``, degrees.
        buttons (np.ndarray): ``[T, B]`` float32 held buttons.
        camera_quantized (np.ndarray): ``[T, 2]`` float32 mu-law turn.
        weapon_ids (np.ndarray): ``[T]`` int64 weapon id.
        action_substeps (np.ndarray): ``[T, S, B + 2]`` float32 ordered substeps.
        action_substep_valid (np.ndarray): ``[T, S]`` bool.
        team_id (int): engine team of the first tick (0 for a seat without media).
        covered (int): leading frames within ``max_tick_gap_seconds`` of the last tick.
    """

    states: np.ndarray
    camera_abs: np.ndarray
    buttons: np.ndarray
    camera_quantized: np.ndarray
    weapon_ids: np.ndarray
    action_substeps: np.ndarray
    action_substep_valid: np.ndarray
    team_id: int
    covered: int

    @classmethod
    def from_ticks(
        cls, table: TickTable, media: MediaRecord, start_frame: int, spec: WindowSpec
    ) -> "PlayerFrames":
        """A present player's frames for the window starting at source frame ``start_frame``."""
        ratio = TICK_RATE / float(media.fps)
        if float(media.fps) <= 0 or abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"tick rate {TICK_RATE} is not a multiple of fps {media.fps}")
        frames = cls.absent(spec)
        rows, covered = frame_endpoint_rows(table.t, float(media.fps), start_frame, spec)
        alive_pad = np.arange(spec.pixel_frames) < covered

        def column(values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=np.float32)[rows]

        states = np.stack(
            [
                column(table.x),
                column(table.y),
                column(table.z),
                column(table.yaw),
                column(table.pitch),
                column(table.is_alive).astype(np.float32) * alive_pad.astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)
        camera_abs = np.stack([column(table.pitch), column(table.yaw)], axis=-1).astype(np.float32)
        if covered > 0:
            ticks = dict(
                timestamps=table.t,
                active_actions=table.active,
                delta_pitch=table.delta_pitch,
                delta_yaw=table.delta_yaw,
                start_frame=int(start_frame),
                num_frames=int(covered),
                video_fps=float(media.fps),
                skip_frame=spec.skip_frame,
                max_tick_gap_seconds=spec.max_tick_gap_seconds,
                button_names=spec.button_names,
            )
            actions = align_ticks_to_video_frames(
                **ticks, input_weapons=table.input_weapon, camera_encoding=spec.camera_encoding
            )
            frames.buttons[:covered] = actions.buttons
            frames.camera_quantized[:covered] = actions.camera
            frames.weapon_ids[:covered] = actions.weapon
            substeps = align_ticks_to_ordered_substeps(
                **ticks,
                num_substeps=spec.num_substeps,
                camera_delta_scale=spec.camera_delta_scale,
            )
            frames.action_substeps[:covered] = substeps.values
            frames.action_substep_valid[:covered] = substeps.valid
        team_id = int(table.team_num[0])
        if not 0 <= team_id < 4:
            raise ValueError(f"team_num {team_id} outside the 4-entry team vocabulary")
        return cls(
            states=states,
            camera_abs=camera_abs,
            buttons=frames.buttons,
            camera_quantized=frames.camera_quantized,
            weapon_ids=frames.weapon_ids,
            action_substeps=frames.action_substeps,
            action_substep_valid=frames.action_substep_valid,
            team_id=team_id,
            covered=int(covered),
        )

    @classmethod
    def absent(cls, spec: WindowSpec) -> "PlayerFrames":
        """The frames of a seat with no media in the round: all zeros (dead, silent, team 0)."""
        t, width, substeps = spec.pixel_frames, len(spec.button_names), spec.num_substeps
        return cls(
            states=np.zeros((t, 6), dtype=np.float32),
            camera_abs=np.zeros((t, 2), dtype=np.float32),
            buttons=np.zeros((t, width), dtype=np.float32),
            camera_quantized=np.zeros((t, 2), dtype=np.float32),
            weapon_ids=np.zeros((t,), dtype=np.int64),
            action_substeps=np.zeros((t, substeps, width + 2), dtype=np.float32),
            action_substep_valid=np.zeros((t, substeps), dtype=np.bool_),
            team_id=0,
            covered=0,
        )
