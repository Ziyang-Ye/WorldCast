"""A synthetic player for the data tests: tick tables and media rows in memory, label files."""

from pathlib import Path

import numpy as np

from worldcast.data.game import TICK_RATE
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS, flash_label_path, scope_label_path
from worldcast.data.recordings import MediaRecord, TickTable

TICKS_PER_SECOND = int(TICK_RATE)
#: Frames of the ``recording`` fixture, and the grey level of each.
RECORDING_FRAMES = 12
RECORDING_LEVELS = [20 * i + 10 for i in range(RECORDING_FRAMES)]


def tick_table(
    seconds: float = 4.0, *, dies_at: float | None = None, y: float = 0.0, team: int = 2
) -> TickTable:
    """A player that walks along +x at 250 u/s (at ``y``) and turns 30 degrees per second, holding
    ``forward``; dead from ``dies_at`` seconds on."""
    n = int(seconds * TICKS_PER_SECOND)
    t = np.arange(n, dtype=np.float64) / TICK_RATE
    alive = np.ones(n, dtype=bool) if dies_at is None else t < dies_at
    return TickTable(
        t=t,
        x=(250.0 * t).astype(np.float32).astype(np.float64),
        y=np.full(n, y),
        z=np.zeros(n),
        yaw=(30.0 * t).astype(np.float32).astype(np.float64),
        pitch=np.zeros(n),
        is_alive=alive,
        active=[["forward"]] * n,
        delta_pitch=np.zeros(n),
        delta_yaw=np.full(n, 30.0 / TICK_RATE, np.float32).astype(np.float64),
        input_weapon=["weapon_ak47"] * n,
        team_num=np.full(n, team, np.int64),
    )


def media_record(table: TickTable, slot: int = 0, media_id: str = "m0") -> MediaRecord:
    """The media row of ``table``: a 32 fps recording as long as its ticks."""
    return MediaRecord(
        media_id=media_id,
        match_id=1,
        map_name="de_test",
        round=1,
        player_slot=slot,
        fps=32.0,
        source_frames=len(table) // 2,
        ticks_path=f"{media_id}.parquet",
        ticks_rows=len(table),
        ticks_file_size=1,
    )


def unscoped_signals(latent_frames: int) -> dict[str, np.ndarray]:
    """Observer signals of a player that is never flashed or scoped, labels known."""
    signals = {key: np.zeros(latent_frames, np.int64) for key in OBSERVER_SIGNAL_KEYS}
    signals["obs_flash_valid"][:] = 1
    signals["obs_scope_valid"][:] = 1
    return signals


def write_observer_labels(root: Path, media_id: str, source_frames: int, **curves) -> None:
    """Flash and scope label files of a recording of ``source_frames`` source frames, one sample
    per source frame: ``lum`` (default 0.5 everywhere, threshold 0.85), ``scoped_vis`` and ``level``
    (default 0)."""
    flash, scope = flash_label_path(root, media_id), scope_label_path(root, media_id)
    flash.parent.mkdir(parents=True, exist_ok=True)
    scope.parent.mkdir(parents=True, exist_ok=True)
    lum = np.asarray(curves.get("lum", np.full(source_frames, 0.5)), np.float16)
    np.savez(flash, lum=lum, hot_threshold=np.array(0.85, np.float32), stride=np.array(1, np.int16))
    zeros = np.zeros(source_frames)
    np.savez(
        scope,
        scoped_vis=np.asarray(curves.get("scoped_vis", zeros), np.uint8),
        level=np.asarray(curves.get("level", zeros), np.int8),
        corner_max=zeros.astype(np.float16),
        center=zeros.astype(np.float16),
        attack2=zeros.astype(np.uint8),
        weapon_id=zeros.astype(np.int16),
        ncov=np.array(source_frames, np.int32),
        thresholds=np.array([0.04, 0.08], np.float32),
    )
