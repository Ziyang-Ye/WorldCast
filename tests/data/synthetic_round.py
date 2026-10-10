"""A synthetic recorded round for the data tests: three players standing in a box room.

The client (slot 0) and its teammate (slot 1) are on team 2, an enemy (slot 5) on team 3. Each
stands still with its eye at height 0 and looks along +x, except for a time in which it looks
along -x. The round is ten seconds: one cached window of 41 latent frames per player. The recorded
frames are rendered from a texture on the room's walls, so two players that look at the same
surface see the same picture.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from tests.data.support import TICKS_PER_SECOND, write_observer_labels
from worldcast.data import recordings, window
from worldcast.data.camera import c2w_from_state_rows, half_angle_tangents
from worldcast.data.controls import count_until
from worldcast.data.game import EYE_HEIGHT
from worldcast.data.latents import FRAME_SIZE, window_key
from worldcast.data.map_mesh import MapMesh, MeshLibrary
from worldcast.data.recordings import MediaIndex, MediaRecord, RoundIndexRow, TickTable
from worldcast.data.window import DataPaths

#: Half the room's side along x and y, and along z, u: walls at +-400, floor and ceiling at +-200.
HALF_SIDE, HALF_HEIGHT = 400.0, 200.0
SECONDS = 10.5
SOURCE_FRAMES = int(SECONDS * 32)
CLIENT, TEAMMATE, ENEMY = "1-de_test-r01-p00", "1-de_test-r01-p01", "1-de_test-r01-p05"
#: Per player: slot, team, eye position ``(x, y)`` and the seconds ``(from, to]`` it looks along -x.
PLAYERS = {
    CLIENT: (0, 2, (0.0, 0.0), (7.03, SECONDS)),
    TEAMMATE: (1, 2, (-100.0, 0.0), (3.03, 4.0)),
    ENEMY: (5, 3, (-200.0, 100.0), (0.0, SECONDS)),
}


def tick_table(media_id: str, *, flashed: bool = False) -> TickTable:
    """The ticks of one of the round's players."""
    _, team, (x, y), (turn_from, turn_to) = PLAYERS[media_id]
    n = int(SECONDS * TICKS_PER_SECOND)
    t = np.arange(n, dtype=np.float64) / TICKS_PER_SECOND
    return TickTable(
        t=t,
        x=np.full(n, x),
        y=np.full(n, y),
        z=np.full(n, -EYE_HEIGHT),
        yaw=np.where((t > turn_from) & (t <= turn_to), 180.0, 0.0),
        pitch=np.zeros(n),
        is_alive=np.ones(n, dtype=bool),
        active=[[] for _ in range(n)],
        delta_pitch=np.zeros(n),
        delta_yaw=np.zeros(n),
        input_weapon=["weapon_ak47"] * n,
        team_num=np.full(n, team, np.int64),
    )


def media_record(media_id: str) -> MediaRecord:
    return MediaRecord(
        media_id=media_id,
        match_id=1,
        map_name="de_test",
        round=1,
        player_slot=PLAYERS[media_id][0],
        fps=32.0,
        source_frames=SOURCE_FRAMES,
        ticks_path=f"{media_id}/ticks.parquet",
        ticks_rows=int(SECONDS * TICKS_PER_SECOND),
        ticks_file_size=0,
        video_path=f"{media_id}/video.mp4",
    )


def room_mesh() -> MapMesh:
    """The room as a collision mesh."""
    import trimesh

    return MapMesh(trimesh.creation.box(extents=(2 * HALF_SIDE, 2 * HALF_SIDE, 2 * HALF_HEIGHT)))


def render(c2w: np.ndarray, tans: tuple[float, float]) -> np.ndarray:
    """The room seen from the camera ``c2w`` ``[4, 4]``: ``[384, 672, 3]`` uint8, a smooth texture
    of the wall point each pixel centre looks at."""
    h, w = FRAME_SIZE
    u, v = np.meshgrid((np.arange(w) + 0.5) / w * 2 - 1, (np.arange(h) + 0.5) / h * 2 - 1)
    rays = np.stack([u * tans[0], v * tans[1], np.ones_like(u)], axis=-1) @ c2w[:3, :3].T
    eye = c2w[:3, 3]
    half = np.array([HALF_SIDE, HALF_SIDE, HALF_HEIGHT])
    with np.errstate(divide="ignore"):
        exits = np.where(rays > 0, (half - eye) / rays, (-half - eye) / rays)
    p = eye + exits.min(-1, keepdims=True) * rays
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    grey = 0.5 + 0.2 * np.sin(x / 7 + y / 5 + z / 6) + 0.2 * np.cos(x / 9 - y / 4 + z / 8)
    tint = np.stack([grey, grey * 0.9 + 0.05, grey * 0.8 + 0.1], axis=-1)
    return np.clip(255 * tint, 0, 255).astype(np.uint8)


@dataclass
class SyntheticRound:
    """The round's data in memory and on disk.

    Attributes:
        tables (dict[str, TickTable]): the players' ticks by media id.
        media_index (MediaIndex): their media rows.
        paths (DataPaths): the label files and the latent cache under the round's directory.
        meshes (MeshLibrary): the room.
    """

    tables: dict[str, TickTable]
    media_index: MediaIndex
    paths: DataPaths
    meshes: MeshLibrary

    def row(self, media_id: str = CLIENT, start_frame: int = 0) -> RoundIndexRow:
        """The round-index row of a player's window."""
        return RoundIndexRow(
            media_id=media_id,
            start_frame=start_frame,
            match_id=1,
            round=1,
            map_name="de_test",
            player_slot=PLAYERS[media_id][0],
        )

    def read_ticks(self, media: MediaRecord, dataset_root: Path, **_) -> TickTable:
        """Stands in for :func:`worldcast.data.recordings.read_player_ticks`."""
        return self.tables[media.media_id]

    def frame(self, media_id: str, source_frame: int) -> np.ndarray:
        """The recorded frame of a player at a source frame."""
        table = self.tables[media_id]
        row = int(count_until(table.t, source_frame / 32.0)) - 1
        c2w = c2w_from_state_rows(table.states([row]))[0].numpy().astype(np.float64)
        return render(c2w, half_angle_tangents())

    def recorded_frames(self) -> type:
        """Stands in for :class:`worldcast.data.video.VideoFrames`."""
        this = self

        class RecordedFrames:
            def __init__(self, dataset_root, media) -> None:
                self.frames: dict[tuple[str, int], np.ndarray] = {}

            def __enter__(self):
                return self

            def __exit__(self, *exc) -> None:
                self.frames.clear()

            def get(self, media_id: str, frame: int) -> np.ndarray:
                key = (str(media_id), int(frame))
                if key not in self.frames:
                    self.frames[key] = this.frame(*key)
                return self.frames[key]

        return RecordedFrames


def write_round(root: Path, *, flashed_teammate_frames: tuple[int, ...] = ()) -> SyntheticRound:
    """Write the round's label files and latent cache under ``root``.

    Args:
        root (Path): an empty directory.
        flashed_teammate_frames (tuple[int, ...]): source frames at which the teammate is
            flash-white.
    """
    tables = {media_id: tick_table(media_id) for media_id in PLAYERS}
    records = {media_id: media_record(media_id) for media_id in PLAYERS}
    paths = DataPaths(
        dataset_root=root / "recordings",
        media_index=root / "media_index.jsonl",
        latent_cache_root=root / "latent_cache",
        visibility_label_root=root / "visibility",
        observer_signal_label_root=root / "observer_signals",
    )
    for directory in (paths.latent_cache_root, paths.visibility_label_root):
        directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for media_id in PLAYERS:
        lum = np.full(SOURCE_FRAMES, 0.5)
        if media_id == TEAMMATE:
            lum[list(flashed_teammate_frames)] = 0.95
        write_observer_labels(paths.observer_signal_label_root, media_id, SOURCE_FRAMES, lum=lum)
        latents = rng.standard_normal((1, 41, 48, 24, 42)).astype(np.float16)
        np.savez(paths.latent_cache_root / f"{media_id}.npz", **{window_key(0): latents})
        visible = np.zeros((10, SOURCE_FRAMES), bool)
        np.savez(
            paths.visibility_label_root / f"{media_id}.npz",
            _binary_visible=visible,
            _binary_eval_valid=~visible,
        )
    return SyntheticRound(
        tables=tables,
        media_index=MediaIndex(records),
        paths=paths,
        meshes=MeshLibrary({"de_test": room_mesh()}),
    )


def patch_readers(monkeypatch, synthetic: SyntheticRound) -> None:
    """Serve the round's ticks and frames from memory: the tick parquets and the videos are the
    only files of a round that the tests do not write."""
    from worldcast.data import memory_selection

    monkeypatch.setattr(window, "read_player_ticks", synthetic.read_ticks)
    monkeypatch.setattr(recordings, "read_player_ticks", synthetic.read_ticks)
    monkeypatch.setattr(memory_selection, "VideoFrames", synthetic.recorded_frames())


__all__ = [
    "CLIENT",
    "ENEMY",
    "PLAYERS",
    "TEAMMATE",
    "SyntheticRound",
    "patch_readers",
    "render",
    "replace",
    "write_round",
]
