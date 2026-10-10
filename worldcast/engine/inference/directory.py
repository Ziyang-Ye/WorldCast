"""The shared world state on a directory shared by the clients: the paper's evaluation setup."""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .world_state import PublishedBlock, StepRecord, WorldState, WorldStateError

__all__ = ["DirectoryWorldState"]

#: Prefix of a file while it is written; it matches neither ``blk_*.npy`` nor an exact path, so a
#: partial file is invisible.
_PARTIAL_PREFIX = ".pub-"


class DirectoryWorldState(WorldState):
    """The shared world state on one directory shared by the clients of a session, fresh per
    session::

        <media>/win_<window_start:06d>/blk_<f0:06d>.json   the published block, written first
        <media>/win_<window_start:06d>/blk_<f0:06d>.npy    its 4 latents, float32, written last
        <media>/steps/step_<t_target:06d>.json             the step record
        <media>/state/state_<t_target:06d>.json            the position (closed loop)
        <media>/DONE.json                                  the client publishes nothing more

    Every file goes to a ``.pub-*`` file in its directory, is fsynced and renamed into place, so a
    reader sees nothing or a whole file. Republishing different bytes at a key raises. A client
    writes its position for block ``t`` before its step record, so a client that has waited for the
    step record can read the position too.

    Args:
        root (str | Path): the directory.
        client (str): this client's media id (:class:`WorldState`).
        poll_s (float): the lockstep poll period, s.
    """

    def __init__(self, root: str | Path, *, client: str, poll_s: float) -> None:
        super().__init__(client=client, poll_s=poll_s)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # media -> {t_first: (block, npy)} as of the last refresh, plus this client's own
        self._index: dict[str, dict[int, tuple[PublishedBlock, Path]]] = {}
        self._seen: set[Path] = set()
        self._steps: dict[str, dict[int, StepRecord]] = {}
        self.refresh()

    def _block_paths(self, block: PublishedBlock) -> tuple[Path, Path]:
        window = self.root / block.media_id / f"win_{block.window_start:06d}"
        stem = window / f"blk_{block.f0:06d}"
        return stem.with_suffix(".npy"), stem.with_suffix(".json")

    def _step_path(self, media_id: str, t_target: int) -> Path:
        return self.root / media_id / "steps" / f"step_{int(t_target):06d}.json"

    def _position_path(self, media_id: str, t_target: int) -> Path:
        return self.root / media_id / "state" / f"state_{int(t_target):06d}.json"

    def _done_path(self, media_id: str) -> Path:
        return self.root / media_id / "DONE.json"

    @staticmethod
    def _write(path: Path, data: bytes | np.ndarray) -> None:
        """Write ``path`` atomically: a partial file beside it, fsync, rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, partial = tempfile.mkstemp(
            prefix=_PARTIAL_PREFIX, suffix=path.suffix, dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                if isinstance(data, np.ndarray):
                    np.save(handle, data)
                else:
                    handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, path)
        except BaseException:
            Path(partial).unlink(missing_ok=True)
            raise

    # -------------------------------------------------------------------------------- publishing
    def _store_block(self, block: PublishedBlock, latents: np.ndarray) -> None:
        """The record first, the latents last; republishing identical bytes is a no-op."""
        npy, record = self._block_paths(block)
        if npy.exists() and record.exists():
            held = str(json.loads(record.read_text()).get("latents_sha256"))
            if held == block.latents_sha256:
                return
            raise WorldStateError(
                f"{npy} already holds a DIFFERENT block (sha {held[:16]} vs"
                f" {block.latents_sha256[:16]}): one key, one block; use a fresh world-state"
                " directory per session"
            )
        self._write(record, json.dumps(asdict(block), sort_keys=True).encode())
        self._write(npy, latents)
        self._seen.add(npy)  # a client knows its own blocks without polling for them
        self._index.setdefault(self.client, {})[block.t_first] = (block, npy)

    def _store_step(self, record: StepRecord) -> None:
        """Publishing the same record again is a no-op."""
        body = json.dumps(asdict(record), sort_keys=True).encode()
        path = self._step_path(self.client, record.t_target)
        if path.exists():
            if path.read_bytes() == body:
                return
            raise WorldStateError(
                f"{path} already holds a DIFFERENT step record: one client, one record per block"
            )
        self._write(path, body)

    def _store_position(self, t_target: int, latent_frames: list[int], xyz: np.ndarray) -> None:
        body = json.dumps({"latent_frames": latent_frames, "xyz": xyz.tolist()}).encode()
        self._write(self._position_path(self.client, t_target), body)

    def _store_done(self, status: str, note: str) -> None:
        body = dict(media_id=self.client, status=status, note=note)
        self._write(self._done_path(self.client), json.dumps(body, sort_keys=True).encode())

    # ----------------------------------------------------------------------------------- reading
    def refresh(self) -> None:
        """Index the blocks whose record and latents have landed since the last call."""
        for media_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            index = self._index.setdefault(media_dir.name, {})
            for npy in sorted(media_dir.glob("win_*/blk_*.npy")):
                record = npy.with_suffix(".json")
                if npy in self._seen or not record.exists():
                    continue  # the record lands first: an .npy without one is not published
                block = PublishedBlock(**json.loads(record.read_text()))
                self._seen.add(npy)
                index[block.t_first] = (block, npy)

    def published_blocks(self, media_id: str) -> list[PublishedBlock]:
        """The indexed blocks of ``media_id``, by ``t_first``."""
        index = self._index.get(str(media_id)) or {}
        return [index[t][0] for t in sorted(index)]

    def read_steps(self, media_id: str, *, upto: int) -> list[StepRecord]:
        """The step files of ``media_id`` up to ``upto``; a file is read once."""
        cache = self._steps.setdefault(str(media_id), {})
        for path in sorted((self.root / str(media_id) / "steps").glob("step_*.json")):
            t = int(path.stem.split("_", 1)[1])
            if t <= upto and t not in cache:
                record = json.loads(path.read_text())
                cache[t] = StepRecord(
                    media_id=str(record["media_id"]),
                    t_target=int(record["t_target"]),
                    withdrawn=tuple(int(v) for v in record["withdrawn"]),
                    resident=tuple(int(v) for v in record["resident"]),
                )
        return [cache[t] for t in sorted(cache) if t <= upto]

    def has_step(self, media_id: str, t_target: int) -> bool:
        """Whether the step file of ``t_target`` exists."""
        return self._step_path(media_id, t_target).exists()

    def read_position(self, media_id: str, t_target: int) -> tuple[list[int], np.ndarray] | None:
        """The position file of ``t_target``, or None while it does not exist."""
        path = self._position_path(media_id, t_target)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        latent_frames = [int(k) for k in record["latent_frames"]]
        return latent_frames, np.asarray(record["xyz"], np.float64).reshape(-1, 3)

    def done_status(self, media_id: str) -> str | None:
        """The status in ``DONE.json``, or None while the file does not exist."""
        path = self._done_path(media_id)
        return str(json.loads(path.read_text())["status"]) if path.exists() else None

    def _read_block(self, block: PublishedBlock) -> np.ndarray:
        return np.load(self._index[block.media_id][block.t_first][1])
