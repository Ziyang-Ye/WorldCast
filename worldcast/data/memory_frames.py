"""Memory frames (Sec. 3.3): a block of a player's window and what a block's window reads of it.

The memory frames of a block's window are the four latent frames of one block, the client's own or
another player's (:class:`MemoryBlock`). Besides their latents and cameras it reads that player's
recorded states and controls at the block's 16 video frames and the observer signals and fields of
view of its four latent frames (:func:`memory_frame_inputs`). The client and the training windows
build both here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .camera import c2w_from_state_rows, camera_tans
from .labels import load_observer_signals
from .latents import BLOCK, block_span, last_video_frames, video_frames_of
from .recordings import MediaRecord, TickTable
from .window import PlayerFrames, WindowSpec, frame_endpoint_rows

__all__ = [
    "MEMORY_FRAME_FIELDS",
    "MEMORY_FRAMES_PREFIX",
    "MemoryBlock",
    "MemoryFrames",
    "batch_keys",
    "block_memory_frames",
    "memory_frame_inputs",
    "recorded_block",
]

#: Prefix of the batch keys of a window's memory frames (``memory_frames_latents``, ...).
MEMORY_FRAMES_PREFIX = "memory_frames_"
#: The :class:`PlayerFrames` fields a window reads at the memory frames' 16 video frames.
MEMORY_FRAME_FIELDS = (
    "states",
    "buttons",
    "view_deltas",
    "weapon_ids",
    "control_substeps",
    "control_substep_valid",
)


@dataclass(frozen=True)
class MemoryBlock:
    """One block of a player's window that can serve as memory frames.

    Attributes:
        media_id (str): the player's media id.
        slot (int): its player slot.
        window_start (int): the ``start_frame`` of the player's window; with ``f0`` the block's
            address, as the block is published.
        f0 (int): the block's first latent frame in that window.
        t_first (int): source frame of its first latent frame
            (:func:`~worldcast.data.latents.block_span`).
        t_last (int): source frame of its last latent frame.
        c2w (torch.Tensor): ``[4, 4, 4]`` float32 cameras of its latent frames.
    """

    media_id: str
    slot: int
    window_start: int
    f0: int
    t_first: int
    t_last: int
    c2w: torch.Tensor

    @classmethod
    def at(
        cls, media_id: str, slot: int, window_start: int, f0: int, c2w: torch.Tensor
    ) -> "MemoryBlock":
        """The block at the address ``(window_start, f0)``, with its source-frame span."""
        t_first, t_last = block_span(window_start, f0)
        return cls(str(media_id), int(slot), int(window_start), int(f0), t_first, t_last, c2w)


def recorded_block(
    slot: int,
    media: MediaRecord,
    table: TickTable,
    start_frame: int,
    f0: int,
    spec: WindowSpec,
    *,
    endpoints: tuple[np.ndarray, int] | None = None,
) -> MemoryBlock | None:
    """A block of a player's window, keyed at the player's recorded cameras.

    Args:
        slot (int): the player's slot.
        media (MediaRecord): the player's media row.
        table (TickTable): the player's tick table.
        start_frame (int): the window's first source frame.
        f0 (int): the block's first latent frame in the window.
        spec (WindowSpec): the window (its video-frame count bounds the covered cut).
        endpoints (tuple[np.ndarray, int] | None): :func:`~worldcast.data.window.
            frame_endpoint_rows` of the window, when the caller has them.

    Returns:
        MemoryBlock | None: the block with the cameras at its four latent frames' last video frames,
        or None when the recording does not cover the block or the player is dead in it.
    """
    start_frame, f0 = int(start_frame), int(f0)
    rows, covered = endpoints or frame_endpoint_rows(table.t, float(media.fps), start_frame, spec)
    last = last_video_frames(f0, BLOCK)
    if last[-1] >= covered:
        return None
    if not table.is_alive[rows[last]].all():
        return None
    states = table.states(rows[last])
    return MemoryBlock.at(media.media_id, slot, start_frame, f0, c2w_from_state_rows(states))


def memory_frame_inputs(
    frames: PlayerFrames, signals: Mapping[str, np.ndarray], f0: int
) -> dict[str, torch.Tensor]:
    """What a window reads of a memory block besides its latents and cameras.

    Args:
        frames (PlayerFrames): the block's player over its window.
        signals (Mapping[str, np.ndarray]): that player's per-latent observer signals.
        f0 (int): the block's first latent frame in the window (``>= 1``).

    Returns:
        dict[str, torch.Tensor]: the player at the 16 video frames of the block (``states`` ``[16,
        6]``, ``buttons`` ``[16, 11]``, ``view_deltas`` ``[16, 2]`` float32, ``weapon_ids``
        ``[16]`` int64, ``control_substeps`` ``[16, 4, 13]`` float32, ``control_substep_valid``
        ``[16, 4]`` bool), the observer signals ``obs_*`` ``[4]`` int64 of its four latent frames
        and ``tans`` ``[4, 2]`` float32 (``tan(hfov / 2), tan(vfov / 2)`` per latent frame, narrowed
        while the player is scoped).
    """
    rows = video_frames_of(f0, BLOCK)
    out = {key: torch.from_numpy(getattr(frames, key)[rows]) for key in MEMORY_FRAME_FIELDS}
    for key, value in signals.items():
        out[key] = torch.from_numpy(np.asarray(value)[f0 : f0 + BLOCK])
    out["tans"] = torch.from_numpy(camera_tans(signals, frames.weapon_ids, range(f0, f0 + BLOCK)))
    return out


def block_memory_frames(
    block: MemoryBlock,
    latents: torch.Tensor | np.ndarray,
    frames: PlayerFrames,
    signals: Mapping[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """The memory frames a window reads from ``block``.

    Args:
        block (MemoryBlock): the block.
        latents (torch.Tensor | np.ndarray): its latents ``[4, 48, 24, 42]``.
        frames (PlayerFrames): the block's player over its window.
        signals (Mapping[str, np.ndarray]): that player's per-latent observer signals.

    Returns:
        dict[str, torch.Tensor]: ``latents`` ``[4, 48, 24, 42]`` float32, ``c2w`` ``[4, 4, 4]``
        float32 and :func:`memory_frame_inputs`.
    """
    latents = torch.as_tensor(latents).float()
    if tuple(latents.shape[:1]) != (BLOCK,):
        raise RuntimeError(f"memory frames are {BLOCK} latent frames, got {tuple(latents.shape)}")
    return {
        "latents": latents,
        "c2w": torch.as_tensor(block.c2w).float(),
        **memory_frame_inputs(frames, signals, block.f0),
    }


def batch_keys(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The tensors of :func:`block_memory_frames` under their batch keys,
    ``memory_frames_<name>``."""
    return {MEMORY_FRAMES_PREFIX + name: value for name, value in tensors.items()}


class MemoryFrames:
    """The memory frames of one client's windows: a retrieved block's inputs.

    The client's own blocks read its own frames and signals; another player's frames are built once
    per window of that player and cached.

    Args:
        client_media (str): the client's media id.
        client_frames (PlayerFrames): the client's frames of its window.
        client_signals (Mapping[str, np.ndarray]): the client's observer signals (``[N]`` each).
        round_slots (Mapping[int, MediaRecord]): the round's media rows by slot.
        tick_tables (Mapping[int, TickTable]): the round's tick tables by slot.
        spec (WindowSpec): the client's window (another player's frames are built for the same N
            from that player's ``start_frame``).
        observer_signal_label_root (str | Path): holds ``flashlabels/`` and ``scopelabels/``.
    """

    def __init__(
        self,
        *,
        client_media: str,
        client_frames: PlayerFrames,
        client_signals: Mapping[str, np.ndarray],
        round_slots: Mapping[int, MediaRecord],
        tick_tables: Mapping[int, TickTable],
        spec: WindowSpec,
        observer_signal_label_root: str | Path,
    ) -> None:
        self.client_media = str(client_media)
        self.client_frames = client_frames
        self.client_signals = client_signals
        self.round_slots = dict(round_slots)
        self.tick_tables = dict(tick_tables)
        self.spec = spec
        self.observer_signal_label_root = observer_signal_label_root
        self._players: dict[tuple[int, int], tuple[PlayerFrames, dict[str, np.ndarray]]] = {}

    def _player(self, slot: int, start_frame: int) -> tuple[PlayerFrames, dict[str, np.ndarray]]:
        key = (slot, start_frame)
        if key not in self._players:
            media = self.round_slots[slot]
            frames = PlayerFrames.from_ticks(self.tick_tables[slot], media, start_frame, self.spec)
            signals = load_observer_signals(
                self.observer_signal_label_root,
                media.media_id,
                source_frames=media.source_frames,
                start_frame=start_frame,
                latent_frames=int(self.spec.latent_frames),
            )
            self._players[key] = frames, signals
        return self._players[key]

    def __call__(self, block: MemoryBlock, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        """The memory frames' tensors for ``block``.

        Args:
            block (MemoryBlock): the retrieved block.
            latents (torch.Tensor): the block's latents ``[4, 48, 24, 42]``, from the client's own
                store (own block) or as its player published them.

        Returns:
            dict[str, torch.Tensor]: :func:`block_memory_frames`.
        """
        if block.media_id == self.client_media:
            frames, signals = self.client_frames, self.client_signals
        else:
            frames, signals = self._player(block.slot, block.window_start)
        return block_memory_frames(block, latents, frames, signals)
