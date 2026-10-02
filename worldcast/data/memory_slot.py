"""Published blocks as memory candidates, and the frames of the memory slot's block (Sec. 3.3).

:func:`published_block_candidate` is the ``candidate_at`` of
:class:`worldcast.engine.inference.pool.PeerBlocks`. :class:`MemorySlotFrames` gives the window what
it needs about the retrieved block besides its latents: its player's recorded rows of the block's 16
pixel frames, the observer signals, cameras and fields of view of its four latent frames.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from worldcast.player_state.projection import (
    EYE_HEIGHT,
    HFOV_DEGREES,
    c2w_from_state_rows,
    camera_tans,
)

from .actions import OPENCS2_WEAPONS
from .media import MediaRecord
from .obs_signals import ObsSignals, load_obs_signals
from .player_frames import PlayerFrames, WindowSpec, frame_endpoint_rows
from .ticks import TickTable

__all__ = ["BLOCK", "ROWS_PER_LATENT", "published_block_candidate", "MemorySlotFrames"]

#: Latent frames per block (one memory entry, one memory slot).
BLOCK = 4
#: Pixel frames per latent frame.
ROWS_PER_LATENT = 4


def published_block_candidate(
    *,
    slot: int,
    media: MediaRecord,
    table: TickTable,
    window_start: int,
    f0: int,
    spec: WindowSpec,
    eye_height: float = EYE_HEIGHT,
) -> tuple[dict | None, str]:
    """A peer's published block as a memory candidate, keyed at the peer's recorded cameras.

    Args:
        slot (int): the peer's player slot.
        media (MediaRecord): the peer's media row.
        table (TickTable): the peer's tick table.
        window_start (int): the peer's rollout start (source frame, 32 fps).
        f0 (int): the block's first latent frame in the peer's rollout.
        spec (WindowSpec): the reader's window (its pixel-frame count bounds the covered cut).
        eye_height (float): camera height above the feet, u.

    Returns:
        tuple[dict | None, str]: ``(block, "")`` with ``media_id, slot, window_start, f0,
        orig_first, orig_last`` (source frames of the first and last latent frame) and ``c2w``
        ``[4, 4, 4]`` float32 (the cameras at the four latent frames' last pixel frames), or
        ``(None, "uncovered" | "dead")`` when the recording does not cover the block or the peer is
        dead.
    """
    window_start, f0 = int(window_start), int(f0)
    stride = int(spec.skip_frame) * ROWS_PER_LATENT
    rows, covered = frame_endpoint_rows(table.t, float(media.fps), window_start, spec)
    if ROWS_PER_LATENT * (f0 + BLOCK - 1) >= covered:
        return None, "uncovered"
    tick_rows = [int(rows[ROWS_PER_LATENT * (f0 + j)]) for j in range(BLOCK)]
    if any(not table.is_alive[r] for r in tick_rows):
        return None, "dead"
    pose = np.stack([table.x, table.y, table.z, table.yaw, table.pitch], axis=-1)
    state_rows = np.concatenate(
        [pose.astype(np.float32)[tick_rows], np.ones((BLOCK, 1), np.float32)], axis=1
    )
    block = dict(
        media_id=str(media.media_id),
        slot=int(slot),
        window_start=window_start,
        f0=f0,
        orig_first=window_start + stride * f0,
        orig_last=window_start + stride * (f0 + BLOCK - 1),
        c2w=c2w_from_state_rows(state_rows, eye_height=eye_height),
    )
    return block, ""


class MemorySlotFrames:
    """The frames of the block in the memory slot, for one client (peers' frames are cached).

    Args:
        own_media (str): the client's media id; its own blocks use ``own_frames`` and ``own_obs``.
        own_frames (PlayerFrames): the client's frames of its window.
        own_obs (ObsSignals): the client's per-latent observer signals (``[N]`` each).
        round_slots (Mapping[int, MediaRecord]): the round's media rows by slot.
        tick_tables (Mapping[int, TickTable]): the round's tick tables by slot.
        spec (WindowSpec): the client's window (a peer's frames are built for the same N from the
            peer's ``window_start``).
        obs_signal_label_root (str | Path): holds ``flashlabels/`` and ``scopelabels/``.
        hfov_degrees (float): unscoped horizontal field of view of every camera.
    """

    def __init__(
        self,
        *,
        own_media: str,
        own_frames: PlayerFrames,
        own_obs: ObsSignals,
        round_slots: Mapping[int, MediaRecord],
        tick_tables: Mapping[int, TickTable],
        spec: WindowSpec,
        obs_signal_label_root: str | Path,
        hfov_degrees: float = HFOV_DEGREES,
    ) -> None:
        self.own_media = str(own_media)
        self.own_frames = own_frames
        self.own_obs = own_obs
        self.round_slots = dict(round_slots)
        self.tick_tables = dict(tick_tables)
        self.spec = spec
        self.obs_signal_label_root = obs_signal_label_root
        self.hfov_degrees = float(hfov_degrees)
        self._peers: dict[tuple[int, int], tuple[PlayerFrames, ObsSignals]] = {}

    def _peer(self, slot: int, window_start: int) -> tuple[PlayerFrames, ObsSignals]:
        key = (slot, window_start)
        if key not in self._peers:
            media = self.round_slots[slot]
            frames = PlayerFrames.from_ticks(self.tick_tables[slot], media, window_start, self.spec)
            obs = load_obs_signals(
                self.obs_signal_label_root,
                media.media_id,
                video_frames=int(media.video_frames),
                start_frame=window_start,
                latent_frames=int(self.spec.latent_frames),
                skip_frame=int(self.spec.skip_frame),
            )
            self._peers[key] = frames, obs
        return self._peers[key]

    def __call__(self, block: Mapping, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        """The memory slot's tensors for ``block``.

        Args:
            block (Mapping): an entry of :attr:`worldcast.engine.inference.pool.PeerBlocks.blocks`
                (``media_id``, ``slot``, ``window_start``, ``f0``, ``c2w`` ``[4, 4, 4]``).
            latents (torch.Tensor): the block's latents ``[4, 48, 24, 42]``, from the client's own
                store (own block) or as the peer published them.

        Returns:
            dict[str, torch.Tensor]: keyed by ``wp_slot_`` suffix: ``latents`` ``[4, C, H, W]``
            float32, ``c2w`` ``[4, 4, 4]`` float32, ``states`` ``[16, 6]``, ``camera_abs`` ``[16,
            2]``, ``buttons`` ``[16, B]``, ``camera_quantized`` ``[16, 2]`` float32, ``weapon_ids``
            ``[16]`` int64, ``action_substeps`` ``[16, 4, B + 2]`` float32, ``action_substep_valid``
            ``[16, 4]`` bool, ``obs_<key>`` ``[4]`` (observer signals of the four latent frames) and
            ``tans`` ``[4, 2]`` float32 (``tan(hfov/2), tan(vfov/2)`` per latent frame, narrowed
            while the player is scoped).
        """
        f0 = int(block["f0"])
        if str(block["media_id"]) == self.own_media:
            frames, obs = self.own_frames, self.own_obs
        else:
            frames, obs = self._peer(int(block["slot"]), int(block["window_start"]))
        lat = torch.as_tensor(latents).float()
        if tuple(lat.shape[:1]) != (BLOCK,):
            raise RuntimeError(f"a memory slot holds {BLOCK} latents, got {tuple(lat.shape)}")
        # the 16 pixel frames of latent frames f0 .. f0 + 3 (f0 >= 1: latent 0 is the sink)
        rows = list(range(ROWS_PER_LATENT * f0 - 3, ROWS_PER_LATENT * (f0 + BLOCK - 1) + 1))
        out = {
            "latents": lat,
            "c2w": torch.as_tensor(block["c2w"]).float(),
            "states": torch.from_numpy(frames.states[rows]),
            "camera_abs": torch.from_numpy(frames.camera_abs[rows]),
            "buttons": torch.from_numpy(frames.buttons[rows]),
            "camera_quantized": torch.from_numpy(frames.camera_quantized[rows]),
            "weapon_ids": torch.from_numpy(frames.weapon_ids[rows]),
            "action_substeps": torch.from_numpy(frames.action_substeps[rows]),
            "action_substep_valid": torch.from_numpy(frames.action_substep_valid[rows]),
        }
        signals = obs.as_dict()
        for key, value in signals.items():
            out["obs_" + key] = torch.from_numpy(np.asarray(value)[f0 : f0 + BLOCK])
        out["tans"] = torch.from_numpy(
            camera_tans(
                signals,
                frames.weapon_ids,
                range(f0, f0 + BLOCK),
                self.hfov_degrees,
                weapon_names=OPENCS2_WEAPONS,
            )
        )
        return out
