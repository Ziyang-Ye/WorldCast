"""Closed-loop inference (Sec. 3.4; App. "State model in detail", Complementary filter and Within
a block).

Each client reads its own position off its new latent frames with the state model and fuses it with
the complementary filter of Eq. (4). It publishes the position once per block and places the other
clients where they published themselves, extrapolated over the block from their controls
(:mod:`worldcast.player_state.extrapolation`).
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol

import numpy as np
import torch

from worldcast.data.camera import c2w_from_state_rows
from worldcast.data.controls import state_model_controls
from worldcast.data.latents import (
    BLOCK,
    SOURCE_FRAMES_PER_LATENT,
    last_video_frames,
    source_frame,
    video_frames_of,
)
from worldcast.data.memory_frames import MemoryBlock
from worldcast.data.recordings import ALIVE_INDEX, TickTable
from worldcast.modeling.depth_head import DepthFn
from worldcast.modeling.state_model import (
    DISPLACEMENT_UNIT_U,
    STATE_MODEL_WINDOW_LATENT_FRAMES,
    STATE_MODEL_WINDOW_STRIDE,
    StateModel,
    map_index,
)

from .extrapolation import (
    ClientCameras,
    PhysicsPrior,
    PositionTrack,
    PredictedStateTable,
    player_motion,
)
from .predicted_visibility import PredictedVisibility

__all__ = [
    "FUSION_WEIGHT",
    "ClosedLoop",
    "ComplementaryFilter",
    "PositionExchange",
    "StateReader",
    "window_of",
]

#: Weight of the place estimate in Eq. (4).
FUSION_WEIGHT = 0.5


# -------------------------------------------------------------------------- state model and Eq. (4)
def window_of(latent_frame: int) -> tuple[int, int]:
    """``(window, position)`` of a latent frame in the state model's windows: latent frame 0 is
    read from window 0, latent frame ``40w + j`` (j in 1..40) from window ``w`` at position ``j``.
    """
    f = int(latent_frame)
    if f < 0:
        raise ValueError(f"latent frame {f}")
    if f == 0:
        return 0, 0
    w = (f - 1) // STATE_MODEL_WINDOW_STRIDE
    return w, f - STATE_MODEL_WINDOW_STRIDE * w


class StateReader:
    """The state model in the block loop.

    Window ``w`` covers the client's latent frames ``40w .. 40w + 40``; a window still being
    generated is read with its future latent frames zeroed (every part of the model is causal).

    Args:
        model (StateModel): the state model.
        ticks (TickTable): the client's recorded tick stream, read without jump recall
            (``read_player_ticks(..., jump_recall=False)``), as the state model was trained.
        map_name (str): the round's map, one of :data:`~worldcast.modeling.state_model.MAPS`
            (:func:`~worldcast.modeling.state_model.map_index` refuses another).
        source_fps (float): source frames per second of the recording.
        start_frame (int): source frame of latent frame 0.
        device (torch.device | str): where the model runs.
    """

    def __init__(
        self,
        model: StateModel,
        ticks: TickTable,
        *,
        map_name: str,
        source_fps: float,
        start_frame: int,
        device: torch.device | str,
    ) -> None:
        self.model, self.ticks, self.source_fps = model, ticks, float(source_fps)
        self.start_frame = int(start_frame)
        self.device = torch.device(device)
        self.map_id = torch.tensor([map_index(map_name)], device=self.device)
        self._controls: dict[int, torch.Tensor] = {}

    def _controls_of(self, w: int) -> torch.Tensor:
        if w not in self._controls:
            controls = state_model_controls(
                timestamps=self.ticks.t,
                held_buttons=self.ticks.active,
                delta_pitch=self.ticks.delta_pitch,
                delta_yaw=self.ticks.delta_yaw,
                start_frame=source_frame(self.start_frame, STATE_MODEL_WINDOW_STRIDE * w),
                source_fps=self.source_fps,
                latent_frames=STATE_MODEL_WINDOW_LATENT_FRAMES,
            )
            self._controls[w] = torch.from_numpy(controls)[None].to(self.device).float()
        return self._controls[w]

    def _window_input(self, latents: torch.Tensor, first: int, last: int) -> np.ndarray:
        """The model's input ``[41, 48, 24, 42]`` fp16 for the window that starts at latent frame
        ``first``: its frames up to position ``last`` from ``latents``, the later ones zero."""
        if first + last >= int(latents.shape[0]):
            raise ValueError(
                f"latent frame {first + last} requested with {int(latents.shape[0])} generated"
            )
        # fp16, as the state model was trained on the fp16 latent cache
        v = np.zeros((STATE_MODEL_WINDOW_LATENT_FRAMES,) + tuple(latents.shape[1:]), np.float16)
        v[: last + 1] = np.asarray(latents[first : first + last + 1], dtype=np.float32).astype(
            np.float16
        )
        return v

    @torch.no_grad()
    def read(
        self, latents: torch.Tensor, latent_frames: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read the state model at ``latent_frames``.

        Args:
            latents (torch.Tensor): ``[N, 48, 24, 42]`` generated latent frames (CPU), ``N`` > every
                latent frame read.
            latent_frames (Sequence[int]): latent frames to read.

        Returns:
            tuple[np.ndarray, np.ndarray]: the place-head estimates ``A_f`` ``[n, 3]`` float64 (u)
            and the motion-head displacements ``Delta_f`` ``[n, 3]`` float32 (units of 64 u).
        """
        latent_frames = [int(f) for f in latent_frames]
        place = np.zeros((len(latent_frames), 3), np.float64)
        displacement = np.zeros((len(latent_frames), 3), np.float32)
        by_window: dict[int, list[tuple[int, int]]] = {}
        for i, f in enumerate(latent_frames):
            w, j = window_of(f)
            by_window.setdefault(w, []).append((i, j))
        for w, items in sorted(by_window.items()):
            last = max(j for _, j in items)
            v = self._window_input(latents, STATE_MODEL_WINDOW_STRIDE * w, last)
            output = self.model(
                torch.from_numpy(v)[None].to(self.device).float(), self._controls_of(w), self.map_id
            )
            window_place = output.place[0].float().cpu().numpy()
            window_displacement = output.displacement[0].float().cpu().numpy()
            for i, j in items:
                place[i] = window_place[j].astype(np.float64)
                displacement[i] = window_displacement[j]
        return place, displacement


class ComplementaryFilter:
    """The complementary filter of Eq. (4), latent frame by latent frame.

    ``p_f = 1/2 (p_{f-1} + 64 Delta_f) + 1/2 (A_f - A_0 + p_0)``: the motion head's displacement
    ``Delta_f`` (in units of 64 u) advances the estimate, and the place head's estimate ``A_f``
    corrects it, entering as a displacement from its first value ``A_0`` (App. "State model in
    detail", Complementary filter). Positions are float64, u.

    Args:
        p0 (np.ndarray): ``[3]`` the given initial position ``p_0`` (the round start), u.
    """

    def __init__(self, p0: np.ndarray) -> None:
        self.p0 = np.asarray(p0, np.float64).reshape(3)
        self.place0 = None
        self.p = None
        self.frame = -1

    def step(self, f: int, place: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        """The estimate ``p_f`` of the next latent frame ``f``.

        Args:
            f (int): the latent frame; frames are fed in order from 0.
            place (np.ndarray): ``[3]`` the place-head estimate ``A_f``, u.
            displacement (np.ndarray): ``[3]`` the motion-head displacement ``Delta_f``, units of
                64 u (unused at ``f = 0``).

        Returns:
            np.ndarray: ``[3]`` float64 ``p_f``, u.
        """
        f = int(f)
        if f != self.frame + 1:
            raise ValueError(f"latent frame {f} after {self.frame}: the filter runs frame by frame")
        place = np.asarray(place, np.float64).reshape(3)
        if f == 0:
            self.place0, self.p = place, self.p0.copy()
        else:
            q = self.p + np.asarray(displacement, np.float64).reshape(3) * DISPLACEMENT_UNIT_U
            self.p = q - FUSION_WEIGHT * (q - (place - self.place0 + self.p0))
        self.frame = f
        return self.p.copy()


# -------------------------------------------------------------------------------------- closed loop
class PositionExchange(Protocol):
    """Where the clients publish their estimated positions, once per block: the shared world state
    (:class:`worldcast.engine.inference.world_state.WorldState`). A block is named by
    ``t_target``, the source frame of its first latent frame."""

    def publish_position(
        self, t_target: int, latent_frames: Sequence[int], xyz: np.ndarray
    ) -> None:
        """Publish this client's positions ``xyz [n, 3]`` at ``latent_frames`` for its block at
        ``t_target``."""
        ...

    def read_position(self, media_id: str, t_target: int) -> tuple[list[int], np.ndarray] | None:
        """``(latent_frames, xyz [n, 3] float64)`` of ``media_id`` for block ``t_target``, or
        None."""
        ...


class ClosedLoop:
    """One client's predicted states.

    Its state model and complementary filter, its cameras, the round's player-state table
    (rewritten in place block by block; slots other than the round's clients keep the recording) and
    the predicted visibility built on it.

    Args:
        reader (StateReader): the client's state model in the block loop.
        cameras (ClientCameras): the client's cameras.
        table (PredictedStateTable): the round's player-state table.
        visibility (PredictedVisibility): the client's visibility labels.
        world_state (PositionExchange): where the clients' positions are published.
        others (Sequence[str]): the other clients with a position track.
    """

    def __init__(
        self,
        *,
        reader: StateReader,
        cameras: ClientCameras,
        table: PredictedStateTable,
        visibility: PredictedVisibility,
        world_state: PositionExchange,
        others: Sequence[str],
    ) -> None:
        self.reader, self.cameras, self.table = reader, cameras, table
        self.visibility, self.world_state, self.others = visibility, world_state, tuple(others)
        self.filter = ComplementaryFilter(cameras.track.xyz[0])

    @classmethod
    def build(
        cls,
        batch: dict[str, torch.Tensor],
        *,
        reader: StateReader,
        client: str,
        client_slot: int,
        round_media: Mapping[int, str],
        others: Sequence[str],
        latent_frames: int,
        start_frame: int,
        prior: PhysicsPrior,
        depth_fn: DepthFn,
        world_state: PositionExchange,
    ) -> "ClosedLoop":
        """Set up the closed loop of one client.

        Args:
            batch (dict[str, torch.Tensor]): the whole round (batch size 1); its ``player_states``
                is replaced by the predicted table (a copy of the recording with block 1 written).
            reader (StateReader): the client's state model in the block loop.
            client (str): the client's media id.
            client_slot (int): the client's slot.
            round_media (Mapping[int, str]): media id of every slot of the round.
            others (Sequence[str]): the round's other clients.
            latent_frames (int): latent frames of the client's window of the round.
            start_frame (int): source frame of latent frame 0.
            prior (PhysicsPrior): the motion prior of the extrapolation.
            depth_fn (DepthFn): the depth head.
            world_state (PositionExchange): where the clients' positions are published.

        Returns:
            ClosedLoop: the client's predicted states.
        """
        batch["player_states"] = batch["player_states"].clone()
        alive = batch["player_states"][0, :, :, ALIVE_INDEX].numpy() > 0.5
        slots = {}
        for slot, media in sorted(round_media.items()):
            if (slot != client_slot and media not in others) or not alive[slot, 1:].any():
                continue
            motion = player_motion(batch, slot, prior=prior)
            slots[slot] = (PositionTrack(media, start_frame, motion.start_state[:3]), motion)
        if client_slot not in slots:
            raise ValueError(f"{client} is not alive after the round start: nothing to estimate")
        table = PredictedStateTable(batch["player_states"], slots)
        table.advance(1)  # block 1 starts from the round-start positions
        track, motion = slots[client_slot]
        return cls(
            reader=reader,
            cameras=ClientCameras(track, motion, latent_frames),
            table=table,
            visibility=PredictedVisibility(batch, depth_fn=depth_fn),
            world_state=world_state,
            others=[m for m in others if table.track_of(m)],
        )

    def _t_target(self, s: int) -> int:
        """The source frame of the first latent frame of block ``s``: its name in the exchange."""
        return source_frame(self.cameras.track.start_frame, s)

    def publish_position(self, s: int, latents: torch.Tensor) -> None:
        """Estimate this client's positions at the latent frames ``s-4 .. s-1`` (latent frame 0
        for block 1) from its ``latents`` ``[>= s, 48, 24, 42]`` (the state model, Eq. (4)), extend
        its own track and publish them for block ``s``."""
        s = int(s)
        frames = [0] if s == 1 else list(range(s - BLOCK, s))
        place, displacement = self.reader.read(latents, frames)
        xyz = np.stack(
            [self.filter.step(f, place[i], displacement[i]) for i, f in enumerate(frames)]
        )
        if s > 1:
            self.cameras.track.append(frames, xyz)
        self.world_state.publish_position(self._t_target(s), frames, xyz)

    def read_positions(self, s: int) -> None:
        """Take the other clients' positions for block ``s`` and write the table's video frames of
        the block.

        The visibility geometry of the video frames that changed is re-read. A client without a
        record holds its last position.
        """
        for other in self.others:
            record = self.world_state.read_position(other, self._t_target(s))
            # block 1 publishes latent frame 0, the round-start position every track starts with
            if record is None or record[0] == [0]:
                continue
            self.table.track_of(other).append(*record)
        touched = self.table.advance(s)
        if touched:
            self.visibility.update_frames(np.unique(np.concatenate(list(touched.values()))))

    def _round_latent(self, block: MemoryBlock) -> int:
        """The latent frame of the client's round at which another client's block starts."""
        offset = block.t_first - self.cameras.track.start_frame
        if offset % SOURCE_FRAMES_PER_LATENT:
            raise ValueError(f"{block.media_id}: block off the round's latent grid")
        return offset // SOURCE_FRAMES_PER_LATENT

    def as_generated(self, block: MemoryBlock, recorded: torch.Tensor) -> MemoryBlock:
        """Another client's block with the cameras it was generated from: that client's predicted
        positions and its own view angles.

        Args:
            block (MemoryBlock): the block, keyed at its recorded cameras.
            recorded (torch.Tensor): ``[1, P, T, 6]`` the recorded player-state table of the round.
        """
        last = torch.as_tensor(last_video_frames(self._round_latent(block), BLOCK))
        states = recorded[0, block.slot, last].float().clone()
        states[:, :3] = self.table.states[0, block.slot, last, :3]
        return replace(block, c2w=c2w_from_state_rows(states.numpy()))

    def inputs_as_generated(self, inputs: dict[str, Any], block: MemoryBlock) -> dict[str, Any]:
        """The memory frames' inputs (:func:`~worldcast.data.memory_frames.memory_frame_inputs`)
        with the block's player at the predicted positions it was generated from."""
        frames = torch.as_tensor(video_frames_of(self._round_latent(block), BLOCK))
        states = inputs["states"].clone()
        states[:, :3] = self.table.states[0, block.slot, frames, :3].to(states.dtype)
        return dict(inputs, states=states)
