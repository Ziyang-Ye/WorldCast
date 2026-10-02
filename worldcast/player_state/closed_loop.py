"""Closed-loop deployment (Sec. 3.4, App. "Closed-loop deployment").

Each client reads its own position off its new latent frames with the state model and fuses it
with Eq. (6). It publishes the position once per block and draws the other clients where they
published themselves, extrapolated over the block from their controls
(``worldcast.player_state.extrapolate``). A knot is the position at one latent frame.
"""

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from worldcast.data.actions import ControlTicks, state_model_controls
from worldcast.modeling.state_model import (
    POSITION_UNIT_U,
    WINDOW_LATENTS,
    WINDOW_STRIDE,
    StateModel,
    map_id_of,
)

from .extrapolate import (
    BLOCK,
    SOURCE_FRAMES_PER_LATENT,
    OwnCameraPlan,
    PositionTrack,
    PredictedStateTable,
    player_motion,
)
from .predicted_visibility import PredictedVisibility
from .projection import c2w_from_state_rows

__all__ = [
    "FUSION_WEIGHT",
    "window_of",
    "StateReader",
    "PositionFilter",
    "StateExchange",
    "ClosedLoop",
]

#: Weight of the place estimate in Eq. (6).
FUSION_WEIGHT = 0.5


def window_of(knot: int) -> tuple[int, int]:
    """``(window, position)`` of a knot: knot 0 is read from window 0, knot ``40w + j`` (j in 1..40)
    from window ``w``."""
    k = int(knot)
    if k < 0:
        raise ValueError(f"knot {k}")
    if k == 0:
        return 0, 0
    w = (k - 1) // WINDOW_STRIDE
    return w, k - WINDOW_STRIDE * w


class StateReader:
    """The state model in the block loop.

    Window ``w`` covers latent frames ``40w .. 40w + 40`` of the rollout; a window still being
    generated is read with its future latent frames zeroed (every part of the model is causal).

    Args:
        model (StateModel): the state model.
        ticks (ControlTicks): the client's raw tick stream.
        media_id (str): the client's media id (it selects the map).
        fps (float): source frame rate.
        start_frame (int): source frame of latent frame 0.
        device (torch.device | str): where the model runs.
    """

    def __init__(
        self,
        model: StateModel,
        ticks: ControlTicks,
        *,
        media_id: str,
        fps: float,
        start_frame: int,
        device,
    ) -> None:
        self.model, self.ticks, self.fps = model, ticks, float(fps)
        self.start_frame = int(start_frame)
        self.device = torch.device(device)
        self.map_id = torch.tensor([map_id_of(media_id)], device=self.device)
        self._controls: dict[int, torch.Tensor] = {}

    def _controls_of(self, w: int) -> torch.Tensor:
        if w not in self._controls:
            start = self.start_frame + WINDOW_STRIDE * SOURCE_FRAMES_PER_LATENT * w
            u = state_model_controls(self.ticks, start, self.fps, WINDOW_LATENTS)
            self._controls[w] = torch.from_numpy(u)[None].to(self.device).float()
        return self._controls[w]

    @torch.no_grad()
    def read(self, latents: torch.Tensor, knots: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Read the state model at ``knots``.

        Args:
            latents (torch.Tensor): ``[N, 48, 24, 42]`` generated latent frames (CPU), ``N`` > every
                knot.
            knots (Sequence[int]): latent frames to read.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``A_world`` ``[n, 3]`` float64 (u) and ``inc`` ``[n, 3]``
            float32 (units of 64 u).
        """
        knots = [int(k) for k in knots]
        place = np.zeros((len(knots), 3), np.float64)
        inc = np.zeros((len(knots), 3), np.float32)
        by_window: dict[int, list[tuple[int, int]]] = {}
        for i, k in enumerate(knots):
            w, j = window_of(k)
            by_window.setdefault(w, []).append((i, j))
        for w, items in sorted(by_window.items()):
            last = max(j for _, j in items)
            first = WINDOW_STRIDE * w
            if first + last >= int(latents.shape[0]):
                raise ValueError(
                    f"knot {first + last} requested with {int(latents.shape[0])} latents generated"
                )
            # fp16, as the state model was trained on the fp16 latent cache
            v = np.zeros((WINDOW_LATENTS,) + tuple(latents.shape[1:]), np.float16)
            v[: last + 1] = np.asarray(latents[first : first + last + 1], dtype=np.float32).astype(
                np.float16
            )
            window_inc, window_place = self.model(
                torch.from_numpy(v)[None].to(self.device).float(), self._controls_of(w), self.map_id
            )
            window_place = window_place[0].float().cpu().numpy()
            window_inc = window_inc[0].float().cpu().numpy()
            for i, j in items:
                place[i] = window_place[j].astype(np.float64)
                inc[i] = window_inc[j]
        return place, inc


class PositionFilter:
    """Eq. (6) at the constant weight 1/2, knot by knot.

    ``p_0 = S_0``, ``p_k = 1/2 (p_{k-1} + 64 inc_k) + 1/2 (A_k - A_0 + S_0)``; positions in u,
    float64.

    Args:
        s0 (np.ndarray): ``[3]`` the round-start position, u.
    """

    def __init__(self, s0) -> None:
        self.s0 = np.asarray(s0, np.float64).reshape(3)
        self.a0 = None
        self.p = None
        self.k = -1

    def step(self, k: int, place, inc) -> np.ndarray:
        """Fuse the next knot ``k``: ``place`` ``[3]`` u, ``inc`` ``[3]`` units of 64 u."""
        k = int(k)
        if k != self.k + 1:
            raise ValueError(f"knot {k} after knot {self.k}: the filter runs knot by knot")
        place = np.asarray(place, np.float64).reshape(3)
        if k == 0:
            self.a0, self.p = place, self.s0.copy()
        else:
            q = self.p + np.asarray(inc, np.float64).reshape(3) * POSITION_UNIT_U
            self.p = q - FUSION_WEIGHT * (q - (place - self.a0 + self.s0))
        self.k = k
        return self.p.copy()


class StateExchange:
    """Position records in the live pool directory, one per client and block.

    ``<root>/<client>/state/state_<t:06d>.json`` = ``{"knots": [...], "xyz": [[x, y, z], ...]}``. A
    client writes its record for block ``t`` before its step record, so a peer that has waited for
    the step record can read the position too; the plain prefix has no step records, and there
    :meth:`wait` is the lock-step.

    Args:
        root (str | Path): the live pool directory.
        client (str): this client's media id.
        poll_s (float): polling interval of :meth:`wait`, seconds.
        fatal (bool): a timeout raises (config ``pool.fail_on_timeout``).
    """

    def __init__(self, root, client: str, *, poll_s: float, fatal: bool) -> None:
        self.root, self.client = Path(root), str(client)
        self.poll_s, self.fatal = float(poll_s), bool(fatal)

    def _path(self, client: str, t_target: int) -> Path:
        return self.root / client / "state" / f"state_{int(t_target):06d}.json"

    def publish(self, t_target: int, knots: Sequence[int], xyz: np.ndarray) -> None:
        """Write this client's record for block ``t_target`` (atomically)."""
        path = self._path(self.client, t_target)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        record = {"knots": [int(k) for k in knots], "xyz": np.asarray(xyz, np.float64).tolist()}
        tmp.write_text(json.dumps(record))
        os.replace(tmp, path)

    def read(self, client: str, t_target: int) -> tuple[list[int], np.ndarray] | None:
        """``(knots, xyz [n, 3] float64)`` of ``client`` at block ``t_target``, or None."""
        path = self._path(client, t_target)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        knots = [int(k) for k in record["knots"]]
        return knots, np.asarray(record["xyz"], np.float64).reshape(-1, 3)

    def wait(
        self,
        peers: Sequence[str],
        t_target: int,
        *,
        max_wait_s: float,
        is_done: Callable[[str], bool],
    ) -> None:
        """Until every peer has published block ``t_target`` or is done."""
        start = time.monotonic()
        while True:
            late = [p for p in peers if not is_done(p) and not self._path(p, t_target).exists()]
            if not late:
                return
            if time.monotonic() - start >= float(max_wait_s):
                if self.fatal:
                    raise TimeoutError(f"no position record of {late} for block {t_target}")
                return
            time.sleep(self.poll_s)


class ClosedLoop:
    """One client's predicted states.

    Its state-model reader and Eq. (6) filter, its own camera plan, the round's player-state table
    (rewritten in place block by block; seats other than the round's clients keep the recording) and
    the predicted visibility built on it.

    Args:
        reader (StateReader): the client's state model read-out.
        own_cameras (OwnCameraPlan): the client's own cameras.
        table (PredictedStateTable): the round's player-state table.
        visibility (PredictedVisibility): the client's visibility labels.
        exchange (StateExchange): where the clients' positions are published.
        peers (Sequence[str]): the other clients with a position track.
    """

    def __init__(
        self,
        *,
        reader: StateReader,
        own_cameras: OwnCameraPlan,
        table: PredictedStateTable,
        visibility: PredictedVisibility,
        exchange: StateExchange,
        peers: Sequence[str],
    ) -> None:
        self.reader, self.own_cameras, self.table = reader, own_cameras, table
        self.visibility, self.exchange, self.peers = visibility, exchange, tuple(peers)
        self.filter = PositionFilter(own_cameras.track.xyz[0])

    @classmethod
    def build(
        cls,
        batch: dict[str, torch.Tensor],
        *,
        reader: StateReader,
        me: str,
        my_slot: int,
        round_media: Mapping[int, str],
        clients: Sequence[str],
        n_latents: int,
        start_frame: int,
        motion_kwargs: dict,
        eye_height: float,
        depth_fn,
        pose_radius: float,
        exchange: StateExchange,
    ) -> "ClosedLoop":
        """Set up the closed loop of one client.

        Args:
            batch (dict[str, torch.Tensor]): the whole round; its ``player_states`` is replaced by
                the predicted table (a copy of the recording with block 1 written).
            reader (StateReader): the client's state model read-out.
            me (str): the client's media id.
            my_slot (int): the client's seat.
            round_media (Mapping[int, str]): media id of every seat of the round.
            clients (Sequence[str]): the round's other clients.
            n_latents (int): latent frames of the rollout.
            start_frame (int): source frame of latent frame 0.
            motion_kwargs (dict): ``camera_delta_scale``, ``channels`` and ``prior`` of
                :func:`worldcast.player_state.extrapolate.player_motion`.
            eye_height (float): camera height above the feet, u.
            depth_fn (DepthFn): the picture depth head.
            pose_radius (float): radius of the extra visibility test points, u.
            exchange (StateExchange): where the clients' positions are published.

        Returns:
            ClosedLoop: the client's predicted states.
        """
        batch["player_states"] = batch["player_states"].clone()
        alive = batch["player_states"][0, :, :, 5].numpy() > 0.5
        seats = {}
        for slot, media in sorted(round_media.items()):
            if (slot != my_slot and media not in clients) or not alive[slot, 1:].any():
                continue
            motion = player_motion(batch, slot, **motion_kwargs)
            seats[slot] = (PositionTrack(media, start_frame, motion.s0[:3]), motion)
        if my_slot not in seats:
            raise ValueError(f"{me} is not alive after the round start: nothing to estimate")
        table = PredictedStateTable(batch["player_states"], seats)
        table.advance(1)  # block 1 starts from the round-start positions
        track, motion = seats[my_slot]
        visibility = PredictedVisibility(
            batch,
            depth_fn=depth_fn,
            camera_delta_scale=motion_kwargs["camera_delta_scale"],
            pose_radius=pose_radius,
        )
        return cls(
            reader=reader,
            own_cameras=OwnCameraPlan(track, motion, n_latents, eye_height=eye_height),
            table=table,
            visibility=visibility,
            exchange=exchange,
            peers=[m for m in clients if table.track_of(m)],
        )

    def publish_own(self, s: int, t_target: int, latents: torch.Tensor) -> None:
        """Read this client's knots ``s-4 .. s-1`` (knot 0 for block 1) off its ``latents``
        ``[>= s, 48, 24, 42]``, fuse them (Eq. (6)), extend its own track and publish them for block
        ``s``."""
        s = int(s)
        knots = [0] if s == 1 else list(range(s - BLOCK, s))
        place, inc = self.reader.read(latents, knots)
        xyz = np.stack([self.filter.step(k, place[i], inc[i]) for i, k in enumerate(knots)])
        if s > 1:
            self.own_cameras.track.append(knots, xyz)
        self.exchange.publish(t_target, knots, xyz)

    def read_peers(self, s: int, t_target: int) -> None:
        """Take the other clients' records for block ``s`` and write the table's rows of the block.

        The visibility geometry of the rows that changed is re-read. A client without a record holds
        its last position.
        """
        for peer in self.peers:
            record = self.exchange.read(peer, t_target)
            if record is None or record[0] == [0]:
                continue
            self.table.track_of(peer).append(*record)
        touched = self.table.advance(s)
        if touched:
            self.visibility.update_rows(np.unique(np.concatenate(list(touched.values()))))

    def rekey(
        self, block: dict | None, recorded: torch.Tensor, *, start_frame: int, stride: int
    ) -> dict | None:
        """A peer's memory entry keyed at the cameras its publisher drew it from (its predicted
        position, its own view angles); ``recorded``: the recorded round table ``[1, P, T, 6]``."""
        if block is None:
            return None
        orig = int(block["window_start"]) + stride * (int(block["f0"]) + np.arange(BLOCK))
        if ((orig - start_frame) % stride).any():
            raise ValueError(f"{block['media_id']}: block off the round's latent grid")
        rows = 4 * ((orig - start_frame) // stride)
        state_rows = recorded[0, int(block["slot"]), torch.as_tensor(rows)].float().clone()
        state_rows[:, :3] = self.table.states[0, int(block["slot"]), torch.as_tensor(rows), :3]
        c2w = c2w_from_state_rows(state_rows.numpy(), eye_height=self.own_cameras.eye_height)
        return dict(block, c2w=c2w)

    def slot_rows(self, frames: dict, block: Mapping, *, start_frame: int, stride: int) -> dict:
        """The memory slot's 16 pixel rows (``frames``, :class:`~worldcast.data.memory_slot.
        MemorySlotFrames` output) at the positions its publisher was drawn at."""
        first = (int(block["window_start"]) + stride * int(block["f0"]) - start_frame) // stride
        rows = torch.arange(4 * first - 3, 4 * (first + BLOCK - 1) + 1)
        states = frames["states"].clone()
        states[:, :3] = self.table.states[0, int(block["slot"]), rows, :3].to(states.dtype)
        return dict(frames, states=states)
