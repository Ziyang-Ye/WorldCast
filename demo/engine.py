"""The interface between a worker and its engine (``EngineProtocol``), and the records that cross it.

A worker holds one engine for its whole life (models loaded once). Per player session it calls :meth:`start`, then
:meth:`step` once per block; ``step`` asks for the block's controls as late as it can (right before the block's
denoising ladder) and yields the block's frames as they are decoded. The shared world state goes through these hooks:
the player's own state rides on every :class:`Frame` (with the peers where this client placed them); the peers'
latest states come in through :meth:`receive_state`; everything else the engines exchange (scene-state blocks, step
records, the paper engine's own player-state messages) is an opaque :class:`PeerMessage`, out through
:meth:`take_messages` and in through :meth:`receive_message`.

``step`` runs on the worker's engine thread. ``receive_*`` and ``remove_peer`` are called from the network thread at
any time; an engine queues what they bring and applies it at its next block boundary.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from demo.library import RoundStart, Seat

#: Wire precision of positions (engine units) and angles (degrees).
_ROUND = 2


@dataclass
class PlayerState:
    """One player at one frame, as its state model sees it.

    ``t`` is room time in seconds (the end of the frame's input slot); ``x, y, z`` engine units (feet), ``yaw`` /
    ``pitch`` degrees (Source convention: yaw counter-clockwise from +x, pitch > 0 looks down); ``buttons`` the held
    buttons as a bitmask over ``PAPER_ACTION_BUTTONS``; ``vx, vy`` ground velocity, units/s (for extrapolation).
    """

    seat: int
    t: float
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    alive: bool = True
    weapon: int = 0
    buttons: int = 0
    vx: float = 0.0
    vy: float = 0.0

    def wire(self) -> list:
        r = lambda v: round(float(v), _ROUND)  # noqa: E731
        return [
            self.seat,
            round(self.t, 4),
            r(self.x),
            r(self.y),
            r(self.z),
            r(self.yaw),
            r(self.pitch),
            int(self.alive),
            int(self.weapon),
            int(self.buttons),
            r(self.vx),
            r(self.vy),
        ]

    @classmethod
    def from_wire(cls, w: Sequence) -> "PlayerState":
        return cls(
            seat=int(w[0]),
            t=float(w[1]),
            x=float(w[2]),
            y=float(w[3]),
            z=float(w[4]),
            yaw=float(w[5]),
            pitch=float(w[6]),
            alive=bool(w[7]),
            weapon=int(w[8]),
            buttons=int(w[9]),
            vx=float(w[10]),
            vy=float(w[11]),
        )


@dataclass
class BlockActions:
    """One block of controls, ``F`` frames, in the encodings of ``worldcast.data.actions``.

    ``buttons [F, 11]`` float32 {0, 1} in ``PAPER_ACTION_BUTTONS`` order; ``turn [F, 2]`` float32 degrees turned
    during the frame (pitch, yaw); ``camera [F, 2]`` the same turn mu-law encoded (``noclip``, the model's input);
    ``weapon [F]`` int64 fine weapon ids; ``substeps [F, 4, 13]`` float32 per 1/64 s: buttons held, pitch / 5,
    yaw / 5 (the peer-material layout) with ``substep_valid [F, 4]``; ``times [F]`` room seconds of each frame;
    ``input_seq [F]`` the newest browser input tick folded into each frame (-1: none yet).
    """

    block: int
    first_frame: int
    times: np.ndarray
    buttons: np.ndarray
    turn: np.ndarray
    camera: np.ndarray
    weapon: np.ndarray
    substeps: np.ndarray
    substep_valid: np.ndarray
    input_seq: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.buttons.shape[0])


@dataclass
class Frame:
    """One decoded frame: the player's own state at it, the other players where this client placed them (their
    published states, extrapolated), and the picture as ``rgb [H, W, 3]`` uint8 or as ``jpeg`` bytes already encoded
    by the engine."""

    index: int
    state: PlayerState
    peers: list[PlayerState] = field(default_factory=list)
    rgb: np.ndarray | None = None
    jpeg: bytes | None = None


@dataclass
class PeerMessage:
    """What one engine sends the others of its room, opaque to the network: ``kind`` (``scene`` = a scene-state block,
    kept for late joiners; anything else is relayed only), the writer's seat and block, small JSON ``meta`` and the
    ``payload`` (a paper scene-state block is four float32 latents ``[4, 48, 24, 42]``, 774 kB)."""

    seat: int
    block: int
    kind: str
    meta: dict[str, Any]
    payload: bytes


class EngineProtocol(Protocol):
    """What a worker needs from an engine. ``demo.mock_engine.MockEngine`` implements it without a GPU."""

    fps: float
    frames_per_step: int

    def start(self, round_start: RoundStart, seat: Seat, peers: Sequence[Seat]) -> PlayerState:
        """Begin a session at the round start (models stay loaded); returns the player's initial state."""

    def step(self, controls: Callable[[], BlockActions]) -> Iterator[Frame]:
        """Generate the next block: call ``controls()`` once, as late as possible (right before the ladder), and
        yield the block's frames as they are decoded."""

    def receive_state(self, state: PlayerState) -> None:
        """A peer's newest state (async mode: the engine extrapolates it over the block)."""

    def remove_peer(self, seat: int) -> None:
        """A peer left the room."""

    def take_messages(self) -> list[PeerMessage]:
        """Messages for the room's other engines produced since the last call."""

    def receive_message(self, message: PeerMessage) -> None:
        """A message from another engine of the room."""

    def stats(self) -> dict[str, Any]:
        """Small numbers for the HUD (e.g. scene blocks shared)."""
