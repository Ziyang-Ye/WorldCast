"""The client a worker runs, behind one interface (:class:`ClientProtocol`).

A worker holds one client for its whole life (models loaded once). It calls
:meth:`ClientProtocol.warm_up` once, before it registers with the coordinator. Per player it calls
:meth:`ClientProtocol.start`, then :meth:`ClientProtocol.step` once per block, then
:meth:`ClientProtocol.stop`; ``step`` asks for the block's controls as late as it can, right before
the block's denoising, and yields the block's frames as they are decoded. Every :class:`LiveFrame`
carries the players as the client's player state field read them: its own state and the other
players' (:class:`LivePlayerState`), each with where it comes from. What the clients of a room
exchange (player states, memory entries, withdrawals) is an opaque :class:`ClientMessage`.
``warm_up``, ``start``, ``step`` and ``stop`` run on the worker's client thread;
``receive_message`` and ``remove_player`` come from the network thread. A client with
``blocks_in_flight`` 2 (the WorldCast client whose decoder runs on its own device) has ``step``
request the block and return at once; the worker takes the frames on another thread, in order, and
calls ``step`` for the next block before it has them all.

* :class:`worldcast.demo.worldcast_client.WorldCastClient`: the WorldCast client of one GPU.
* :class:`worldcast.demo.mock.MockClient`: no GPU; pre-rendered frames or a grid world.
"""

import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from worldcast.engine.inference.controls import LiveBlockControls
from worldcast.engine.inference.field_rows import FieldRows

from .config import DemoConfig
from .library import Library, RoundStart, Seat

__all__ = [
    "LIVE_STATE",
    "MEMORY_ENTRY",
    "ClientMessage",
    "ClientProtocol",
    "LiveControls",
    "LiveFrame",
    "LivePlayerState",
    "Outbox",
    "button_bits",
    "make_client",
]

#: The kinds of :class:`ClientMessage` the demo itself reads (those of
#: :attr:`worldcast.engine.inference.messages.MessagePart.kind`): a player's live state, the rows
#: its camera used in a block (``live``), and a memory entry; the coordinator keeps the newest of
#: both for late joiners.
LIVE_STATE, MEMORY_ENTRY = "live", "scene"
#: Decimals on the wire: positions (engine units), angles (degrees) and velocities; room seconds;
#: a field marker's place in the frame (fractions of its width and height) and its depth (u).
_DECIMALS, _TIME_DECIMALS, _MARKER_DECIMALS, _DEPTH_DECIMALS = 2, 4, 4, 1
#: Decimals of the degrees a player has turned on the wire: those of the page's input samples.
_TURN_DECIMALS = 4


@dataclass
class LivePlayerState:
    """The player state of one player at one frame, as a frame's header carries it: the row the
    client's player state field read
    (:meth:`~worldcast.engine.inference.client.Client.field_rows`).

    ``slot``: the player slot; ``t``: room time, s (the end of the frame's input slot);
    ``x, y, z``: feet, engine units; ``yaw``, ``pitch``: degrees (yaw counter-clockwise from +x,
    pitch > 0 looks down); ``vx, vy``: ground velocity, units/s; ``src``: where the row comes
    from, a code of :class:`~worldcast.engine.inference.field_rows.FieldRows`: ``OWN`` (the
    player's own client), ``LIVE`` (another client that plays it) or ``RECORDED`` (the seat's
    recording, a seat nobody plays); ``drawn``: whether the field draws the player at the frame
    (its visibility label), ``None`` where the client has no label (its own row, the first
    frame).
    """

    slot: int
    t: float
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    alive: bool = True
    vx: float = 0.0
    vy: float = 0.0
    src: int = FieldRows.OWN
    drawn: bool | None = None

    def wire(self) -> list:
        """``[slot, t, x, y, z, yaw, pitch, alive, vx, vy, src, drawn]``; ``drawn`` is 1, 0 or
        ``null``."""
        numbers = (self.x, self.y, self.z, self.yaw, self.pitch, self.vx, self.vy)
        x, y, z, yaw, pitch, vx, vy = (round(float(v), _DECIMALS) for v in numbers)
        t = round(float(self.t), _TIME_DECIMALS)
        drawn = None if self.drawn is None else int(self.drawn)
        return [self.slot, t, x, y, z, yaw, pitch, int(self.alive), vx, vy, int(self.src), drawn]

    @classmethod
    def from_wire(cls, wire: Sequence) -> "LivePlayerState":
        """The state of :meth:`wire`."""
        slot, t, x, y, z, yaw, pitch, alive, vx, vy, src, drawn = wire
        drawn = None if drawn is None else bool(drawn)
        return cls(int(slot), t, x, y, z, yaw, pitch, bool(alive), vx, vy, int(src), drawn)


@dataclass
class LiveControls(LiveBlockControls):
    """A block's controls taken from a player's browser, one row per frame: the live controls of
    :class:`~worldcast.engine.inference.controls.LiveBlockControls`, their ``times`` on the room
    clock, ``input_seq [F]`` the newest browser input folded into each frame (-1: none), and
    ``turned [F, 2]`` float64 how far the player has turned in the session through each frame
    (pitch, yaw), degrees: the turn of every input sample folded into the controls up to and
    including the frame's row.

    Raises:
        ValueError: ``turned`` is not ``[F, 2]``.
    """

    input_seq: np.ndarray
    turned: np.ndarray

    def __post_init__(self) -> None:
        super().__post_init__()
        self.turned = np.asarray(self.turned, dtype=np.float64)
        if self.turned.shape != (self.num_frames, 2):
            raise ValueError(
                f"live controls of {self.num_frames} rows have turned [{self.num_frames}, 2]; got"
                f" {self.turned.shape}"
            )

    @property
    def num_frames(self) -> int:
        return int(self.buttons.shape[0])


@dataclass
class LiveFrame:
    """One decoded frame of a player's view.

    Attributes:
        index (int): the frame's index in the session's video.
        rgb (np.ndarray): ``[H, W, 3]`` uint8.
        state (LivePlayerState): the player's own state at the frame: the row the player state
            field read at the frame's latent frame.
        turned (tuple[float, float]): how far the player had turned in the session through the
            frame (pitch, yaw), degrees: :attr:`LiveControls.turned` of the frame's row, the turn
            of the input samples folded into its view, all of which the view holds but what the
            pitch limit cut.
        camera_pitch (float): the pitch of the camera at the frame itself, degrees (> 0 looks
            down); ``state`` has the pitch at the last frame of the frame's latent frame.
        others (list[LivePlayerState]): the states of the other players the client has in the
            world at the frame (a hidden one is left out).
        input_seq (int): the newest browser input folded into the frame (-1: none).
        markers (list[tuple[int, float, float, float]]): ``(slot, u, v, depth)`` of every other
            player in front of the camera: where the player state field puts its feet, ``u`` and
            ``v`` fractions of the frame's width and height from its top left corner, and its
            depth, u.
        status (str): how the player's own row is known
            (:attr:`~worldcast.engine.inference.field_rows.FieldRows.status`).
        version (int): which version of its block (``play.refresh``): 0 the block's first, then
            one per refresh asked for (:meth:`ClientProtocol.refresh`).
        buttons (int): the buttons held in the frame's row of the controls, bit ``i`` for
            ``CONTROL_BUTTONS[i]`` (:func:`button_bits`).
        weapon (int): the weapon held in that row, an id of ``OPENCS2_WEAPONS``.
        preview (bool): a frame of a refresh's preview (``play.preview_step``), which its
            version's own frame replaces where it has not gone.
    """

    index: int
    rgb: np.ndarray
    state: LivePlayerState
    turned: tuple[float, float]
    camera_pitch: float
    others: list[LivePlayerState] = field(default_factory=list)
    input_seq: int = -1
    markers: list[tuple[int, float, float, float]] = field(default_factory=list)
    status: str = ""
    version: int = 0
    buttons: int = 0
    weapon: int = 0
    preview: bool = False

    def marker_wire(self) -> list[list]:
        """:attr:`markers` as the frame header carries them: ``[[slot, u, v, depth], ...]``."""
        u_v, depth_u = _MARKER_DECIMALS, _DEPTH_DECIMALS
        return [
            [int(slot), round(float(u), u_v), round(float(v), u_v), round(float(depth), depth_u)]
            for slot, u, v, depth in self.markers
        ]

    def look_wire(self) -> dict[str, Any]:
        """How much of the player's turn the frame holds, as its header carries it: ``tn``
        :attr:`turned` ``[pitch, yaw]`` with the decimals of the page's input samples, and ``cp``
        :attr:`camera_pitch`."""
        return {
            "tn": [round(float(degrees), _TURN_DECIMALS) for degrees in self.turned],
            "cp": round(float(self.camera_pitch), _DECIMALS),
        }

    def controls_wire(self) -> dict[str, int]:
        """The controls of the frame's row as its header carries them, as the page's input samples
        say them: ``b`` :attr:`buttons`, ``w`` :attr:`weapon`."""
        return {"b": int(self.buttons), "w": int(self.weapon)}


def button_bits(row: np.ndarray) -> int:
    """The buttons a row of the controls holds (``[B]``, any share of the frame), bit ``i`` for
    ``CONTROL_BUTTONS[i]``."""
    return sum(1 << i for i, held in enumerate(np.asarray(row).tolist()) if held > 0)


@dataclass
class ClientMessage:
    """What one client sends the other clients of its room, opaque to the network.

    ``slot``: the sender's player slot; ``kind``: :data:`LIVE_STATE` or :data:`MEMORY_ENTRY` (kept
    for late joiners), or any other kind, only relayed; small JSON ``meta``; the ``payload``.
    """

    slot: int
    kind: str
    meta: dict[str, Any]
    payload: bytes


class ClientProtocol(Protocol):
    """What a worker needs from a client."""

    frames_per_block: int
    #: The blocks the worker may have started whose frames it has not taken to the end: 1, the
    #: next block starts when the last one's frames are all taken; 2, the next block starts while
    #: the last one's frames are still being decoded, and ``step`` only requests its block.
    blocks_in_flight: int
    #: Called after each message for the room's other clients is queued, on the thread that queued
    #: it: the worker then takes and sends it at once.
    notify: Callable[[], None] | None
    #: The cards a refresh may run on (``play.refresh``): 1, the client's own, where a refresh asked
    #: for while another is generated waits for it; more (the WorldCast client's
    #: ``refresh_devices``), and a refresh starts at once on a card that is free
    #: (:attr:`refresh_card_free`).
    refresh_cards: int
    #: With more than one card: called from any thread when a card has no job left.
    card_freed: Callable[[], None] | None

    def warm_up(self) -> None:
        """Run whatever makes the first session's blocks as fast as the later ones (the WorldCast
        client compiles its generator and captures its CUDA graphs); what it would send the other
        clients is dropped."""

    def start(self, round_start: RoundStart, seat: Seat) -> None:
        """Join the room's session as the player of ``seat`` of ``round_start``; the room's other
        players are in the world from their first live state on. The caller has taken the messages
        of the client's earlier session (:meth:`stop`)."""

    def step(self, controls: Callable[[], LiveControls]) -> Iterator[LiveFrame]:
        """Generate the next block; yield its frames as they are decoded. A session has no last
        block: a block always yields at least one frame, and none raises.

        ``controls()`` is called once, right before the block's denoising. With
        ``blocks_in_flight`` > 1 the block is requested by the call itself, and the iterators of
        successive calls are taken in order. With ``play.refresh`` the newest version of the block
        before is committed first, and a refresh of the block generated while its frames are
        decoded may take the decode over: the iterator then yields the refresh's frames
        (:attr:`LiveFrame.version`), from its switch on, which may come again for frames it
        yielded already (their newer version), and that refresh's own iterator yields none.
        """

    @property
    def can_refresh(self) -> bool:
        """Whether the newest block can be generated again from newer controls (:meth:`refresh`;
        ``play.refresh``)."""

    @property
    def refresh_card_free(self) -> bool:
        """With more than one card: whether one is free, so that a refresh asked for now starts at
        once."""

    def refresh(
        self,
        controls: Callable[[], LiveControls],
        switch: Callable[[float], int],
        generated: Callable[[], None] | None = None,
        preview_step: int = 0,
    ) -> Iterator[LiveFrame]:
        """Generate the newest block again; yield its frames from the first one not yet sent on,
        as they are decoded (``play.refresh``).

        The worker calls it on its event loop's thread, from the block's controls' cut on (its
        frames may not be decoded yet), and it returns at once: the refresh waits for the block's
        first denoising. ``controls()`` is called once, right before the refresh's denoising.
        ``switch(seconds)`` is called when the frames are about to be decoded, with what the
        client takes to decode a latent frame's (four) frames: the first frame of the block to
        yield (the frames before it are not); one past the block's last yields none.
        ``generated()``, if given, is called once the new version is generated, before its frames
        are decoded (from any thread). With ``preview_step`` (``play.preview_step``) the x0 of that
        denoising step is decoded as soon as it is there, and its frames (:attr:`LiveFrame.preview`)
        come first, from ``switch`` on as the version's own do, which follow. The iterator
        is taken after the step's, on the thread that takes those. The block's ``live`` part
        leaves with the next :meth:`step` (or :meth:`stop`), with the controls of the newest
        version. With ``play.refresh`` the worker also calls :meth:`step` on its event loop's
        thread, where it must return at once (the block is decoded as the iterator is taken).
        """

    def stop(self) -> None:
        """Leave the session; the messages it queues (the WorldCast client's ``done``) are taken
        after it, and sent after all the others."""

    def remove_player(self, slot: int) -> None:
        """The player of ``slot`` left the room: it is hidden from the next block on, whether or
        not its own last message arrives."""

    def take_messages(self) -> list[ClientMessage]:
        """The messages for the room's other clients produced since the last call."""

    def receive_message(self, message: ClientMessage) -> None:
        """A message from another client of the room."""


class Outbox:
    """The messages a client produced since the last :meth:`take_messages`.

    Attributes:
        notify (Callable[[], None] | None): called after each :meth:`put`, on the thread that put
            (:attr:`ClientProtocol.notify`).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: list[ClientMessage] = []
        self.notify: Callable[[], None] | None = None

    def put(self, message: ClientMessage) -> None:
        """Queue a message for the room's other clients (any thread), then :attr:`notify`."""
        with self._lock:
            self._messages.append(message)
        if self.notify is not None:
            self.notify()

    def take_messages(self) -> list[ClientMessage]:
        """The queued messages, in order; the queue is empty afterwards."""
        with self._lock:
            out, self._messages = self._messages, []
        return out


def make_client(config: DemoConfig, library: Library) -> ClientProtocol:
    """The configured client: the mock, or the WorldCast client with its models loaded."""
    if config.client.kind == "mock":
        from .mock import MockClient

        return MockClient(config.client.mock, library, refresh=config.play.refresh)
    from .worldcast_client import WorldCastClient

    return WorldCastClient(
        config.client.worldcast, config.scene_state, library, refresh=config.play.refresh
    )
