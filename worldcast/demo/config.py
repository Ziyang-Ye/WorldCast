"""The demo's settings (docs/demo.md, "Config"): the defaults below, with YAML files merged over
them in order, then dotted ``--set key=value`` overrides. ``tools/serve_demo.py --print-config``
prints the resolved settings."""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from worldcast.config.inference import SceneStateConfig
from worldcast.config.loader import build, check_choice, load
from worldcast.engine.inference.serving import ServingOptions

__all__ = [
    "ClientConfig",
    "CoordinatorConfig",
    "DemoConfig",
    "LiveSceneStateConfig",
    "MockConfig",
    "PlayConfig",
    "WorkerConfig",
    "WorldCastClientConfig",
    "load_config",
]

FRAME_ROUTES = ("direct", "proxy")
CLIENT_KINDS = ("mock", "worldcast")
INPUT_MAPPINGS = ("latest", "front", "hybrid", "realtime")
#: With ``play.refresh`` the generator is fed this times the player's yaw by default: the view it
#: generates turns short of the yaw it is fed, more so in a flick than in a steady turn. Bench on
#: an H200 (``look`` over three draws, ``orbit``): the view turns 0.76 of a flick fed 1.25 times,
#: 0.91 fed 1.45 times and 0.96 fed 1.6 (2.0 no further), and a steady turn 1.04, 1.15 and 1.27 of
#: it; 1.45 keeps the worse of the two within 15 per cent.
REFRESH_YAW_GAIN = 1.45
MOCK_PLAYER_STATE_SOURCES = ("live", "gt")
#: ``play.preview_step`` names one of the four denoising steps before the last.
PREVIEW_STEPS = 4


@dataclass
class CoordinatorConfig:
    """The coordinator: where it listens, how browsers reach the workers, the rooms."""

    host: str = "0.0.0.0"
    port: int = 8100
    #: How browsers reach a worker's frames: ``direct``, on the worker's own WebSocket, or
    #: ``proxy``, through the coordinator.
    frame_route: str = "direct"
    max_rooms: int = 32
    #: Seconds an empty room stays open.
    empty_room_s: float = 120.0

    def __post_init__(self) -> None:
        check_choice("coordinator.frame_route", self.frame_route, FRAME_ROUTES)


@dataclass
class WorkerConfig:
    """A GPU worker: where it listens, how it reaches the coordinator and how browsers reach it."""

    host: str = "0.0.0.0"
    port: int = 8101
    coordinator_url: str = "ws://127.0.0.1:8100"
    #: WebSocket base browsers use to reach this worker, e.g. ``ws://gpu-07.lan:8101``; ``None``:
    #: the address the worker connected to the coordinator from.
    advertise_url: str | None = None
    worker_id: str | None = None
    #: JPEG quality of the player's frames while its link keeps up; a socket that falls behind
    #: steps down through 75, 60 and 50, and spectators start a step lower (docs/demo.md,
    #: "Latency").
    jpeg_quality: int = 85
    #: How long a browser's link may take to deliver a frame behind those sent and not yet
    #: acknowledged, at the rate it delivers them, ms; at least two round trips of the link. The
    #: other frames wait on the worker, where a socket that falls behind skips or drops them.
    max_inflight_ms: float = 250.0
    #: Seconds a player may reconnect (page reload) before the seat is released.
    reconnect_grace_s: float = 8.0

    def __post_init__(self) -> None:
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("worker.jpeg_quality must be in [1, 100]")
        if self.max_inflight_ms <= 0:
            raise ValueError("worker.max_inflight_ms must be positive")


@dataclass
class MockConfig:
    """The mock client: its timing (a block is ``frames_per_block`` frames, the first
    ``denoise_ms``, plus or minus ``jitter_ms``, after its controls, the others ``decode_ms``
    apart) and the player states it stands in for."""

    frames_per_block: int = 4
    denoise_ms: float = 180.0
    decode_ms: float = 4.0
    jitter_ms: float = 15.0
    #: The WorldCast client's ``player_state.source`` the mock stands in for: ``live``, the
    #: demo's, where the camera follows the keys; or ``gt``, where the players are drawn into the
    #: camera of the seat's recording while the keys move the view (what live play fixes).
    player_state_source: str = "live"

    def __post_init__(self) -> None:
        if self.frames_per_block < 1:
            raise ValueError("client.mock.frames_per_block must be at least 1")
        check_choice(
            "client.mock.player_state_source", self.player_state_source, MOCK_PLAYER_STATE_SOURCES
        )


@dataclass
class WorldCastClientConfig:
    """The WorldCast client: the inference YAMLs, dotted overrides of them, how it is served.

    The demo plays ``player_state.source: live`` (``configs/demo/worldcast.yaml`` sets it): the
    client's camera follows the player's keys, and the only other players in the world are those
    whom people play, drawn where their clients put them. The client refuses ``gt`` and
    ``predicted``: under live keys the first keeps the camera on the seat's recording, so the world
    drifts with the keys, and the second reads the recorded controls.
    """

    configs: list[str] = field(default_factory=list)
    #: Dotted overrides of the inference YAMLs. A live session attends with FlashAttention-3 when
    #: it is installed (``model.attention: fa3``); ``{model.attention: flash}`` with ``serving:
    #: {compile: false}`` gives exact latents.
    overrides: dict[str, Any] = field(default_factory=dict)
    #: ``taew2_2.pth``, the tiny decoder's weights (``tools/download_weights.py``).
    taehv_path: str | None = None
    #: Fields of ``ServingOptions.low_latency`` to replace, e.g. ``{compile: false}``.
    serving: dict[str, Any] = field(default_factory=dict)

    def serving_options(self) -> ServingOptions:
        """How the client is served: with low latency, on the latest messages (the paper's
        deployment; lockstep is its evaluation's, ``tools/run_session.py``)."""
        if "lockstep" in self.serving:
            raise ValueError("client.worldcast.serving: a live client never runs in lockstep")
        try:
            return ServingOptions.low_latency(self.taehv_path, **self.serving)
        except (TypeError, ValueError) as error:
            raise ValueError(f"client.worldcast: {error}") from None


@dataclass
class ClientConfig:
    """Which client the workers run: ``mock`` (no GPU) or ``worldcast``."""

    kind: str = "mock"
    mock: MockConfig = field(default_factory=MockConfig)
    worldcast: WorldCastClientConfig = field(default_factory=WorldCastClientConfig)

    def __post_init__(self) -> None:
        check_choice("client.kind", self.kind, CLIENT_KINDS)
        if self.kind == "worldcast":
            self.worldcast.serving_options()


@dataclass(frozen=True)
class LiveSceneStateConfig(SceneStateConfig):
    """The scene state of a live session (the inference config's ``scene_state``).

    A client keeps ``bound`` own memory entries, and the coordinator as many per player for late
    joiners. The paper's B = 64 makes a block's retrieval outlast its frames once the memory bank
    is full; 32 keeps a long session in step with its display (docs/demo.md, "Latency").
    """

    bound: int = 32


@dataclass
class PlayConfig:
    """How a player's inputs are read and the frames shown: a worker's settings, which the page
    takes from the worker's ``welcome``."""

    #: Frames the browser shows per second while the blocks keep up (the worker paces them for
    #: it). A block's 16 frames, a second of the recording, last ``16 / fps`` s: at 16, the
    #: recording's rate, the second a block takes on an H200; an H20 needs 15, the 1.067 s a block
    #: takes there in a long session.
    fps: float = 16.0
    #: Frames the browser buffers before it shows the first one; from then on its playout follows
    #: the rate the blocks arrive at (docs/demo.md, "Latency").
    jitter_buffer_frames: int = 1
    #: Frame periods (or two jitters, if more) before it is due that the browser's playout has
    #: every coming frame arrive: it slows down to keep this margin (``Playout`` in
    #: ``static/stream.js``). Less is less latency and more holds.
    margin_frames: float = 1.5
    #: How input samples become a block's controls (:mod:`worldcast.demo.inputs`): ``hybrid``
    #: (``front`` for a block that turns little, ``latest`` for one that turns more, and a gain on
    #: the yaw the generator is fed), ``latest`` (the turn spread over the block), ``front`` (the
    #: turn done in the block's first latent frame, and a refresh's in the first rows after its
    #: switch) or ``realtime``; ``None`` (the default): ``front`` with ``refresh``, ``latest``
    #: without (:attr:`mapping`).
    input_mapping: str | None = None
    #: The generator is fed this times the player's yaw on every mapping (the view it generates
    #: turns 0.75-0.85 of the controls at 25-150 degrees a second), and ``pitch_gain`` times the
    #: pitch; the camera and the page keep the turn made. ``None`` (the default): with ``refresh``
    #: :data:`REFRESH_YAW_GAIN` and 1 for the pitch, without both 1 (:attr:`gains`).
    yaw_gain: float | None = None
    pitch_gain: float | None = None
    #: Degrees the view turns per pixel the mouse moves, until the player sets their own.
    mouse_degrees_per_pixel: float = 0.03
    #: Seconds without an input sample after which a block's cut releases the buttons the browser
    #: held (a hidden page, a lost link); longer than the gaps of a congested link, so that a held
    #: key is not released for a whole block.
    release_after_s: float = 1.0
    #: Seconds a page may stay hidden, and seconds a player may send no key and no mouse movement,
    #: before the seat is released (a forgotten tab holds a GPU for ever, since a hidden page
    #: still acknowledges frames). Not a limit on play: a player who plays is never released, and
    #: any input, or the page shown again, starts both over.
    away_s: float = 900.0
    idle_s: float = 900.0
    #: Generate the block on screen again when the player's input changes, so that the input shows
    #: in the frames not yet sent instead of waiting for the next block (the WorldCast client's
    #: ``live.refresh``; docs/demo.md, "Latency"). The worker then sends each frame shortly
    #: before the display shows it.
    refresh: bool = False
    #: With ``refresh``, a turn of at least this many degrees (pitch or yaw) since the block's last
    #: cut or refresh asks for a refresh; so does any button or weapon change, and a turn that
    #: starts, stops or turns back (:meth:`~worldcast.demo.inputs.InputBuffer.changed`).
    refresh_threshold_deg: float = 1.0
    #: With ``refresh``, ms a frame goes before the pacing's display shows it (the network's
    #: jitter, :class:`~worldcast.demo.worker.SendQueue`); until then a refresh may replace it.
    #: Less lets a refresh replace frames nearer the display, at the risk of a frame late on a
    #: link that jitters more.
    send_lead_ms: float = 40.0
    #: With ``refresh``, a newer version of a frame that went goes too (a refresh's, or its own
    #: after its preview's) while the page has not shown it, ``margin_frames`` frame periods after
    #: the pacing's display less the send lead, and the page shows it in place of the one it holds;
    #: a refresh then replaces the frames from the first the page will not have shown on instead of
    #: the first not sent (:class:`~worldcast.demo.worker.SendQueue`).
    swap_sent: bool = False
    #: With ``refresh``, decode the x0 of this denoising step of a refresh (1-3 of the four; 0:
    #: none) as soon as it is there, and send its frames as the refresh's preview in the places of
    #: those not yet sent, until the refresh's own frames take their places
    #: (:meth:`~worldcast.demo.clients.ClientProtocol.refresh`): an input shows that much sooner,
    #: from a coarser estimate of the block for a while.
    preview_step: int = 0
    #: With ``refresh``, ms a refresh asked for by a turn that starts or turns back waits for more
    #: of the turn after its first sample arrived, unless it has turned 20 degrees by then
    #: (:meth:`~worldcast.demo.inputs.InputBuffer.gathering`); 0: none waits.
    gather_ms: float = 20.0

    def __post_init__(self) -> None:
        if self.input_mapping is not None:
            check_choice("play.input_mapping", self.input_mapping, INPUT_MAPPINGS)
        if self.fps <= 0:
            raise ValueError("play.fps must be positive")
        if self.margin_frames < 0:
            raise ValueError("play.margin_frames must not be negative")
        if self.release_after_s <= 0:
            raise ValueError("play.release_after_s must be positive")
        for name in ("yaw_gain", "pitch_gain"):
            if getattr(self, name) is not None and getattr(self, name) <= 0:
                raise ValueError(f"play.{name} must be positive")
        if self.away_s <= 0 or self.idle_s <= 0:
            raise ValueError("play.away_s and play.idle_s must be positive")
        if self.refresh_threshold_deg <= 0:
            raise ValueError("play.refresh_threshold_deg must be positive")
        if self.send_lead_ms < 0 or self.gather_ms < 0:
            raise ValueError("play.send_lead_ms and play.gather_ms must not be negative")
        if not 0 <= self.preview_step < PREVIEW_STEPS:
            raise ValueError(f"play.preview_step is 0 (none) or a step 1-{PREVIEW_STEPS - 1}")

    @property
    def mapping(self) -> str:
        """The input mapping in use: ``input_mapping``, by default ``front`` with ``refresh`` (a
        refresh cuts the block on screen again, so the turn of each cut is done in its first
        frames) and ``latest`` without (a block's turn spread over the block: a turn that steps
        from block to block shows as a pop)."""
        if self.input_mapping is not None:
            return self.input_mapping
        return "front" if self.refresh else "latest"

    @property
    def gains(self) -> tuple[float, float]:
        """The gains on the turn the generator is fed, ``(pitch, yaw)``: ``pitch_gain`` and
        ``yaw_gain``, by default :data:`REFRESH_YAW_GAIN` on the yaw with ``refresh`` (each
        refresh's turn is done in a few frames, which the view follows that far short) and 1
        otherwise, the turn as made."""
        yaw = REFRESH_YAW_GAIN if self.refresh else 1.0
        return (
            1.0 if self.pitch_gain is None else self.pitch_gain,
            yaw if self.yaw_gain is None else self.yaw_gain,
        )


@dataclass
class DemoConfig:
    """The demo's settings."""

    #: ``library.json`` of the round starts (``tools/build_demo_library.py``); ``None``: the
    #: synthetic arena.
    library: str | None = None
    scene_state: LiveSceneStateConfig = field(default_factory=LiveSceneStateConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    play: PlayConfig = field(default_factory=PlayConfig)

    def __post_init__(self) -> None:
        if "scene_state.bound" in self.client.worldcast.overrides:
            raise ValueError(
                "client.worldcast.overrides: the bound of a live session is the demo's"
                " scene_state.bound, which the coordinator keeps to as well"
            )
        if "live.refresh" in self.client.worldcast.overrides:
            raise ValueError(
                "client.worldcast.overrides: a worker refreshes its blocks with play.refresh,"
                " which sets the client's live.refresh"
            )


def load_config(
    paths: Sequence[str | os.PathLike[str]] = (), overrides: Mapping[str, Any] | None = None
) -> DemoConfig:
    """The defaults, the YAML ``paths`` merged over them in order, then the dotted ``overrides``."""
    return load(paths, overrides, lambda data: build(DemoConfig, data))
