"""The WorldCast client of one GPU behind :class:`worldcast.demo.clients.ClientProtocol`."""

import logging
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator

import numpy as np
import torch

from worldcast.config.inference import InferenceConfig, SceneStateConfig, load_config
from worldcast.config.loader import require_set
from worldcast.data import (
    VIDEO_FRAMES_PER_BLOCK,
    VIDEO_FRAMES_PER_LATENT,
    MediaIndex,
    RoundIndexRow,
)
from worldcast.data.controls import CONTROL_BUTTONS, encode_turn, encode_weapon
from worldcast.data.latents import BLOCK, FIRST_TARGET, FPS
from worldcast.data.recordings import RoundKey
from worldcast.engine.inference.client import Client, Frame, Switch
from worldcast.engine.inference.controls import LiveBlockControls
from worldcast.engine.inference.field_rows import FieldRows
from worldcast.engine.inference.live_rollout import live_round_latents
from worldcast.engine.inference.messages import MessagePart
from worldcast.engine.inference.serving import LOW_LATENCY_ATTENTION
from worldcast.engine.inference.world_state import DONE_OK
from worldcast.player_state.projection import project_players
from worldcast.sampling.window import KV_CACHE_LATENTS, WindowLayout

from .clients import ClientMessage, LiveControls, LiveFrame, LivePlayerState, Outbox, button_bits
from .config import WorldCastClientConfig
from .library import Library, RoundStart, Seat, players_in_first_frame, zooming_weapons

__all__ = [
    "REFRESH_WARM_UP_ROWS",
    "REFRESH_WARM_UP_WINDOWS",
    "WARM_UP_WINDOWS",
    "WINDOW_SHAPES",
    "WorldCastClient",
    "field_markers",
    "field_states",
    "index_row",
    "inference_config",
    "window_shapes",
]

log = logging.getLogger(__name__)

#: Longest wait, s, for the client to take a block's controls.
CONTROLS_TIMEOUT_S = 120.0
#: The shapes of the window of a block, in latent frames: without a retrieved memory entry (17)
#: and with one (21).
WINDOW_SHAPES = tuple(WindowLayout(with_memory=memory).num_frames for memory in (False, True))


def window_shapes(scene_state: bool) -> tuple[int, ...]:
    """The shapes of the window of a block of the generator of ``model.scene_state``, in latent
    frames: :data:`WINDOW_SHAPES`; without scene state those of the ordinary window as it grows,
    29, 33, 37 and 41 (:meth:`~worldcast.sampling.window.WindowLayout.ordinary`)."""
    if scene_state:
        return WINDOW_SHAPES
    starts = range(FIRST_TARGET, KV_CACHE_LATENTS + 1, BLOCK)
    return tuple(sorted({WindowLayout.ordinary(s).num_frames for s in starts}))


#: The warm-up ends once each window shape has run, or after this many window blocks (about 16 s
#: on an H20): the generator is warm after the first, whose calls repeat those of latents 1-24.
WARM_UP_WINDOWS = 16
#: With ``live.refresh`` the warm-up also plays the seat live through the first six blocks and this
#: many window blocks, each refreshed (:meth:`WorldCastClient.warm_up`).
REFRESH_WARM_UP_WINDOWS = 2
#: The rows a warm-up refresh switches at: in the block's second latent frame (the decoder resumes
#: from a snapshot inside the block), and at its start (the whole block again).
REFRESH_WARM_UP_ROWS = (VIDEO_FRAMES_PER_LATENT + 2, 0)


class WorldCastClient(Outbox):
    """The WorldCast client of one GPU; the models load once, in ``__init__``.

    It plays the player's live controls (``player_state.source: live``,
    :class:`~worldcast.engine.inference.live_rollout.LiveRollout`). The clients of a room exchange
    the parts of the paper's messages (:class:`worldcast.engine.inference.messages.MessagePart`) as
    :class:`ClientMessage` payloads, which the coordinator relays. A frame shows every player as
    the client's player state field read it at the frame's latent frame
    (:meth:`~worldcast.engine.inference.client.Client.field_rows`): :func:`field_states` and
    :func:`field_markers`.

    Args:
        config (WorldCastClientConfig): the inference YAMLs, their overrides, how it is served.
        scene_state (SceneStateConfig): the scene state of the session (its bound B).
        library (Library): the round starts the worker serves (:meth:`warm_up` replays the first).
        refresh (bool): the worker refreshes the block on screen (``play.refresh``): the client
            runs ``live.refresh`` (:meth:`refresh`).

    Raises:
        ValueError: the inference config's ``player_state.source`` is not ``live``.
    """

    def __init__(
        self,
        config: WorldCastClientConfig,
        scene_state: SceneStateConfig,
        library: Library,
        *,
        refresh: bool = False,
    ) -> None:
        super().__init__()
        self.frames_per_block = VIDEO_FRAMES_PER_BLOCK
        self.cfg = inference_config(config, scene_state, refresh=refresh)
        source = self.cfg.player_state.source
        if source != "live":
            raise ValueError(
                f"client.worldcast: the demo plays player_state.source live, not {source} (under"
                " live keys gt keeps the camera on the seat's recording, so the world drifts with"
                " the keys, and predicted reads the recorded controls); give player_state.source:"
                " live in client.worldcast.overrides, as configs/demo/worldcast.yaml does"
            )
        self.library = library
        serving = config.serving_options()
        self.client = Client(self.cfg, serving, on_message=self._sent)
        self.blocks_in_flight = 1 if serving.decoder_device is None else 2
        self.slot = -1
        self._media: dict[int, str] = {}  # player slot -> media id, of the session's round

    def warm_up(self) -> None:
        """Generate a replay session before the worker's first player: the first seat of the
        library's first round start, on its recorded player states and controls
        (``player_state.source: gt``, whatever the source of the play) on a round of the first six
        blocks and :data:`WARM_UP_WINDOWS` window blocks (the play's round has one), until each of
        the :data:`WINDOW_SHAPES` has run, for at most :data:`WARM_UP_WINDOWS` window blocks or to
        the round's end. The recorded camera moves, so some blocks retrieve a memory entry and some
        do not. Without scene state (``model.scene_state: false``) the shapes are those of the
        ordinary window as it grows (:func:`window_shapes`).

        What a player's first blocks would otherwise wait for happens here: the generator compiles
        each of its calls and captures its CUDA graph when the call recurs. Latents 1-24 make every
        call a window makes, at window positions 1-17 (each runs five times in its block), and the
        first window block makes the first frame's again. The session also runs the decoder, the
        scene state's writes and retrievals, and both kinds of window. A window shape that did not
        run is named in a warning. The session's messages are dropped.

        With ``live.refresh`` the seat is then played live (:meth:`_warm_up_refreshes`): what a
        refresh runs, and the live play around it, are warm for the first player too. A replica of
        the generator (``refresh_devices``) that a check of either part finds otherwise than the
        client's card, or failing, is let go (``Client.disable_replicas``): the client then plays
        on its own card, its live part warmed up again there.

        Raises:
            ValueError: the round start names no recorded round.
            KeyError: the seat's player has no recording in the round.
        """
        round_start = next(iter(self.library.round_starts.values()))
        seat = round_start.seats[0]
        row = index_row(round_start, seat.slot, MediaIndex.load(self.cfg.paths.media_index))
        latents = FIRST_TARGET + BLOCK * WARM_UP_WINDOWS  # the first six blocks, then the windows
        replay = self.cfg.with_overrides(
            {"player_state.source": "gt", "run.latent_frames": latents}
        )
        play, self.client.cfg = self.client.cfg, replay  # a rollout reads it when it starts
        shapes, t0 = Counter(), time.monotonic()
        wanted = window_shapes(self.cfg.model.scene_state)
        try:
            self.client.start(row)
            while not self.client.finished:
                if all(shapes[n] for n in wanted) or shapes.total() == WARM_UP_WINDOWS:
                    break
                deque(self.client.step(), maxlen=0)  # the recorded controls
                read = self.client.records[-1].read
                if read is not None:
                    shapes[read.window_frames] += 1
        finally:
            self.client.stop()
            self.client.cfg = play
            self.take_messages()
        log.info(
            "warm-up on %s, player %d: %d blocks in %.1f s; window blocks of %s latent frames: %s",
            round_start.id,
            seat.slot,
            len(self.client.records),
            time.monotonic() - t0,
            " / ".join(map(str, wanted)),
            " / ".join(str(shapes[n]) for n in wanted),
        )
        for n in wanted:
            if not shapes[n]:
                log.warning(
                    "warm-up on %s: no window of %d latent frames ran; its generator calls are "
                    "warm, the rest of a session's first one is not",
                    round_start.id,
                    n,
                )
        self._check_replicas()
        if self.cfg.live.refresh:
            try:
                self._warm_up_refreshes(row, seat)
            except Exception:
                if not self.client.replicas:
                    raise
                log.exception("warm-up: the refreshes on the cards failed")
                self.client.replicas_equal.append(False)
                self._check_replicas()
                self._warm_up_refreshes(row, seat)
            self._check_replicas()

    def _check_replicas(self) -> None:
        """Let the generator's replicas go if a check of the warm-up found one otherwise than the
        client's card, or failing: the client then refreshes on its own card alone."""
        checks = self.client.replicas_equal
        if self.client.replicas and not all(checks):
            log.warning(
                "warm-up: %d of %d checks of the generator's replicas found other results than"
                " the client's card, or failed (refresh_devices %s)",
                checks.count(False),
                len(checks),
                ", ".join(self.client.serving.refresh_devices),
            )
            self.client.disable_replicas()

    def _warm_up_refreshes(self, row: RoundIndexRow, seat: Seat) -> None:
        """The warm-up of ``live.refresh``: the seat played live, walking and turning, through the
        first six blocks and :data:`REFRESH_WARM_UP_WINDOWS` window blocks, each window block
        refreshed at each of :data:`REFRESH_WARM_UP_ROWS` on other controls and then on its own:
        the live cut, its save and restore, a refresh's denoising on the CUDA graphs a block's
        takes, the decoder's snapshot and its resumption inside a block and at its start (in the
        layout it decodes in), the page's rows, the commit of a refreshed block. The session's
        messages are dropped. With replicas (``refresh_devices``) the refreshes go to the cards in
        turn, and the refresh on the block's own controls, decoded whole, gives the block's first
        frames again, byte for byte, where the card runs the client's kernels
        (``Client.replicas_equal``)."""
        weapon = encode_weapon(seat.loadout[0]) if seat.loadout else 0
        blocks, refreshes, t0 = FIRST_TARGET // BLOCK + REFRESH_WARM_UP_WINDOWS, 0, time.monotonic()
        cards = self.client.refresh_cards
        try:
            self.client.start(row)
            for k in range(blocks):
                frames = list(self.client.step(lambda k=k: _warm_up_controls(k, weapon, 12.0)))
                if not self.client.can_refresh:
                    continue
                first = frames[0].index
                for row_of_switch, yaw in zip(REFRESH_WARM_UP_ROWS, (-30.0, 12.0)):
                    on = {"card": (refreshes + k) % cards} if cards > 1 else {}
                    again = self.client.refresh(
                        lambda k=k, yaw=yaw: _warm_up_controls(k, weapon, yaw),
                        lambda seconds, at=first + row_of_switch: at,
                        **on,
                    )
                    images = [frame.image for frame in again]
                    refreshes += 1
                if cards > 1:  # the own controls from the block's start: its first frames
                    same = len(images) == len(frames) and all(
                        np.array_equal(a, b.image) for a, b in zip(images, frames)
                    )
                    self.client.replicas_equal.append(same)
        finally:
            self.client.stop()
            self.take_messages()
        log.info(
            "warm-up of refresh on player %d: %d blocks, %d refreshes in %.1f s",
            seat.slot,
            blocks,
            refreshes,
            time.monotonic() - t0,
        )

    def start(self, round_start: RoundStart, seat: Seat) -> None:
        """Start the client on the round-index row of the seat's player (:func:`index_row`). The
        other seats are nowhere until their players' first live states.

        Only a seat whose first frame shows no other player is played: the window of every block
        attends that frame, in which the player state field says nobody is there. The library
        offers such seats (:func:`~worldcast.demo.library.players_in_first_frame`); the labels are
        read again here, since a library built another way would put the people of a recorded
        round in the world's first frame without a word. Nor is a seat whose loadout holds a weapon
        that zooms: live play does not track the scope, and the generator draws a black scope view
        from the first frame when the scope is unknown for the weapon held.

        Raises:
            ValueError: the round start names no recorded round, the loadout of the seat holds a
                weapon that zooms, or the first frame of the seat may show another player.
            KeyError: the seat's player has no recording in the round.
        """
        if zooming := zooming_weapons(seat.loadout):
            raise ValueError(
                f"seat {seat.slot} of {round_start.id}: its loadout holds {', '.join(zooming)},"
                " which zooms, and live play does not track the scope; rebuild the library with"
                " tools/build_demo_library.py (it leaves such seats out)"
            )
        media_index = MediaIndex.load(self.cfg.paths.media_index)
        row = index_row(round_start, seat.slot, media_index)
        require_set(self.cfg.paths, "paths", ["visibility_label_root"])
        seen = players_in_first_frame(
            self.cfg.paths.visibility_label_root, row.media_id, seat.slot, row.start_frame
        )
        if seen:
            raise ValueError(
                f"seat {seat.slot} of {round_start.id}: players {', '.join(map(str, seen))} may be"
                " seen in its first frame, which live play does not show them in; rebuild the"
                " library with tools/build_demo_library.py (it offers only seats that show"
                " nobody) from this paths.visibility_label_root"
            )
        self.slot = seat.slot
        self._media = {s: m.media_id for s, m in media_index.slots(_round_key(round_start)).items()}
        self.client.start(row)

    def start_row(self, row: RoundIndexRow) -> None:
        """Start on a round-index row as it is, without the library's rules on its seat
        (``tools/bench_live.py`` plays a client of an example case so)."""
        self.slot = row.player_slot
        key = (row.match_id, row.map_name, row.round)
        media_index = MediaIndex.load(self.cfg.paths.media_index)
        self._media = {s: m.media_id for s, m in media_index.slots(key).items()}
        self.client.start(row)

    @property
    def can_refresh(self) -> bool:
        """Whether the newest block can be refreshed (:meth:`refresh`): with ``refresh``, a block
        after the first six."""
        return self.client.can_refresh

    @property
    def refresh_cards(self) -> int:
        """The cards a refresh may run on: its own and its replicas' (``refresh_devices``)."""
        return self.client.refresh_cards

    @property
    def refresh_card_free(self) -> bool:
        """Whether one of its cards is free (``refresh_devices``): a refresh asked for now starts at
        once."""
        return self.client.refresh_card_free

    @property
    def card_freed(self) -> Callable[[], None] | None:
        """Called from any thread when one of its cards has no job left."""
        return self.client.card_freed

    @card_freed.setter
    def card_freed(self, freed: Callable[[], None] | None) -> None:
        self.client.card_freed = freed

    def refresh(
        self,
        controls: Callable[[], LiveControls],
        switch: Switch,
        generated: Callable[[], None] | None = None,
        preview_step: int = 0,
    ) -> Iterator[LiveFrame]:
        """Generate the newest block again from newer controls
        (:meth:`~worldcast.engine.inference.client.Client.refresh`, on a free card); yield its
        frames from the first one ``switch`` says has not been sent, as :meth:`step` yields a
        block's, decoded as the iterator is taken (after its preview's, with ``preview_step``);
        ``generated()`` once its latents are ready. The next :meth:`step` commits the newest
        version."""
        taken: list[LiveControls] = []
        ready = threading.Event()

        def take() -> LiveControls:
            taken.append(controls())
            ready.set()
            return taken[0]

        frames = self.client.refresh(take, switch, generated, preview_step=preview_step)
        return self._frames(frames, taken, ready)

    def step(self, controls: Callable[[], LiveControls]) -> Iterator[LiveFrame]:
        """Request the next block; each frame carries the players as the client's player state
        field read them at its latent frame, at the room time of its row of the controls, how far
        the player had turned through that row and the camera's pitch at the frame itself
        (:meth:`~worldcast.engine.inference.client.Client.camera_pitch`). The block is requested
        by the call, its frames are decoded as the iterator is taken; with ``refresh``, a refresh
        delivered while they are can take over (a frame may come again, of a newer version)."""
        taken: list[LiveControls] = []
        ready = threading.Event()

        def take() -> LiveControls:
            taken.append(controls())
            ready.set()
            return taken[0]

        return self._frames(self.client.step(take), taken, ready)

    def _frames(
        self, frames: Iterator[Frame], taken: list[LiveControls], ready: threading.Event
    ) -> Iterator[LiveFrame]:
        read: tuple[tuple[int, int], FieldRows, list] | None = None
        for frame in frames:
            # a frame of a block that may be refreshed comes with its version's controls (a
            # refresh's can come from the step's iterator); a session's first frame (latent 0) can
            # be decoded before its block takes the controls
            block = frame.controls
            if block is None:
                if not ready.wait(CONTROLS_TIMEOUT_S):
                    raise RuntimeError("the client did not take the block's controls")
                block = taken[0]
            row = max(frame.index - 1, 0) % VIDEO_FRAMES_PER_BLOCK  # the first frame: the first row
            turned = block.turned[row]
            if frame.index == 0:  # the round's recorded first frame: none of the block's turn
                turned = turned - block.turn[0]
            # video frames 4f-3 .. 4f are latent frame f
            latent = -(-frame.index // VIDEO_FRAMES_PER_LATENT)
            if read is None or read[0] != (latent, frame.version):
                rows = self.client.field_rows(latent) if frame.rows is None else frame.rows
                read = (latent, frame.version), rows, field_markers(rows)
            _, rows, markers = read
            own, others = field_states(rows, float(block.times[row]))
            pitch = frame.camera_pitch
            yield LiveFrame(
                index=frame.index,
                rgb=frame.image,
                state=own,
                turned=tuple(turned.tolist()),
                camera_pitch=self.client.camera_pitch(frame.index) if pitch is None else pitch,
                others=others,
                input_seq=int(block.input_seq[row]),
                markers=markers,
                status=rows.status,
                version=frame.version,
                buttons=button_bits(block.buttons[row]),
                weapon=int(block.weapon[row]),
                preview=frame.preview,
            )

    def stop(self) -> None:
        """Stop the client; its ``done`` part stays queued for the room's other clients."""
        self.client.stop()

    def _sent(self, part: MessagePart) -> None:
        """A message part of this client (from its generation thread): queue it."""
        self.put(ClientMessage(self.slot, part.kind, {}, part.encode()))

    def receive_message(self, message: ClientMessage) -> None:
        """Hand another client's message part to the client."""
        self.client.receive(MessagePart.decode(message.payload))

    def remove_player(self, slot: int) -> None:
        """The player of ``slot`` left: the client takes in a ``done`` (``ok``) of its seat, so it
        is hidden from the next block on, whether or not its own ``done`` arrives."""
        self.client.receive(MessagePart(self._media[int(slot)], "done", status=DONE_OK))


def _warm_up_controls(k: int, weapon: int, yaw: float) -> LiveBlockControls:
    """The ``k``-th block's controls of the refresh warm-up: forward held, ``yaw`` degrees turned
    over the block, ``weapon`` in hand, at the session times of its video frames."""
    buttons = np.zeros((VIDEO_FRAMES_PER_BLOCK, len(CONTROL_BUTTONS)), np.float32)
    buttons[:, CONTROL_BUTTONS.index("forward")] = 1.0
    turn = np.zeros((VIDEO_FRAMES_PER_BLOCK, 2), np.float32)
    turn[:, 1] = yaw / VIDEO_FRAMES_PER_BLOCK
    return LiveBlockControls(
        buttons=buttons,
        view_deltas=np.stack([encode_turn(row) for row in turn]),
        weapon=np.full(VIDEO_FRAMES_PER_BLOCK, weapon, np.int64),
        turn=turn,
        times=(VIDEO_FRAMES_PER_BLOCK * k + 1 + np.arange(VIDEO_FRAMES_PER_BLOCK)) / FPS,
    )


def inference_config(
    config: WorldCastClientConfig, scene_state: SceneStateConfig, *, refresh: bool = False
) -> InferenceConfig:
    """The inference config a worker's client runs: the YAMLs of ``config.configs`` with the
    settings of live play over them, the one place the demo's inference config is made.

    Over the inference defaults and the YAMLs the demo sets, in this order: FlashAttention-3
    (:data:`~worldcast.engine.inference.serving.LOW_LATENCY_ATTENTION`) and the round a live client
    of the generator (``model.scene_state``) keeps
    (:func:`~worldcast.engine.inference.live_rollout.live_round_latents`), then
    ``config.overrides`` (``configs/demo/worldcast.yaml``: the live source, the idle context), then
    the demo's bound B and, with ``refresh`` (``play.refresh``), ``live.refresh``. Every other
    key is the evaluation's.

    Args:
        config (WorldCastClientConfig): the inference YAMLs and their overrides.
        scene_state (SceneStateConfig): the scene state of the session (its bound B).
        refresh (bool): the worker refreshes the block on screen.
    """
    overrides = {
        "model.attention": LOW_LATENCY_ATTENTION,
        **config.overrides,
        "scene_state.bound": scene_state.bound,
    }
    if refresh:
        overrides["live.refresh"] = True
    cfg = load_config(config.configs, overrides)
    if "run.latent_frames" in config.overrides:
        return cfg
    return cfg.with_overrides({"run.latent_frames": live_round_latents(cfg.model.scene_state)})


def field_states(rows: FieldRows, t: float) -> tuple[LivePlayerState, list[LivePlayerState]]:
    """The players the player state field read, as a frame's header carries them at room time ``t``.

    Args:
        rows (FieldRows): the rows of one latent frame.
        t (float): room seconds.

    Returns:
        tuple[LivePlayerState, list[LivePlayerState]]: the client's own state, and the other
        players' except the hidden ones, by slot.
    """
    own, others = None, []
    for slot, source in enumerate(rows.source.tolist()):
        if source == FieldRows.HIDDEN:
            continue
        x, y, z, yaw, pitch, alive = rows.states[slot].tolist()
        drawn = None if rows.drawn is None or source == FieldRows.OWN else bool(rows.drawn[slot])
        state = LivePlayerState(slot, t, x, y, z, yaw, pitch, alive > 0.5, src=source, drawn=drawn)
        if source == FieldRows.OWN:
            own = state
        else:
            others.append(state)
    return own, others


def field_markers(rows: FieldRows) -> list[tuple[int, float, float, float]]:
    """Where the player state field puts every other player shown in front of the client's camera
    (:func:`~worldcast.player_state.projection.project_players` onto a 1 x 1 grid, from the own
    row's position and angles): ``(slot, u, v, depth)``, ``u`` and ``v`` fractions of the frame's
    width and height from its top left corner, the depth in u."""
    source = rows.source.tolist()
    shown = [s for s, src in enumerate(source) if src not in (FieldRows.OWN, FieldRows.HIDDEN)]
    if not shown:
        return []
    states = torch.from_numpy(rows.states).float()
    own = states[source.index(FieldRows.OWN)]
    uv, _, in_front, depth = project_players(
        states[shown, :3][None, None],
        own[:3][None, None],
        own[3].view(1, 1),
        own[4].view(1, 1),
        grid_h=1,
        grid_w=1,
    )
    return [
        (slot, float(uv[0, 0, i, 0]), float(uv[0, 0, i, 1]), float(depth[0, 0, i]))
        for i, slot in enumerate(shown)
        if in_front[0, 0, i]
    ]


def _round_key(round_start: RoundStart) -> RoundKey:
    """The recorded round of a round start, as the media index keys it.

    Raises:
        ValueError: the round start names no recorded round.
    """
    if round_start.match_id is None or round_start.round is None:
        raise ValueError(f"round start {round_start.id} names no recorded round (match_id, round)")
    return round_start.match_id, round_start.map, round_start.round


def index_row(round_start: RoundStart, slot: int, media_index: MediaIndex) -> RoundIndexRow:
    """The round-index row of the client of player ``slot``. The other players of a room are
    found by what they publish, not by the row.

    Raises:
        ValueError: the round start names no recorded round.
        KeyError: the player has no recording in the round.
    """
    key = _round_key(round_start)
    slots = media_index.slots(key)
    if slot not in slots:
        raise KeyError(f"player slot {slot} has no recording in round {key}")
    return RoundIndexRow(
        media_id=slots[slot].media_id,
        start_frame=round_start.start_frame,
        match_id=round_start.match_id,
        round=round_start.round,
        map_name=round_start.map,
        player_slot=slot,
    )
