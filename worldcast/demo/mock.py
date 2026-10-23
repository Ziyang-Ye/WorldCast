"""The mock client: a stand-in for the WorldCast client that needs no GPU.

It keeps the real client's timing shape and traffic, plays pre-rendered frames of the seat (or draws
a grid world), places the player with a fake state model and shows the other players as the
WorldCast client does, live (or, as ``gt``, on their recordings), so the whole demo runs on a
laptop.
"""

import bisect
import copy
import math
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from worldcast.data.controls import CONTROL_BUTTONS
from worldcast.data.game import (
    ASPECT_TAN_RATIO,
    EYE_HEIGHT,
    GRAVITY_U_PER_S2,
    HFOV_DEGREES,
    PITCH_MAX_DEGREES,
    PITCH_MIN_DEGREES,
)
from worldcast.data.latents import (
    BLOCK,
    FPS,
    FRAME_SIZE,
    LATENT_CHANNELS,
    LATENT_HEIGHT,
    LATENT_WIDTH,
    VIDEO_FRAMES_PER_BLOCK,
    VIDEO_FRAMES_PER_LATENT,
)
from worldcast.engine.inference.field_rows import FieldRows

from .clients import (
    LIVE_STATE,
    MEMORY_ENTRY,
    ClientMessage,
    LiveControls,
    LiveFrame,
    LivePlayerState,
    Outbox,
    button_bits,
)
from .config import MockConfig
from .library import PLAYER_COLORS, Library, RoundStart, Seat

__all__ = ["MockClient"]

#: Bytes of a block's latents ``[4, 48, 24, 42]`` in float32, as the WorldCast client publishes them
#: (the paper counts half precision).
BLOCK_LATENT_BYTES = 4 * BLOCK * LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH
#: The mock's frames have the WorldCast client's size, pixels.
FRAME_HEIGHT, FRAME_WIDTH = FRAME_SIZE
#: Ground speeds, units/s: running, and the factors of walking (shift) and crouching.
RUN_SPEED, WALK_FACTOR, DUCK_FACTOR = 240.0, 0.52, 0.34
#: Velocity relaxation towards the wished velocity, 1/s.
ACCELERATION = 10.0
#: Jump speed, units/s.
JUMP_SPEED = 300.0
#: Integration substeps per frame.
SUBSTEPS = 4
#: A drawn player's height and half width, units, and the nearest depth at which one is drawn.
PLAYER_HEIGHT, PLAYER_HALF_WIDTH, NEAR_DEPTH = 72.0, 16.0, 24.0
#: The rows of another player's live states the mock keeps: two blocks of 16 frames.
PEER_ROWS = 2 * VIDEO_FRAMES_PER_BLOCK
_BUTTON = {name: i for i, name in enumerate(CONTROL_BUTTONS)}


class Kinematics:
    """The fake state model: a player's position from its controls, on ground without bounds."""

    def __init__(self, slot: int, spawn: Sequence[float]) -> None:
        x, y, z, yaw, pitch = (float(v) for v in spawn)
        self.state = LivePlayerState(slot=slot, t=0.0, x=x, y=y, z=z, yaw=yaw, pitch=pitch)
        self.ground, self.vz = z, 0.0

    def _move(self, h: float) -> None:
        """One substep of ground motion."""
        s = self.state
        s.x, s.y = s.x + s.vx * h, s.y + s.vy * h

    def advance(
        self, buttons: np.ndarray, turn: np.ndarray, t: float, dt: float
    ) -> LivePlayerState:
        """The state after one frame.

        Args:
            buttons (np.ndarray): ``[11]`` the frame's held buttons (``CONTROL_BUTTONS``).
            turn (np.ndarray): ``[2]`` degrees turned during the frame (pitch, yaw).
            t (float): room seconds of the frame.
            dt (float): the frame's length, s.
        """
        s = self.state
        s.yaw = (s.yaw + float(turn[1]) + 180.0) % 360.0 - 180.0
        s.pitch = float(np.clip(s.pitch + float(turn[0]), PITCH_MIN_DEGREES, PITCH_MAX_DEGREES))
        forward = float(buttons[_BUTTON["forward"]] - buttons[_BUTTON["back"]])
        right = float(buttons[_BUTTON["move_right"]] - buttons[_BUTTON["move_left"]])
        norm = math.hypot(forward, right)
        speed = RUN_SPEED * (WALK_FACTOR if buttons[_BUTTON["speed"]] else 1.0)
        speed *= DUCK_FACTOR if buttons[_BUTTON["duck"]] else 1.0
        cos, sin = math.cos(math.radians(s.yaw)), math.sin(math.radians(s.yaw))
        wish_x = wish_y = 0.0
        if norm > 0:
            f, r = forward / norm, right / norm
            wish_x, wish_y = speed * (f * cos + r * sin), speed * (f * sin - r * cos)
        h = dt / SUBSTEPS
        blend = 1.0 - math.exp(-ACCELERATION * h)
        for _ in range(SUBSTEPS):
            if s.z <= self.ground + 1e-3 and self.vz <= 0:  # on the ground
                s.vx += (wish_x - s.vx) * blend
                s.vy += (wish_y - s.vy) * blend
                if buttons[_BUTTON["jump"]]:
                    self.vz = JUMP_SPEED
            self.vz -= GRAVITY_U_PER_S2 * h
            self._move(h)
            s.z = max(self.ground, s.z + self.vz * h)
            if s.z <= self.ground:
                self.vz = 0.0
        s.t = float(t)
        return LivePlayerState(**vars(s))


def extrapolate(state: LivePlayerState, t: float, horizon: float) -> LivePlayerState:
    """``state`` moved on with its velocity to room time ``t``, for at most ``horizon`` seconds."""
    dt = float(np.clip(t - state.t, 0.0, horizon))
    return LivePlayerState(
        **{**vars(state), "t": t, "x": state.x + state.vx * dt, "y": state.y + state.vy * dt}
    )


class GridWorld:
    """The synthetic view: a dark sky, a lit grid floor, every other player a coloured pillar."""

    SCALE = 2  # the floor is drawn at half resolution, then upscaled
    SPACING = 128.0  # grid spacing, units
    FOG = 2600.0  # distance at which the floor fades out, units
    LINE_PX, LINE_GROWTH = 1.5, 0.02  # a grid line's width in pixels, and its growth per unit
    #: Per colour channel: the floor, its grid lines, the sky, the horizon's haze.
    PALETTE = ((12, 168, 4, 40), (13, 190, 6, 48), (18, 220, 14, 70))
    HAZE_FALLOFF, FLOOR_HAZE = 9.0, 0.5  # the haze's decay off the horizon; its share on the floor
    LABEL_COLOR, LABEL_SIZE, LABEL_GAP = "#f5f5f7", 13, 6  # a player's label: size and gap, pixels

    def __init__(self) -> None:
        w, h = FRAME_WIDTH // self.SCALE, FRAME_HEIGHT // self.SCALE
        # pixels per unit of tangent, across and up: the recordings' 16:9 camera over the frame
        across = math.tan(math.radians(HFOV_DEGREES) / 2)
        self.focal = (FRAME_WIDTH / 2 / across, FRAME_HEIGHT / 2 / (across * ASPECT_TAN_RATIO))
        jj, ii = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        self.a = (jj * self.SCALE - FRAME_WIDTH / 2) / self.focal[0]
        self.b = -(ii * self.SCALE - FRAME_HEIGHT / 2) / self.focal[1]
        self.font = ImageFont.load_default(size=self.LABEL_SIZE)

    @staticmethod
    def axes(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The camera's forward, right and up axes."""
        psi, theta = math.radians(yaw), math.radians(pitch)
        f = np.array(
            [math.cos(psi) * math.cos(theta), math.sin(psi) * math.cos(theta), -math.sin(theta)]
        )
        r = np.array([math.sin(psi), -math.cos(psi), 0.0])
        return f, r, np.cross(r, f)

    def place(
        self, camera: LivePlayerState, players: list[LivePlayerState]
    ) -> list[tuple[LivePlayerState, float, float, float]]:
        """The players in ``camera``'s view.

        Returns:
            list[tuple[LivePlayerState, float, float, float]]: per player, the player with
            ``drawn`` set (in front of the camera and within the frame's width), and where its
            feet are: ``u`` and ``v``, fractions of the frame's width and height from its top left
            corner, and its depth, u.
        """
        f, r, up = self.axes(camera.yaw, camera.pitch)
        eye = np.array([camera.x, camera.y, camera.z + EYE_HEIGHT])
        placed = []
        for p in players:
            rel = np.array([p.x, p.y, p.z]) - eye
            depth = float(rel @ f)
            across, upward = (focal / max(depth, NEAR_DEPTH) for focal in self.focal)
            u = 0.5 + float(rel @ r) * across / FRAME_WIDTH
            v = 0.5 - float(rel @ up) * upward / FRAME_HEIGHT
            half = PLAYER_HALF_WIDTH * across / FRAME_WIDTH
            drawn = depth > NEAR_DEPTH and u + half > 0.0 and u - half < 1.0
            placed.append((replace(p, drawn=drawn), u, v, depth))
        return placed

    def render(
        self, view: LivePlayerState, placed: list[tuple[LivePlayerState, float, float, float]]
    ) -> np.ndarray:
        """The view of ``view`` as ``[H, W, 3]`` uint8: the floor it sees, and the players
        :meth:`place` drew, where it put them."""
        f, r, u = self.axes(view.yaw, view.pitch)
        eye = np.array([view.x, view.y, view.z + EYE_HEIGHT])
        dz = f[2] + self.a * r[2] + self.b * u[2]
        hit = dz < -1e-4
        dist = np.where(hit, eye[2] / np.maximum(-dz, 1e-4), 0.0)
        gx = eye[0] + dist * (f[0] + self.a * r[0] + self.b * u[0])
        gy = eye[1] + dist * (f[1] + self.a * r[1] + self.b * u[1])
        width = self.LINE_PX + dist * self.LINE_GROWTH
        edge = 0.5 - np.maximum(
            np.abs((gx / self.SPACING) % 1.0 - 0.5), np.abs((gy / self.SPACING) % 1.0 - 0.5)
        )
        line = np.clip(1.0 - edge * self.SPACING / width, 0.0, 1.0)
        fog = np.clip(1.0 - dist / self.FOG, 0.0, 1.0) * hit
        glow = np.exp(-np.abs(dz) * self.HAZE_FALLOFF)
        rgb = np.empty(dz.shape + (3,), np.float32)
        for c, (floor, grid, sky, haze) in enumerate(self.PALETTE):
            lit = floor * fog + grid * line * fog**3 + haze * glow * self.FLOOR_HAZE
            rgb[..., c] = np.where(hit, lit, sky + haze * glow)
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize(
            (FRAME_WIDTH, FRAME_HEIGHT), Image.BILINEAR
        )
        self._draw_players(ImageDraw.Draw(img), placed)
        return np.asarray(img)

    def _draw_players(
        self, draw: ImageDraw.ImageDraw, placed: list[tuple[LivePlayerState, float, float, float]]
    ) -> None:
        for p, u, v, depth in sorted(placed, key=lambda item: -item[3]):
            if not p.drawn:
                continue
            cx, foot = u * FRAME_WIDTH, v * FRAME_HEIGHT
            height = PLAYER_HEIGHT / depth * self.focal[1]
            half = PLAYER_HALF_WIDTH / depth * self.focal[0]
            box = [cx - half, foot - height, cx + half, foot]
            color = PLAYER_COLORS[p.slot % len(PLAYER_COLORS)]
            draw.rounded_rectangle(box, radius=max(1.0, half), fill=color)
            label = (cx, foot - height - self.LABEL_GAP)
            draw.text(label, f"P{p.slot + 1}", fill=self.LABEL_COLOR, anchor="ms", font=self.font)


class ClipPlayer:
    """Frames of a pre-rendered clip (a directory of JPEGs), looped."""

    def __init__(self, directory: Path) -> None:
        self.files = sorted(directory.glob("*.jpg"))
        if not self.files:
            raise FileNotFoundError(f"no .jpg frames in {directory}")

    def frame(self, index: int) -> np.ndarray:
        """Frame ``index`` of the looped clip, ``[H, W, 3]`` uint8."""
        with Image.open(self.files[index % len(self.files)]) as im:
            return np.asarray(im.convert("RGB"))


class MockClient(Outbox):
    """A stand-in client without a GPU.

    It keeps the real client's timing shape (the controls are taken, ``denoise_ms`` of fake
    denoising, then ``decode_ms`` per frame), integrates the player's controls into a position (run
    / walk / crouch speeds, jump and gravity) and shows the other players as the WorldCast client
    does under the ``player_state.source`` it stands in for (``client.mock.player_state_source``):

    * ``live``: the players are drawn into the camera the keys move. At each block's controls cut,
      before the fake denoising, it publishes the block's rows as its live state. Another player
      who plays is drawn at the newest of its rows at or before the frame's time, past them moved
      on with its velocity for at most a block. Nobody else is in the world: a seat nobody plays,
      and a seat before its player's first live state or after the player leaves, are hidden.
    * ``gt``: the players stand at their seats' spawns and are drawn into the camera of the
      player's own recording, its spawn, while the keys move the floor: every other player stays on
      its spot of the frame as the view moves, the drift of the WorldCast client's ``gt`` under
      live keys. Nothing positional is published.

    Per sixteen frames it publishes a fake memory entry of the real size, so the coordinator's
    relay carries the real traffic.

    With ``refresh`` (``play.refresh``) it refreshes its newest block as the WorldCast client
    does: :meth:`refresh` moves the player through the block again from where it started, on the
    new controls, waits out the fake denoising and yields the frames from the switch on; the
    block's live state leaves with the next block, with the newest controls. With a preview step it
    yields the frames of the switch's latent frame, marked, that share of the denoising in, and the
    version's own after it. It has one card.
    """

    blocks_in_flight = 1
    refresh_cards = 1
    card_freed = None

    def __init__(self, config: MockConfig, library: Library, *, refresh: bool = False) -> None:
        super().__init__()
        self.cfg, self.library = config, library
        self.frames_per_block = int(config.frames_per_block)
        self.slot, self.frame_index = -1, 0
        self.body: Kinematics | None = None
        self.recording: LivePlayerState | None = None
        self.clip: ClipPlayer | None = None
        self.seats: dict[int, Seat] = {}  # the other seats of the round start
        self._rows: dict[int, list[LivePlayerState]] = {}  # another live player's, by time
        self._grid = GridWorld()
        self.refreshes = bool(refresh)
        # the newest block while it may be refreshed: the body before it, its first frame and the
        # rows of its newest version
        self._open: tuple[Kinematics, int, list[LivePlayerState]] | None = None

    @property
    def can_refresh(self) -> bool:
        """Whether the newest block can be refreshed (``refresh``, a block stepped)."""
        return self._open is not None

    @property
    def refresh_card_free(self) -> bool:
        """Its one card is free whenever the worker asks: a refresh waits for the one before."""
        return True

    def warm_up(self) -> None:
        """Nothing: the mock's first blocks are as fast as its later ones."""

    def start(self, round_start: RoundStart, seat: Seat) -> None:
        """Place the player at the seat's spawn, with its clip."""
        self.stop()
        self._open = None
        self.slot, self.frame_index = seat.slot, 0
        self.body = Kinematics(seat.slot, seat.spawn)
        self.recording = LivePlayerState(seat.slot, 0.0, *seat.spawn)
        self.clip = ClipPlayer(self.library.path(seat.clip)) if seat.clip else None
        with self._lock:
            self.seats = {s.slot: s for s in round_start.seats if s.slot != seat.slot}

    def step(self, controls: Callable[[], LiveControls]) -> Iterator[LiveFrame]:
        """Take the controls and move the player through the block (under ``live``, publish the
        rows), wait out the fake denoising, then yield the block's frames; per sixteen frames a
        fake memory entry is queued. With ``refresh`` the rows of the block before leave first."""
        cfg, live = self.cfg, self.cfg.player_state_source == "live"
        self._publish_open()
        before = copy.deepcopy(self.body) if self.refreshes else None
        taken = controls()
        views = [
            self.body.advance(taken.buttons[j], taken.turn[j], float(taken.times[j]), 1 / FPS)
            for j in range(taken.num_frames)
        ]
        if self.refreshes:
            self._open = before, self.frame_index, views
        elif live:
            rows = {"rows": [view.wire() for view in views]}
            self.put(ClientMessage(self.slot, LIVE_STATE, rows, b""))
        time.sleep(
            max(0.0, cfg.denoise_ms + random.uniform(-cfg.jitter_ms, cfg.jitter_ms)) / 1000.0
        )
        for j, view in enumerate(views):
            started = time.perf_counter()
            frame = self._frame(self.frame_index, taken, j, view)
            self.frame_index += 1
            if self.frame_index % VIDEO_FRAMES_PER_BLOCK == 0:
                meta = {"frames": [self.frame_index - VIDEO_FRAMES_PER_BLOCK, self.frame_index]}
                entry = ClientMessage(self.slot, MEMORY_ENTRY, meta, bytes(BLOCK_LATENT_BYTES))
                self.put(entry)
            time.sleep(max(0.0, cfg.decode_ms / 1000.0 - (time.perf_counter() - started)))
            yield frame

    def refresh(
        self,
        controls: Callable[[], LiveControls],
        switch: Callable[[float], int],
        generated: Callable[[], None] | None = None,
        preview_step: int = 0,
    ) -> Iterator[LiveFrame]:
        """Move the player through the newest block again from where it started, on new
        controls, wait out the fake denoising (then ``generated()``), then yield the frames from
        the first one ``switch`` names on (``refresh``); with ``preview_step`` the frames of the
        switch's latent frame, marked, ``preview_step`` quarters of the denoising in."""
        if self._open is None:
            raise RuntimeError("the mock refreshes its newest block, with refresh on")
        cfg = self.cfg
        before, first, _ = self._open
        body = copy.deepcopy(before)
        taken = controls()
        views = [
            body.advance(taken.buttons[j], taken.turn[j], float(taken.times[j]), 1 / FPS)
            for j in range(taken.num_frames)
        ]
        self.body, self._open = body, (before, first, views)
        denoise_s = max(0.0, cfg.denoise_ms + random.uniform(-cfg.jitter_ms, cfg.jitter_ms)) / 1e3
        decode_s = cfg.decode_ms / 1000.0 * VIDEO_FRAMES_PER_LATENT  # four frames at a time
        if preview_step:
            time.sleep(denoise_s * preview_step / 4)
            denoise_s *= 1 - preview_step / 4
            start = max(switch(decode_s) - first, 0)
            for j in range(start, min(start + VIDEO_FRAMES_PER_LATENT, len(views))):
                yield replace(self._frame(first + j, taken, j, views[j]), preview=True)
        time.sleep(denoise_s)
        if generated is not None:
            generated()
        for j in range(max(switch(decode_s) - first, 0), len(views)):
            started = time.perf_counter()
            frame = self._frame(first + j, taken, j, views[j])
            time.sleep(max(0.0, cfg.decode_ms / 1000.0 - (time.perf_counter() - started)))
            yield frame

    def _frame(self, index: int, taken: LiveControls, j: int, view: LivePlayerState) -> LiveFrame:
        """Frame ``index``, row ``j`` of controls ``taken``, at the player's ``view``."""
        live = self.cfg.player_state_source == "live"
        camera = view if live else replace(self.recording, t=view.t)
        placed = self._grid.place(camera, self._others(view.t, live))
        rgb = self.clip.frame(index) if self.clip else self._grid.render(view, placed)
        return LiveFrame(
            index=index,
            rgb=rgb,
            state=camera,
            turned=tuple(taken.turned[j].tolist()),
            camera_pitch=view.pitch,
            others=[p for p, *_ in placed],
            input_seq=int(taken.input_seq[j]),
            markers=[(p.slot, u, v, depth) for p, u, v, depth in placed if depth > NEAR_DEPTH],
            status=self.cfg.player_state_source,
            buttons=button_bits(taken.buttons[j]),
            weapon=int(taken.weapon[j]),
        )

    def _publish_open(self) -> None:
        """With ``refresh``, under ``live``: the newest block's live state, of its newest
        version."""
        if self._open is None:
            return
        _, _, views = self._open
        self._open = None
        if self.cfg.player_state_source == "live":
            rows = {"rows": [view.wire() for view in views]}
            self.put(ClientMessage(self.slot, LIVE_STATE, rows, b""))

    def _others(self, t: float, live: bool) -> list[LivePlayerState]:
        """The other players at room time ``t``, those hidden left out (the class docstring)."""
        horizon = self.frames_per_block / FPS
        players = []
        with self._lock:
            for slot, seat in sorted(self.seats.items()):
                if not live:
                    players.append(LivePlayerState(slot, t, *seat.spawn, src=FieldRows.RECORDED))
                elif self._rows.get(slot):
                    players.append(row_at(self._rows[slot], t, horizon))
        return players

    def stop(self) -> None:
        """Forget the other players; the messages not yet taken stay queued for the worker (with
        ``refresh``, the newest block's live state among them)."""
        self._publish_open()
        with self._lock:
            self._rows.clear()

    def receive_message(self, message: ClientMessage) -> None:
        """Keep the newest rows of another player's live states; other kinds are dropped."""
        if message.kind != LIVE_STATE:
            return
        rows = [LivePlayerState.from_wire(wire) for wire in message.meta["rows"]]
        with self._lock:
            live = (replace(row, src=FieldRows.LIVE) for row in rows)
            kept = [*self._rows.get(message.slot, ()), *live]
            self._rows[message.slot] = sorted(kept, key=lambda row: row.t)[-PEER_ROWS:]

    def remove_player(self, slot: int) -> None:
        """The player of ``slot`` left: its seat is hidden until a player takes it again and
        publishes."""
        with self._lock:
            self._rows.pop(int(slot), None)


def row_at(rows: list[LivePlayerState], t: float, horizon: float) -> LivePlayerState:
    """Another player at room time ``t`` from its rows, by time: the newest at or before ``t``
    (before them all, the oldest); past the newest, moved on with its velocity for at most
    ``horizon`` s."""
    newest = bisect.bisect_right([row.t for row in rows], t) - 1
    if newest == len(rows) - 1:
        return extrapolate(rows[-1], t, horizon)
    return replace(rows[max(newest, 0)], t=t)
