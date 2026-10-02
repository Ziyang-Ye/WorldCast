"""A stand-in engine for laptops: plays pre-rendered frames (or draws a grid world) and fakes the state model.

It keeps the real engine's timing shape (the controls are taken, ``step_ms`` of fake denoising, then ``decode_ms``
per frame), integrates the player's controls into a position (run / walk / crouch speeds, jump and gravity, inside
the walkable area of the round's radar image), extrapolates the peers' published states over at most one block, and
publishes fake scene-state blocks of the real size, so the coordinator's relay carries the real traffic.
"""

import math
import random
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demo.actions import NUM_BUTTONS, PITCH_LIMIT
from demo.config import MockConfig
from demo.engine import BlockActions, Frame, PeerMessage, PlayerState
from demo.library import Library, Radar, RoundStart, Seat
from worldcast.data.actions import PAPER_ACTION_BUTTONS

FRAME_W, FRAME_H = 672, 384
HFOV_DEGREES = 106.26
EYE_HEIGHT = 64.0
#: Ground speeds, units/s: running, and the factors of walking (shift) and crouching.
RUN_SPEED, WALK_FACTOR, DUCK_FACTOR = 240.0, 0.52, 0.34
ACCELERATION = 10.0  # 1/s, velocity relaxation towards the wished velocity
JUMP_SPEED, GRAVITY = 300.0, 800.0
SUBSTEPS = 4
#: Seat colours (the project page's client colours, then Apple system colours); the browser uses the same list.
SEAT_COLORS = (
    "#E67F27",
    "#3D94E8",
    "#3FB576",
    "#BF5AF2",
    "#FF375F",
    "#64D2FF",
    "#FFD60A",
    "#AC8E68",
    "#5E5CE6",
    "#30D158",
)
_B = {name: i for i, name in enumerate(PAPER_ACTION_BUTTONS)}


class Walkable:
    """Where a player may stand: the opaque pixels of the round's radar image (drawn from the nav mesh)."""

    def __init__(self, radar: Radar, image: Path) -> None:
        with Image.open(image) as im:
            self.mask = np.asarray(im.convert("RGBA"))[..., 3] > 0
        self.radar = radar

    def __call__(self, x: float, y: float) -> bool:
        r = self.radar
        col, row = int(r.cx + r.scale * (x - r.x0)), int(r.cy + r.scale * (r.y0 - y))
        return (
            0 <= row < self.mask.shape[0]
            and 0 <= col < self.mask.shape[1]
            and bool(self.mask[row, col])
        )


class Kinematics:
    """The fake state model: one player's position from its own controls, kept on ``walkable`` ground if given."""

    def __init__(self, seat: int, spawn: Sequence[float], walkable: Walkable | None = None) -> None:
        x, y, z, yaw, pitch = (float(v) for v in spawn)
        self.state = PlayerState(seat=seat, t=0.0, x=x, y=y, z=z, yaw=yaw, pitch=pitch)
        self.ground = z
        self.vz = 0.0
        self.walkable = walkable if walkable is not None and walkable(x, y) else None

    def _move(self, h: float) -> None:
        """One substep of ground motion; against an edge, slide along it."""
        s, ok = self.state, self.walkable
        x, y = s.x + s.vx * h, s.y + s.vy * h
        if ok is None or ok(x, y):
            s.x, s.y = x, y
        elif ok(x, s.y):
            s.x, s.vy = x, 0.0
        elif ok(s.x, y):
            s.y, s.vx = y, 0.0
        else:
            s.vx = s.vy = 0.0

    def advance(
        self, buttons: np.ndarray, turn: np.ndarray, weapon: int, t: float, dt: float
    ) -> PlayerState:
        s = self.state
        s.yaw = (s.yaw + float(turn[1]) + 180.0) % 360.0 - 180.0
        s.pitch = float(np.clip(s.pitch + float(turn[0]), -PITCH_LIMIT, PITCH_LIMIT))
        forward = float(buttons[_B["forward"]] - buttons[_B["back"]])
        right = float(buttons[_B["move_right"]] - buttons[_B["move_left"]])
        norm = math.hypot(forward, right)
        speed = RUN_SPEED * (WALK_FACTOR if buttons[_B["speed"]] else 1.0)
        speed *= DUCK_FACTOR if buttons[_B["duck"]] else 1.0
        cos, sin = math.cos(math.radians(s.yaw)), math.sin(math.radians(s.yaw))
        wish_x = wish_y = 0.0
        if norm > 0:
            f, r = forward / norm, right / norm
            wish_x, wish_y = speed * (f * cos + r * sin), speed * (f * sin - r * cos)
        h = dt / SUBSTEPS
        blend = 1.0 - math.exp(-ACCELERATION * h)
        for _ in range(SUBSTEPS):
            airborne = s.z > self.ground + 1e-3 or self.vz > 0
            if not airborne:
                s.vx += (wish_x - s.vx) * blend
                s.vy += (wish_y - s.vy) * blend
                if buttons[_B["jump"]]:
                    self.vz = JUMP_SPEED
            self.vz -= GRAVITY * h
            self._move(h)
            s.z = max(self.ground, s.z + self.vz * h)
            if s.z <= self.ground:
                self.vz = 0.0
        s.t, s.weapon = float(t), int(weapon)
        s.buttons = int(sum(1 << i for i in range(NUM_BUTTONS) if buttons[i] > 0.5))
        return PlayerState(**vars(s))


def extrapolate(state: PlayerState, t: float, horizon: float) -> PlayerState:
    """``state`` moved on with its own velocity to room time ``t``, for at most ``horizon`` seconds."""
    dt = float(np.clip(t - state.t, 0.0, horizon))
    return PlayerState(
        **{**vars(state), "t": t, "x": state.x + state.vx * dt, "y": state.y + state.vy * dt}
    )


class GridWorld:
    """The synthetic view: a dark sky, a lit grid floor and every other player as a coloured pillar."""

    SCALE = 2  # the floor is drawn at half resolution, then upscaled
    CELL = 128.0  # grid spacing, units
    FOG = 2600.0  # distance at which the floor fades out, units

    def __init__(self) -> None:
        w, h = FRAME_W // self.SCALE, FRAME_H // self.SCALE
        self.focal = (FRAME_W / 2) / math.tan(math.radians(HFOV_DEGREES) / 2)
        jj, ii = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        self.a = (jj * self.SCALE - FRAME_W / 2) / self.focal
        self.b = -(ii * self.SCALE - FRAME_H / 2) / self.focal
        self.sky = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        try:
            self.font = ImageFont.load_default(size=13)
        except TypeError:  # Pillow < 10.1
            self.font = ImageFont.load_default()

    @staticmethod
    def axes(yaw: float, pitch: float):
        psi, theta = math.radians(yaw), math.radians(pitch)
        f = np.array(
            [math.cos(psi) * math.cos(theta), math.sin(psi) * math.cos(theta), -math.sin(theta)]
        )
        r = np.array([math.sin(psi), -math.cos(psi), 0.0])
        return f, r, np.cross(r, f)

    def render(self, me: PlayerState, peers: list[PlayerState]) -> np.ndarray:
        f, r, u = self.axes(me.yaw, me.pitch)
        eye = np.array([me.x, me.y, me.z + EYE_HEIGHT])
        dz = f[2] + self.a * r[2] + self.b * u[2]
        hit = dz < -1e-4
        dist = np.where(hit, eye[2] / np.maximum(-dz, 1e-4), 0.0)
        gx = eye[0] + dist * (f[0] + self.a * r[0] + self.b * u[0])
        gy = eye[1] + dist * (f[1] + self.a * r[1] + self.b * u[1])
        width = 1.5 + dist * 0.02
        edge = 0.5 - np.maximum(
            np.abs((gx / self.CELL) % 1.0 - 0.5), np.abs((gy / self.CELL) % 1.0 - 0.5)
        )
        line = np.clip(1.0 - edge * self.CELL / width, 0.0, 1.0)
        fog = np.clip(1.0 - dist / self.FOG, 0.0, 1.0) * hit
        glow = np.exp(-np.abs(dz) * 9.0)
        rgb = np.empty(dz.shape + (3,), np.float32)
        for c, (floor, grid, top, haze) in enumerate(
            ((12, 168, 4, 40), (13, 190, 6, 48), (18, 220, 14, 70))
        ):
            rgb[..., c] = np.where(
                hit, floor * fog + grid * line * fog**3 + haze * glow * 0.5, top + haze * glow
            )
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize(
            (FRAME_W, FRAME_H), Image.BILINEAR
        )
        self._draw_players(ImageDraw.Draw(img), eye, f, r, u, peers)
        return np.asarray(img)

    def _draw_players(self, draw, eye, f, r, u, peers) -> None:
        visible = []
        for p in peers:
            rel = np.array([p.x, p.y, p.z]) - eye
            depth = float(rel @ f)
            if depth > 24.0:
                visible.append((depth, p, rel))
        for depth, p, rel in sorted(visible, key=lambda v: -v[0]):
            cx = FRAME_W / 2 + float(rel @ r) / depth * self.focal
            foot = FRAME_H / 2 - float(rel @ u) / depth * self.focal
            height, half = 72.0 / depth * self.focal, 16.0 / depth * self.focal
            if cx + half < 0 or cx - half > FRAME_W:
                continue
            color = SEAT_COLORS[p.seat % len(SEAT_COLORS)]
            box = [cx - half, foot - height, cx + half, foot]
            draw.rounded_rectangle(box, radius=max(1.0, half), fill=color)
            draw.text(
                (cx, foot - height - 6),
                f"P{p.seat + 1}",
                fill="#f5f5f7",
                anchor="ms",
                font=self.font,
            )


class ClipPlayer:
    """Frames of a pre-rendered clip (a directory of JPEGs), looped."""

    def __init__(self, directory: Path) -> None:
        self.files = sorted(directory.glob("*.jpg"))
        if not self.files:
            raise FileNotFoundError(f"no .jpg frames in {directory}")

    def frame(self, index: int) -> np.ndarray:
        with Image.open(self.files[index % len(self.files)]) as im:
            return np.asarray(im.convert("RGB"))


class MockEngine:
    """:class:`demo.engine.EngineProtocol` without a GPU."""

    def __init__(self, config: MockConfig, library: Library, fps: float = 16.0) -> None:
        self.cfg, self.library, self.fps = config, library, float(fps)
        self.frames_per_step = int(config.frames_per_step)
        self._lock = threading.Lock()
        self._peers: dict[int, PlayerState] = {}
        self._outbox: list[PeerMessage] = []
        self._grid: GridWorld | None = None
        self._masks: dict[str, Walkable] = {}
        self.counts = {"scene_out": 0, "scene_in": 0}

    def start(self, round_start: RoundStart, seat: Seat, peers: Sequence[Seat]) -> PlayerState:
        self.round, self.seat = round_start, seat
        self.body = Kinematics(seat.seat, seat.spawn, self._walkable(round_start))
        self.frame_index = 0
        self.clip = ClipPlayer(self.library.path(seat.clip)) if seat.clip else None
        if self.clip is None and self._grid is None:
            self._grid = GridWorld()
        with self._lock:
            self._peers.clear()
            self._outbox.clear()
        self.counts = {"scene_out": 0, "scene_in": 0}
        return PlayerState(**vars(self.body.state))

    def _walkable(self, round_start: RoundStart) -> Walkable | None:
        radar = round_start.radar
        if radar is None:
            return None
        if radar.image not in self._masks:
            self._masks[radar.image] = Walkable(radar, self.library.path(radar.image))
        return self._masks[radar.image]

    def step(self, controls: Callable[[], BlockActions]) -> Iterator[Frame]:
        cfg = self.cfg
        actions = controls()
        time.sleep(max(0.0, cfg.step_ms + random.uniform(-cfg.jitter_ms, cfg.jitter_ms)) / 1000.0)
        horizon = self.frames_per_step / self.fps
        for j in range(actions.num_frames):
            t0 = time.perf_counter()
            me = self.body.advance(
                actions.buttons[j],
                actions.turn[j],
                int(actions.weapon[j]),
                float(actions.times[j]),
                1.0 / self.fps,
            )
            with self._lock:
                peers = [extrapolate(p, me.t, horizon) for p in self._peers.values()]
            rgb = self.clip.frame(self.frame_index) if self.clip else self._grid.render(me, peers)
            self.frame_index += 1
            if self.frame_index % cfg.scene_block_frames == 0:
                self._publish_scene_block(actions.block)
            time.sleep(max(0.0, cfg.decode_ms / 1000.0 - (time.perf_counter() - t0)))
            yield Frame(index=self.frame_index - 1, state=me, peers=peers, rgb=rgb)

    def _publish_scene_block(self, block: int) -> None:
        meta = {"frames": [self.frame_index - self.cfg.scene_block_frames, self.frame_index]}
        with self._lock:
            self._outbox.append(
                PeerMessage(
                    seat=self.seat.seat,
                    block=block,
                    kind="scene",
                    meta=meta,
                    payload=bytes(self.cfg.scene_block_bytes),
                )
            )
        self.counts["scene_out"] += 1

    def receive_state(self, state: PlayerState) -> None:
        with self._lock:
            old = self._peers.get(state.seat)
            if old is None or state.t >= old.t:
                self._peers[state.seat] = state

    def remove_peer(self, seat: int) -> None:
        with self._lock:
            self._peers.pop(int(seat), None)

    def take_messages(self) -> list[PeerMessage]:
        with self._lock:
            out, self._outbox = self._outbox, []
        return out

    def receive_message(self, message: PeerMessage) -> None:
        self.counts["scene_in"] += message.kind == "scene"

    def stats(self) -> dict[str, int]:
        return dict(self.counts)
