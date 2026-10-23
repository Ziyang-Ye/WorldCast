"""Live play: a client's player states under live controls.

The own rows follow the keys and are published as the block is denoised with them; written block by
block they are the motion of the whole round; the state model reads them as ticks, causally; the
pitch stays in range in every camera integral; a block that turns as the one before is prepared as
it is cut, however its rows share the turn; a wall's slowdown is kept over the next block while it
holds its movement keys, at a quarter of the way at the least, so that a state model that reads a
glitch cannot lock the camera, and the estimator steps with the camera as far as the view followed
it; a block that runs at a wall the depth head saw ahead stops before it at its cut (not when it
jumps, backs away or its cut has another frame's surface), and the estimator steps with the stopped
displacement; with ``live.wall_keys`` the movement keys of a block are cleared from the row its
camera is stopped at, in every place the controls go; every block stops at the edge of the map's
walkable extent, and the own position never moves farther in a latent frame than the prior's top
speed. Nobody but the players a human plays is in the world, the first frame included: another live
player is found by its first part, drawn from its rows (those that arrived before the cut
included), hidden while its client is done, and a seat nobody plays is nowhere, its recorded death
no corpse, and can be taken; a block of another player is keyed at its published cameras; the
target labels are tested again with the cameras of the cut, against the context of the block's own
labels, reprojected once per set of cameras; the cut's z-test and view angles are torch's arithmetic
operation by operation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import worldcast.player_state.live as live_module
import worldcast.player_state.predicted_visibility as predicted_visibility_module
from tests.data.support import tick_table
from tests.player_state.support import (
    PRIOR,
    REPO,
    flat_depth,
    open_ground,
    own_position_read,
    round_batch,
    standing_round,
)
from worldcast.data.camera import c2w_from_state_rows, camera_rays, half_angle_tangents
from worldcast.data.controls import (
    CAMERA_DELTA_SCALE,
    CONTROL_BUTTONS,
    OPENCS2_BUTTONS,
    button_bits,
    buttons_from_bits,
    count_until,
    encode_weapon,
    quantize_camera_delta,
    state_model_controls,
)
from worldcast.data.game import PITCH_MAX_DEGREES, TICK_RATE
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.latents import (
    BLOCK,
    LATENT_GRID,
    RECENT,
    source_frame,
    video_frame_count,
    video_frames_of,
)
from worldcast.data.memory_frames import MemoryBlock
from worldcast.data.recordings import ALIVE_INDEX
from worldcast.engine.inference.controls import BlockControls
from worldcast.engine.inference.idle_context import is_idle
from worldcast.modeling.state_model import (
    STATE_MODEL_WINDOW_LATENT_FRAMES,
    STATE_MODEL_WINDOW_STRIDE,
    StateModel,
    load_cell_table,
)
from worldcast.player_state.closed_loop import StateReader, window_of
from worldcast.player_state.extrapolation import MOVEMENT_COLUMNS, player_motion
from worldcast.player_state.live import (
    LiveLoop,
    LiveMotion,
    LiveStateReader,
    live_observer_signals,
)
from worldcast.player_state.live_peers import (
    LivePeer,
    PeerRows,
    block_substeps,
    packed_substeps,
    prior_displacements,
)
from worldcast.player_state.live_position import CONTINUATION_FLOOR, Walkable, wall_clearance
from worldcast.player_state.predicted_visibility import (
    POSE_RADIUS_U,
    PredictedVisibility,
    ztest_labels,
)
from worldcast.player_state.projection import table_angles
from worldcast.player_state.states import (
    PlayerState,
    continuous_row_state,
    derive_death_state,
    integrate_camera_angles,
    pack_substeps,
    wrap_degrees,
)

LATENTS = 33
#: The players stand still: the client (slot 0) at the origin looking along +x, slot 1 1600 u
#: ahead and 600 u to the left, slots 2 and 3 behind it.
POSITIONS = [[0.0, 0.0, 0.0], [1600.0, 600.0, 0.0], [-2000.0, 0.0, 0.0], [-2200.0, 300.0, 0.0]]
MEDIA = {slot: f"7-de_nuke-r03-p{slot:02d}" for slot in range(len(POSITIONS))}
FORWARD, JUMP = CONTROL_BUTTONS.index("forward"), CONTROL_BUTTONS.index("jump")
LEFT, ATTACK = CONTROL_BUTTONS.index("move_left"), CONTROL_BUTTONS.index("attack")
AK47, AWP = encode_weapon("ak47"), encode_weapon("awp")
CELLS = REPO / "configs" / "state_model" / "cells.json"
#: The share of the prior's step the motion head sees while a wall blocks the player: the view
#: stands still.
WALL = 0.0
#: A gathered window's labels, which a cut tests again.
WINDOW_LABELS = ("client_visibility", "client_visibility_valid")


class Room:
    """The ``live`` parts of a room in memory: a part reaches the other clients' inboxes as it is
    published, unless its sender is ``held`` back (its parts wait there), and is applied at their
    next refresh. ``done`` holds the status of the clients that are done."""

    def __init__(self) -> None:
        self.exchanges: dict[str, "Exchange"] = {}
        self.held: dict[str, list] = {}
        self.done: dict[str, str] = {}

    def join(self, client: str) -> "Exchange":
        self.exchanges[client] = Exchange(self, client)
        return self.exchanges[client]

    def send(self, part) -> None:
        if part.media_id in self.held:
            self.held[part.media_id].append(part)
            return
        for client, exchange in self.exchanges.items():
            if client != part.media_id:
                exchange.inbox.append(part)


class Exchange:
    """One client's view of a :class:`Room` (a ``LiveExchange``)."""

    def __init__(self, room: Room, client: str) -> None:
        self.room, self.client = room, client
        self.inbox: list = []
        self.parts: dict[str, list] = {}
        self.refreshes = 0

    def refresh(self) -> None:
        self.refreshes += 1
        for part in self.inbox:
            self.parts.setdefault(part.media_id, []).append(part)
        self.inbox = []

    def publish_live_state(self, f0, *, times, rows, buttons, turn, weapon) -> None:
        part = SimpleNamespace(
            media_id=self.client,
            f0=int(f0),
            times=np.array(times, np.float64),
            rows=np.array(rows, np.float32),
            buttons=button_bits(buttons),
            turn=np.array(turn, np.float32),
            weapon=np.asarray(weapon).astype(np.uint8),
        )
        self.parts.setdefault(self.client, []).append(part)
        self.room.send(part)

    def live_part(self, media_id, f0):
        return next((p for p in self.parts.get(media_id, []) if p.f0 == f0), None)

    def recent_live(self, media_id, n=2):
        parts = sorted(self.parts.get(media_id, []), key=lambda part: float(part.times[-1]))
        return parts[-n:]

    def done_status(self, media_id):
        return self.room.done.get(media_id)


class OracleReader:
    """Stands in for the state model: it reads the own position where the keys put it
    (:func:`~tests.player_state.support.own_position_read`) and records the ticks of the final
    rows."""

    def __init__(self) -> None:
        self.motion: LiveMotion | None = None
        self.rows = 0

    def read(self, latents, latent_frames):
        return own_position_read(self.motion, latent_frames)

    def append(self, buttons, turn) -> None:
        self.rows += len(buttons)


class CameraReader(OracleReader):
    """The state model of a player whose view follows the camera, a wall's stop and a continuation's
    slowdown included: its heads read the displacement of the camera."""

    def read(self, latents, latent_frames):
        camera, start = self.motion.displacement, self.motion.start_state[:3]
        frames = [int(f) for f in latent_frames]
        place = np.stack([start + camera[4 * f] for f in frames])
        steps = [(camera[4 * f] - camera[max(4 * f - 4, 0)]) / 64.0 for f in frames]
        return place, np.stack(steps).astype(np.float32)


class GlitchReader(CameraReader):
    """:class:`CameraReader` whose motion head reads nothing on the latent frames in ``glitch``."""

    def __init__(self, glitch: set[int]) -> None:
        super().__init__()
        self.glitch = glitch

    def read(self, latents, latent_frames):
        place, steps = super().read(latents, latent_frames)
        for i, f in enumerate(latent_frames):
            if int(f) in self.glitch:
                steps[i] = 0.0
        return place, steps


class LostReader(CameraReader):
    """The state model of a client whose place head is lost: it reads a track 2,000 u off that moves
    with the camera, and the motion head the displacement stopped at a wall."""

    def read(self, latents, latent_frames):
        place, steps = super().read(latents, latent_frames)
        return place + [2000.0, 2000.0, 0.0], steps


class AheadReader(CameraReader):
    """:class:`CameraReader` with the place head 40 u ahead along +x from latent frame 1 on."""

    def read(self, latents, latent_frames):
        place, steps = super().read(latents, latent_frames)
        ahead = 40.0 * (np.asarray(latent_frames) > 0)
        return place + ahead[:, None] * [1.0, 0.0, 0.0], steps


class WallReader(OracleReader):
    """The state model of a player that walks into a wall: the place head stays at the round
    start, and the motion head sees :data:`WALL` of every step of the prior."""

    def read(self, latents, latent_frames):
        place, steps = super().read(latents, latent_frames)
        start = np.repeat(self.motion.start_state[None, :3], len(place), 0)
        return start, steps * np.float32(WALL)


def live_loop(
    room: Room,
    slot: int,
    *,
    latents: int = LATENTS,
    dies: dict | None = None,
    reader: OracleReader | None = None,
    depth: float = 5000.0,
    walkable: Walkable | None = None,
    moving: bool = False,
    wall_keys: str = "held",
    dead_zone_u: float = 0.0,
    hop: str = "step",
) -> LiveLoop:
    """The live player states of the client at ``slot`` in ``room``; ``dies`` ``{slot: video
    frame}``: a player's recording is dead from then on; ``reader``: the state model
    (:class:`OracleReader` by default); ``depth``: the depth head sees a surface that far in front
    of every latent frame, u; ``walkable``: the map's extent (open ground by default);
    ``moving``: the round of ten players that move and turn, with random weapons
    (:func:`~tests.player_state.support.round_batch`), not the standing four; ``wall_keys``:
    ``live.wall_keys``; ``dead_zone_u``, ``hop``: the settings of that name."""
    if moving:
        batch = round_batch(0, batch=1, latents=latents, client_slots=(slot,))
    else:
        batch = standing_round(POSITIONS, latents=latents, client_slot=slot)
    frames = video_frame_count(latents)
    batch["window_tans"] = torch.tensor([0.4, 0.2]).expand(1, latents, 2).clone()  # scoped
    batch.update({key: torch.ones(1, latents, dtype=torch.long) for key in OBSERVER_SIGNAL_KEYS})
    batch["buttons"] = torch.zeros(1, frames, len(CONTROL_BUTTONS))
    batch["view_deltas"] = torch.zeros(1, frames, 2)
    batch["weapon"] = torch.full((1, frames), AK47)
    if not moving:
        batch["player_weapon_ids"][:] = AK47
    for dead, frame in (dies or {}).items():
        batch["player_states"][0, dead, frame:, ALIVE_INDEX] = 0.0
    reader = reader or OracleReader()
    loop = LiveLoop.build(
        batch,
        reader=reader,
        client=MEDIA[slot],
        client_slot=slot,
        round_media=MEDIA,
        latent_frames=latents,
        start_frame=0,
        prior=PRIOR,
        walkable=walkable or open_ground(),
        depth_fn=flat_depth(depth),
        world_state=room.join(MEDIA[slot]),
        device="cpu",
        wall_keys=wall_keys,
        dead_zone_u=dead_zone_u,
        hop=hop,
    )
    reader.motion = loop.motion
    return loop


def keys(*held: int) -> np.ndarray:
    buttons = np.zeros((16, len(CONTROL_BUTTONS)), np.float32)
    buttons[:, list(held)] = 1.0
    return buttons


def session_times(s: int, clock: float = 0.0) -> np.ndarray:
    """The session times of block ``s``'s video frames: 1/16 s apart from ``clock``."""
    return clock + np.asarray(video_frames_of(s, BLOCK), np.float64) / 16.0


def numbered(n: int) -> torch.Tensor:
    """``n`` stand-ins of latent frames ``[n, 1, 1, 1]``, each holding its latent frame."""
    return torch.arange(n).float().view(-1, 1, 1, 1)


def prepare(loop: LiveLoop, s: int) -> None:
    loop.publish_position(s, numbered(s))
    loop.read_positions(s)


def cut(loop: LiveLoop, s: int, buttons=None, *, turn=(0.0, 0.0), clock=0.0, weapon=AK47) -> None:
    buttons = keys() if buttons is None else buttons
    turn = loop.clamp_pitch(np.tile(np.asarray(turn, np.float32), (16, 1)))
    loop.take_controls(s, buttons, turn, np.full(16, weapon), session_times(s, clock))


def play(loop: LiveLoop, s: int, buttons=None, **kwargs) -> None:
    prepare(loop, s)
    cut(loop, s, buttons, **kwargs)


def prior_result(given: tuple) -> tuple:
    return prior_displacements([given], prior=PRIOR)[0]


def own_rows(loop: LiveLoop, s: int) -> torch.Tensor:
    return loop.batch["player_states"][0, loop.slot, video_frames_of(s, BLOCK)]


# ------------------------------------------------------------------------------------ own rows
def test_the_own_rows_follow_the_keys_and_are_what_the_client_publishes():
    room = Room()
    loop, watch = live_loop(room, 0), live_loop(room, 2)
    turn = (0.0, 1.5)  # degrees per video frame
    for s in range(1, LATENTS, BLOCK):
        play(loop, s, keys(FORWARD), turn=turn)
        play(watch, s)
    # the client's rows are where its keys put it: the motion of its whole round
    batch = standing_round(POSITIONS, latents=LATENTS)
    values, valid = block_substeps(keys(FORWARD).repeat(8, 0), np.tile(turn, (128, 1)))
    batch["player_control_substeps"][0, 0, 1:] = torch.from_numpy(values)
    batch["player_control_substep_valid"][0, 0, 1:] = torch.from_numpy(valid)
    motion = player_motion(batch, 0, prior=PRIOR)
    states = loop.batch["player_states"][0, 0].double().numpy()
    np.testing.assert_allclose(states[1:, :3], motion.displacement[1:], atol=0.5)
    np.testing.assert_allclose(states[1:, 3], motion.yaw[1:], atol=1e-4)
    assert states[-1, 3] == pytest.approx(1.5 * 128 - 360.0, abs=1e-3)
    # a block is published as it is denoised: the rows of the table, and the cameras of its rays
    parts = room.exchanges[MEDIA[0]].parts[MEDIA[0]]
    assert [part.f0 for part in parts] == list(range(1, LATENTS, BLOCK))
    for part in parts:
        rows = own_rows(loop, part.f0)
        assert np.array_equal(part.rows, rows[:, :5].numpy())
        assert np.array_equal(part.times, session_times(part.f0))
        assert np.array_equal(part.buttons, button_bits(keys(FORWARD)))
        cameras = loop.cameras.for_block(part.f0)[part.f0 : part.f0 + BLOCK]
        published = np.column_stack([part.rows[3::4], np.ones(4, np.float32)])
        assert torch.equal(cameras, c2w_from_state_rows(published))
    # the other client draws it where it published itself, but for the blend of a block's start
    # into what it drew from the rows before (their extension holds the yaw); the standing
    # players never move
    for s in range(1, LATENTS, BLOCK):
        drawn = watch.batch["player_states"][0, 0, video_frames_of(s, BLOCK), :3].numpy()
        assert np.abs(drawn - own_rows(loop, s)[:, :3].numpy()).max() < 0.1
        assert np.array_equal(drawn[-1], own_rows(loop, s)[-1, :3].numpy())
    for slot in (1, 3):
        assert (loop.batch["player_states"][0, slot, :, :3] == torch.tensor(POSITIONS[slot])).all()
    # the own controls replace the recording's, and the field of view is the unscoped one
    assert loop.reader.rows == video_frame_count(LATENTS) - 1  # the final rows, as ticks
    assert (loop.batch["window_tans"][0, 1:] == torch.tensor(half_angle_tangents())).all()
    assert loop.batch["window_tans"][0, 0].tolist() == pytest.approx([0.4, 0.2])


def game_keys(rows: int, seed: int) -> np.ndarray:
    """``[rows, 11]`` keys as a player holds them: a few movement and gait keys, changed about
    every 12 video frames, and jump taps."""
    rng = np.random.default_rng(seed)
    choices = [
        CONTROL_BUTTONS.index(name)
        for name in ("forward", "back", "move_left", "move_right", "speed", "duck")
    ]
    buttons, held = np.zeros((rows, len(CONTROL_BUTTONS)), np.float32), []
    for row in buttons:
        if rng.random() < 0.08:
            held = list(rng.choice(choices, size=rng.integers(0, 3), replace=False))
        row[held] = 1.0
        row[JUMP] = float(rng.random() < 0.02)
    return buttons


def test_the_motion_written_block_by_block_is_the_motion_of_the_whole_round():
    blocks = 12
    buttons = game_keys(16 * blocks, seed=3)
    turn = np.random.default_rng(4).normal(0.0, 3.0, (16 * blocks, 2)).astype(np.float32)
    batch = standing_round(POSITIONS, latents=1 + BLOCK * blocks)
    values, valid = block_substeps(buttons, turn)
    batch["player_control_substeps"][0, 0, 1:] = torch.from_numpy(values)
    batch["player_control_substep_valid"][0, 0, 1:] = torch.from_numpy(valid)
    whole = player_motion(batch, 0, prior=PRIOR)

    packed = pack_substeps(batch["player_control_substeps"], batch["player_control_substep_valid"])
    motions = []
    for provisional in (False, True):
        motion = LiveMotion(whole.start_state, packed[0, 0, 0], len(whole.yaw), prior=PRIOR)
        for k in range(blocks):
            rows = slice(1 + 16 * k, 17 + 16 * k)
            if provisional:  # another block's controls first: nothing of them stays
                other = pack_substeps(
                    *map(torch.from_numpy, block_substeps(buttons[::-1][rows], -turn[rows]))
                )
                motion.write(*prior_result(motion.prior_input(rows.start, other)), final=False)
            motion.write(
                *prior_result(motion.prior_input(rows.start, packed[0, 0, rows])), final=True
            )
        assert motion.final_row == 16 * blocks
        motions.append(motion)
    # the integral of the whole round's keys, but for the order of float32 sums
    for motion in motions:
        np.testing.assert_allclose(motion.displacement, whole.displacement, atol=1e-3)
        np.testing.assert_allclose(motion.yaw, whole.yaw, atol=1e-4)
        np.testing.assert_allclose(motion.pitch, whole.pitch, atol=1e-4)
    for name in ("displacement", "prior_displacement", "yaw", "pitch"):
        assert np.array_equal(getattr(motions[0], name), getattr(motions[1], name)), name
    with pytest.raises(ValueError, match="rows from 5 after the final row 192"):
        motions[0].prior_input(5, packed[0, 0, 1:17])


def test_a_write_stops_at_the_wall_ahead_then_keeps_a_walls_slowdown():
    """A block of W from rest stops 50 u along its way; the slowdown of a wall the block before
    found keeps half of what is left along it. The prior's displacement runs on, the camera's
    stops and is half of what is left; across the way and in height nothing changes."""
    batch = standing_round(POSITIONS, latents=1 + BLOCK)
    packed = pack_substeps(batch["player_control_substeps"], batch["player_control_substep_valid"])
    start = batch["player_states"][0, 0, 0].double().numpy()
    motion = LiveMotion(start, packed[0, 0, 0], video_frame_count(1 + BLOCK), prior=PRIOR)
    values, valid = map(torch.from_numpy, block_substeps(keys(FORWARD), np.zeros((16, 2))))
    given = motion.prior_input(1, pack_substeps(values, valid))
    along = np.array([1.0, 0.0])
    motion.write(*prior_result(given), final=True, stops=[(along, 50.0)], kept=(along, 0.5))
    prior, camera = motion.prior_displacement[1:17], motion.displacement[1:17]
    assert prior[-1, 0] > 150.0
    np.testing.assert_allclose(
        camera[:, 0], 0.5 * np.minimum(prior[:, 0], 50.0), rtol=0.0, atol=1e-12
    )
    assert np.array_equal(camera[:, 1:], prior[:, 1:])


def test_the_pitch_stays_in_range_in_every_camera_integral():
    """200 rows of +3 degrees: the effective turn stops at 89 degrees, where the 13-wide integral
    clamps; the 6-wide one, which does not clamp, and the own camera agree with it."""
    latents = 1 + BLOCK * 13
    room = Room()
    loop = live_loop(room, 0, latents=latents)
    for s in range(1, latents, BLOCK):
        play(loop, s, turn=(3.0, 0.0))
    pitch = loop.motion.pitch[1 : 1 + 16 * 13]
    assert pitch.max() == PITCH_MAX_DEGREES and pitch[-170:].min() == PITCH_MAX_DEGREES
    continuous = continuous_row_state(loop.batch)[0, :, 0, 1].double().numpy()
    six_wide = PlayerState.from_batch(loop.batch)
    _, integral = table_angles(six_wide.table(), six_wide.controls)
    own = loop.motion.pitch[4 * np.arange(latents)]
    np.testing.assert_allclose(continuous, own, atol=1e-3)
    np.testing.assert_allclose(integral[0, :, 0].double().numpy(), own, atol=1e-3)
    assert own[-1] == pytest.approx(89.0, abs=1e-3)


@pytest.mark.parametrize("first_latent", [False, True], ids=["spread", "front"])
def test_a_block_that_turns_as_the_one_before_is_prepared_as_it_is_cut(first_latent):
    """The turn is held row by row, however a block's rows share it (spread over the block, or
    all in its first latent frame: the demo's ``play.input_mapping``): retrieval and the anchor of
    the rays look where the block will look."""
    turn = np.zeros((16, 2), np.float32)
    if first_latent:
        turn[:4, 1] = 10.0
    else:
        turn[:, 1] = 2.5
    loop = live_loop(Room(), 0)
    for s in (1, 5, 9):
        prepare(loop, s)
        provisional = own_rows(loop, s).clone()
        loop.take_controls(s, keys(FORWARD), turn, np.full(16, AK47), session_times(s))
        if s > 1:  # the first block is prepared with no turn held
            assert torch.equal(own_rows(loop, s), provisional)
    rows = video_frames_of(9, BLOCK)
    yaw = loop.motion.yaw[rows[0] - 1 : rows[-1] + 1]
    np.testing.assert_allclose(np.diff(yaw), turn[:, 1], atol=1e-4)


def record_steps(loop: LiveLoop) -> list[tuple[np.ndarray, np.ndarray]]:
    """The prior's and the camera's steps the own position is fed from now on."""
    fed, step = [], loop.position.step

    def recorded(place, motion, prior_step, camera_step):
        fed.append((prior_step, camera_step))
        return step(place, motion, prior_step, camera_step)

    loop.position.step = recorded
    return fed


def block_motion(loop: LiveLoop, s: int) -> tuple[np.ndarray, np.ndarray]:
    """The own displacement over block ``s``'s rows from the row before it: the camera's and the
    prior's, ``[16, 3]`` u each."""
    rows, motion = video_frames_of(s, BLOCK), loop.motion
    camera = motion.displacement[rows] - motion.displacement[rows[0] - 1]
    return camera, motion.prior_displacement[rows] - motion.prior_displacement[rows[0] - 1]


@pytest.mark.parametrize(
    ("held", "yaw", "share"),
    [
        ((FORWARD,), 14.0, CONTINUATION_FLOOR),  # the same keys, turning less than 15 degrees
        ((FORWARD, JUMP), 0.0, CONTINUATION_FLOOR),  # jump is no movement key
        ((FORWARD,), 15.0, 1.0),
        ((FORWARD, LEFT), 0.0, 1.0),
    ],
    ids=["turning-14", "jumping", "turning-15", "another-key"],
)
def test_a_wall_slows_the_next_block_while_it_holds_its_movement_keys(held, yaw, share):
    """The motion head finds the view of the newest latent frame of block 1 stood still: block 5
    keeps a quarter of its way along that latent frame's step (the floor: the view follows a camera
    that moves), from its prepare on, unless its movement keys change or it turns 15 degrees."""
    loop = live_loop(Room(), 0, reader=WallReader())
    play(loop, 1, keys(FORWARD))
    prepare(loop, 5)
    assert loop.position.blocked_progress == CONTINUATION_FLOOR
    blocked = loop.motion.prior_displacement[16] - loop.motion.prior_displacement[12]
    along = blocked[:2] / np.linalg.norm(blocked[:2])
    across = np.array([-along[1], along[0]])
    # prepared with block 1's keys held: slowed along the step, unchanged across it
    camera, prior = block_motion(loop, 5)
    assert prior[-1, :2] @ along > 200.0
    np.testing.assert_allclose(
        camera[:, :2] @ along, CONTINUATION_FLOOR * (prior[:, :2] @ along), rtol=1e-6
    )
    np.testing.assert_allclose(camera[:, :2] @ across, prior[:, :2] @ across, atol=1e-9)
    # at the cut, with the block's own keys and turn
    cut(loop, 5, keys(*held), turn=(0.0, yaw / 16))
    camera, prior = block_motion(loop, 5)
    np.testing.assert_allclose(camera[:, :2] @ along, share * (prior[:, :2] @ along), rtol=1e-6)
    np.testing.assert_allclose(camera[:, :2] @ across, prior[:, :2] @ across, atol=1e-9)
    assert np.array_equal(camera[:, 2], prior[:, 2])


def test_the_own_position_is_fed_the_slowed_camera_and_the_prior_it_is_read_against():
    """The slowed block's camera is the prior's displacement with the wall's slowdown; the prior's
    own displacement is the one of the same keys without a wall. The own position is fed both:
    it steps with the camera, as far as the view followed it, and the wall test of the next block
    reads the camera that moved."""
    walled, free = (live_loop(Room(), 0, reader=reader) for reader in (WallReader(), None))
    for loop in (walled, free):
        play(loop, 1, keys(FORWARD))
        play(loop, 5, keys(FORWARD))
    camera, prior = block_motion(walled, 5)
    assert not np.allclose(camera, prior, atol=1.0)
    assert np.array_equal(walled.motion.prior_displacement, free.motion.prior_displacement)
    assert np.array_equal(free.motion.displacement, free.motion.prior_displacement)
    fed = record_steps(walled)
    prepare(walled, 9)
    motion = walled.motion
    prior_steps, camera_steps = (
        np.stack([track[4 * f] - track[4 * f - 4] for f in range(5, 9)])
        for track in (motion.prior_displacement, motion.displacement)
    )
    assert np.array_equal(np.stack([f[0] for f in fed]), prior_steps)
    assert np.array_equal(np.stack([f[1] for f in fed]), camera_steps)
    assert not np.allclose(camera_steps, prior_steps, atol=1.0)


def block_paths(loop: LiveLoop, blocks: int) -> list[int]:
    """The camera's path of each of ``blocks`` blocks of W held, u (rounded)."""
    paths = []
    for k in range(blocks):
        s = 1 + BLOCK * k
        play(loop, s, keys(FORWARD))
        rows = own_rows(loop, s)[:, :2].numpy()
        paths.append(round(float(np.linalg.norm(rows[-1] - rows[0]))))
    return paths


def test_a_glitch_of_the_motion_head_costs_one_slowed_block_not_the_camera():
    """The view follows the camera (the state model's heads read its displacement) but the motion
    head reads nothing on latent frames 11 and 12, the newest two of a block: the next block keeps
    a quarter of its way, the view moves with it, and the block after runs whole. Were the
    slowdown kept at what the head read, the view of a camera that stood still would stand still
    for good."""
    blocks = 16
    loop = live_loop(Room(), 0, latents=1 + BLOCK * blocks + BLOCK, reader=GlitchReader({11, 12}))
    assert block_paths(loop, blocks) == [180, 230, 230, 57] + [230] * 12


def test_a_view_that_stands_still_keeps_the_camera_creeping_at_the_floor():
    """The state model sees the view stand still whatever the camera does (a wall the depth head
    does not see): every block after the first keeps a quarter of its way, 57 u of 230, so that the
    view can show it moved; nothing else moves it."""
    blocks = 10
    loop = live_loop(Room(), 0, latents=1 + BLOCK * blocks + BLOCK, reader=WallReader())
    assert block_paths(loop, blocks) == [180] + [57] * 9


# ------------------------------------------------------------------------------ the wall ahead
#: The depth head of the wall tests sees a wall this far ahead of the eye on every latent frame.
WALL_U = 120.0


def wall_loop(reader: OracleReader | None = None) -> LiveLoop:
    """A client that stands :data:`WALL_U` before a wall the depth head sees."""
    return live_loop(Room(), 0, reader=reader or CameraReader(), depth=WALL_U)


def test_the_surface_ahead_is_read_off_a_latent_frame_in_the_world():
    """The client stands at the origin looking along +x, its eye 64 u above its feet; after a block
    forward its camera stands 104 u on, stopped before the wall."""
    loop = wall_loop()
    surface = loop.visibility.surface(0, numbered(1))
    assert surface.shape == (24 * 42, 3)
    np.testing.assert_allclose(surface[:, 0], WALL_U, rtol=0.0, atol=1e-4)
    assert surface[:, 2].min() == pytest.approx(64.0 - 0.75 * WALL_U * 23 / 24, abs=1e-3)
    play(loop, 1, keys(FORWARD))
    later = loop.visibility.surface(4, numbered(5))
    np.testing.assert_allclose(later[:, 0], 104.0 + WALL_U, rtol=0.0, atol=1e-4)


def test_a_block_stops_where_the_depth_head_saw_a_wall_ahead():
    """Running at the wall, block 1's camera stops 16 u (half the hull) before it; the prior runs
    on, and the own position steps with the stopped displacement."""
    loop = wall_loop()
    play(loop, 1, keys(FORWARD))
    camera, prior = block_motion(loop, 1)
    stop = WALL_U - 16.0
    assert prior[-1, 0] > stop + 10.0
    np.testing.assert_allclose(camera[:, 0], np.minimum(prior[:, 0], stop), rtol=0.0, atol=1e-4)
    assert np.array_equal(camera[:, 1:], prior[:, 1:])
    fed = record_steps(loop)
    prepare(loop, 5)
    stopped = loop.motion.displacement
    assert np.array_equal(
        np.stack([f[1] for f in fed]),
        np.stack([stopped[4 * f] - stopped[4 * f - 4] for f in range(1, 5)]),
    )
    assert loop.cameras.track.xyz[4, 0] == pytest.approx(stop, abs=1.0)


@pytest.mark.parametrize(
    ("first", "held"),
    [((FORWARD,), (FORWARD, JUMP)), ((), (CONTROL_BUTTONS.index("back"),))],
    ids=["jumping", "away"],
)
def test_a_block_that_jumps_or_moves_away_from_the_wall_is_not_stopped(first, held):
    """After block 1 (running at the wall and stopped, the state model reading the prior's
    displacement, so that no wall's slowdown is kept; or standing), block 5 jumps on or backs away:
    not stopped."""
    loop = wall_loop(OracleReader())
    play(loop, 1, keys(*first))
    play(loop, 5, keys(*held))
    camera, prior = block_motion(loop, 5)
    assert np.abs(prior[-1, :2]).max() > WALL_U and np.array_equal(camera, prior)


@pytest.mark.parametrize(
    ("reader", "moved_on"), [(CameraReader, False), (OracleReader, True)], ids=["held", "moved-on"]
)
def test_a_block_after_the_motion_head_said_the_view_moved_past_a_stop_is_not_stopped(
    reader, moved_on
):
    """Block 1 stops before the wall the depth head sees. Where the state model reads the stopped
    displacement, block 5 is stopped again; where it reads the prior's, the view moved on past the
    stop (the wall is not there): it is forgotten, and block 5 is not stopped."""
    loop = wall_loop(reader())
    play(loop, 1, keys(FORWARD))
    play(loop, 5, keys(FORWARD))
    assert loop.position.moved_past_wall is moved_on
    camera, prior = block_motion(loop, 5)
    assert np.array_equal(camera, prior) is moved_on


# --------------------------------------------------------------------------- a wall in the world
#: The static wall's plane across +x, x (u); a push stops half the hull before it.
WALL_X = 300.0
#: The cameras' rays through the depth head's cells, in camera space (axial depth 1).
RAYS = camera_rays(half_angle_tangents(), LATENT_GRID)


class StaticWall:
    """The depth head of a world with a floor at height 0 and a wall across +x at :data:`WALL_X`,
    ``width`` u wide around y = 0, seen exactly through the camera of each video frame of a client
    (:attr:`loop`); it comes into view at latent frame ``seen_from`` (from behind a corner). The
    latents it reads hold their latent frame (:func:`numbered`)."""

    def __init__(self, width: float, seen_from: int) -> None:
        self.width, self.seen_from, self.loop = width, seen_from, None

    def __call__(self, latents: torch.Tensor) -> np.ndarray:
        out = np.zeros((len(latents), 4, *LATENT_GRID), np.float32)
        for i, f in enumerate(latents[:, 0, 0, 0].round().long().tolist()):
            for c, v in enumerate(video_frames_of(f)):
                c2w = self.loop.visibility.cameras[v].numpy()
                rays, eye = RAYS @ c2w[:3, :3].T, c2w[:3, 3]
                with np.errstate(divide="ignore", invalid="ignore"):
                    wall = (WALL_X - eye[0]) / rays[:, 0]
                    beside = np.abs(eye[1] + wall * rays[:, 1]) > self.width / 2
                    floor = -eye[2] / rays[:, 2]
                wall[~(wall > 0) | beside | (f < self.seen_from)] = np.inf
                floor[~(floor > 0)] = np.inf
                out[i, c] = np.log(np.minimum(np.minimum(wall, floor), 5000.0)).reshape(LATENT_GRID)
        return out


def static_wall_loop(
    reader: OracleReader, *, width: float = np.inf, seen_from: int = 0, wall_keys: str = "held"
) -> LiveLoop:
    """A client at the origin facing +x, :data:`WALL_X` before a :class:`StaticWall`."""
    loop = live_loop(Room(), 0, reader=reader, wall_keys=wall_keys)
    loop.visibility.depth_fn = StaticWall(width, seen_from)
    loop.visibility.depth_fn.loop = loop
    return loop


def play_prepared(loop: LiveLoop, s: int, *held: int) -> torch.Tensor:
    """:func:`play` block ``s`` with the keys ``held``; returns its own rows as prepared."""
    prepare(loop, s)
    prepared = own_rows(loop, s).clone()
    cut(loop, s, keys(*held))
    return prepared


def test_a_block_is_prepared_stopped_where_its_cut_stops():
    """The wall 300 u ahead is too far for the depth head's cells at latent frame 0; from latent
    frame 4, 118 u before it, it is found when block 5 is prepared: its rows stop 16 u before it
    as prepared (retrieval and the rays' anchor) and as cut."""
    loop = static_wall_loop(LostReader())
    play(loop, 1, keys(FORWARD))
    prepared = play_prepared(loop, 5, FORWARD)
    assert torch.equal(prepared, own_rows(loop, 5))
    _, prior = block_motion(loop, 5)
    assert prior[-1, 0] > 200.0
    assert float(own_rows(loop, 5)[-1, 0]) == pytest.approx(WALL_X - 16.0, abs=1e-4)


def test_a_push_stops_at_a_wall_for_as_long_as_it_lasts():
    """A client whose place head is lost runs at a wall for 3 s, stands for 1 s and pushes on for
    4 s. Block 5 reaches the wall and stops 16 u (half the hull) before it. From there its depth
    head no longer sees the wall's lower part, and every later block stops at the wall block 5
    stopped before, as prepared and as cut: the camera never passes the stop and never steps
    back."""
    loop = static_wall_loop(LostReader())
    stop = WALL_X - 16.0  # as the depth head's float32 cells place it
    held = [(FORWARD,)] * 3 + [()] + [(FORWARD,)] * 4
    for s, keys_held in zip(range(1, LATENTS, BLOCK), held):
        assert play_prepared(loop, s, *keys_held)[:, 0].max() <= stop + 1e-4
    assert loop.status == "degraded"
    x = loop.batch["player_states"][0, 0, :, 0].numpy()
    assert x.max() <= stop + 1e-4 and (np.diff(x) >= 0.0).all()
    held_from = video_frames_of(5, BLOCK)[-1]
    assert x[held_from] == pytest.approx(stop, abs=1e-4) and (x[held_from:] == x[held_from]).all()
    assert (loop.cameras.track.xyz[8:] == loop.cameras.track.xyz[8]).all()
    # a level view 16 u before the wall sees nothing of it below 52 u
    surface = loop.visibility.surface(LATENTS - 1, numbered(LATENTS))
    assert wall_clearance(surface - loop.cameras.track.xyz[-1], np.array([1.0, 0.0])) == np.inf


def test_a_wall_ahead_stops_a_block_measured_from_where_it_starts():
    """The place head reads 40 u ahead of the camera from latent frame 1 on, so the own position at
    latent frame 4 is ahead of the camera that sees the wall come into view there: block 5 still
    stops 16 u before the wall, measured from where it starts."""
    loop = static_wall_loop(AheadReader(), seen_from=4)
    play(loop, 1, keys(FORWARD))
    play(loop, 5, keys(FORWARD))
    start, seen_from = loop.cameras.track.xyz[4, 0], float(loop.visibility.cameras[16, 0, 3])
    assert start - seen_from == pytest.approx(27.42, abs=0.005)
    assert float(own_rows(loop, 5)[:, 0].max()) == pytest.approx(WALL_X - 16.0, abs=1e-4)


class PassedReader(CameraReader):
    """:class:`CameraReader` whose place head reads the player up to 100 u farther along +x than
    the camera stopped, as far as the prior went, and whose motion head sees the view move on, by
    less than :data:`~worldcast.player_state.live_position.PASSED_SHARE` of the prior's step."""

    def read(self, latents, latent_frames):
        place, _ = super().read(latents, latent_frames)
        _, steps = own_position_read(self.motion, latent_frames)
        frames = [4 * int(f) for f in latent_frames]
        ahead = self.motion.prior_displacement[frames, 0] - self.motion.displacement[frames, 0]
        return place + np.minimum(ahead, 100.0)[:, None] * [1.0, 0.0, 0.0], steps * np.float32(0.45)


@pytest.mark.parametrize(
    ("then", "reader", "width"),
    [("strafe", LostReader, 200.0), ("jump", LostReader, np.inf), ("run", PassedReader, np.inf)],
)
def test_a_wall_a_block_stopped_before_is_forgotten_another_way_or_past_it(then, reader, width):
    """A client stops before a wall (block 5), then strafes past its end (the wall 200 u wide) or
    jumps through it (a block that leaves the ground is not stopped; jump on its first two rows,
    landed within it), its place head lost; or runs on while its place head, followed, reads it
    100 u past the stop. The wall is forgotten and the next block runs on past it."""
    loop = static_wall_loop(reader(), width=width)
    block_9 = keys(LEFT) if then == "strafe" else keys(FORWARD)
    block_9[:2, JUMP] = float(then == "jump")
    blocks = (1, 5, 9) if then == "run" else (1, 5, 9, 13)
    for s, buttons in zip(blocks, [keys(FORWARD), keys(FORWARD), block_9, keys(FORWARD)]):
        play(loop, s, buttons)
    camera, prior = block_motion(loop, blocks[-1])
    assert np.array_equal(camera, prior) and own_rows(loop, blocks[-1])[-1, 0] > WALL_X


def test_a_block_stops_at_the_edge_of_the_walkable_extent_and_the_own_position_with_it():
    """A client runs along +x on a strip of the map that ends at x = 768 u: as prepared and as cut,
    every block stops there (within a sample step of 16 u) and the own position steps with the
    stop; no later block carries the camera on or back."""
    strip = Walkable(np.array([[i, 0] for i in range(-2, 2)]))
    loop = live_loop(Room(), 0, reader=CameraReader(), walkable=strip)
    for s in range(1, LATENTS, BLOCK):
        assert play_prepared(loop, s, FORWARD)[:, 0].max() <= 768.0
    x = loop.batch["player_states"][0, 0, :, 0].numpy()
    assert 750.0 < x.max() <= 768.0 and (np.diff(x) >= 0.0).all()
    edge = video_frames_of(13, BLOCK)[-1]
    assert (x[edge:] == x[edge]).all()
    assert (loop.cameras.track.xyz[14:] == loop.cameras.track.xyz[14]).all()


# ------------------------------------------------------------------------------ the keys at a stop
#: The walkable strip of the edge tests: it ends at x = 768 u.
STRIP = Walkable(np.array([[i, 0] for i in range(-2, 2)]))
MOVEMENT = list(MOVEMENT_COLUMNS)


class TickReader(CameraReader):
    """:class:`CameraReader` that keeps the buttons of every block it is given as ticks."""

    def __init__(self) -> None:
        super().__init__()
        self.buttons: list[np.ndarray] = []

    def append(self, buttons, turn) -> None:
        super().append(buttons, turn)
        self.buttons.append(np.array(buttons))


def release(loop: LiveLoop, s: int, buttons: np.ndarray, *, turn=(0.0, 0.0)) -> np.ndarray:
    """What :meth:`LiveLoop.release_keys` makes of ``buttons`` for block ``s``."""
    return loop.release_keys(s, buttons, loop.clamp_pitch(np.tile(np.float32(turn), (16, 1))))


def play_released(loop: LiveLoop, s: int, buttons: np.ndarray, **kwargs) -> np.ndarray:
    """:func:`play` block ``s`` as the rollout's cut does: the movement keys are released where the
    camera is stopped, and the block is cut with the buttons that remain; returns them."""
    prepare(loop, s)
    taken = release(loop, s, buttons, turn=kwargs.get("turn", (0.0, 0.0)))
    cut(loop, s, taken, **kwargs)
    return taken


def stop_row(buttons: np.ndarray) -> int:
    """The row the forward key is cleared from (16: none)."""
    held = buttons[:, FORWARD] > 0.5
    row = int(np.argmin(held)) if not held.all() else 16
    assert held[:row].all() and not held[row:].any()
    return row


def test_the_keys_stay_in_the_controls_unless_live_wall_keys_says_otherwise():
    """``held`` (the default) does not move the block to find a stop: its buttons are the
    player's, and the rows are those a loop that was never asked gives."""
    loop = wall_loop()
    assert loop.wall_keys == "held"
    prior = [0]
    integrate = loop._integrate

    def counted(own, times):
        prior[0] += 1
        return integrate(own, times)

    loop._integrate = counted
    pressed = keys(FORWARD)
    prepare(loop, 1)
    before = prior[0]
    assert release(loop, 1, pressed) is pressed and prior[0] == before
    cut(loop, 1, pressed)
    assert prior[0] == before + 1
    camera, rows = block_motion(loop, 1)
    np.testing.assert_allclose(camera[:, 0], np.minimum(rows[:, 0], WALL_U - 16.0), atol=1e-4)


def test_a_block_that_runs_at_a_wall_loses_its_movement_keys_from_the_row_its_camera_stops_at():
    """The wall 120 u ahead stops block 1's camera 16 u before it: on the row the camera is first
    cut, the forward key goes, and with it the prior's run, which the keys of the rows before it
    set going and nothing carries on. Every other button, the weapon and the turn stay; the
    block's ticks, its published part and its own rows are those of the keys that remain."""
    reader = TickReader()
    held = live_loop(Room(), 0, reader=CameraReader(), depth=WALL_U)
    loop = live_loop(Room(), 0, reader=reader, depth=WALL_U, wall_keys="walls")
    pressed = keys(FORWARD, ATTACK)
    play(held, 1, pressed)
    _, run = block_motion(held, 1)
    row = int(np.argmax(run[:, 0] > WALL_U - 16.0))
    assert 0 < row < 15
    taken = play_released(loop, 1, pressed)
    assert stop_row(taken) == row
    assert np.array_equal(taken[:row], pressed[:row])
    assert np.array_equal(np.delete(taken, FORWARD, 1), np.delete(pressed, FORWARD, 1))
    assert np.array_equal(reader.buttons[-1], taken)
    part = loop.world_state.parts[MEDIA[0]][-1]
    assert np.array_equal(buttons_from_bits(part.buttons), taken)
    assert np.array_equal(part.weapon, np.full(16, AK47))
    camera, kept_run = block_motion(loop, 1)
    assert camera[:, 0].max() <= WALL_U - 16.0 + 1e-4
    steps = np.diff(np.concatenate([[0.0], kept_run[:, 0]]))
    assert (np.diff(steps[row:]) <= 1e-9).all() and steps[-1] < steps[row - 1]
    assert kept_run[-1, 0] < run[-1, 0] - 20.0
    assert np.array_equal(kept_run[:row], run[:row])


def test_the_rows_of_the_released_block_are_those_of_a_block_cut_with_the_keys_that_remain():
    """The block is moved twice, once with the keys pressed to find its stop row and once with the
    keys that remain: the rows, the camera, the prior and the own position are those of a loop
    that is given the remaining keys at once."""
    pressed = keys(FORWARD)
    walls = live_loop(Room(), 0, reader=CameraReader(), depth=WALL_U, wall_keys="walls")
    taken = play_released(walls, 1, pressed)
    assert 0 < stop_row(taken) < 16
    plain = live_loop(Room(), 0, reader=CameraReader(), depth=WALL_U)
    play(plain, 1, taken)
    for name in ("displacement", "prior_displacement", "yaw", "pitch"):
        assert np.array_equal(getattr(walls.motion, name), getattr(plain.motion, name)), name
    assert torch.equal(walls.batch["player_states"], plain.batch["player_states"])
    for key in ("player_control_substeps", "player_control_substep_valid"):
        assert torch.equal(walls.batch[key], plain.batch[key])
    for name in ("velocity", "hold", "flight"):
        assert torch.equal(
            getattr(walls.motion.prior_state, name), getattr(plain.motion.prior_state, name)
        )


def test_only_the_edge_of_the_extent_clears_the_keys_in_edge_mode():
    """On a strip of the map that ends at x = 768 u the blocks run on with their keys until the one
    whose camera the edge stops, which loses its forward key from that row on, and so does every
    block after it; a wall the depth head sees clears nothing."""
    loop = live_loop(Room(), 0, reader=CameraReader(), walkable=STRIP, wall_keys="edge")
    rows = []
    for s in range(1, LATENTS, BLOCK):
        rows.append(stop_row(play_released(loop, s, keys(FORWARD))))
    first = next(i for i, row in enumerate(rows) if row < 16)
    assert 0 < rows[first] < 16 and rows[:first] == [16] * first and first >= 2
    assert rows[first + 1 :] == [0] * (len(rows) - first - 1) and len(rows) > first + 1
    x = loop.batch["player_states"][0, 0, :, 0].numpy()
    # the edge is searched for along the block's way, in steps of 16 u: the camera of a block that
    # stands on the strip's last cell moves within that step, and the position behind it is
    # where the stop was found
    assert 750.0 < x.max() <= 768.0 and (np.diff(x) > -0.1).all()
    for mode, released in (("edge", False), ("walls", True)):
        loop = wall_loop_of(mode)
        pressed = keys(FORWARD)
        prepare(loop, 1)
        assert (release(loop, 1, pressed) is not pressed) is released


def wall_loop_of(mode: str) -> LiveLoop:
    return live_loop(Room(), 0, reader=CameraReader(), depth=WALL_U, wall_keys=mode)


def test_a_block_after_a_stop_is_cleared_from_its_first_row_while_the_plane_is_kept():
    """Block 5 stops 16 u before the wall the depth head finds from latent frame 4; from there the
    depth head no longer sees its lower part and the plane block 5 stopped at stops the blocks that
    push on: every one of them is released from row 0. Block 13 turns away (more than
    CONTINUATION_DEGREES over its rows) and keeps its keys."""
    loop = static_wall_loop(LostReader(), wall_keys="walls")
    assert stop_row(play_released(loop, 1, keys(FORWARD))) == 16
    contact = stop_row(play_released(loop, 5, keys(FORWARD)))
    assert 0 < contact < 16
    for s in (9, 13):
        assert stop_row(play_released(loop, s, keys(FORWARD))) == 0
    pressed = keys(FORWARD)
    prepare(loop, 17)
    assert release(loop, 17, pressed, turn=(0.0, 4.0)) is pressed
    assert stop_row(release(loop, 17, pressed)) == 0


def test_a_block_that_leaves_the_ground_keeps_its_keys_at_a_wall():
    loop = static_wall_loop(LostReader(), wall_keys="walls")
    play_released(loop, 1, keys(FORWARD))
    play_released(loop, 5, keys(FORWARD))
    jumping = keys(FORWARD)
    jumping[:2, JUMP] = 1.0
    prepare(loop, 9)
    assert release(loop, 9, jumping) is jumping
    assert stop_row(release(loop, 9, keys(FORWARD))) == 0


def test_a_block_with_a_movement_key_and_no_wall_is_the_same_whatever_wall_keys_is():
    walls = live_loop(Room(), 0, wall_keys="walls")
    held = live_loop(Room(), 0)
    for s, pressed in zip(range(1, LATENTS, BLOCK), [keys(FORWARD), keys(FORWARD, LEFT)] * 4):
        prepare(walls, s)
        assert release(walls, s, pressed) is pressed
        cut(walls, s, pressed)
        play(held, s, pressed)
    for name in ("displacement", "prior_displacement", "yaw", "pitch"):
        assert np.array_equal(getattr(walls.motion, name), getattr(held.motion, name)), name
    assert torch.equal(walls.batch["player_states"], held.batch["player_states"])
    assert np.array_equal(walls.cameras.track.xyz, held.cameras.track.xyz)


def test_a_player_who_pushes_a_wall_is_idle_in_the_controls_the_generator_reads():
    """After the block that meets the wall, a push on is a block with no button and no turn: the
    idle stretch's definition (:func:`~worldcast.engine.inference.idle_context.is_idle`), where the
    keys pressed are not."""
    loop = static_wall_loop(LostReader(), wall_keys="walls")
    play_released(loop, 1, keys(FORWARD))
    play_released(loop, 5, keys(FORWARD))
    for s in (9, 13, 17):
        pressed = keys(FORWARD)
        taken = play_released(loop, s, pressed)
        idle = BlockControls(taken, np.zeros((16, 2), np.float32), np.full(16, AK47))
        assert is_idle(idle) and not is_idle(BlockControls(pressed, idle.view_deltas, idle.weapon))


def test_the_own_position_is_bounded_at_the_top_speed_of_the_prior():
    """A place head 40 u ahead of the camera from latent frame 1 on would carry the own position a
    run's step and more on: it moves the prior's top speed (245 u/s) over a latent frame at most."""
    loop = live_loop(Room(), 0, reader=AheadReader())
    for s in range(1, LATENTS, BLOCK):
        play(loop, s, keys(FORWARD))
    top = PRIOR.top_speed / 4.0
    steps = np.linalg.norm(np.diff(loop.cameras.track.xyz[:, :2], axis=0), axis=1)
    assert loop.position.top_step_u == top == 61.25
    assert steps.max() == pytest.approx(top, abs=1e-9)


class NoisyReader(OracleReader):
    """The state model of a standing client whose place head is lost: 60 u of noise around a point
    1,000 u away, now and then 3,000 u off; the motion head 4 u of noise."""

    def __init__(self) -> None:
        super().__init__()
        self.rng = np.random.default_rng(0)

    def read(self, latents, latent_frames):
        n = len(latent_frames)
        place = self.rng.normal(0.0, 60.0, (n, 3)) + [1000.0, 0.0, 0.0]
        place[self.rng.random(n) < 0.3] += [0.0, 3000.0, 0.0]
        return place, self.rng.normal(0.0, 4.0 / 64.0, (n, 3)).astype(np.float32)


def test_a_standing_client_never_moves_whatever_its_state_model_reads():
    loop = live_loop(Room(), 0, reader=NoisyReader())
    for s in range(1, LATENTS, BLOCK):
        play(loop, s)
    assert (loop.batch["player_states"][0, 0, :, :3] == 0.0).all()
    assert (loop.cameras.track.xyz == 0.0).all()


# ---------------------------------------------------------------------------- the state model
class ControlsSeen:
    """Stands in for the state model: records the controls of every window it reads."""

    def __init__(self) -> None:
        self.controls = []

    def __call__(self, latents, controls, map_id):
        self.controls.append(controls[0].clone())
        frames = int(latents.shape[1])
        return SimpleNamespace(
            place=torch.zeros(1, frames, 3), displacement=torch.zeros(1, frames, 3)
        )


def test_the_state_model_reads_the_live_rows_as_ticks():
    recorded = tick_table(seconds=14.0)  # holds forward and turns, which live play replaces
    model = ControlsSeen()
    reader = LiveStateReader(
        model, recorded, map_name="de_nuke", source_fps=32.0, start_frame=96, device="cpu"
    )
    rng = np.random.default_rng(0)
    buttons = (rng.random((32, len(CONTROL_BUTTONS))) < 0.4).astype(np.float32)
    turn = rng.normal(0.0, 4.0, (32, 2)).astype(np.float32)
    reader.append(buttons, turn)
    for frames in ([0], [1, 2, 3, 4], [5, 6, 7, 8]):  # every latent frame is read once, in order
        reader.read(torch.zeros(9, 1, 1, 1), frames)
    controls = model.controls[0].numpy()  # window 0: latent frames 0 .. 40
    # latent frame 0 reads the recording, as the closed loop's reader does
    window = state_model_controls(
        timestamps=recorded.t,
        held_buttons=recorded.active,
        delta_pitch=recorded.delta_pitch,
        delta_yaw=recorded.delta_yaw,
        start_frame=96,
        source_fps=32.0,
        latent_frames=41,
    )
    assert np.array_equal(controls[0], window[0])
    # latent frames 1-8: one tick per substep of the rows, then idle ticks
    columns = [OPENCS2_BUTTONS.index(name) for name in CONTROL_BUTTONS]
    rows = controls[1:9].reshape(32, 4, 16)
    assert np.array_equal(rows[..., columns], np.repeat(buttons[:, None], 4, 1))
    np.testing.assert_allclose(
        rows[..., 13:15] * CAMERA_DELTA_SCALE * 4, turn[:, None].repeat(4, 1), atol=1e-5
    )
    assert (rows[..., 15] == 1.0).all()
    idle = controls[9:]
    assert (idle[..., :15] == 0.0).all() and (idle[..., 15] == 1.0).all()

    # the model is causal: later rows change nothing at the latent frames read before them
    torch.manual_seed(5)
    tiny = StateModel(load_cell_table(CELLS), dim=64, layers=2).eval()
    early, late = (
        LiveStateReader(
            tiny, recorded, map_name="de_nuke", source_fps=32.0, start_frame=96, device="cpu"
        )
        for _ in range(2)
    )
    early.append(buttons[:16], turn[:16])
    late.append(buttons, turn)
    latents = torch.randn(9, 48, 24, 42)
    for reader in (early, late):
        reader.read(latents, [0])
    for a, b in zip(early.read(latents, [1, 2, 3, 4]), late.read(latents, [1, 2, 3, 4])):
        assert np.array_equal(a, b)
    for reader in (early, late):
        reader.read(latents, [5, 6, 7])
    assert not np.array_equal(early.read(latents, [8])[0], late.read(latents, [8])[0])


# ------------------------------------------------------------------------- the other players
def test_another_live_player_is_drawn_from_its_rows_found_by_its_first_part():
    room = Room()
    walker, idle = live_loop(room, 0), live_loop(room, 1)
    for s in range(1, LATENTS, BLOCK):
        for loop, buttons in ((walker, keys(FORWARD)), (idle, keys())):
            play(loop, s, buttons)
    # the walker's first part reaches the idle client before its first block, which then draws
    # it from that block on where it published itself
    walker_rows = walker.batch["player_states"][0, 0]
    drawn = idle.batch["player_states"][0, 0]
    assert not idle.visibility.live[0, 0] and idle.visibility.live[1:, 0].all()
    np.testing.assert_allclose(drawn[1:, :3], walker_rows[1:, :3], atol=0.5)
    assert drawn[-1, 0] > 1500.0
    # the idle player's first part reaches the walker after its first block: nobody is in the
    # walker's world before that block, the first frame included, then the idle player is, and
    # stays put while the client walks
    shown = walker.visibility.live[:, 1]
    assert not shown[: video_frames_of(5, 1)[0]].any()
    assert shown[video_frames_of(5, LATENTS - 5)].all()
    assert (walker.batch["player_states"][0, 1, :, :3] == torch.tensor(POSITIONS[1])).all()
    # alive on every row: no corpse, ever
    assert (drawn[:, ALIVE_INDEX] == 1.0).all()
    assert walker.in_world(LATENTS - 1).tolist() == [False, True, False, False]
    assert walker.in_world(0).tolist() == [False] * 4


def test_a_part_that_arrives_before_the_cut_is_drawn_at_it():
    room = Room()
    a, b = live_loop(room, 0), live_loop(room, 1)
    for s in (1, 5):
        play(a, s)
        play(b, s, keys(FORWARD))
    prepare(a, 9)
    provisional = a.batch["player_states"][0, 1, video_frames_of(9, BLOCK), :3].clone()
    play(b, 9, keys(FORWARD))  # arrives between a's prepare and its cut
    refreshes = room.exchanges[MEDIA[0]].refreshes
    cut(a, 9)
    assert room.exchanges[MEDIA[0]].refreshes == refreshes + 1
    final = a.batch["player_states"][0, 1, video_frames_of(9, BLOCK), :3].numpy()
    published = room.exchanges[MEDIA[1]].live_part(MEDIA[1], 9).rows[:, :3]
    # the block before drew the peer from its older rows: the new part is blended in over the
    # block, all of its offset from them at the block's first video frame kept at the video frame
    # before the block, 0.75, 0.5, 0.25 and none of it at the block's latent frames
    assert np.abs(provisional.numpy() - published).max() > 1.0  # held keys, a block behind
    first = room.exchanges[MEDIA[0]].live_part(MEDIA[1], 1)  # all a had of b at its block 5's cut
    older = LivePeer(PRIOR)
    older.update(
        [PeerRows(first.times, first.rows, buttons_from_bits(first.buttons), first.weapon)]
    )
    drawn_from = older.at(session_times(9), np.zeros(2), final=False).rows[:, :3]
    offset = published[0] - drawn_from[0]
    assert np.abs(offset).max() > 100.0
    weights = np.interp(np.arange(16), [-1, 3, 7, 11, 15], [1.0, 0.75, 0.5, 0.25, 0.0])
    np.testing.assert_allclose(final, published - weights[:, None] * offset, atol=1e-3)
    assert np.abs(final[-1] - published[-1]).max() < 1e-3


def test_a_player_who_turns_96_degrees_a_block_is_drawn_on_every_block():
    """Each part arrives after the other client's cut, so every block turns the player 96 degrees
    from the rows the block before was drawn from: blended in, the player is never hidden and
    turns one way only."""
    room = Room()
    a, b = live_loop(room, 0), live_loop(room, 1)
    for s in range(1, LATENTS, BLOCK):
        play(a, s)
        play(b, s, keys(FORWARD), turn=(0.0, -6.0))
    shown = a.visibility.live[:, 1]
    assert not shown[: video_frames_of(5, 1)[0]].any()  # drawn from its first part on
    assert shown[video_frames_of(5, LATENTS - 5)].all()
    yaw = a.batch["player_states"][0, 1, video_frames_of(9, LATENTS - 9), 3].double()
    turned = wrap_degrees(yaw.diff())
    assert float(turned.max()) <= 1e-3 and float(turned.min()) >= -96.0 / 16 - 1e-3
    assert a.in_world(LATENTS - 1).tolist() == [False, True, False, False]


def test_lost_and_late_parts_never_raise():
    room = Room()
    a, b = live_loop(room, 0), live_loop(room, 1)
    rng = np.random.default_rng(1)
    for s in range(1, LATENTS, BLOCK):
        if rng.random() < 0.5:
            room.held.setdefault(MEDIA[1], [])
        for loop in rng.permutation([a, b]):
            play(loop, s, keys(FORWARD) if loop is b else keys(), clock=0.3 if loop is b else 0.0)
        if MEDIA[1] in room.held and rng.random() < 0.5:
            parts = room.held.pop(MEDIA[1])
            for part in parts[1:]:  # the first is lost
                room.send(part)
    drawn = a.batch["player_states"][0, 1, :, :3].numpy()
    assert np.isfinite(drawn).all() and (np.diff(drawn[17:, 0]) >= -1e-3).all()


def test_a_player_whose_client_is_done_is_hidden_from_the_next_block():
    room = Room()
    a, b = live_loop(room, 0), live_loop(room, 1)
    for s in (1, 5, 9):
        play(a, s)
        play(b, s)
    room.done[MEDIA[1]] = "ok"
    play(a, 13)
    assert a.visibility.live[video_frames_of(9, BLOCK), 1].all()
    assert not a.visibility.live[video_frames_of(13, BLOCK), 1].any()
    assert (a.batch["player_states"][0, 1, :, ALIVE_INDEX] == 1.0).all()


def test_a_player_who_takes_the_seat_of_one_who_left_is_drawn_where_it_is_not_slid_there():
    room = Room()
    a, b = live_loop(room, 0), live_loop(room, 1)
    for k, s in enumerate(range(1, 17, BLOCK)):  # b walks two blocks, then stands
        play(b, s, keys(FORWARD) if k < 2 else keys())
        play(a, s)
    left = a.batch["player_states"][0, 1, video_frames_of(13, BLOCK), 0]
    assert left[-1] > 1900.0 and a.visibility.live[video_frames_of(13, BLOCK), 1].all()
    room.done[MEDIA[1]] = "ok"
    for s in (17, 21):  # hidden from the next block on
        play(a, s)
        assert not a.visibility.live[video_frames_of(s, BLOCK), 1].any()
    # another player takes the seat, on the session's clock, where the seat spawns: its first
    # part reaches a, and the seat's done is cleared with it
    taker = live_loop(room, 1)
    play(taker, 1, clock=5.9)
    room.done.pop(MEDIA[1])
    play(a, 25)
    frames = video_frames_of(25, BLOCK)
    assert a.visibility.live[frames, 1].all()
    # drawn at its spawn from the block's first frame, not slid from where the leaver stood
    drawn = a.batch["player_states"][0, 1, frames, :3]
    assert (drawn[:, 0] == 1600.0).all() and (drawn[:, 1] == 600.0).all()


def test_a_client_alone_has_nobody_in_its_world_the_first_frame_included():
    room = Room()
    a = live_loop(room, 0)
    assert not a.visibility.live.any() and a.in_world(0).tolist() == [False] * 4
    for s in range(1, LATENTS, BLOCK):
        play(a, s, keys(FORWARD))
    # the recorded players stand at their spawns, 1600 u ahead and behind: none is in the world
    assert not a.visibility.live.any() and not a.in_world(LATENTS - 1).any()
    assert not a.peers


def test_a_seat_nobody_plays_is_nowhere_its_recorded_death_leaves_no_corpse_and_it_can_be_taken():
    room = Room()
    a = live_loop(room, 0, dies={1: 40})
    for s in range(1, 21, BLOCK):
        play(a, s)
    assert not a.visibility.live[:, 1].any()
    assert (a.batch["player_states"][0, 1, :, ALIVE_INDEX] == 1.0).all()
    # a human takes the seat on the session's clock: drawn from its first block, unblended, as
    # nothing was shown there
    b = live_loop(room, 1)
    for k, s in enumerate(range(21, LATENTS, BLOCK)):
        play(b, 1 + BLOCK * k, keys(FORWARD), clock=80 / 16)
        play(a, s)
    assert not a.visibility.live[:81, 1].any() and a.visibility.live[81:, 1].all()
    np.testing.assert_allclose(
        a.batch["player_states"][0, 1, 81:, :3], b.batch["player_states"][0, 1, 1:49, :3], atol=0.5
    )
    derive_death_state(a.batch["player_states"][:, :, ::4].permute(0, 2, 1, 3))  # no resurrection


# ------------------------------------------------------------------------- memory entries
def test_another_players_block_is_keyed_at_its_published_cameras():
    room = Room()
    a, b = live_loop(room, 0), live_loop(room, 1)
    for s in (1, 5):
        play(a, s, keys(FORWARD))
        play(b, s, keys(FORWARD, JUMP), turn=(0.5, -30.0), weapon=AWP)
    a.world_state.refresh()
    part = a.world_state.live_part(MEDIA[1], 5)
    block = a.block_at(1, 0, 5)
    rows = np.column_stack([part.rows[3::4], np.ones(4, np.float32)])
    assert torch.equal(block.c2w, c2w_from_state_rows(rows))
    assert (block.media_id, block.slot, block.f0) == (MEDIA[1], 1, 5)
    assert a.block_at(1, 0, 9) is None  # no part yet: the block is never admitted
    inputs = {"states": torch.zeros(16, 6), "kept": "as given"}
    peer = a.inputs_as_generated(inputs, block, part)
    assert peer["kept"] == "as given"
    assert torch.equal(peer["states"][:, :5], torch.from_numpy(part.rows))
    assert (peer["states"][:, 5] == 1.0).all()
    turn = np.tile(np.float32([0.5, -30.0]), (16, 1))
    deltas = np.stack([quantize_camera_delta(t, clip=False) for t in turn])
    assert np.array_equal(peer["view_deltas"].numpy(), deltas)
    assert np.array_equal(peer["buttons"].numpy(), keys(FORWARD, JUMP))
    values, _ = block_substeps(keys(FORWARD, JUMP), turn)
    assert np.array_equal(peer["control_substeps"].numpy(), values)
    assert peer["weapon_ids"].tolist() == [AWP] * 16
    assert (
        peer["obs_scope_valid"].tolist() == [0] * 4 and peer["obs_flash_valid"].tolist() == [1] * 4
    )
    assert torch.equal(peer["tans"], torch.tensor(half_angle_tangents()).expand(4, 2))
    # the client's own block: from its part, the rows and controls the round batch holds
    own_block = MemoryBlock.at(MEDIA[0], 0, 0, 5, a.cameras.for_block(5)[5:9])
    own = a.inputs_as_generated(inputs, own_block, a.world_state.live_part(MEDIA[0], 5))
    assert torch.equal(own["states"], own_rows(a, 5))
    assert own["obs_scope_valid"].tolist() == [1] * 4
    assert torch.equal(
        own["control_substeps"],
        torch.from_numpy(block_substeps(keys(FORWARD), np.zeros((16, 2)))[0]),
    )


def test_nothing_flashes_and_the_scope_of_a_weapon_that_zooms_is_unknown():
    signals = live_observer_signals(np.array([AK47, AWP, encode_weapon("aug"), 0]))
    assert {key: value.tolist() for key, value in signals.items()} == {
        "obs_flash_flag": [0, 0, 0, 0],
        "obs_flash_valid": [1, 1, 1, 1],
        "obs_scope_on": [0, 0, 0, 0],
        "obs_scope_level": [0, 0, 0, 0],
        "obs_scope_valid": [1, 0, 0, 1],
    }
    assert sorted(signals) == sorted(OBSERVER_SIGNAL_KEYS)
    assert all(value.dtype == np.int64 for value in signals.values())


# ------------------------------------------------------------------------------- visibility
def test_the_target_labels_are_tested_again_with_the_cameras_of_the_cut():
    """Prepared with no turn, the other player, standing ahead, is labelled visible on the whole
    target block; a half turn at the cut takes it out of view from the block's middle on. The
    surface of the context read when the window was gathered is the one read after the cut: the cut
    changes no camera or point of the context, so the labels are those of a test from scratch."""
    room = Room()
    a, b = live_loop(room, 0, latents=41), live_loop(room, 1, latents=41)
    for s in range(1, 25, BLOCK):
        play(a, s)
        play(b, s)
    s = 25
    prepare(a, s)
    labelled = a.visibility.for_block(s, a.batch, torch.zeros(41, 1, 1, 1))
    target = video_frames_of(s, BLOCK)
    window = {
        "client_visibility": labelled["client_visibility"][:, :, target].clone(),
        "client_visibility_valid": labelled["client_visibility_valid"][:, :, target].clone(),
    }
    assert window["client_visibility"][0, 1].all()
    cut(a, s, turn=(0.0, 180.0 / 16))
    retested = a.visibility.retest_target(window, s)
    visible = retested["client_visibility"][0, 1].bool()
    assert visible[:4].all() and not visible[-4:].any()
    first = s - RECENT
    from_scratch = PredictedVisibility.target_labels(
        a.visibility,
        target,
        a.visibility._depth(torch.zeros(41, 1, 1, 1)[first:s], first),
        video_frames_of(first, RECENT),
    )
    for key, labels in zip(("client_visibility", "client_visibility_valid"), from_scratch):
        assert torch.equal(retested[key][0].bool(), labels.T), key
    assert a.visibility.drawn(s).tolist() == [False, True, False, False]
    assert a.visibility.drawn(s + 3).tolist() == [False] * 4
    assert a.visibility.drawn(0) is None
    a.visibility.drawn(s)[:] = True  # the page's copy
    assert a.visibility.drawn(s).tolist() == [False, True, False, False]
    # another block labelled in between: its context is not block 25's
    a.visibility.target_visible(21, torch.zeros(41, 1, 1, 1))
    with pytest.raises(
        ValueError,
        match="block 25 is tested again right after its for_block; the last frames labelled are"
        " the video frames 81-96",
    ):
        a.visibility.retest_target(window, s)


def test_a_cut_with_the_cameras_its_block_was_prepared_with_reprojects_nothing(monkeypatch):
    """The reprojection of the context into the target frames' cameras is kept per set of
    cameras: a cut whose rows are those its block was prepared with, and a second test of the same
    cut, reproject nothing; a cut that turns reprojects once. The labels are those of a test from
    scratch either way."""
    room = Room()
    a, b = live_loop(room, 0, latents=41), live_loop(room, 1, latents=41)
    for s in range(1, 25, BLOCK):
        play(a, s)
        play(b, s)
    reprojected = []
    zbuffer = predicted_visibility_module._zbuffer

    def counted(*args, **kwargs):
        reprojected.append(1)
        return zbuffer(*args, **kwargs)

    monkeypatch.setattr(predicted_visibility_module, "_zbuffer", counted)
    s, target = 25, video_frames_of(25, BLOCK)
    prepare(a, s)
    labelled = a.visibility.for_block(s, a.batch, torch.zeros(41, 1, 1, 1))
    window = {key: labelled[key][:, :, target].clone() for key in WINDOW_LABELS}
    assert len(reprojected) == 1
    cut(a, s)
    first = a.visibility.retest_target(window, s)
    again = a.visibility.retest_target(window, s)
    assert len(reprojected) == 1
    for key in WINDOW_LABELS:
        assert torch.equal(first[key], again[key]) and torch.equal(first[key], window[key]), key
    a.cameras.track.xyz[s - 1] += 1.0  # the block's camera steps
    a._write_table(s)
    moved = a.visibility.retest_target(window, s)
    assert len(reprojected) == 2
    monkeypatch.setattr(predicted_visibility_module, "_zbuffer", zbuffer)
    context = a.visibility._context
    from_scratch = PredictedVisibility._reprojected_labels(a.visibility, *context)
    for key, labels in zip(WINDOW_LABELS, from_scratch):
        assert torch.equal(moved[key][0].bool(), labels.T), key


def test_the_cuts_z_test_is_ztest_labels_operation_by_operation():
    """The z-test of a cut (numpy, its nine test points at once) gives the labels of
    :func:`ztest_labels` (torch, test point by test point) to the bit: players behind, beside and
    in front of the cameras, on cell borders, in holes of the surface and exactly at its margin."""
    rng = np.random.default_rng(3)
    tans = half_angle_tangents()
    for case in range(60):
        r, p = int(rng.integers(1, 17)), int(rng.integers(1, 11))
        yaw, pitch = rng.uniform(-180, 180, r), rng.uniform(-60, 60, r)
        rows = np.column_stack([rng.uniform(-500, 500, (r, 3)), yaw, pitch, np.ones(r)])
        c2w = c2w_from_state_rows(rows).double()
        ahead = rng.uniform(-200, 3000, (r, p, 1)) * c2w[:, None, :3, 2].numpy()
        points = c2w[:, None, :3, 3].numpy() + ahead + rng.normal(0, 600, (r, p, 3))
        if case % 4 == 0:
            points = points.round()
        depth = rng.uniform(50, 4000, (r, *LATENT_GRID))
        depth[rng.random(depth.shape) < 0.3] = np.nan
        if case % 5 == 0:
            depth[:] = 777.0
        live = rng.random((r, p)) < 0.8
        want = ztest_labels(
            torch.from_numpy(points),
            c2w,
            torch.from_numpy(depth),
            live=torch.from_numpy(live),
            tans=tans,
            pose_radius=POSE_RADIUS_U,
        )
        got = live_module._ztest(points, c2w.numpy(), depth, live, tans)
        for expected, labels in zip(want, got):
            assert np.array_equal(expected.numpy(), labels), case


def test_the_view_angles_of_a_cut_are_the_camera_integral_operation_by_operation():
    """The view angles a cut integrates (several players at once) are those of
    :func:`integrate_camera_angles` of each player alone, to the bit: the yaw wrapped, the pitch
    clamped row by row, invalid substeps skipped."""
    rng = np.random.default_rng(4)
    for case in range(40):
        players, rows = int(rng.integers(1, 9)), int(rng.integers(1, 20))
        buttons = (rng.random((players, rows, len(CONTROL_BUTTONS))) < 0.3).astype(np.float32)
        turn = rng.normal(0, rng.choice([0.01, 2.0, 40.0]), (players, rows, 2)).astype(np.float32)
        packed = []
        for k in range(players):
            values, valid = block_substeps(buttons[k], turn[k])
            if case % 3 == 0:
                valid = rng.random(valid.shape) < 0.7
            packed.append(packed_substeps(values, valid))
            assert np.array_equal(
                packed[-1], pack_substeps(torch.from_numpy(values), torch.from_numpy(valid))
            )
        start = np.column_stack([rng.uniform(-720, 720, players), rng.uniform(-100, 100, players)])
        yaw, pitch = live_module._view_angles(start[:, 0], start[:, 1], np.stack(packed))
        for k in range(players):
            initial = torch.zeros(1, 1, 6)
            initial[0, 0, 3], initial[0, 0, 4] = float(start[k, 0]), float(start[k, 1])
            substeps = torch.cat([torch.zeros(1, 4, 14), torch.from_numpy(packed[k])])
            want_yaw, want_pitch = integrate_camera_angles(initial, substeps[None, None])
            assert np.array_equal(want_yaw[0, 0].numpy(), yaw[k]), case
            assert np.array_equal(want_pitch[0, 0].numpy(), pitch[k]), case


# ------------------------------------------------------------- a session that has no end
def test_every_other_player_stands_at_its_first_row_until_it_is_in_the_world():
    """The recorded players of the round move, and a round that slides could not follow them (the
    rows it vacates repeat its last): every other player's rows stand at its round start on every
    video frame, holding nothing, and so does the facing its first block starts turning from; they
    stay so while the client plays on alone."""
    recorded = round_batch(0, batch=1, latents=LATENTS, client_slots=(1,))
    assert (
        recorded["player_states"][0, :, 1:, :5] != recorded["player_states"][0, :, :1, :5]
    ).any()
    loop = live_loop(Room(), 1, moving=True)
    batch, others = loop.batch, [slot for slot in range(10) if slot != 1]

    def held() -> None:
        states, weapon = batch["player_states"][0, others], batch["player_weapon_ids"][0, others]
        assert (states[:, 1:, :5] == states[:, :1, :5]).all()
        assert (batch["player_control_substeps"][0, others, 1:] == 0.0).all()
        assert not batch["player_control_substep_valid"][0, others, 1:].any()
        assert (weapon[:, 1:] == weapon[:, :1]).all()
        for facing in loop._batch_facing:
            assert (facing[others] == facing[others, :1]).all()

    held()
    play(loop, 1, keys(FORWARD))
    play(loop, 5, keys(FORWARD, JUMP), turn=(0.0, 30.0))
    held()


def test_nobody_is_in_the_first_frame_whoever_joins_and_leaves():
    """The pin on latent frame 1 (the depth head reads the first frame with it) does not carry the
    first frame's label: nobody is in the world at video frame 0, however late a player joins and
    whenever it leaves, so the label of the first frame never reads the depth."""
    room = Room()
    a, b = live_loop(room, 0, latents=145), live_loop(room, 1, latents=145)
    for k in range(36):
        play(a, 1 + 4 * k, keys(FORWARD))
        if 12 <= k < 30:  # the second player's own blocks 0-17, 12 s late
            play(b, 1 + 4 * (k - 12), keys(FORWARD), clock=12.0)
        if k == 30:
            room.done[MEDIA[1]] = "ok"
        assert not a.visibility.live[0].any() and not b.visibility.live[0].any(), k
    assert a.visibility.live[video_frames_of(13, 1)[0] :, 1].any()  # it was in the world


#: The source frame of the state model tests' latent frame 0.
START = 96


def reference_controls(recorded, live_active, live_turn, w):
    """The state model's controls of window ``w`` from the whole tick stream: the recorded ticks up
    to latent frame 0, then the live ones, then idle ticks to the window's end."""
    first = STATE_MODEL_WINDOW_STRIDE * w
    kept = int(count_until(recorded.t, START / 32.0))
    anchor = recorded.t[kept - 1]
    last = source_frame(START, first + STATE_MODEL_WINDOW_LATENT_FRAMES - 1)
    ticks = max(len(live_active), int(np.ceil((last / 32.0 - anchor) * TICK_RATE)) + 1)
    turn = np.zeros((ticks, 2), np.float32)
    turn[: len(live_turn)] = live_turn
    controls = state_model_controls(
        timestamps=np.concatenate(
            [recorded.t[:kept], anchor + np.arange(1, ticks + 1) / TICK_RATE]
        ),
        held_buttons=list(recorded.active[:kept]) + live_active + [[]] * (ticks - len(live_active)),
        delta_pitch=np.concatenate([recorded.delta_pitch[:kept], turn[:, 0]]),
        delta_yaw=np.concatenate([recorded.delta_yaw[:kept], turn[:, 1]]),
        start_frame=source_frame(START, first),
        source_fps=32.0,
        latent_frames=STATE_MODEL_WINDOW_LATENT_FRAMES,
    )
    return torch.from_numpy(controls)[None].float()


def test_the_ticks_a_reader_trims_change_no_bit_of_any_windows_controls():
    """A reader that drops the live ticks and the windows it will not read again gives, over 1,100
    blocks (18 minutes, 28 state-model windows), the controls of the whole tick stream bit for bit,
    for every window at its first and its last read and at every 50th block; and it holds at most
    a window of ticks (10.25 s at 64 a second, and two of margin) and two windows."""
    recorded = tick_table(seconds=14.0)
    model = ControlsSeen()
    reader = LiveStateReader(
        model, recorded, map_name="de_nuke", source_fps=32.0, start_frame=START, device="cpu"
    )
    latents = torch.zeros(4500, 1, 1, 1)
    reader.read(latents, [0])  # block 1 is prepared before its cut
    rng = np.random.default_rng(5)
    active, turns, checked, most = [], np.zeros((0, 2), np.float32), 0, 0
    for k in range(1100):
        s = 1 + 4 * k
        buttons = (rng.random((16, len(CONTROL_BUTTONS))) < 0.25).astype(np.float32)
        turn = rng.normal(0.0, 3.0, (16, 2)).astype(np.float32)
        reader.append(buttons, turn)  # the cut of block s
        for row in buttons:
            active += [[CONTROL_BUTTONS[i] for i in np.flatnonzero(row > 0.5)]] * 4
        turns = np.concatenate([turns, np.repeat(turn / np.float32(4), 4, 0)])
        model.controls.clear()
        reader.read(latents, list(range(s, s + 4)))  # the prepare of the block after it
        most = max(most, len(reader._live_active))
        assert len(reader._windows) <= 2, k
        if k % 50 == 0 or s % 40 in (1, 37):
            (w, _), (controls,) = window_of(s), model.controls
            assert torch.equal(controls[None], reference_controls(recorded, active, turns, w))
            checked += 1
    assert checked >= 56 and most == 658 and reader._live_dropped > 69_000


def test_a_reader_gives_the_state_model_what_the_whole_session_gives_it_as_the_round_slides():
    """The latent frames a reader is given are the round's, which slides: it numbers them by the
    session's and keeps the window's itself. Read as the loop reads them (every latent frame once,
    in order, from a round that holds nothing else of the session: NaN), over 14 blocks (latent
    frames 0-56, two windows, a slide from the seventh block on), its outputs are those of the
    paper's reader on the whole session's latent frames, bit for bit; a reader that does not count
    the slides reads other frames."""
    torch.manual_seed(5)
    tiny = StateModel(load_cell_table(CELLS), dim=64, layers=2).eval()
    recorded = tick_table(seconds=14.0)

    def reader(cls) -> StateReader:
        return cls(
            tiny, recorded, map_name="de_nuke", source_fps=32.0, start_frame=START, device="cpu"
        )

    live, counting_no_slide, paper = (
        reader(LiveStateReader),
        reader(LiveStateReader),
        reader(StateReader),
    )
    paper._controls_of = live._controls_of  # the paper's reader, on the live reader's controls
    session = torch.randn(57, 48, 24, 42)
    rng = np.random.default_rng(1)
    for k in range(14):
        s = 1 + 4 * k
        buttons = (rng.random((16, len(CONTROL_BUTTONS))) < 0.4).astype(np.float32)
        turn = rng.normal(0.0, 4.0, (16, 2)).astype(np.float32)
        live.append(buttons, turn)
        if counting_no_slide is not None:
            counting_no_slide.append(buttons, turn)
        if s - live.offset > 25:  # the round slides once the next block does not fit
            live.offset += 4
        frames = [0] if k == 0 else list(range(s - 4, s))
        round_ = torch.full((29, 48, 24, 42), float("nan"))
        for f in frames:
            round_[f - live.offset] = session[f]
        mine = live.read(round_, [f - live.offset for f in frames])
        assert all(np.array_equal(a, b) for a, b in zip(mine, paper.read(session, frames))), k
        if counting_no_slide is not None:
            lost = counting_no_slide.read(round_, [f - live.offset for f in frames])
            if live.offset:
                assert not all(np.array_equal(a, b) for a, b in zip(lost, mine))
                counting_no_slide = None
