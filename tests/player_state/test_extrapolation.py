"""Extrapolation within a block: the physics prior, the position track, the client's cameras and
the round's table under predicted states."""

import numpy as np
import pytest
import torch

from tests.player_state.support import PRIOR, round_batch
from worldcast.data.controls import CONTROL_BUTTONS
from worldcast.data.game import EYE_HEIGHT
from worldcast.player_state.extrapolation import (
    JUMP_COLUMN,
    JUMP_U_PER_S,
    MOVEMENT_COLUMNS,
    ClientCameras,
    Motion,
    PositionTrack,
    PredictedStateTable,
    PriorState,
    _relaxed_velocity,
    extrapolate,
    physics_prior_displacement,
    physics_prior_track,
    player_motion,
)

N_LATENTS = 9
T_ROWS = 1 + 4 * (N_LATENTS - 1)


def _idle(rows: int = T_ROWS) -> torch.Tensor:
    """Packed substeps ``[1, rows, 4, 14]`` with no key held, all valid."""
    controls = torch.zeros(1, rows, 4, 14)
    controls[..., -1] = 1.0
    return controls


# -------------------------------------------------------------------------------- the physics prior
def test_no_keys_no_displacement():
    start = torch.tensor([[[10.0, 20.0, 30.0]]])
    track = physics_prior_track(_idle(), torch.zeros(1, T_ROWS), start, prior=PRIOR)
    assert torch.equal(track, start.expand(1, T_ROWS, 3))


def test_the_priors_top_speed_is_the_running_speed():
    """The fastest target speed of the table, which the speed ramp reaches after 24 substeps of a
    key held: 245 u/s."""
    assert PRIOR.top_speed == 245.0


def test_forward_moves_along_the_yaw_and_speeds_up():
    assert [CONTROL_BUTTONS[c] for c in MOVEMENT_COLUMNS] == [
        "forward",
        "back",
        "move_left",
        "move_right",
    ]
    controls = _idle()
    controls[0, :, :, MOVEMENT_COLUMNS[0]] = 1.0
    track = physics_prior_track(
        controls, torch.full((1, T_ROWS), 90.0), torch.zeros(1, 1, 3), prior=PRIOR
    )[0]
    assert torch.equal(track[0], torch.zeros(3))  # row 0's keys precede the window
    steps = track[1:, 1] - track[:-1, 1]
    # the speed ramp, up to the walking speed of the last column (245 u/s at 16 rows per second)
    assert bool((steps[1:] >= steps[:-1]).all()) and float(steps[0]) < 4.0
    assert float(steps[-1]) == pytest.approx(245.0 / 16.0, abs=1e-3)
    assert float(track[:, 0].abs().max()) < 1e-4 and not bool(track[:, 2].any())
    # an invalid substep freezes the integrator: the player stops where it is
    controls[0, 3:, :, -1] = 0.0
    frozen = physics_prior_track(
        controls, torch.full((1, T_ROWS), 90.0), torch.zeros(1, 1, 3), prior=PRIOR
    )[0]
    assert torch.equal(frozen[:3], track[:3])
    assert float(frozen[4, 1] - frozen[3, 1]) == pytest.approx(float(frozen[3, 1] - frozen[2, 1]))


@pytest.mark.parametrize(
    "keys, yaw, direction",
    [
        (("move_right",), 0.0, (0.0, -1.0)),  # looking along +x, the right is -y
        (("move_left",), 0.0, (0.0, 1.0)),
        (("back",), 0.0, (-1.0, 0.0)),
        (("move_right",), 90.0, (1.0, 0.0)),
        (("forward", "move_left"), 0.0, (0.7071, 0.7071)),
        (("move_left", "move_right"), 0.0, (0.0, 0.0)),  # they cancel
    ],
)
def test_the_movement_keys_move_the_player_in_its_own_frame(keys, yaw, direction):
    controls = _idle()
    for key in keys:
        controls[0, :, :, CONTROL_BUTTONS.index(key)] = 1.0
    track = physics_prior_track(
        controls, torch.full((1, T_ROWS), yaw), torch.zeros(1, 1, 3), prior=PRIOR
    )[0]
    moved = track[-1, :2]
    distance = float(moved.norm())
    if direction == (0.0, 0.0):
        assert distance == 0.0
    else:
        assert distance > 100.0  # two seconds at up to 245 u/s
        assert (moved / distance).tolist() == pytest.approx(list(direction), abs=1e-4)
    assert not bool(track[:, 2].any())


def test_a_jump_rises_at_the_engines_impulse_and_falls_under_gravity():
    controls = _idle()
    controls[0, 1, 0, JUMP_COLUMN] = 1.0
    z = physics_prior_track(controls, torch.zeros(1, T_ROWS), torch.zeros(1, 1, 3), prior=PRIOR)[
        0, :, 2
    ]
    # four 64 Hz substeps from the take-off: 260, 247.5, 235, 222.5 u/s
    assert float(z[1]) == pytest.approx((4 * JUMP_U_PER_S - 75.0) / 64.0)
    assert int(z.argmax()) == 5 and bool(z[8] < z[5])


def test_the_integrator_goes_on_from_the_state_a_call_left():
    """Running forward, a jump in row 10 and a frozen substep in row 12: integrated in two calls
    with the state the first left (mid-flight, at speed) it is the one call, bit for bit; the
    track is that displacement summed from video frame 1 on."""
    controls = _idle()
    controls[0, :, :, MOVEMENT_COLUMNS[0]] = 1.0
    controls[0, 10, 1, JUMP_COLUMN] = 1.0
    controls[0, 12, 2] = 0.0  # invalid: the integrator holds, the player moves on
    yaw = torch.linspace(0.0, 60.0, T_ROWS)[None]
    whole, end = physics_prior_displacement(controls, yaw, prior=PRIOR)
    first, middle = physics_prior_displacement(controls[:, :11], yaw[:, :11], prior=PRIOR)
    assert middle.hold.tolist() == [24] and middle.flight.tolist() == [37]
    assert float(middle.velocity[0, 2]) == JUMP_U_PER_S - 2 * 12.5
    second, last = physics_prior_displacement(
        controls[:, 11:], yaw[:, 11:], prior=PRIOR, state=middle
    )
    assert torch.equal(torch.cat([first, second], 1), whole)
    for key in ("velocity", "hold", "flight"):
        assert torch.equal(getattr(last, key), getattr(end, key))
    assert end.flight.tolist() == [0] and float(end.velocity[0, 2]) == 0.0
    track = physics_prior_track(controls, yaw, torch.zeros(1, 1, 3), prior=PRIOR)
    assert torch.equal(track[:, 1:] - track[:, :1], torch.cumsum(whole[:, 1:], 1))
    # the state at rest gives the track from rest
    rest = PriorState.at_rest(1)
    again, _ = physics_prior_displacement(controls, yaw, prior=PRIOR, state=rest)
    assert torch.equal(again, whole)


def _torch_relaxation(want: torch.Tensor, relaxes: torch.Tensor, velocity: torch.Tensor):
    """The relaxation of the planar velocity as torch runs it, substep by substep: the reference
    the prior's numpy scan follows operation by operation."""
    wanted = torch.linalg.norm(want, dim=-1, keepdim=True)
    up, down = torch.tensor(PRIOR.alpha_up), torch.tensor(PRIOR.alpha_down)
    out, planar = [], velocity
    for target, speed, relaxing in zip(want.unbind(1), wanted.unbind(1), relaxes.unbind(1)):
        if relaxing.any():
            alpha = torch.where(speed >= torch.linalg.norm(planar, dim=-1, keepdim=True), up, down)
            relaxed = planar + alpha * (target - planar)
            planar = torch.where(relaxing[:, None], relaxed, planar)
        out.append(planar)
    return torch.stack(out, 1)


def test_the_relaxation_is_torchs_bit_for_bit(monkeypatch):
    """The planar velocity relaxes as torch's float32 arithmetic has it, to the bit, wherever the
    velocity's own speed lies against the keys': far from them, so close that torch's norm decides,
    at rest, slowed to the smallest float32 velocities. Running at the keys' speed, the velocity
    stays a few units of rounding short of it, the same bits substep after substep: torch's norm is
    asked a few times over 400 substeps of three players."""
    norms, norm = [], torch.linalg.norm
    monkeypatch.setattr(torch.linalg, "norm", lambda *a, **k: norms.append(1) or norm(*a, **k))
    rng = np.random.default_rng(0)
    asked = 0
    for case in range(80):
        players, substeps = int(rng.integers(1, 12)), int(rng.choice([4, 64, 188, 400]))
        heading = np.deg2rad(rng.uniform(-180, 180, (players, 1)))
        heading = heading + np.cumsum(
            rng.normal(0, rng.choice([0.0, 0.01]), (players, substeps)), 1
        )
        speed = rng.choice([0.0, 34.0, 245.0, 111.8], (players, 1)) * (
            rng.random((players, 1)) < 0.9
        )
        speed = np.repeat(speed, substeps, 1)
        speed[:, substeps // 2 :] *= rng.choice([1.0, 0.5, 0.0])
        want = np.stack([speed * np.cos(heading), speed * np.sin(heading)], -1).astype(np.float32)
        relaxes = rng.random((players, substeps)) < rng.choice([1.0, 0.9])
        velocity = (rng.normal(0, 1, (players, 2)) * rng.choice([0.0, 250.0, 1e-20, 1e-43])).astype(
            np.float32
        )
        norms.clear()
        got = _relaxed_velocity(
            velocity, want, norm(torch.from_numpy(want), dim=-1).numpy(), relaxes, PRIOR
        )
        asked += len(norms)
        expected = _torch_relaxation(
            torch.from_numpy(want), torch.from_numpy(relaxes), torch.from_numpy(velocity)
        )
        assert np.array_equal(got.view(np.int32), expected.numpy().view(np.int32)), case
    assert asked > 0
    for heading in np.deg2rad([0.0, 30.0, 47.0, 90.0, 123.4, -150.0]):
        speeds = np.array([245.0, 111.8, 34.0])[:, None, None]
        want = np.broadcast_to(speeds * [np.cos(heading), np.sin(heading)], (3, 400, 2))
        want = np.ascontiguousarray(want, np.float32)
        norms.clear()
        _relaxed_velocity(
            np.zeros((3, 2), np.float32),
            want,
            norm(torch.from_numpy(want), dim=-1).numpy(),
            np.ones((3, 400), bool),
            PRIOR,
        )
        assert len(norms) <= 10


def _round_batch() -> dict:
    """One round of three living players."""
    batch = round_batch(0, batch=1, players=3, latents=N_LATENTS, client_slots=(0,))
    batch["player_states"][..., 5] = 1.0
    return batch


def test_player_motion_integrates_a_players_own_controls():
    batch = _round_batch()
    motion = player_motion(batch, 1, prior=PRIOR)
    assert motion.yaw.shape == motion.pitch.shape == (T_ROWS,)
    assert motion.displacement.shape == (T_ROWS, 3)
    assert motion.yaw.dtype == motion.displacement.dtype == np.float64
    assert np.array_equal(motion.start_state, batch["player_states"][0, 1, 0].double().numpy())
    assert motion.yaw[0] == float(batch["player_states"][0, 1, 0, 3])
    assert not motion.displacement[0].any() and motion.displacement[-1].any()
    two = {key: torch.cat([value, value]) for key, value in batch.items()}
    with pytest.raises(ValueError, match="batch size 1"):
        player_motion(two, 1, prior=PRIOR)


# ------------------------------------------------------------------------------- the position track
def test_a_track_grows_latent_frame_by_latent_frame():
    track = PositionTrack("m", 64, [1.0, 2.0, 3.0])
    assert len(track) == 1 and track.start_frame == 64
    track.append([1, 2], [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert len(track) == 3 and track.xyz[2].tolist() == [7.0, 8.0, 9.0]
    # the newest position published at a latent frame or at the one before it
    assert track.latest_published(2) == 2 and track.latest_published(3) == 2
    assert track.latest_published(4) is None and track.latest_published(-1) is None
    assert track.require_published(1) == 1
    with pytest.raises(ValueError, match="no position published at latent frame 4"):
        track.require_published(4)
    with pytest.raises(ValueError, match=r"latent frames \[4\] after 3 positions"):
        track.append([4], [[0.0, 0.0, 0.0]])  # latent frame 3 is missing
    with pytest.raises(ValueError, match="non-finite"):
        track.append([3], [[0.0, float("nan"), 0.0]])


def _linear_motion() -> Motion:
    """A player that moves 1 u along x per video frame and turns 1 degree per video frame."""
    frames = np.arange(T_ROWS, dtype=np.float64)
    displacement = np.stack([frames, np.zeros(T_ROWS), np.zeros(T_ROWS)], -1)
    return Motion(
        yaw=frames.copy(),
        pitch=np.zeros(T_ROWS),
        displacement=displacement,
        start_state=np.zeros(6),
    )


def test_extrapolate_adds_the_displacement_since_the_published_position():
    track = PositionTrack("m", 0, [100.0, 0.0, 0.0])
    track.append([1, 2], [[110.0, 0.0, 0.0], [120.0, 0.0, 0.0]])
    got = extrapolate(track, _linear_motion(), np.array([2, 2, 0]), np.array([8, 12, 3]))
    assert got[:, 0].tolist() == [120.0, 124.0, 103.0]


def test_client_cameras_extrapolate_the_block_and_hold_after_it():
    track = PositionTrack("m", 0, [100.0, 0.0, 0.0])
    cameras = ClientCameras(track, _linear_motion(), N_LATENTS)
    first = cameras.as_predicted()
    assert first.shape == (N_LATENTS, 4, 4)
    # block 1 from latent frame 0: latent frames 1 .. 4 at video frames 4 .. 16; later frames
    # hold
    assert first[:, 0, 3].tolist() == [100.0, 104.0, 108.0, 112.0, 116.0] + [116.0] * 4
    assert bool((first[:, 2, 3] == EYE_HEIGHT).all())
    assert cameras.as_predicted() is first  # cached until the track grows

    track.append([1, 2, 3, 4], [[105.0, 0.0, 0.0]] * 3 + [[130.0, 0.0, 0.0]])
    rows = cameras.block_rows(5)
    assert rows.shape == (4, 6) and rows[:, 0].tolist() == [134.0, 138.0, 142.0, 146.0]
    assert rows[:, 3].tolist() == [20.0, 24.0, 28.0, 32.0] and bool((rows[:, 5] == 1.0).all())
    known = cameras.for_block(5)
    assert known[:, 0, 3].tolist() == [
        100.0,
        105.0,
        105.0,
        105.0,
        130.0,
        134.0,
        138.0,
        142.0,
        146.0,
    ]
    assert cameras.as_predicted()[5:, 0, 3].tolist() == [134.0, 138.0, 142.0, 146.0]


def test_the_predicted_table_rewrites_the_clients_rows_block_by_block():
    batch = _round_batch()
    states = batch["player_states"]
    states[0, 1, 9:, 5] = 0.0  # slot 1 dies in block 3: its dead frames keep the recording
    recorded = states.clone()
    slots = {}
    for p in (0, 1):  # slot 2 is not a client
        motion = player_motion(batch, p, prior=PRIOR)
        slots[p] = (PositionTrack(f"m{p}", 0, motion.start_state[:3]), motion)
    table = PredictedStateTable(states, slots)
    assert table.track_of("m1") is slots[1][0] and table.track_of("m2") is None

    touched = table.advance(1)
    assert sorted(touched) == [0, 1] and touched[1].tolist() == list(range(1, 9))
    track, motion = slots[0]
    want = extrapolate(track, motion, np.zeros(16, np.int64), np.arange(1, 17))
    assert np.array_equal(states[0, 0, 1:17, :3].numpy(), want.astype(np.float32))
    assert torch.equal(states[0, 0, 17:, :3], states[0, 0, 16:17, :3].expand(T_ROWS - 17, 3))
    assert torch.equal(states[0, 2], recorded[0, 2]) and torch.equal(
        states[..., 3:5], recorded[..., 3:5]
    )
    assert torch.equal(states[0, 1, 9:, :3], recorded[0, 1, 9:, :3])

    held = states[0, 0, 16, :3].clone()
    table.advance(5)  # nothing published at latent frame 4: the client holds its position
    assert torch.equal(states[0, 0, 17:, :3], held.expand(T_ROWS - 17, 3))
