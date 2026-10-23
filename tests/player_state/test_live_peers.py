"""Other live players: a peer is drawn at the client's session times from the rows its own camera
used, extended past the newest by its held keys, then coasting, frozen and hidden; parts that
overlap in session time never step back; newer parts are blended in however far they turn, so a
peer that turns and runs off is drawn on every block, and only a jump farther than a peer runs
while it is drawn past its newest row hides one block; a peer hidden before is drawn at once, and
no part order raises. Every player alive at the first frame stays alive on every frame."""

import numpy as np
import pytest
import torch

from tests.player_state.support import PRIOR, standing_round
from worldcast.data.controls import CONTROL_BUTTONS, align_ticks_to_ordered_substeps
from worldcast.player_state.extrapolation import physics_prior_displacement, physics_prior_track
from worldcast.player_state.live_peers import (
    PEER_HIDE_S,
    PEER_HOLD_ROWS,
    PEER_HORIZON_ROWS,
    PEER_KEPT_ROWS,
    PEER_SNAP_U,
    LivePeer,
    PeerBlock,
    PeerRows,
    block_substeps,
    keep_alive,
    prior_displacements,
)
from worldcast.player_state.projection import table_angles
from worldcast.player_state.states import (
    PlayerState,
    continuous_row_state,
    pack_substeps,
    wrap_degrees,
)

FORWARD, JUMP, ATTACK = (CONTROL_BUTTONS.index(name) for name in ("forward", "jump", "attack"))
#: The peer's video frames are 1/16 s apart in session time; its first at this time, s.
T0 = 10.0


def _keys(rows: int, *held: int) -> np.ndarray:
    buttons = np.zeros((rows, len(CONTROL_BUTTONS)), np.float32)
    buttons[1:, list(held)] = 1.0
    return buttons


def _walk(buttons: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """A peer's own rows ``[n, 5]``: the physics prior's continuous track of its keys over all its
    video frames from the origin, its yaw, pitch 0."""
    values, valid = block_substeps(buttons, np.zeros((len(buttons), 2)))
    controls = pack_substeps(torch.from_numpy(values), torch.from_numpy(valid))[None]
    yaw32 = torch.from_numpy(np.asarray(yaw, np.float32))[None]
    xyz = physics_prior_track(controls, yaw32, torch.zeros(1, 1, 3), prior=PRIOR)[0].double()
    return np.column_stack([xyz.numpy(), yaw, np.zeros(len(yaw))])


def _runner(blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """The rows and keys of a peer that runs along x at the prior's full speed (forward held from
    long before its video frame 0, where it is at the origin)."""
    frames, run_up = 1 + 16 * blocks, 64
    buttons = _keys(frames + run_up, FORWARD)
    rows = _walk(buttons, np.zeros(frames + run_up))[run_up:]
    rows[:, 0] -= rows[0, 0]
    return rows, buttons[run_up:]


def _parts(
    rows: np.ndarray, buttons: np.ndarray, *, blocks: range, shift: float = 0.0
) -> list[PeerRows]:
    """The parts of the peer's blocks ``blocks`` (video frames ``1 + 16 k .. 16 + 16 k``)."""
    times = T0 + (np.arange(len(rows)) + shift) / 16.0
    return [
        PeerRows(times[s], rows[s], buttons[s], np.full(16, 3))
        for s in (slice(1 + 16 * k, 17 + 16 * k) for k in blocks)
    ]


def _sent(rows: np.ndarray, buttons: np.ndarray, cuts: np.ndarray, fps: float) -> list[PeerRows]:
    """The parts of the peer's blocks as it sends them: the times of block ``k``'s video frames are
    its controls cut ``cuts[k]`` minus ``(15 - j) / fps``, so that two cuts less than 15 frames
    apart give parts that overlap in session time."""
    times = cuts[:, None] - (15 - np.arange(16)) / fps
    spans = [slice(1 + 16 * k, 17 + 16 * k) for k in range(len(cuts))]
    return [PeerRows(t, rows[s], buttons[s], np.full(16, 3)) for t, s in zip(times, spans)]


def _times(first_row: int, shift: float = 0.0) -> np.ndarray:
    """The session times of the 16 video frames from ``first_row`` on the peer's clock."""
    return T0 + (first_row + np.arange(16) + shift) / 16.0


def _fresh(parts: list[PeerRows], times: np.ndarray) -> np.ndarray:
    """The rows a peer that has never been drawn gives at ``times``: unblended."""
    peer = LivePeer(PRIOR)
    peer.update(parts)
    return peer.at(times, np.zeros(2), final=False).rows


# ------------------------------------------------------------------------------------- substeps
def test_block_substeps_hold_the_buttons_and_split_the_turn():
    buttons = _keys(3, FORWARD, ATTACK)
    values, valid = block_substeps(buttons, [[0.0, 0.0], [4.0, -8.0], [-1.0, 30.0]])
    assert values.shape == (3, 4, 13) and values.dtype == np.float32
    assert valid.shape == (3, 4) and valid.all()
    np.testing.assert_array_equal(values[1, :, :11], np.tile(buttons[1], (4, 1)))
    np.testing.assert_array_equal(values[0, :, :11], 0.0)
    np.testing.assert_allclose(values[:, :, 11], [[0.0] * 4, [0.2] * 4, [-0.05] * 4], atol=1e-7)
    np.testing.assert_allclose(values[:, :, 12], [[0.0] * 4, [-0.4] * 4, [1.5] * 4], atol=1e-7)
    with pytest.raises(ValueError, match=r"buttons are \[n, 11\]"):
        block_substeps(np.zeros((3, 10)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match=r"turn \[n, 2\]"):
        block_substeps(np.zeros((3, 11)), np.zeros((2, 2)))


def test_block_substeps_are_the_tick_layout_with_one_tick_per_substep():
    rng = np.random.default_rng(0)
    frames = 6
    buttons = (rng.random((frames, 11)) < 0.4).astype(np.float32)
    turn = rng.normal(0.0, 5.0, (frames, 2)).astype(np.float32)
    ticks = 4 * frames
    aligned = align_ticks_to_ordered_substeps(
        timestamps=(1 + np.arange(ticks)) / 64.0 + 1.0 / 16.0,
        held_buttons=[
            [name for name, held in zip(CONTROL_BUTTONS, buttons[k // 4]) if held]
            for k in range(ticks)
        ],
        delta_pitch=np.repeat(turn[:, 0] / 4, 4),
        delta_yaw=np.repeat(turn[:, 1] / 4, 4),
        start_frame=4,
        num_frames=frames,
        source_fps=32.0,
        max_tick_gap_seconds=1.5 / 64,
    )
    values, valid = block_substeps(buttons, turn)
    np.testing.assert_array_equal(valid, aligned.valid)
    np.testing.assert_allclose(values, aligned.values, rtol=1e-6, atol=0)


def test_the_prior_of_several_players_in_one_call_is_each_one_alone():
    """The batched call pads the shorter inputs in front with invalid substeps: each player's
    displacement and integrator are, bit for bit, those of its input alone, from rest or from
    the integrator an earlier call left (a runner at full speed, airborne)."""
    rng = np.random.default_rng(1)
    run_up = _keys(40, FORWARD)
    run_up[-1, JUMP] = 1.0
    values, valid = block_substeps(run_up, np.zeros((40, 2)))
    _, running = physics_prior_displacement(
        pack_substeps(torch.from_numpy(values), torch.from_numpy(valid))[None],
        torch.zeros(1, 40),
        prior=PRIOR,
    )
    assert running.flight.tolist() == [36] and float(running.velocity[0, 0]) > 200.0
    inputs = []
    for rows, state in ((20, None), (48, running), (64, None)):
        buttons = (rng.random((rows, 11)) < 0.3).astype(np.float32)
        values, valid = block_substeps(buttons, np.zeros((rows, 2)))
        controls = pack_substeps(torch.from_numpy(values), torch.from_numpy(valid))
        yaw = torch.from_numpy(rng.normal(0, 40, rows).astype(np.float32))
        inputs.append((controls, yaw, state))
    results = prior_displacements(inputs, prior=PRIOR)
    assert [tuple(moved.shape) for moved, _ in results] == [(20, 3), (48, 3), (64, 3)]
    for (controls, yaw, state), (moved, after) in zip(inputs, results):
        alone, last = physics_prior_displacement(
            controls[None], yaw[None], prior=PRIOR, state=state
        )
        assert torch.equal(moved, alone[0])
        for key in ("velocity", "hold", "flight"):
            assert torch.equal(getattr(after, key), getattr(last, key))


# ------------------------------------------------------------------------------------------ alive
def test_every_player_alive_at_the_first_frame_is_alive_on_every_frame_never_a_corpse():
    states = torch.zeros(1, 4, 6, 6)
    states[0, 0, :, 5] = 1.0  # alive throughout
    states[0, 1, :3, 5] = 1.0  # dies at video frame 3
    states[0, 2, :, :3] = 50.0  # dead from frame 0; slot 3 has no recording
    assert keep_alive(states) is None
    assert states[0, :, :, 5].tolist() == [[1.0] * 6, [1.0] * 6, [0.0] * 6, [0.0] * 6]
    assert (states[0, 2, :, :3] == 50.0).all()


# ----------------------------------------------------------------------------------- the timeline
def test_peer_rows_refuse_other_shapes_and_times_that_do_not_increase():
    with pytest.raises(ValueError, match="a peer's rows are"):
        PeerRows(np.zeros(16), np.zeros((16, 6)), np.zeros((16, 11)), np.zeros(16))
    with pytest.raises(ValueError, match="a peer's rows are"):
        PeerRows(np.zeros(16), np.zeros((16, 5)), np.zeros((15, 11)), np.zeros(16))
    with pytest.raises(ValueError, match="a peer's rows are increasing times"):
        PeerRows([1.0, 1.0], np.zeros((2, 5)), np.zeros((2, 11)), np.zeros(2))
    with pytest.raises(ValueError, match=r"\[n >= 1\]"):
        PeerRows(np.zeros(0), np.zeros((0, 5)), np.zeros((0, 11)), np.zeros(0))


def test_a_peer_without_rows_or_a_block_of_other_length_is_refused():
    peer = LivePeer(PRIOR)
    with pytest.raises(ValueError, match="none has arrived"):
        peer.at(_times(1), np.zeros(2), final=True)
    rows = np.zeros((17, 5))
    peer.update(_parts(rows, _keys(17), blocks=range(1)))
    with pytest.raises(ValueError, match="a block has 16 times"):
        peer.at(_times(1)[:12], np.zeros(2), final=True)


def test_between_rows_the_peer_is_interpolated_and_turns_along_the_shortest_arc():
    rows = np.zeros((17, 5))
    rows[:, 0] = 100.0 * np.arange(17)
    rows[:, 3] = np.where(np.arange(17) % 2, -170.0, 170.0)
    rows[:, 4] = np.arange(17)
    buttons = _keys(17)
    buttons[1::2, ATTACK] = 1.0
    peer = LivePeer(PRIOR)
    peer.update([PeerRows(T0 + np.arange(17) / 16.0, rows, buttons, np.arange(17))])
    block = peer.at(_times(0, shift=0.5), np.zeros(2), final=False)
    np.testing.assert_allclose(block.rows[:, 0], 50.0 + 100.0 * np.arange(16))
    np.testing.assert_allclose(block.rows[:, 3], [-180.0] * 16)
    np.testing.assert_allclose(block.rows[:, 4], 0.5 + np.arange(16))
    assert block.buttons[:, ATTACK].tolist() == [0.0, 1.0] * 8
    assert block.weapon.tolist() == list(range(16))
    assert not block.snapped and block.present.all()  # a new peer is drawn from its first block


def test_a_peer_holding_w_is_its_own_continuous_prior_track():
    frames = 1 + 16 * 6
    buttons = _keys(frames)
    buttons[21:41, CONTROL_BUTTONS.index("move_left")] = 1.0
    buttons[21:, FORWARD] = 1.0
    yaw = np.zeros(frames)
    yaw[21:41] = 3.0 * np.arange(20)
    yaw[41:] = yaw[40]
    rows = _walk(buttons, yaw)
    peer = LivePeer(PRIOR)
    peer.update(_parts(rows, buttons, blocks=range(5)))  # video frames up to 80
    block = peer.at(_times(73), np.zeros(2), final=True)  # 73-80 received, 81-88 extended
    assert float(np.abs(block.rows[:, :3] - rows[73:89, :3]).max()) <= 0.5
    np.testing.assert_allclose(block.rows[:, 3], yaw[73:89])


def test_keys_are_held_8_rows_then_the_peer_coasts_to_rest():
    frames = 1 + 16 * 3
    buttons = _keys(frames, FORWARD, ATTACK)
    buttons[-1, JUMP] = 1.0
    rows = _walk(buttons, np.full(frames, 30.0))
    peer = LivePeer(PRIOR)
    peer.update(_parts(rows, buttons, blocks=range(3)))
    past = [peer.at(_times(48 + 16 * k), np.zeros(2), final=False) for k in range(3)]
    held = np.concatenate([b.buttons for b in past])[: PEER_HORIZON_ROWS + 1]  # newest row first
    assert held[:, FORWARD].tolist() == [1.0] * (PEER_HOLD_ROWS + 1) + [0.0] * 24
    assert held[:, ATTACK].tolist() == [1.0] * (PEER_HOLD_ROWS + 1) + [0.0] * 24
    assert held[:, JUMP].tolist() == [1.0] + [0.0] * PEER_HORIZON_ROWS
    assert not held[PEER_HOLD_ROWS + 1 :].any()
    xy = np.concatenate([b.rows[:, :2] for b in past])
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    running = np.linalg.norm(rows[48, :2] - rows[47, :2])
    np.testing.assert_allclose(speed[:PEER_HOLD_ROWS], running, rtol=1e-3)
    coasting = speed[PEER_HOLD_ROWS:PEER_HORIZON_ROWS]  # airborne until the jump lands
    assert (np.diff(coasting) <= 1e-9).all() and coasting[4] < 0.5 * running
    assert coasting[-1] < 1e-3 and not speed[PEER_HORIZON_ROWS:].any()
    np.testing.assert_allclose(np.concatenate([b.rows[:, 3] for b in past]), 30.0)


def test_the_extension_advances_at_the_peers_own_frame_period():
    frames = 1 + 16 * 4
    buttons = _keys(frames, FORWARD)
    rows = _walk(buttons, np.zeros(frames))
    times = T0 + np.arange(frames) / 15.0  # a peer at 15 fps
    peer = LivePeer(PRIOR)
    spans = (slice(1, 17), slice(17, 33), slice(33, 49))
    peer.update([PeerRows(times[s], rows[s], buttons[s], np.zeros(16)) for s in spans])
    block = peer.at(times[48:64], np.zeros(2), final=True)  # its newest row, then 15 past it
    # the keys are held over the first 8 rows past the newest: the peer's own track at 8/15 s
    held = slice(0, PEER_HOLD_ROWS + 1)
    assert float(np.abs(block.rows[held, :3] - rows[48 : 49 + PEER_HOLD_ROWS, :3]).max()) <= 0.5
    assert float(np.linalg.norm(rows[56, :2] - rows[48, :2])) > 100.0


def test_the_newest_48_rows_are_kept():
    rows, buttons = _runner(4)
    peer = LivePeer(PRIOR)
    peer.update(_parts(rows, buttons, blocks=range(4)))  # video frames 1-64: 17-64 are kept
    block = peer.at(_times(9), np.zeros(2), final=False)
    assert PEER_KEPT_ROWS == 48
    assert block.present.tolist() == [False] * 8 + [True] * 8
    np.testing.assert_array_equal(block.rows[:8], np.repeat(rows[17:18], 8, 0))


def test_a_silent_peer_is_frozen_after_32_rows_and_hidden_after_3_s():
    frames = 1 + 16 * 2
    buttons = _keys(frames, FORWARD)
    rows = _walk(buttons, np.zeros(frames))
    peer = LivePeer(PRIOR)
    peer.update(_parts(rows, buttons, blocks=range(2)))
    peer.at(_times(17), np.zeros(2), final=True)
    newest = T0 + 32 / 16.0
    frozen = peer.at(newest + (PEER_HORIZON_ROWS + np.arange(16)) / 16.0, np.zeros(2), final=False)
    assert (frozen.rows == frozen.rows[0]).all() and frozen.present.all()
    assert not frozen.buttons.any()
    late = peer.at(newest + PEER_HIDE_S - 0.5 + np.arange(16) / 16.0, np.zeros(2), final=False)
    assert PEER_HIDE_S == 3.0 and late.present.tolist() == [True] * 9 + [False] * 7
    assert (late.rows == frozen.rows[0]).all()
    # back where it stood, 4 s after its newest row: the track before the silence is dropped
    standing = np.repeat(frozen.rows[:1], 16, 0)
    peer.update([PeerRows(newest + 4.0 + np.arange(16) / 16.0, standing, _keys(16), np.zeros(16))])
    gone = peer.at(newest + 0.5 + np.arange(16) / 16.0, np.zeros(2), final=False)
    assert not gone.snapped and not gone.present.any()


@pytest.mark.parametrize("silent", [True, False])
def test_a_peer_back_after_it_was_hidden_is_drawn_at_once_unblended(silent):
    # silent: no row for 4 s, then back 200 u away; else its parts were held up 4 s in delivery
    # while it walked 200 u: either way the client's last block hid it
    rows, buttons = np.zeros((1 + 16 * 6, 5)), _keys(1 + 16 * 6)
    rows[17:, 0] = np.minimum(np.arange(len(rows) - 17) * 6.25, 200.0)
    parts = _parts(rows, buttons, blocks=range(6), shift=64.0 if silent else 0.0)
    parts[0] = _parts(rows, buttons, blocks=range(1))[0]
    peer = LivePeer(PRIOR)
    peer.update(parts[:1])
    drawn = peer.at(_times(9), np.zeros(2), final=True)
    assert drawn.present.all()
    hidden = peer.at(_times(73), drawn.rows[-1, 3:5], final=True)  # 3.56 s past its newest row
    assert not hidden.present.any() and not hidden.snapped
    peer.update(parts[1:])
    times = _times(89 + (64 if silent else 0))
    back = peer.at(times, hidden.rows[-1, 3:5], final=True)
    assert not back.snapped and back.present.all()
    np.testing.assert_array_equal(back.rows, _fresh(parts, times))
    np.testing.assert_allclose(back.rows[:, 0], 200.0)


@pytest.mark.parametrize("hidden", [True, False])
def test_a_player_in_the_seat_of_one_who_left_is_drawn_at_once_only_if_the_block_hid_the_leaver(
    hidden,
):
    # a runner, drawn on its first block; its player is done on the next (hidden), where it
    # would have run on; another player takes the seat and stands at the origin, 1 s on
    rows, buttons = _runner(4)
    standing = np.zeros_like(rows)
    peer = LivePeer(PRIOR)
    peer.update(_parts(rows, buttons, blocks=range(1)))
    drawn = peer.at(_times(17), np.zeros(2), final=True)
    assert drawn.present.all() and drawn.rows[-1, 0] > 100.0
    second = peer.at(_times(33), drawn.rows[-1, 3:5], final=True, hidden=hidden)
    assert second.present.any() != hidden and not second.snapped
    peer.update(_parts(standing, _keys(len(standing)), blocks=range(2, 4)))
    back = peer.at(_times(41), second.rows[-1, 3:5], final=True)
    assert back.present.all() and not back.snapped
    if hidden:  # nothing drawn is slid from: at its spot on every frame
        np.testing.assert_array_equal(back.rows[:, :3], 0.0)
    else:  # the leaver's last place, blended to the spot over the block
        assert back.rows[0, 0] > 100.0 and back.rows[-1, 0] == 0.0


@pytest.mark.parametrize("early", [0.1, 0.4])
def test_parts_that_overlap_in_session_time_never_step_back(early):
    fps, blocks = 15.0, 8
    rows, buttons = _runner(blocks)
    # the peer's controls cut came `early` s before 16 frames after the last: its pacing lead grew
    cuts = 100.0 + np.cumsum([16 / fps] * 3 + [16 / fps - early] + [16 / fps] * 4)
    parts = _sent(rows, buttons, cuts, fps)
    assert parts[3].times[0] < parts[2].times[-1]
    peer, previous, drawn = LivePeer(PRIOR), np.zeros(2), []
    for k in range(blocks):  # the client's block draws half inside the received rows
        peer.update(parts[k : k + 1])
        block = peer.at(cuts[0] + (16 * k - 7 + np.arange(16)) / fps, previous, final=True)
        previous = block.rows[-1, 3:5]
        assert block.present.all()
        drawn.append(block.rows[:, 0])
    assert (np.diff(np.concatenate(drawn)) > 0.0).all()
    # all of them at once, read every 1/63 s across the overlap: the newer part's rows count from
    # its first time on
    peer = LivePeer(PRIOR)
    peer.update(parts[1:5][::-1])
    t = np.linspace(parts[3].times[0] - 0.5, parts[3].times[0] + 0.5, 64).reshape(4, 16)
    x = np.concatenate([peer.at(times, np.zeros(2), final=False).rows[:, 0] for times in t])
    assert (np.diff(x) > 0.0).all()
    newer = np.concatenate([_fresh(parts[3:4], times)[:, 0] for times in t])
    on_the_newer = t.ravel() >= parts[3].times[0]
    np.testing.assert_allclose(x[on_the_newer], newer[on_the_newer])


# ------------------------------------------------------------------------------- blocks in a row
def test_a_new_peer_is_drawn_from_its_first_block_unblended():
    rows, buttons = _runner(3)
    peer = LivePeer(PRIOR)
    peer.update(_parts(rows, buttons, blocks=range(2)))
    first = peer.at(_times(25), np.zeros(2), final=False)
    assert not first.snapped and first.present.all()
    first = peer.at(_times(25), np.zeros(2), final=True)
    peer.update(_parts(rows, buttons, blocks=range(2, 3)))
    second = peer.at(_times(41), first.rows[-1, 3:5], final=True)
    assert not second.snapped and second.present.all()


def test_a_newer_part_that_moves_the_peer_is_blended_over_the_block():
    rows, buttons = _runner(6)
    moved = rows.copy()
    moved[65:81, 1] += 120.0  # the peer's own correction, from its video frame 65 on
    moved[65:81, 3] += 20.0
    early = _parts(rows, buttons, blocks=range(4))
    late = _parts(moved, buttons, blocks=range(4, 5))
    peer = LivePeer(PRIOR)
    peer.update(early)
    drawn = peer.at(_times(57), np.zeros(2), final=True)
    peer.update(late)
    block = peer.at(_times(73), drawn.rows[-1, 3:5], final=True)
    new, old = _fresh(early + late, _times(73)), _fresh(early, _times(73))
    delta = new[0] - old[0]
    assert not block.snapped and block.present.all()
    assert 100.0 < np.hypot(*delta[:2]) < 200.0 and 15.0 < delta[3] < 25.0
    np.testing.assert_allclose(block.rows[0], old[0] + delta / 16, atol=1e-9)
    np.testing.assert_allclose(block.rows[-1], new[-1], atol=1e-9)
    # per latent frame (video frames 3, 7, 11, 15): at most a quarter of the offset beyond the
    # rows' own step (the first also carries the earlier rows' step to the block)
    steps = np.diff(np.concatenate([drawn.rows[-1:], block.rows[3::4]]), axis=0)
    own = np.diff(np.concatenate([new[:1], new[3::4]]), axis=0)
    own[0] += old[0] - drawn.rows[-1]
    bound = np.hypot(*delta[:2]) / 4 + np.linalg.norm(own[:, :2], axis=1)
    assert (np.linalg.norm(steps[:, :2], axis=1) <= bound + 1e-9).all()
    assert (np.abs(steps[:, 3]) <= abs(delta[3]) / 4 + np.abs(own[:, 3]) + 1e-9).all()


def _jump(jump: float) -> tuple[PeerBlock, PeerBlock, np.ndarray, np.ndarray, np.ndarray]:
    """A running peer drawn up to its video frame 72 from its parts up to 64, whose next part puts
    it ``jump`` u aside from its video frame 65 on: the client's next two blocks (video frames
    73-88 and 89-104), and the rows a peer never drawn gives at the first from the parts with the
    jump and without it, and at the second with it."""
    rows, buttons = _runner(7)
    moved = rows.copy()
    moved[65:, 1] += jump
    early, late = _parts(rows, buttons, blocks=range(4)), _parts(moved, buttons, blocks=range(4, 5))
    peer = LivePeer(PRIOR)
    peer.update(early)
    drawn = peer.at(_times(57), np.zeros(2), final=True)
    assert drawn.present.all()
    peer.update(late)
    block = peer.at(_times(73), drawn.rows[-1, 3:5], final=True)
    after = peer.at(_times(89), block.rows[-1, 3:5], final=True)
    fresh = _fresh(early + late, _times(73)), _fresh(early, _times(73))
    return block, after, *fresh, _fresh(early + late, _times(89))


def test_a_jump_of_600_u_is_blended_over_the_block():
    """600 u, more than two blocks of running, is less than a peer runs while it is drawn past
    its newest row: blended in, the peer slides there over the block."""
    block, after, new, old, later = _jump(600.0)
    np.testing.assert_allclose(new[0] - old[0], [4.9437, 600.0, 0.0, 0.0, 0.0], atol=1e-4)
    assert not block.snapped and block.present.all()
    np.testing.assert_allclose(block.rows[0], old[0] + (new[0] - old[0]) / 16, atol=1e-9)
    np.testing.assert_allclose(block.rows[-1], new[-1], atol=1e-9)
    assert not after.snapped and after.present.all()
    np.testing.assert_allclose(after.rows, later, atol=1e-9)


def test_a_jump_of_1100_u_hides_one_block():
    """1100 u is more than a peer runs while it is drawn past its newest row: its rows jumped, and
    the block is hidden instead of sliding there; the next block draws it."""
    block, after, new, old, later = _jump(1100.0)
    assert PEER_SNAP_U == 1024.0
    np.testing.assert_allclose(new[0] - old[0], [4.9437, 1100.0, 0.0, 0.0, 0.0], atol=1e-4)
    assert block.snapped and not block.present.any()
    np.testing.assert_array_equal(block.rows, new)
    assert not after.snapped and after.present.all()
    np.testing.assert_allclose(after.rows, later, atol=1e-9)


@pytest.mark.parametrize("turn", [10.0, 50.0, 120.0, 179.0])
def test_a_turn_of_any_size_is_blended_along_the_shortest_arc_across_the_seam(turn):
    rows = np.zeros((1 + 16 * 6, 5))
    rows[:, 3] = 175.0
    moved = rows.copy()
    moved[65:, 3] = wrap_degrees(torch.tensor(175.0 + turn)).item()  # -175 for a turn of 10
    buttons = _keys(len(rows))
    early, late = _parts(rows, buttons, blocks=range(4)), _parts(moved, buttons, blocks=range(4, 5))
    peer = LivePeer(PRIOR)
    peer.update(early)
    drawn = peer.at(_times(57), np.zeros(2), final=True)
    peer.update(late)
    block = peer.at(_times(73), drawn.rows[-1, 3:5], final=True)
    assert not block.snapped and block.present.all()
    along = wrap_degrees(torch.from_numpy(block.rows[:, 3] - 175.0)).numpy()  # through 180
    np.testing.assert_allclose(along, turn * (1 + np.arange(16)) / 16, atol=1e-9)


def test_a_standing_peer_drawn_a_second_late_that_turns_and_runs_off_is_drawn_on_every_block():
    """A peer stands facing 161 degrees, then turns 49 degrees in half a second and runs, and the
    client draws it about a second late (each part two blocks after its own): when the turn
    arrives the peer has run 300 u from where the client draws it standing, and the newer rows
    move it 198 u and turn it 49 degrees there. Blended from where the client drew it, the peer
    turns and runs off on the client's view frame by frame instead of vanishing for a block, never
    faster than it runs."""
    frames = 1 + 16 * 7
    buttons = _keys(frames)
    buttons[41:, FORWARD] = 1.0
    yaw = np.full(frames, 161.0)
    yaw[41:49] = 161.0 - 6.125 * np.arange(1, 9)
    yaw[49:] = yaw[48]
    rows = _walk(buttons, yaw)
    parts = _parts(rows, buttons, blocks=range(7))
    peer, previous, drawn = LivePeer(PRIOR), np.zeros(2), []
    for k in range(2, 7):
        peer.update(parts[: k - 1])  # the newest part is block k - 2's: two blocks late
        block = peer.at(_times(1 + 16 * k), previous, final=True)
        assert block.present.all() and not block.snapped
        previous = block.rows[-1, 3:5]
        drawn.append(block.rows)
    x = np.concatenate(drawn)  # video frames 33-112
    assert np.hypot(*(rows[65, :2] - x[32, :2])) == pytest.approx(301.48, abs=0.01)
    seam = _fresh(parts[:3], _times(65))[0] - _fresh(parts[:2], _times(65))[0]
    assert seam[3] == -49.0 and np.hypot(*seam[:2]) == pytest.approx(198.43, abs=0.01)
    np.testing.assert_array_equal(x[:32], np.tile([0.0, 0.0, 0.0, 161.0, 0.0], (32, 1)))
    np.testing.assert_allclose(x[32:48, 3], 161.0 - 49.0 * (1 + np.arange(16)) / 16, atol=1e-9)
    np.testing.assert_array_equal(x[48:, 3], 112.0)
    # the largest step of a video frame is about the peer's own run (15.31 u a frame)
    assert np.hypot(*np.diff(rows[:, :2], axis=0).T).max() == pytest.approx(15.31, abs=0.01)
    assert np.hypot(*np.diff(x[:, :2], axis=0).T).max() == pytest.approx(15.34, abs=0.01)
    np.testing.assert_allclose(x[-1], _fresh(parts[:5], _times(97))[-1], atol=1e-9)


def test_a_provisional_block_leaves_nothing_behind():
    rows, buttons = _runner(6)
    times = T0 + np.arange(len(rows)) / 16.0
    # parts of 8 rows, each a correction of the peer's own track: every one moves the peer
    spans = [slice(1 + 8 * j, 9 + 8 * j) for j in range(12)]
    halves = [
        PeerRows(times[s], rows[s] + [0.0, 30.0 * j, 0.0, 4.0 * j, 0.0], buttons[s], np.full(8, 3))
        for j, s in enumerate(spans)
    ]
    drawn = {}
    for probed in (False, True):
        peer, previous, blocks = LivePeer(PRIOR), np.zeros(2), []
        for k in range(6):
            peer.update(halves[2 * k : 2 * k + 1])  # at the client's prepare
            if probed:
                peer.at(_times(16 * k + 7) + 0.1, previous, final=False)
            peer.update(halves[2 * k + 1 : 2 * k + 2])  # before the controls cut
            blocks.append(peer.at(_times(16 * k + 7), previous, final=True))
            previous = blocks[-1].rows[-1, 3:5]
        drawn[probed] = blocks
    assert not any(block.snapped for block in drawn[False])
    for plain, probed in zip(drawn[False], drawn[True]):
        np.testing.assert_array_equal(plain.rows, probed.rows)
        np.testing.assert_array_equal(plain.values, probed.values)
        np.testing.assert_array_equal(plain.present, probed.present)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lost_late_duplicated_and_reordered_parts_never_raise_nor_step_back(seed):
    rng = np.random.default_rng(seed)
    blocks = 30
    rows, buttons = _runner(blocks)
    arrivals: dict[int, list[PeerRows]] = {}
    for k, part in enumerate(_parts(rows, buttons, blocks=range(blocks))):
        if rng.random() < 0.2:
            continue  # lost
        for _ in range(2 if rng.random() < 0.2 else 1):  # duplicated, each copy late or not
            arrivals.setdefault(k + 1 + int(rng.integers(0, 4)), []).append(part)
    peer, previous, drawn, seen = LivePeer(PRIOR), np.zeros(2), [], False
    for step in range(blocks + 4):
        arrived = arrivals.get(step, [])
        rng.shuffle(arrived)  # reordered
        half = len(arrived) // 2
        peer.update(arrived[:half])
        times = _times(16 * step + 1, shift=-0.37)
        if seen or half:
            peer.at(times + 0.2, previous, final=False)  # prepared on predicted times
        peer.update(arrived[half:])  # arrived before the controls cut
        seen = seen or bool(arrived)
        if seen:
            block = peer.at(times, previous, final=True)
            previous = block.rows[-1, 3:5]
            drawn.append(block.rows[:, 0])
    x = np.concatenate(drawn)
    assert len(drawn) >= blocks and (np.diff(x) >= -1e-6).all()


# ---------------------------------------------------------------------------- facing in the field
def _write_blocks(batch: dict, slot: int, peer: LivePeer, blocks: list) -> None:
    """Write the peer's blocks ``(first row, session times, parts that arrived before)`` into the
    round's table as a client does: each block's turns start at the continuous columns of the
    latent frame before it."""
    for first, times, parts in blocks:
        peer.update(parts)
        previous = continuous_row_state(batch)[0, (first - 1) // 4, slot, 0:2].numpy()
        block = peer.at(times, previous, final=True)
        video = slice(first, first + 16)
        batch["player_states"][0, slot, video, :5] = torch.from_numpy(block.rows).float()
        batch["player_control_substeps"][0, slot, video] = torch.from_numpy(block.values)
        batch["player_control_substep_valid"][0, slot, video] = torch.from_numpy(block.valid)


def _seam(batch: dict) -> tuple[LivePeer, list, int]:
    """Yaw across the seam, pitch up and down, drawn between the peer's rows (0.3 rows after)."""
    batch["player_states"][0, 1, 0, 3:5] = torch.tensor([-100.0, 5.0])  # the seat's first frame
    frames = 1 + 16 * 5
    rows = np.zeros((frames, 5))
    rows[:, 3] = wrap_degrees(torch.from_numpy(160.0 + 1.5 * np.arange(frames))).numpy()
    rows[:, 4] = 20.0 * np.sin(np.arange(frames) / 10.0)
    parts = _parts(rows, _keys(frames), blocks=range(5), shift=-0.3)
    blocks = [(1 + 16 * k, _times(1 + 16 * k), parts[k : k + 1]) for k in range(4)]
    return LivePeer(PRIOR), blocks, 0


def _pitch_limit(batch: dict) -> tuple[LivePeer, list, int]:
    """Looking down at 89 degrees, a flick to 60 in a part that arrives one block late: blended,
    its rows would pass 89."""
    batch["player_states"][0, 1, 0, 3:5] = torch.tensor([0.0, 89.0])
    frames = 1 + 16 * 7
    rows = np.zeros((frames, 5))
    rows[:, 4] = 89.0
    rows[65:67, 4] = [60.0, 75.0]
    parts = _parts(rows, _keys(frames), blocks=range(7))
    arrive = [parts[0:1], parts[1:2], parts[2:3], parts[3:4], [], parts[4:6], parts[6:7]]
    blocks = [(1 + 16 * k, _times(1 + 16 * k) - 1.0, arrive[k]) for k in range(7)]
    return LivePeer(PRIOR), blocks, 0


def _takeover(batch: dict) -> tuple[LivePeer, list, int]:
    """A human takes the seat from video frame 49 on: its recording turned before, and its rows
    are not the integral of its substeps."""
    turning = torch.zeros(49, 4, 13)
    turning[:, :, 11], turning[:, :, 12] = 0.1, 0.4  # 2 degrees of pitch, 8 of yaw per row
    batch["player_control_substeps"][0, 1, :49] = turning
    recorded = torch.arange(49.0)
    batch["player_states"][0, 1, :49, 3] = wrap_degrees(30.0 + 8.3 * recorded)
    batch["player_states"][0, 1, :49, 4] = -40.0 + 1.9 * recorded
    frames = 1 + 16 * 6
    rows = np.zeros((frames, 5))
    rows[:, 3] = wrap_degrees(torch.from_numpy(-120.0 - 3.0 * np.arange(frames))).numpy()
    rows[:, 4] = 10.0
    parts = _parts(rows, _keys(frames), blocks=range(6))
    blocks = [(1 + 16 * k, _times(1 + 16 * k), parts[k : k + 1]) for k in range(3, 6)]
    return LivePeer(PRIOR), blocks, 13


def _long(batch: dict) -> tuple[LivePeer, list, int]:
    """100 blocks of fast turning (30 degrees per video frame, 480 per second) and pitch."""
    rng = np.random.default_rng(0)
    frames = 1 + 16 * 100
    rows = np.zeros((frames, 5))
    rows[:, 3] = wrap_degrees(torch.from_numpy(np.cumsum(rng.normal(0.0, 30.0, frames)))).numpy()
    rows[:, 4] = np.clip(np.cumsum(rng.normal(0.0, 2.0, frames)), -80.0, 80.0)
    parts = _parts(rows, _keys(frames), blocks=range(100))
    blocks = [(1 + 16 * k, _times(1 + 16 * k), parts[k : k + 1]) for k in range(100)]
    return LivePeer(PRIOR), blocks, 0


@pytest.mark.parametrize(
    "scenario, latents",
    [(_seam, 17), (_pitch_limit, 29), (_takeover, 25), (_long, 401)],
    ids=["seam", "pitch_limit", "takeover", "long"],
)
def test_the_written_substeps_reproduce_the_rows_facing_in_both_tables(scenario, latents):
    batch = standing_round([[0.0, 0.0, 0.0], [800.0, 0.0, 0.0]], latents=latents)
    peer, blocks, start = scenario(batch)
    _write_blocks(batch, 1, peer, blocks)
    states = batch["player_states"]
    assert float(states[0, 1, :, 4].abs().max()) <= 89.0
    yaw, pitch = states[0, 1, ::4, 3][start:], states[0, 1, ::4, 4][start:]
    continuous = continuous_row_state(batch)[0, start:, 1]
    folded = PlayerState.from_batch(batch)
    yaw6, pitch6 = table_angles(folded.table(), folded.controls)
    for table_yaw, table_pitch in (
        (continuous[:, 0], continuous[:, 1]),
        (yaw6[0, start:, 1], pitch6[0, start:, 1]),
    ):
        assert float(wrap_degrees(table_yaw - yaw).abs().max()) <= 1e-3
        assert float((table_pitch - pitch).abs().max()) <= 1e-3


def test_the_extension_is_integrated_as_far_as_the_times_drawn_need_and_batches_with_others():
    """A newer row's extension is integrated up to the row of the block's last time: from rest over
    the 32 warm rows (all there are, if fewer) and the rows past the newest the times need, none
    while the received rows reach them; later times go on from the integrator where it stopped,
    which gives the rows of one call bit for bit; never past the horizon."""
    rows, buttons = _runner(5)
    parts = _parts(rows, buttons, blocks=range(5))
    peer, other = LivePeer(PRIOR), LivePeer(PRIOR)
    peer.update(parts[:3])  # video frames 1-48
    other.update(parts[:1])  # 1-16
    assert peer.prior_inputs(_times(33)) == []  # 33-48: received
    inputs = peer.prior_inputs(_times(41)) + other.prior_inputs(_times(9))
    assert [(int(c.shape[0]), state) for c, _, state in inputs] == [(32 + 8, None), (16 + 8, None)]
    peer.update(parts[1:3])  # delivered again: the input still holds
    results = prior_displacements(inputs, prior=PRIOR)
    peer.extend(results[:1])
    other.extend(results[1:])
    assert peer.prior_inputs(_times(41)) == [] and other.prior_inputs(_times(9)) == []
    batched = peer.at(_times(41), np.zeros(2), final=False).rows
    later = peer.prior_inputs(_times(49))  # 8 rows more, from the integrator
    assert [(int(c.shape[0]), state is None) for c, _, state in later] == [(8, False)]
    peer.extend(prior_displacements(later, prior=PRIOR))
    at_once = LivePeer(PRIOR)
    at_once.update(parts[:3])
    assert np.array_equal(
        peer.at(_times(49), np.zeros(2), final=False).rows,
        at_once.at(_times(49), np.zeros(2), final=False).rows,
    )
    assert np.array_equal(batched, at_once.at(_times(41), np.zeros(2), final=False).rows)
    far = peer.prior_inputs(_times(48 + 100))
    assert [int(c.shape[0]) for c, _, _ in far] == [PEER_HORIZON_ROWS - 16]
    with pytest.raises(ValueError, match="one result per input of its latest prior_inputs"):
        peer.extend([])
    ((moved, state),) = prior_displacements(far, prior=PRIOR)
    with pytest.raises(ValueError, match="the displacement of its latest prior input"):
        peer.extend([(moved[1:], state)])
    stale = peer.prior_inputs(_times(48 + 100))
    peer.update(parts[3:])  # a newer row between the input and its result: it asks on its own
    peer.extend(prior_displacements(stale, prior=PRIOR))
    fresh = peer.prior_inputs(_times(73))
    assert [(int(c.shape[0]), state) for c, _, state in fresh] == [(32 + 8, None)]
    peer.extend(prior_displacements(fresh, prior=PRIOR))
    extended = peer.at(_times(73), np.zeros(2), final=False)
    np.testing.assert_allclose(extended.rows, _fresh(parts, _times(73)), atol=1e-3)


def test_the_blend_integrates_the_rows_the_last_block_was_drawn_from_as_far_as_it_needs():
    """A block drawn at video frames 25-40 from the rows up to 32 integrates 8 rows past them; with
    newer rows the next block's blend reads those older rows at its first time, 9 rows past them:
    one row more, in the same prior call, and the block is the one integrated at once."""
    rows, buttons = _runner(3)
    parts = _parts(rows, buttons, blocks=range(3))
    lazy, at_once = LivePeer(PRIOR), LivePeer(PRIOR)
    for peer in (lazy, at_once):
        peer.update(parts[:2])
    drawn = lazy.at(_times(25), np.zeros(2), final=True)
    at_once.extend(prior_displacements(at_once.prior_inputs(_times(48 + 100)), prior=PRIOR))
    assert np.array_equal(at_once.at(_times(25), np.zeros(2), final=True).rows, drawn.rows)
    for peer in (lazy, at_once):
        peer.update(parts[2:])
    inputs = lazy.prior_inputs(_times(41))
    assert [(int(c.shape[0]), state is None) for c, _, state in inputs] == [(40, True), (1, False)]
    lazy.extend(prior_displacements(inputs, prior=PRIOR))
    block = lazy.at(_times(41), drawn.rows[-1, 3:5], final=True)
    assert block.present.all() and not block.snapped
    assert np.array_equal(block.rows, at_once.at(_times(41), drawn.rows[-1, 3:5], final=True).rows)
