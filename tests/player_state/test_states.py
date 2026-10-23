"""Player states at latent rate and the player state table."""

import math

import pytest
import torch

from tests.player_state.support import TSD, round_batch, same_columns
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.latents import video_frames_of
from worldcast.player_state.field import field_builder
from worldcast.player_state.live_peers import keep_alive
from worldcast.player_state.states import (
    DEATH_TAU,
    FIELD_CONDITION_KEYS,
    PlayerState,
    camera_turns,
    check_block_alive_agreement,
    continuous_block_state,
    continuous_row_state,
    death_state_for_table,
    derive_death_state,
    held_fraction,
    integrate_camera_angles,
    latent_frame_rows,
    memory_continuous_columns,
    owned_rows,
    pack_substeps,
    player_state_conditions,
    wrap_degrees,
)


def test_a_latent_frame_is_read_at_its_last_video_frame_and_owns_four():
    assert latent_frame_rows(9).tolist() == [0, 4, 8]
    assert owned_rows(3).tolist() == [[0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8]]
    with pytest.raises(ValueError):
        latent_frame_rows(10)


def test_pack_substeps_zeroes_invalid_substeps_and_appends_the_flag():
    substeps = torch.ones(1, 1, 2, 2, 3)
    valid = torch.tensor([[[[True, False], [False, True]]]])
    packed = pack_substeps(substeps, valid)
    assert packed.shape == (1, 1, 2, 2, 4)
    assert packed[0, 0, 0].tolist() == [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]


def test_held_fraction_counts_the_valid_substeps_only():
    controls = torch.zeros(1, 1, 1, 4, 3)  # one button, one turn, the valid flag
    controls[..., :3, 2] = 1.0
    controls[..., 0, 0] = 1.0
    controls[..., 3, 0] = 1.0  # held on an invalid substep: not counted
    assert held_fraction(controls, 0).item() == pytest.approx(1 / 3)
    assert held_fraction(torch.zeros(1, 1, 1, 4, 3), 0).item() == 0.0


def _turning() -> torch.Tensor:
    """Packed substeps ``[1, 1, 4, 2, 4]`` (one button, pitch, yaw, valid) of one player over four
    rows: the turns of row 0 precede the window; one substep of row 3 is not valid."""
    substeps = torch.zeros(1, 1, 4, 2, 4)
    substeps[..., 3] = 1.0
    substeps[0, 0, :, 0, 1] = torch.tensor([9.0, 4.0, 0.0, -1.0])  # pitch, units of 5 degrees
    substeps[0, 0, :, 0, 2] = torch.tensor([9.0, 4.0, 0.0, 1.0])  # yaw
    substeps[0, 0, 3, 1, 1:3] = 7.0
    substeps[0, 0, 3, 1, 3] = 0.0  # not valid: its turns do not count
    return substeps


def test_the_turn_of_a_row_is_the_sum_of_its_valid_substeps():
    pitch, yaw = camera_turns(_turning(), frame_dim=2)
    assert pitch[0, 0].tolist() == [0.0, 20.0, 0.0, -5.0]
    assert yaw[0, 0].tolist() == [0.0, 20.0, 0.0, 5.0]


def test_camera_integral_wraps_the_yaw_and_clamps_the_pitch_row_by_row():
    initial = torch.tensor([[[0.0, 0.0, 0.0, 170.0, 80.0, 1.0]]])
    yaw, pitch = integrate_camera_angles(initial, _turning())
    # 170 + 20 wraps to -170
    assert yaw[0, 0].tolist() == [170.0, -170.0, -170.0, -165.0]
    # 80 + 20 is clamped to 89 and the excess is discarded: the next -5 starts from 89
    assert pitch[0, 0].tolist() == [80.0, 89.0, 89.0, 84.0]
    assert wrap_degrees(torch.tensor([180.0, -181.0])).tolist() == [-180.0, 179.0]


def test_camera_integral_can_clamp_the_summed_pitch_instead():
    initial = torch.tensor([[[0.0, 0.0, 0.0, 170.0, 80.0, 1.0]]])
    yaw, pitch = integrate_camera_angles(initial, _turning(), clamp_pitch_sum=True)
    assert yaw[0, 0].tolist() == [170.0, -170.0, -170.0, -165.0]
    # 80 + 20 = 100 is banked: after the next -5 the sum is 95, still beyond the range
    assert pitch[0, 0].tolist() == [80.0, 89.0, 89.0, 89.0]
    initial[..., 4] = 60.0  # a sum that stays in the range is the row-by-row pitch
    _, pitch = integrate_camera_angles(initial, _turning(), clamp_pitch_sum=True)
    assert pitch[0, 0].tolist() == [60.0, 80.0, 80.0, 75.0]


def test_from_batch_folds_the_video_frames_to_latent_frames():
    batch = round_batch(0, latents=5)
    states = PlayerState.from_batch(batch)
    rows = [0, 4, 8, 12, 16]
    assert torch.equal(states.xyz, batch["player_states"][:, :, rows, :3].permute(0, 2, 1, 3))
    assert torch.equal(states.yaw, batch["player_states"][:, :, rows, 3].permute(0, 2, 1))
    assert torch.equal(states.alive, batch["player_states"][:, :, rows, 5].permute(0, 2, 1) > 0.5)
    assert torch.equal(states.weapon_ids, batch["player_weapon_ids"][:, :, rows].permute(0, 2, 1))
    packed = pack_substeps(batch["player_control_substeps"], batch["player_control_substep_valid"])
    assert states.controls.shape == (2, 5, 10, 16, 14)
    # latent frame 2 owns video frames 5 .. 8, four substeps each; latent frame 0 its one frame
    assert torch.equal(states.controls[:, 2], packed[:, :, 5:9].flatten(2, 3))
    assert torch.equal(states.controls[:, 0, :, :4], packed[:, :, 0])
    assert states.table().shape == (2, 5, 10, 6) and states.continuous is None


def test_player_state_refuses_bad_batches():
    batch = round_batch(0)
    for key in ("client_slot", "player_weapon_ids", "player_control_substep_valid"):
        with pytest.raises(KeyError, match=key):
            PlayerState.from_batch({k: v for k, v in batch.items() if k != key})
    bad = dict(batch, player_team_ids=batch["player_team_ids"][:, :5])
    with pytest.raises(ValueError):
        PlayerState.from_batch(bad)
    bad = dict(batch, player_continuous_columns=torch.zeros(2, 3, 10, 7))
    with pytest.raises(ValueError):
        PlayerState.from_batch(bad)


def test_continuous_columns_carry_the_death_state_of_the_whole_round():
    batch = round_batch(1, batch=1, latents=9, client_slots=(3,))
    states = batch["player_states"]
    states[0, 0, :, 5] = 1.0
    states[0, 0, 13:, 5] = 0.0  # slot 0 dies after latent frame 3 (video frame 12)
    states[0, 1, :, 5] = 0.0  # slot 1 is dead from the first frame: no death site, no corpse
    columns = continuous_row_state(batch)
    assert columns.shape == (1, 9, 10, 7)
    yaw, pitch = integrate_camera_angles(
        states[:, :, 0],
        pack_substeps(batch["player_control_substeps"], batch["player_control_substep_valid"]),
    )
    assert torch.equal(columns[0, :, :, 0], yaw[0, :, ::4].T)  # at the latent frames' last frames
    assert torch.equal(columns[0, :, :, 1], pitch[0, :, ::4].T)
    assert columns[0, :, 0, 2].tolist() == [0.0] * 4 + [1.0] * 5  # corpse
    assert columns[0, :, 0, 3].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert torch.equal(columns[0, 4:, 0, 4:], states[0, 0, 12, :3].expand(5, 3))  # the death site
    assert not bool(columns[0, :, 1, 2].any())

    table = PlayerState.from_batch(dict(batch, player_continuous_columns=columns)).table()
    assert table.shape == (1, 9, 10, 13)
    death, derived = death_state_for_table(table), death_state_for_table(table[..., :6])
    assert torch.equal(death.corpse, derived.corpse)
    assert torch.equal(death.frozen_states[..., :3], derived.frozen_states[..., :3])
    assert death.dying[0, 5, 0].item() == pytest.approx(math.exp(-2.0 / DEATH_TAU))


def test_a_player_dead_from_the_first_frame_counts_its_death_from_the_offset():
    """A table that is a part of a longer round (its first frame is the round's frame ``offset``):
    a player dead from the first frame has been dead for ``offset`` frames more, to the cap of 32;
    a player that dies in it, and every other column, are what they are without the offset."""
    states = torch.ones(1, 40, 3, 6)
    states[:, :, 0, 5] = 0.0  # never alive
    states[:, 10:, 1, 5] = 0.0  # dies after frame 9
    plain, moved = derive_death_state(states), derive_death_state(states, offset=5)
    assert plain.time_since_death[0, :4, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert moved.time_since_death[0, :4, 0].tolist() == [6.0, 7.0, 8.0, 9.0]
    assert moved.time_since_death[0, 26:, 0].tolist() == [32.0] * 14
    assert plain.time_since_death[0, 10:13, 1].tolist() == [1.0, 2.0, 3.0]
    assert torch.equal(moved.time_since_death[..., 1:], plain.time_since_death[..., 1:])
    assert torch.equal(moved.dying, plain.dying) and torch.equal(moved.corpse, plain.corpse)
    assert torch.equal(moved.frozen_states, plain.frozen_states)


@pytest.mark.parametrize("s", [1, 5, 13, 29])
def test_a_blocks_continuous_columns_are_the_rounds_after_its_rows_change(s):
    """The rows of block ``s`` are written again (turns in every row, the block's first included,
    positions, valid flags; one player drifts to the pitch clamp): its four latent frames' columns,
    written on their own, are those of the whole round, and the earlier latent frames keep theirs.
    """
    batch = round_batch(4, batch=2, latents=33)
    columns = continuous_row_state(batch)
    earlier = columns[:, :s].clone()
    g = torch.Generator().manual_seed(s)
    rows = slice(4 * s - 3, 4 * s + 13)
    substeps = batch["player_control_substeps"][:, :, rows]
    substeps[:] = torch.randn(substeps.shape, generator=g) * 4.0
    substeps[:, 2, :, :, 11] = 3.0  # pitch turn / 5 per substep: into the clamp within the block
    batch["player_control_substep_valid"][:, :, rows] = torch.rand(2, 10, 16, 4, generator=g) < 0.9
    batch["player_states"][:, :, rows, :3] += 75.0
    continuous_block_state(columns, batch, s)
    whole = continuous_row_state(batch)
    block = slice(s, s + 4)
    same_columns(columns[:, block], whole[:, block])
    assert torch.equal(columns[:, :s], earlier) and torch.equal(whole[:, :s], earlier)
    assert columns[:, s + 3, 2, 1].tolist() == [89.0, 89.0]


def _torch_block_columns(columns, batch, s, shift):
    """A block's continuous columns as torch integrates them: the reference
    :func:`continuous_block_state` follows operation by operation."""
    first, rows = 4 * s - 3, 4 * s + 13
    turn_pitch, turn_yaw = camera_turns(
        pack_substeps(
            batch["player_control_substeps"][:, :, first - 1 : rows, :, -2:],
            batch["player_control_substep_valid"][:, :, first - 1 : rows],
        ),
        frame_dim=2,
    )
    yaw = wrap_degrees(columns[:, s - 1, :, 0, None] + turn_yaw[:, :, 1:].cumsum(dim=2))
    current, pitch = columns[:, s - 1, :, 1], []
    for row in range(1, 17):
        current = (current + turn_pitch[:, :, row]).clamp(-89.0, 89.0)
        pitch.append(current)
    pitch = torch.stack(pitch, 2)
    latent = latent_frame_rows(rows)
    death = derive_death_state(
        batch["player_states"][:, :, :rows].index_select(2, latent).permute(0, 2, 1, 3), shift
    )
    block = slice(s, s + 4)
    return torch.cat(
        [
            yaw[:, :, 3::4].permute(0, 2, 1)[..., None],
            pitch[:, :, 3::4].permute(0, 2, 1)[..., None],
            death.corpse[:, block, :, None].float(),
            death.time_since_death[:, block, :, None],
            death.frozen_states[:, block, :, :3],
        ],
        -1,
    )


@pytest.mark.parametrize("s", [1, 5, 13, 25])
def test_a_blocks_continuous_columns_are_torchs_bit_for_bit(s):
    """A block's continuous columns are torch's integral of its turns and death state to the bit:
    turns in every row (signed zeros and invalid substeps included), the pitch clamp, players dead
    and dead from the first row, a round that slid."""
    for seed, shift in ((1, 0), (2, 4), (3, 12)):
        batch = round_batch(seed, batch=2, latents=33)
        columns = continuous_row_state(batch)
        g = torch.Generator().manual_seed(seed)
        rows = slice(4 * s - 3, 4 * s + 13)
        substeps = batch["player_control_substeps"][:, :, rows]
        substeps[:] = torch.randn(substeps.shape, generator=g) * 4.0
        substeps[:, 2, :, :, 11] = 3.0
        substeps[:, 3, :8, :, 12] = -0.0
        batch["player_control_substep_valid"][:, :, rows] = (
            torch.rand(2, 10, 16, 4, generator=g) < 0.9
        )
        batch["player_states"][1, 4, 40:, 5] = 0.0  # dies in the round
        batch["player_states"][:, 6, :, 5] = 0.0  # dead from the first row
        expected = _torch_block_columns(columns, batch, s, shift)
        continuous_block_state(columns, batch, s, shift)
        written = columns[:, s : s + 4]
        assert torch.equal(written.view(torch.int32), expected.view(torch.int32)), (seed, shift)


@pytest.mark.parametrize("shift", [4, 12])
def test_a_round_that_slid_has_the_columns_of_the_session(shift):
    """A round of 29 latent frames that slid by ``shift`` latent frames (the first two stay) gives
    its newest block, the round's latent frames 25-28, the columns of the session's latent frames
    ``25 + shift ..`` on a round of 121: the yaw taken on from the block before with the turn of
    every row, and, for the players dead or absent from the first row, the time since death counted
    from the session's first row (the round's would be 26 .. 29 where the session's is 30 .. 32
    from shift 4)."""
    session = round_batch(7, batch=1, latents=121, client_slots=(3,))
    keep_alive(session["player_states"])  # one life per round, as in live play
    whole = continuous_row_state(session)
    latents = [0, 1, *range(2 + shift, 29 + shift)]
    rows = [r for f in latents for r in video_frames_of(f)]
    slid = {
        key: value[:, :, rows] if value.ndim >= 3 and value.shape[2] == 481 else value
        for key, value in session.items()
        if key in ("player_states", "player_control_substeps", "player_control_substep_valid")
    }
    columns = whole[:, latents].clone()
    columns[:, 25:] = 0.0  # the block's own columns are written
    continuous_block_state(columns, slid, 25, shift)
    same_columns(columns[:, 25:], whole[:, 25 + shift : 29 + shift])
    never_alive = ~(session["player_states"][0, :, 0, 5] > 0.5)
    assert never_alive.any()
    expected = [min(26 + shift + i, 32) for i in range(4)]
    counted = columns[0, 25:, :, TSD][:, never_alive]
    assert counted.tolist() == [[v] * int(never_alive.sum()) for v in expected]


def test_a_resurrection_is_refused():
    table = PlayerState.from_batch(round_batch(0)).table()
    table[:, :, 0, 5] = 1.0
    table[:, 3, 0, 5] = 0.0
    with pytest.raises(ValueError, match="one life per round"):
        death_state_for_table(table)
    with pytest.raises(ValueError):
        death_state_for_table(table[..., :5])


def test_conditions_carry_the_table_the_controls_and_the_observer_signals():
    batch = round_batch(0)
    states = PlayerState.from_batch(batch)
    visible = torch.ones_like(states.alive)
    conditions = player_state_conditions(states, visible)
    assert (
        tuple(conditions)
        == FIELD_CONDITION_KEYS
        == field_builder().condition_keys
        == (
            "player_state_table",
            "player_controls",
            "client_slot",
            "player_team_ids",
            "player_alive",
            "player_visible",
            "player_weapons",
        )
    )
    assert conditions["player_alive"].dtype == conditions["player_visible"].dtype == torch.float32
    with_signals = player_state_conditions(states, visible, observer_signals=batch)
    assert list(with_signals)[7:] == list(OBSERVER_SIGNAL_KEYS)
    assert all(with_signals[key].dtype == torch.long for key in OBSERVER_SIGNAL_KEYS)


def test_the_memory_frames_continuous_columns():
    """Every other player keeps its first-frame columns on the memory frames; the client's row
    takes the source player's angles and position at each memory frame's last video frame, with no
    corpse and no time since death."""
    g = torch.Generator().manual_seed(0)
    rows = torch.randn(2, 9, 3, 7, generator=g)
    memory_states = torch.arange(16, dtype=torch.float32)[None, :, None].expand(2, 16, 6).clone()
    memory_states[1] += 100.0
    columns = memory_continuous_columns(rows, memory_states.double(), torch.tensor([2, 0]))
    assert columns.shape == (2, 4, 3, 7) and columns.dtype == torch.float32
    for b, client in enumerate((2, 0)):
        last = [float(100 * b + t) for t in (3, 7, 11, 15)]
        for column in (0, 1, 4, 5, 6):  # yaw, pitch and the frozen x, y, z
            assert columns[b, :, client, column].tolist() == last
        assert not bool(columns[b, :, client, 2:4].any())  # corpse, time since death
        others = [p for p in range(3) if p != client]
        assert torch.equal(columns[b, :, others], rows[b, :1, others].expand(4, 2, 7))


def test_the_alive_condition_must_agree_with_the_table():
    derived = torch.tensor([[[True], [True], [False]]])
    window = torch.tensor([[[0.0], [1.0], [0.0]]])  # zero outside the call's frames 1 .. 2
    check_block_alive_agreement(derived, window, frame_offset=1, num_frames=2)
    with pytest.raises(ValueError):
        check_block_alive_agreement(derived, 1.0 - window, frame_offset=1, num_frames=2)
