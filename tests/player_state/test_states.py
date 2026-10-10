"""Player states at latent rate and the player state table."""

import math

import pytest
import torch

from tests.player_state.support import round_batch
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.player_state.field import field_builder
from worldcast.player_state.states import (
    DEATH_TAU,
    FIELD_CONDITION_KEYS,
    PlayerState,
    camera_turns,
    check_block_alive_agreement,
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


def test_a_player_dead_from_the_first_frame_counts_its_death_from_the_first_frame():
    states = torch.ones(1, 40, 3, 6)
    states[:, :, 0, 5] = 0.0  # never alive
    states[:, 10:, 1, 5] = 0.0  # dies after frame 9
    death = derive_death_state(states)
    assert death.time_since_death[0, :4, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert death.time_since_death[0, 31:, 0].tolist() == [32.0] * 9  # to the cap
    assert death.time_since_death[0, 10:13, 1].tolist() == [1.0, 2.0, 3.0]


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
