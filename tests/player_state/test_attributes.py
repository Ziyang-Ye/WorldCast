"""The attributes the player state field writes of a player."""

import pytest
import torch

from worldcast.modeling.state_injector import FIELD_CONTROLS
from worldcast.player_state.attributes import (
    CONTROL_COLUMNS,
    control_fractions,
    corpse_id_plane,
    death_planes,
    enemy_mask,
    headcount,
    live_id_values,
    team_sign,
    weapon_values,
)


def test_the_control_channels_are_nine_of_the_eleven_buttons():
    # forward, back, strafe left, strafe right, jump, crouch, walk, attack, reload (the recordings
    # call the strafes move_left / move_right, crouch duck and walk speed)
    assert FIELD_CONTROLS == (
        "forward",
        "back",
        "move_left",
        "move_right",
        "jump",
        "duck",
        "speed",
        "attack",
        "reload",
    )
    assert CONTROL_COLUMNS == (0, 1, 2, 3, 4, 5, 6, 7, 9)  # without attack2 and look_at_weapon
    controls = torch.zeros(1, 1, 1, 16, 14)
    controls[..., 13] = 1.0
    controls[..., :8, 9] = 1.0  # reload held for half of the latent frame
    fractions = control_fractions(controls)
    assert fractions.shape == (1, 1, 1, 9)
    assert fractions[0, 0, 0].tolist() == [0.0] * 8 + [0.5]


def test_team_sign_and_enemy_mask():
    team_ids = torch.tensor([[2, 2, 3, 3, 0]])
    eligible = torch.ones(1, 3, 5, dtype=torch.bool)
    enemy = enemy_mask(team_ids, torch.tensor([1]), eligible)
    assert enemy.tolist() == [[[False, False, True, True, True]] * 3]
    assert team_sign(enemy, torch.float32)[0, 0].tolist() == [-1.0, -1.0, 1.0, 1.0, 1.0]
    assert not bool(enemy_mask(team_ids, torch.tensor([1]), ~eligible).any())


def test_weapon_values_are_rows_of_the_embedding():
    embedding = torch.arange(52 * 4, dtype=torch.float32).reshape(52, 4)
    weapons = torch.tensor([[[0, 51]]])
    assert torch.equal(weapon_values(weapons, embedding)[0, 0], embedding[[0, 51]])


def test_headcount_saturates_at_four_players():
    weights = torch.zeros(1, 1, 6, 2, 2)
    weights[0, 0, :3, 0, 0] = torch.tensor([1.0, 0.5, 0.5])
    assert headcount(weights).item() == 0.5
    weights[0, 0, 3:, 1, 1] = 1.0
    assert headcount(weights).item() == 1.0


def test_identity_bands():
    enemy = torch.tensor([[[False, True, False]]])
    assert live_id_values(enemy)[0, 0].tolist() == pytest.approx([0.20, 0.52, 0.24])
    corpse = torch.zeros(1, 1, 3, 1, 2)
    corpse[0, 0, 2, 0, 0] = 0.5
    corpse[0, 0, 1, 0, 0] = 0.25
    plane = corpse_id_plane(corpse)
    assert plane[0, 0, 0].tolist() == pytest.approx([0.84 * 0.5, 0.0])  # the arg-max corpse
    with pytest.raises(ValueError):
        live_id_values(torch.zeros(1, 1, 11, dtype=torch.bool))
    with pytest.raises(ValueError):
        corpse_id_plane(torch.zeros(1, 1, 11, 1, 1))


def test_death_planes_take_the_maximum_over_players():
    corpse = torch.zeros(1, 1, 2, 1, 2)
    corpse[0, 0, 0, 0, 0], corpse[0, 0, 1, 0, 0] = 1.0, 0.5
    dying, plane = death_planes(corpse, torch.tensor([[[0.2, 0.8]]]))
    assert dying[0, 0, 0].tolist() == pytest.approx([0.4, 0.0])
    assert plane[0, 0, 0].tolist() == [1.0, 0.0]
