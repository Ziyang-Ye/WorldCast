"""The player attributes of the player state field."""

import pytest
import torch

from worldcast.player_state import attributes as new


def test_control_indices_errors():
    with pytest.raises(ValueError):
        new.control_indices(("forward", "back"), ("forward", "reload"))
    with pytest.raises(ValueError):
        new.duck_index(("forward", "back"))


def test_team_sign_and_enemy_mask_shapes():
    team_ids = torch.tensor([[2, 2, 3, 3, 0]])
    eligible = torch.ones(1, 3, 5, dtype=torch.bool)
    enemy = new.enemy_mask(team_ids, torch.tensor([1]), eligible)
    assert enemy.tolist() == [[[False, False, True, True, True]] * 3]
    assert new.team_sign(enemy, torch.float32)[0, 0].tolist() == [-1.0, -1.0, 1.0, 1.0, 1.0]
