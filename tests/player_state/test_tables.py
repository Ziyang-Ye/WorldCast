"""The peer tables."""

import pytest
import torch

from worldcast.player_state import tables as new


def test_player_states_refuse_bad_batches(synth):
    batch = synth.make_round_batch(0)
    with pytest.raises(KeyError):
        new.PlayerStates.from_batch({k: v for k, v in batch.items() if k != "observer_slot"})
    bad = dict(batch, player_team_ids=batch["player_team_ids"][:, :5])
    with pytest.raises(ValueError):
        new.PlayerStates.from_batch(bad)
    bad = dict(batch, peer_continuous_columns=torch.zeros(2, 3, 10, 7))
    with pytest.raises(ValueError):
        new.PlayerStates.from_batch(bad)
