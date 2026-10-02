"""The GT-label visibility."""

import pytest

from worldcast.player_state import tables as new_material
from worldcast.player_state import visibility as new


def test_gt_label_visibility_requires_labels(synth):
    batch = synth.make_round_batch(0)
    material = new_material.PlayerStates.from_batch(batch)
    del batch["observer_visibility_valid"]
    with pytest.raises(ValueError):
        new.GTLabelVisibility()(batch, material)
