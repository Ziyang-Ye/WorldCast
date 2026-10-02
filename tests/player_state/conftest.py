"""Fixtures of the player-state tests."""

import types

import pytest

from tests.player_state.synthetic_round import compact_batch, make_round_batch


@pytest.fixture
def synth():
    return types.SimpleNamespace(make_round_batch=make_round_batch, compact_batch=compact_batch)
