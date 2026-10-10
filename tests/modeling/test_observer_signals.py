"""The observer-signal embedding: flash and scope, each with a learned "unknown"."""

import dataclasses

import pytest
import torch

from tests.modeling.support import randomize_
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.modeling.observer_signals import (
    ObserverSignalConfig,
    ObserverSignalEmbedding,
    ObserverSignals,
)

TINY = ObserverSignalConfig(hidden=8)


def _signals(flash=1, flash_valid=1, scope_on=1, level=2, scope_valid=1) -> ObserverSignals:
    values = (flash, flash_valid, scope_on, level, scope_valid)
    return ObserverSignals(*(torch.tensor([[value]]) for value in values))


def test_the_signals_of_a_condition_dict():
    """The record's fields are the condition keys without their ``obs_`` prefix, in order; the
    signals come all together or not at all."""
    conditions = {key: torch.full((1, 3), i) for i, key in enumerate(OBSERVER_SIGNAL_KEYS)}
    signals = ObserverSignals.from_conditions({**conditions, "weapon": torch.zeros(1, 9)})
    names = [field.name for field in dataclasses.fields(ObserverSignals)]
    assert names == [key.removeprefix("obs_") for key in OBSERVER_SIGNAL_KEYS]
    assert all(getattr(signals, name) is conditions["obs_" + name] for name in names)
    assert torch.equal(signals.to("cpu").scope_level, conditions["obs_scope_level"])
    assert ObserverSignals.from_conditions({"weapon": torch.zeros(1, 9)}) is None
    with pytest.raises(ValueError, match="arrive together or not at all"):
        ObserverSignals.from_conditions({k: v for k, v in conditions.items() if "valid" in k})
    with pytest.raises(ValueError, match=r"share one shape \[B, F\], got .*'scope_on': \(1, 2\)"):
        ObserverSignals.from_conditions({**conditions, "obs_scope_on": torch.zeros(1, 2)})


def test_the_embedding_starts_at_zero():
    torch.manual_seed(0)
    embedding = ObserverSignalEmbedding(16, TINY)
    out = embedding(ObserverSignals(*(torch.ones(2, 5, dtype=torch.long) for _ in range(5))))
    assert out.shape == (2, 5, 16) and not bool(out.any())


def test_an_invalid_signal_reads_unknown():
    embedding = randomize_(ObserverSignalEmbedding(16, TINY), 1).eval()
    with torch.no_grad():
        known = embedding(_signals())
        # an unknown flash does not depend on the flag; the scope's flag covers both its signals
        assert torch.equal(
            embedding(_signals(flash=0, flash_valid=0)), embedding(_signals(flash_valid=0))
        )
        assert torch.equal(
            embedding(_signals(scope_on=0, level=0, scope_valid=0)),
            embedding(_signals(scope_valid=0)),
        )
        for changed in (dict(flash=0), dict(scope_on=0), dict(level=1), dict(flash_valid=0)):
            assert not torch.equal(embedding(_signals(**changed)), known), changed
        # values beyond the tables are clamped to their last row
        assert torch.equal(embedding(_signals(flash=7, level=9)), known)
