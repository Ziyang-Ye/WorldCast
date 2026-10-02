"""Fixtures of the modeling tests."""

from types import SimpleNamespace

import pytest

from tests.modeling.tiny_model import TINY, new_kv_cache, random_c2w, randomize_, tiny_config


@pytest.fixture
def tiny():
    """Builders of a tiny generator."""
    return SimpleNamespace(
        tiny_config=tiny_config,
        random_c2w=random_c2w,
        new_kv_cache=new_kv_cache,
        randomize_=randomize_,
        TINY=TINY,
    )
