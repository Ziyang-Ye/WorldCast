"""Seeding of python, numpy and torch."""

import random

import numpy as np
import torch

from worldcast.utils.seed import set_seed


def _draws() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.rand()), torch.randn(3)


def test_set_seed_restarts_the_three_streams():
    set_seed(20260917)
    first = _draws()
    set_seed(20260917)
    again = _draws()
    assert first[:2] == again[:2] and torch.equal(first[2], again[2])
    assert torch.equal(first[2], torch.randn(3, generator=torch.Generator().manual_seed(20260917)))
    set_seed(20260918)
    other = _draws()
    assert first[0] != other[0] and first[1] != other[1] and not torch.equal(first[2], other[2])
