"""The state injector ``Conv_0`` of Eq. (2) and the field's weapon embedding."""

import math

import pytest
import torch
from torch import nn

from tests.modeling.support import randomize_
from worldcast.modeling.state_injector import (
    FIELD_CHANNEL_NAMES,
    StateInjector,
    StateInjectorConfig,
    log_compressed_depth,
)


def test_paper_architecture():
    """23 channels -> Conv3x3 to 32 -> SiLU -> Conv1x1 to the model width, after the second DiT
    block; four weapon channels of 52 weapons."""
    with torch.device("meta"):
        injector = StateInjector(3072)
    assert injector.dit_block == 1
    assert injector.stem.weight.shape == (32, 23, 3, 3) and injector.stem.in_channels == 23
    assert injector.proj.weight.shape == (3072, 32, 1, 1)
    assert injector.weapon_embedding.weight.shape == (52, 4)
    assert sum(p.numel() for p in injector.parameters()) == 108_240


def test_a_fresh_injector_draws_as_trained():
    """The weapon embedding, a stem over the first 19 channels and ``proj``, in that order; the
    stem's columns of the last four channels and ``proj`` start at zero."""
    torch.manual_seed(5)
    injector = StateInjector(16)
    after = torch.rand(1)
    torch.manual_seed(5)
    weapon = nn.Embedding(52, 4)
    stem = nn.Conv2d(19, 32, kernel_size=3, padding=1)
    nn.Conv2d(32, 16, kernel_size=1)
    assert torch.equal(torch.rand(1), after)
    assert FIELD_CHANNEL_NAMES[19:] == ("dying", "corpse", "identity_live", "identity_corpse")
    assert torch.equal(injector.weapon_embedding.weight, weapon.weight)
    assert torch.equal(injector.stem.weight[:, :19], stem.weight)
    assert torch.equal(injector.stem.bias, stem.bias)
    assert not bool(injector.stem.weight[:, 19:].any())
    assert not bool(injector.proj.weight.any()) and not bool(injector.proj.bias.any())


def test_the_delta_is_the_convolution_of_each_frame_in_token_order():
    injector = randomize_(StateInjector(16, StateInjectorConfig(hidden=8)), 1)
    field = torch.randn(2, 3, 23, 4, 6, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        delta = injector.delta(field)
        frame = injector.proj(torch.nn.functional.silu(injector.stem(field[1, 2:3])))[0]
    assert delta.shape == (2, 3 * 24, 16)
    # token (row 3, column 4) of frame 2
    assert torch.equal(delta[1, 2 * 24 + 3 * 6 + 4], frame[:, 3, 4])


def test_the_injector_adds_the_delta_to_each_copy_of_the_calls_frames():
    injector = randomize_(StateInjector(16, StateInjectorConfig(hidden=8)), 1)
    g = torch.Generator().manual_seed(3)
    field = torch.randn(1, 2, 23, 4, 6, generator=g)
    hidden = torch.randn(1, 2 * 24, 16, generator=g)
    with torch.no_grad():
        injected = injector(hidden, field)
        both = injector(torch.cat([hidden, hidden], 1), field, copies=2)
    assert torch.equal(injected, hidden + injector.delta(field))
    assert torch.equal(both[:, :48], injected) and torch.equal(both[:, 48:], injected)


def test_the_injector_names_a_field_that_does_not_cover_the_hidden_state():
    injector = StateInjector(16, StateInjectorConfig(hidden=8))
    hidden = torch.zeros(2, 2 * 24, 16)
    for shape in ((2, 2, 19, 4, 6), (2, 23, 4, 6)):  # other channels, no frame axis
        with pytest.raises(ValueError, match=r"the field must be \[B, F, 23, h, w\]"):
            injector(hidden, torch.zeros(shape))
    for field, copies in (
        (torch.zeros(1, 2, 23, 4, 6), 1),  # one sample would broadcast over the two
        (torch.zeros(2, 1, 23, 4, 6), 1),  # another number of frames
        (torch.zeros(2, 2, 23, 4, 6), 2),  # more copies than the hidden state holds
        (torch.zeros(2, 2, 23, 4, 6), 0),
    ):
        with pytest.raises(ValueError, match=f"{copies} copies of the field .* do not cover"):
            injector(hidden, field, copies=copies)


def test_the_depth_channel_is_log_compressed_to_the_unit_interval():
    depth = torch.tensor([-5.0, 0.0, 4096.0, 1e6])
    assert log_compressed_depth(depth).tolist() == [0.0, 0.0, 1.0, 1.0]
    middle = log_compressed_depth(torch.tensor(63.0)).item()
    assert middle == pytest.approx(math.log(64.0) / math.log(4097.0), rel=1e-6)
