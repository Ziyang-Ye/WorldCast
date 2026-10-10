"""The visibility probe of training: footprint pooling, the five geometry features, its BCE."""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.modeling.support import randomize_
from worldcast.modeling.visibility_probe import (
    VisibilityProbe,
    VisibilityProbeConfig,
    VisibilityProbeInputs,
    visibility_bce,
)

GRID_H, GRID_W, DIM = 2, 3, 8


def _probe() -> VisibilityProbe:
    return randomize_(VisibilityProbe(DIM, VisibilityProbeConfig(dit_block=2, hidden=4)), 1)


def _inputs(frames: int = 3) -> VisibilityProbeInputs:
    """One player whose footprint is the single token (row 1, column 2) of every frame."""
    footprint = torch.zeros(1, frames, 1, GRID_H, GRID_W)
    footprint[..., 1, 2] = 2.0
    return VisibilityProbeInputs(
        footprint=footprint,
        depth=torch.full((1, frames, 1), 4096.0),
        relative_yaw=torch.full((1, frames, 1), math.pi / 2),
        in_front=torch.ones(1, frames, 1, dtype=torch.bool),
    )


def test_released_shape():
    with torch.device("meta"):
        probe = VisibilityProbe(3072)
    assert probe.dit_block == 20
    assert probe.mlp[0].in_features == 3072 + 5 and probe.mlp[0].out_features == 256
    assert sum(p.numel() for p in probe.parameters()) == 860_161


def test_logits_are_the_mlp_on_the_pooled_tokens_and_the_geometry():
    probe = _probe()
    hidden = torch.randn(1, 3 * GRID_H * GRID_W, DIM, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        logits = probe(hidden, _inputs())
        token = hidden[0, 2 * 6 + 1 * GRID_W + 2]  # the footprint's token of window frame 2
        # depth 4096 -> 1, sin and cos of a quarter turn, coverage 2 / 4, in front
        geometry = torch.tensor([1.0, 1.0, math.cos(math.pi / 2), 0.5, 1.0])
        want = probe.mlp(torch.cat([probe.norm(token), geometry]))
    assert logits.shape == (1, 3, 1) and logits.dtype == torch.float32
    torch.testing.assert_close(logits[0, 2], want, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match=r"is not \[B, F h w, dim\] of the footprint"):
        probe(hidden[:, :-1], _inputs())  # the token grid is the footprint's


def test_the_probe_reads_a_detached_hidden_state():
    probe = _probe()
    hidden = torch.randn(1, 3 * GRID_H * GRID_W, DIM, requires_grad=True)
    logits = probe(hidden, _inputs())
    logits.sum().backward()
    assert hidden.grad is None and probe.mlp[0].weight.grad is not None


def test_bce_balances_the_classes_over_the_confirmed_labels():
    """One positive against two negatives weighs 2; the unknown label has no weight."""
    logits = torch.tensor([[[0.0, 2.0, -2.0, 5.0]]])
    visible = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
    valid = torch.tensor([[[True, True, True, False]]])
    loss = visibility_bce(logits, visible, valid)
    softplus = F.softplus(torch.tensor([0.0, 2.0, -2.0]))
    want = (2 * softplus[0] + softplus[1] + softplus[2]) / 4
    assert loss.shape == () and loss.item() == pytest.approx(want.item(), rel=1e-6)


def test_bce_caps_the_positive_weight():
    logits = torch.zeros(1, 1, 31)
    visible = torch.zeros(1, 1, 31)
    visible[..., 0] = 1.0
    valid = torch.ones(1, 1, 31, dtype=torch.bool)
    logits[..., 0] = 3.0
    loss = visibility_bce(logits, visible, valid)
    # 30 negatives per positive would weigh 30: the weight is capped at 10
    want = (10 * F.softplus(torch.tensor(-3.0)) + 30 * math.log(2.0)) / 40
    assert loss.item() == pytest.approx(want.item(), rel=1e-6)
    with pytest.raises(ValueError, match="one shape"):
        visibility_bce(logits, visible[..., :30], valid)
