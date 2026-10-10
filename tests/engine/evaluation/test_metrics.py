"""The metrics against known values, and their summary."""

from pathlib import Path

import pytest
import torch

from tests.engine.evaluation.support import l1
from worldcast.engine.evaluation.metrics import (
    LPIPS,
    pixel_mse,
    psnr,
    score_window,
    ssim,
    summarize,
)


def test_psnr_is_that_of_the_pooled_mse():
    black = torch.zeros(2, 3, 4, 4)
    assert pixel_mse(black, black + 0.1) == pytest.approx(0.01)
    assert psnr(black, black + 0.1) == pytest.approx(20.0)
    assert psnr(black, black + 1.0) == 0.0
    assert psnr(black, black) == pytest.approx(120.0)  # the MSE floor
    # of one MSE pooled over the frames, (0 + 0.04) / 2: the frames' own PSNRs average 67 dB
    uneven = torch.stack([torch.zeros(3, 4, 4), torch.full((3, 4, 4), 0.2)])
    assert psnr(black, uneven) == pytest.approx(16.9897, abs=1e-4)


def test_ssim_known_values():
    pytest.importorskip("torchmetrics")
    x = torch.rand(3, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-6)
    # constant frames: SSIM = (2 a b + C1) / (a^2 + b^2 + C1), C1 = 0.01^2
    a, b = torch.full((2, 3, 32, 32), 0.5), torch.full((2, 3, 32, 32), 0.25)
    expected = (2 * 0.5 * 0.25 + 1e-4) / (0.25 + 0.0625 + 1e-4)
    assert ssim(a, b) == pytest.approx(expected, rel=1e-4)  # float32 window moments


def test_lpips_of_identical_frames_is_zero():
    pytest.importorskip("lpips")
    weights = Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
    if not weights.is_file():
        pytest.skip(f"torchvision's AlexNet weights are not downloaded ({weights})")
    x = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    lpips = LPIPS("cpu")
    assert lpips(x, x) == pytest.approx(0.0, abs=1e-6)
    assert lpips(x, 1 - x) > 0.1


def test_a_window_is_scored_after_its_first_frame():
    def same(generated: torch.Tensor, truth: torch.Tensor) -> float:
        return float((generated == truth).float().mean())

    g = torch.Generator().manual_seed(2)
    truth = torch.rand(9, 3, 8, 8, generator=g)
    generated = truth.clone()
    generated[0] = 1 - truth[0]
    latents = torch.randn(3, 4, 2, 2, generator=g)
    rollout = latents.clone()
    rollout[0] += 5
    rollout[1:] += 0.5
    scores = score_window(generated, truth, rollout, latents, lpips=l1, ssim=same)
    assert list(scores) == ["psnr", "ssim", "lpips", "pixel_mse", "latent_mse"]
    assert scores["pixel_mse"] == 0.0 and scores["psnr"] == pytest.approx(120.0)
    assert scores["ssim"] == 1.0 and scores["lpips"] == 0.0
    assert scores["latent_mse"] == pytest.approx(0.25)
    generated[1:] = truth[1:] + 0.1  # clamped to [0, 1] before it is scored
    clamped = (truth[1:] + 0.1).clamp(0, 1)
    scores = score_window(generated, truth, rollout, latents, lpips=l1, ssim=same)
    assert scores["pixel_mse"] == pytest.approx(float((clamped - truth[1:]).square().mean()))
    assert scores["lpips"] == pytest.approx(float((clamped - truth[1:]).abs().mean()))
    with pytest.raises(ValueError, match="differ in shape"):
        score_window(generated[:5], truth, rollout, latents, lpips=l1, ssim=same)


def test_summarize_means_by_map_and_stratum():
    rows = [("a", "vis?", 1.0), ("b", "vis?", 2.0), ("a", "vis0.5", 4.0)]
    samples = [
        {"map_name": m, "stratum": s, "psnr": v, "ssim": v, "lpips": v}
        | {"pixel_mse": v / 10, "latent_mse": v / 100}
        for m, s, v in rows
    ]
    out = summarize(samples)
    assert out["sample_count"] == 3 and out["psnr"] == pytest.approx(7 / 3)
    assert out["by_map"]["a"]["psnr"] == 2.5 and out["by_map"]["b"]["sample_count"] == 1
    assert out["by_stratum"]["vis?"]["latent_mse"] == pytest.approx(0.015)
    assert list(out["by_map"]) == ["a", "b"]
    with pytest.raises(ValueError, match="no windows to summarize"):
        summarize([])
