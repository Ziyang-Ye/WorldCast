"""GT-paired metrics of one window and their summary over the windows.

A window is scored on every decoded frame after the first (the first is the recorded first frame,
copied into the rollout) and on every latent after the first. PSNR is one pooled MSE over the scored
frames, in dB; SSIM is torchmetrics' (11 x 11 Gaussian window), LPIPS the AlexNet variant of the
``lpips`` package; each is one call over all scored frames. A summary is the plain mean of the
per-window values.

LPIPS loads ``torchvision``'s ImageNet AlexNet (``alexnet-owt-7be5be79.pth``, downloaded from
download.pytorch.org into ``$TORCH_HOME/hub/checkpoints`` on first use) and the package's own linear
layers.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

__all__ = [
    "LPIPS",
    "METRICS",
    "MSE_FLOOR",
    "FrameMetric",
    "pixel_mse",
    "psnr",
    "score_window",
    "ssim",
    "summarize",
]

#: The scored quantities of a window, in result-row order.
METRICS = ("psnr", "ssim", "lpips", "pixel_mse", "latent_mse")
#: Floor of the MSE inside PSNR (a perfect match reads 120 dB).
MSE_FLOOR = 1e-12

#: ``metric(generated, truth) -> float`` on ``[T, 3, H, W]`` frames in ``[0, 1]``.
FrameMetric = Callable[[torch.Tensor, torch.Tensor], float]


def pixel_mse(generated: torch.Tensor, truth: torch.Tensor) -> float:
    """Mean squared error of ``[T, 3, H, W]`` frames in ``[0, 1]``, pooled over all of them."""
    return float((generated.float() - truth.float()).square().mean().item())


def psnr(generated: torch.Tensor, truth: torch.Tensor) -> float:
    """PSNR in dB of ``[T, 3, H, W]`` frames in ``[0, 1]``: of their pooled MSE."""
    return -10.0 * math.log10(max(pixel_mse(generated, truth), MSE_FLOOR))


def ssim(generated: torch.Tensor, truth: torch.Tensor) -> float:
    """Mean SSIM of ``[T, 3, H, W]`` frames in ``[0, 1]``."""
    from torchmetrics.functional.image import structural_similarity_index_measure

    value = structural_similarity_index_measure(
        generated.float().clamp(0, 1), truth.float().clamp(0, 1), data_range=1.0
    )
    return float(value.item())


class LPIPS:
    """LPIPS (AlexNet) on one device, built on first use."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._model = None

    def __call__(self, generated: torch.Tensor, truth: torch.Tensor) -> float:
        """Mean LPIPS of ``[T, 3, H, W]`` frames in ``[0, 1]``."""
        if self._model is None:
            import lpips

            self._model = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
            self._model.requires_grad_(False)
        left = (generated.float().clamp(0, 1) * 2.0 - 1.0).to(self.device)
        right = (truth.float().clamp(0, 1) * 2.0 - 1.0).to(self.device)
        with torch.no_grad():
            return float(self._model(left, right).mean().item())


def score_window(
    generated: torch.Tensor,
    truth: torch.Tensor,
    generated_latents: torch.Tensor,
    truth_latents: torch.Tensor,
    *,
    lpips: FrameMetric,
    ssim: FrameMetric = ssim,
) -> dict[str, float]:
    """The metrics of one window.

    Args:
        generated (Tensor): ``[T, 3, H, W]`` decoded rollout in ``[0, 1]``, ``T = 1 + 4 (F - 1)``.
        truth (Tensor): ``[T, 3, H, W]`` the window's cached latents decoded the same way.
        generated_latents (Tensor): ``[F, 48, 24, 42]`` the rollout's latents.
        truth_latents (Tensor): ``[F, 48, 24, 42]`` the cached latents.
        lpips (FrameMetric): LPIPS, for example :class:`LPIPS`.
        ssim (FrameMetric): SSIM; default :func:`ssim`.

    Returns:
        dict[str, float]: :data:`METRICS` over frames and latents ``1 ..``.
    """
    if generated.shape != truth.shape or generated_latents.shape != truth_latents.shape:
        raise ValueError("generated and GT windows differ in shape")
    frames, ground_truth = generated[1:].clamp(0, 1), truth[1:].clamp(0, 1)
    latent_mse = (generated_latents[1:].float() - truth_latents[1:].float()).square().mean()
    return {
        "psnr": psnr(frames, ground_truth),
        "ssim": ssim(frames, ground_truth),
        "lpips": lpips(frames, ground_truth),
        "pixel_mse": pixel_mse(frames, ground_truth),
        "latent_mse": float(latent_mse.item()),
    }


def _mean(samples: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {"sample_count": len(samples)}
    out.update({key: sum(s[key] for s in samples) / len(samples) for key in METRICS})
    return out


def summarize(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Means over the windows (in the given order), by map and by stratum.

    Args:
        samples (Sequence[Mapping]): per-window rows with :data:`METRICS`, ``map_name`` and
            ``stratum``.

    Returns:
        dict: ``sample_count``, the :data:`METRICS` means, ``by_map`` and ``by_stratum``.
    """
    if not samples:
        raise ValueError("no windows to summarize")
    groups: dict[str, dict[str, list]] = {"by_map": {}, "by_stratum": {}}
    for sample in samples:
        groups["by_map"].setdefault(str(sample["map_name"]), []).append(sample)
        groups["by_stratum"].setdefault(str(sample["stratum"]), []).append(sample)
    out: dict[str, Any] = _mean(samples)
    for name, group in groups.items():
        out[name] = {key: _mean(rows) for key, rows in sorted(group.items())}
    return out
