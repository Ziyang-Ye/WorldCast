"""A tiny stage config and an evaluator over synthetic windows, with stand-ins for the VAE, SSIM and
LPIPS (CPU)."""

import dataclasses
from pathlib import Path

import torch

from tests.engine.training.support import RUN_DIMS, make_item, write_prompt_embedding
from worldcast.config.training import TrainConfig, paper_config
from worldcast.engine.evaluation.evaluator import Evaluator
from worldcast.engine.evaluation.protocols import Protocol, select_windows
from worldcast.engine.generator import load_prompt_embeds


def l1(generated: torch.Tensor, truth: torch.Tensor) -> float:
    """A stand-in for LPIPS: the mean absolute pixel difference."""
    return float((generated - truth).abs().mean())


def l1_similarity(generated: torch.Tensor, truth: torch.Tensor) -> float:
    """A stand-in for SSIM: one minus the mean absolute pixel difference."""
    return 1.0 - l1(generated, truth)


class PixelStub(torch.nn.Module):
    """A stand-in VAE: three channels of each latent at latent resolution, four frames per latent
    after the first."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(()))

    def decode(self, z: torch.Tensor, scale: list[torch.Tensor]) -> torch.Tensor:
        x = z[:, :3] * self.gain
        return torch.cat([x[:, :, :1], x[:, :, 1:].repeat_interleave(4, dim=2)], dim=2)


def stage_config(tmp_path: Path, stage: str, **overrides) -> TrainConfig:
    """The paper config of ``stage`` with the tiny generator, SDPA and a prompt-embedding file."""
    prompt = tmp_path / "prompt.safetensors"
    if not prompt.exists():
        write_prompt_embedding(prompt)
    return paper_config(
        stage,
        {
            "model.dims": RUN_DIMS,
            "model.attention": "sdpa",
            "run.output_dir": str(tmp_path / f"run{stage}"),
            "data.prompt_embedding": str(prompt),
            **overrides,
        },
    )


def tiny_evaluator(
    cfg: TrainConfig, protocol: Protocol, index_rows: list[dict], count: int = 2
) -> Evaluator:
    """An evaluator of ``count`` synthetic windows (``make_item``) with stand-ins for the VAE, SSIM
    and LPIPS."""
    windows = select_windows(index_rows, dataclasses.replace(protocol, count=count))
    items = {}
    for window in windows:
        items[window.index] = make_item(100 + window.index)
        identity = {"media_id": window.media_id, "start_frame": window.start_frame}
        items[window.index]["metadata"] = identity
    return Evaluator(
        cfg,
        protocol,
        windows=windows,
        load_item=lambda window: items[window.index],
        prompt_embeds=load_prompt_embeds(cfg.data.prompt_embedding, None, "cpu"),
        vae=PixelStub(),
        device="cpu",
        lpips=l1,
        ssim=l1_similarity,
    )
