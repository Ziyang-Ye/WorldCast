"""Scoring a generator on an evaluation protocol (``tools/evaluate.py``; in training through
:class:`~worldcast.engine.evaluation.validation.Validation`).

Per window: the cached latents and the window's conditions (prompt, controls, the player state field
of the GT player states gated by the GT visibility labels, as in training), the window's noise from
``torch.Generator(seed=noise_seed)`` (the whole 41-latent draw in the generator's dtype, frame 0
dropped), the protocol's sampler, then the rollout and the cached latents decoded with the Wan2.2
VAE and scored (:func:`~worldcast.engine.evaluation.metrics.score_window`). Ranks take every
``world_size``-th window.

Scoring a window draws from the window's own generator only. Building the models does draw from
the global RNG (their random initialisation: the VAE in :meth:`Evaluator.from_config`, LPIPS on the
first window scored), which is why the trainer restores every RNG state after a validation.
"""

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldcast.config.training import WINDOW_LATENT_FRAMES, TrainConfig
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.recordings import MediaIndex
from worldcast.data.training import BucketWindow, BucketWindows
from worldcast.data.window import DataPaths, WindowSpec, collate_windows
from worldcast.distributed.process_group import DistInfo
from worldcast.engine.generator import client_cameras, load_prompt_embeds, load_vae, window_inputs
from worldcast.engine.inference.decode import decode_pixels
from worldcast.engine.stage import built_state, field_config, generator_config
from worldcast.modeling.build import EMA_KEY, build_inference_generator, read_generator
from worldcast.modeling.ray_embedding import RayConditions
from worldcast.modeling.wan22.attention import attention_kernel
from worldcast.modeling.wan22.model import CausalGeneratorAdapter, WorldCastGenerator
from worldcast.modeling.wan22.training import TrainingForward
from worldcast.modeling.wan22.vae import Wan22VAE
from worldcast.player_state import field_builder
from worldcast.sampling.sampler import Sampler
from worldcast.utils.precision import enable_tf32, generator_dtype

from .metrics import LPIPS, METRICS, FrameMetric, score_window, ssim, summarize
from .protocols import EvalWindow, Protocol, load_windows
from .samplers import sample_bidirectional, sample_block_causal, sample_four_step

__all__ = ["Evaluator", "load_eval_generator"]

log = logging.getLogger(__name__)


def load_eval_generator(
    cfg: TrainConfig,
    checkpoint: str | Path,
    *,
    weights: str = EMA_KEY,
    device: torch.device | str,
) -> WorldCastGenerator:
    """The generator of a stage for evaluation
    (:func:`~worldcast.modeling.build.build_inference_generator`: without the visibility probe, in
    the generator's dtype on ``device``), from the weights the stage's generator takes of the
    checkpoint.

    Args:
        cfg (TrainConfig): the stage.
        checkpoint (str | Path): a release ``.safetensors`` file, or a training checkpoint.
        weights (str): the payload entry of a training checkpoint to score.
        device (torch.device | str): where the generator lives.
    """
    config = generator_config(cfg)
    return build_inference_generator(
        config,
        built_state(read_generator(checkpoint, weights), config),
        device=device,
        attention=attention_kernel(cfg.model.attention, device),
    )


def _rays(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """The ray embedding's conditions of a window without memory frames, from the client's
    recorded cameras."""
    slot = int(batch["client_slot"][0])
    cameras = client_cameras(
        batch["player_states"][0, slot],
        {key: np.asarray(batch[key][0]) for key in OBSERVER_SIGNAL_KEYS},
        batch["player_weapon_ids"][0, slot],
        int(batch["latents"].shape[1]),
    )
    rays = RayConditions.contiguous(cameras["window_c2w"], cameras["window_tans"])
    return rays.conditions(device)


def _window_dataset(cfg: TrainConfig, windows: list[EvalWindow]) -> BucketWindows:
    """The windows' loader items (``BucketWindows.load_window(window.index)``) from the stage's data
    paths and window sampling (camera encoding included)."""
    paths = DataPaths.from_config(cfg.data, "data")
    rows = [
        BucketWindow(
            media_id=w.media_id,
            start_frame=w.start_frame,
            match_id=int(w.record["match_id"]),
            round=int(w.record["round"]),
            weight=1.0,
        )
        for w in windows
    ]
    spec = WindowSpec(cfg.stage.latent_frames, camera_encoding=cfg.stage.camera_encoding)
    return BucketWindows(
        rows,
        media_index=MediaIndex.load(paths.media_index),
        paths=paths,
        spec=spec,
    )


class Evaluator:
    """One protocol on one stage's model, on one device: the windows, their loader, the prompt
    embedding, the VAE and LPIPS. :meth:`from_config` builds it from the stage's config.

    A bidirectional model (stages 1_long, 2, 2s) is sampled on the whole window, a block-causal
    one (stages 3 and 4) block by block; the four-step sampler needs a block-causal model, and
    the four-step generator of stage 4 that sampler.

    Args:
        cfg (TrainConfig): the stage: its model switches and field.
        protocol (Protocol): :data:`~worldcast.engine.evaluation.protocols.UNIPC` or
            :data:`~worldcast.engine.evaluation.protocols.FOUR_STEP`.
        windows (list[EvalWindow]): the windows to score.
        load_item (Callable): ``load_item(window)``: the window's loader item (no batch axis).
        prompt_embeds (Tensor): ``[1, 512, 4096]`` the fixed prompt's embedding.
        vae (Wan22VAE): decodes the rollout and the cached latents.
        device (torch.device | str): where the models run, in the generator's dtype there.
        lpips (FrameMetric | None): ``lpips(generated, truth)``; default :class:`LPIPS`.
        ssim (FrameMetric): ``ssim(generated, truth)``; default
            :func:`~worldcast.engine.evaluation.metrics.ssim`.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        protocol: Protocol,
        *,
        windows: list[EvalWindow],
        load_item: Callable[[EvalWindow], dict],
        prompt_embeds: torch.Tensor,
        vae: Wan22VAE,
        device: torch.device | str,
        lpips: FrameMetric | None = None,
        ssim: FrameMetric = ssim,
    ) -> None:
        stage, recipe = cfg.run.stage, cfg.stage.recipe
        self.bidirectional = recipe == "bidirectional"
        if protocol.sampler == "four_step" and self.bidirectional:
            raise ValueError(f"stage {stage} is bidirectional: score it with the sampler unipc")
        if protocol.sampler == "unipc" and recipe == "distillation":
            raise ValueError(
                f"stage {stage} is the four-step generator: score it with the sampler four_step"
            )
        if cfg.stage.latent_frames != WINDOW_LATENT_FRAMES:
            raise ValueError(
                f"the protocols score windows of {WINDOW_LATENT_FRAMES} latent frames; stage"
                f" {stage} trains on {cfg.stage.latent_frames}"
            )
        self.cfg, self.protocol, self.windows = cfg, protocol, list(windows)
        self.device = torch.device(device)
        self.dtype = generator_dtype(self.device)
        self.load_item, self.prompt_embeds, self.vae = load_item, prompt_embeds, vae
        self.lpips = lpips if lpips is not None else LPIPS(self.device)
        self.ssim = ssim
        self.field = field_config(cfg, bidirectional=self.bidirectional)

    @classmethod
    def from_config(
        cls,
        cfg: TrainConfig,
        protocol: Protocol,
        *,
        index: str | Path,
        device: torch.device | str,
        count: int | None = None,
    ) -> "Evaluator":
        """The protocol's windows of ``index`` (the first ``count``), loaded from ``cfg.data``; the
        prompt embedding of ``cfg.data.prompt_embedding`` and the VAE of ``cfg.model.wan22_root``.
        ``cfg.validation.index_sha256`` is the sha256 of an index you rebuilt.

        On CUDA it sets the process's TF32 switches as ``protocol.tf32`` says.
        """
        device = torch.device(device)
        if device.type == "cuda":
            enable_tf32(matmul=protocol.tf32)
        protocol, windows = load_windows(index, protocol, cfg.validation.index_sha256)
        windows = windows[:count]
        dataset = _window_dataset(cfg, windows)
        return cls(
            cfg,
            protocol,
            windows=windows,
            load_item=lambda window: dataset.load_window(window.index),
            prompt_embeds=load_prompt_embeds(
                cfg.data.prompt_embedding, cfg.model.wan22_root, device
            ),
            vae=load_vae(cfg.model.wan22_root, device),
            device=device,
        )

    def inputs(self, window: EvalWindow) -> tuple[torch.Tensor, dict[str, Any]]:
        """The window's cached latents ``[1, 41, 48, 24, 42]`` and its generator conditions."""
        item = self.load_item(window)
        meta = item["metadata"]
        if (meta["media_id"], int(meta["start_frame"])) != (window.media_id, window.start_frame):
            raise RuntimeError(f"the loader served {meta['media_id']}@{meta['start_frame']}")
        batch = collate_windows([item])
        scene_state = self.cfg.stage.scene_state
        rays = None
        if scene_state and self.protocol.sampler != "four_step":
            rays = _rays(batch, self.device)
        inputs = window_inputs(
            batch,
            prompt_embeds=self.prompt_embeds,
            device=self.device,
            dtype=self.dtype,
            observer_signals=scene_state,
            rays=rays,
        )
        return inputs.latents, inputs.conditions

    def rollout(
        self,
        generator: WorldCastGenerator,
        window: EvalWindow,
        clean: torch.Tensor,
        conditions: Mapping[str, Any],
    ) -> torch.Tensor:
        """The protocol's rollout of one window, ``[1, 41, 48, 24, 42]`` in the generator's
        dtype."""
        rng = torch.Generator(device=self.device).manual_seed(window.noise_seed)
        noise = torch.randn(clean.shape, generator=rng, device=self.device, dtype=self.dtype)
        noise, first_frame = noise[:, 1:].contiguous(), clean[:, :1]
        builder = field_builder(self.field, visibility_gate=self.cfg.model.visibility_gate)
        input_dtype = self.dtype if self.device.type == "cuda" else None
        steps = self.protocol.denoising_steps
        if self.bidirectional:
            module = TrainingForward(generator, builder, input_dtype=input_dtype)
            return sample_bidirectional(module, noise, first_frame, conditions, steps=steps)
        cache = generator.allocate_kv_cache(clean.shape[1])
        if self.protocol.sampler == "four_step":
            # as the reference rows were scored: float32 timesteps and player states reach the
            # generator uncast
            input_dtype = None
        call = CausalGeneratorAdapter(generator, builder, input_dtype=input_dtype)
        sampler = Sampler.create(call, cache, context_noise=0, rng=rng)
        if self.protocol.sampler == "four_step":
            return sample_four_step(sampler, noise, first_frame, conditions)
        return sample_block_causal(sampler, noise, first_frame, conditions, steps=steps)

    @torch.no_grad()
    def score(self, generator: WorldCastGenerator, window: EvalWindow) -> dict[str, Any]:
        """One window's row: its identity and the
        :data:`~worldcast.engine.evaluation.metrics.METRICS`."""
        clean, conditions = self.inputs(window)
        latents = self.rollout(generator, window, clean, conditions)
        metrics = score_window(
            decode_pixels(self.vae, latents),
            decode_pixels(self.vae, clean),
            latents[0],
            clean[0],
            lpips=self.lpips,
            ssim=self.ssim,
        )
        return {
            "index": window.index,
            "media_id": window.media_id,
            "start_frame": window.start_frame,
            "map_name": window.map_name,
            "stratum": window.stratum,
            "noise_seed": window.noise_seed,
            **metrics,
        }

    def run(
        self, generator: WorldCastGenerator, *, info: DistInfo | None = None
    ) -> dict[str, Any] | None:
        """Score this rank's windows (every ``world_size``-th) and gather them.

        Args:
            generator (WorldCastGenerator): the model to score.
            info (DistInfo | None): this rank of the process group; ``None``: one process.

        Returns:
            dict | None: on rank 0 the result row (the protocol's fields, the means overall,
            ``by_map``, ``by_stratum`` and the per-window ``samples``); ``None`` elsewhere.
        """
        rank, world = (info.rank, info.world_size) if info is not None else (0, 1)
        samples = []
        for window in self.windows[rank::world]:
            started = time.perf_counter()
            samples.append(self.score(generator, window))
            scores = " ".join(f"{key}={samples[-1][key]:.4f}" for key in METRICS[:3])
            seconds = time.perf_counter() - started
            log.info("%s@%d %s (%.0f s)", window.media_id, window.start_frame, scores, seconds)
        if info is not None and info.initialized:
            import torch.distributed as dist

            gathered: list = [None] * world
            dist.all_gather_object(gathered, samples)
            samples = [sample for part in gathered for sample in part]
        if rank != 0:
            return None
        samples.sort(key=lambda sample: sample["index"])
        return {**self.protocol.fields(), **summarize(samples), "samples": samples}
