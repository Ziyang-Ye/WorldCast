"""In-training validation: the trainer's EMA weights scored on the validation windows every
``validation.interval`` steps."""

import logging
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from worldcast.config.training import TrainConfig
from worldcast.distributed.fsdp import full_state_dict, live_module
from worldcast.distributed.process_group import DistInfo
from worldcast.engine.checkpoint.training import generator_state
from worldcast.engine.optim.ema import ShardedEMA
from worldcast.modeling.wan22.model import WorldCastGenerator
from worldcast.utils.weights import module_from_state

from .evaluator import Evaluator
from .metrics import METRICS
from .protocols import UNIPC

__all__ = ["Validation", "gather_generator"]

log = logging.getLogger(__name__)


@torch.no_grad()
def gather_generator(
    model: nn.Module, ema: ShardedEMA | None = None, *, device: torch.device, dtype: torch.dtype
) -> WorldCastGenerator:
    """A plain copy of a trainer's generator on every rank (a collective under FSDP).

    Args:
        model (nn.Module): the trainer's model (``GeneratorForward``, FSDP-wrapped or not).
        ema (ShardedEMA | None): its EMA, whose weights are copied instead of the live ones.
        device (torch.device): where the copy lives.
        dtype (torch.dtype): bf16 on the paper path: the weights FSDP's mixed precision computes
            with.
    """
    with nullcontext() if ema is None else ema.applied(model):
        full = generator_state(full_state_dict(model, on_every_rank=True))
        state = {
            key: value.to(device=device, dtype=dtype, copy=True) for key, value in full.items()
        }
        del full
    source = live_module(model).generator
    generator = module_from_state(
        lambda: WorldCastGenerator(source.config, attention=source.attention), state
    )
    return generator.eval().requires_grad_(False)


class Validation:
    """In-training validation: the EMA weights on the validation windows with the protocol
    :data:`~worldcast.engine.evaluation.protocols.UNIPC`.

    :class:`~worldcast.engine.training.trainer.Trainer` builds it when ``validation.interval`` is
    set (1000 steps in the paper's runs), calls it on every rank, appends its row to
    ``metrics.jsonl`` and restores every RNG state afterwards. Each rank gathers a copy of the
    generator in the training dtype (:func:`gather_generator`, about 10 GB in bf16) and scores its
    share of the windows; the trainer's weights are restored.

    Args:
        cfg (TrainConfig): the run; its ``validation.index`` is the window index.
        info (DistInfo): the process.

    Attributes:
        evaluator (Evaluator | None): scores the windows; built from ``cfg`` on the first call
            (inside the trainer's RNG capture: building it draws from the global RNG).
    """

    def __init__(self, cfg: TrainConfig, info: DistInfo) -> None:
        self.cfg, self.info = cfg, info
        self.evaluator: Evaluator | None = None

    def __call__(
        self, model: nn.Module, *, step: int, ema: ShardedEMA | None = None
    ) -> dict[str, Any] | None:
        """Score the trainer's ``model`` (its EMA weights when ``ema`` is given) at ``step``.

        Returns:
            dict[str, Any] | None: on rank 0 the row (``event: validation``, the step, the
            ``weight_source`` and the evaluator's result row); ``None`` elsewhere.
        """
        if self.evaluator is None:
            self.evaluator = Evaluator.from_config(
                self.cfg, UNIPC, index=self.cfg.validation.index, device=self.info.device
            )
        evaluator = self.evaluator
        generator = gather_generator(model, ema, device=evaluator.device, dtype=evaluator.dtype)
        try:
            scores = evaluator.run(generator, info=self.info)
        finally:
            del generator
            if evaluator.device.type == "cuda":
                torch.cuda.empty_cache()
        if scores is None:
            return None
        means = " ".join(f"{key}={scores[key]:.6f}" for key in METRICS[:3])
        log.info("validation step=%d %s windows=%d", step, means, scores["sample_count"])
        weight_source = "ema" if ema is not None else "live"
        return {"event": "validation", "step": int(step), "weight_source": weight_source, **scores}
