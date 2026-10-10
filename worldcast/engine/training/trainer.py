"""The step loop every training stage shares: metrics, validation, checkpoints and resume.

:mod:`worldcast.engine.training.build` builds a stage's trainer; the recipes
(:mod:`worldcast.engine.training.recipes`) are its steps.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch
from torch import nn

from worldcast.config.training import TrainConfig
from worldcast.distributed.process_group import DistInfo, capture_rng_state, restore_rng_state
from worldcast.engine.checkpoint.training import load_rank_state, save_checkpoint
from worldcast.engine.evaluation.validation import Validation
from worldcast.engine.optim.ema import ShardedEMA
from worldcast.sampling.schedulers import FlowMatchScheduler
from worldcast.utils.precision import generator_dtype

__all__ = ["STEP_TIMES", "BatchStream", "Trainer"]

log = logging.getLogger(__name__)

T = TypeVar("T")

#: The times of an optimizer step that every recipe measures, s: the wait for the batches, the
#: forwards and backwards, the clip and the optimizer.
STEP_TIMES = ("data_wait_sec", "forward_backward_time_sec", "optimizer_time_sec")


class BatchStream(Protocol):
    """The batches of one rank, resumable: :class:`worldcast.data.stream.ResumableDataStream`."""

    def next_batch(self) -> dict[str, Any]:
        """The next collated batch."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """The stream's position."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Continue from a position."""
        ...


class Trainer(ABC):
    """The step loop of every stage: metrics, validation, checkpoints and resume.

    A recipe implements :meth:`train_step`, :meth:`checkpoint_models`, :meth:`optimizers` and
    :attr:`ema_model`.

    Args:
        cfg (TrainConfig): the run.
        info (DistInfo): this process.
        data (BatchStream): this rank's batches.
        prompt_embeds (Tensor): ``[1, L, 4096]`` the fixed prompt's embedding.
    """

    def __init__(
        self, cfg: TrainConfig, info: DistInfo, *, data: BatchStream, prompt_embeds: torch.Tensor
    ) -> None:
        self.cfg, self.info, self.data = cfg, info, data
        self.prompt_embeds, self.device = prompt_embeds, info.device
        #: The compute dtype: the generator's on the process's device.
        self.dtype = generator_dtype(info.device)
        self.scheduler = FlowMatchScheduler()
        self.step = 0
        self.ema: ShardedEMA | None = None
        output = cfg.run.output_dir
        self.metrics_path = Path(output) / "metrics.jsonl" if output else None
        self.validation = Validation(cfg, info) if cfg.validation.interval else None

    # ------------------------------------------------------------------------------- the recipe
    @abstractmethod
    def train_step(self) -> dict[str, Any]:
        """One optimizer step; it advances :attr:`step` by one.

        Returns:
            dict[str, Any]: this rank's metrics of the step, which :meth:`log` reduces over the
            ranks by key.
        """

    @abstractmethod
    def checkpoint_models(self) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """``{file: {payload key: state}}`` of the weights to save (a collective under FSDP)."""

    @abstractmethod
    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """The optimizers by name, as a checkpoint keeps them."""

    @property
    @abstractmethod
    def ema_model(self) -> nn.Module:
        """The module the EMA follows and validation scores."""

    # --------------------------------------------------------------------------------- the loop
    def fit(self) -> None:
        """Train to ``run.max_steps``, validating every ``validation.interval`` steps, saving every
        ``checkpoint.interval`` steps and at the end. Without ``run.output_dir`` nothing is
        written."""
        cfg = self.cfg
        while self.step < cfg.run.max_steps:
            self.log(self.train_step())
            if self.validation is not None and self.step % cfg.validation.interval == 0:
                self.validate()
            last = self.step >= cfg.run.max_steps
            due = cfg.checkpoint.interval and self.step % cfg.checkpoint.interval == 0
            if cfg.run.output_dir and (due or last):
                self.save()

    def validate(self) -> None:
        """Score the EMA weights (the live ones before the EMA starts) on the validation windows
        and append the row to ``metrics.jsonl``; every RNG state is restored, so the run continues
        as without it."""
        rng = capture_rng_state(self.device)
        try:
            row = self.validation(self.ema_model, step=self.step, ema=self.ema)
        finally:
            restore_rng_state(rng, self.device)
        if row is not None:
            self._append_row(row)

    def accumulate(
        self, micro_batch: Callable[[dict[str, Any], int], T], times: dict[str, float]
    ) -> list[T]:
        """The ``optim.grad_accum_steps`` micro-batches of one optimizer step.

        Args:
            micro_batch (Callable): ``micro_batch(batch, index)``: the forward and the backward of
                one micro-batch.
            times (dict[str, float]): receives the wait for the batches (``data_wait_sec``) and
                the time in ``micro_batch`` (``forward_backward_time_sec``).

        Returns:
            list: what ``micro_batch`` returned, per micro-batch.
        """
        out = []
        for index in range(self.cfg.optim.grad_accum_steps):
            started = time.perf_counter()
            batch = self.data.next_batch()
            times["data_wait_sec"] += time.perf_counter() - started
            started = time.perf_counter()
            out.append(micro_batch(batch, index))
            times["forward_backward_time_sec"] += self.seconds_since(started)
        return out

    def start_step(self) -> float:
        """Start the step's clock (and, on CUDA, its peak-memory count)."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return time.perf_counter()

    def seconds_since(self, started: float) -> float:
        """Wall seconds since ``started`` (``time.perf_counter``), once the GPU is done."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - started

    def resources(self, started: float) -> dict[str, float]:
        """The step's time and, on CUDA, its peak memory (reset at the step's start)."""
        out = {"step_time_sec": self.seconds_since(started)}
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device)
            out["max_memory_allocated_gib"] = peak / 2**30
        return out

    # ------------------------------------------------------------------------------ the metrics
    def log(self, metrics: Mapping[str, Any]) -> None:
        """Append a step's row to ``metrics.jsonl`` on rank 0, and log it every
        ``run.log_interval`` steps.

        The row: the step, a timestamp, the mean over ranks of each float (the losses and norms),
        the maximum over ranks of the keys ending in ``_sec`` or ``_gib`` (the times and the
        memory), and rank 0's other values (the ``lr_*``, flags, draws).
        """
        floats = sorted(
            k for k, v in metrics.items() if isinstance(v, float) and not k.startswith("lr_")
        )
        maxima = [k for k in floats if k.endswith(("_sec", "_gib"))]
        means = [k for k in floats if k not in maxima]
        values = torch.tensor(
            [metrics[k] for k in means + maxima], dtype=torch.float64, device=self.device
        )
        if self.info.initialized:
            import torch.distributed as dist

            mean_values, max_values = values[: len(means)], values[len(means) :]
            if means:
                dist.all_reduce(mean_values, op=dist.ReduceOp.SUM)
            if maxima:
                dist.all_reduce(max_values, op=dist.ReduceOp.MAX)
            values = torch.cat([mean_values / self.info.world_size, max_values])
        if not self.info.is_main:
            return
        reduced = dict(zip(means + maxima, values.tolist()))
        row = {**{k: v for k, v in metrics.items() if k not in reduced}, **reduced}
        self._append_row({"step": self.step, **row})
        if self.cfg.run.log_interval and self.step % self.cfg.run.log_interval == 0:
            shown = (f"{k}={v:.6g}" for k, v in row.items() if isinstance(v, float))
            log.info("step=%d %s", self.step, " ".join(shown))

    def _append_row(self, row: Mapping[str, Any]) -> None:
        """Append ``row``, with its ``step`` and a UTC timestamp first, to ``metrics.jsonl``."""
        if self.metrics_path is None:
            return
        stamped = {"step": row["step"], "timestamp": datetime.now(timezone.utc).isoformat(), **row}
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped) + "\n")

    # ------------------------------------------------------------------- the EMA, save, resume
    def start_ema(self) -> None:
        """Start the EMA from the live weights of :attr:`ema_model`."""
        self.ema = ShardedEMA.start(self.ema_model, self.cfg.ema.decay)

    def save(self) -> Path:
        """Write ``checkpoint_model_<step>/`` (:mod:`worldcast.engine.checkpoint.training`)."""
        if not self.cfg.run.output_dir:
            raise ValueError("run.output_dir is not set")
        models = self.checkpoint_models()
        rank_state = {
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers().items()},
            "rng": capture_rng_state(self.device),
            "data": self.data.state_dict(),
            "ema": None if self.ema is None else self.ema.state_dict(),
        }
        metadata = {
            "step": self.step,
            "base_seed": self.cfg.run.seed,
            "stage": self.cfg.run.stage,
            "run_name": self.cfg.run.name,
        }
        return save_checkpoint(
            self.cfg.run.output_dir,
            step=self.step,
            info=self.info,
            metadata=metadata,
            models=models if self.info.is_main else {},
            rank_state=rank_state,
            keep=self.cfg.checkpoint.keep,
            keep_shards=self.cfg.checkpoint.keep_shards,
        )

    def resume(self, directory: str | Path) -> None:
        """Continue exactly from a checkpoint of :meth:`save` (its weights were loaded when the
        models were built): optimizers, step, EMA, data position and RNG; same topology."""
        rank = load_rank_state(directory, self.info)
        if int(rank["base_seed"]) != self.cfg.run.seed:
            raise RuntimeError("the checkpoint's run.seed differs from this run's")
        for name, optimizer in self.optimizers().items():
            optimizer.load_state_dict(rank["optimizers"][name])
        self.step = int(rank["step"])
        if rank["ema"] is not None:
            self.ema = ShardedEMA.resume(self.ema_model, self.cfg.ema.decay, rank["ema"])
        self.data.load_state_dict(rank["data"])
        restore_rng_state(rank["rng"], self.device)
