#!/usr/bin/env python3
"""Score a checkpoint on the 64 validation windows (``docs/training.md``, "Evaluation").

    python tools/evaluate.py --config configs/train/stage2.yaml --config train_paths.yaml \\
        --checkpoint runs/stage2/checkpoint_model_025000/model_ema.pt \\
        --out runs/eval/stage2.json
    torchrun --nproc_per_node 4 tools/evaluate.py ...     # the windows split over the GPUs

The windows are those of ``validation.index`` (set in train_paths.yaml). The stage config gives the
model, its data and the sampler: ``unipc`` (20 UniPC steps, the whole window at once for the
bidirectional stages, block by block for stage 3) for stages 1_long to 3, ``four_step`` (the four
denoising steps) for stage 4; ``--sampler four_step`` scores a stage-3 model in four steps.
``--checkpoint`` is a release ``.safetensors`` file or a training checkpoint (its ``--weights``
entry, EMA by default). Writes one JSON with PSNR, SSIM, LPIPS, pixel and latent MSE (means overall,
by map and by stratum) and every window's scores.
"""

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from worldcast import distributed as D
from worldcast.config.loader import add_config_args, parse_overrides, usage_errors
from worldcast.config.training import load_train_config
from worldcast.engine.evaluation import PROTOCOLS, Evaluator, load_eval_generator
from worldcast.modeling.build import EMA_KEY


def main(argv: Sequence[str] | None = None) -> int:
    """Score ``--checkpoint`` and write ``--out``; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--checkpoint", required=True, help=".safetensors or a training .pt")
    parser.add_argument(
        "--weights", default=EMA_KEY, help="entry of a training checkpoint to score"
    )
    parser.add_argument("--out", required=True, help="the result JSON")
    parser.add_argument(
        "--sampler", choices=sorted(PROTOCOLS), help="default: four_step for stage 4, else unipc"
    )
    parser.add_argument("--windows", type=int, help="score only the first N windows")
    args = parser.parse_args(argv)

    with usage_errors(parser):
        cfg = load_train_config(args.config, parse_overrides(args.overrides))
    if not cfg.validation.index:
        parser.error("set validation.index (train_paths.yaml)")
    sampler = args.sampler or ("four_step" if cfg.stage.recipe == "distillation" else "unipc")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    info = D.init_distributed()
    try:
        started = time.time()
        evaluator = Evaluator.from_config(
            cfg,
            PROTOCOLS[sampler],
            index=cfg.validation.index,
            device=info.device,
            count=args.windows,
        )
        generator = load_eval_generator(
            cfg, args.checkpoint, weights=args.weights, device=info.device
        )
        with torch.no_grad():
            scores = evaluator.run(generator, info=info)
        if scores is not None:  # rank 0
            result = {
                "stage": cfg.run.stage,
                "checkpoint": str(args.checkpoint),
                "weights": args.weights,
                **scores,
            }
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=1) + "\n")
            means = " ".join(f"{k}={result[k]:.6f}" for k in ("psnr", "ssim", "lpips"))
            print(
                f"{sampler} {result['sample_count']} windows: {means}"
                f" ({time.time() - started:.0f} s) -> {out}"
            )
    finally:
        D.destroy_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
