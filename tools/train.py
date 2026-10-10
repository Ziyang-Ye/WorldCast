#!/usr/bin/env python3
"""Train one WorldCast stage (docs/training.md)::

    torchrun --nnodes N --nproc_per_node 8 [--node_rank R --master_addr A --master_port P] \\
        tools/train.py --config configs/train/stage2.yaml --config train_paths.yaml \\
        --set run.output_dir=runs/stage2 --set checkpoint.init=...

``--config`` may be repeated (files merged in order); ``--set key=value`` overrides one dotted key;
``--set model.attention=sdpa`` runs without flash-attention (not bit-equal).
``--resume auto`` (the default) continues from the newest complete checkpoint in
``run.output_dir``, a directory resumes from that checkpoint, ``none`` starts fresh; resume is exact
and needs the checkpoint's world size. ``--print-config`` prints the resolved config.
"""

import argparse
import logging
import os
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import yaml

from worldcast import distributed as D
from worldcast.config.loader import (
    add_config_args,
    config_to_dict,
    parse_overrides,
    usage_errors,
)
from worldcast.config.training import load_train_config
from worldcast.engine.checkpoint.training import find_latest_checkpoint, is_resumable
from worldcast.engine.training.build import build_trainer


def main(argv: Sequence[str] | None = None) -> int:
    """Train the stage of the config, or print the config; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument(
        "--resume",
        default="auto",
        metavar="auto|none|DIR",
        help="auto: the newest complete checkpoint in run.output_dir; DIR: that checkpoint; none",
    )
    parser.add_argument("--print-config", action="store_true", help="print the config and exit")
    args = parser.parse_args(argv)

    with usage_errors(parser):
        cfg = load_train_config(args.config, parse_overrides(args.overrides))
    resolved = yaml.safe_dump(config_to_dict(cfg), sort_keys=False)
    if args.print_config:
        print(resolved)
        return 0
    if not cfg.run.output_dir:
        parser.error("set run.output_dir")
    if args.resume == "auto":
        resume = find_latest_checkpoint(cfg.run.output_dir)
    else:
        resume = None if args.resume == "none" else Path(args.resume)
        if resume is not None and not is_resumable(resume):
            parser.error(f"--resume {resume}: not a complete checkpoint with its resume files")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    info = D.init_distributed()
    try:
        if info.is_main:
            out = Path(cfg.run.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.yaml").write_text(resolved)
            batch = cfg.data.batch_size * cfg.optim.grad_accum_steps * info.world_size
            resuming = f", resuming {resume}" if resume else ""
            print(f"stage {cfg.run.stage}, {info.world_size} GPUs, batch {batch}{resuming}")
        build_trainer(cfg, info, resume=resume).fit()
    except Exception:
        if info.world_size == 1:
            raise
        # The other ranks wait in a collective, and the teardown can block before Python prints
        # the traceback: print it now and leave without the teardown (torchrun stops the others).
        print(f"rank {info.rank} failed:", file=sys.stderr)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        D.destroy_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
