#!/usr/bin/env python3
"""Encode the fixed prompt with umT5-XXL once, so clients never build the 11 GB text encoder.

Run it on a GPU in bf16: the paper's embedding was the bf16 umT5 forward on CUDA, and CPU bf16 GEMMs
differ from it. The release ships the result (``fixed_prompt_umt5xxl_bf16.safetensors``); this is
how it was made. Needs the ``text`` extra (``pip install -e ".[text]"``) and the encoder of the
Wan2.2 snapshot under ``paths.wan22_root`` (``python tools/download_weights.py --out-dir weights
--with-t5``).

Example::

    python tools/make_prompt_embedding.py --config weights/paths.yaml \\
        --out weights/worldcast/fixed_prompt_umt5xxl_bf16.safetensors
"""

import argparse
import sys
from collections.abc import Sequence

from worldcast.config.inference import load_config
from worldcast.config.loader import add_config_args, parse_overrides, usage_errors
from worldcast.modeling.wan22.text_encoder import (
    FIXED_PROMPT,
    load_text_encoder,
    save_prompt_embedding,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Encode the fixed prompt and write ``--out``; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--out", required=True, help="output .safetensors")
    parser.add_argument("--device", help="torch device, e.g. cuda or cuda:1")
    args = parser.parse_args(argv)

    with usage_errors(parser):
        overrides = parse_overrides(args.overrides)
        if args.device:
            overrides["run.device"] = args.device
        cfg = load_config(args.config, overrides)
        cfg.paths.require("wan22_root")
    embeds = load_text_encoder(cfg.paths.wan22_root, device=cfg.run.device).encode_prompt()
    save_prompt_embedding(embeds, args.out)
    print(f"{FIXED_PROMPT!r}: embeds {tuple(embeds.shape)} {embeds.dtype} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
