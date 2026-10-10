#!/usr/bin/env python3
"""Decode a client's ``latents.npy`` into a 16 fps 672 x 384 mp4 with the streaming Wan2.2 VAE.

The VAE is ``Wan2.2_VAE.pth`` under ``paths.wan22_root``, in the generator's dtype on the device.
The paper decoded each client on a GPU with the VAE in bf16 and chunks of 8 latents (the defaults
here; the CPU decodes in float32). The mp4 is lossy and its bytes depend on the ffmpeg build:
compare ``latents.npy``, never the mp4.

Example::

    python tools/decode.py --config weights/paths.yaml \\
        --latents runs/examples/dust2_r09/*/*/latents.npy
"""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from worldcast.config.inference import load_config
from worldcast.config.loader import add_config_args, parse_overrides, usage_errors
from worldcast.engine.generator import load_vae
from worldcast.engine.inference.decode import DEFAULT_DECODE_CHUNK, decode_to_mp4, load_latents


def main(argv: Sequence[str] | None = None) -> int:
    """Decode the named ``latents.npy`` files; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--latents", required=True, nargs="+", help="latents.npy [N, 48, 24, 42]")
    parser.add_argument("--out", help="mp4 path (default: video.mp4 next to the latents)")
    parser.add_argument("--chunk", type=int, default=DEFAULT_DECODE_CHUNK, help="latents per call")
    parser.add_argument("--device", help="torch device, e.g. cuda or cuda:1")
    args = parser.parse_args(argv)
    if args.out and len(args.latents) != 1:
        parser.error("--out names one file: give one --latents, or omit --out")

    with usage_errors(parser):
        overrides = parse_overrides(args.overrides)
        if args.device:
            overrides["run.device"] = args.device
        cfg = load_config(args.config, overrides)
        cfg.paths.require("wan22_root")
        vae = load_vae(cfg.paths.wan22_root, cfg.run.device)  # a missing VAE file: a usage error
    for path in args.latents:
        out = Path(args.out) if args.out else Path(path).with_name("video.mp4")
        t0 = time.time()
        frames = decode_to_mp4(vae, load_latents(path), out, chunk=args.chunk)
        print(f"{path} -> {out}: {frames} frames ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
