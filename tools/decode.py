#!/usr/bin/env python3
"""Decode a client's ``latents.npy`` into a 16 fps 672x384 mp4 with the streaming Wan2.2 VAE.

The paper decoded each client in a separate process, on a GPU, with the VAE in bf16 and chunks of 8 latents (the
defaults here). The mp4 is lossy (libx264, quality 8) and its bytes depend on the ffmpeg build: compare
``latents.npy`` for exactness, never the mp4.

Examples::

    python tools/decode.py \\
        --latents runs/dust2-r09/2392812-de_dust2-r09-s000000/2392812-de_dust2-r09-p09/latents.npy \\
        --vae weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
    python tools/decode.py --latents runs/x/latents.npy --config weights/paths.yaml   # VAE from paths.wan22_root
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run from a checkout without installing the package
    sys.path.insert(0, str(REPO))

from worldcast.config.loader import add_config_args, load_cli_config, parse_overrides  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--latents", required=True, nargs="+", help="latents.npy file(s) [N, 48, 24, 42]"
    )
    parser.add_argument(
        "--out", help="mp4 path (default: video.mp4 next to the latents; one input only)"
    )
    parser.add_argument(
        "--vae", help="Wan2.2_VAE.pth (default: <paths.wan22_root>/Wan2.2_VAE.pth from --config)"
    )
    add_config_args(parser)
    parser.add_argument(
        "--chunk", type=int, default=8, help="latents per streaming chunk (paper: 8)"
    )
    parser.add_argument("--device", default="cuda", help="torch device of the VAE (default: cuda)")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="VAE dtype (paper: bf16)",
    )
    args = parser.parse_args(argv)
    if args.out and len(args.latents) != 1:
        raise SystemExit("--out names one file: give one --latents, or omit --out")

    import torch

    from worldcast.engine.inference.decode import decode_to_mp4, load_latents
    from worldcast.modeling.wan22.vae import VAE_CHECKPOINT_NAME, load_wan22_vae

    vae_path = args.vae
    if vae_path is None:
        cfg = load_cli_config(args.config, parse_overrides(args.overrides))
        cfg.paths.require("wan22_root")
        vae_path = str(Path(cfg.paths.wan22_root) / VAE_CHECKPOINT_NAME)
    vae = load_wan22_vae(vae_path, device=args.device, dtype=getattr(torch, args.dtype))
    for path in args.latents:
        out = Path(args.out) if args.out else Path(path).with_name("video.mp4")
        t0 = time.time()
        with torch.no_grad():
            frames = decode_to_mp4(vae, load_latents(path), out, chunk=args.chunk)
        print(f"{path} -> {out}: {frames} frames ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
