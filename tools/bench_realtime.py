#!/usr/bin/env python3
"""One client's latency on a GPU: input to photon, denoising, frame rate.

One client of a round (recorded player states and controls, no other client) runs through latent 24
and then ``--blocks`` blocks under each setting, paced as the demo's workers pace
(``worldcast.demo.latency``): a block starts as late as a display at ``--fps`` allows, so its
controls are taken as late as possible. Per setting: the input-to-photon latency of inputs spread
evenly over the session (with 5 ms assumed for the network and 10 ms for the display), the time
from a block's controls to its first frame, the frames generated per second, the stages of a block
(the DiT, the decoder per block and per latent frame, the latents to the first frame, the period
between blocks) and whether the latents equal those of the first setting bit for bit.

    python tools/bench_realtime.py --config weights/paths.yaml \\
        --config examples/data/mirage_r16/config.yaml --index-row 0 --blocks 24 \\
        --taehv weights/taew2_2.pth --out-dir runs/bench

Settings (``--settings``): ``exact`` (the paper's latents: ``model.attention=flash``, no
``torch.compile``), ``fast`` (the demo's default: ``torch.compile`` and FlashAttention-3). Writes
``<out-dir>/bench.json``.

The decoder (``--decoder``): ``taehv``, the tiny decoder (the demo's), or ``wan``, the Wan2.2 VAE
(the paper's frames). ``--decoder-device cuda:1`` runs it on a second card
(``CUDA_VISIBLE_DEVICES=0,1``), with the next block starting while the frames of this one are
decoded, as a worker does; ``--fps`` is then a rate the decoder sustains (docs/demo.md, "The
original VAE on a second card"). ``--decoder-channels-last`` keeps the VAE's decoding convolutions
in the channels-last layout: faster, and not the paper's pixels byte for byte:

    python tools/bench_realtime.py ... --decoder wan --decoder-device cuda:1 --fps 11
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from worldcast.config.inference import load_config
from worldcast.config.loader import add_config_args, parse_overrides, usage_errors
from worldcast.data import load_round_index_row
from worldcast.demo.config import PlayConfig
from worldcast.demo.latency import SETTINGS, UNMEASURED_BLOCKS, compare_settings
from worldcast.engine.inference.serving import ServingOptions
from worldcast.modeling.wan22.attention import flash_attention_backend


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the client of ``--index-row`` under each setting; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument(
        "--index-row", type=int, required=True, help="the client's row of the round index"
    )
    parser.add_argument("--blocks", type=int, default=24, help="blocks after latent 24")
    parser.add_argument("--taehv", help="taew2_2.pth, the tiny decoder (--decoder taehv)")
    parser.add_argument("--decoder", choices=("taehv", "wan"), default="taehv")
    parser.add_argument(
        "--decoder-device",
        help="where the decoder runs, e.g. cuda:1 (default: the client's device); the next block"
        " then starts while the frames of this one are decoded",
    )
    parser.add_argument(
        "--decoder-channels-last",
        action="store_true",
        help="the Wan2.2 VAE's decoding convolutions in the channels-last layout: faster, not the"
        " paper's pixels byte for byte (--decoder wan)",
    )
    parser.add_argument("--settings", nargs="+", choices=tuple(SETTINGS), default=tuple(SETTINGS))
    parser.add_argument(
        "--fps", type=float, default=PlayConfig.fps, help="frames shown per second (play.fps)"
    )
    parser.add_argument("--out-dir", required=True, help="receives bench.json")
    args = parser.parse_args(argv)
    if args.blocks < UNMEASURED_BLOCKS + 2:
        parser.error(
            f"--blocks: at least {UNMEASURED_BLOCKS + 2} (the first {UNMEASURED_BLOCKS} are not"
            " measured)"
        )

    if args.decoder == "taehv" and not args.taehv:
        parser.error(
            "--taehv: the tiny decoder's weights (taew2_2.pth) are needed with --decoder taehv"
        )

    with usage_errors(parser):
        ServingOptions.low_latency(
            args.taehv,
            decoder=args.decoder,
            decoder_device=args.decoder_device,
            decoder_channels_last=args.decoder_channels_last,
        )
        overrides = {**parse_overrides(args.overrides), "run.max_blocks": args.blocks}
        cfg = load_config(args.config, overrides)
        cfg.paths.require("round_index")
        row = load_round_index_row(cfg.paths.round_index, args.index_row)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result: dict = dict(
        index_row=args.index_row,
        blocks=args.blocks,
        display_fps=args.fps,
        decoder=args.decoder,
        decoder_device=args.decoder_device,
        decoder_channels_last=args.decoder_channels_last,
        attention=flash_attention_backend(),
    )
    measured = compare_settings(
        cfg,
        row,
        args.taehv,
        args.settings,
        fps=args.fps,
        decoder=args.decoder,
        decoder_device=args.decoder_device,
        decoder_channels_last=args.decoder_channels_last,
    )
    for name, entry in measured:
        result[name] = entry
        print(json.dumps({name: entry}), flush=True)
        (out / "bench.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
