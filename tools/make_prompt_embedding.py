#!/usr/bin/env python3
"""Encode the fixed prompt once with umT5-XXL and save it, so clients never build the 11 GB text encoder.

Run this once, on a GPU, in bf16: the paper client's embedding was the bf16 umT5 forward on CUDA, and CPU bf16
GEMMs are not bit-equal to it. The release ships the result (``fixed_prompt_umt5xxl_bf16.safetensors``); this script
is how it was made, and how to make it for another prompt.

Needs the ``text`` extra (``pip install -e ".[text]"``) and the umT5-XXL encoder of the Wan2.2 snapshot
(``python tools/download_weights.py --out-dir weights --with-t5``).

Example::

    python tools/make_prompt_embedding.py --wan22-root weights/Wan2.2-TI2V-5B \\
        --out weights/fixed_prompt_umt5xxl_bf16.safetensors
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run from a checkout without installing the package
    sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    from worldcast.modeling.wan22.text_encoder import FIXED_PROMPT

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--wan22-root",
        required=True,
        help="Wan2.2-TI2V-5B snapshot (models_t5_umt5-xxl-enc-bf16.pth, google/umt5-xxl/)",
    )
    parser.add_argument("--out", required=True, help="output .safetensors")
    parser.add_argument(
        "--prompt", default=FIXED_PROMPT, help=f"default: {FIXED_PROMPT!r} (the paper's)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device (default: cuda; the paper's embedding was made on a GPU)",
    )
    args = parser.parse_args(argv)

    import torch

    from worldcast.modeling.wan22.text_encoder import TextEncoder

    encoder = TextEncoder.from_pretrained(args.wan22_root, device=args.device, dtype=torch.bfloat16)
    embedding = encoder.encode_prompt(args.prompt)
    embedding.save(args.out)
    print(
        f"{args.prompt!r}: {embedding.seq_len} tokens, embeds {tuple(embedding.embeds.shape)} "
        f"{embedding.embeds.dtype} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
