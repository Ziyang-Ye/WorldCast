#!/usr/bin/env python3
"""Download the release weights and the parts of the Wan2.2-TI2V-5B snapshot a client needs.

From the WorldCast repository (docs/inference.md, "Weights"): the four-step generator (bf16), the
state model, the depth head and its read-out, and the fixed prompt's umT5 embedding, all
``.safetensors``, with the repository's ``config.json`` (its description). From
``Wan-AI/Wan2.2-TI2V-5B`` at the pinned revision: ``config.json`` (backbone
dimensions), ``Wan2.2_VAE.pth`` (decode) and the umT5 tokenizer; with ``--with-t5`` also the 11 GB
umT5-XXL encoder (only to re-make the prompt embedding). The Wan2.2 backbone weights are not
needed: the WorldCast checkpoint replaces all of them. From ``madebyollin/taehv`` (MIT) at a pinned
commit: ``taew2_2.pth``, the tiny decoder of the web demo, checked by sha256.

``--with-training-checkpoints`` also fetches the EMA weights of stages 2, 2s and 3 (fp32, 20 GB
each), which training starts from (docs/training.md): stage 3 from stage 2s, stage 4 from stage 3
with stage 2 (Table 1's WorldCast) as teacher and critic. Inference does not read them.

``--with-metrics`` also fetches the weights of the paper's metrics (``tools/evaluate_paper.py``;
docs/training.md, "The paper's metrics") into ``<out-dir>/metrics/``, from where each is published,
checked by its sha256 (``worldcast.hub.METRIC_WEIGHTS``): ``detector``, the player detector of
Rendered and Unmatched (in the WorldCast repository, AGPL-3.0), ``gim_dkm``, GIM's dense matcher
(843 MB, from its authors), ``loftr``, LoFTR's outdoor weights (where kornia fetches them), and
``i3d``, the I3D detector of FVD; alone it fetches all four, or the comma-separated ones it names.
Each sets its ``paths`` key once it is in place.

Writes ``<out-dir>/paths.yaml``, the ``paths`` entries to pass as a ``--config``, before it fetches
the tiny decoder, the training checkpoints and the metrics' weights. A file the repository does not
hold ends the tool with its name. When the Hugging Face Hub cannot be reached, the files already in
``--out-dir`` are taken as they are, and the tool ends naming every one that is not there; so does
a tiny decoder that cannot be fetched (``--skip-taehv`` goes without it).

Example::

    python tools/download_weights.py --out-dir weights
    python tools/run_client.py --config weights/paths.yaml ...
    python tools/download_weights.py --out-dir weights --with-metrics i3d,loftr
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from worldcast.hub import METRIC_WEIGHTS, download_weights

ROOT = Path(__file__).resolve().parents[1]


def _metric_keys(text: str) -> list[str]:
    keys = [key.strip() for key in text.split(",") if key.strip()]
    unknown = [key for key in keys if key not in METRIC_WEIGHTS]
    if not keys or unknown:
        raise argparse.ArgumentTypeError(
            f"{text!r}: name some of {', '.join(METRIC_WEIGHTS)}, separated by commas"
        )
    return keys


def main(argv: Sequence[str] | None = None) -> int:
    """Download the release weights into ``--out-dir``; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True, help="download directory")
    parser.add_argument("--with-t5", action="store_true", help="also the umT5-XXL encoder (11 GB)")
    parser.add_argument("--skip-wan22", action="store_true", help="not the Wan2.2 snapshot")
    parser.add_argument("--skip-taehv", action="store_true", help="not the tiny decoder")
    parser.add_argument(
        "--with-training-checkpoints",
        action="store_true",
        help="also the stage-2, 2s and 3 checkpoints training starts from (fp32, 20 GB each)",
    )
    parser.add_argument(
        "--with-metrics",
        nargs="?",
        const=list(METRIC_WEIGHTS),
        default=[],
        type=_metric_keys,
        metavar="KEYS",
        help=f"also weights of the paper's metrics: {','.join(METRIC_WEIGHTS)} (default: all)",
    )
    args = parser.parse_args(argv)
    try:
        paths = download_weights(
            args.out_dir,
            ROOT / "configs" / "state_model",
            wan22=not args.skip_wan22,
            t5=args.with_t5,
            taehv=not args.skip_taehv,
            training_checkpoints=args.with_training_checkpoints,
            metrics=args.with_metrics,
        )
    except ConnectionError as error:  # and files are not in --out-dir yet: it names them
        parser.error(str(error))
    except FileNotFoundError as error:  # a file a repository does not hold
        parser.error(f"{error} ({Path(args.out_dir) / 'paths.yaml'} names what was fetched)")
    print(f"{paths}:\n{paths.read_text()}", end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
