#!/usr/bin/env python3
"""Download the release weights and the parts of the Wan2.2-TI2V-5B snapshot a client needs.

From the WorldCast repository (docs/inference.md, "Weights"): the four-step generator (bf16), the
state model, the depth head and its read-out, and the fixed prompt's umT5 embedding, all
``.safetensors``, with the repository's ``config.json`` (its description). From
``Wan-AI/Wan2.2-TI2V-5B`` at the pinned revision: ``config.json`` (backbone dimensions),
``Wan2.2_VAE.pth`` (decode) and the umT5 tokenizer; with ``--with-t5`` also the 11 GB umT5-XXL
encoder (only to re-make the prompt embedding). The Wan2.2 backbone weights are not needed: the
WorldCast checkpoint replaces all of them.

Writes ``<out-dir>/paths.yaml``, the ``paths`` entries to pass as a ``--config``. A file the
repository does not hold ends the tool with its name. When the Hugging Face Hub cannot be reached,
the files already in ``--out-dir`` are taken as they are, and the tool ends naming every one that
is not there.

Example::

    python tools/download_weights.py --out-dir weights
    python tools/run_client.py --config weights/paths.yaml ...
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from worldcast.hub import download_weights

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Download the release weights into ``--out-dir``; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True, help="download directory")
    parser.add_argument("--with-t5", action="store_true", help="also the umT5-XXL encoder (11 GB)")
    parser.add_argument("--skip-wan22", action="store_true", help="not the Wan2.2 snapshot")
    args = parser.parse_args(argv)
    try:
        paths = download_weights(
            args.out_dir,
            ROOT / "configs" / "state_model",
            wan22=not args.skip_wan22,
            t5=args.with_t5,
        )
    except (
        ConnectionError,
        FileNotFoundError,
    ) as error:  # files that are not in --out-dir, or that
        parser.error(str(error))  # the repository does not hold: it names them
    print(f"{paths}:\n{paths.read_text()}", end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
