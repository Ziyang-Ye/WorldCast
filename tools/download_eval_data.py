#!/usr/bin/env python3
"""Download the evaluation data of the paper's evaluation sets and write each set's config.

The data is the paper's own: the latents of the windows the sets read, the GT visibility, flash and
scope labels of their clients or views, the tick tables of every player of their rounds and their
rows of the media index, from the dataset ``ZiyangYe/WorldCast-data`` on Hugging Face (derived
from OpenCS2, CC BY 4.0). Its manifest, checked by its sha256, gives every file's sha256; a file
already in place with that digest is kept. The sets:

- ``validation``: the 64 windows of ``configs/data/validation_index.jsonl`` (``tools/evaluate.py``);
- ``windows``: the 512 views of the 256 ten-second windows of Tables 1 and 2
  (``configs/eval/window_views.jsonl``);
- ``rounds``: the 96 clients of the 32 rounds of Table 3 (``configs/eval/round_index.jsonl``).

The files go to ``<out-dir>/eval/`` and ``<out-dir>/eval/<set>.yaml`` names their paths, for
``--config``: ``data`` and ``validation.index`` for ``validation``, ``paths`` for the others.

Examples::

    python tools/download_eval_data.py --out-dir data/evaluation rounds
    python tools/run_session.py --config weights/paths.yaml \\
        --config data/evaluation/eval/rounds.yaml --all-rounds --round-length --gpus 0,1,2 \\
        --out-dir runs/table3
    python tools/download_eval_data.py --out-dir data/evaluation validation windows
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from worldcast.hub import EVAL_SETS, download_eval_data, write_eval_data_config

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch the named sets into ``--out-dir``; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sets", nargs="+", choices=list(EVAL_SETS), help="the evaluation sets")
    parser.add_argument("--out-dir", required=True, help="receives eval/")
    parser.add_argument("--source", help="a local copy of the dataset on Hugging Face")
    args = parser.parse_args(argv)
    for name in dict.fromkeys(args.sets):
        try:
            fetched = download_eval_data(name, args.out_dir, ROOT, args.source)
        except ConnectionError as error:
            parser.error(f"{error}; pass --source <a local copy of the dataset>")
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        config = write_eval_data_config(name, args.out_dir, ROOT)
        print(f"{name}: {fetched} files fetched -> {config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
