#!/usr/bin/env python3
"""Check the inputs of example cases on the CPU, without weights.

Loads every client's inputs as a client does (round and media index, the tick tables of all ten
players, the visibility and observer-signal labels, the first frame) and checks that the case is
one round whose clients all cover its length, and that each first frame is the reference run's
(``examples/manifest.json``). After a GPU run, ``tools/verify_reference.py check`` compares its
latents with the reference run's.

Examples::

    python tools/check_examples.py examples/data/*/config.yaml         # every downloaded case
    python tools/check_examples.py examples/data/dust2_r09/config.yaml
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from worldcast.config.inference import load_config
from worldcast.config.loader import usage_errors
from worldcast.engine.inference.reference import check_inputs, read_reference_runs
from worldcast.hub import EXAMPLES_MANIFEST

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Check the cases of the given configs; returns 1 if a case fails a check, else 0."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "configs", nargs="+", help="the cases' config.yaml (tools/download_examples.py writes them)"
    )
    args = parser.parse_args(argv)

    runs = read_reference_runs(ROOT / EXAMPLES_MANIFEST)
    failed = 0
    for config in args.configs:
        with usage_errors(parser):
            cfg = load_config(config)
        print(config)
        try:
            lines = check_inputs(cfg, runs)
        except (OSError, ValueError) as error:  # a missing file, a failed check
            lines, failed = [f"FAILED: {error}"], failed + 1
        except ModuleNotFoundError as error:  # pyarrow, the tick tables' reader
            parser.error(f"{error}: install the package with its dependencies, pip install -e .")
        for line in lines:
            print(f"  {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
