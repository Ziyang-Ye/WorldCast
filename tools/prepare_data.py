#!/usr/bin/env python3
"""Build the files training reads, from the OpenCS2 dataset and the game's map files (docs/data.md).

One subcommand per artefact; ``--help`` after a subcommand lists its options.

Examples::

    python tools/prepare_data.py media-index --help
    python tools/prepare_data.py latent-cache --help
"""

import argparse
import importlib
import sys
from collections.abc import Sequence

from worldcast.config.loader import usage_errors

#: Subcommand -> the module of ``worldcast.data.preparation`` that builds it.
STEPS = {
    "media-index": "media_index",
    "manifest": "manifest",
    "observer-labels": "observer_labels",
    "latent-cache": "latent_cache",
    "buckets": "buckets",
    "visibility": "visibility",
    "meshes": "meshes",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the named step; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    steps = parser.add_subparsers(dest="step", required=True)
    for name, module in STEPS.items():
        step = importlib.import_module(f"worldcast.data.preparation.{module}")
        sub = steps.add_parser(
            name,
            help=step.__doc__.splitlines()[0],
            description=step.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        step.add_arguments(sub)
        # a step's error ends with the usage of its own parser (buckets: of its subcommand's)
        sub.set_defaults(run=step.run, parser=sub)
    args = parser.parse_args(argv)
    with usage_errors(args.parser):
        return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
