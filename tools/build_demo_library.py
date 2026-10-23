#!/usr/bin/env python3
"""Build the library of round starts the demo's rooms begin from (docs/demo.md, "Library").

The manifest names, per round start, the map, the start frame and each seat's rendered video; its
format is in ``worldcast/demo/build_library.py``. A seat of a recorded round is offered only if the
GT visibility labels show no other player in its first frame (``visibility_label_root`` of the
manifest), and a seat whose loadout holds a weapon that zooms is left out (live play does not
track the scope); a round start none of whose seats qualifies is left out. Each round start logs
how many of its seats it offers and why a seat was left out. A manifest the builder refuses, or
one that leaves no round start, ends with a message and status 2.

Example::

    python tools/build_demo_library.py --manifest library.yaml --out-dir demo-library
    python tools/serve_demo.py --role local --set library=demo-library/library.json
"""

import argparse
import logging
import sys
from collections.abc import Sequence

from worldcast.demo.build_library import build_library


def main(argv: Sequence[str] | None = None) -> int:
    """Build the library of ``--manifest`` into ``--out-dir``; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, help="the manifest YAML")
    parser.add_argument("--out-dir", required=True, help="receives library.json and its files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        library = build_library(args.manifest, args.out_dir)
    except ValueError as error:
        parser.error(str(error))
    print(library)
    return 0


if __name__ == "__main__":
    sys.exit(main())
