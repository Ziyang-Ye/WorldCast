#!/usr/bin/env python3
"""Download the data and the expected videos of the example cases, and write each case's config.

``examples/manifest.json`` lists each file with its size and sha256; a file already in place with
that digest is kept. The files come from the WorldCast repository on Hugging Face, or without the
Hub from a local copy of it (``--source``), and go to ``<out-dir>/data/<case>/`` and
``<out-dir>/expected/<case>.mp4``; ``<out-dir>/data/<case>/config.yaml`` names the case's paths
and its length, for ``--config``. Without the Hub, a file that is neither in place nor in the
Hub's cache ends the tool with its name.

Examples::

    python tools/download_examples.py --out-dir examples                         # every case
    python tools/download_examples.py --out-dir examples dust2_r09 mirage_r16    # some cases
    python tools/download_examples.py --out-dir examples --source /path/to/copy  # from a local copy
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from worldcast.config.loader import usage_errors
from worldcast.hub import EXAMPLES_MANIFEST, download_example, example_cases, write_example_config

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch the named example cases into ``--out-dir``; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cases", nargs="*", help="case names (default: every case)")
    parser.add_argument("--out-dir", required=True, help="receives data/<case>/ and expected/")
    parser.add_argument("--source", help="a local copy of the repository on Hugging Face")
    args = parser.parse_args(argv)
    manifest = ROOT / EXAMPLES_MANIFEST
    with usage_errors(parser):
        cases = example_cases(manifest, args.cases)
    for name, case in cases.items():
        try:
            fetched = download_example(case, args.out_dir, args.source)
        except ConnectionError as error:
            parser.error(f"{error}; pass --source <a local copy of the repository>")
        except FileNotFoundError as error:  # not in the repository, or not in --source
            parser.error(str(error))
        config = write_example_config(name, case, args.out_dir, manifest)
        size = sum(file["bytes"] for file in case["files"].values()) / 1e6
        print(f"{name}: {len(case['files'])} files, {size:.1f} MB ({fetched} fetched) -> {config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
