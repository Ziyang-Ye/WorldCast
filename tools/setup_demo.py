#!/usr/bin/env python3
"""Get a machine ready to serve WorldCast Live and print the command that serves it (docs/demo.md,
"Quick start").

Downloads the release weights and the tiny decoder (``tools/download_weights.py``), checks that the
library of round starts and its overviews load as the coordinator reads them and that the data paths
file is in place, and prints the one command that serves the demo on the cards you name. A GPU
worker needs a library: its warm-up replays the library's first seat. ``--check-only`` downloads
nothing and only checks.

Example::

    python tools/setup_demo.py --library demo-library/library.json --gpus 0,1
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "demo" / "worldcast.yaml"
#: Where ``configs/demo/worldcast.yaml`` reads the weights and the data paths, from the repository.
WEIGHTS, DATA_PATHS = "weights", "data/paths.yaml"


def check_library(path: str | Path) -> str:
    """A line on the library of ``path``: its round starts per map and the maps with overview
    images, each overview checked as the coordinator reads it.

    Raises:
        ValueError: the library or an overview does not load, or the library has no round start.
        KeyError: an overview lacks a field.
        OSError: a file cannot be read.
    """
    from worldcast.demo.library import Library
    from worldcast.demo.map_picker import MapPicker

    library = Library.load(str(path))
    picker = MapPicker.load(library)
    per_map: dict[str, int] = {}
    for round_start in library.round_starts.values():
        per_map[round_start.map] = per_map.get(round_start.map, 0) + 1
    drawn = sorted(
        name
        for name, seats in picker.maps.items()
        if any(floor.image for floor in seats.overview.floors)
    )
    maps = ", ".join(f"{name} {count}" for name, count in sorted(per_map.items()))
    pictures = f"overviews for {', '.join(drawn)}" if drawn else "no overview images (plain panels)"
    return f"{len(library.round_starts)} round starts ({maps}); {pictures}"


def serve_command(library: str, gpus: str) -> str:
    """The command that serves the demo from the repository root."""
    return (
        f"python tools/serve_demo.py --role all --gpus {gpus} --config {CONFIG.relative_to(ROOT)}"
        f" \\\n    --set library={library}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare the demo's files and print how to serve it; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--library", required=True, help="the demo library's library.json")
    parser.add_argument("--gpus", default="0", help="the cards to serve on: 0,1,2")
    parser.add_argument("--check-only", action="store_true", help="download nothing")
    args = parser.parse_args(argv)

    try:
        print(f"library: {check_library(args.library)}")
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"the library {args.library} does not load: {error}")
    if not (ROOT / DATA_PATHS).is_file():
        parser.error(
            f"{DATA_PATHS} is missing: it names the dataset files the closed loop reads"
            " (docs/inference.md, 'Data')"
        )
    if not args.check_only:
        from worldcast.hub import download_weights

        try:
            paths = download_weights(ROOT / WEIGHTS, ROOT / "configs" / "state_model")
        except (ConnectionError, FileNotFoundError) as error:
            parser.error(str(error))
        print(f"weights: {paths}")
    print("\nServe the demo, then open http://localhost:8100:\n")
    print(serve_command(args.library, args.gpus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
