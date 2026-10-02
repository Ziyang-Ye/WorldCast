#!/usr/bin/env python3
"""Download the data and the expected videos of the example cases (examples/README.md) into ``examples/``.

The files are under ``examples/`` in the WorldCast release repository (config ``weights.hf_repo_id``, as for
``tools/download_weights.py``). ``examples/manifest.json`` lists each file with its size and sha256; a file already
in place with the right digest is skipped.

Examples::

    python tools/download_examples.py                         # every case
    python tools/download_examples.py dust2_r09 mirage_r16    # some cases
    python tools/download_examples.py --source /path/to/copy  # from a local copy of the repository
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run from a checkout without installing the package
    sys.path.insert(0, str(REPO))

from worldcast.config.loader import add_config_args, load_cli_config, parse_overrides  # noqa: E402

MANIFEST = REPO / "examples" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "cases", nargs="*", help="case names (default: every case of examples/manifest.json)"
    )
    parser.add_argument(
        "--source", help="copy from this directory (holding examples/) instead of downloading"
    )
    add_config_args(parser)
    args = parser.parse_args(argv)

    cases = json.loads(MANIFEST.read_text())["cases"]
    unknown = [c for c in args.cases if c not in cases]
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; known: {', '.join(cases)}")
    todo = args.cases or list(cases)

    if args.source:
        source = Path(args.source)

        def fetch(name: str) -> Path:
            return source / name

    else:
        w = load_cli_config(args.config, parse_overrides(args.overrides)).weights
        from huggingface_hub import hf_hub_download

        def fetch(name: str) -> Path:
            return Path(hf_hub_download(w.hf_repo_id, name, revision=w.hf_revision))

    for case in todo:
        files = cases[case]["files"]
        fetched = 0
        for name, meta in files.items():
            target = REPO / name
            if (
                target.is_file()
                and target.stat().st_size == meta["bytes"]
                and sha256(target) == meta["sha256"]
            ):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fetch(name), target)
            if sha256(target) != meta["sha256"]:
                target.unlink()
                raise SystemExit(f"{name}: sha256 differs from examples/manifest.json")
            fetched += 1
        size = sum(meta["bytes"] for meta in files.values()) / 1e6
        print(
            f"{case}: {len(files)} files, {size:.1f} MB ({fetched} fetched, {len(files) - fetched}"
            " already there)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
