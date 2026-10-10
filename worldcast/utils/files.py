"""Files: content digests."""

import hashlib
import os

__all__ = ["sha256_file"]


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
