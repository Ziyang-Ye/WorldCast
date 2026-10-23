"""Files: content digests, and writes that leave a whole file or none."""

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

__all__ = ["atomic_write", "json_line", "sha256_file", "write_jsonl"]


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: str | os.PathLike[str], write: Callable[[BinaryIO], Any]) -> None:
    """Write a file through a temporary file in its directory, renamed into place, so that a file
    at ``path`` is always whole; the temporary file is removed if ``write`` raises.

    Args:
        path (str | os.PathLike[str]): the file; its directory is created.
        write (Callable[[BinaryIO], Any]): writes the file's bytes to the open handle it is given.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            write(handle)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_line(row: Mapping[str, Any]) -> str:
    """One JSONL line: sorted keys, compact separators, ASCII, no NaN."""
    return (
        json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    )


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Write ``rows`` as JSON lines (:func:`json_line`) with :func:`atomic_write`."""
    atomic_write(path, lambda handle: handle.writelines(json_line(row).encode() for row in rows))
