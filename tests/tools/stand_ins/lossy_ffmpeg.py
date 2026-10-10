"""ffmpeg that loses part of what it is given: ``lossy_ffmpeg.py LOSS FFMPEG ARGUMENTS...`` runs
``FFMPEG ARGUMENTS...``. Where they write an mp4 (the last argument), LOSS is one of

- ``frames``: the frames it wrote, its ``mdat`` box, read back as zeros and its index, the
  ``moov`` box, whole;
- ``tail``: the frames from the 4 KiB page of the file at the middle of the mdat on;
- ``page``: that page alone;
- ``bytes``: :data:`BYTES` bytes in the middle of its largest frame, which still decodes, with
  errors;
- ``input``: the first video it reads (``-i``) holds its first :data:`INPUT_SECONDS` only, as a
  client's ``video.mp4`` that is short.

The tests of the mp4 writer and of the examples' scripts name it as their ffmpeg (the
``lossy_ffmpeg`` fixture of ``tests/conftest.py``).
"""

import os
import subprocess
import sys
from pathlib import Path

from tests.tools.support import mp4_boxes

#: A page of the file, bytes.
PAGE = 4096
#: What ``input`` keeps of the first video it reads, s.
INPUT_SECONDS = 3
#: What ``bytes`` loses of a frame, bytes.
BYTES = 256


def frame_sizes(ffmpeg: str, path: Path) -> list[int]:
    """The sizes of the frames of an mp4 of one video stream, bytes, in file order."""
    copy = [ffmpeg, "-v", "error", "-i", str(path), "-c", "copy", "-f", "framemd5"]
    listed = subprocess.run([*copy, "-"], capture_output=True, text=True, check=True).stdout
    return [int(line.split(",")[4]) for line in listed.splitlines() if not line.startswith("#")]


loss, ffmpeg, *arguments = sys.argv[1:]
out = Path(arguments[-1]) if arguments[-1].endswith(".mp4") else None
if out and loss == "input":
    first = Path(arguments[arguments.index("-i") + 1])
    cut = first.with_name(f".cut{first.suffix}")
    keep = ["-t", str(INPUT_SECONDS), "-i", str(first), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    subprocess.run([ffmpeg, "-v", "error", "-y", *keep, str(cut)], check=True)
    os.replace(cut, first)
status = subprocess.call([ffmpeg, *arguments])
if status == 0 and out and loss != "input":
    ((start, end),) = [(start, end) for kind, start, end in mp4_boxes(out) if kind == "mdat"]
    page = max(start, (start + end) // 2 // PAGE * PAGE)
    lost = {"frames": (start, end), "tail": (page, end), "page": (page, min(page + PAGE, end))}
    if loss == "bytes":  # the mdat holds the frames one after the other
        sizes = frame_sizes(ffmpeg, out)
        largest = sizes.index(max(sizes))
        middle = start + sum(sizes[:largest]) + sizes[largest] // 2
        lost["bytes"] = (middle - BYTES // 2, middle + BYTES // 2)
    first_byte, end_byte = lost[loss]
    with open(out, "r+b") as video:
        video.seek(first_byte)
        video.write(bytes(end_byte - first_byte))
sys.exit(status)
