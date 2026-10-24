"""ffmpeg that loses part of what it is given: ``lossy_ffmpeg.py LOSS FFMPEG ARGUMENTS...`` runs
``FFMPEG ARGUMENTS...``. Where they write an mp4 (the last argument), LOSS is one of

- ``frames``: the frames it wrote, its ``mdat`` box, read back as zeros and its index, the
  ``moov`` box, whole;
- ``tail``: the frames from the 4 KiB page of the file at the middle of the mdat on;
- ``page``: that page alone;
- ``bytes``: :data:`BYTES` bytes of its largest frame, which still decodes, with errors: of the
  spans from the middle of the frame on, :data:`STEP` bytes apart, the first whose loss the
  decoder reports with an error of its own. Which losses inside a frame the decoder finds
  depends on the frame's bytes, which differ by x264 build and by the machine's cores: after
  other losses it conceals the frame without an error, or decodes it as if it were whole;
- ``input``: the first video it reads (``-i``) holds its first :data:`INPUT_SECONDS` only, as a
  client's ``video.mp4`` that is short.

The tests of the mp4 writer and of the examples' scripts name it as their ffmpeg (the
``lossy_ffmpeg`` fixture of ``tests/conftest.py``).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.tools.support import mp4_boxes

#: A page of the file, bytes.
PAGE = 4096
#: What ``input`` keeps of the first video it reads, s.
INPUT_SECONDS = 3
#: What ``bytes`` loses of a frame, bytes.
BYTES = 256
#: How far apart the spans start that ``bytes`` tries, bytes.
STEP = 64


def frame_sizes(ffmpeg: str, path: Path) -> list[int]:
    """The sizes of the frames of an mp4 of one video stream, bytes, in file order."""
    copy = [ffmpeg, "-v", "error", "-i", str(path), "-c", "copy", "-f", "framemd5"]
    listed = subprocess.run([*copy, "-"], capture_output=True, text=True, check=True).stdout
    return [int(line.split(",")[4]) for line in listed.splitlines() if not line.startswith("#")]


def reported_loss(ffmpeg: str, path: Path, mdat: int) -> tuple[int, int]:
    """Where the ``bytes`` loss of the mp4 ``path`` starts and ends, bytes into the file; the
    content of its ``mdat`` box starts at byte ``mdat``."""
    sizes = frame_sizes(ffmpeg, path)
    largest = sizes.index(max(sizes))
    frame_start = mdat + sum(sizes[:largest])  # the mdat holds the frames one after the other
    frame_end = frame_start + sizes[largest]
    data = path.read_bytes()
    with tempfile.TemporaryDirectory() as scratch:
        trial = Path(scratch) / path.name
        for at in range((frame_start + frame_end) // 2 - BYTES // 2, frame_end - BYTES + 1, STEP):
            trial.write_bytes(data[:at] + bytes(BYTES) + data[at + BYTES :])
            decode = [ffmpeg, "-v", "error", "-xerror", "-i", str(trial), "-f", "null", "-"]
            errors = subprocess.run(decode, capture_output=True, text=True).stderr
            # the decoder's own error, not ffmpeg's word on a frame the decoder concealed
            if errors.startswith("[h264 @ "):
                return at, at + BYTES
    sys.exit(f"the decoder reports no loss of {BYTES} bytes in the largest frame of {path}")


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
    if loss == "bytes":
        lost["bytes"] = reported_loss(ffmpeg, out, start)
    first_byte, end_byte = lost[loss]
    with open(out, "r+b") as video:
        video.seek(first_byte)
        video.write(bytes(end_byte - first_byte))
sys.exit(status)
