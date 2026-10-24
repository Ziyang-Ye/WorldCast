"""The demo's wire formats (docs/demo.md, "Protocol").

Control messages are JSON text frames ``{"t": <type>, ...}``; payloads (video frames, the clients'
messages) are binary frames: a 4-byte big-endian header length, a JSON header, the payload bytes.
Video frames are JPEG.
"""

import functools
import io
import json
import struct
from collections.abc import Callable

import numpy as np
from aiohttp import ClientWebSocketResponse, web

__all__ = [
    "HEARTBEAT_S",
    "LINK_MESSAGE_BYTES",
    "WebSocket",
    "encode_jpeg",
    "load_jpeg_encoder",
    "pack",
    "send",
    "unpack",
]

#: Largest message between a worker and the coordinator, bytes (a memory entry is 774 kB).
LINK_MESSAGE_BYTES = 64 << 20
#: Seconds between two pings of a worker's sockets: its link to the coordinator, its browsers.
HEARTBEAT_S = 10.0
#: A WebSocket from either end: the one a server accepted, the one a client opened.
WebSocket = web.WebSocketResponse | ClientWebSocketResponse
_LENGTH = struct.Struct(">I")


def pack(header: dict, payload: bytes = b"") -> bytes:
    """A binary frame: ``len(header) | header JSON | payload``."""
    head = json.dumps(header, separators=(",", ":")).encode()
    return _LENGTH.pack(len(head)) + head + payload


def unpack(data: bytes) -> tuple[dict, bytes]:
    """The header and the payload of a binary frame."""
    (n,) = _LENGTH.unpack_from(data)
    start = _LENGTH.size
    return json.loads(data[start : start + n]), data[start + n :]


async def send(ws: WebSocket | None, message: dict | bytes) -> None:
    """Send a JSON message or a binary frame; a socket that is gone (``None``, closed, reset, lost)
    drops it."""
    if ws is None or ws.closed:
        return
    try:
        if isinstance(message, bytes):
            await ws.send_bytes(message)
        else:
            await ws.send_json(message)
    except (ConnectionError, RuntimeError):
        pass


@functools.cache
def _torchvision_jpeg() -> Callable | None:
    """torchvision's JPEG encoder (libjpeg-turbo, without the GIL), or None without torchvision."""
    try:
        from torchvision.io import encode_jpeg as encode
    except ImportError:
        return None
    return encode


def load_jpeg_encoder() -> None:
    """Load the JPEG encoder now, not with a session's first frame (importing torchvision takes
    seconds)."""
    encode_jpeg(np.zeros((16, 16, 3), np.uint8), 85)


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """``[H, W, 3]`` uint8 -> JPEG bytes (4:2:0).

    torchvision's encoder releases the GIL; Pillow's, the fallback, holds it for the whole frame and
    so stalls the client's generation thread, which prepares the next block meanwhile.
    """
    # torch.from_numpy warns on a read-only array (the mock client's frames)
    encode = _torchvision_jpeg() if rgb.flags.writeable else None
    if encode is not None:
        import torch

        chw = torch.from_numpy(rgb).permute(2, 0, 1)
        return encode(chw, quality=int(quality)).numpy().tobytes()
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()
