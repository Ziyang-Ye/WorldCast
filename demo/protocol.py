"""Wire formats shared by the coordinator, the workers and the browser (docs/demo.md, "Protocol").

Control messages are JSON text frames ``{"t": <type>, ...}``. Payloads (video frames, scene-state blocks) are binary
frames: a 4-byte big-endian header length, a JSON header, then the payload bytes.
"""

import io
import json
import secrets
import struct
from typing import Any

import numpy as np
from PIL import Image

_LEN = struct.Struct(">I")


def pack(header: dict[str, Any], payload: bytes = b"") -> bytes:
    """A binary frame: ``len(header) | header JSON | payload``."""
    head = json.dumps(header, separators=(",", ":")).encode()
    return _LEN.pack(len(head)) + head + payload


def unpack(data: bytes) -> tuple[dict[str, Any], bytes]:
    (n,) = _LEN.unpack_from(data)
    return json.loads(data[4 : 4 + n]), data[4 + n :]


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """``[H, W, 3]`` uint8 -> JPEG bytes (4:2:0)."""
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def new_token() -> str:
    return secrets.token_urlsafe(18)
