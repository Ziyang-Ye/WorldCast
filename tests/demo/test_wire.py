"""The wire formats: a binary frame packs and unpacks; a socket that is gone drops the message; a
frame encodes as JPEG (torchvision's encoder when installed, as it releases the GIL; Pillow
otherwise)."""

import asyncio
import io

import numpy as np
import pytest
from PIL import Image

from worldcast.demo import wire
from worldcast.demo.wire import pack, send, unpack


def test_a_binary_frame_round_trips():
    header, payload = {"kind": "scene", "slot": 3, "meta": {"frames": [0, 16]}}, bytes(range(200))
    assert unpack(pack(header, payload)) == (header, payload)
    assert unpack(pack(header)) == (header, b"")


def test_send_picks_the_frame_type_and_drops_on_a_gone_socket():
    class Socket:
        def __init__(self, closed=False, broken=False):
            self.closed, self.broken, self.sent = closed, broken, []

        async def send_json(self, message):
            if self.broken:
                raise ConnectionResetError
            self.sent.append(("json", message))

        async def send_bytes(self, data):
            self.sent.append(("bytes", data))

    async def scenario():
        live, closed, broken = Socket(), Socket(closed=True), Socket(broken=True)
        for ws in (live, closed, broken, None):
            await send(ws, {"t": "roster"})
        await send(live, b"frame")
        return live.sent, closed.sent, broken.sent

    live, closed, broken = asyncio.run(scenario())
    assert live == [("json", {"t": "roster"}), ("bytes", b"frame")] and closed == broken == []


def _frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:96, 0:160]
    rgb = np.stack([1.5 * x, 2.5 * y, x + y], -1) + rng.normal(0.0, 1.0, (96, 160, 3))
    return rgb.clip(0, 255).astype(np.uint8)


def _decoded(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as image:
        assert image.format == "JPEG" and image.size == (160, 96)
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def test_torchvision_encodes_when_installed(monkeypatch):
    pytest.importorskip("torchvision")
    encode, calls = wire._torchvision_jpeg(), []

    def spy(chw, quality):
        calls.append((tuple(chw.shape), quality))
        return encode(chw, quality=quality)

    monkeypatch.setattr(wire, "_torchvision_jpeg", lambda: spy)
    rgb = _frame()
    assert np.abs(_decoded(wire.encode_jpeg(rgb, 85)) - rgb).mean() < 3.0
    assert calls == [((3, 96, 160), 85)]


@pytest.mark.parametrize("case", ["no torchvision", "read-only frame"])
def test_pillow_encodes_otherwise(monkeypatch, case):
    rgb = _frame()
    if case == "no torchvision":
        monkeypatch.setattr(wire, "_torchvision_jpeg", lambda: None)
    else:  # the mock client's frames; torch.from_numpy would warn on them
        monkeypatch.setattr(wire, "_torchvision_jpeg", lambda: pytest.fail("torchvision used"))
        rgb.flags.writeable = False
    assert np.abs(_decoded(wire.encode_jpeg(rgb, 85)) - rgb).mean() < 3.0
