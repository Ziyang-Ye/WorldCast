"""The page's shell (``static/index.html``) as the coordinator serves it: the viewport and the
colours of a phone's browser, the manifest that lets a phone open the page full screen and the
icons it names. No browser runs here (``test_environments.py`` runs the shell's notices in
Chrome)."""

import asyncio
import io
import json
import re

from PIL import Image

from tests.demo.support import STATIC, demo


async def fetch(path: str) -> tuple[int, str, bytes]:
    async with demo(1) as (local, http):
        response = await http.get(f"{local.url}{path}")
        return response.status, response.headers.get("Content-Type", ""), await response.read()


def shell() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def test_the_shell_sizes_itself_to_the_device_and_asks_for_no_zoom_lock():
    head = shell()
    viewport = re.search(r'<meta name="viewport" content="([^"]+)"', head).group(1)
    assert "width=device-width" in viewport and "initial-scale=1" in viewport
    assert "maximum-scale" not in viewport and "user-scalable" not in viewport
    assert '<meta name="color-scheme" content="dark">' in head
    assert '<meta name="theme-color" content="#000000">' in head
    assert '<meta name="format-detection" content="telephone=no">' in head
    assert "<noscript>" in head and 'id="unsupported" hidden' in head


def test_the_manifest_names_icons_that_the_coordinator_serves():
    manifest = re.search(r'<link rel="manifest" href="([^"]+)"', shell()).group(1)
    status, kind, body = asyncio.run(fetch(manifest))
    assert status == 200 and kind.startswith("application/manifest+json")
    data = json.loads(body)
    assert data["start_url"] == "/" and data["display"] == "standalone"
    assert data["background_color"] == data["theme_color"] == "#000000"
    kinds = {}
    for icon in data["icons"]:
        status, kind, body = asyncio.run(fetch(icon["src"]))
        assert status == 200 and kind.startswith(icon["type"]), icon
        kinds[icon["sizes"]] = body
    assert set(kinds) == {"any", "192x192", "512x512"}
    for size in (192, 512):
        assert Image.open(io.BytesIO(kinds[f"{size}x{size}"])).size == (size, size)


def test_the_shell_offers_the_phones_home_screen_an_icon_and_a_title():
    head = shell()
    touch = re.search(r'<link rel="apple-touch-icon" href="([^"]+)"', head).group(1)
    status, kind, body = asyncio.run(fetch(touch))
    assert status == 200 and kind == "image/png"
    assert Image.open(io.BytesIO(body)).size == (180, 180)
    assert '<meta name="apple-mobile-web-app-capable" content="yes">' in head
    assert '<meta name="mobile-web-app-capable" content="yes">' in head
    assert '<meta name="apple-mobile-web-app-title" content="WorldCast">' in head
