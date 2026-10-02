"""Low-latency serving of one WorldCast client: a resident engine that streams frames block by block.

``Engine`` (:mod:`worldcast.engine.realtime.engine`) keeps the models loaded and turns one block of controls into decoded
frames, which it yields as each latent frame is decoded. Its defaults reproduce the paper client bit for bit; every
latency option is opt-in (:class:`~worldcast.engine.realtime.config.RealtimeConfig`). docs/latency.md has the latency
budget and the recommended settings.
"""

from .config import RealtimeConfig

__all__ = [
    "RealtimeConfig",
    "Engine",
    "EngineModels",
    "RoundSpec",
    "Frame",
    "BlockControls",
    "LiveControls",
]

_LAZY = {
    "Engine": "engine",
    "EngineModels": "engine",
    "RoundSpec": "engine",
    "Frame": "engine",
    "BlockControls": "controls",
    "LiveControls": "controls",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(f"worldcast.engine.realtime.{_LAZY[name]}"), name)
    raise AttributeError(f"module 'worldcast.engine.realtime' has no attribute {name!r}")
