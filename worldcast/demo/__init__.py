"""WorldCast Live, the networked multiplayer web demo: coordinator, GPU workers, browser UI
(docs/demo.md).

The names below are imported on first use (PEP 562): the coordinator and the mock client load no
torch.
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "clients": (
            "ClientMessage",
            "ClientProtocol",
            "LiveControls",
            "LiveFrame",
            "LivePlayerState",
            "make_client",
        ),
        "config": ("DemoConfig", "load_config"),
        "coordinator": ("Coordinator",),
        "library": ("Library", "RoundStart", "Seat", "synthetic_library"),
        "local": ("LocalDemo",),
        "mock": ("MockClient",),
        "pacing": ("Pacing",),
        "worker": ("Worker",),
        "worldcast_client": ("WorldCastClient",),
    },
)
