"""Scene state: memory entries, the memory bank and retrieval (Sec. 3.3).

:class:`MemoryBank` works on cameras and depth maps alone, in numpy on the CPU (``publish``,
``withdraw``, ``retrieve``); :class:`SceneState` is one client's bank, fed with its own and the
other clients' blocks and a depth head.

The names below are imported on first use (PEP 562).
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "bank": ("MemoryBank", "MemoryEntry", "RetrieveResult"),
        "state": ("CopyMismatchError", "FollowStats", "SceneState", "StepSource"),
    },
)
