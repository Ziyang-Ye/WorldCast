"""Distributed training: the process group, the RNG states of every rank and FSDP.

The names below are imported on first use (PEP 562).
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "fsdp": (
            "MIN_UNIT_PARAMS",
            "clip_grad_norm",
            "fsdp_wrap",
            "full_state_dict",
            "group_all_reduce",
            "is_fsdp",
            "live_module",
            "no_sync",
            "sharded_parameters",
        ),
        "process_group": (
            "DistInfo",
            "barrier",
            "capture_rng_state",
            "destroy_distributed",
            "init_distributed",
            "restore_rng_state",
            "use_deterministic_algorithms",
        ),
    },
)
