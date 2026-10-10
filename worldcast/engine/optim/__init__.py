"""The optimizer: parameter groups, AdamW, the per-group clip and the EMA.

The names below are imported on first use (PEP 562).
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "builder": (
            "BETAS",
            "GROUPS",
            "MAX_GRAD_NORM",
            "MODULE_LRS",
            "WEIGHT_DECAY",
            "build_optimizer",
            "clip_per_group_",
            "group_grad_norms",
            "parameter_group",
        ),
        "ema": ("ShardedEMA",),
    },
)
