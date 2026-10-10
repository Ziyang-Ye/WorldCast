"""Training checkpoints: saving and resuming a run.

The names below are imported on first use (PEP 562).
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "training": (
            "CRITIC_FILE",
            "CRITIC_KEY",
            "EMA_FILE",
            "MODEL_FILE",
            "checkpoint_dir",
            "find_latest_checkpoint",
            "generator_state",
            "is_resumable",
            "load_rank_state",
            "load_wan22_backbone",
            "prune_checkpoints",
            "prune_shards",
            "release_state",
            "save_checkpoint",
        ),
    },
)
