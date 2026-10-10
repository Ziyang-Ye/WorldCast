"""Evaluation: the two protocols on the validation windows, from a checkpoint or inside training.

:class:`Evaluator` scores a generator on a :class:`Protocol` (:data:`UNIPC`, :data:`FOUR_STEP`) with
the samplers and the metrics of this package; :class:`Validation` is its use inside training. The
names below are imported on first use (PEP 562).
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "evaluator": ("Evaluator", "load_eval_generator"),
        "metrics": (
            "LPIPS",
            "METRICS",
            "MSE_FLOOR",
            "FrameMetric",
            "pixel_mse",
            "psnr",
            "score_window",
            "ssim",
            "summarize",
        ),
        "protocols": (
            "FOUR_STEP",
            "PROTOCOLS",
            "UNIPC",
            "WINDOW_SET",
            "EvalWindow",
            "Protocol",
            "check_selection",
            "load_windows",
            "read_index",
            "select_windows",
            "selection_sha256",
            "window_noise_seed",
        ),
        "samplers": ("FlowFn", "sample_bidirectional", "sample_block_causal", "sample_four_step"),
        "validation": ("Validation", "gather_generator"),
    },
)
