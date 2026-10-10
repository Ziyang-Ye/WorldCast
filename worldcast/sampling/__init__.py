"""Sampling: the flow-matching table and the denoising steps, the block-causal sampler, the window
of a block.

The names below are imported on first use (PEP 562).
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "conditions": ("control_conditions",),
        "sampler": ("CONTEXT_NOISE", "DENOISING_STEPS", "CausalGenerator", "Sampler"),
        "schedulers": (
            "NUM_TRAIN_TIMESTEPS",
            "TIMESTEP_SHIFT",
            "BlockCall",
            "ContextWriteNoise",
            "DenoisingExit",
            "FlowMatchScheduler",
            "context_noise_seed",
            "denoise_block",
            "draw_noise",
            "entry_noise",
            "flow_to_x0",
            "nearest_index",
            "noise_context",
            "paired_context_noise",
            "renoise_frames",
            "run_denoising_steps",
            "shift_denoising_steps",
            "shift_sigma",
            "table_sigma",
            "x0_to_flow",
        ),
        "unipc": ("FlowUniPCSolver",),
        "window": (
            "CONTINUOUS_COLUMNS_KEY",
            "KV_CACHE_LATENTS",
            "MEMORY_CONTINUOUS_COLUMNS_KEY",
            "ROUND_CONTINUOUS_COLUMNS_KEY",
            "WINDOW_ENTRIES",
            "WindowEntry",
            "WindowLayout",
            "gather_window",
        ),
    },
)
