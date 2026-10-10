"""Inference (the client), training, evaluation, checkpoints and the optimizer.

Beside the subpackages, what they share of the generator: how it is loaded and called
(:mod:`worldcast.engine.generator`) and how a training stage configures it
(:mod:`worldcast.engine.stage`). The names below are imported on first use (PEP 562).
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "generator": (
            "WindowInputs",
            "backbone_config",
            "client_cameras",
            "load_prompt_embeds",
            "load_vae",
            "window_conditions",
            "window_inputs",
        ),
        "stage": ("built_state", "field_config", "generator_config"),
    },
)
