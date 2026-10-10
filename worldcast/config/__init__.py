"""Configuration: the inference and training configs, and the YAML loader they share.

``load_config`` and ``load_train_config`` merge YAML files and dotted ``--set`` overrides into an
:class:`InferenceConfig` and a :class:`TrainConfig`; ``paper_config`` is a training stage's paper
run. A tool adds ``--config`` / ``--set`` with ``add_config_args`` and reads the overrides with
``parse_overrides``.
"""

from .inference import InferenceConfig, SceneStateConfig, load_config
from .loader import add_config_args, build, config_to_dict, load, parse_overrides
from .training import (
    STAGES,
    WINDOW_LATENT_FRAMES,
    TrainConfig,
    load_train_config,
    paper_config,
)

__all__ = [
    "STAGES",
    "WINDOW_LATENT_FRAMES",
    "InferenceConfig",
    "SceneStateConfig",
    "TrainConfig",
    "add_config_args",
    "build",
    "config_to_dict",
    "load",
    "load_config",
    "load_train_config",
    "paper_config",
    "parse_overrides",
]
