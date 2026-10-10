"""A training stage's generator, as training and evaluation build it: its config, the weights it
takes of a checkpoint and its player state field."""

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from worldcast.config.training import TrainConfig
from worldcast.data.latents import BLOCK
from worldcast.modeling.visibility_probe import VisibilityProbeConfig
from worldcast.modeling.wan22.model import GeneratorConfig, worldcast_module
from worldcast.player_state import PlayerStateFieldConfig

from .generator import backbone_config

__all__ = ["built_state", "field_config", "generator_config"]

#: The modules a checkpoint may carry although its generator is built without them: the visibility
#: probe (training only) and the modules of scene state (zero in the checkpoints of a model without
#: scene state that kept them). The :class:`GeneratorConfig` field of a module's name says whether
#: it is built.
_OPTIONAL_MODULES = ("visibility_probe", "observer_signals", "ray_embedding")


def built_state(
    state: Mapping[str, torch.Tensor], config: GeneratorConfig
) -> dict[str, torch.Tensor]:
    """A checkpoint's ``state`` without the visibility probe and the modules of scene state
    where the generator of ``config`` does not build them. Every other key stays, and must be the
    generator's."""
    unbuilt = [module for module in _OPTIONAL_MODULES if not getattr(config, module)]
    return {key: value for key, value in state.items() if worldcast_module(key) not in unbuilt}


def generator_config(cfg: TrainConfig, *, score_model: bool = False) -> GeneratorConfig:
    """The stage's generator: Wan2.2-TI2V-5B (the snapshot's dimensions when ``model.wan22_root``
    has them), the modules the stage trains and the ablation switches.

    Args:
        cfg (TrainConfig): the run.
        score_model (bool): the distillation's teacher and critic instead, the stage-2 model:
            without scene state.
    """
    stage = cfg.stage
    base = backbone_config(cfg.model.wan22_root)
    scene_state = stage.scene_state and not score_model
    injector = dataclasses.replace(base.state_injector, dit_block=cfg.model.state_injector_block)
    config = dataclasses.replace(
        base,
        state_injector=injector if stage.player_state_field else None,
        observer_signals=base.observer_signals if scene_state else None,
        ray_embedding=scene_state,
        field_downsample=cfg.model.field_downsample,
        visibility_probe=VisibilityProbeConfig() if scene_state else None,
    )
    return _replace_nested(config, cfg.model.dims) if cfg.model.dims else config


def _replace_nested(config: Any, overrides: Mapping[str, Any]) -> Any:
    """``dataclasses.replace`` by dotted names (``controls.hidden``); a ``None`` section is
    left alone."""
    top: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = {}
    for dotted, value in overrides.items():
        head, _, rest = dotted.partition(".")
        if rest:
            nested.setdefault(head, {})[rest] = value
        else:
            top[head] = value
    for head, inner in nested.items():
        if getattr(config, head) is not None:
            top[head] = _replace_nested(getattr(config, head), inner)
    return dataclasses.replace(config, **top)


def field_config(cfg: TrainConfig, *, bidirectional: bool) -> PlayerStateFieldConfig:
    """The player state field of a generator.

    Its confidence restarts at each block of the attention: one block of the whole window when
    bidirectional, the first frame and blocks of 4 latent frames otherwise. Without the visibility
    gate (Table 5, "Trained without visibility") the field has no confidence either: a floor of 1
    maps every label to 1.

    Args:
        cfg (TrainConfig): the run.
        bidirectional (bool): the generator attends over the whole window.
    """
    blocks = (
        dict(frames_per_block=cfg.stage.latent_frames, first_frame_alone=False)
        if bidirectional
        else dict(frames_per_block=BLOCK, first_frame_alone=True)
    )
    floor = {} if cfg.model.visibility_gate else {"confidence_floor": 1.0}
    return PlayerStateFieldConfig(**blocks, **floor)
