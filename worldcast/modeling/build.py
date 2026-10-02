"""Build the generator from a config and load a checkpoint into it (docs/checkpoints.md)."""

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldcast.engine.checkpoint.formats import read_generator_ema, remap_generator_state
from worldcast.modeling.wan22.attention import AttentionFn
from worldcast.modeling.wan22.model import GeneratorConfig, WorldCastGenerator
from worldcast.utils.precision import cast_parameters_bf16

from .action import ActionConfig
from .state_injector import FIELD_CHANNELS, StateInjectorConfig

__all__ = [
    "build_generator",
    "generator_config_from_inference",
    "generator_config_from_snapshot",
    "load_generator",
]

#: ``config.json`` keys of the Wan2.2-TI2V-5B snapshot that set :class:`GeneratorConfig` fields.
_SNAPSHOT_KEYS = (
    "dim",
    "eps",
    "ffn_dim",
    "freq_dim",
    "in_dim",
    "num_heads",
    "num_layers",
    "out_dim",
    "text_len",
)
#: ``config.json`` keys that are not read (the generator's values are fixed).
_SNAPSHOT_IGNORED = ("patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size")


def generator_config_from_snapshot(
    config_json: str | Path, base: GeneratorConfig = GeneratorConfig()
) -> GeneratorConfig:
    """``base`` with the backbone dimensions of a Wan2.2-TI2V-5B ``config.json``.

    Args:
        config_json (str | Path): the snapshot's ``config.json``; an unknown key is an error.
        base (GeneratorConfig): the other fields.
    """
    raw = json.loads(Path(config_json).read_text())
    model_type = raw.get("model_type", "ti2v")
    if model_type not in ("t2v", "ti2v"):
        raise ValueError(f"unsupported Wan2.2 model_type {model_type!r}")
    known = (*_SNAPSHOT_KEYS, *_SNAPSHOT_IGNORED, "model_type")
    unknown = sorted(k for k in raw if not k.startswith("_") and k not in known)
    if unknown:
        raise ValueError(f"{config_json}: unexpected config keys {unknown}")
    return dataclasses.replace(base, **{k: raw[k] for k in _SNAPSHOT_KEYS if k in raw})


def generator_config_from_inference(
    cfg: Any, base: GeneratorConfig = GeneratorConfig()
) -> GeneratorConfig:
    """``base`` (the backbone) with the conditioning of an inference config.

    Args:
        cfg (InferenceConfig): a :class:`worldcast.config.inference.InferenceConfig`.
        base (GeneratorConfig): the backbone fields.
    """
    model, field = cfg.model, cfg.model.player_field
    channels = 6 + len(field.action_signals) + int(field.weapon_channels) + 2 + 2
    if channels != FIELD_CHANNELS:
        raise NotImplementedError(
            f"the config describes a {channels}-channel field, not {FIELD_CHANNELS}"
        )
    action = ActionConfig(
        button_dim=int(model.action_button_dim),
        camera_dim=int(model.action_camera_dim),
        weapon_vocab_size=int(model.action_weapon_vocab_size),
        weapon_embedding_dim=int(model.action_weapon_embedding_dim),
        vae_time_compression_ratio=int(model.vae_time_compression),
        history_frames=int(model.action_history_frames),
        hidden_dim=int(model.action_hidden_dim),
        adaln_rank=int(model.action_adaln_rank),
    )
    injector = StateInjectorConfig(
        write_block=int(field.inject_after_block),
        field_channels=channels,
        hidden=int(field.stem_hidden),
        weapon_vocab=int(model.action_weapon_vocab_size),
        weapon_channels=int(field.weapon_channels),
    )
    return dataclasses.replace(
        base,
        in_dim=int(model.latent_channels),
        out_dim=int(model.latent_channels),
        patch_size=tuple(int(v) for v in model.patch_size),
        action=action,
        state_injector=injector,
        ray_unit_u=float(model.ray_unit_u),
        obs_signal_hidden=int(model.obs_signal_hidden),
    )


def build_generator(
    config: GeneratorConfig,
    state: Mapping[str, torch.Tensor],
    *,
    device: str | torch.device = "cuda",
    attention: AttentionFn | None = None,
) -> WorldCastGenerator:
    """The generator with ``state`` assigned: bf16 parameters on ``device``, eval mode, no grad.

    Args:
        config (GeneratorConfig): the architecture.
        state (Mapping[str, Tensor]): a remapped state (:func:`remap_generator_state`).
        device (str | torch.device): where the generator lives.
        attention (AttentionFn | None): the kernel; flash-attn by default, ``sdpa_attention`` on
            the CPU.
    """
    with torch.device("meta"):
        model = WorldCastGenerator(config, attention=attention)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise KeyError(
            f"checkpoint/model key mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    wrong = sorted(k for k, v in state.items() if tuple(v.shape) != tuple(expected[k].shape))
    if wrong:
        detail = [(k, tuple(state[k].shape), tuple(expected[k].shape)) for k in wrong[:4]]
        raise ValueError(f"checkpoint/model shape mismatch: {detail}")
    model.load_state_dict(dict(state), strict=True, assign=True)
    cast_parameters_bf16(model)
    model.to(device=device)
    model.requires_grad_(False)
    return model.eval()


def load_generator(
    checkpoint: str | Path,
    config: GeneratorConfig = GeneratorConfig(),
    *,
    device: str | torch.device = "cuda",
    attention: AttentionFn | None = None,
    weights_only: bool = True,
) -> WorldCastGenerator:
    """Read a checkpoint and build the generator from it.

    A ``.safetensors`` file (the release format) holds the generator's own key names and loads as it
    is; a torch checkpoint's ``generator_ema`` goes through :func:`remap_generator_state`.

    Args:
        checkpoint (str | Path): the checkpoint file.
        config (GeneratorConfig): the architecture.
        device (str | torch.device): where the generator lives.
        attention (AttentionFn | None): the kernel (:func:`build_generator`).
        weights_only (bool): safe unpickling of a torch checkpoint (:func:`read_generator_ema`).
    """
    if str(checkpoint).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(str(checkpoint), device="cpu")
    else:
        state = remap_generator_state(read_generator_ema(checkpoint, weights_only=weights_only))
    return build_generator(config, state, device=device, attention=attention)
