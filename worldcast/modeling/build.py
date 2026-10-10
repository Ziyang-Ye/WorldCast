"""Read the generator's weights and build the inference generator from them (docs/inference.md,
"Weights").

A release ``.safetensors`` file holds the generator's state under its own key names; a training
checkpoint holds it in a payload entry (``generator_ema``, ``generator``) under the ``model.``
root.
"""

import dataclasses
import json
from collections.abc import Collection, Mapping
from pathlib import Path

import torch

from worldcast.modeling.wan22.attention import PAPER_ATTENTION, AttentionFn, attention_kernel
from worldcast.modeling.wan22.model import (
    PATCH_SIZE,
    GeneratorConfig,
    WorldCastGenerator,
    worldcast_module,
)
from worldcast.utils.precision import GENERATOR_DTYPE, generator_dtype
from worldcast.utils.weights import is_safetensors, module_from_state, read_state_dict

__all__ = [
    "CHECKPOINT_ROOT",
    "EMA_KEY",
    "ONLINE_KEY",
    "build_inference_generator",
    "generator_config_from_snapshot",
    "load_generator",
    "optional_modules",
    "read_generator",
]

#: Root of every generator key in an entry of a training checkpoint: an entry is the state of the
#: trainer's module around the generator, saved under one root whatever that module calls the
#: generator. A release file has the generator's own key names, without a root.
CHECKPOINT_ROOT = "model."
#: Payload entries of a training checkpoint: the EMA weights (evaluated and released) and the
#: online weights (resumed).
EMA_KEY = "generator_ema"
ONLINE_KEY = "generator"

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
    "text_dim",
    "text_len",
)
#: ``config.json`` keys the generator fixes, with its values: another value is refused.
_SNAPSHOT_FIXED = {
    "model_type": "ti2v",
    "patch_size": list(PATCH_SIZE),
    "cross_attn_norm": True,
    "qk_norm": True,
    "window_size": [-1, -1],
}


def generator_config_from_snapshot(config_json: str | Path) -> GeneratorConfig:
    """The paper's :class:`GeneratorConfig` with the backbone dimensions of a Wan2.2-TI2V-5B
    ``config.json``; a key the generator does not know, or a value it does not honour, is an
    error."""
    raw = json.loads(Path(config_json).read_text())
    unknown = sorted(
        key
        for key in raw
        if not key.startswith("_") and key not in (*_SNAPSHOT_KEYS, *_SNAPSHOT_FIXED)
    )
    if unknown:
        raise ValueError(f"{config_json}: unexpected config keys {unknown}")
    other = {
        key: raw[key] for key, fixed in _SNAPSHOT_FIXED.items() if raw.get(key, fixed) != fixed
    }
    if other:
        raise ValueError(f"{config_json}: the generator fixes {_SNAPSHOT_FIXED}, not {other}")
    return GeneratorConfig(**{key: raw[key] for key in _SNAPSHOT_KEYS if key in raw})


def read_generator(path: str | Path, key: str | None = None) -> dict[str, torch.Tensor]:
    """The generator's weights of a file, under the generator's key names.

    Args:
        path (str | Path): a release ``.safetensors`` file, or a training checkpoint.
        key (str | None): the payload entry of a training checkpoint; ``None``: :data:`EMA_KEY`
            when the file has it, else :data:`ONLINE_KEY`. A release file holds one state:
            ``key`` does not apply to it.
    """
    payload = read_state_dict(path)
    if is_safetensors(path):
        return payload
    if key is None:
        key = EMA_KEY if EMA_KEY in payload else ONLINE_KEY
    if key not in payload:
        raise KeyError(f"{path} has no {key!r} entry (keys: {sorted(payload)[:8]})")
    state = payload[key]
    outside = sorted(name for name in state if not name.startswith(CHECKPOINT_ROOT))
    if outside:
        raise KeyError(f"{path}: {key!r} keys outside the {CHECKPOINT_ROOT!r} root: {outside[:4]}")
    return {name[len(CHECKPOINT_ROOT) :]: value for name, value in state.items()}


def optional_modules(config: GeneratorConfig, keys: Collection[str]) -> GeneratorConfig:
    """``config`` with the optional modules that the weights of ``keys`` hold: the state injector,
    the ray embedding and the observer-signal embedding (a generator trained without scene state
    has neither of the last two)."""
    held = {worldcast_module(key) for key in keys}
    return dataclasses.replace(
        config,
        state_injector=config.state_injector if "state_injector" in held else None,
        observer_signals=config.observer_signals if "observer_signals" in held else None,
        ray_embedding="ray_embedding" in held,
    )


def build_inference_generator(
    config: GeneratorConfig,
    state: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str = "cpu",
    attention: AttentionFn | None = None,
) -> WorldCastGenerator:
    """The generator with ``state`` assigned, on ``device``, in eval mode, without gradients and
    without the visibility probe (training only).

    The parameters are cast to bf16, as in the paper's runs; on the CPU they are then held in
    float32 (:func:`~worldcast.utils.precision.generator_dtype`) and the default kernel is
    ``sdpa_attention``, so the generator runs there on float32 inputs.

    Args:
        config (GeneratorConfig): the architecture.
        state (Mapping[str, Tensor]): every key of the generator (:func:`read_generator`); the
            visibility probe's are left out.
        device (torch.device | str): where the generator lives: CUDA (the paper's bf16 generator)
            or the CPU.
        attention (AttentionFn | None): the kernel; by default flash-attention 2 on CUDA, the
            paper's, and ``sdpa_attention`` on the CPU.
    """
    if torch.device(device).type not in ("cuda", "cpu"):
        raise ValueError(f"the generator runs on CUDA or on the CPU (float64 RoPE), not {device}")
    if attention is None:
        attention = attention_kernel(PAPER_ATTENTION, device)
    config = dataclasses.replace(config, visibility_probe=None)
    state = {
        key: value for key, value in state.items() if worldcast_module(key) != "visibility_probe"
    }
    model = module_from_state(lambda: WorldCastGenerator(config, attention=attention), state)
    model.to(dtype=GENERATOR_DTYPE)
    model.to(device=device, dtype=generator_dtype(device))
    model.requires_grad_(False)
    return model.eval()


def load_generator(
    checkpoint: str | Path,
    config: GeneratorConfig = GeneratorConfig(),
    *,
    device: torch.device | str = "cpu",
    attention: AttentionFn | None = None,
) -> WorldCastGenerator:
    """Read a checkpoint and build the inference generator from it.

    Args:
        checkpoint (str | Path): a release ``.safetensors`` file, or a training checkpoint with a
            :data:`EMA_KEY` entry.
        config (GeneratorConfig): the dimensions; the optional modules are those the checkpoint
            holds (:func:`optional_modules`).
        device (torch.device | str): where the generator lives.
        attention (AttentionFn | None): the kernel (:func:`build_inference_generator`).
    """
    state = read_generator(checkpoint, EMA_KEY)
    config = optional_modules(config, state)
    return build_inference_generator(config, state, device=device, attention=attention)
