"""The released checkpoint: its ``generator_ema`` state and the map of its keys onto the model."""

import re
from collections.abc import Mapping
from pathlib import Path

import torch

__all__ = [
    "DROPPED_KEYS",
    "DROPPED_PREFIXES",
    "GENERATOR_EMA_KEY",
    "KEY_RENAMES",
    "read_generator_ema",
    "remap_generator_state",
]

#: Payload key of the EMA generator in a checkpoint.
GENERATOR_EMA_KEY = "generator_ema"
#: Checkpoint modules that never run at inference: the visibility probe (training only).
DROPPED_PREFIXES = ("visibility_head.",)
#: Two scalars of an unused splat kernel of the field stem.
DROPPED_KEYS = (
    "peer_raster.raster_write.raster_bias",
    "peer_raster.raster_write.raster_log_kappa",
)
#: Checkpoint module prefix -> module of the generator (backbone keys are unchanged).
KEY_RENAMES = (
    ("peer_raster.raster_write.", "state_injector."),
    ("worldplay_memory.", "rays."),
    ("opencs2_action.", "action."),
)
_ADALN = re.compile(r"^(blocks\.\d+|head)\.opencs2_action_adaln\.")
_ROOT = "model."
_FSDP_ROOT = "model._fsdp_wrapped_module."


def remap_generator_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint keys (under ``model.``) -> generator keys; the tensors are not copied.

    Args:
        state (Mapping[str, Tensor]): a ``generator_ema`` state.

    Returns:
        dict[str, Tensor]: the state with the keys of
        :class:`~worldcast.modeling.wan22.model.WorldCastGenerator`.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = key.replace(_FSDP_ROOT, _ROOT, 1)
        if not name.startswith(_ROOT):
            raise KeyError(f"generator_ema key {key!r} is outside the wrapper's 'model.' root")
        name = name[len(_ROOT) :]
        if name in DROPPED_KEYS or name.startswith(DROPPED_PREFIXES):
            continue
        for source, target in KEY_RENAMES:
            if name.startswith(source):
                name = target + name[len(source) :]
                break
        name = _ADALN.sub(r"\1.action_adaln.", name)
        if name in out:
            raise KeyError(f"two generator_ema keys map to {name!r}")
        out[name] = value
    return out


def read_generator_ema(path: str | Path, *, weights_only: bool = True) -> dict[str, torch.Tensor]:
    """``payload["generator_ema"]`` of a checkpoint, memory-mapped.

    Args:
        path (str | Path): the checkpoint file.
        weights_only (bool): safe unpickling; a training checkpoint, which carries non-tensor
            metadata, may need ``False`` (only for files you trust).
    """
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=weights_only)
    if GENERATOR_EMA_KEY not in payload:
        raise KeyError(f"{path} has no {GENERATOR_EMA_KEY!r} entry (keys: {sorted(payload)[:8]})")
    return payload[GENERATOR_EMA_KEY]
