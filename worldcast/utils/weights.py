"""Weight files: reading them, and a module that holds their tensors."""

import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

__all__ = [
    "check_state_dict",
    "indexed_count",
    "is_safetensors",
    "leading_size",
    "module_from_state",
    "read_state_dict",
]

_Module = TypeVar("_Module", bound=nn.Module)


def is_safetensors(path: str | Path) -> bool:
    """Whether ``path`` names a ``.safetensors`` file (the release's weights) rather than a torch
    file."""
    return str(path).endswith(".safetensors")


def read_state_dict(path: str | Path) -> dict[str, Any]:
    """The content of a weight file, on the CPU.

    Args:
        path (str | Path): a ``.safetensors`` file, or a torch file, which is unpickled with
            ``weights_only`` (tensors and plain containers) and memory-mapped when it is a zip
            archive (every file torch has written by default since 1.6).

    Returns:
        dict[str, Any]: the tensors by name; a torch checkpoint's payload as it was saved.
    """
    if is_safetensors(path):
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(
        str(path), map_location="cpu", mmap=zipfile.is_zipfile(path), weights_only=True
    )


def indexed_count(state: Mapping[str, Any], prefix: str) -> int:
    """How many numbered submodules ``<prefix>0.``, ``<prefix>1.``, ... a state dict holds."""
    indices = [int(key[len(prefix) :].split(".")[0]) for key in state if key.startswith(prefix)]
    if not indices:
        raise KeyError(f"the weights have no {prefix!r} keys: they are another module's")
    return 1 + max(indices)


def leading_size(state: Mapping[str, torch.Tensor], key: str) -> int:
    """``state[key].shape[0]``: a width a module is sized with from its weights."""
    if key not in state:
        raise KeyError(f"the weights have no {key!r}: they are another module's")
    return int(state[key].shape[0])


def check_state_dict(
    expected: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor]
) -> list[str]:
    """Check weights against a module's state: a key the module does not have, or another shape,
    is an error.

    Args:
        expected (Mapping[str, Tensor]): the module's ``state_dict()`` (``meta`` is enough).
        state (Mapping[str, Tensor]): the weights, under the module's key names.

    Returns:
        list[str]: the module's keys the weights lack, sorted.
    """
    unexpected = sorted(set(state) - set(expected))
    if unexpected:
        raise KeyError(f"the weights hold keys the module does not have: {unexpected[:8]}")
    wrong = [
        (key, tuple(value.shape), tuple(expected[key].shape))
        for key, value in state.items()
        if tuple(value.shape) != tuple(expected[key].shape)
    ]
    if wrong:
        raise ValueError(f"the weights' and the module's shapes differ: {wrong[:4]}")
    return sorted(set(expected) - set(state))


def module_from_state(build: Callable[[], _Module], state: Mapping[str, torch.Tensor]) -> _Module:
    """The module ``build()`` holding the tensors of ``state``, every key of it (strict).

    The module is built on ``meta`` and the tensors are assigned: no parameter is initialised, so
    the global RNG does not advance, and the parameters keep the dtypes of ``state``.

    Args:
        build (Callable[[], nn.Module]): constructs the module.
        state (Mapping[str, Tensor]): its weights, on the CPU.
    """
    with torch.device("meta"):
        module = build()
    missing = check_state_dict(module.state_dict(), state)
    if missing:
        raise KeyError(f"the weights are missing keys of {type(module).__name__}: {missing[:8]}")
    module.load_state_dict(dict(state), strict=True, assign=True)
    return module
