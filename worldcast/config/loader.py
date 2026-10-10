"""The config loader: YAML files merged in order, dotted ``key=value`` overrides, dataclasses.

The inference and training configs both load through it. The files are merged, the overrides
are set in the merged mapping, and the dataclasses are built from the result. Loading is strict: a
key written twice in one mapping, an unknown key and a value of the wrong type are errors, and each
dataclass validates its own values in ``__post_init__``.
"""

import argparse
import contextlib
import copy
import dataclasses
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "Paths",
    "add_config_args",
    "build",
    "check_at_least",
    "check_choice",
    "config_to_dict",
    "load",
    "merge",
    "parse_overrides",
    "read_yaml",
    "require_set",
    "set_dotted",
    "usage_errors",
]

T = TypeVar("T")
#: One config file or several, merged in order.
Paths = str | os.PathLike[str] | Sequence[str | os.PathLike[str]]


class _Loader(yaml.SafeLoader):
    """``yaml.safe_load`` that refuses a key written twice in one mapping (the first would be lost
    silently) and reads a number in scientific notation as a number."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        keys = [self.construct_object(key, deep=True) for key, _ in node.value]
        twice = sorted({str(key) for key in keys if keys.count(key) > 1})
        if twice:
            raise ValueError(f"keys written twice: {twice}{node.start_mark}")
        return super().construct_mapping(node, deep=deep)


# YAML 1.1 reads ``1e-5`` and ``1.5e3`` as strings: it wants a dot and a signed exponent
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` (files merged in order over the defaults) and ``--set KEY=VALUE``."""
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="YAML",
        help="config file merged over the defaults; repeat to merge several in order",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="override one dotted key (YAML-typed), e.g. --set section.key=value",
    )


@contextlib.contextmanager
def usage_errors(parser: argparse.ArgumentParser) -> Iterator[None]:
    """Report a config a tool cannot run with as the parser's usage error.

    Inside the block, a file that cannot be read (``OSError``), a file that is not YAML and every
    refusal of the loader or of a config's own validation (``ValueError``) end the tool with its
    usage message and exit status 2 instead of a traceback.
    """
    try:
        yield
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


def check_choice(key: str, value: object, allowed: Sequence[object]) -> None:
    """Raise ``ValueError`` unless the config value ``value`` of ``key`` is one of ``allowed``."""
    if value not in allowed:
        raise ValueError(f"{key} must be one of {allowed}, got {value!r}")


def check_at_least(key: str, value: float, minimum: float) -> None:
    """Raise ``ValueError`` unless the config value ``value`` of ``key`` is ``>= minimum``."""
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value!r}")


def require_set(section: Any, where: str, names: Iterable[str]) -> None:
    """Raise ``ValueError`` naming the keys ``<where>.<name>`` of a config section that are unset
    (None or empty)."""
    missing = [f"{where}.{name}" for name in names if not getattr(section, name)]
    if missing:
        raise ValueError(f"not set: {', '.join(missing)} (a config file or --set names them)")


def merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """``extra`` merged into a copy of ``base``: mappings recursively, everything else replaced."""
    out = copy.deepcopy(dict(base))
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_yaml(path: str | os.PathLike[str], _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """A YAML mapping; ``inherit: <path relative to the file>`` merges the file under it first."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"inherit cycle through {path}")
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader) or {}
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: the top level of a config must be a mapping")
    data = dict(data)
    parent = data.pop("inherit", None)
    if parent is None:
        return data
    return merge(read_yaml(path.parent / str(parent), _seen + (path,)), data)


def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """``["run.seed=1", "optim.lr=1e-5"]`` -> ``{"run.seed": 1, "optim.lr": 1e-05}``: the values
    are read as YAML, like a file's."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, text = item.split("=", 1)
        out[key.strip()] = yaml.load(text, Loader=_Loader)
    return out


def set_dotted(data: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``data`` with every dotted key of ``overrides`` set: ``{"run.seed": 1}`` sets
    ``data["run"]["seed"]``, replacing what is there; a mapping-valued setting takes one of its
    keys (``data.collision_meshes.de_dust2``) or a whole mapping."""
    out = copy.deepcopy(dict(data))
    for dotted, value in overrides.items():
        *sections, key = str(dotted).split(".")
        node = out
        for section in sections:
            if node.get(section) is None:
                node[section] = {}
            node = node[section]
            if not isinstance(node, dict):
                raise ValueError(f"override {dotted!r}: {section!r} is a value, not a section")
        node[key] = copy.deepcopy(value)
    return out


def build(cls: type[T], data: Mapping[str, Any], where: str = "") -> T:
    """The dataclass ``cls`` of nested mappings; unknown keys, wrong types: ``ValueError``."""
    if not isinstance(data, Mapping):
        raise ValueError(f"config section {where or '<root>'} must be a mapping, got {data!r}")
    hints = get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ValueError(f"unknown config keys in {where or '<root>'}: {unknown}")
    return cls(
        **{
            name: _coerce(hints[name], data[name], f"{where}.{name}" if where else name)
            for name in names
            if name in data
        }
    )


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Plain nested dicts and lists of a config, the inverse of :func:`build`."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if dataclasses.is_dataclass(value):
            value = config_to_dict(value)
        elif isinstance(value, (tuple, list)):
            value = list(value)
        elif isinstance(value, dict):
            value = dict(value)
        out[f.name] = value
    return out


def load(
    paths: Paths, overrides: Mapping[str, Any] | None, from_dict: Callable[[dict[str, Any]], T]
) -> T:
    """Merge the YAML files in order, set the dotted ``overrides`` in the merged mapping
    (:func:`set_dotted`) and build with ``from_dict``."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    data: dict[str, Any] = {}
    for path in paths:
        data = merge(data, read_yaml(path))
    return from_dict(set_dotted(data, overrides or {}))


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin, args = get_origin(tp), get_args(tp)
    if tp is Any:
        return value
    if tp is str and isinstance(value, int) and not isinstance(value, bool):
        return str(value)  # YAML reads a bare 4 as a number: --set run.stage=4
    if dataclasses.is_dataclass(tp):
        return build(tp, value, where)
    if origin in (Union, UnionType):
        if value is None and NoneType in args:
            return None
        return _coerce(next(a for a in args if a is not NoneType), value, where)
    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where} must be a list, got {value!r}")
        if origin is list or (len(args) == 2 and args[1] is Ellipsis):
            return origin(_coerce(args[0], v, where) for v in value)
        if len(value) != len(args):
            raise ValueError(f"{where} must have {len(args)} entries, got {list(value)!r}")
        return tuple(_coerce(a, v, where) for a, v in zip(args, value))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ValueError(f"{where} must be a mapping, got {value!r}")
        return {
            _coerce(args[0], k, where): _coerce(args[1], v, f"{where}.{k}")
            for k, v in value.items()
        }
    if tp is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} must be an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} must be a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a string, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported config type {tp!r}")
