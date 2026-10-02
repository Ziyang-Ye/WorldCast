"""Command-line config handling shared by the scripts: YAML files merged in order, then ``--set key=value``."""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = REPO / "configs" / "infer" / "worldcast_4step.yaml"


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="YAML",
        help=(
            "config file; repeat to merge several in order (default:"
            f" {DEFAULT_CONFIG.relative_to(REPO)})"
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="override one dotted config key, e.g. --set run.seed=1 --set paths.out_dir=out/x",
    )


def _merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """``["run.seed=1", "paths.out_dir=out"]`` -> ``{"run.seed": 1, "paths.out_dir": "out"}`` (YAML-typed values)."""
    import yaml

    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def load_cli_config(configs: list[str] | None, overrides: Mapping[str, Any]):
    """Merge the YAML files, apply the dotted overrides, validate: an :class:`worldcast.config.inference.InferenceConfig`."""
    import yaml

    from .inference import config_from_dict, with_overrides

    data: dict[str, Any] = {}
    for path in configs or [str(DEFAULT_CONFIG)]:
        with open(path, encoding="utf-8") as fh:
            data = _merge(data, yaml.safe_load(fh) or {})
    cfg = config_from_dict(data)
    return with_overrides(cfg, overrides) if overrides else cfg
