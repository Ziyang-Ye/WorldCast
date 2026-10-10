"""Helpers of the tool tests: the tools as modules, the engine tests' synthetic round with its
config, as an object and as a YAML file, and the boxes of an mp4."""

import importlib.util
import struct
from pathlib import Path

import yaml

from worldcast.config.inference import InferenceConfig
from worldcast.config.loader import config_to_dict

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load_tool(name: str):
    """``tools/<name>.py`` as a module (the tools are scripts, not a package)."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_round(
    tmp_path: Path, monkeypatch, *, max_blocks: int, seconds: float = 8.5, **config
) -> tuple[dict, InferenceConfig]:
    """The synthetic round with one client and tiny random weights, its tick tables served for the
    rest of the test: the world and the client's config (``max_blocks`` blocks after latent 24,
    output under ``tmp_path / "run"``; ``seconds``: the round's recorded length; ``config``: more
    arguments of the engine tests' config)."""
    import torch

    import tests.engine.inference.support as sw

    torch.set_num_threads(1)
    world = sw.make_world(tmp_path / "world", clients=(0,), seconds=seconds)
    sw.patch_ticks(monkeypatch, world["tables"])
    weights = sw.make_weights(tmp_path / "weights")
    return world, sw.config(world, weights, tmp_path / "run", max_blocks=max_blocks, **config)


def mp4_boxes(path: Path) -> list[tuple[str, int, int]]:
    """An mp4's top-level boxes in file order: each one's kind and where its content starts and
    the box ends, bytes into the file. A size of 1 is followed by the box's 64-bit size; a size of
    0 runs to the end of the file."""
    data, boxes, at = path.read_bytes(), [], 0
    while at < len(data):
        size, kind = struct.unpack(">I4s", data[at : at + 8])
        header = 8
        if size == 1:
            (size,), header = struct.unpack(">Q", data[at + 8 : at + 16]), 16
        elif size == 0:
            size = len(data) - at
        if size < header:
            raise ValueError(f"{path}: a box of {size} bytes at byte {at}")
        boxes.append((kind.decode(), at + header, at + size))
        at += size
    return boxes


def write_config(cfg: InferenceConfig, path: Path) -> str:
    """``cfg`` as a YAML file without the entries a tool sets itself."""
    data = config_to_dict(cfg)
    for key in ("world_state_dir", "out_dir"):
        data["paths"].pop(key)
    data["run"].pop("index_row")
    path.write_text(yaml.safe_dump(data))
    return str(path)
