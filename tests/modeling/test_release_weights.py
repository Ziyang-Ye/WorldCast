"""The released weight files hold exactly the tensors of the modules that load them.

Only the safetensors headers are read (names and shapes), never the tensors. The files are looked
up in ``WORLDCAST_WEIGHTS_DIR`` (default ``weights/worldcast/``, where ``tools/download_weights.py
--out-dir weights`` puts them); a test skips when its file is absent.
"""

import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from worldcast import hub
from worldcast.modeling.depth_head import DepthHead, DepthReadout
from worldcast.modeling.state_model import StateModel, load_cell_table
from worldcast.modeling.wan22.model import GeneratorConfig, WorldCastGenerator
from worldcast.modeling.wan22.text_encoder import TEXT_DIM, TEXT_LEN

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = Path(os.environ.get("WORLDCAST_WEIGHTS_DIR") or REPO / "weights" / "worldcast")
PROMPT_EMBEDDING = hub.WEIGHTS["prompt_embedding"]


def _state_model() -> StateModel:
    return StateModel(load_cell_table(REPO / "configs" / "state_model" / "cells.json"))


#: Release file -> the module it loads into, and its tensor count.
FILES = {
    "worldcast_4step_bf16.safetensors": (lambda: WorldCastGenerator(GeneratorConfig()), 946),
    "state_model.safetensors": (_state_model, 301),
    "depth_head.safetensors": (DepthHead, 130),
    "depth_readout.safetensors": (DepthReadout, 10),
}


def _header(name: str) -> dict[str, tuple[int, ...]]:
    path = WEIGHTS / name
    if not path.is_file():
        pytest.skip(f"{path} is absent (set WORLDCAST_WEIGHTS_DIR to the release weights)")
    with safe_open(str(path), framework="pt") as f:
        return {key: tuple(f.get_slice(key).get_shape()) for key in f.keys()}


@pytest.mark.parametrize("name", FILES)
def test_a_release_file_holds_the_tensors_of_its_module(name):
    got = _header(name)
    build, count = FILES[name]
    with torch.device("meta"):
        want = {key: tuple(value.shape) for key, value in build().state_dict().items()}
    assert len(want) == count
    assert got.keys() == want.keys(), (
        sorted(want.keys() - got.keys())[:8],
        sorted(got.keys() - want.keys())[:8],
    )
    assert got == want


def test_the_prompt_embedding_is_one_padded_umt5_embedding():
    assert _header(PROMPT_EMBEDDING) == {"prompt_embeds": (1, TEXT_LEN, TEXT_DIM)}


def test_every_file_of_the_download_is_checked():
    assert {*hub.WEIGHTS.values()} <= {*FILES, PROMPT_EMBEDDING}
