"""The state model: its cell table, its components, the two heads' outputs and loading."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tests.modeling.support import randomize_
from worldcast.modeling.state_model import (
    CELL_SIZE,
    STATE_MODEL_WINDOW_LATENT_FRAMES,
    CellTable,
    StateModel,
    StateModelOutput,
    load_cell_table,
    load_state_model,
    map_index,
)

CELLS = Path(__file__).resolve().parents[2] / "configs" / "state_model" / "cells.json"


@pytest.fixture(scope="module")
def table() -> CellTable:
    return load_cell_table(CELLS)


@pytest.fixture(scope="module")
def model(table) -> StateModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # building the model warns of nothing
        model = StateModel(table, dim=64, layers=2)
    return randomize_(model, 1, scale=0.05).eval()


def _window(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, STATE_MODEL_WINDOW_LATENT_FRAMES, 48, 24, 42, generator=g)
    return latents, torch.randn(1, STATE_MODEL_WINDOW_LATENT_FRAMES, 16, 16, generator=g)


def test_the_cell_table(table, tmp_path):
    """1860 occupied cells of 256 x 256 x 64 u over the four maps, in map order."""
    assert table.num_cells == 1860
    assert [int((table.cell_map == m).sum()) for m in range(4)] == [362, 461, 489, 548]
    assert table.cell_ijk[0].tolist() == [-9, -4, -1]
    cells = json.loads(CELLS.read_text())
    other = tmp_path / "cells.json"
    other.write_text(json.dumps({**cells, "cell_size_u": [128.0, 128.0, 64.0]}))
    with pytest.raises(ValueError, match="another cell lattice"):
        load_cell_table(other)
    other.write_text(json.dumps({**cells, "map_order": cells["map_order"][::-1]}))
    with pytest.raises(ValueError, match="map order .*, expected .*'de_ancient'"):
        load_cell_table(other)
    for another_json in ({"cell_size_u": cells["cell_size_u"], "decay_s": 0.5}, [1, 2]):
        other.write_text(json.dumps(another_json))
        with pytest.raises(ValueError, match="not a cell table"):
            load_cell_table(other)


def test_the_map_index_of_a_map():
    assert [map_index(name) for name in ("de_ancient", "de_mirage")] == [0, 2]
    with pytest.raises(ValueError, match="knows the maps .* not 'de_inferno'"):
        map_index("de_inferno")


def test_paper_architecture(table):
    """Encoder, trunk (16 layers of width 1280), motion head, action encoder (4 layers) and place
    head: the 349M parameters of the released file, 301 tensors with the place head's tables."""
    with torch.device("meta"):
        model = StateModel(table)
    assert [name for name, _ in model.named_children()] == [
        "encoder",
        "trunk",
        "motion_head",
        "action_encoder",
        "place_head",
    ]
    assert len(model.encoder.blocks) == 5 and len(model.trunk.transformer.layers) == 16
    assert len(model.action_encoder.transformer.layers) == 4
    assert sum(p.numel() for p in model.parameters()) == 349_467_194
    assert len(model.state_dict()) == 301


def test_the_place_heads_lattice(model):
    """A cell's origin is its lattice index times the cell size; the 4 x 4 sub-cells are 64 u wide
    and the offset reaches half a sub-cell."""
    head = model.place_head
    assert head.cell_origin[0].tolist() == [-2304.0, -1024.0, -64.0]
    assert head.sub_cell_centre[0].tolist() == [32.0, 32.0, 32.0]
    assert head.sub_cell_centre[6].tolist() == [160.0, 96.0, 32.0]  # column 2 of row 1
    assert head.offset_bound.tolist() == [32.0] * 3
    assert head.cell_mask.shape == (4, 1860) and head.cell_mask.sum(1).tolist() == [
        362,
        461,
        489,
        548,
    ]


def test_forward_gives_the_displacement_and_the_place_of_every_latent_frame(model, table):
    latents, controls = _window()
    with torch.no_grad():
        output = model(latents, controls, torch.tensor([2]))
    assert isinstance(output, StateModelOutput)
    assert (
        output.displacement.shape == output.place.shape == (1, STATE_MODEL_WINDOW_LATENT_FRAMES, 3)
    )
    # every place lies in an occupied cell of the map asked for (de_mirage)
    size = torch.tensor(CELL_SIZE)
    corners = torch.tensor(table.cell_ijk[table.cell_map == 2]) * size
    place = output.place[0, :, None]
    assert bool(((place >= corners) & (place <= corners + size)).all(-1).any(-1).all())


def test_the_trunk_and_the_action_encoder_are_causal(model):
    """A latent frame's displacement depends on the frames and controls up to it."""
    latents, controls = _window()
    later_latents, later_controls = latents.clone(), controls.clone()
    later_latents[:, 30:] += 1.0
    later_controls[:, 30:] += 1.0
    with torch.no_grad():
        output = model(latents, controls, torch.tensor([0]))
        later = model(later_latents, later_controls, torch.tensor([0]))
    torch.testing.assert_close(
        later.displacement[:, :30], output.displacement[:, :30], rtol=0, atol=1e-5
    )
    assert not torch.allclose(later.displacement[:, 30:], output.displacement[:, 30:], atol=1e-3)


def test_forward_names_a_wrong_input(model):
    latents, controls = _window()
    with pytest.raises(ValueError, match=r"windows \[B, 41, 48, 24, 42\]"):
        model(latents[:, :40], controls[:, :40], torch.tensor([0]))
    with pytest.raises(ValueError, match=r"windows \[B, 41, 48, 24, 42\]"):
        model(latents[..., :12, :21], controls, torch.tensor([0]))
    with pytest.raises(ValueError, match=r"controls must be \[1, 41, 16, 16\]"):
        model(latents, controls[..., :14], torch.tensor([0]))
    with pytest.raises(ValueError, match="map_id must be a \\[B\\] tensor"):
        model(latents, controls, 0)
    for map_id in (torch.tensor([4]), torch.tensor([1.0])):
        with pytest.raises(ValueError, match="integer indices of"):
            model(latents, controls, map_id)


def test_load_refuses_another_cell_table(model, table, tmp_path):
    """The file's own lattice buffers are what the model computes with, so the table given must
    be the one they were built from."""
    path = tmp_path / "state_model.safetensors"
    save_file(model.state_dict(), str(path))
    other = CellTable(cell_ijk=table.cell_ijk[::-1].copy(), cell_map=table.cell_map)
    with pytest.raises(ValueError, match="another cell table \\(cell_origin differs\\)"):
        load_state_model(path, other)
    other = CellTable(cell_ijk=table.cell_ijk, cell_map=np.roll(table.cell_map, 1))
    with pytest.raises(ValueError, match="another cell table \\(cell_mask differs\\)"):
        load_state_model(path, other)
    save_file({"inp.weight": torch.zeros(1)}, str(path))
    with pytest.raises(KeyError, match="no 'encoder.proj.weight': they are another module's"):
        load_state_model(path, table)


def test_load_builds_at_the_files_size_without_a_random_draw(model, table, tmp_path):
    path = tmp_path / "state_model.safetensors"
    save_file(model.state_dict(), str(path))
    torch.manual_seed(0)
    before = torch.get_rng_state()
    loaded = load_state_model(path, table)
    assert torch.equal(torch.get_rng_state(), before)
    assert not loaded.training and len(loaded.trunk.transformer.layers) == 2
    assert not any(p.requires_grad for p in loaded.parameters())
    state, want = loaded.state_dict(), model.state_dict()
    assert state.keys() == want.keys() and all(torch.equal(state[k], want[k]) for k in want)
    latents, controls = _window(seed=1)
    with torch.no_grad():
        a, b = (m(latents, controls, torch.tensor([1])) for m in (loaded, model))
    assert torch.equal(a.displacement, b.displacement) and torch.equal(a.place, b.place)
