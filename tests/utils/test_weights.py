"""Weight files: safetensors, or a torch file unpickled with ``weights_only``; and the module that
holds their tensors."""

import pickle

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from worldcast.utils.weights import (
    check_state_dict,
    indexed_count,
    is_safetensors,
    leading_size,
    module_from_state,
    read_state_dict,
)


class _Code:
    """Something a weight file must not carry."""


def test_a_safetensors_file_gives_its_tensors(tmp_path):
    state = {"a.weight": torch.arange(6.0).view(2, 3), "b": torch.tensor([1, 2])}
    path = tmp_path / "weights.safetensors"
    save_file(state, str(path))
    assert is_safetensors(path) and is_safetensors(str(path))
    assert not is_safetensors(tmp_path / "weights.pt")
    read = read_state_dict(path)
    assert read.keys() == state.keys()
    assert all(torch.equal(read[k], state[k]) and read[k].dtype == state[k].dtype for k in state)


def test_a_torch_file_gives_its_payload_of_tensors_and_plain_containers(tmp_path):
    payload = {"generator": {"model.a": torch.ones(2, dtype=torch.bfloat16)}, "step": 3}
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    read = read_state_dict(str(path))
    assert read["step"] == 3 and read["generator"]["model.a"].dtype == torch.bfloat16
    assert torch.equal(read["generator"]["model.a"], payload["generator"]["model.a"])
    assert read["generator"]["model.a"].device.type == "cpu"


def test_a_torch_file_of_the_format_before_zip_archives_is_read(tmp_path):
    path = tmp_path / "weights.pth"
    torch.save({"a": torch.arange(3.0)}, path, _use_new_zipfile_serialization=False)
    assert read_state_dict(path)["a"].tolist() == [0.0, 1.0, 2.0]


def test_a_torch_file_with_code_is_refused(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"state": _Code()}, path)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        read_state_dict(path)


def test_indexed_count_counts_the_numbered_submodules():
    state = {"blocks.0.weight": 0, "blocks.11.bias": 0, "dblocks.2.weight": 0, "out.weight": 0}
    assert indexed_count(state, "blocks.") == 12 and indexed_count(state, "dblocks.") == 3
    with pytest.raises(KeyError, match="no 'layers.' keys"):
        indexed_count(state, "layers.")


def test_leading_size_is_the_first_axis_of_a_weight():
    state = {"proj.weight": torch.zeros(5, 3), "proj.bias": torch.zeros(5)}
    assert leading_size(state, "proj.weight") == 5
    with pytest.raises(KeyError, match="no 'inp.weight': they are another module's"):
        leading_size(state, "inp.weight")


def test_check_state_dict_names_what_does_not_fit():
    expected = {"a": torch.zeros(2, 3), "b": torch.zeros(4), "c": torch.zeros(1)}
    assert check_state_dict(expected, expected) == []
    assert check_state_dict(expected, {"b": torch.ones(4)}) == ["a", "c"]  # the keys it lacks
    with pytest.raises(KeyError, match="does not have: \\['d'\\]"):
        check_state_dict(expected, {**expected, "d": torch.zeros(1)})
    with pytest.raises(ValueError, match="shapes differ: \\[\\('a', \\(3, 2\\), \\(2, 3\\)\\)\\]"):
        check_state_dict(expected, {"a": torch.zeros(3, 2)})


def test_module_from_state_assigns_the_tensors_without_a_random_draw():
    state = {"weight": torch.arange(6.0).view(2, 3).to(torch.bfloat16), "bias": torch.ones(2)}
    torch.manual_seed(0)
    before = torch.get_rng_state()
    module = module_from_state(lambda: nn.Linear(3, 2), state)
    assert torch.equal(torch.get_rng_state(), before)
    assert module.weight.dtype == torch.bfloat16 and module.bias.dtype == torch.float32
    assert torch.equal(module.weight, state["weight"]) and torch.equal(module.bias, state["bias"])
    with pytest.raises(KeyError, match="missing keys of Linear: \\['bias'\\]"):
        module_from_state(lambda: nn.Linear(3, 2), {"weight": state["weight"]})
