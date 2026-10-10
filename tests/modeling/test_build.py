"""Reading the generator's weights and building the inference generator from them."""

import json

import pytest
import torch
from safetensors.torch import save_file

from tests.modeling.support import (
    FRAME_TOKENS,
    LATENT_H,
    LATENT_W,
    TINY,
    field_builder,
    tiny_config,
    tiny_generator,
    window_conditions,
)
from worldcast.modeling.build import (
    build_inference_generator,
    generator_config_from_snapshot,
    load_generator,
    read_generator,
)
from worldcast.modeling.visibility_probe import VisibilityProbeConfig
from worldcast.modeling.wan22.attention import sdpa_attention
from worldcast.modeling.wan22.model import CausalGeneratorAdapter, GeneratorConfig

PROBE = VisibilityProbeConfig(dit_block=2, hidden=8)


def _state(**overrides) -> dict[str, torch.Tensor]:
    return dict(tiny_generator(**overrides).state_dict())


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_a_release_file_holds_the_generators_keys(tmp_path):
    state = _state(visibility_probe=PROBE)
    path = tmp_path / "generator.safetensors"
    save_file(state, str(path))
    assert _same(read_generator(path), state)


def test_a_training_checkpoint_holds_them_under_the_model_root(tmp_path):
    online, ema = _state(seed=3), _state(seed=5)
    path = tmp_path / "checkpoint.pt"
    payload = {
        "generator": {"model." + k: v for k, v in online.items()},
        "generator_ema": {"model." + k: v for k, v in ema.items()},
        "step": 7,
    }
    torch.save(payload, path)
    assert _same(read_generator(path), ema)  # the EMA weights when the file has them
    assert _same(read_generator(path, "generator"), online)
    with pytest.raises(KeyError, match="no 'critic' entry"):
        read_generator(path, "critic")

    torch.save({"generator": payload["generator"]}, path)
    assert _same(read_generator(path), online)
    torch.save({"generator": {**payload["generator"], "step": torch.zeros(1)}}, path)
    with pytest.raises(KeyError, match="outside the 'model.' root"):
        read_generator(path)


def test_the_inference_generator_holds_the_bf16_weights_frozen_and_draws_from_no_rng():
    """On the CPU the generator is float32 and holds the weights as in the paper's runs: rounded to
    bf16; its RoPE table stays complex128."""
    state = _state()
    torch.manual_seed(0)
    before = torch.get_rng_state()
    generator = build_inference_generator(tiny_config(), state)
    assert torch.equal(torch.get_rng_state(), before)
    assert not generator.training and not any(p.requires_grad for p in generator.parameters())
    assert generator.freqs.dtype == torch.complex128
    built = generator.state_dict()
    assert built.keys() == state.keys()
    for key, value in state.items():
        assert built[key].dtype == torch.float32
        assert torch.equal(built[key], value.to(torch.bfloat16).float()), key
    del state["head.modulation"]
    with pytest.raises(KeyError, match="missing keys of WorldCastGenerator: .*head.modulation"):
        build_inference_generator(tiny_config(), state)


def test_the_generator_built_on_the_cpu_runs_there():
    """By default the CPU: float32 parameters and ``sdpa_attention``."""
    generator = build_inference_generator(tiny_config(), _state())
    assert generator.attention is sdpa_attention
    with pytest.raises(ValueError, match="on CUDA or on the CPU"):
        build_inference_generator(tiny_config(), _state(), device="mps")
    adapter = CausalGeneratorAdapter(generator, field_builder)
    noisy = torch.randn(1, 2, TINY["in_dim"], LATENT_H, LATENT_W)
    flow = adapter(
        noisy,
        torch.full((1, 2), 500.0),
        window_conditions(2),
        kv_cache=generator.allocate_kv_cache(2, frame_tokens=FRAME_TOKENS),
        frame_offset=0,
    )
    assert flow.shape == noisy.shape and flow.dtype == torch.float32
    assert bool(torch.isfinite(flow).all())


def test_the_inference_generator_leaves_the_visibility_probe_out():
    state = _state(visibility_probe=PROBE)
    config = tiny_config(visibility_probe=PROBE)
    generator = build_inference_generator(config, state)
    assert generator.visibility_probe is None
    left_out = set(state) - set(generator.state_dict())
    assert (
        left_out == {k for k in state if k.startswith("visibility_probe.")} and len(left_out) == 8
    )


def test_load_generator_reads_the_ema_weights(tmp_path):
    ema = _state(visibility_probe=PROBE)
    path = tmp_path / "checkpoint.pt"
    torch.save({"generator_ema": {"model." + k: v for k, v in ema.items()}}, path)
    generator = load_generator(path, tiny_config())
    assert generator.visibility_probe is None
    weight = generator.state_dict()["patch_embedding.weight"]
    assert torch.equal(weight, ema["patch_embedding.weight"].to(torch.bfloat16).float())
    torch.save({"generator": {}}, path)
    with pytest.raises(KeyError, match="no 'generator_ema' entry"):
        load_generator(path, tiny_config())


def test_load_generator_builds_the_optional_modules_its_file_holds(tmp_path):
    """A generator trained without scene state has no ray embedding and no observer-signal
    embedding, a stage-1 generator no state injector either; the same config loads each."""
    path = tmp_path / "generator.safetensors"
    save_file(_state(observer_signals=None, ray_embedding=False), str(path))
    generator = load_generator(path, tiny_config())
    assert generator.observer_signals is None and generator.ray_embedding is None
    assert generator.state_injector is not None
    save_file(_state(observer_signals=None, ray_embedding=False, state_injector=None), str(path))
    generator = load_generator(path, tiny_config())
    assert generator.state_injector is None
    save_file(_state(), str(path))
    generator = load_generator(path, tiny_config())
    assert None not in (generator.observer_signals, generator.ray_embedding)


def test_generator_config_from_a_wan22_snapshot(tmp_path):
    path = tmp_path / "config.json"
    snapshot = {"_class_name": "WanModel", "model_type": "ti2v", "patch_size": [1, 2, 2], **TINY}
    path.write_text(json.dumps(snapshot))
    config = generator_config_from_snapshot(path)
    assert (config.dim, config.num_layers, config.ffn_dim) == (32, 4, 64)
    assert (config.text_dim, config.text_len) == (24, 7)
    assert config == GeneratorConfig(**TINY)  # the other fields are the paper's
    path.write_text(json.dumps({**snapshot, "num_frames": 81}))
    with pytest.raises(ValueError, match="unexpected config keys \\['num_frames'\\]"):
        generator_config_from_snapshot(path)
    for key, value in (("model_type", "t2v"), ("patch_size", [1, 4, 4]), ("qk_norm", False)):
        path.write_text(json.dumps({**snapshot, key: value}))
        with pytest.raises(ValueError, match=f"the generator fixes .* not .*{key}"):
            generator_config_from_snapshot(path)
