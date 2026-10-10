"""The generator: its dimensions and config checks, its conditions, a call on the KV cache, the
sampler's adapter and the coarse field of the ablation."""

import math
from types import SimpleNamespace

import pytest
import torch

from tests.modeling.support import (
    FRAME_TOKENS,
    GRID_H,
    GRID_W,
    LATENT_H,
    LATENT_W,
    TINY,
    field_builder,
    tiny_config,
    tiny_generator,
    window_conditions,
)
from worldcast.modeling.state_injector import FIELD_CHANNELS, StateInjectorConfig
from worldcast.modeling.visibility_probe import VisibilityProbeConfig
from worldcast.modeling.wan22.attention import sdpa_attention
from worldcast.modeling.wan22.dit import KVCache
from worldcast.modeling.wan22.model import (
    CallPlan,
    CausalGeneratorAdapter,
    GeneratorConditions,
    GeneratorConfig,
    WorldCastGenerator,
    call_inputs,
    coarsen_field,
    conditions_with_field,
    worldcast_module,
)

#: first frame 0 | memory frames 1-2 | recent context 3-4 | target frames 5-6
WINDOW, TARGET = 7, 5
FIELD_KEYS = field_builder.condition_keys


def _latents(frames: int, seed: int = 9) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, frames, TINY["in_dim"], LATENT_H, LATENT_W, generator=g)


def _cache(generator: WorldCastGenerator, frames: int, **kwargs) -> KVCache:
    return generator.allocate_kv_cache(frames, frame_tokens=FRAME_TOKENS, **kwargs)


def _conditions(generator, **kwargs) -> GeneratorConditions:
    conditions = window_conditions(WINDOW, anchor=TARGET, **kwargs)
    return conditions_with_field(generator, field_builder, conditions)


class _RecordingBuilder:
    """A field builder that records its calls and returns ones."""

    condition_keys = FIELD_KEYS

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, conditions, weapon_embedding, frame_offset, num_frames):
        self.calls.append((conditions, weapon_embedding, frame_offset, num_frames))
        return torch.ones(1, num_frames, FIELD_CHANNELS, GRID_H, GRID_W)


# ------------------------------------------------------------------------------------- architecture
def test_paper_architecture():
    """The default config is Wan2.2-TI2V-5B with the paper's generator's inputs: the 946 tensors
    and 5,098,162,792 parameters of the released four-step file."""
    with torch.device("meta"):
        model = WorldCastGenerator(GeneratorConfig())
    state = model.state_dict()
    assert len(state) == 946
    assert sum(int(math.prod(v.shape)) for v in state.values()) == 5_098_162_792
    assert sum(p.numel() for p in model.ray_embedding.parameters()) == 9_461_760
    assert model.state_injector.stem.weight.shape == (32, 23, 3, 3)
    assert model.controls.net[0].normalized_shape == (900,)
    assert (len(model.blocks), model.state_injector.dit_block) == (30, 1)


def test_shape_with_the_visibility_probe():
    """The generator of stages 2s to 4 adds the probe: 8 tensors, 860,161 parameters."""
    with torch.device("meta"):
        model = WorldCastGenerator(GeneratorConfig(visibility_probe=VisibilityProbeConfig()))
    assert len(model.state_dict()) == 954
    assert sum(p.numel() for p in model.visibility_probe.parameters()) == 860_161
    assert model.visibility_probe.dit_block == 20


def test_config_checks():
    with pytest.raises(ValueError, match="even head_dim"):
        GeneratorConfig(dim=30, num_heads=4)
    for factor in (0, 1.5):
        with pytest.raises(ValueError, match="field_downsample must be a positive integer"):
            GeneratorConfig(field_downsample=factor)
    with pytest.raises(ValueError, match=r"freq_dim must be even .* got 15"):
        GeneratorConfig(freq_dim=15)
    with pytest.raises(TypeError, match="attention must be a kernel"):
        WorldCastGenerator(tiny_config(), attention="sdpa")
    with pytest.raises(ValueError, match="state_injector.dit_block"):
        GeneratorConfig(state_injector=StateInjectorConfig(dit_block=30))
    with pytest.raises(ValueError, match="visibility_probe.dit_block"):
        GeneratorConfig(visibility_probe=VisibilityProbeConfig(dit_block=0))
    with pytest.raises(ValueError, match="visibility_probe.dit_block"):
        GeneratorConfig(visibility_probe=VisibilityProbeConfig(dit_block=30))
    late = GeneratorConfig(state_injector=StateInjectorConfig(dit_block=22))
    assert late.state_injector.dit_block == 22


def test_every_key_belongs_to_the_backbone_or_to_one_conditioning_module():
    generator = tiny_generator(visibility_probe=VisibilityProbeConfig(dit_block=2, hidden=8))
    modules = {key: worldcast_module(key) for key in generator.state_dict()}
    assert set(modules.values()) == {
        None,
        "controls",
        "state_injector",
        "observer_signals",
        "ray_embedding",
        "visibility_probe",
    }
    assert modules["blocks.0.control_adaln.1.weight"] == "controls"
    assert modules["head.control_adaln.3.bias"] == "controls"
    assert modules["blocks.0.self_attn.q.weight"] is None


def test_a_fresh_generator_starts_from_the_backbone():
    """The conditioning starts at zero, so a generator built on a stage's checkpoint starts as that
    stage's generator: the prologue adds nothing, the field's injection is zero and the flow is
    zero (Wan2.2 zero-initialises the head)."""
    torch.manual_seed(0)
    generator = WorldCastGenerator(tiny_config(), attention=sdpa_attention).eval()
    cond = _conditions(generator)
    with torch.no_grad():
        prologue = generator.prologue(cond, (2, GRID_H, GRID_W), TARGET, torch.float32)
        tokens = torch.randn(1, 2 * FRAME_TOKENS, TINY["dim"])
        assert torch.equal(prologue.embed(tokens), tokens)
        assert torch.equal(generator.state_injector(tokens, prologue.field), tokens)
        assert not bool(generator.blocks[0].control_adaln(prologue.control_embedding).any())
        flow = generator(
            _latents(1), torch.zeros(1, 1), cond, kv_cache=_cache(generator, 1), frame_offset=0
        )
    assert flow.shape == (1, 1, TINY["out_dim"], LATENT_H, LATENT_W) and not bool(flow.any())


def test_the_kv_cache_of_a_generator():
    """The cache holds the keys and values of every DiT block, in the dtype and on the device of the
    parameters unless the caller names another."""
    generator = tiny_generator()
    cache = _cache(generator, 5, batch_size=3)
    head_dim = TINY["dim"] // TINY["num_heads"]
    assert len(cache.keys) == len(cache.values) == TINY["num_layers"]
    assert cache.keys[0].shape == (3, 5 * FRAME_TOKENS, TINY["num_heads"], head_dim)
    assert cache.keys[0].dtype == torch.float32 and cache.end == 0
    assert _cache(generator, 1, dtype=torch.bfloat16).values[0].dtype == torch.bfloat16
    assert generator.allocate_kv_cache(1).keys[0].shape[1] == 252  # the paper's token grid


# --------------------------------------------------------------------------------------- conditions
def test_conditions_from_a_condition_dict():
    conditions = window_conditions(WINDOW, anchor=TARGET)
    field = torch.zeros(1, WINDOW, FIELD_CHANNELS, GRID_H, GRID_W)
    kwargs = dict(player_state_field=field, field_keys=FIELD_KEYS)
    cond = GeneratorConditions.from_conditions(conditions, **kwargs)
    assert cond.buttons is conditions["buttons"] and cond.player_state_field is field
    assert cond.rays.anchor_c2w is conditions["ray_anchor_c2w"]
    assert cond.observer_signals.scope_level is conditions["obs_scope_level"]
    with pytest.raises(ValueError, match="no consumer"):
        GeneratorConditions.from_conditions({**conditions, "depth": field}, **kwargs)
    with pytest.raises(ValueError, match="no consumer"):  # the keys of another field builder
        GeneratorConditions.from_conditions(conditions, player_state_field=field)
    with pytest.raises(ValueError, match="missing"):
        GeneratorConditions.from_conditions(
            {k: v for k, v in conditions.items() if k != "view_deltas"}, **kwargs
        )
    with pytest.raises(ValueError, match="cameras arrive together"):
        GeneratorConditions.from_conditions(
            {k: v for k, v in conditions.items() if k != "ray_frame_tans"}, **kwargs
        )
    bare = {k: v for k, v in conditions.items() if not k.startswith(("ray_", "obs_"))}
    cond = GeneratorConditions.from_conditions(bare, **kwargs)
    assert cond.rays is None and cond.observer_signals is None


def test_the_field_is_built_from_the_conditions_and_the_weapon_embedding():
    generator = tiny_generator()
    conditions = window_conditions(WINDOW)
    builder = _RecordingBuilder()
    cond = conditions_with_field(generator, builder, conditions)
    assert builder.calls == []  # built when the generator asks for it
    field = cond.call_field(3, 2)
    assert field.shape == (1, 2, FIELD_CHANNELS, GRID_H, GRID_W)
    built_from, weapon_embedding, frame_offset, num_frames = builder.calls[0]
    assert built_from is conditions and (frame_offset, num_frames) == (3, 2)
    assert weapon_embedding is generator.state_injector.weapon_embedding.weight


def test_a_field_given_for_the_window_is_cut_to_the_call():
    conditions = window_conditions(WINDOW)
    g = torch.Generator().manual_seed(4)
    window_field = torch.randn(1, WINDOW, FIELD_CHANNELS, GRID_H, GRID_W, generator=g)
    cond = GeneratorConditions.from_conditions(
        conditions, player_state_field=window_field, field_keys=FIELD_KEYS
    )
    assert torch.equal(cond.call_field(3, 2), window_field[:, 3:5])
    prologue = tiny_generator().prologue(cond, (2, GRID_H, GRID_W), 3, torch.float32)
    assert torch.equal(prologue.field, window_field[:, 3:5])


def test_a_generator_without_the_state_injector_needs_no_field_builder():
    bare = tiny_generator(state_injector=None)
    conditions = {k: v for k, v in window_conditions(2).items() if k not in FIELD_KEYS}
    assert conditions_with_field(bare, None, conditions).player_state_field is None
    adapter = CausalGeneratorAdapter(bare)
    with torch.no_grad():
        flow = adapter(
            _latents(2),
            torch.full((1, 2), 500.0),
            conditions,
            kv_cache=_cache(bare, 2),
            frame_offset=0,
        )
    assert flow.shape == (1, 2, TINY["out_dim"], LATENT_H, LATENT_W)
    with pytest.raises(ValueError, match="needs a field builder"):
        conditions_with_field(tiny_generator(), None, conditions)


# ------------------------------------------------------------------------------- the KV-cache call
def test_a_call_plan_places_a_call_in_the_window_and_the_cache():
    cache = _cache(tiny_generator(), 4)
    grid = (2, GRID_H, GRID_W)
    assert CallPlan.at(grid, 0, cache) == CallPlan(grid, 0, 0, 2 * FRAME_TOKENS)
    with pytest.raises(ValueError, match="must start there"):
        CallPlan.at(grid, 1, cache)  # the cache is empty: a write starts at 0
    cache.end = 2 * FRAME_TOKENS
    plan = CallPlan.at(grid, 2, cache)
    assert (plan.frame_offset, plan.start, plan.end) == (2, 2 * FRAME_TOKENS, 4 * FRAME_TOKENS)


def test_a_call_writes_its_range_of_the_cache_and_a_repeat_rewrites_it():
    generator = tiny_generator()
    cond = _conditions(generator)
    x, cache = _latents(WINDOW), _cache(generator, WINDOW)
    t = torch.full((1, 2), 16.0)
    with torch.no_grad():
        generator(x[:, :1], t[:, :1], cond, kv_cache=cache, frame_offset=0)
        assert cache.end == FRAME_TOKENS
        flow = generator(x[:, 1:3], t, cond, kv_cache=cache, frame_offset=1)
        assert cache.end == 3 * FRAME_TOKENS and flow.dtype == torch.float32
        keys = cache.keys[0].clone()
        assert bool(keys[:, : 3 * FRAME_TOKENS].any()) and not bool(
            keys[:, 3 * FRAME_TOKENS :].any()
        )
        # a denoising step on a range the cache already holds replaces its keys and values
        again = generator(x[:, 1:3], t, cond, kv_cache=cache, frame_offset=1)
        assert torch.equal(again, flow) and torch.equal(cache.keys[0], keys)
        generator(x[:, 3:5], t, cond, kv_cache=cache, frame_offset=1)
    assert cache.end == 3 * FRAME_TOKENS and not torch.equal(cache.keys[0], keys)


def test_a_call_reads_its_own_frames_of_the_window():
    """The controls, the field, the observer signals and the cameras of the frames a call does not
    cover leave its flow unchanged (the control history reaches 19 video frames back)."""
    generator = tiny_generator()
    x, t = _latents(2), torch.full((1, 2), 500.0)

    def flow(conditions):
        cond = conditions_with_field(generator, field_builder, conditions)
        with torch.no_grad():
            return generator(x, t, cond, kv_cache=_cache(generator, 2), frame_offset=0)

    conditions = window_conditions(WINDOW, anchor=TARGET)
    later = {k: v.clone() for k, v in conditions.items()}
    later["buttons"][:, 5:] = 1 - later["buttons"][:, 5:]  # video frames of latent frame 2 on
    later["player_state_table"][:, 2:] += 50.0
    later["obs_flash_flag"][:, 2:] = 1 - later["obs_flash_flag"][:, 2:]
    later["ray_frame_tans"][:, 2:] += 0.1
    assert torch.equal(flow(later), flow(conditions))
    own = {k: v.clone() for k, v in conditions.items()}
    own["buttons"][:, 4] = 1 - own["buttons"][:, 4]  # the last video frame of latent frame 1
    assert not torch.equal(flow(own), flow(conditions))


def test_a_generator_without_the_optional_modules():
    """Stage 1's generator has no state injector, observer-signal embedding or ray embedding: the
    call needs none of their conditions, and the field builder's entries are left to it."""
    generator = tiny_generator(state_injector=None, observer_signals=None, ray_embedding=False)
    conditions = {
        k: v for k, v in window_conditions(3).items() if not k.startswith(("ray_", "obs_"))
    }
    cond = conditions_with_field(generator, field_builder, conditions)
    with torch.no_grad():
        prologue = generator.prologue(cond, (3, GRID_H, GRID_W), 0, torch.float32)
        flow = generator(
            _latents(3),
            torch.full((1, 3), 16.0),
            cond,
            kv_cache=_cache(generator, 3),
            frame_offset=0,
        )
    assert (prologue.signal_embedding, prologue.ray_embedding, prologue.field) == (None,) * 3
    assert flow.shape == (1, 3, TINY["out_dim"], LATENT_H, LATENT_W) and bool(flow.any())


def test_the_conditions_must_match_the_generators_modules():
    """A condition without its module, and a module without its condition, are errors; only the
    cameras may be absent for a generator with a ray embedding (stage 4)."""
    conditions = window_conditions(3)
    field = torch.zeros(1, 3, FIELD_CHANNELS, GRID_H, GRID_W)
    kwargs = dict(player_state_field=field, field_keys=FIELD_KEYS)
    full = GeneratorConditions.from_conditions(conditions, **kwargs)
    grid = (3, GRID_H, GRID_W)

    def prologue(cond: GeneratorConditions, **overrides):
        return tiny_generator(**overrides).prologue(cond, grid, 0, torch.float32)

    def without(*prefixes: str) -> GeneratorConditions:
        kept = {k: v for k, v in conditions.items() if not k.startswith(prefixes)}
        return GeneratorConditions.from_conditions(kept, **kwargs)

    with pytest.raises(ValueError, match="embeds the observer signals: the conditions need"):
        prologue(without("obs_"))
    with pytest.raises(ValueError, match="arrive together or not at all"):
        without("obs_flash_valid")  # one of the signals
    with pytest.raises(ValueError, match="no field"):
        prologue(GeneratorConditions.from_conditions(conditions, field_keys=FIELD_KEYS))
    with pytest.raises(ValueError, match="no ray embedding"):
        prologue(full, ray_embedding=False)
    with pytest.raises(ValueError, match="embeds none"):
        prologue(full, observer_signals=None)
    with pytest.raises(ValueError, match="no state injector"):
        prologue(full, state_injector=None)
    assert prologue(without("ray_")).ray_embedding is None


def test_a_call_names_the_input_that_does_not_fit():
    generator = tiny_generator()
    cond = _conditions(generator)
    x, cache = _latents(2), _cache(generator, WINDOW)

    def call(noisy=x, timestep=torch.full((1, 2), 500.0), cond=cond):
        return generator(noisy, timestep, cond, kv_cache=cache, frame_offset=0)

    with pytest.raises(ValueError, match=r"timestep must be \[B, 2\]"):
        call(timestep=torch.full((1, 1), 500.0))
    with pytest.raises(ValueError, match="the latents must be torch.float32"):
        call(noisy=x.to(torch.bfloat16))
    short = window_conditions(WINDOW)
    short.update({key: value[:, :1] for key, value in short.items() if key.startswith("obs_")})
    with pytest.raises(ValueError, match="the observer signals must span the window"):
        call(cond=conditions_with_field(generator, field_builder, short))
    short = window_conditions(WINDOW)
    short.update({key: short[key][:, :1] for key in ("ray_frame_c2w", "ray_frame_tans")})
    with pytest.raises(ValueError, match="the cameras cover 1 latent frames"):
        call(cond=conditions_with_field(generator, field_builder, short))
    for prompt in (torch.zeros(1, TINY["text_len"] + 1, TINY["text_dim"]), torch.zeros(1, 5, 4)):
        other_prompt = {**window_conditions(WINDOW), "prompt_embeds": prompt}
        with pytest.raises(ValueError, match=r"every prompt must be \[L <= 7, 24\]"):
            call(cond=conditions_with_field(generator, field_builder, other_prompt))
    with pytest.raises(ValueError, match="starts at a token of the cache, not at -24"):
        generator(x, torch.full((1, 2), 500.0), cond, kv_cache=cache, frame_offset=-1)
    with pytest.raises(TypeError, match="takes GeneratorConditions .* got dict"):
        call(cond=window_conditions(WINDOW))


def test_a_call_refuses_latents_its_patches_do_not_tile():
    """A latent of odd height or width would lose its last row or column to the patch embedding."""
    generator = tiny_generator()
    cond, cache = _conditions(generator), _cache(generator, WINDOW)
    for shape in (
        (1, 2, TINY["in_dim"], LATENT_H + 1, LATENT_W),
        (1, 2, TINY["in_dim"], LATENT_H, LATENT_W - 1),
        (1, 2, TINY["in_dim"] + 1, LATENT_H, LATENT_W),
        (2, TINY["in_dim"], LATENT_H, LATENT_W),
    ):
        with pytest.raises(ValueError, match=r"the latents must be \[B, F, 8, H, W\] with H and W"):
            generator(
                torch.zeros(shape), torch.full((1, 2), 500.0), cond, kv_cache=cache, frame_offset=0
            )


def test_a_call_refuses_inputs_of_different_batches():
    """One sample of timesteps or of a condition would broadcast over the latents' samples."""
    generator = tiny_generator()
    x, t = _latents(2), torch.full((1, 2), 500.0)

    def call(timestep=t, **conditions):
        conditions = {**window_conditions(WINDOW, anchor=TARGET), **conditions}
        cond = conditions_with_field(generator, field_builder, conditions)
        return generator(x, timestep, cond, kv_cache=_cache(generator, WINDOW), frame_offset=0)

    two = window_conditions(WINDOW, batch=2, anchor=TARGET)
    with pytest.raises(ValueError, match="the latents hold 1 samples, the timesteps 2"):
        call(timestep=t.expand(2, 2))
    with pytest.raises(ValueError, match="the timesteps 1 and the controls 2"):
        call(**two)
    with pytest.raises(ValueError, match=r"the controls hold 1 samples; .*'prompt_embeds': 2"):
        call(prompt_embeds=two["prompt_embeds"])
    with pytest.raises(
        ValueError, match="the controls hold 1 samples; .*'the observer signals': 2"
    ):
        call(**{key: value for key, value in two.items() if key.startswith("obs_")})
    with pytest.raises(ValueError, match=r"the controls hold 1 samples; .*'the cameras': 2"):
        call(**{key: value for key, value in two.items() if key.startswith("ray_")})


# ------------------------------------------------------------------------------------- the adapter
def test_the_inputs_are_cast_before_the_field_is_built():
    """The sampler casts only latents and timesteps; ``call_inputs`` casts the conditions too, and
    the field builder sees the cast tensors."""
    conditions = window_conditions(WINDOW, anchor=TARGET)
    builder = _RecordingBuilder()
    weapon_embedding = torch.nn.Embedding(52, 4).to(torch.bfloat16)
    generator = SimpleNamespace(state_injector=SimpleNamespace(weapon_embedding=weapon_embedding))
    x, t = _latents(2), torch.full((1, 2), 937.5)
    cond, noisy, timestep = call_inputs(generator, builder, torch.bfloat16, conditions, x, t)
    cond.call_field(1, 2)
    seen, seen_weapon_embedding, _, _ = builder.calls[0]
    assert torch.equal(noisy, x.to(torch.bfloat16)) and noisy.dtype == torch.bfloat16
    assert timestep.float().unique().tolist() == [936.0]
    assert seen.keys() == conditions.keys()
    assert seen["player_state_table"].dtype == torch.bfloat16
    assert torch.equal(seen["player_state_table"], conditions["player_state_table"].bfloat16())
    assert seen["player_weapons"] is conditions["player_weapons"]  # long: not cast
    assert seen_weapon_embedding is weapon_embedding.weight
    assert cond.view_deltas.dtype == cond.rays.frame_c2w.dtype == torch.bfloat16
    # without an input dtype nothing is cast
    cond, noisy, timestep = call_inputs(generator, builder, None, conditions, x, t)
    assert noisy is x and timestep is t and cond.buttons is conditions["buttons"]


def test_adapter_calls_the_generator_on_the_samplers_cache():
    generator = tiny_generator()
    conditions = window_conditions(WINDOW, anchor=TARGET)
    adapter = CausalGeneratorAdapter(generator, field_builder)
    x, t, cache = _latents(1), torch.full((1, 1), 16.0), _cache(generator, 1)
    with torch.no_grad():
        flow = adapter(x, t, conditions, kv_cache=cache, frame_offset=0)
        cond = conditions_with_field(generator, field_builder, conditions)
        want = generator(x, t, cond, kv_cache=_cache(generator, 1), frame_offset=0)
    assert torch.equal(flow, want) and cache.end == FRAME_TOKENS


# ------------------------------------------------------------------------------- the field ablation
def test_coarsen_field_copies_each_cells_maximum_back():
    field = torch.arange(15, dtype=torch.float32).reshape(1, 1, 1, 3, 5)
    assert coarsen_field(field, 1) is field
    coarse = coarsen_field(field, 2)  # 2 x 2 cells; the last row and column are narrower
    assert coarse[0, 0, 0].tolist() == [
        [6.0, 6.0, 8.0, 8.0, 9.0],
        [6.0, 6.0, 8.0, 8.0, 9.0],
        [11.0, 11.0, 13.0, 13.0, 14.0],
    ]
