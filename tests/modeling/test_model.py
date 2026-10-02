"""The block-causal DiT: its dimensions, its conditions and the input cast of the causal adapter."""

import math

import pytest
import torch

from worldcast.modeling.wan22.model import CausalGeneratorAdapter, GeneratorConditions, KVCache

WINDOW = 7  # sink 1 | slot 2 | recent 2 | target 2
SLOTS = [1, 2]
TARGET = 5
LATENT_H, LATENT_W = 8, 12  # token grid 4 x 6


#: (first window frame, frames, timestep): the context pass writes at t = 16, then the 3-rung ladder
#: on the target
CALLS = [
    (0, 1, 16.0),
    (1, 2, 16.0),
    (3, 2, 16.0),
    (TARGET, 2, 1000.0),
    (TARGET, 2, 937.5),
    (TARGET, 2, 833.25),
]


def _window_inputs(tiny, batch: int = 2, players: int = 4, seed: int = 21):
    g = torch.Generator().manual_seed(seed)
    rows = 1 + 4 * (WINDOW - 1)
    # peers in front of the observer (slot 0 at the origin, looking along +x), all alive, labels
    # random
    states = torch.zeros(batch, WINDOW, players, 6)
    states[..., 0] = torch.rand(batch, 1, players, generator=g) * 400 + 150
    states[..., 1] = (torch.rand(batch, 1, players, generator=g) - 0.5) * 300
    states[..., 0] += torch.arange(WINDOW)[None, :, None] * 5.0
    states[:, :, 0, :3] = 0.0
    states[..., 3] = (torch.rand(batch, WINDOW, players, generator=g) - 0.5) * 360
    states[..., 5] = 1.0
    actions = torch.zeros(batch, WINDOW, players, 16, 14)
    actions[..., :11] = torch.randint(0, 2, (batch, WINDOW, players, 16, 11), generator=g).float()
    actions[..., -3:-1] = (
        torch.rand(batch, WINDOW, players, 16, 2, generator=g) - 0.5
    ) * 0.2  # pitch, yaw
    actions[..., -1] = 1.0
    peer = dict(
        peer_states=states,
        peer_actions=actions,
        peer_observer_slot=torch.zeros(batch, dtype=torch.long),
        peer_team_ids=torch.tensor([[0, 0, 1, 1]] * batch),
        peer_weapons=torch.randint(0, 52, (batch, WINDOW, players), generator=g),
    )
    alive = torch.ones(batch, WINDOW, players)
    visible = torch.randint(0, 2, (batch, WINDOW, players), generator=g).float()
    visible[:, :, 1] = 1.0  # at least one peer always written

    frame_c2w = tiny.random_c2w(batch, WINDOW, seed=seed + 1)
    shared = dict(
        prompt_embeds=torch.randn(batch, 5, tiny.TINY["text_dim"], generator=g),
        button_condition=torch.randint(0, 2, (batch, rows, 11), generator=g).float(),
        camera_condition=torch.randn(batch, rows, 2, generator=g),
        weapon_condition=torch.randint(0, 52, (batch, rows), generator=g),
        obs_flash_flag=torch.randint(0, 2, (batch, WINDOW), generator=g),
        obs_flash_valid=torch.randint(0, 2, (batch, WINDOW), generator=g),
        obs_scope_on=torch.randint(0, 2, (batch, WINDOW), generator=g),
        obs_scope_level=torch.randint(0, 3, (batch, WINDOW), generator=g),
        obs_scope_valid=torch.randint(0, 2, (batch, WINDOW), generator=g),
        state_wp_frame_c2w=frame_c2w,
        state_wp_frame_tans=torch.rand(batch, WINDOW, 2, generator=g) + 0.5,
        state_wp_anchor_c2w=frame_c2w[:, TARGET],
        state_wp_memory_c2w=frame_c2w[:, SLOTS],
        state_wp_memory_frames=torch.tensor([SLOTS] * batch),
    )
    noise = torch.randn(len(CALLS), batch, 2, tiny.TINY["in_dim"], LATENT_H, LATENT_W, generator=g)
    return peer, alive, visible, shared, noise


def test_from_dict_rejects_unknown_and_partial_keys(tiny):
    _, _, _, shared, _ = _window_inputs(tiny)
    field = torch.zeros(2, WINDOW, 23, 4, 6)
    with pytest.raises(ValueError, match="no consumer"):
        GeneratorConditions.from_dict({**shared, "state_memory_latents": field}, state_field=field)
    partial = {k: v for k, v in shared.items() if k != "state_wp_frame_tans"}
    with pytest.raises(ValueError, match="together"):
        GeneratorConditions.from_dict(partial, state_field=field)
    without_rays = {k: v for k, v in shared.items() if not k.startswith("state_wp_")}
    assert GeneratorConditions.from_dict(without_rays, state_field=field).rays is None


def test_paper_dimensions():
    """The default config is Wan2.2-TI2V-5B + the deploy config; 948 checkpoint tensors minus the 2
    dead ones."""
    from worldcast.modeling.wan22.model import GeneratorConfig, WorldCastGenerator

    with torch.device("meta"):
        model = WorldCastGenerator(GeneratorConfig())
    state = model.state_dict()
    assert len(state) == 956 - 8 - 2
    params = sum(int(math.prod(v.shape)) for v in state.values())
    assert (
        sum(
            int(math.prod(model.rays.ray_mlp[i].weight.shape)) + model.rays.ray_mlp[i].bias.numel()
            for i in (0, 2)
        )
        == 9_461_760
    )
    assert model.state_injector.stem.weight.shape == (32, 23, 3, 3)
    assert model.action.net[0].normalized_shape == (900,)
    # the checkpoint's 5,099,022,955 parameters minus the visibility head (LayerNorm 2 x 3072,
    # Linear(3072 + 5, 256), Linear(256, 256), Linear(256, 1) = 860,161) and the two dead scalars
    assert params == 5_099_022_955 - 860_161 - 2


def test_adapter_applies_the_root_cast_before_the_field_build(tiny):
    """The sampler casts only latents and timesteps; the adapter casts the conditions as the FSDP
    root did (torch's ``_cast_forward_inputs``), and the field builder sees the cast peer
    tensors."""
    from torch.distributed.utils import _cast_forward_inputs

    peer, alive, visible, shared, noise = _window_inputs(tiny)
    legacy = {
        **shared,
        **peer,
        "peer_alive": alive[:, 1:3],
        "peer_visible": visible[:, 1:3],
    }
    seen = {}

    class StubGenerator:
        weapon_embedding = torch.nn.Embedding(52, 4).to(torch.bfloat16)

        def __call__(self, noisy, timestep, cond, *, kv_cache, current_start, cache_start):
            seen.update(noisy=noisy, timestep=timestep, cond=cond, kv_cache=kv_cache)
            seen["starts"] = (current_start, cache_start)
            seen["field"] = cond.state_field(1, 2)
            return torch.zeros(noisy.shape, dtype=torch.float32)

    def field_builder(conditions, weapon_weight, frame_offset, num_frames):
        seen["builder"] = (conditions, weapon_weight, frame_offset, num_frames)
        return torch.zeros(2, num_frames, 23, 4, 6)

    generator = StubGenerator()
    adapter = CausalGeneratorAdapter(generator, field_builder)
    cache = KVCache([torch.zeros(1, dtype=torch.bfloat16)], [torch.zeros(1, dtype=torch.bfloat16)])
    x = noise[0, :, :2]
    t = torch.full((2, 2), 937.5)
    adapter(x, t, legacy, kv_cache=cache, current_start=252, cache_start=252)

    (ref_x, ref_t, ref_cond), _ = _cast_forward_inputs(torch.bfloat16, x, t, legacy)
    assert torch.equal(seen["noisy"], ref_x) and seen["noisy"].dtype == torch.bfloat16
    assert seen["timestep"].float().unique().tolist() == [936.0]
    conditions, weapon_weight, frame_offset, num_frames = seen["builder"]
    assert conditions.keys() == ref_cond.keys()
    for key, value in ref_cond.items():
        assert conditions[key].dtype == value.dtype and torch.equal(conditions[key], value), key
    assert (
        conditions["peer_states"].dtype == torch.bfloat16
        and conditions["peer_weapons"].dtype == torch.long
    )
    assert weapon_weight is generator.weapon_embedding.weight and (frame_offset, num_frames) == (
        1,
        2,
    )
    cond = seen["cond"]
    assert cond.camera.dtype == torch.bfloat16 and cond.rays.frame_c2w.dtype == torch.bfloat16
    assert torch.equal(cond.rays.memory_frames, legacy["state_wp_memory_frames"])
    assert seen["kv_cache"] is cache and seen["starts"] == (252, 252)
