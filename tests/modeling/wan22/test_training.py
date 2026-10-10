"""The training forward: the whole window under an attention mask (bidirectional, or teacher
forcing over ``[context | noisy]``)."""

from types import MappingProxyType

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
    tiny_generator,
    window_conditions,
)
from worldcast.modeling.state_injector import FIELD_CHANNELS
from worldcast.modeling.visibility_probe import VisibilityProbeConfig, VisibilityProbeInputs
from worldcast.modeling.wan22 import training
from worldcast.modeling.wan22.model import conditions_with_field
from worldcast.modeling.wan22.training import TrainingForward, forward_train

#: first frame 0 | two blocks of four latent frames
WINDOW = 9
BLOCKS = [(0, 1), (1, 4), (5, 4)]


def _window(seed: int = 9, batch: int = 1):
    g = torch.Generator().manual_seed(seed)
    shape = (batch, WINDOW, TINY["in_dim"], LATENT_H, LATENT_W)
    noisy, clean = torch.randn(shape, generator=g), torch.randn(shape, generator=g)
    timestep = torch.tensor([[0.0] + [700.0] * 4 + [300.0] * 4] * batch)
    return noisy, clean, timestep, torch.full((batch, WINDOW), 16.0)


def _conditions(generator, *, observer_signals: bool = True, batch: int = 1):
    conditions = window_conditions(WINDOW, batch=batch, anchor=5, seed=4)
    if not observer_signals:
        conditions = {k: v for k, v in conditions.items() if not k.startswith("obs_")}
    return conditions, conditions_with_field(generator, field_builder, conditions)


def test_teacher_forcing_is_the_block_causal_rollout():
    """Under teacher forcing a block's noisy frames attend to the context frames of the blocks
    before it and to themselves: the flow of each block equals the KV-cache call on its noisy
    frames after the context frames before it were written (the field on both copies, no observer
    signals, which the context copy never receives)."""
    generator = tiny_generator(observer_signals=None)
    _, cond = _conditions(generator, observer_signals=False)
    noisy, clean, timestep, context_timestep = _window()
    with torch.no_grad():
        flow = forward_train(
            generator,
            noisy,
            timestep,
            cond,
            context_latents=clean,
            context_timestep=context_timestep,
            field_on_context=True,
        )
        cache = generator.allocate_kv_cache(WINDOW, frame_tokens=FRAME_TOKENS)
        for start, n in BLOCKS:
            frames, at = slice(start, start + n), dict(kv_cache=cache, frame_offset=start)
            block = generator(noisy[:, frames], timestep[:, frames], cond, **at)
            torch.testing.assert_close(block, flow[:, frames], rtol=0, atol=1e-4)
            generator(clean[:, frames], context_timestep[:, frames], cond, **at)  # the context
    assert flow.shape == noisy.shape and flow.dtype == torch.float32


def test_the_bidirectional_window_attends_both_ways():
    """Without a context copy every frame sees every frame: a change of the last frame changes
    the flow of the first; under teacher forcing it does not reach an earlier block."""
    generator = tiny_generator()
    _, cond = _conditions(generator)
    noisy, clean, timestep, _ = _window()
    changed = noisy.clone()
    changed[:, -1] += 1.0
    with torch.no_grad():
        both = [forward_train(generator, x, timestep, cond) for x in (noisy, changed)]
        causal = [
            forward_train(generator, x, timestep, cond, context_latents=clean)
            for x in (noisy, changed)
        ]
    assert not torch.equal(both[0][:, 0], both[1][:, 0])
    assert torch.equal(causal[0][:, :5], causal[1][:, :5])
    assert not torch.equal(causal[0][:, 5:], causal[1][:, 5:])


def test_the_field_reaches_the_context_copy_only_on_request():
    """``field_on_context`` adds the field to the context copy as well: the two forwards differ by
    the injection alone, and are one once its projection is zero."""
    generator = tiny_generator()
    _, cond = _conditions(generator)
    noisy, clean, timestep, _ = _window()

    def flows():
        with torch.no_grad():
            return [
                forward_train(
                    generator, noisy, timestep, cond, context_latents=clean, field_on_context=on
                )
                for on in (False, True)
            ]

    noisy_only, both = flows()
    assert not torch.equal(noisy_only, both)
    for parameter in generator.state_injector.proj.parameters():
        torch.nn.init.zeros_(parameter)
    noisy_only, both = flows()
    assert torch.equal(noisy_only, both)


def test_gradient_checkpointing_changes_no_number():
    generator = tiny_generator().train()
    _, cond = _conditions(generator)
    noisy, clean, timestep, context_timestep = _window()
    results = []
    for enabled in (False, True):
        generator.gradient_checkpointing = enabled
        generator.zero_grad()
        flow = forward_train(
            generator,
            noisy,
            timestep,
            cond,
            context_latents=clean,
            context_timestep=context_timestep,
        )
        flow.square().sum().backward()
        grads = {n: p.grad.clone() for n, p in generator.named_parameters() if p.grad is not None}
        results.append((flow.detach(), grads))
    (flow_a, grads_a), (flow_b, grads_b) = results
    assert torch.equal(flow_a, flow_b) and grads_a.keys() == grads_b.keys()
    assert all(torch.equal(grads_a[name], grads_b[name]) for name in grads_a)
    assert "state_injector.weapon_embedding.weight" in grads_a  # the field is built in the forward


def test_the_visibility_probe_reads_the_window_without_training_the_backbone():
    probe = VisibilityProbeConfig(dit_block=2, hidden=16)
    generator = tiny_generator(visibility_probe=probe).train()
    _, cond = _conditions(generator, batch=2)
    noisy, clean, timestep, _ = _window(batch=2)
    g = torch.Generator().manual_seed(1)
    inputs = VisibilityProbeInputs(
        footprint=torch.rand(2, WINDOW, 4, GRID_H, GRID_W, generator=g),
        depth=torch.rand(2, WINDOW, 4, generator=g) * 500,
        relative_yaw=torch.rand(2, WINDOW, 4, generator=g),
        in_front=torch.ones(2, WINDOW, 4, dtype=torch.bool),
    )
    for context_latents in (None, clean):
        generator.zero_grad()
        flow, logits = forward_train(
            generator, noisy, timestep, cond, context_latents=context_latents, visibility=inputs
        )
        assert flow.shape == noisy.shape and logits.shape == (2, WINDOW, 4)
        logits.sum().backward()
        trained = {n.split(".")[0] for n, p in generator.named_parameters() if p.grad is not None}
        assert trained == {"visibility_probe"}


def test_argument_checks():
    generator = tiny_generator()
    _, cond = _conditions(generator)
    noisy, clean, timestep, context_timestep = _window()
    with pytest.raises(ValueError, match="shape of noisy"):
        forward_train(generator, noisy, timestep, cond, context_latents=clean[:, 1:])
    with pytest.raises(ValueError, match="need context_latents"):
        forward_train(generator, noisy, timestep, cond, context_timestep=context_timestep)
    with pytest.raises(ValueError, match="need context_latents"):
        forward_train(generator, noisy, timestep, cond, field_on_context=True)
    probe = VisibilityProbeInputs(*(torch.zeros(1) for _ in range(4)))
    with pytest.raises(ValueError, match="no visibility probe"):
        forward_train(generator, noisy, timestep, cond, visibility=probe)


def test_training_forward_casts_its_inputs_and_builds_the_field(monkeypatch):
    """With ``input_dtype`` the wrapper casts every floating input itself, the conditions (any
    mapping) and the keyword arguments included, before the field is built."""
    generator = tiny_generator()
    conditions, _ = _conditions(generator)
    conditions = MappingProxyType(conditions)
    noisy, clean, timestep, context_timestep = _window()
    seen = {}

    def recorded(generator, noisy, timestep, cond, **kwargs):
        seen.update(noisy=noisy, timestep=timestep, cond=cond, kwargs=kwargs)
        return noisy.float()

    class Builder:
        condition_keys = field_builder.condition_keys

        def __call__(self, conditions, weapon_embedding, frame_offset, num_frames):
            seen["builder"] = conditions
            return torch.zeros(1, num_frames, FIELD_CHANNELS, GRID_H, GRID_W)

    monkeypatch.setattr(training, "forward_train", recorded)
    TrainingForward(generator, Builder(), input_dtype=torch.bfloat16)(
        noisy,
        timestep,
        conditions,
        context_latents=clean,
        context_timestep=context_timestep,
    )
    seen["cond"].call_field(0, WINDOW)
    bf16 = torch.bfloat16
    assert seen["noisy"].dtype == bf16 and torch.equal(seen["noisy"], noisy.to(bf16))
    assert seen["timestep"].dtype == bf16 and torch.equal(seen["timestep"], timestep.to(bf16))
    assert torch.equal(seen["kwargs"]["context_latents"], clean.to(bf16))
    assert seen["kwargs"]["context_timestep"].dtype == bf16
    assert seen["builder"].keys() == conditions.keys()
    for key, value in conditions.items():
        got = seen["builder"][key]
        want = bf16 if value.is_floating_point() else value.dtype
        assert got.dtype == want and torch.equal(got, value.to(want)), key
    cond = seen["cond"]
    assert cond.view_deltas.dtype == bf16 and cond.weapon.dtype == torch.long
