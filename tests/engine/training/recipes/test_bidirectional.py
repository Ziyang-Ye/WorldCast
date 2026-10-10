"""Stage 1 on a tiny model (CPU): the initialisation from a Wan2.2 backbone (the controls fresh),
one step on raw frames through a stand-in VAE, and the stage-1 rules (one global clip, no EMA, no
WorldCast module built)."""

import math

import pytest
import torch

from tests.engine.training.support import TINY_DIMS, FixedBatches
from tests.modeling.support import randomize_
from worldcast import distributed as D
from worldcast.config.training import paper_config
from worldcast.engine.optim import build_optimizer
from worldcast.engine.stage import field_config, generator_config, worldcast_module
from worldcast.engine.training.build import build_training_generator
from worldcast.engine.training.recipes.bidirectional import BidirectionalTrainer
from worldcast.engine.training.recipes.flow_matching import (
    GeneratorForward,
    TrainerParts,
    encode_raw_frames,
)
from worldcast.engine.training.recipes.teacher_forcing import TeacherForcingTrainer
from worldcast.modeling.wan22.attention import sdpa_attention
from worldcast.modeling.wan22.model import WorldCastGenerator


class StandInVAE:
    """``encode(pixels [1, 3, T, 384, 672], scale) -> [1, 48, F, 24, 42]``: a fixed linear map."""

    def encode(self, x, scale):
        pooled = torch.nn.functional.avg_pool3d(x, kernel_size=(1, 16, 16))  # [1, 3, T, 24, 42]
        frames = torch.cat(
            [pooled[:, :, :1], pooled[:, :, 1:].unflatten(2, (-1, 4)).mean(3)], dim=2
        )
        return frames.repeat(1, 16, 1, 1, 1)  # [1, 48, F, 24, 42]


def _parts(cfg) -> tuple[TrainerParts, WorldCastGenerator]:
    """Stage-1 parts on a generator initialised from a backbone alone."""
    config = generator_config(cfg)
    torch.manual_seed(0)
    full = randomize_(WorldCastGenerator(config), 1)
    backbone = {k: v for k, v in full.state_dict().items() if worldcast_module(k) != "controls"}
    generator, fresh = build_training_generator(
        config, backbone, new_modules=cfg.stage.new_modules, attention=sdpa_attention
    )
    assert fresh == ["controls"]
    assert all(torch.equal(generator.state_dict()[k], v) for k, v in backbone.items())
    forward = GeneratorForward(generator, field_config(cfg, bidirectional=True))
    g = torch.Generator().manual_seed(2)
    rows = 1 + 4 * (21 - 1)
    batch = {
        "frames": torch.rand(1, 3, rows, 384, 672, generator=g) * 2 - 1,
        "buttons": (torch.rand(1, rows, 11, generator=g) < 0.3).float(),
        "view_deltas": torch.randn(1, rows, 2, generator=g) * 0.1,
        "weapon": torch.randint(0, 52, (1, rows), generator=g),
    }
    parts = TrainerParts(
        generator=forward,
        optimizer=build_optimizer(forward.named_parameters(), cfg.optim.lr),
        data=FixedBatches([batch]),
        prompt_embeds=torch.randn(1, 5, TINY_DIMS["text_dim"], generator=g),
        info=D.DistInfo(),
        vae=StandInVAE(),
    )
    return parts, generator


def test_stage1_initialisation_and_step():
    cfg = paper_config("1", {"model.dims": TINY_DIMS, "data.batch_size": 1})
    config = generator_config(cfg)
    assert config.state_injector is None and not config.observer_signals
    assert not config.ray_embedding and config.visibility_probe is None
    parts, generator = _parts(cfg)
    assert [g["name"] for g in parts.optimizer.param_groups] == ["backbone"]
    trainer = BidirectionalTrainer(cfg, parts)
    before = {n: p.detach().clone() for n, p in generator.named_parameters()}
    metrics = trainer.train_step()
    assert trainer.step == 1 and math.isfinite(metrics["loss"]) and metrics["loss"] > 0
    assert metrics["loss"] == metrics["flow_loss"] and metrics["memory_window"] == 0.0
    # one global clip (no group norms), the backbone's learning rate, no EMA
    assert metrics["grad_norm"] > 0 and not any(k.startswith("grad_norm_") for k in metrics)
    assert metrics["lr_backbone"] == 1e-5 and trainer.ema is None
    changed = {n for n, p in generator.named_parameters() if not torch.equal(p, before[n])}
    assert changed


def test_a_recipe_trains_the_stages_of_its_attention():
    stage3 = paper_config("3", {"model.dims": TINY_DIMS})
    with pytest.raises(ValueError, match="stage 3 trains by teacher_forcing, not with Bidirect"):
        BidirectionalTrainer(stage3, None)
    stage2 = paper_config("2", {"model.dims": TINY_DIMS})
    with pytest.raises(ValueError, match="stage 2 trains by bidirectional, not with TeacherForc"):
        TeacherForcingTrainer(stage2, None)


def test_raw_frames_are_encoded_one_sample_at_a_time():
    frames = torch.rand(2, 3, 5, 32, 32, generator=torch.Generator().manual_seed(0))
    latents = encode_raw_frames(
        StandInVAE(), frames, device=torch.device("cpu"), dtype=torch.float32
    )
    assert latents.shape == (2, 2, 48, 2, 2) and latents.dtype == torch.float32
    # latent frame 0 is video frame 0, pooled 16 x 16; channel c repeats the three colours
    assert torch.allclose(latents[1, 0, 4], frames[1, 1, 0].reshape(2, 16, 2, 16).mean((1, 3)))
