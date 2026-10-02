"""The sampler built from the config."""

import pytest
import torch

from tests.sampling.fakes import FakeCore, NewGenerator
from worldcast import sampling as S
from worldcast.config.inference import InferenceConfig
from worldcast.modeling.wan22.model import KVCache


def test_sampler_from_config():
    cfg = InferenceConfig()
    core = FakeCore()
    cache = KVCache.allocate(
        num_blocks=1,
        num_heads=2,
        head_dim=3,
        capacity_latents=41,
        frame_seq_length=cfg.model.frame_seq_length,
    )
    sampler = S.Sampler.from_config(cfg, NewGenerator(core), cache)
    assert sampler.ladder.tolist() == pytest.approx([1000.0, 937.5, 833.3333, 625.0], abs=1e-3)
    assert (sampler.context_noise, sampler.frame_seq_length) == (16, 252)
    wrong = NewGenerator(core)
    wrong.input_dtype = torch.float32
    with pytest.raises(ValueError):
        S.Sampler.from_config(cfg, wrong, cache)
