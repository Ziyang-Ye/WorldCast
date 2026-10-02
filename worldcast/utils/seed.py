"""Seeding of python, numpy and torch."""

import random

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: int) -> None:
    """Seed ``random``, numpy and torch on the CPU and every CUDA device."""
    # Attribution: these four seeding calls follow CausVid's set_seed (github.com/tianweiy/CausVid
    # at fab2440f, MIT) via Self Forcing (github.com/guandeh17/Self-Forcing, Apache-2.0).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
