"""The tests' stand-ins in every Python process that the examples' scripts start in their tests
(``tests/tools/test_examples_scripts.py`` puts this directory on ``PYTHONPATH``).

The synthetic round's tick tables are served without pyarrow (``WORLDCAST_TEST_TICKS``: a pickle of
them by media id), a fingerprint table stands in for the examples' manifest
(``WORLDCAST_TEST_MANIFEST``), and the tests' tiny VAE for the Wan2.2 one: it says the device and
the GPUs its process was given, and fails on the GPU ``WORLDCAST_TEST_FAILING_GPU`` names. A client
says how often it polls the world state. CUDA finds as many GPUs as ``WORLDCAST_TEST_GPUS`` says
(none without it).
"""

import os
import pickle

import pytest
import torch

import worldcast.engine.generator
import worldcast.hub
from tests.engine.inference.support import patch_ticks
from tests.modeling.support import make_tiny_vae
from worldcast.engine.inference.directory import DirectoryWorldState

FAILING_GPU = os.environ.get("WORLDCAST_TEST_FAILING_GPU", "none")
directory_init = DirectoryWorldState.__init__


def load_vae(wan22_root, device):
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"the tiny VAE on {device}, CUDA_VISIBLE_DEVICES={gpus}", flush=True)
    if gpus == FAILING_GPU:
        raise RuntimeError(f"CUDA out of memory on GPU {gpus}")
    return make_tiny_vae()


def polling_init(self, root, *, client, poll_s):
    print(f"{client} polls the world state: poll_s={poll_s}", flush=True)
    directory_init(self, root, client=client, poll_s=poll_s)


with open(os.environ["WORLDCAST_TEST_TICKS"], "rb") as file:
    patch_ticks(pytest.MonkeyPatch(), pickle.load(file))
worldcast.engine.generator.load_vae = load_vae
worldcast.hub.EXAMPLES_MANIFEST = os.environ["WORLDCAST_TEST_MANIFEST"]
DirectoryWorldState.__init__ = polling_init
torch.cuda.device_count = lambda: int(os.environ.get("WORLDCAST_TEST_GPUS", "0"))
