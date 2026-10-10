"""Fixtures of the trainer tests (CPU).

* ``one_thread`` (autouse): torch runs single-threaded. The CPU backward of an embedding lookup
  accumulates repeated indices in a thread-dependent order, so with several threads the weapon
  embedding's gradient differs in the last bits from run to run.
* The trainers run under FSDP on the shared ``gloo`` group (``tests/conftest.py``).
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
