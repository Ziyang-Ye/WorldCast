"""The attention kernels."""

import pytest
import torch

from worldcast.modeling.wan22 import attention


def test_flash_attention_refuses_cpu():
    q = torch.zeros(1, 2, 1, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        attention.flash_attention(q, q, q)
    assert attention.flash_attention_backend() in ("fa3", "fa2", "none")
