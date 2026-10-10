"""The digests of an array: the sha256 of its float32 bytes, and its first 32 hex digits."""

import hashlib

import numpy as np
import torch

from worldcast.utils.fingerprints import fingerprint, float32_array, sha256_float32


def test_known_values():
    # sha256 of sixteen zero bytes, and of the little-endian float32 bytes of [1, 2]
    assert fingerprint(np.zeros(4, np.float32)) == "374708fff7719dd5979ec875d56cd228"
    assert fingerprint(np.array([1.0, 2.0])) == "b9c80b5adeca450753a16950c3cc655d"


def test_the_fingerprint_is_the_head_of_the_sha256():
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    digest = sha256_float32(values)
    assert digest == hashlib.sha256(values.astype(np.float32).tobytes()).hexdigest()
    assert len(digest) == 64 and fingerprint(values) == digest[:32]


def test_float32_array_is_a_view_of_a_contiguous_float32_array():
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert float32_array(values) is values
    strided = float32_array(values[:, ::2])
    assert strided.flags["C_CONTIGUOUS"] and strided.tolist() == [[0.0, 2.0], [3.0, 5.0]]
    assert float32_array(torch.tensor([1.5], dtype=torch.bfloat16)).dtype == np.float32


def test_a_tensor_is_read_as_float32():
    values = torch.tensor([[0.1, -2.5], [3.0, 1e-3]])
    assert fingerprint(values) == fingerprint(values.numpy())
    # a bf16 tensor's fingerprint is that of its rounded values, not of the float32 ones
    assert fingerprint(values.to(torch.bfloat16)) == fingerprint(values.to(torch.bfloat16).float())
    assert fingerprint(values.to(torch.bfloat16)) != fingerprint(values)
