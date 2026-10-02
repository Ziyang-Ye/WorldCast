"""The length of a client's window."""

import pytest


def test_window_length_matches_memory_deploy():
    """``client_latents`` is ``memory_deploy.py:1119-1125``: clip to coverage and max_blocks, round to 1 + 4k."""
    from worldcast.config.inference import InferenceConfig, with_overrides
    from worldcast.engine.inference.client import client_latents

    cfg = InferenceConfig()

    def old_rule(n_req, covered, max_blocks, first=25):
        n = min(n_req, 1 + max(0, covered - 1) // 4)
        if max_blocks:
            n = min(n, first + 4 * max_blocks)
        n = 1 + ((n - 1) // 4) * 4
        if n < first + 4:
            raise ValueError
        return n

    for covered in (0, 100, 113, 116, 117, 120, 1761, 5000):
        for max_blocks in (0, 1, 3):
            c = with_overrides(cfg, {"run.max_blocks": max_blocks})
            try:
                want = old_rule(441, covered, max_blocks)
            except ValueError:
                with pytest.raises(ValueError):
                    client_latents(c, covered)
                continue
            assert client_latents(c, covered) == want, (covered, max_blocks)
