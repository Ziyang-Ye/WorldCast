"""Fakes of the sampler tests: a deterministic causal-DiT core, a text encoder and conditions."""

from typing import Any

import torch
from torch.distributed.utils import _cast_forward_inputs

from worldcast.modeling.wan22.model import KVCache

BF16 = torch.bfloat16


def _update_kv(cache: dict[str, Any], key: torch.Tensor, value: torch.Tensor, cache_start: int):
    """Write into one block's cache dict with end-index tensors (nothing is evicted here)."""
    capacity = cache["k"].shape[1]
    n = key.shape[1]
    cache_end = int(cache_start) + n
    global_end = int(cache["global_end_index"].item())
    local_end = int(cache["local_end_index"].item())
    if cache_end > global_end:
        assert int(cache_start) == global_end, "KV cache updates must be contiguous when advancing"
        assert local_end + n <= capacity
        local_start = local_end
        local_end += n
    else:
        local_end += cache_end - global_end
        local_start = local_end - n
        assert local_start >= 0
    cache["k"][:, local_start:local_end] = key
    cache["v"][:, local_start:local_end] = value
    cache["global_end_index"].fill_(cache_end)
    cache["local_end_index"].fill_(local_end)
    return cache["k"][:, :local_end], cache["v"][:, :local_end]


class FakeCore:
    def __init__(self, *, heads: int = 2, head_dim: int = 3, patch=(2, 2)):
        self.heads, self.head_dim, self.patch = heads, head_dim, patch
        self.records: list[dict[str, Any]] = []

    def __call__(self, noisy, timestep, conditions, kv_cache, current_start, cache_start):
        """``kv_cache``: a :class:`KVCache`, or a list of per-block cache dicts."""
        b, f, c, h, w = noisy.shape
        ph, pw = self.patch
        tokens = (
            noisy.float().reshape(b, f, c, h // ph, ph, w // pw, pw).permute(0, 1, 3, 5, 2, 4, 6)
        )
        tokens = tokens.reshape(b, f * (h // ph) * (w // pw), c * ph * pw)
        need = self.heads * self.head_dim
        reps = (need + tokens.shape[-1] - 1) // tokens.shape[-1]
        base = tokens.repeat(1, 1, reps)[..., :need]
        t_term = (
            timestep.float().repeat_interleave((h // ph) * (w // pw), dim=1)[..., None] / 1000.0
        )
        attended = []
        new = isinstance(kv_cache, KVCache)
        if new:
            blocks = len(kv_cache.keys)
            cache_in = [(kv_cache.end, kv_cache.end)] * blocks
            end = kv_cache.span(int(cache_start), base.shape[1])
        else:
            blocks = len(kv_cache)
            cache_in = [
                (int(c["global_end_index"].item()), int(c["local_end_index"].item()))
                for c in kv_cache
            ]
        for i in range(blocks):
            key = (torch.sin(base * (i + 1)) + t_term).reshape(b, -1, self.heads, self.head_dim)
            value = (torch.cos(base * (i + 2)) - t_term).reshape(b, -1, self.heads, self.head_dim)
            if new:
                dtype = kv_cache.keys[i].dtype
                k_all, v_all = kv_cache.update(
                    i, key.to(dtype), value.to(dtype), int(cache_start), end
                )
            else:
                cache = kv_cache[i]
                k_all, v_all = _update_kv(
                    cache, key.to(cache["k"].dtype), value.to(cache["v"].dtype), cache_start
                )
            attended.append(k_all.float().mean() + 0.5 * v_all.float().mean())
        if new:
            kv_cache.end = end
        context = torch.stack(attended).sum()
        cond_term = torch.zeros(())
        summary = []
        for name in sorted(conditions):
            value = conditions[name]
            if torch.is_tensor(value):
                if value.numel():
                    cond_term = cond_term + value.float().mean() * (1 + len(summary) % 3)
                summary.append(
                    (name, tuple(value.shape), str(value.dtype), float(value.float().sum()))
                )
            else:
                summary.append((name, repr(value)))
        flow = torch.tanh(
            noisy.float() * 0.7
            + timestep.float()[..., None, None, None] / 1000.0
            + 0.1 * context
            + 0.01 * cond_term
        )
        assert bool(torch.isfinite(flow).all()), "the fake generator must stay finite"
        self.records.append(
            dict(
                x=noisy.detach().clone(),
                t=timestep.detach().clone(),
                current_start=int(current_start),
                cache_start=int(cache_start),
                cache_in=cache_in,
                context=float(context),
                conditions=summary,
            )
        )
        return flow.float()


class NewGenerator:
    """The release ``CausalGenerator`` protocol over the same core, with the same bf16 input
    cast."""

    input_dtype = BF16

    def __init__(self, core: FakeCore):
        self.core = core

    def __call__(self, noisy, timestep, conditions, *, kv_cache, current_start, cache_start):
        (noisy, timestep, conditions), _ = _cast_forward_inputs(
            BF16, noisy, timestep, dict(conditions)
        )
        return self.core(noisy, timestep, conditions, kv_cache, current_start, cache_start)


class FakeTextEncoder:
    def __init__(self, text_len: int = 5, dim: int = 6):
        self.text_len, self.dim = text_len, dim

    def __call__(self, text_prompts):
        b = len(text_prompts)
        base = torch.arange(b * self.text_len * self.dim, dtype=torch.float32).reshape(
            b, self.text_len, self.dim
        )
        return {"prompt_embeds": torch.sin(base / 7.0)}


def make_conditions(
    frames: int, *, players: int = 4, seed: int = 0, peer_width: int = 6
) -> dict[str, torch.Tensor]:
    """Random generator conditions of a window / round of ``frames`` latents (pixel rows 1 +
    4(frames-1))."""
    g = torch.Generator().manual_seed(seed)
    rows = 1 + 4 * (frames - 1)
    randn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return {
        "button_condition": (torch.rand(1, rows, 11, generator=g) > 0.5).float(),
        "camera_condition": randn(1, rows, 2),
        "weapon_condition": torch.randint(0, 52, (1, rows), generator=g),
        "peer_states": randn(1, frames, players, peer_width),
        "peer_actions": randn(1, frames, players, 16, 14),
        "peer_observer_slot": torch.tensor([1]),
        "peer_team_ids": torch.randint(0, 2, (1, players), generator=g),
        "peer_alive": torch.rand(1, frames, players, generator=g) > 0.2,
        "peer_visible": torch.rand(1, frames, players, generator=g) > 0.5,
        "peer_weapons": torch.randint(0, 52, (1, frames, players), generator=g),
        "state_wp_memory_c2w": randn(1, 0, 4, 4),
        "state_wp_anchor_c2w": randn(1, 4, 4),
        "state_wp_memory_frames": torch.zeros(1, 0, dtype=torch.long),
        "state_wp_frame_c2w": randn(1, frames, 4, 4),
        "state_wp_frame_tans": torch.rand(1, frames, 2, generator=g) + 0.5,
        "obs_flash_flag": randn(1, frames),
        "obs_flash_valid": torch.ones(1, frames),
        "obs_scope_on": torch.zeros(1, frames),
        "obs_scope_level": torch.zeros(1, frames, dtype=torch.long),
        "obs_scope_valid": torch.ones(1, frames),
    }


def assert_same(a, b, where: str = "") -> None:
    """Exact structural equality: tensors by ``torch.equal`` and dtype, mappings by key, sequences
    by item."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        assert torch.is_tensor(a) and torch.is_tensor(b), where
        assert a.dtype == b.dtype, f"{where}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{where}: shape {tuple(a.shape)} != {tuple(b.shape)}"
        assert torch.equal(a, b), f"{where}: values differ"
        if a.is_floating_point():
            assert bool(
                torch.isfinite(a).all()
            ), f"{where}: non-finite values (equal, but the check is void)"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{where}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            assert_same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f"{where}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"
