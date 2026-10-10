"""The attention kernels, their names, and the attention masks of training."""

import math

import pytest
import torch

from worldcast.config.inference import ModelConfig
from worldcast.config.training import TrainModelConfig
from worldcast.modeling.wan22 import attention
from worldcast.modeling.wan22.attention import MaskLayout


def _qkv(length: int, heads: int = 2, head_dim: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, length, heads, head_dim, generator=g) for _ in range(3))


# ------------------------------------------------------------------------------------------ kernels
def test_sdpa_attention_is_softmax_of_the_scaled_products():
    q, k, v = _qkv(5)
    weights = torch.softmax(torch.einsum("bqnd,bknd->bnqk", q, k) / math.sqrt(4), dim=-1)
    want = torch.einsum("bnqk,bknd->bqnd", weights, v)
    out = attention.sdpa_attention(q.to(torch.float64), k, v)
    assert out.dtype == torch.float64  # the dtype of q, computed in that of v
    torch.testing.assert_close(out.float(), want, rtol=1e-5, atol=1e-6)


def test_flash_attention_refuses_cpu():
    q = torch.zeros(1, 2, 1, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="needs CUDA"):
        attention.flash_attention(q, q, q)


def test_optional_kernels_are_found_by_module_name(monkeypatch):
    """``flash_attention`` is flash-attention 2 alone; a kernel whose module is missing says which
    module it needs."""
    found = {("flash_attn",): object(), ("flash_attn_interface", "flash_attn3_hub"): None}
    monkeypatch.setattr(attention, "_first_module", found.__getitem__)
    assert attention.flash_attention_backend() == "fa2" and not attention.fa3_available()
    q = torch.zeros(1, 2, 1, 8)
    with pytest.raises(RuntimeError, match="flash_attn_interface or flash_attn3_hub is not inst"):
        attention.fa3_attention(q, q, q)
    found[("flash_attn",)] = None
    assert attention.flash_attention_backend() == "none"


def test_kernels_by_the_name_of_the_setting():
    """On CUDA the kernel the setting names; elsewhere SDPA, the one kernel that runs there. The
    setting defaults to the paper's kernel."""
    assert attention.ATTENTION_KERNELS[attention.PAPER_ATTENTION] is attention.flash_attention
    assert ModelConfig().attention == TrainModelConfig().attention == attention.PAPER_ATTENTION
    assert attention.attention_kernel("flash", "cuda") is attention.flash_attention
    assert attention.attention_kernel("fa3", torch.device("cuda:1")) is attention.fa3_attention
    assert attention.attention_kernel("sdpa", "cuda") is attention.sdpa_attention
    for name in ("flash", "fa3", "sdpa"):
        assert attention.attention_kernel(name, "cpu") is attention.sdpa_attention
    with pytest.raises(
        ValueError, match="unknown attention kernel 'nope' \\(flash \\| fa3 \\| sdpa"
    ):
        attention.attention_kernel("nope", "cpu")


# -------------------------------------------------------------------------------------------- masks
def test_causal_blocks_are_the_first_frame_and_blocks_of_four():
    assert attention.causal_blocks(9) == [(0, 1), (1, 5), (5, 9)]
    assert attention.causal_blocks(7) == [(0, 1), (1, 5), (5, 7)]  # a last block that is not full
    assert attention.causal_blocks(1) == [(0, 1)]
    with pytest.raises(ValueError, match="at least one latent frame"):
        attention.causal_blocks(0)


def test_layouts_of_the_stages():
    bidirectional = MaskLayout(21, 252, teacher_forcing=False)
    assert bidirectional.blocks == [(0, 21)] and bidirectional.total_length == 5292
    teacher_forcing = MaskLayout(21, 252, teacher_forcing=True)
    assert teacher_forcing.blocks == [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 21)]
    assert teacher_forcing.total_length == 2 * 5292


def test_bidirectional_mask_is_full():
    assert bool(attention.dense_attention_mask(MaskLayout(3, 2, teacher_forcing=False)).all())


def test_teacher_forcing_structure():
    """What the stage-3 mask means: context block g sees context blocks <= g; noisy block g sees
    context blocks < g and its own noisy block."""
    layout = MaskLayout(9, 1, teacher_forcing=True)  # blocks (0,1) (1,5) (5,9)
    mask = attention.dense_attention_mask(layout)
    context, noisy = slice(0, 9), slice(9, 18)
    block = torch.tensor([0, 1, 1, 1, 1, 2, 2, 2, 2])
    assert torch.equal(mask[context, context], block[None, :] <= block[:, None])
    assert torch.equal(mask[noisy, context], block[None, :] < block[:, None])
    assert torch.equal(mask[noisy, noisy], block[None, :] == block[:, None])
    assert not bool(mask[context, noisy].any())


def test_teacher_forcing_intervals():
    """A query's own interval, the context before it and the query itself: frames (0) (1 2) of two
    tokens each."""
    starts, ends, context_ends = attention.attention_intervals(
        MaskLayout(3, 2, teacher_forcing=True)
    )
    assert starts.tolist() == [0, 0, 0, 0, 0, 0, 6, 6, 8, 8, 8, 8]
    assert ends.tolist() == [2, 2, 6, 6, 6, 6, 8, 8, 12, 12, 12, 12]
    assert context_ends.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]


def test_masked_attention_ignores_the_masked_keys():
    layout = MaskLayout(3, 2, teacher_forcing=True)
    mask = attention.dense_attention_mask(layout)
    q, k, v = _qkv(layout.total_length)
    out = attention.masked_attention(q, k, v, mask)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, 2:6] += 1.0  # the second context block: unseen by the first frame's context tokens
    changed_v[:, 2:6] += 1.0
    again = attention.masked_attention(q, changed_k, changed_v, mask)
    assert torch.equal(again[:, :2], out[:, :2]) and not torch.equal(again[:, 2:6], out[:, 2:6])
    assert torch.equal(again[:, 6:], out[:, 6:])  # the noisy copies see the first frame only


@pytest.mark.skipif(not attention.FLEX_ATTENTION_AVAILABLE, reason="needs torch >= 2.5")
def test_the_flex_mask_is_the_dense_mask():
    """Every entry of the flex ``BlockMask`` equals the dense mask's, a padded row attends to
    itself alone, and the two kernels agree."""
    layout = MaskLayout(3, 2, teacher_forcing=True)
    block_mask = attention.flex_block_mask(layout, "cpu")
    assert tuple(block_mask.shape[-2:]) == (128, 128)  # padded to flex attention's tile
    dense = attention.dense_attention_mask(layout)
    index = torch.arange(128)
    flex = block_mask.mask_mod(0, 0, index[:, None], index[None, :])
    assert torch.equal(flex[:12, :12], dense)
    assert torch.equal(flex[12:], (index[12:, None] == index[None, :]))
    q, k, v = _qkv(layout.total_length)
    out = attention.masked_flex_attention(q, k, v, block_mask, compiled=False)
    want = attention.masked_sdpa_attention(q, k, v, dense)
    torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-6)


def test_the_training_mask_is_built_once():
    layout, device = MaskLayout(2, 3, teacher_forcing=False), torch.device("cpu")
    mask = attention.training_mask(layout, device)
    assert mask.shape == (6, 6) and mask.dtype == torch.bool
    assert attention.training_mask(MaskLayout(2, 3, teacher_forcing=False), "cpu") is mask


def test_mask_errors():
    q = torch.zeros(1, 6, 1, 4)
    with pytest.raises(ValueError, match="does not match"):
        attention.masked_sdpa_attention(q, q, q, torch.ones(5, 5, dtype=torch.bool))
    with pytest.raises(TypeError, match="unsupported attention mask"):
        attention.masked_attention(q, q, q, "mask")
