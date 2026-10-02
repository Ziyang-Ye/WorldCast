"""The umT5 text encoder module and the precomputed prompt embedding."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from worldcast.modeling.wan22 import text_encoder as T

REPO = Path(__file__).resolve().parents[2]


def test_import_does_not_call_cuda():
    """Importing the module never touches CUDA."""
    code = (
        "import torch\n"
        "def boom(*a, **k):\n"
        "    raise RuntimeError('CUDA touched at import')\n"
        "torch.cuda.current_device = boom\n"
        "torch.cuda.is_available = boom\n"
        "import worldcast.modeling.wan22.text_encoder\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def _embedding(dtype=torch.bfloat16, seq_len: int = len(T.FIXED_PROMPT_TOKEN_IDS), dim: int = 8):
    e = torch.randn(1, T.TEXT_LEN, dim, generator=torch.Generator().manual_seed(17)).to(dtype)
    e[:, seq_len:] = 0
    return e


def test_prompt_embedding_round_trip(tmp_path):
    pytest.importorskip("safetensors")
    emb = T.PromptEmbedding(prompt=T.FIXED_PROMPT, embeds=_embedding(), seq_len=10)
    emb.save(str(tmp_path / "fixed_prompt.safetensors"))
    back = T.PromptEmbedding.load(str(tmp_path / "fixed_prompt.safetensors"))
    assert back.prompt == T.FIXED_PROMPT and back.seq_len == 10
    assert back.embeds.dtype == torch.bfloat16 and torch.equal(back.embeds, emb.embeds)
    with pytest.raises(ValueError, match="expected"):
        T.PromptEmbedding.load(str(tmp_path / "fixed_prompt.safetensors"), prompt="another prompt")


def test_prompt_embedding_validation():
    with pytest.raises(ValueError, match="padding"):
        T.PromptEmbedding(prompt=T.FIXED_PROMPT, embeds=_embedding(seq_len=11), seq_len=10)
    with pytest.raises(ValueError, match="tokens"):
        T.PromptEmbedding(prompt=T.FIXED_PROMPT, embeds=_embedding(seq_len=9), seq_len=9)
    with pytest.raises(ValueError, match=r"\[1, 512"):
        T.PromptEmbedding(prompt=T.FIXED_PROMPT, embeds=torch.zeros(1, 256, 8), seq_len=10)
    T.PromptEmbedding(prompt="some other prompt", embeds=_embedding(seq_len=9), seq_len=9)
