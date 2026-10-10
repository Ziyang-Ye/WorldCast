"""The umT5 text encoder module and the precomputed prompt embedding."""

import functools
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from worldcast.modeling.wan22 import text_encoder as T

REPO = Path(__file__).resolve().parents[3]


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


#: A two-layer encoder; the vocabulary reaches the fixed prompt's largest token id.
TINY = T.UMT5Config(
    vocab_size=224_738, dim=16, dim_attn=16, dim_ffn=32, num_heads=2, num_layers=2, num_buckets=8
)


def _text_encoder(ids_of) -> T.TextEncoder:
    """A tiny random encoder behind a tokenizer that gives ``prompt`` the ids ``ids_of(prompt)``,
    padded with 0 to 512."""

    def tokenize(prompts):
        ids = torch.zeros(len(prompts), T.TEXT_LEN, dtype=torch.long)
        for row, prompt in enumerate(prompts):
            tokens = ids_of(prompt)
            ids[row, : len(tokens)] = torch.tensor(tokens)
        return ids, (ids > 0).long()

    torch.manual_seed(0)
    return T.TextEncoder(T.UMT5Encoder(TINY), tokenize)


def test_the_padding_rows_of_an_embedding_are_zero():
    encoder = _text_encoder(lambda prompt: [3 + len(word) for word in prompt.split()] + [1])
    embeds = encoder(["a bb ccc", "dddd"])
    assert embeds.shape == (2, T.TEXT_LEN, 16) and not embeds.requires_grad
    for row, tokens in enumerate((4, 2)):
        assert bool(embeds[row, :tokens].any(dim=-1).all()) and not bool(embeds[row, tokens:].any())
    ids, mask = encoder.tokenizer(["a bb ccc"])
    assert bool(encoder.encoder(ids, mask)[0, 4:].any())  # the encoder itself leaves them nonzero
    assert torch.equal(encoder("a bb ccc"), embeds[:1])  # a string is one prompt


def test_encode_prompt_checks_the_fixed_prompts_token_ids():
    encoder = _text_encoder(lambda prompt: list(T.FIXED_PROMPT_TOKEN_IDS))
    embeds = encoder.encode_prompt()
    assert embeds.shape == (1, T.TEXT_LEN, 16) and torch.equal(embeds, encoder([T.FIXED_PROMPT]))
    assert bool(embeds[:, :10].any(dim=-1).all()) and not bool(embeds[:, 10:].any())
    other = _text_encoder(lambda prompt: [5, 6, 1])
    with pytest.raises(ValueError, match="not Wan2.2's umT5"):
        other.encode_prompt()


def test_the_loader_reads_the_snapshots_encoder_and_tokenizer(tmp_path, monkeypatch):
    """The checkpoint is assigned without a random draw, in its dtype; the encoder is umT5-XXL's,
    built here at a tiny size."""
    torch.manual_seed(3)
    state = {k: v.to(torch.bfloat16) for k, v in T.UMT5Encoder(TINY).state_dict().items()}
    torch.save(state, tmp_path / T.T5_CHECKPOINT_NAME)
    tokenizers = []
    monkeypatch.setattr(T, "UMT5Encoder", functools.partial(T.UMT5Encoder, TINY))
    monkeypatch.setattr(T, "UMT5Tokenizer", lambda *args, **kwargs: tokenizers.append(args) or len)
    before = torch.get_rng_state()
    encoder = T.load_text_encoder(tmp_path)
    assert torch.equal(torch.get_rng_state(), before)
    assert tokenizers == [(tmp_path / T.TOKENIZER_SUBDIR,)] and encoder.tokenizer is len
    loaded = encoder.encoder.state_dict()
    assert all(torch.equal(loaded[k], v) and loaded[k].dtype == v.dtype for k, v in state.items())
    assert not encoder.encoder.training


def test_the_tokenizer_names_its_extra_before_the_checkpoint_is_read(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # the import fails
    with pytest.raises(ImportError, match=r'pip install "worldcast\[text\]"'):
        T.UMT5Tokenizer(tmp_path)
    with pytest.raises(ImportError, match="text extra"):  # not the missing checkpoint
        T.load_text_encoder(tmp_path)


def _embedding(dtype=torch.bfloat16, tokens: int = len(T.FIXED_PROMPT_TOKEN_IDS), dim: int = 8):
    e = torch.randn(1, T.TEXT_LEN, dim, generator=torch.Generator().manual_seed(17)).to(dtype)
    e[:, tokens:] = 0
    return e


def test_prompt_embedding_round_trip(tmp_path):
    path, embeds = tmp_path / "fixed_prompt.safetensors", _embedding()
    T.save_prompt_embedding(embeds, path)
    back = T.load_prompt_embedding(path)
    assert back.dtype == torch.bfloat16 and torch.equal(back, embeds)
    with safe_open(str(path), framework="pt") as handle:
        assert handle.metadata() == {
            "format": "worldcast.prompt_embedding.v1",
            "prompt": T.FIXED_PROMPT,
            "seq_len": "10",
        }


def test_load_refuses_a_file_that_is_not_the_fixed_prompts_embedding(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_file({"weight": torch.zeros(2)}, str(path))
    with pytest.raises(ValueError, match="not the fixed prompt's embedding: .*'format': None"):
        T.load_prompt_embedding(path)
    metadata = {"format": "worldcast.prompt_embedding.v1", "prompt": "a corridor", "seq_len": "3"}
    save_file({"prompt_embeds": _embedding()}, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="its metadata says .*'prompt': 'a corridor'"):
        T.load_prompt_embedding(path)
    metadata.update(prompt=T.FIXED_PROMPT, seq_len="10")
    save_file({"prompt_embeds": _embedding(tokens=11)}, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="not zero after the prompt's 10 tokens"):
        T.load_prompt_embedding(path)


def test_prompt_embedding_validation(tmp_path):
    path = tmp_path / "fixed_prompt.safetensors"
    with pytest.raises(ValueError, match="not zero after the prompt's 10 tokens"):
        T.save_prompt_embedding(_embedding(tokens=11), path)
    with pytest.raises(ValueError, match=r"must be \[1, 512, dim\], got \(1, 256, 8\)"):
        T.save_prompt_embedding(torch.zeros(1, 256, 8), path)
    assert not path.exists()
