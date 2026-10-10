"""The umT5-XXL text encoder and tokenizer of Wan2.2 (encoder only), for the fixed prompt.

The prompt is fixed (:data:`FIXED_PROMPT`), so its embedding is a constant: a client loads the
precomputed file (:func:`load_prompt_embedding`) when its config names one, and builds umT5 only to
encode the prompt otherwise. The embedding is ``[1, 512, 4096]`` bf16, from bf16 parameters on CUDA
(a CPU bf16 run is not bit-equal), with the rows past the prompt's 10 tokens exactly 0.
"""

import html
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldcast.utils.precision import HALF_DTYPES
from worldcast.utils.weights import module_from_state, read_state_dict

__all__ = [
    "FIXED_PROMPT",
    "FIXED_PROMPT_TOKEN_IDS",
    "T5_CHECKPOINT_NAME",
    "TEXT_DIM",
    "TEXT_LEN",
    "TOKENIZER_SUBDIR",
    "GELU",
    "T5Attention",
    "T5Block",
    "T5FeedForward",
    "T5LayerNorm",
    "T5RelativeEmbedding",
    "TextEncoder",
    "Tokenizer",
    "UMT5Config",
    "UMT5Encoder",
    "UMT5Tokenizer",
    "load_prompt_embedding",
    "load_text_encoder",
    "save_prompt_embedding",
]

#: The prompt of every WorldCast block.
FIXED_PROMPT = "first-person Counter-Strike 2 gameplay"
#: umT5 token ids of :data:`FIXED_PROMPT` after whitespace cleaning, ``</s>`` (id 1) included; the
#: padding id is 0. Tokens: ``▁first - person ▁Counter - Strike ▁ 2 ▁gameplay </s>``.
FIXED_PROMPT_TOKEN_IDS: tuple[int, ...] = (2330, 280, 22247, 68811, 280, 224737, 273, 278, 92819, 1)
#: Token length the tokenizer pads / truncates to.
TEXT_LEN = 512
#: umT5-XXL hidden size.
TEXT_DIM = 4096

#: File names inside the Wan2.2-TI2V-5B snapshot directory.
T5_CHECKPOINT_NAME = "models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER_SUBDIR = "google/umt5-xxl"


# ========================================================================================== encoder
@dataclass(frozen=True)
class UMT5Config:
    """Encoder dimensions; the defaults are umT5-XXL's."""

    vocab_size: int = 256384
    dim: int = 4096
    dim_attn: int = 4096
    dim_ffn: int = 10240
    num_heads: int = 64
    num_layers: int = 24
    num_buckets: int = 32


class GELU(nn.Module):
    """tanh-approximated GELU written out as umT5 does (``F.gelu``'s bits may differ)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Elementwise, any shape."""
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        )


class T5LayerNorm(nn.Module):
    """RMS norm without mean subtraction: the statistic in fp32, the result in the weight dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., dim]`` -> the same shape."""
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in HALF_DTYPES:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    """Multi-head self-attention with an additive position bias, unscaled (T5 has no 1/sqrt(d))."""

    def __init__(self, dim: int, dim_attn: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None, pos_bias: torch.Tensor
    ) -> torch.Tensor:
        """``x`` ``[B, L, C]``, ``mask`` ``[B, L]`` (0 = padding) or None, ``pos_bias``
        ``[1, heads, L, L]`` -> ``[B, L, C]``."""
        b, n, c = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(b, -1, n, c)
        k = self.k(x).view(b, -1, n, c)
        v = self.v(x).view(b, -1, n, c)

        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        attn_bias += pos_bias
        if mask is not None:
            attn_bias.masked_fill_(mask.view(b, 1, 1, -1) == 0, torch.finfo(x.dtype).min)

        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)
        return self.o(x.reshape(b, -1, n * c))


class T5FeedForward(nn.Module):
    """Gated GELU feed-forward: ``fc2(fc1(x) * gelu(gate(x)))``."""

    def __init__(self, dim: int, dim_ffn: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, L, dim]`` -> the same shape."""
        return self.fc2(self.fc1(x) * self.gate(x))


class T5RelativeEmbedding(nn.Module):
    """Bidirectional bucketed relative-position bias (umT5: one table per block)."""

    def __init__(self, num_buckets: int, num_heads: int, max_dist: int = 128) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        """-> bias ``[1, heads, lq, lk]`` in the embedding's dtype."""
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(
            lq, device=device
        ).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        return self.embedding(rel_pos).permute(2, 0, 1).unsqueeze(0).contiguous()

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets // 2
        rel_buckets = (rel_pos > 0).long() * num_buckets
        rel_pos = torch.abs(rel_pos)
        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact)
                / math.log(self.max_dist / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Block(nn.Module):
    """One encoder layer: pre-norm self-attention and pre-norm gated FFN, each with a residual."""

    def __init__(
        self, dim: int, dim_attn: int, dim_ffn: int, num_heads: int, num_buckets: int
    ) -> None:
        super().__init__()
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """``[B, L, dim]`` under ``mask`` ``[B, L]`` (0 = padding) or ``None`` -> the same shape."""
        pos_bias = self.pos_embedding(x.size(1), x.size(1))
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=pos_bias)
        return x + self.ffn(self.norm2(x))


class UMT5Encoder(nn.Module):
    """The umT5 encoder (a position bias per block), with the keys of Wan2.2's umT5 checkpoint."""

    def __init__(self, config: UMT5Config = UMT5Config()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [
                T5Block(
                    config.dim,
                    config.dim_attn,
                    config.dim_ffn,
                    config.num_heads,
                    config.num_buckets,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = T5LayerNorm(config.dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``ids`` ``[B, L]`` int64 and ``mask`` ``[B, L]`` (0 = padding) -> ``[B, L, dim]``."""
        x = self.token_embedding(ids)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


# ======================================================================================== tokenizer
def _basic_clean(text: str) -> str:
    import ftfy

    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    import regex

    return regex.sub(r"\s+", " ", text).strip()


class UMT5Tokenizer:
    """The umT5 tokenizer as Wan2.2 uses it: ftfy and whitespace cleaning, ``</s>`` appended,
    padded to ``seq_len``.

    Args:
        path (str | Path): the ``google/umt5-xxl`` directory of the Wan2.2-TI2V-5B snapshot
            (``transformers.AutoTokenizer`` resolves it to ``T5TokenizerFast``).
        seq_len (int): the padded length.
    """

    def __init__(self, path: str | Path, seq_len: int = TEXT_LEN) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                'the umT5 tokenizer needs the text extra: pip install "worldcast[text]"'
            ) from error

        self.path = str(path)
        self.seq_len = int(seq_len)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

    def __call__(self, prompts: str | Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Token ids and mask of ``B`` prompts (a string is one prompt), both ``[B, seq_len]``
        int64; the mask is 1 on the tokens."""
        if isinstance(prompts, str):
            prompts = [prompts]
        cleaned = [_whitespace_clean(_basic_clean(p)) for p in prompts]
        out = self.tokenizer(
            cleaned,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
            add_special_tokens=True,
        )
        return out.input_ids, out.attention_mask


# ===================================================================================== text encoder
#: ``fn(prompts) -> (ids, mask)``, both ``[B, 512]`` int64.
Tokenizer = Callable[[Sequence[str]], tuple[torch.Tensor, torch.Tensor]]


class TextEncoder(nn.Module):
    """Prompts -> zero-padded umT5 embeddings ``[B, 512, 4096]``, on the device and in the dtype of
    the encoder's parameters (paper: CUDA, bf16).

    Args:
        encoder (UMT5Encoder): the encoder; put in eval mode, without gradients.
        tokenizer (Tokenizer): ``fn(prompts) -> (ids, mask)``, for example :class:`UMT5Tokenizer`.
    """

    def __init__(self, encoder: UMT5Encoder, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.tokenizer = tokenizer

    @torch.no_grad()
    def forward(self, prompts: str | Sequence[str]) -> torch.Tensor:
        """``B`` prompts (a string is one prompt) -> ``[B, seq_len, dim]`` in the encoder's dtype,
        the padding rows 0."""
        ids, mask = self.tokenizer([prompts] if isinstance(prompts, str) else list(prompts))
        device = self.encoder.token_embedding.weight.device
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        embeds = self.encoder(ids, mask)
        # Attribution: zeroing each prompt's padding rows in place follows CausVid's WanTextEncoder
        # (github.com/tianweiy/CausVid at fab2440f, MIT) via Self Forcing
        # (github.com/guandeh17/Self-Forcing, Apache-2.0); the rest is Wan2.2's T5EncoderModel
        # (Apache-2.0).
        for prompt_embeds, seq_len in zip(embeds, seq_lens):
            prompt_embeds[seq_len:] = 0.0
        return embeds

    def encode_prompt(self) -> torch.Tensor:
        """The embedding ``[1, 512, dim]`` of :data:`FIXED_PROMPT`; raises if the tokenizer does
        not give the prompt its ids (:data:`FIXED_PROMPT_TOKEN_IDS`)."""
        ids, mask = self.tokenizer([FIXED_PROMPT])
        if tuple(ids[0, : int(mask.gt(0).sum())].tolist()) != FIXED_PROMPT_TOKEN_IDS:
            raise ValueError("the tokenizer is not Wan2.2's umT5: other ids for the fixed prompt")
        return self([FIXED_PROMPT])


def load_text_encoder(wan22_root: str | Path, *, device: torch.device | str = "cpu") -> TextEncoder:
    """The umT5 encoder and tokenizer of a Wan2.2-TI2V-5B snapshot.

    Args:
        wan22_root (str | Path): the snapshot directory.
        device (torch.device | str): where the encoder lives; it keeps the dtype of the
            snapshot's checkpoint (bf16).
    """
    # the tokenizer first: it fails without its extra, before the checkpoint is read
    tokenizer = UMT5Tokenizer(Path(wan22_root) / TOKENIZER_SUBDIR, seq_len=TEXT_LEN)
    state = read_state_dict(Path(wan22_root) / T5_CHECKPOINT_NAME)
    return TextEncoder(module_from_state(UMT5Encoder, state).to(device), tokenizer)


# ============================================================================ the shipped embedding
_EMBEDDING_KEY = "prompt_embeds"
#: The metadata of a prompt embedding file: its format, the prompt and the prompt's token count.
_EMBEDDING_METADATA = {
    "format": "worldcast.prompt_embedding.v1",
    "prompt": FIXED_PROMPT,
    "seq_len": str(len(FIXED_PROMPT_TOKEN_IDS)),
}


def _check_prompt_embedding(embeds: torch.Tensor) -> None:
    """The fixed prompt's embedding is ``[1, 512, dim]``, zero after the prompt's 10 tokens."""
    if embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[1] != TEXT_LEN:
        raise ValueError(
            f"the prompt embedding must be [1, {TEXT_LEN}, dim], got {tuple(embeds.shape)}"
        )
    tokens = len(FIXED_PROMPT_TOKEN_IDS)
    if bool(embeds[:, tokens:].ne(0).any()):
        raise ValueError(f"the prompt embedding is not zero after the prompt's {tokens} tokens")


def save_prompt_embedding(embeds: torch.Tensor, path: str | Path) -> None:
    """Write the fixed prompt's embedding ``[1, 512, dim]`` (:meth:`TextEncoder.encode_prompt`) as
    safetensors, in its dtype, with the metadata ``{format, prompt, seq_len}``."""
    from safetensors.torch import save_file

    _check_prompt_embedding(embeds)
    tensor = embeds.detach().to("cpu").contiguous()
    save_file({_EMBEDDING_KEY: tensor}, str(path), metadata=_EMBEDDING_METADATA)


def load_prompt_embedding(path: str | Path, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """The fixed prompt's embedding ``[1, 512, dim]`` of a file written by
    :func:`save_prompt_embedding`, on ``device``; a file of another format or for another prompt
    is refused."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        found = {key: metadata.get(key) for key in _EMBEDDING_METADATA}
        if found != _EMBEDDING_METADATA:
            raise ValueError(
                f"{path}: not the fixed prompt's embedding: its metadata says {found}, not"
                f" {_EMBEDDING_METADATA}"
            )
        embeds = handle.get_tensor(_EMBEDDING_KEY)
    _check_prompt_embedding(embeds)
    return embeds.to(device)
