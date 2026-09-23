"""Fertility: how many tokens a tokenizer needs per word.

Words are whitespace-separated units of the *normalized* text (danalm.data.pipeline.normalize),
the same definition for every tokenizer, so the comparison is fair. Lower is better: fewer tokens
per word means shorter sequences, faster inference and more text per context window.
"""

from typing import Any

from tokenizers import Tokenizer


def fertility(tok: Tokenizer, texts: list[str]) -> dict[str, Any]:
    """Tokens per word and characters per token over `texts` (no special tokens added)."""
    encodings = tok.encode_batch(texts, add_special_tokens=False)
    tokens = sum(len(e.ids) for e in encodings)
    words = sum(len(t.split()) for t in texts)
    chars = sum(len(t) for t in texts)
    return {
        "texts": len(texts),
        "words": words,
        "tokens": tokens,
        "fertility": tokens / words if words else float("nan"),
        "chars_per_token": chars / tokens if tokens else float("nan"),
    }


def train_flops_per_token(vocab_size: int, d_model: int, n_layers: int) -> float:
    """Approximate training FLOPs per token (6 x parameters in matmuls) for a Llama-style model:
    12 d^2 per layer (attention 4 d^2 + SwiGLU MLP 8 d^2) plus the tied output head d x V.
    Attention-score FLOPs are left out; they do not depend on the vocabulary."""
    return 6 * (n_layers * 12 * d_model**2 + d_model * vocab_size)
