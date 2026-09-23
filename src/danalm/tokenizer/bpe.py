"""DanaLM's byte-level BPE tokenizer (Arabic + English + Arabizi): build, train, save, load.

Text is normalized by `danalm.data.pipeline.normalize` *before* it reaches the tokenizer (at
training and at inference), so the tokenizer itself has no normalizer and decodes losslessly.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

# Letters include combining marks (\p{M}) so diacritics never split an Arabic word.
_AFTER_WORDS = r"|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
_CONTRACTIONS = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
PRETOKENIZERS = {
    # GPT-4 / Llama 3 style: numbers are separate chunks of up to 3 digits.
    "standard": _CONTRACTIONS + r"|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+" + _AFTER_WORDS,
    # Same, but a digit inside a word or at its edge stays with it, because in Arabizi digits are
    # letters ("3andi", "ma3a", "sa7"). Pure numbers are still split into chunks of up to 3.
    "arabizi": _CONTRACTIONS
    + r"|[^\r\n\p{L}\p{M}\p{N}]?\p{N}?[\p{L}\p{M}]+(?:\p{N}[\p{L}\p{M}]+)*(?:\p{N}(?!\p{N}))?"
    + _AFTER_WORDS,
}


@dataclass(frozen=True)
class TokenizerConfig:
    """The `tokenizer:` section of a tokenizer config. No defaults: every value comes from YAML."""

    vocab_size: int  # final size, including special and placeholder tokens
    min_frequency: int
    pretokenizer: str  # a key of PRETOKENIZERS
    eos_token: str  # end of text; also separates documents
    pad_token: str
    extra_special_tokens: list[str]  # chat/format control tokens
    reserved_tokens: int  # spare <|reserved_i|> control tokens for later phases
    placeholder_tokens: list[str]  # PII placeholders: atomic, never merged, kept when decoding
    corpus: str  # JSONL file with a "text" field
    out_dir: str

    def special_tokens(self) -> list[str]:
        """Control tokens in id order: eos (id 0), pad (id 1), extras, reserved."""
        reserved = [f"<|reserved_{i}|>" for i in range(self.reserved_tokens)]
        return [self.eos_token, self.pad_token, *self.extra_special_tokens, *reserved]


def build(cfg: TokenizerConfig) -> Tokenizer:
    """An untrained tokenizer: regex pre-split, then byte-level BPE (no unknown tokens)."""
    tok = Tokenizer(models.BPE())
    split = pre_tokenizers.Split(Regex(PRETOKENIZERS[cfg.pretokenizer]), behavior="isolated")
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [split, pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def train(cfg: TokenizerConfig, texts: Iterable[str]) -> Tokenizer:
    """Train on `texts`. Placeholders are cut out of the text first, so BPE never spends merges
    on pieces of them; they are added afterwards as atomic (non-special) tokens."""
    tok = build(cfg)
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size - len(cfg.placeholder_tokens),
        min_frequency=cfg.min_frequency,
        special_tokens=cfg.special_tokens(),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    placeholders = re.compile("|".join(map(re.escape, cfg.placeholder_tokens)))
    tok.train_from_iterator((placeholders.sub(" ", t) for t in texts), trainer=trainer)
    tok.add_tokens([AddedToken(p, special=False, normalized=False) for p in cfg.placeholder_tokens])
    return tok


def save(tok: Tokenizer, cfg: TokenizerConfig, out_dir: str | Path) -> None:
    """Save in Hugging Face format, so `AutoTokenizer.from_pretrained(out_dir)` works."""
    from transformers import PreTrainedTokenizerFast

    special = cfg.special_tokens()
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token=cfg.eos_token,
        pad_token=cfg.pad_token,
        additional_special_tokens=special[2:],
    )
    hf.save_pretrained(str(out_dir))
