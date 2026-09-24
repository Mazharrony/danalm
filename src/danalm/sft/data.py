"""SFT examples as token ids (Phase 5, D-029).

Format: <|user|>{message}<|assistant|>{"intent": ..., "reply": ...}<|endoftext|>. The message is
normalized as in the SFT data (which also masks PII), at training and at inference. The loss
covers only the answer: the JSON and the end token.
"""

import random
from dataclasses import dataclass
from typing import Any

import torch
from tokenizers import Tokenizer

from danalm.data.pipeline import normalize

IGNORE = -100  # the model's ignore_index for targets without a loss
INTENT_PREFIX = '{"intent": "'


@dataclass(frozen=True)
class ChatTokens:
    """Ids of the special tokens of the chat format."""

    user: int
    assistant: int
    eos: int
    pad: int

    @classmethod
    def from_tokenizer(cls, tok: Tokenizer, special: dict[str, str]) -> "ChatTokens":
        ids = {role: tok.token_to_id(text) for role, text in special.items()}
        missing = [special[role] for role, i in ids.items() if i is None]
        if missing:
            raise ValueError(f"the tokenizer has no special tokens {missing}")
        return cls(**ids)


def prompt_ids(tok: Tokenizer, chat: ChatTokens, message: str, norm: dict[str, Any]) -> list[int]:
    """<|user|> message <|assistant|>: what the model sees before it answers."""
    text = normalize(message, **norm)
    return [chat.user, *tok.encode(text, add_special_tokens=False).ids, chat.assistant]


def example_ids(
    tok: Tokenizer, chat: ChatTokens, message: str, target: str, norm: dict[str, Any]
) -> tuple[list[int], list[int]]:
    """(ids, labels) of one example. labels equal ids on the answer and the end token, and are
    IGNORE on the prompt."""
    prompt = prompt_ids(tok, chat, message, norm)
    answer = [*tok.encode(target, add_special_tokens=False).ids, chat.eos]
    return prompt + answer, [IGNORE] * len(prompt) + answer


def collate(
    examples: list[tuple[list[int], list[int]]], pad: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the longest example. Returns (x, y) for next-token prediction: y[t] is the
    label of token t + 1, and padding and prompt positions are IGNORE. Padding comes after the
    real tokens, so with causal attention it cannot change them."""
    n = max(len(ids) for ids, _ in examples) - 1
    x = torch.full((len(examples), n), pad, dtype=torch.long)
    y = torch.full((len(examples), n), IGNORE, dtype=torch.long)
    for i, (ids, labels) in enumerate(examples):
        k = len(ids) - 1
        x[i, :k] = torch.tensor(ids[:-1])
        y[i, :k] = torch.tensor(labels[1:])
    return x, y


def length_batches(
    lengths: list[int], batch_size: int, chunk_batches: int, rng: random.Random
) -> list[list[int]]:
    """Batches of example indices with similar lengths, so little padding: shuffle, sort chunks
    of chunk_batches * batch_size examples by length, cut them into batches, shuffle the batches.
    Every index appears exactly once."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    chunk = batch_size * chunk_batches
    batches = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start : start + chunk], key=lambda i: lengths[i])
        batches += [part[b : b + batch_size] for b in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def label_continuations(tok: Tokenizer, names: list[str]) -> tuple[list[int], list[list[int]]]:
    """(prefix ids, one continuation per intent) for scoring an intent as the value of "intent".
    The continuation runs to the closing quote and comma, tokenized together with the prefix, as
    in the training targets ('{"intent": "<name>", "reply": ...')."""
    prefix = tok.encode(INTENT_PREFIX, add_special_tokens=False).ids
    conts = []
    for name in names:
        full = tok.encode(f'{INTENT_PREFIX}{name}",', add_special_tokens=False).ids
        if full[: len(prefix)] != prefix:
            raise ValueError(f"the intent prefix tokenizes differently before {name!r}")
        conts.append(full[len(prefix) :])
    return prefix, conts
