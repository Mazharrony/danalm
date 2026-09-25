"""The chat format and the answer schema (D-029), without PyTorch: shared by SFT training and
evaluation and by the torch-free inference package (danalm.infer).

Format: <|user|>{message}<|assistant|>{"intent": ..., "reply": ...}<|endoftext|>. The message is
normalized as in the SFT data (which also masks PII), at training and at inference.
"""

import json
from dataclasses import dataclass
from typing import Any

from tokenizers import Tokenizer

from danalm.text import normalize

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


def parse_answer(text: str, intents: set[str]) -> dict[str, Any]:
    """parsed: the text is JSON. valid: a JSON object with exactly "intent" and "reply", both
    strings, and an intent from the taxonomy (D-029)."""
    try:
        obj = json.loads(text)
    except ValueError:
        return {"parsed": False, "valid": False, "intent": None, "reply": None}
    if not isinstance(obj, dict):
        return {"parsed": True, "valid": False, "intent": None, "reply": None}
    intent = obj.get("intent") if isinstance(obj.get("intent"), str) else None
    reply = obj.get("reply") if isinstance(obj.get("reply"), str) else None
    valid = set(obj) == {"intent", "reply"} and intent in intents and reply is not None
    return {"parsed": True, "valid": valid, "intent": intent, "reply": reply}
