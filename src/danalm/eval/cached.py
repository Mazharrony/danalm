"""The D-029 development metrics with KV-cache decoding, for any backend (Phase 7, D-034).

evaluate_cached returns the same metrics and per-row predictions as
danalm.sft.evaluate.evaluate, which recomputes the whole sequence at every step, so a PyTorch
model, its KV-cache path and its ONNX variants can be compared row by row.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from danalm.infer.decode import Step, greedy_from, label_logprobs_from, prefill
from danalm.sft.evaluate import summarize
from danalm.sft.format import ChatTokens, label_continuations, parse_answer, prompt_ids
from danalm.text import detect_lang


def evaluate_cached(
    step: Step,
    empty: Callable[[int], list[np.ndarray]],
    tok: Tokenizer,
    chat: ChatTokens,
    rows: list[dict[str, Any]],
    intents: list[str],
    norm: dict[str, Any],
    reply_langs: dict[str, list[str]],
    ev: dict[str, Any],
    batch: int,
    label_batch: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Metrics and predictions for rows with message, intent and variety. Prompts of equal
    length share a step (batch at a time); intent scores run label_batch messages at a time."""
    prompts = [prompt_ids(tok, chat, r["message"], norm) for r in rows]
    prefix, conts = label_continuations(tok, intents)
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    gens: list[Any] = [None] * len(rows)
    scores: list[Any] = [None] * len(rows)
    for n, idxs in by_len.items():
        for s in range(0, len(idxs), batch):
            chunk = idxs[s : s + batch]
            last, past = prefill(step, [prompts[i] for i in chunk], empty(len(chunk)))
            out = greedy_from(step, last, past, n, chat.eos, ev["max_new_tokens"])
            for k in range(0, len(chunk), label_batch):
                part = [p[k : k + label_batch] for p in past]
                lp = label_logprobs_from(step, part, n, prefix, conts, chat.pad)
                for j, i in enumerate(chunk[k : k + label_batch]):
                    scores[i] = lp[j]
            for k, i in enumerate(chunk):
                gens[i] = out[k]
    names = set(intents)
    preds = []
    for r, (ids, finished), lp in zip(rows, gens, scores, strict=True):
        probs = np.exp(lp - lp.max())
        probs /= probs.sum()
        text = tok.decode(ids, skip_special_tokens=False)
        ans = parse_answer(text, names)
        pred = ans["intent"] if ans["valid"] else None
        reply_ok = bool(ans["valid"] and detect_lang(ans["reply"]) in reply_langs[r["variety"]])
        preds.append({
            "message": r["message"], "variety": r["variety"], "intent": r["intent"],
            "output": text, "finished": finished, "parsed": ans["parsed"], "valid": ans["valid"],
            "pred_intent": pred, "reply": ans["reply"] if ans["valid"] else None,
            "reply_lang_ok": reply_ok, "lik_intent": intents[int(probs.argmax())],
            "lik_conf": float(probs.max()), "conf": float(probs[intents.index(pred)]) if pred else 0.0,
        })  # fmt: skip
    return summarize(preds, ev["target_accuracy"]), preds
