"""Evaluating the fine-tuned model (Phase 5 development metrics, D-029; Phase 6 reuses them).

- parse_answer: is the generated text valid JSON with exactly "intent" and "reply" as strings
  and a known intent?
- greedy_answers: greedy decoding, batched over prompts of equal length, so no padding and the
  same result as one prompt at a time. No KV cache.
- label_logprobs: the log-probability of every intent as the value of "intent", for intent by
  likelihood and the confidence score (D-030).
- intent_metrics and coverage_at: accuracy, macro-F1, and the coverage at a target accuracy.
"""

import json
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn

from danalm.data.pipeline import detect_lang
from danalm.sft.data import IGNORE, ChatTokens, collate, label_continuations, prompt_ids

Autocast = Callable[[], AbstractContextManager]


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


@torch.no_grad()
def greedy_answers(
    model: nn.Module,
    prompts: list[list[int]],
    max_new: int,
    eos: int,
    batch_size: int,
    autocast: Autocast,
) -> list[tuple[list[int], bool]]:
    """(generated ids without the end token, whether the end token came) for every prompt."""
    device = next(model.parameters()).device
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    out: list[tuple[list[int], bool]] = [([], False)] * len(prompts)
    for idxs in by_len.values():
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start : start + batch_size]
            x = torch.tensor([prompts[i] for i in chunk], device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            steps = []
            for _ in range(max_new):
                with autocast():
                    logits, _ = model(x)
                nxt = logits[:, -1].float().argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
                steps.append(nxt)
                done |= nxt == eos
                x = torch.cat([x, nxt[:, None]], dim=1)
                if bool(done.all()):
                    break
            for i, g in zip(chunk, torch.stack(steps, dim=1).tolist(), strict=True):
                out[i] = (g[: g.index(eos)], True) if eos in g else (g, False)
    return out


@torch.no_grad()
def label_logprobs(
    model: nn.Module,
    prompts: list[list[int]],
    prefix: list[int],
    conts: list[list[int]],
    batch_examples: int,
    autocast: Autocast,
) -> torch.Tensor:
    """(len(prompts), len(conts)) log-probabilities of each continuation after prompt + prefix."""
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(prompts), batch_examples):
        seqs, spans = [], []
        for p in prompts[start : start + batch_examples]:
            for c in conts:
                seq = p + prefix + c
                spans.append((len(seq) - len(c), len(seq)))
                seqs.append(seq)
        x = torch.zeros((len(seqs), max(map(len, seqs))), dtype=torch.long)
        for i, q in enumerate(seqs):
            x[i, : len(q)] = torch.tensor(q)
        x = x.to(device)
        with autocast():
            logits, _ = model(x)
        logits = logits[:, :-1].float()
        # token t is predicted from position t - 1
        tok_lp = logits.gather(-1, x[:, 1:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        mask = torch.zeros_like(tok_lp, dtype=torch.bool)
        for i, (a, b) in enumerate(spans):
            mask[i, a - 1 : b - 1] = True
        rows.append((tok_lp * mask).sum(-1).view(-1, len(conts)).cpu())
    return torch.cat(rows)


@torch.no_grad()
def answer_loss(
    model: nn.Module,
    examples: list[tuple[list[int], list[int]]],
    pad: int,
    batch_size: int,
    autocast: Autocast,
) -> float:
    """Mean loss per answer token with teacher forcing (the validation loss of D-029)."""
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        x, y = collate(examples[start : start + batch_size], pad)
        x, y = x.to(device), y.to(device)
        with autocast():
            logits, _ = model(x)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=IGNORE, reduction="sum",
        )  # fmt: skip
        total += loss.item()
        count += int((y != IGNORE).sum())
    return total / count


def intent_metrics(gold: list[str], pred: list[str | None]) -> dict[str, float]:
    """Accuracy, and macro-F1 over the intents that occur in `gold`. A None prediction (an
    invalid answer) counts as wrong."""
    pairs = list(zip(gold, pred, strict=True))
    f1 = []
    for c in sorted(set(gold)):
        tp = sum(g == c and p == c for g, p in pairs)
        fp = sum(g != c and p == c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"accuracy": sum(g == p for g, p in pairs) / len(pairs), "macro_f1": sum(f1) / len(f1)}


def coverage_at(confidence: list[float], correct: list[bool], target: float) -> dict[str, Any]:
    """D-030: the lowest confidence threshold at which the answers at or above it are at least
    `target` correct, the share of all answers at or above it (coverage) and their accuracy.
    Answers with equal confidence are always on the same side. Coverage is 0 if no threshold
    reaches the target."""
    pairs = sorted(zip(confidence, correct, strict=True), key=lambda p: -p[0])
    best: dict[str, Any] = {"threshold": None, "coverage": 0.0, "accuracy": None}
    right = 0
    for k, (c, ok) in enumerate(pairs, start=1):
        right += ok
        if k < len(pairs) and pairs[k][0] == c:
            continue
        if right / k >= target:
            best = {"threshold": c, "coverage": k / len(pairs), "accuracy": right / k}
    return best


def summarize(preds: list[dict[str, Any]], target_accuracy: float) -> dict[str, Any]:
    """The development metrics of D-029 overall and per variety, plus the D-030 coverage."""

    def block(ps: list[dict[str, Any]]) -> dict[str, Any]:
        gold = [p["intent"] for p in ps]
        gen = intent_metrics(gold, [p["pred_intent"] for p in ps])
        lik = intent_metrics(gold, [p["lik_intent"] for p in ps])
        return {
            "n": len(ps),
            "parse_rate": sum(p["parsed"] for p in ps) / len(ps),
            "valid_json": sum(p["valid"] for p in ps) / len(ps),
            "intent_accuracy": gen["accuracy"],
            "intent_macro_f1": gen["macro_f1"],
            "reply_lang": sum(p["reply_lang_ok"] for p in ps) / len(ps),
            "lik_intent_accuracy": lik["accuracy"],
            "unfinished": sum(not p["finished"] for p in ps) / len(ps),
        }

    out = block(preds)
    out["coverage"] = coverage_at(
        [p["conf"] for p in preds],
        [p["pred_intent"] == p["intent"] for p in preds],
        target_accuracy,
    )
    varieties = sorted({p["variety"] for p in preds})
    out["per_variety"] = {v: block([p for p in preds if p["variety"] == v]) for v in varieties}
    return out


def evaluate(
    model: nn.Module,
    tok: Tokenizer,
    chat: ChatTokens,
    rows: list[dict[str, Any]],
    intents: list[str],
    norm: dict[str, Any],
    reply_langs: dict[str, list[str]],
    ev: dict[str, Any],
    autocast: Autocast,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Metrics and per-row predictions for rows with message, intent and variety. `ev` holds
    max_new_tokens, gen_batch, score_batch and target_accuracy."""
    was_training = model.training
    model.eval()
    prompts = [prompt_ids(tok, chat, r["message"], norm) for r in rows]
    gens = greedy_answers(model, prompts, ev["max_new_tokens"], chat.eos, ev["gen_batch"], autocast)
    prefix, conts = label_continuations(tok, intents)
    probs = label_logprobs(model, prompts, prefix, conts, ev["score_batch"], autocast).softmax(-1)
    model.train(was_training)
    names = set(intents)
    preds = []
    for r, (ids, finished), p in zip(rows, gens, probs, strict=True):
        text = tok.decode(ids, skip_special_tokens=False)
        ans = parse_answer(text, names)
        pred = ans["intent"] if ans["valid"] else None
        reply_ok = bool(ans["valid"] and detect_lang(ans["reply"]) in reply_langs[r["variety"]])
        preds.append({
            "message": r["message"], "variety": r["variety"], "intent": r["intent"],
            "output": text, "finished": finished, "parsed": ans["parsed"], "valid": ans["valid"],
            "pred_intent": pred, "reply": ans["reply"] if ans["valid"] else None,
            "reply_lang_ok": reply_ok, "lik_intent": intents[int(p.argmax())],
            "lik_conf": float(p.max()), "conf": float(p[intents.index(pred)]) if pred else 0.0,
        })  # fmt: skip
    return summarize(preds, ev["target_accuracy"]), preds
