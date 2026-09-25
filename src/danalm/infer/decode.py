"""Greedy answers and intent scores with a KV cache, for any backend (Phase 7, D-034).

A backend is a step function: step(ids, positions, past) -> (logits, present).
- ids (batch, new) and positions (new,) are int64; the positions are shared by the batch.
- past and present hold two float32 arrays per layer, keys then values, each (batch, kv_heads,
  length, head_dim); the first step gets an empty cache (length 0).
- logits (batch, new, vocab) are float32.
DanaLM.step behind danalm.model.export.numpy_step and the ONNX graph behind
danalm.infer.onnx.OnnxStep are the two backends. The decoding matches danalm.sft.evaluate
(greedy_answers, label_logprobs), which recomputes the whole sequence at every step.
"""

from collections.abc import Callable

import numpy as np

Step = Callable[[np.ndarray, np.ndarray, list[np.ndarray]], tuple[np.ndarray, list[np.ndarray]]]


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Log-probabilities over the last axis, in float32."""
    x = x.astype(np.float32, copy=False)
    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def prefill(
    step: Step, prompts: list[list[int]], empty: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run prompts of equal length through the model: the logits at their last position
    (batch, vocab) and the cache after them."""
    ids = np.asarray(prompts, dtype=np.int64)
    logits, past = step(ids, np.arange(ids.shape[1], dtype=np.int64), empty)
    return logits[:, -1], past


def greedy_from(
    step: Step,
    last_logits: np.ndarray,
    past: list[np.ndarray],
    start: int,
    eos: int,
    max_new: int,
) -> list[tuple[list[int], bool]]:
    """Greedy continuation of prefilled prompts (the next position is `start`): (generated ids
    without the end token, whether the end token came), at most max_new tokens, as in
    danalm.sft.evaluate.greedy_answers. Finished rows keep producing the end token."""
    done = np.zeros(last_logits.shape[0], dtype=bool)
    steps = []
    nxt = last_logits.argmax(-1)
    for n in range(max_new):
        nxt = np.where(done, eos, nxt)
        steps.append(nxt)
        done |= nxt == eos
        if done.all() or n == max_new - 1:
            break
        logits, past = step(
            nxt[:, None].astype(np.int64), np.array([start + n], dtype=np.int64), past
        )
        nxt = logits[:, -1].argmax(-1)
    out = []
    for g in np.stack(steps, axis=1).tolist():
        out.append((g[: g.index(eos)], True) if eos in g else (g, False))
    return out


def label_logprobs_from(
    step: Step,
    past: list[np.ndarray],
    start: int,
    prefix: list[int],
    conts: list[list[int]],
    pad: int,
) -> np.ndarray:
    """(batch, len(conts)) log-probability of each continuation after the prompt and the prefix,
    from the cache of the prompts (the next position is `start`), as in
    danalm.sft.evaluate.label_logprobs. One step reads the prefix; one more step reads every
    continuation at once, each after its own copy of the cache, right-padded (with causal
    attention the padding cannot change the real tokens)."""
    b, n = past[0].shape[0], len(conts)
    ids = np.tile(np.asarray(prefix, dtype=np.int64), (b, 1))
    logits, past = step(ids, np.arange(start, start + len(prefix), dtype=np.int64), past)
    scores = log_softmax(logits[:, -1])[:, [c[0] for c in conts]]
    longest = max(len(c) for c in conts)
    if longest == 1:
        return scores
    inp = np.full((n, longest - 1), pad, dtype=np.int64)
    for i, c in enumerate(conts):
        inp[i, : len(c) - 1] = c[:-1]
    copies = [np.repeat(p, n, axis=0) for p in past]  # row m * n + i: message m, intent i
    at = start + len(prefix)
    logits, _ = step(np.tile(inp, (b, 1)), np.arange(at, at + longest - 1, dtype=np.int64), copies)
    lp = log_softmax(logits).reshape(b, n, longest - 1, -1)
    for i, c in enumerate(conts):
        for j in range(1, len(c)):
            scores[:, i] += lp[:, i, j - 1, c[j]]
    return scores
