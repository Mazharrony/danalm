"""From a customer message to an intent, a reply, a confidence and a route (Phase 7, D-034).

A model directory holds the graph, tokenizer.json and danalm.json (written by
scripts/export_onnx.py and scripts/quantize_onnx.py): the intents, the special tokens, the
normalization, the answer length and the confidence threshold fixed on the real dev set.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from danalm.infer.decode import greedy_from, label_logprobs_from, prefill
from danalm.infer.onnx import OnnxStep
from danalm.sft.format import ChatTokens, label_continuations, parse_answer, prompt_ids
from danalm.text import normalize


class Predictor:
    def __init__(self, model_dir: str | Path, threads: int, threshold: float | None = None):
        d = Path(model_dir)
        self.meta: dict[str, Any] = json.loads((d / "danalm.json").read_text(encoding="utf-8"))
        self.tok = Tokenizer.from_file(str(d / self.meta["tokenizer_file"]))
        self.chat = ChatTokens.from_tokenizer(self.tok, self.meta["special"])
        self.intents: list[str] = self.meta["intents"]
        self.prefix, self.conts = label_continuations(self.tok, self.intents)
        self.step = OnnxStep(d / self.meta["model_file"], threads)
        self.threshold = self.meta.get("threshold") if threshold is None else threshold
        if self.threshold is None:
            raise ValueError(f"{d / 'danalm.json'} has no threshold yet; pass one")

    def predict(self, message: str) -> dict[str, Any]:
        """The answer, its confidence (the intent's share of the 21 intent likelihoods, D-030)
        and the route: "on_device" when the answer is valid and confident enough, else
        "escalate". message_masked is the normalized text with PII masked: what may leave the
        device, e.g. with an escalation."""
        start = time.perf_counter()
        norm = self.meta["normalize"]
        prompt = prompt_ids(self.tok, self.chat, message, norm)
        last, past = prefill(self.step, [prompt], self.step.empty(1))
        [(ids, finished)] = greedy_from(
            self.step, last, past, len(prompt), self.chat.eos, self.meta["max_new_tokens"]
        )
        answer = parse_answer(self.tok.decode(ids, skip_special_tokens=False), set(self.intents))
        scores = label_logprobs_from(
            self.step, past, len(prompt), self.prefix, self.conts, self.chat.pad
        )[0]
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        conf = float(probs[self.intents.index(answer["intent"])]) if answer["valid"] else 0.0
        on_device = answer["valid"] and conf >= self.threshold
        return {
            "intent": answer["intent"] if answer["valid"] else None,
            "reply": answer["reply"] if answer["valid"] else None,
            "confidence": round(conf, 4),
            "route": "on_device" if on_device else "escalate",
            "valid_json": answer["valid"],
            "finished": finished,
            "message_masked": normalize(message, **norm),
            "latency_ms": round(1000 * (time.perf_counter() - start), 1),
        }
