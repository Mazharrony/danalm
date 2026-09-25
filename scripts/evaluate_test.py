"""Score the Phase 5 model and the CAMeLBERT baseline on the human test set (Phase 6, D-032).

Usage: uv run python scripts/evaluate_test.py --config configs/eval/phase6.yaml
Refuses a test file whose SHA-256 is not phase6.test_sha256, and refuses to run while the teacher
server is up. Writes to phase6.out_dir:
- danalm_predictions.jsonl: every answer (output, valid JSON, intent, reply, confidence);
- camelbert_predictions.jsonl: the baseline's intent per message;
- test_metrics.json: DanaLM's metrics (D-029) and its coverage at the threshold fixed on the SFT
  validation split (D-030), the baseline's accuracy and macro-F1, and latency and memory: per
  message, batch 1, on the CPU (float32, phase6.latency.cpu_threads threads, measured first, before
  the GPU is touched) and on the GPU (bf16), after phase6.latency.warmup warm-up messages.
"""

import contextlib
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from danalm.config import config_from_cli
from danalm.data.intents import load_intents
from danalm.data.pipeline import normalize
from danalm.model.transformer import DanaLM, ModelConfig
from danalm.sft.data import ChatTokens, prompt_ids
from danalm.sft.evaluate import Autocast, evaluate, greedy_answers, intent_metrics
from danalm.teacher.server import assert_teacher_stopped
from danalm.utils.seed import set_seed


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def load_test(p: dict[str, Any]) -> list[dict[str, Any]]:
    """The test rows as message/intent/variety/id, after checking the file is the scored one."""
    raw = Path(p["test_set"]).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != p["test_sha256"]:
        raise ValueError(f"test set changed: sha256 {digest} != {p['test_sha256']} (D-032)")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return [{"id": r["id"], "message": r["text"], "intent": r["intent"], "variety": r["variety"]}
            for r in rows]  # fmt: skip


def latency(model: torch.nn.Module, prompts: list[list[int]], eos: int, max_new: int, warmup: int,
            device: str, autocast: Autocast, proc: psutil.Process) -> dict[str, Any]:  # fmt: skip
    """Seconds per message (batch 1, greedy), after `warmup` messages; peak process memory seen."""
    times, lengths, rss_peak = [], [], proc.memory_info().rss
    for i, p in enumerate(prompts[:warmup] + prompts):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        [(ids, _)] = greedy_answers(model, [p], max_new, eos, 1, autocast)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - start)
            lengths.append(len(ids) + 1)
        rss_peak = max(rss_peak, proc.memory_info().rss)
    times.sort()
    return {"messages": len(times), "median_s": statistics.median(times),
            "p95_s": times[math.ceil(0.95 * len(times)) - 1], "mean_answer_tokens": statistics.mean(lengths),
            "tokens_per_s": sum(lengths) / sum(times), "rss_peak_mb": rss_peak / 2**20}  # fmt: skip


def main() -> None:
    cfg = config_from_cli(__doc__)
    assert_teacher_stopped(cfg["teacher"]["host"], cfg["teacher"]["port"])
    set_seed(cfg["seed"], cfg["deterministic"])
    p, d, ev = cfg["phase6"], cfg["sft_data"], cfg["eval"]
    out = Path(p["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = load_test(p)
    intents = [i["name"] for i in load_intents(d["intents_file"])]
    selected = json.loads(Path(p["selected"]).read_text(encoding="utf-8"))
    state = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)
    tok = Tokenizer.from_file(d["tokenizer_file"])
    chat = ChatTokens.from_tokenizer(tok, d["special"])
    prompts = [prompt_ids(tok, chat, r["message"], d["normalize"]) for r in rows]
    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    # CPU first, before any CUDA context exists (D-032: float32, a few threads, batch 1)
    torch.set_num_threads(p["latency"]["cpu_threads"])
    model = DanaLM(ModelConfig(**state["model_config"])).eval()
    model.load_state_dict(state["model"])
    weights_mb = sum(t.numel() * t.element_size() for t in model.parameters()) / 2**20
    cpu = latency(model, prompts, chat.eos, ev["max_new_tokens"], p["latency"]["warmup"], "cpu",
                  contextlib.nullcontext, proc)  # fmt: skip
    cpu["rss_before_model_mb"] = rss_before / 2**20

    def autocast() -> torch.autocast:
        return torch.autocast("cuda", dtype=torch.bfloat16)

    model = model.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    gpu = latency(model, prompts, chat.eos, ev["max_new_tokens"], p["latency"]["warmup"], "cuda",
                  autocast, proc)  # fmt: skip
    gpu["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    del gpu["rss_peak_mb"]

    # the scored run: the same decoding as the Phase 5 development metrics
    metrics, preds = evaluate(model, tok, chat, rows, intents, d["normalize"], d["reply_langs"],
                              ev, autocast)  # fmt: skip
    for r, pr in zip(rows, preds, strict=True):
        pr["id"] = r["id"]
    # None: no threshold reached the target accuracy on the development set, so nothing is answered
    threshold = selected["metrics"]["coverage"]["threshold"]
    covered = [pr for pr in preds if threshold is not None and pr["conf"] >= threshold]
    metrics["coverage_at_dev_threshold"] = {
        "threshold": threshold, "answered": len(covered), "coverage": len(covered) / len(preds),
        "accuracy": (sum(pr["pred_intent"] == pr["intent"] for pr in covered) / len(covered)
                     if covered else None),
    }  # fmt: skip
    write_jsonl(out / "danalm_predictions.jsonl", preds)
    del model
    torch.cuda.empty_cache()

    cb_dir = p["camelbert_dir"]
    cb_tok = AutoTokenizer.from_pretrained(cb_dir)
    cb = AutoModelForSequenceClassification.from_pretrained(cb_dir).to("cuda").eval()
    texts = [normalize(r["message"], **d["normalize"]) for r in rows]
    with torch.no_grad(), autocast():
        enc = cb_tok(texts, truncation=True, max_length=p["camelbert_max_length"], padding=True,
                     return_tensors="pt").to("cuda")  # fmt: skip
        labels = cb(**enc).logits.argmax(-1).tolist()
    cb_preds = [{"id": r["id"], "message": r["message"], "intent": r["intent"],
                 "pred_intent": cb.config.id2label[k]} for r, k in zip(rows, labels, strict=True)]  # fmt: skip
    write_jsonl(out / "camelbert_predictions.jsonl", cb_preds)
    gold = [r["intent"] for r in rows]
    result = {
        "test_set": p["test_set"], "test_sha256": p["test_sha256"], "messages": len(rows),
        "danalm_checkpoint": selected["checkpoint"], "danalm": metrics,
        "camelbert": intent_metrics(gold, [c["pred_intent"] for c in cb_preds]),
        "latency": {"cpu_float32": cpu, "gpu_bf16": gpu, "weights_mb_float32": weights_mb,
                    "cpu_threads": p["latency"]["cpu_threads"], "cpu": platform.processor() or platform.machine()},
    }  # fmt: skip
    text = json.dumps(result, indent=2)
    (out / "test_metrics.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
