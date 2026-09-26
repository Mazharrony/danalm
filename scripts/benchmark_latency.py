"""Latency and memory of every system on the CPU, batch 1 (Phase 7, D-034 protocol).

Usage: uv run python scripts/benchmark_latency.py --config configs/deploy/phase7.yaml
Messages: a seeded sample of development messages (latency.per_set from each set in
phase7.dev_sets, or only the sets named in latency.sets; never the test set), after
latency.warmup warm-up messages. Systems:
- pytorch-fp32: PyTorch float32 without the cache (the setting of D-032);
- pytorch-fp32-cache: the same model with the KV cache;
- onnx-<variant>: the ONNX variants through danalm.infer.predictor.
For each system and each thread count in latency.threads, a fresh worker process (this script
with --worker) measures, per message: the answer alone (greedy decoding to the end token) and
the full prediction (answer plus the 21-intent confidence, timed on its own; for ONNX, the
service's predict()), plus answer tokens per second, peak process memory (resident) and load
time. ONNX workers never import PyTorch, as in the service. Workers opt out of Windows power
throttling and record it. Writes phase7.latency_results.
"""

import argparse
import contextlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from danalm.config import config_from_cli, load_config


def sample_messages(p: dict[str, Any], lat: dict[str, Any]) -> list[str]:
    rng = random.Random(lat["seed"])
    out = []
    names = lat.get("sets") or list(p["dev_sets"])
    for path in (p["dev_sets"][k] for k in names):
        rows = []
        for f in [path] if isinstance(path, str) else path:  # one file or a list of files
            with open(f, encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh]
        out += [r["message"] for r in rng.sample(rows, lat["per_set"])]
    return out


def summary(times: list[float]) -> dict[str, float]:
    s = sorted(times)
    return {"median_s": statistics.median(s), "p95_s": s[math.ceil(0.95 * len(s)) - 1]}


def worker(config: str, system: str, threads: int) -> dict[str, Any]:
    """Measure one system with one thread count in this process."""
    from danalm.utils.power import disable_power_throttling

    throttling_off = disable_power_throttling()
    cfg = load_config(config)
    p, lat, d, ev = cfg["phase7"], cfg["latency"], cfg["sft_data"], cfg["eval"]
    proc = psutil.Process()
    rss_start = proc.memory_info().rss
    messages = sample_messages(p, lat)
    deploy = Path(cfg["deploy"]["out_dir"])
    t0 = time.perf_counter()
    if system.startswith("onnx-"):
        from danalm.infer.decode import greedy_from, prefill
        from danalm.infer.predictor import Predictor
        from danalm.sft.format import prompt_ids

        pred = Predictor(deploy / system.removeprefix("onnx-"), threads=threads, threshold=0.5)
        size = (deploy / system.removeprefix("onnx-") / "model.onnx").stat().st_size

        def answer(m: str) -> int:
            prompt = prompt_ids(pred.tok, pred.chat, m, pred.meta["normalize"])
            last, past = prefill(pred.step, [prompt], pred.step.empty(1))
            [(ids, _)] = greedy_from(
                pred.step, last, past, len(prompt), pred.chat.eos, pred.meta["max_new_tokens"]
            )
            return len(ids) + 1

        def full(m: str) -> None:
            pred.predict(m)

    else:
        import torch
        from tokenizers import Tokenizer

        from danalm.data.intents import load_intents
        from danalm.infer.decode import greedy_from, label_logprobs_from, prefill
        from danalm.model.export import empty_cache, numpy_step
        from danalm.model.transformer import DanaLM, ModelConfig
        from danalm.sft.evaluate import greedy_answers, label_logprobs
        from danalm.sft.format import ChatTokens, label_continuations, prompt_ids

        torch.set_num_threads(threads)
        checkpoint = json.loads(Path(cfg["deploy"]["selected"]).read_text(encoding="utf-8"))["checkpoint"]  # fmt: skip
        size = Path(checkpoint).stat().st_size
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = DanaLM(ModelConfig(**state["model_config"])).eval()
        model.load_state_dict(state["model"])
        tok = Tokenizer.from_file(d["tokenizer_file"])
        chat = ChatTokens.from_tokenizer(tok, d["special"])
        prefix, conts = label_continuations(tok, [i["name"] for i in load_intents(d["intents_file"])])  # fmt: skip
        max_new, null = ev["max_new_tokens"], contextlib.nullcontext
        step = numpy_step(model)

        def answer(m: str) -> int:
            prompt = prompt_ids(tok, chat, m, d["normalize"])
            with torch.no_grad():
                if system == "pytorch-fp32":
                    [(ids, _)] = greedy_answers(model, [prompt], max_new, chat.eos, 1, null)
                    return len(ids) + 1
                last, past = prefill(step, [prompt], empty_cache(model, 1))
                [(ids, _)] = greedy_from(step, last, past, len(prompt), chat.eos, max_new)
                return len(ids) + 1

        def full(m: str) -> None:
            prompt = prompt_ids(tok, chat, m, d["normalize"])
            with torch.no_grad():
                if system == "pytorch-fp32":
                    greedy_answers(model, [prompt], max_new, chat.eos, 1, null)
                    label_logprobs(model, [prompt], prefix, conts, 1, null)
                    return
                last, past = prefill(step, [prompt], empty_cache(model, 1))
                greedy_from(step, last, past, len(prompt), chat.eos, max_new)
                label_logprobs_from(step, past, len(prompt), prefix, conts, chat.pad)

    load_s = time.perf_counter() - t0
    rss_peak = proc.memory_info().rss
    ans_times, full_times, tokens = [], [], []
    for k, m in enumerate(messages[: lat["warmup"]] + messages):
        start = time.perf_counter()
        n = answer(m)
        mid = time.perf_counter()
        full(m)
        end = time.perf_counter()
        if k >= lat["warmup"]:
            ans_times.append(mid - start)
            full_times.append(end - mid)
            tokens.append(n)
        rss_peak = max(rss_peak, proc.memory_info().rss)
    return {
        "messages": len(ans_times), "threads": threads, "power_throttling_off": throttling_off,
        "answer": summary(ans_times), "full": summary(full_times),
        "mean_answer_tokens": statistics.mean(tokens),
        "answer_tokens_per_s": sum(tokens) / sum(ans_times), "load_s": load_s,
        "rss_start_mb": rss_start / 2**20, "rss_peak_mb": rss_peak / 2**20, "file_mb": size / 2**20,
    }  # fmt: skip


def main() -> None:
    if "--worker" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--worker", required=True)
        ap.add_argument("--config", required=True)
        ap.add_argument("--threads", type=int, required=True)
        a = ap.parse_args()
        print("RESULT " + json.dumps(worker(a.config, a.worker, a.threads)), flush=True)
        return
    cfg = config_from_cli(__doc__)
    p, lat = cfg["phase7"], cfg["latency"]
    config_path = sys.argv[sys.argv.index("--config") + 1]
    systems = ["pytorch-fp32", "pytorch-fp32-cache"] + [f"onnx-{v}" for v in p["variants"]]
    results: dict[str, Any] = {"cpu": platform.processor() or platform.machine(),
                               "messages_per_set": lat["per_set"], "seed": lat["seed"],
                               "sets": lat.get("sets") or list(p["dev_sets"])}  # fmt: skip
    for system in systems:
        results[system] = {}
        for threads in lat["threads"]:
            cmd = [sys.executable, __file__, "--worker", system, "--config", config_path,
                   "--threads", str(threads)]  # fmt: skip
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            r = json.loads(next(x for x in out.splitlines() if x.startswith("RESULT "))[7:])
            results[system][f"threads_{threads}"] = r
            print(f"{system:<20} {threads:>2} threads: answer {r['answer']['median_s']:.3f} s "
                  f"(p95 {r['answer']['p95_s']:.3f}); with confidence {r['full']['median_s']:.3f} s, "
                  f"{r['answer_tokens_per_s']:.0f} tok/s, peak {r['rss_peak_mb']:.0f} MB", flush=True)  # fmt: skip
    Path(p["latency_results"]).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
