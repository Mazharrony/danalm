"""Phase 6 results page: DanaLM and the baselines on the human test set, the D-030 bars, errors,
reply quality, latency and memory (D-032).

Usage: uv run python scripts/phase6_report.py --config configs/eval/phase6.yaml
Reads what scripts/evaluate_test.py and scripts/evaluate_teacher.py wrote to phase6.out_dir and
writes phase6.results_md. Every rate has a Wilson 95% interval; macro-F1 differences have a
paired bootstrap 95% interval. Nothing here runs a model.
"""

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli
from danalm.eval.stats import paired_bootstrap, wilson_interval
from danalm.sft.evaluate import intent_metrics


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ⏎ ")


def rate(k: int, n: int) -> str:
    lo, hi = wilson_interval(k, n)
    return f"{100 * k / n:.1f}% ({k}/{n}; {100 * lo:.0f}–{100 * hi:.0f}%)"


def macro_f1(gold: list, pred: list) -> float:
    return intent_metrics(gold, pred)["macro_f1"]


def main() -> None:
    cfg = config_from_cli(__doc__)
    p = cfg["phase6"]
    out = Path(p["out_dir"])
    metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    dana = read_jsonl(out / "danalm_predictions.jsonl")
    systems = {
        "DanaLM (62M, Phase 5 model)": [x["pred_intent"] for x in dana],
        "CAMeLBERT-mix classifier (110M)": [x["pred_intent"] for x in read_jsonl(out / "camelbert_predictions.jsonl")],
        "Qwen3.5-35B-A3B, zero-shot": [x["pred_intent"] for x in read_jsonl(out / "teacher_predictions.jsonl")],
    }  # fmt: skip
    gold = [x["intent"] for x in dana]
    n = len(gold)
    judge = {x["id"]: x for x in read_jsonl(out / "reply_judge.jsonl")}
    bs = p["bootstrap"]
    names = list(systems)
    diffs = {other: paired_bootstrap(gold, systems[names[0]], systems[other], macro_f1,
                                     bs["n_resamples"], bs["seed"]) for other in names[1:]}  # fmt: skip

    valid = sum(x["valid"] for x in dana)
    lang_ok = sum(x["reply_lang_ok"] for x in dana)
    cov = metrics["danalm"]["coverage_at_dev_threshold"]
    judged = [j for j in judge.values() if j["good"] is not None]
    good = sum(j["good"] for j in judged)
    bars = p["bars"]
    gap = diffs[names[1]]
    bar_rows = [
        ("Valid JSON", f"≥ {100 * bars['valid_json']:.0f}%", rate(valid, n), valid / n >= bars["valid_json"]),
        ("Intent macro-F1 vs CAMeLBERT", f"within {100 * bars['macro_f1_gap']:.0f} points",
         f"{100 * gap['diff']:+.1f} points (95% interval {100 * gap['lo']:+.1f} to {100 * gap['hi']:+.1f})",
         gap["diff"] >= -bars["macro_f1_gap"]),
        ("Coverage at the SFT-validation threshold " + f"({cov['threshold']:.3f})",
         f"≥ {100 * bars['coverage']:.0f}% answered, ≥ {100 * bars['coverage_accuracy']:.0f}% of them right",
         f"{100 * cov['coverage']:.1f}% answered ({cov['answered']}/{n}), "
         + ("–" if cov["accuracy"] is None else f"{100 * cov['accuracy']:.1f}% right"),
         cov["coverage"] >= bars["coverage"] and (cov["accuracy"] or 0) >= bars["coverage_accuracy"]),
        ("Reply in the right language", f"≥ {100 * bars['reply_lang']:.0f}%", rate(lang_ok, n),
         lang_ok / n >= bars["reply_lang"]),
    ]  # fmt: skip

    disputed = set()
    with open(p["review_sheet"], encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["row"]) in p["disputed_rows"]:
                disputed.add(row["text"])
    keep = [i for i, x in enumerate(dana) if x["message"] not in disputed]
    per_intent = sorted(Counter(gold).items(), key=lambda kv: (-kv[1], kv[0]))
    errors = [x for x in dana if x["pred_intent"] != x["intent"]]
    lat = metrics["latency"]
    cpu, gpu = lat["cpu_float32"], lat["gpu_bf16"]

    lines = [
        "# Phase 6 results: evaluation on the human test set (English part)",
        "",
        f"Generated on {date.today().isoformat()} by `scripts/phase6_report.py`; do not edit by hand. "
        "Protocol: D-032, bars: D-030 in [DECISIONS.md](../DECISIONS.md).",
        "",
        f"**Test set:** {n} English messages over {len(per_intent)} intents, labelled by the owner (MR), SHA-256 "
        f"`{metrics['test_sha256'][:12]}…`. The overlap check with all training data reports 0 overlaps. The Gulf "
        "Arabic, Arabizi and mixed parts of the test set do not exist yet, so **this evaluation covers English "
        f"only**. With {n} messages, one message is {100 / n:.1f} points, so the intervals are wide.",
        "",
        "## Results",
        "",
        "| System | Intent accuracy (95% interval) | Macro-F1 | Valid JSON | Reply in the right language |",
        "|---|---|---:|---|---|",
        *(
            f"| {name} | {rate(sum(g == q for g, q in zip(gold, pred, strict=True)), n)} | "
            f"{100 * macro_f1(gold, pred):.1f}% | "
            + (f"{rate(valid, n)} | {rate(lang_ok, n)} |" if k == 0 else "– | – |")
            for k, (name, pred) in enumerate(systems.items())
        ),
        "",
        "- Messages without a usable intent, counted as wrong (D-032): "
        + "; ".join(f"{name} {sum(q is None for q in pred)}" for name, pred in systems.items())
        + ". (DanaLM: an answer that is not valid JSON; Qwen: a message its answer lines did not label.)",
        *(f"- Macro-F1 difference, DanaLM minus {other}: {100 * r['diff']:+.1f} points (paired bootstrap 95% "
          f"interval {100 * r['lo']:+.1f} to {100 * r['hi']:+.1f})." for other, r in diffs.items()),
        "",
        "## The D-030 bars (English part only)",
        "",
        "| Bar | Target | Measured | Result |",
        "|---|---|---|---|",
        *(f"| {b} | {target} | {measured} | **{'met' if ok else 'missed'}** |" for b, target, measured, ok in bar_rows),
        "",
        "## Reply quality",
        "",
        f"Qwen judged the {len(judged)} valid DanaLM replies it gave a full verdict for on four yes/no questions: "
        + ", ".join(f"{q.replace('_', ' ')} {rate(sum(j[q] is True for j in judged), len(judged))}"
                    for q in ("answers", "polite_clear", "language", "safe"))
        + f". **All four yes: {rate(good, len(judged))}.** Qwen wrote DanaLM's training replies, so it may be "
        "lenient; every reply is listed at the end for the owner's spot check.",
        "",
        "## Per intent",
        "",
        "| Intent | Messages | " + " | ".join(names) + " |",
        "|---|---:|" + "---:|" * len(names),
        *(f"| {c} | {k} | " + " | ".join(
            f"{sum(g == q == c for g, q in zip(gold, pred, strict=True))}/{k}" for pred in systems.values()) + " |"
          for c, k in per_intent),
        "",
        f"## Sensitivity: without the {len(disputed)} disputed messages",
        "",
        "An independent audit disagreed with the owner's label on review rows "
        f"{', '.join(map(str, p['disputed_rows']))}. Leaving those messages out (the owner's labels stay the "
        "primary result): "
        + "; ".join(f"{name} {rate(sum(gold[i] == pred[i] for i in keep), len(keep))}"
                    for name, pred in systems.items()) + ".",
        "",
        f"## Every DanaLM error ({len(errors)} of {n})",
        "",
        "| Id | Message | Gold | DanaLM intent | Confidence | DanaLM reply |",
        "|---|---|---|---|---:|---|",
        *(f"| {x['id']} | {cell(x['message'])} | {x['intent']} | {x['pred_intent'] or '(invalid answer)'} | "
          f"{x['conf']:.2f} | {cell(x['reply'] or x['output'])} |" for x in errors),
        "",
        "## Latency and memory",
        "",
        f"Per message, batch 1, greedy decoding with the current decoder (no KV cache yet; Phase 7 adds it and "
        f"quantizes). Median and p95 over the {cpu['messages']} messages after warm-up. CPU: {lat['cpu']}, "
        f"{lat['cpu_threads']} threads.",
        "",
        "| | CPU, float32 | GPU (RTX 4070), bf16 |",
        "|---|---:|---:|",
        f"| Median per message | {cpu['median_s']:.2f} s | {gpu['median_s']:.2f} s |",
        f"| p95 per message | {cpu['p95_s']:.2f} s | {gpu['p95_s']:.2f} s |",
        f"| Answer tokens (mean) | {cpu['mean_answer_tokens']:.0f} | {gpu['mean_answer_tokens']:.0f} |",
        f"| Tokens per second | {cpu['tokens_per_s']:.0f} | {gpu['tokens_per_s']:.0f} |",
        f"| Memory | weights {lat['weights_mb_float32']:.0f} MB; process peak {cpu['rss_peak_mb']:.0f} MB "
        f"(before loading the model: {cpu['rss_before_model_mb']:.0f} MB) | peak allocated {gpu['peak_allocated_mb']:.0f} MB |",
        "",
        "## All replies (for the owner's spot check)",
        "",
        "<details><summary>Show the 64 answers</summary>",
        "",
        "| Id | Message | Gold | DanaLM | Reply | Qwen: good? |",
        "|---|---|---|---|---|---|",
        *(f"| {x['id']} | {cell(x['message'])} | {x['intent']} | {x['pred_intent'] or '(invalid)'} | "
          f"{cell(x['reply'] or x['output'])} | "
          + ({True: "yes", False: "no: " + cell(judge[x['id']]['note']), None: "–"}[judge[x['id']]['good']]
             if x['id'] in judge else "–") + " |" for x in dana),
        "",
        "</details>",
    ]  # fmt: skip
    Path(p["results_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
