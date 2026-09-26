"""Phase 7 results page: export, the variants on the development sets and the choice, latency
and memory, and accuracy before and after quantization on the human test set (D-034).

Usage: uv run python scripts/phase7_report.py --config configs/deploy/phase7.yaml
Reads what export_onnx.py, quantize_onnx.py, evaluate_variants.py (dev, then test),
benchmark_latency.py and evaluate_teacher.py (configs/deploy/phase7_judge.yaml) wrote, and writes
phase7.results_md. Nothing here runs a model; the test sections appear once the test run exists.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli
from danalm.eval.stats import wilson_interval
from danalm.text import normalize, reply_fits


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def diff(dev: dict[str, Any], variant: str, split: str, limit: float) -> str:
    """A variant's intent accuracy against onnx-fp32 on one set, in points and in messages, with
    the D-034 limit in messages."""
    a, b = dev["metrics"][f"onnx-{variant}"][split], dev["metrics"]["onnx-fp32"][split]
    d = a["intent_accuracy"] - b["intent_accuracy"]
    return (f"{100 * d:+.3f} points ({round(d * a['n']):+d} of {a['n']} messages; "
            f"the limit is {-limit * a['n']:.2f})")  # fmt: skip


def rate(k: int, n: int) -> str:
    lo, hi = wilson_interval(k, n)
    return f"{100 * k / n:.1f}% ({k}/{n}; {100 * lo:.0f}–{100 * hi:.0f}%)"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fits(x: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """The D-035 reply guard on a saved prediction (as the predictor applies it)."""
    d = cfg["sft_data"]
    return reply_fits(normalize(x["message"], **d["normalize"]), x["reply"], d["reply_langs"])


def broken(path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Answers that are not valid JSON or whose reply the guard rejects."""
    return [x for x in read_jsonl(path) if not x["valid"] or not fits(x, cfg)]


GUARD_NOTE = (
    "Added after the test run: the predictor escalates a reply that does not fit the customer's "
    "language. Applied here to the saved predictions; no model was re-run. The development sets "
    "give the false-alarm check; the test rows are not an independent estimate, because the "
    "test set revealed the problem."
)


def guard_section(
    cfg: dict[str, Any], out: Path, dev: dict[str, Any], test: bool, note: str
) -> list[str]:
    """D-035: what the reply guard escalates, on the saved predictions of every system."""
    runs = [("dev", k) for k in cfg["phase7"]["dev_sets"]] + ([("test", "test")] if test else [])
    rows, flagged = [], []
    for split, k in runs:
        for s, thr in dev["thresholds"].items():
            path = out / f"{split}_{k}_{s}.jsonl"
            if not path.exists():
                continue
            ps = read_jsonl(path)
            bad = [x for x in ps if x["valid"] and not fits(x, cfg)]
            before = [x for x in ps if x["valid"] and x["conf"] >= thr]
            after = [x for x in before if fits(x, cfg)]

            def acc(a: list[dict[str, Any]]) -> str:
                return pct(sum(x["pred_intent"] == x["intent"] for x in a) / len(a) if a else None)

            rows.append(f"| {k} | {s} | {len(bad)} of {len(ps)} | {len(before)} → {len(after)} | "
                        f"{acc(before)} → {acc(after)} |")  # fmt: skip
            flagged += [(k, s, x) for x in bad]
    return [
        "## Reply guard (D-035)",
        "",
        note,
        "",
        "| Set | System | Replies rejected | Answered on the device, before → after | Their intent accuracy |",
        "|---|---|---:|---:|---:|",
        *rows,
        "",
        "Every rejected reply:",
        "",
        "| Set | System | Message | Reply |",
        "|---|---|---|---|",
        *(f"| {k} | {s} | {cell(x['message'])} | {cell(x['reply'])} |" for k, s, x in flagged),
    ]


def cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    cfg = config_from_cli(__doc__)
    p = cfg["phase7"]
    deploy, out = Path(cfg["deploy"]["out_dir"]), Path(p["out_dir"])
    parity = read_json(deploy / "export_parity.json")
    dev = read_json(out / "dev_metrics.json")
    lat = read_json(Path(p["latency_results"]))
    test = read_json(out / "test_metrics.json")
    metas = {v: read_json(deploy / v / "danalm.json") for v in p["variants"]}
    sel = dev["selected"]
    systems = list(dev["metrics"])
    sets = list(p["dev_sets"])
    # the page's wording; the defaults are the D-033 deployment's (D-034), a later one overrides them
    rep = p.get("report", {})
    names = {
        "sft_val": "SFT validation (957, all varieties)",
        "real_dev": "real dev set (804, English)",
        "qtype_dev": "question-type dev set (1,298, English and Saudi Arabic)",
    } | rep.get("set_names", {})
    all_sets = "both sets" if len(sets) == 2 else f"all {len(sets)} development sets"

    lines = [
        f"# {rep.get('title', 'Phase 7 results: quantization and deployment')}",
        "",
        f"Generated on {date.today().isoformat()} by `scripts/phase7_report.py`; do not edit by hand. "
        "Rules: " + rep.get("rules", "D-034 in [DECISIONS.md](../DECISIONS.md), fixed before any "
                            "quantized model was scored") + ". "
        f"The model is {rep.get('model', 'the D-033 model')} (`{dev['checkpoint'].replace(chr(92), '/')}`).",
        "",
        "## Export and variants",
        "",
        f"One ONNX step graph with a KV cache serves both the prompt and each new token. Right after "
        f"the export, PyTorch and ONNX Runtime gave identical greedy answers on "
        f"{parity['identical_greedy_answers']} of {parity['parity_messages']} real-dev messages; the "
        f"largest logit difference after the prompt was {parity['max_logit_diff_after_prompt']:.1e}.",
        "",
        "| Variant | Method | File | Quantized ops | Threshold (real dev, 95% target) |",
        "|---|---|---:|---|---:|",
        *(f"| {v} | {m.get('quantization', {}).get('method', 'float32')} | {m['model_bytes'] / 2**20:.1f} MB | "
          f"{', '.join(m.get('quantized_ops', [])) or '–'} | {m['threshold']:.3f} |" for v, m in metas.items()),
        "",
        "The input embedding table stays float32 in every variant (D-034).",
        "",
        "## Development sets: the check and the choice",
        "",
    ]  # fmt: skip
    for k in sets:
        lines += [
            f"**{names.get(k, k)}**",
            "",
            "| System | Intent accuracy | Macro-F1 | Valid JSON | Reply language | Same intent as the reference | Same answer as the reference | Answered at its threshold (accuracy) |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            *(f"| {s} | {pct(dev['metrics'][s][k]['intent_accuracy'])} | {pct(dev['metrics'][s][k]['intent_macro_f1'])} | "
              f"{pct(dev['metrics'][s][k]['valid_json'])} | {pct(dev['metrics'][s][k]['reply_lang'])} | "
              + ("–" if s == "pytorch-fp32" else pct(dev["agreement"][s][k]["same_intent"])) + " | "
              + ("–" if s == "pytorch-fp32" else pct(dev["agreement"][s][k]["same_answer"])) + " | "
              + f"{pct(dev['coverage_at_threshold'][s][k]['coverage'])} ({pct(dev['coverage_at_threshold'][s][k]['accuracy'])}) |"
              for s in systems),
            "",
        ]  # fmt: skip
    rule = p["rule"]
    lines += [
        "The reference is PyTorch float32 on the CPU, recomputing the whole sequence at every step "
        "(the D-029 decoding).",
        "",
        f"- Exactness (same answer as the reference on at least {pct(rule['min_same_answer'])} of {all_sets}): "
        + "; ".join(f"{s}: **{'met' if ok else 'missed'}**" for s, ok in sel["exactness_ok"].items()) + ".",
        f"- The D-034 rule for a quantized variant, on {all_sets}: intent accuracy at most "
        f"{100 * rule['max_accuracy_drop']:.0f} point below onnx-fp32, valid JSON ≥ {pct(rule['min_valid_json'])}, "
        f"reply language ≥ {pct(rule['min_reply_lang'])}.",
        *(f"  - {v}: **{'passes' if v in sel['passing'] else 'fails'}**. Intent accuracy against onnx-fp32: "
          + "; ".join(f"{names.get(k, k).split(' (')[0]} {diff(dev, v, k, rule["max_accuracy_drop"])}" for k in sets) + ". "
          + ("Valid JSON and reply language within the limits." if all(
              c["valid_json_ok"] and c["reply_lang_ok"] for c in sel["checks"][v].values())
             else "Valid JSON or reply language out of the limits.")
          for v in sel["checks"]),
        f"- **Deployed: {sel['variant']}**, threshold {sel['threshold']:.3f}: {sel['reason']}.",
        "",
    ]  # fmt: skip
    if lat:
        th = [
            k
            for k in next(v for k, v in lat.items() if isinstance(v, dict))
            if k.startswith("threads_")
        ]
        lat_sets = lat.get("sets") or sets
        source = ("each development set" if len(lat_sets) == len(sets) else
                  "each of " + " and ".join(names.get(k, k).split(" (")[0] for k in lat_sets))  # fmt: skip
        lines += [
            "## Latency and memory",
            "",
            f"Batch 1 on the CPU ({lat['cpu']}), {lat['messages_per_set']} messages from {source} "
            f"(seed {lat['seed']}), after warm-up; each system in its own process, with Windows power "
            "throttling turned off for that process (D-034). \"With confidence\" adds the 21-intent "
            "scores that decide the route; it is timed on its own.",
            "",
            "| System | Threads | Answer: median (p95) | With confidence: median (p95) | Answer tokens/s | Peak memory | Load | File |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *(f"| {s} | {r['threads']} | {r['answer']['median_s']:.3f} s ({r['answer']['p95_s']:.3f}) | "
              f"{r['full']['median_s']:.3f} s ({r['full']['p95_s']:.3f}) | {r['answer_tokens_per_s']:.0f} | "
              f"{r['rss_peak_mb']:.0f} MB | {r['load_s']:.1f} s | {r['file_mb']:.0f} MB |"
              for s, by in lat.items() if isinstance(by, dict) for t in th for r in [by[t]]),
            "",
            "pytorch-fp32 is the setting of D-032 (no cache). Its file is the PyTorch checkpoint. The "
            "PyTorch rows include PyTorch's own memory; the ONNX rows never import PyTorch, as in the "
            "service.",
            "",
        ]  # fmt: skip
    if test:
        tm, ta = test["metrics"], test["agreement"]
        n = tm["pytorch-fp32"]["test"]["n"]
        lines += [
            "## Human test set: accuracy before and after quantization",
            "",
            f"The English part (64 messages, SHA-256 `{test['test_sha256'][:12]}…`), scored once per "
            "system after the choice was committed (D-034). "
            + rep.get("test_note", "The test messages come from the test splits of the datasets whose "
                                   "train splits D-033 trained on (see its caveat)."),
            "",
            "| System | Intent accuracy (95% interval) | Macro-F1 | Valid JSON | Reply language | Same intent as PyTorch float32 | Answered at the dev threshold (accuracy) |",
            "|---|---|---:|---:|---:|---:|---|",
            *(f"| {s} | {rate(round(tm[s]['test']['intent_accuracy'] * n), n)} | {pct(tm[s]['test']['intent_macro_f1'])} | "
              f"{pct(tm[s]['test']['valid_json'])} | {pct(tm[s]['test']['reply_lang'])} | "
              + ("–" if s == "pytorch-fp32" else pct(ta[s]["test"]["same_intent"])) + " | "
              + f"{pct(test['coverage_at_dev_threshold'][s]['test']['coverage'])} "
              + f"({pct(test['coverage_at_dev_threshold'][s]['test']['accuracy'])}) |" for s in tm),
            "",
            rep.get("gpu_reference", "Phase 6b scored the same model in bf16 on the GPU: 84.4% "
                                     "([phase6b_eval.md](phase6b_eval.md))."),
            "",
        ]  # fmt: skip
        judge_path = Path(p["judge_dir"]) / "reply_judge.jsonl"
        if judge_path.exists():
            judged = [j for j in map(json.loads, judge_path.read_text(encoding="utf-8").splitlines())
                      if j["good"] is not None]  # fmt: skip
            lines += [
                f"Qwen judged the deployed variant's {len(judged)} valid test replies with the D-032 "
                "prompt: " + ", ".join(f"{q.replace('_', ' ')} {rate(sum(j[q] is True for j in judged), len(judged))}"
                                        for q in ("answers", "polite_clear", "language", "safe"))
                + f". **All four yes: {rate(sum(j['good'] for j in judged), len(judged))}.** "
                + rep.get("judge_reference", "Phase 6b, the same model unquantized: 92.2% (59/64)."),
                "",
                "Broken answers of the deployed variant on the test set: "
                + "; ".join(f"{x['id']} ({'invalid JSON' if not x['valid'] else 'reply: ' + repr(x['reply'][:70])})"
                            for x in broken(out / f"test_test_onnx-{sel['variant']}.jsonl", cfg)) + ".",
                "",
            ]  # fmt: skip
    lines += guard_section(cfg, out, dev, bool(test), rep.get("guard_note", GUARD_NOTE))
    Path(p["results_md"]).write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
