"""Phase 5c results page: the question-type round (D-037), its selection rule and the chosen model.

Usage: uv run python scripts/qtype_round_report.py --config configs/sft/report_qtypes.yaml
Reads <checkpoints_dir>/<run>/metrics.jsonl of every run in report.runs. Among the checkpoints
whose SFT-validation intent accuracy is at least report.min_sft_val and whose real-dev intent
accuracy is at least report.min_real_dev, it picks the highest intent accuracy on the
question-type dev set; within report.tie of the best, the higher real-dev accuracy, then the
higher SFT-validation accuracy. Writes report.selected (with the coverage threshold fixed on the
real dev set, as in D-033), report.plot and report.results_md. When the reply-faithfulness
judgements exist (scripts/faithfulness_sample.py, then scripts/evaluate_teacher.py), the page
compares the earlier model and the chosen one on the same development replies. The human test
set is not used.
"""

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from danalm.config import config_from_cli  # noqa: E402
from danalm.eval.stats import wilson_interval  # noqa: E402


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def rate(k: int, n: int) -> str:
    if not n:
        return "–"
    lo, hi = wilson_interval(k, n)
    return f"{100 * k / n:.1f}% ({k}/{n}; {100 * lo:.0f}–{100 * hi:.0f}%)"


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def pick(cands: list[dict[str, Any]], r: dict[str, Any]) -> dict[str, Any]:
    """The D-037 rule."""
    eligible = [c for c in cands if c["intent_accuracy"] >= r["min_sft_val"]
                and c["dev_real"]["intent_accuracy"] >= r["min_real_dev"]]  # fmt: skip
    if not eligible:
        raise SystemExit("no checkpoint passes the D-037 guards")
    top = max(c["dev_qtype"]["intent_accuracy"] for c in eligible)
    tied = [c for c in eligible if c["dev_qtype"]["intent_accuracy"] >= top - r["tie"]]
    return max(tied, key=lambda c: (c["dev_real"]["intent_accuracy"], c["intent_accuracy"]))


def judged(path: Path) -> list[dict[str, Any]]:
    return [j for j in read_jsonl(path) if j.get("good") is not None]


def main() -> None:
    cfg = config_from_cli(__doc__)
    r = cfg["report"]
    ckpt = Path(r["checkpoints_dir"])
    cands = []
    for label, run in r["runs"].items():
        for line in (ckpt / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            cands.append(json.loads(line) | {"run": run, "label": label})
    best = pick(cands, r)
    path = ckpt / best["run"] / f"epoch_{best['epoch']}.pt"
    selected = {"run": best["run"], "label": best["label"], "epoch": best["epoch"], "checkpoint": str(path),
                "rule": f"D-037 (highest question-type-dev intent accuracy with SFT validation >= {r['min_sft_val']} "
                        f"and real dev >= {r['min_real_dev']}; within {100 * r['tie']:.1f} points: real dev, "
                        "then SFT validation)",
                "metrics": {"coverage": best["dev_real"]["coverage"], "dev_real": best["dev_real"],
                            "dev_qtype": best["dev_qtype"],
                            "sft_val": {k: best[k] for k in ("intent_accuracy", "intent_macro_f1",
                                                             "valid_json", "reply_lang", "val_loss")}}}  # fmt: skip
    Path(r["selected"]).write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8", newline="\n")  # fmt: skip

    base = read_json(r["baseline_out"])
    base_q = base.get("metrics", {})
    prev = read_json(r["previous_selected"]).get("metrics", {})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label in r["runs"]:
        cs = [c for c in cands if c["label"] == label]
        ep = [c["epoch"] for c in cs]
        axes[0].plot(
            ep, [100 * c["dev_qtype"]["intent_accuracy"] for c in cs], marker="o", label=label
        )
        axes[1].plot(
            ep, [100 * c["dev_real"]["intent_accuracy"] for c in cs], marker="o", label=label
        )
        axes[2].plot(ep, [100 * c["intent_accuracy"] for c in cs], marker="o", label=label)
    if base_q:
        axes[0].axhline(100 * base_q["intent_accuracy"], color="grey", ls="--", label="D-033 model")
    axes[1].axhline(100 * r["min_real_dev"], color="grey", ls=":", label="guard (D-037)")
    axes[2].axhline(100 * r["min_sft_val"], color="grey", ls=":", label="guard (D-037)")
    titles = ["Question-type dev: intent accuracy (%)", "Real dev: intent accuracy (%)",
              "SFT validation: intent accuracy (%)"]  # fmt: skip
    for ax, title in zip(axes, titles, strict=True):
        ax.set(title=title, xlabel="epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(r["plot"], dpi=110)

    data = read_json(r["data_stats"])
    splits = {name: read_json(p) for name, p in r["split_stats"].items()}
    preds = read_jsonl(ckpt / best["run"] / f"dev_qtype_predictions_epoch{best['epoch']}.jsonl")
    base_preds = base.get("predictions", [])
    intents = sorted(Counter(p["intent"] for p in preds).items(), key=lambda kv: -kv[1])
    confusions = Counter((p["intent"], p["pred_intent"] or "(invalid answer)") for p in preds
                         if p["pred_intent"] != p["intent"])  # fmt: skip

    def acc_of(ps: list[dict], intent: str) -> str:
        sel = [p for p in ps if p["intent"] == intent]
        return f"{sum(p['pred_intent'] == p['intent'] for p in sel)}/{len(sel)}" if sel else "–"

    lines = [
        "# Phase 5c results: question types the model had not seen (D-037)",
        "",
        f"Generated on {date.today().isoformat()} by `scripts/qtype_round_report.py`; do not edit by hand. "
        "Setup: D-037 in [DECISIONS.md](../DECISIONS.md). The human test set is not used here.",
        "",
        "## Data",
        "",
        *(f"- {name}: {s.get('messages', 0):,} messages mapped; {s.get('dropped_near_copy_of_test', 0)} dropped as "
          f"near copies of test messages; **{s.get('dev', 0):,} held out for the question-type dev set**; "
          f"{s.get('dropped_near_copy_of_dev', 0):,} training candidates dropped as near copies of dev messages; "
          f"{s.get('train_candidates', 0):,} training candidates." for name, s in splits.items()),
        *(f"- {name}: added {data.get(f'{name}/added', 0):,} (dropped: judge disagreed "
          f"{data.get(f'{name}/judge_disagreed', 0):,}, broken reply {data.get(f'{name}/broken_reply', 0):,}, "
          f"duplicate {data.get(f'{name}/duplicate', 0):,}, near copy of a test or dev message "
          f"{data.get(f'{name}/near_copy_of_test_or_real_dev', 0):,})." for name in r["sources"]),
        f"- Final data (final-v5): {data.get('train/examples', 0):,} training and {data.get('val/examples', 0):,} "
        "validation examples (the validation split is unchanged).",
        "",
        "## Sweep and selection",
        "",
        f"![curves]({Path(r['plot']).name})",
        "",
        "| Run | Epoch | Question-type dev | Real dev | SFT validation | Question-type dev: valid JSON |",
        "|---|---:|---:|---:|---:|---:|",
        *(f"| {c['label']} | {c['epoch']} | {pct(c['dev_qtype']['intent_accuracy'])} | "
          f"{pct(c['dev_real']['intent_accuracy'])} | {pct(c['intent_accuracy'])} | {pct(c['dev_qtype']['valid_json'])} |"
          for c in cands),
        "",
        (f"- The D-033 model on the same question-type dev set: {pct(base_q.get('intent_accuracy'))} intent accuracy, "
         f"{pct(base_q.get('intent_macro_f1'))} macro-F1; on the real dev set (D-033): "
         f"{pct(prev.get('dev_real', {}).get('intent_accuracy'))}." if base_q else "- D-033 model: not measured."),
        f"- **Chosen:** {best['label']}, epoch {best['epoch']} (`{path.as_posix()}`): question-type dev "
        f"{pct(best['dev_qtype']['intent_accuracy'])} (macro-F1 {pct(best['dev_qtype']['intent_macro_f1'])}), real dev "
        f"{pct(best['dev_real']['intent_accuracy'])}, SFT validation {pct(best['intent_accuracy'])}.",
        f"- Coverage on the real dev set (the threshold for the test): at confidence ≥ "
        f"{'–' if best['dev_real']['coverage']['threshold'] is None else format(best['dev_real']['coverage']['threshold'], '.3f')}, "
        f"the model answers {pct(best['dev_real']['coverage']['coverage'])} of the real dev messages with "
        f"{pct(best['dev_real']['coverage']['accuracy'])} intent accuracy.",
        "",
        "## Per intent on the question-type dev set",
        "",
        "| Intent | Messages | Chosen model | D-033 model |",
        "|---|---:|---:|---:|",
        *(f"| {i} | {k} | {acc_of(preds, i)} | {acc_of(base_preds, i)} |" for i, k in intents),
        "",
        "Most frequent confusions of the chosen model (gold → predicted):",
        "",
        "| Gold | Predicted | Count |",
        "|---|---|---:|",
        *(f"| {g} | {p} | {n} |" for (g, p), n in confusions.most_common(10)),
    ]  # fmt: skip
    before = judged(Path(r["faithfulness"]["before_dir"]) / "reply_judge.jsonl")
    after = judged(Path(r["faithfulness"]["after_dir"]) / "reply_judge.jsonl")
    if before and after:
        both = {j["id"] for j in before} & {j["id"] for j in after}
        b = {j["id"]: j for j in before if j["id"] in both}
        a = {j["id"]: j for j in after if j["id"] in both}
        n = len(both)
        lines += [
            "",
            "## Reply faithfulness (D-032 judge, the same development replies)",
            "",
            f"{n} development messages ({r['faithfulness']['per_set']} from each dev set, seeded) with a "
            "valid answer from both models, judged by Qwen with the D-032 prompt.",
            "",
            "| | D-033 model | Chosen model |",
            "|---|---:|---:|",
            *(f"| {q.replace('_', ' ')} | {rate(sum(b[i][q] is True for i in both), n)} | "
              f"{rate(sum(a[i][q] is True for i in both), n)} |" for q in ("answers", "polite_clear", "language", "safe")),
            f"| **all four yes** | {rate(sum(b[i]['good'] for i in both), n)} | {rate(sum(a[i]['good'] for i in both), n)} |",
            "",
            f"- Replies judged good only for the chosen model: {sum(a[i]['good'] and not b[i]['good'] for i in both)}; "
            f"only for the D-033 model: {sum(b[i]['good'] and not a[i]['good'] for i in both)}.",
        ]  # fmt: skip
    Path(r["results_md"]).write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
