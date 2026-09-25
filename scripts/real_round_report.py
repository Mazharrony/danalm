"""Phase 5b results page: the real-message round (D-033), its selection rule and the chosen model.

Usage: uv run python scripts/real_round_report.py --config configs/sft/report_real.yaml
Reads <checkpoints_dir>/<run>/metrics.jsonl of every run in report.runs. Among the checkpoints
whose SFT-validation intent accuracy is at least report.min_sft_val, it picks the highest intent
accuracy on the real dev set; within report.tie of the best, the higher real-dev valid-JSON rate,
then the higher SFT-validation accuracy. Writes report.selected (the chosen checkpoint, with the
coverage threshold fixed on the real dev set, for Phase 6), report.plot and report.results_md.
The human test set is not used.
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


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ⏎ ")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pick(cands: list[dict[str, Any]], r: dict[str, Any]) -> dict[str, Any]:
    """The D-033 rule."""
    eligible = [c for c in cands if c["intent_accuracy"] >= r["min_sft_val"]]
    if not eligible:
        raise SystemExit(f"no checkpoint reaches {r['min_sft_val']:.3f} SFT-validation accuracy")
    top = max(c["dev"]["intent_accuracy"] for c in eligible)
    tied = [c for c in eligible if c["dev"]["intent_accuracy"] >= top - r["tie"]]
    return max(tied, key=lambda c: (c["dev"]["valid_json"], c["intent_accuracy"]))


def main() -> None:
    cfg = config_from_cli(__doc__)
    r = cfg["report"]
    ckpt = Path(r["checkpoints_dir"])
    cands = []
    for label, run in r["runs"].items():
        for line in (ckpt / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            m = json.loads(line)
            cands.append(m | {"run": run, "label": label, "dev": m["dev_real"]})
    best = pick(cands, r)
    path = ckpt / best["run"] / f"epoch_{best['epoch']}.pt"
    dev = best["dev"]
    selected = {"run": best["run"], "label": best["label"], "epoch": best["epoch"], "checkpoint": str(path),
                "rule": "D-033 (highest real-dev intent accuracy with SFT-validation accuracy >= "
                        f"{r['min_sft_val']}; within {100 * r['tie']:.1f} points: real-dev valid JSON, "
                        "then SFT-validation accuracy)",
                "metrics": {"coverage": dev["coverage"], "dev_real": dev,
                            "sft_val": {k: best[k] for k in ("intent_accuracy", "intent_macro_f1",
                                                             "valid_json", "reply_lang", "val_loss")}}}  # fmt: skip
    Path(r["selected"]).write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8", newline="\n")  # fmt: skip

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label in r["runs"]:
        cs = [c for c in cands if c["label"] == label]
        axes[0].plot([c["epoch"] for c in cs], [100 * c["dev"]["intent_accuracy"] for c in cs], marker="o", label=label)  # fmt: skip
        axes[1].plot([c["epoch"] for c in cs], [100 * c["intent_accuracy"] for c in cs], marker="o", label=label)  # fmt: skip
    base = read_json(r["baseline_out"]).get("metrics", {})
    if base:
        axes[0].axhline(100 * base["intent_accuracy"], color="grey", ls="--", label="Phase 5 model")
    axes[1].axhline(100 * r["min_sft_val"], color="grey", ls=":", label="guard (D-033)")
    for ax, title in zip(axes, ["Real dev set: intent accuracy (%)", "SFT validation: intent accuracy (%)"], strict=True):  # fmt: skip
        ax.set(title=title, xlabel="epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(r["plot"], dpi=110)

    split = read_json(r["split_stats"])
    data = read_json(r["data_stats"])
    preds = [json.loads(x) for x in (ckpt / best["run"] / f"dev_real_predictions_epoch{best['epoch']}.jsonl")
             .read_text(encoding="utf-8").splitlines()]  # fmt: skip
    base_preds = read_json(r["baseline_out"]).get("predictions", [])
    intents = sorted(Counter(p["intent"] for p in preds).items(), key=lambda kv: -kv[1])
    confusions = Counter((p["intent"], p["pred_intent"] or "(invalid answer)") for p in preds
                         if p["pred_intent"] != p["intent"])  # fmt: skip

    def acc_of(ps: list[dict], intent: str) -> str:
        sel = [p for p in ps if p["intent"] == intent]
        return f"{sum(p['pred_intent'] == p['intent'] for p in sel)}/{len(sel)}" if sel else "–"

    lines = [
        "# Phase 5b results: real customer messages (D-033)",
        "",
        f"Generated on {date.today().isoformat()} by `scripts/real_round_report.py`; do not edit by hand. "
        "Setup: D-033 in [DECISIONS.md](../DECISIONS.md). The human test set is not used here.",
        "",
        "## Data",
        "",
        f"- Real messages (Banking77 and CLINC150 train splits, CC-BY): {split.get('messages', 0):,} mapped; "
        f"{split.get('dropped_near_copy_of_test', 0)} dropped as near copies of test messages; "
        f"**{split.get('dev', 0)} held out as the real dev set**; {split.get('dropped_near_copy_of_dev', 0)} "
        f"training candidates dropped as near copies of dev messages; {split.get('train_candidates', 0):,} "
        "training candidates (at most 500 per intent).",
        *(f"- {name}: added {data.get(f'{name}/added', 0):,} (dropped: judge disagreed "
          f"{data.get(f'{name}/judge_disagreed', 0):,}, broken reply {data.get(f'{name}/broken_reply', 0):,}, "
          f"duplicate {data.get(f'{name}/duplicate', 0):,}, near copy of a test or dev message "
          f"{data.get(f'{name}/near_copy_of_test_or_real_dev', 0):,})." for name in r["sources"]),
        f"- Final data (final-v4): {data.get('train/examples', 0):,} training and {data.get('val/examples', 0):,} "
        "validation examples (the validation split is unchanged).",
        "",
        "## Sweep and selection",
        "",
        f"![curves]({Path(r['plot']).name})",
        "",
        "| Run | Epoch | Real dev: intent acc. | Real dev: macro-F1 | Real dev: valid JSON | SFT val: intent acc. |",
        "|---|---:|---:|---:|---:|---:|",
        *(f"| {c['label']} | {c['epoch']} | {pct(c['dev']['intent_accuracy'])} | {pct(c['dev']['intent_macro_f1'])} | "
          f"{pct(c['dev']['valid_json'])} | {pct(c['intent_accuracy'])} |" for c in cands),
        "",
        f"- Phase 5 model on the same real dev set: {pct(base.get('intent_accuracy'))} intent accuracy, "
        f"{pct(base.get('intent_macro_f1'))} macro-F1." if base else "- Phase 5 model: not measured.",
        f"- **Chosen:** {best['label']}, epoch {best['epoch']} (`{path.as_posix()}`): real dev "
        f"{pct(dev['intent_accuracy'])} (macro-F1 {pct(dev['intent_macro_f1'])}), SFT validation "
        f"{pct(best['intent_accuracy'])}.",
        f"- Coverage on the real dev set (the threshold for Phase 6): at confidence ≥ "
        f"{'–' if dev['coverage']['threshold'] is None else format(dev['coverage']['threshold'], '.3f')}, the model "
        f"answers {pct(dev['coverage']['coverage'])} of the dev messages with {pct(dev['coverage']['accuracy'])} "
        "intent accuracy.",
        "",
        "## Per intent on the real dev set",
        "",
        "| Intent | Messages | Chosen model | Phase 5 model |",
        "|---|---:|---:|---:|",
        *(f"| {i} | {k} | {acc_of(preds, i)} | {acc_of(base_preds, i)} |" for i, k in intents),
        "",
        "Most frequent confusions of the chosen model (gold → predicted):",
        "",
        "| Gold | Predicted | Count |",
        "|---|---|---:|",
        *(f"| {g} | {p} | {n} |" for (g, p), n in confusions.most_common(10)),
    ]  # fmt: skip
    Path(r["results_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
