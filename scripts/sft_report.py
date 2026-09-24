"""Phase 5 results page: the SFT data, the sweeps, the D-029 selection rule and the chosen model.

Usage: uv run python scripts/sft_report.py --config configs/sft/report.yaml
Reads <checkpoints_dir>/<run>/metrics.jsonl for every run in report.rounds (a round whose runs are
not all finished is left out), picks each round's checkpoint with the D-029 rule, then the better
round winner with the same rule (D-031), and writes report.results_md, report.plot and
report.selected (the chosen checkpoint, for Phase 6). Error analysis and examples come from the
chosen checkpoint's validation predictions. Only the SFT validation split is used.
"""

import json
import random
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from danalm.config import config_from_cli  # noqa: E402
from danalm.sft.evaluate import select_checkpoint  # noqa: E402


def cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ⏎ ")


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def read_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    cfg = config_from_cli(__doc__)
    r = cfg["report"]
    ckpt = Path(r["checkpoints_dir"])
    rounds: dict[str, list[dict]] = {}
    for rnd, runs in r["rounds"].items():
        cands = []
        for lr, run in runs.items():
            path = ckpt / run / "metrics.jsonl"
            rows = (
                [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
                if path.exists()
                else []
            )
            if len(rows) < r["epochs"]:
                cands = []
                break
            cands += [m | {"round": rnd, "run": run, "lr": lr} for m in rows]
        if cands:
            rounds[rnd] = cands
    if not rounds:
        raise SystemExit("no finished round yet")
    winners = {rnd: select_checkpoint(c, r["tie"]) for rnd, c in rounds.items()}
    best = select_checkpoint(list(winners.values()), r["tie"])
    chosen_path = ckpt / best["run"] / f"epoch_{best['epoch']}.pt"
    selected = {"round": best["round"], "run": best["run"], "lr": best["lr"], "epoch": best["epoch"],
                "checkpoint": str(chosen_path), "rule": "D-029 (highest greedy intent accuracy; "
                f"within {100 * r['tie']:.1f} points: higher valid JSON, then lower loss)",
                "metrics": {k: v for k, v in best.items() if k not in ("round", "run", "lr")}}  # fmt: skip
    Path(r["selected"]).write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for rnd, cands in rounds.items():
        for lr in dict.fromkeys(c["lr"] for c in cands):
            cs = [c for c in cands if c["lr"] == lr]
            style = "-" if rnd == "round1" else "--"
            ep = [c["epoch"] for c in cs]
            axes[0].plot(
                ep,
                [100 * c["intent_accuracy"] for c in cs],
                style,
                marker="o",
                label=f"{rnd} lr {lr}",
            )
            axes[1].plot(
                ep, [100 * c["valid_json"] for c in cs], style, marker="o", label=f"{rnd} lr {lr}"
            )
            axes[2].plot(ep, [c["val_loss"] for c in cs], style, marker="o", label=f"{rnd} lr {lr}")
    for ax, title in zip(
        axes,
        ["Intent accuracy (%)", "Valid JSON (%)", "Validation loss (answer tokens)"],
        strict=True,
    ):
        ax.set(title=title, xlabel="epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(r["plot"], dpi=110)

    preds = [json.loads(x) for x in (ckpt / best["run"] / f"val_predictions_epoch{best['epoch']}.jsonl")
             .read_text(encoding="utf-8").splitlines()]  # fmt: skip
    confusions = Counter((p["intent"], p["pred_intent"] or "(invalid answer)") for p in preds
                         if p["pred_intent"] != p["intent"])  # fmt: skip
    rng = random.Random(cfg["seed"])
    examples = []
    for v in sorted({p["variety"] for p in preds}):
        right = [p for p in preds if p["variety"] == v and p["pred_intent"] == p["intent"]]
        wrong = [p for p in preds if p["variety"] == v and p["pred_intent"] != p["intent"]]
        examples += rng.sample(right, min(2, len(right))) + rng.sample(wrong, min(2, len(wrong)))

    data = read_json(r["data_stats"]) or {}
    sample = read_json(r["reply_check_sample"]) or {}
    full = read_json(r["reply_check_all"]) or {}
    m = best
    lines = [
        "# Phase 5 results: supervised fine-tuning",
        "",
        f"Generated on {date.today().isoformat()} by `scripts/sft_report.py`; do not edit by hand. "
        "Setup: D-029 and D-031 in [DECISIONS.md](../DECISIONS.md). All numbers are on the SFT "
        "validation split (synthetic, written by the same teacher as the training data); the human "
        "test set is scored only in Phase 6.",
        "",
        "## Data",
        "",
        f"- Reply check (Qwen as proofreader): the 500-reply sample estimated "
        f"{pct(sample.get('estimated_broken_share_of_train'))} broken replies, above the 3% bar, so every reply "
        f"was judged: {full.get('train_broken', '–')} of the training and {full.get('val_broken', '–')} of the "
        "validation replies were marked broken. The judge missed both canary replies, so some broken "
        "replies remain.",
        f"- Broken replies rewritten and accepted: {data.get('train/reply_rewritten', 0) + data.get('val/reply_rewritten', 0):,}; "
        f"dropped: {data.get('train/dropped_broken_reply', 0) + data.get('val/dropped_broken_reply', 0):,}.",
        f"- Top-up examples added to training: {data.get('topup/added', 0):,} (dropped: judge disagreed "
        f"{data.get('topup/judge_disagreed', 0):,}, broken reply {data.get('topup/broken_reply', 0):,}, duplicate "
        f"{data.get('topup/duplicate', 0):,}, near copy of a validation message {data.get('topup/near_copy_of_val', 0):,}).",
        f"- Final data: {data.get('train/examples', 0):,} training and {data.get('val/examples', 0):,} validation examples.",
        "",
        "## Sweep and selection",
        "",
        f"![SFT curves]({Path(r['plot']).name})",
        "",
        "| Round | Peak LR | Epoch | Val loss | Valid JSON | Intent acc. | Macro-F1 | Reply language | Intent by likelihood |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {c['round']} | {c['lr']} | {c['epoch']} | {c['val_loss']:.3f} | {pct(c['valid_json'])} | "
            f"{pct(c['intent_accuracy'])} | {pct(c['intent_macro_f1'])} | {pct(c['reply_lang'])} | "
            f"{pct(c['lik_intent_accuracy'])} |"
            for cands in rounds.values() for c in cands
        ),
        "",
        *(f"- {rnd} winner (D-029 rule): lr {w['lr']}, epoch {w['epoch']}, intent accuracy "
          f"{pct(w['intent_accuracy'])}, valid JSON {pct(w['valid_json'])}." for rnd, w in winners.items()),
        f"- **Chosen:** {best['round']}, lr {best['lr']}, epoch {best['epoch']} (`{chosen_path.as_posix()}`).",
        "",
        "## The chosen model on the validation split",
        "",
        f"Valid JSON {pct(m['valid_json'])} (parses: {pct(m['parse_rate'])}, unfinished answers: "
        f"{pct(m['unfinished'])}); intent accuracy {pct(m['intent_accuracy'])}, macro-F1 "
        f"{pct(m['intent_macro_f1'])}; reply in an allowed language {pct(m['reply_lang'])}; intent by "
        f"likelihood {pct(m['lik_intent_accuracy'])}.",
        "",
        f"Coverage (D-030, on this development split): at confidence ≥ "
        f"{m['coverage']['threshold'] if m['coverage']['threshold'] is not None else '–'}, the model answers "
        f"{pct(m['coverage']['coverage'])} of the messages with {pct(m['coverage']['accuracy'])} intent accuracy. "
        "Phase 6 fixes this threshold here and applies it to the test set.",
        "",
        "| Variety | Messages | Valid JSON | Intent acc. | Macro-F1 | Reply language |",
        "|---|---:|---:|---:|---:|---:|",
        *(f"| {v} | {x['n']} | {pct(x['valid_json'])} | {pct(x['intent_accuracy'])} | "
          f"{pct(x['intent_macro_f1'])} | {pct(x['reply_lang'])} |" for v, x in m["per_variety"].items()),
        "",
        "Most frequent confusions (gold → predicted):",
        "",
        "| Gold | Predicted | Count |",
        "|---|---|---:|",
        *(f"| {g} | {p} | {n} |" for (g, p), n in confusions.most_common(r["top_confusions"])),
        "",
        f"Examples (two right and two wrong per variety, seed {cfg['seed']}):",
        "",
        "| Variety | Message | Gold | Model output |",
        "|---|---|---|---|",
        *(f"| {p['variety']} | {cell(p['message'])} | {p['intent']} | {cell(p['output'])} |" for p in examples),
    ]  # fmt: skip
    Path(r["results_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
