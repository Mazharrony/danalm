"""Compare the model sizes' sanity runs and apply the size rule fixed in advance (Phase 3).

Usage: uv run python scripts/phase3_report.py --config configs/model/report.yaml
Reads the latest runs/<candidate>-*/sanity.json of every candidate and writes report.results_md:
parameters, initial loss vs ln(vocab), the one-batch overfit, speed, peak VRAM, MFU and hours per
pass over the train shards, then the chosen size: the largest one that meets every rule.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli


def latest(runs_dir: str, name: str) -> dict[str, Any]:
    paths = sorted(Path(runs_dir).glob(f"{name}-*/sanity.json"))
    if not paths:
        raise FileNotFoundError(f"no sanity run for {name}; run scripts/model_sanity.py first")
    return {**json.loads(paths[-1].read_text(encoding="utf-8")), "run": paths[-1].parent.name}


def passes(s: dict[str, Any], rule: dict[str, Any]) -> list[str]:
    """The rule conditions a candidate fails (empty = it passes)."""
    failed = []
    if s["speed"]["hours_per_train_pass"] > rule["max_hours_per_pass"]:
        failed.append(f"> {rule['max_hours_per_pass']} h per pass")
    if abs(s["initial_loss"] - s["ln_vocab"]) > rule["max_init_loss_gap"]:
        failed.append("initial loss too far from ln(vocab)")
    if rule["overfit_must_pass"] and not s["overfit"]["passed"]:
        failed.append("did not overfit one batch")
    return failed


def main() -> None:
    cfg = config_from_cli(__doc__)
    r = cfg["report"]
    results = {name: latest(cfg["paths"]["runs_dir"], name) for name in r["candidates"]}
    rule = r["rule"]
    lines = [
        "# Phase 3 results: model sanity checks and size",
        "",
        f"Generated on {date.today().isoformat()} by `scripts/phase3_report.py` from the sanity"
        " runs; do not edit by hand. Decision: D-026 in [DECISIONS.md](../DECISIONS.md).",
        "",
        "| | " + " | ".join(n.removeprefix("model-") for n in results) + " |",
        "|---|" + "---:|" * len(results),
    ]
    rows = [
        ("Parameters (total)", lambda s: f"{s['params']['total'] / 1e6:.1f}M"),
        ("Non-embedding parameters", lambda s: f"{s['params']['non_embedding'] / 1e6:.1f}M"),
        ("Initial loss (ln vocab = {:.2f})".format(next(iter(results.values()))["ln_vocab"]),
         lambda s: f"{s['initial_loss']:.3f}"),
        ("Overfit one batch (first → last loss)",
         lambda s: f"{s['overfit']['first']:.2f} → {s['overfit']['last']:.3f}"),
        ("Training speed", lambda s: f"{s['speed']['tokens_per_s']:,} tok/s"),
        ("Peak VRAM", lambda s: f"{s['speed']['peak_vram_gb']} GB"),
        ("MFU (bf16 peak 61 TFLOPS)", lambda s: f"{s['speed']['mfu']:.0%}"),
        ("Hours per pass over the train shards",
         lambda s: f"{s['speed']['hours_per_train_pass']:.1f}"),
    ]  # fmt: skip
    for label, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(fmt(s) for s in results.values()) + " |")
    failed = {n: passes(s, rule) for n, s in results.items()}
    ok = [n for n in results if not failed[n]]
    chosen = max(ok, key=lambda n: results[n]["params"]["total"]) if ok else None
    lines += [
        "",
        f"**Rule (fixed before the runs, `configs/model/report.yaml`):** the largest size whose"
        f" pass over the train shards takes at most {rule['max_hours_per_pass']} h, whose initial"
        f" loss is within {rule['max_init_loss_gap']} of ln(vocab) and that overfits one batch.",
        "",
        *(f"- {n}: {'passes' if not f else 'fails: ' + ', '.join(f)}" for n, f in failed.items()),
        "",
        f"**Chosen: {chosen.removeprefix('model-') if chosen else 'none (no size meets the rule)'}**",
    ]
    Path(r["results_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
