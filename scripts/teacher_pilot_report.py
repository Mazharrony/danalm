"""Compare the two teacher-pilot runs and apply the pre-declared decision rule.

Usage: uv run python scripts/teacher_pilot_report.py --config configs/sft/pilot_report.yaml
Reads each run's stats.json and verified.jsonl (from verify_sft.py) and the judge summary, and
writes the results page given by report.results_md.
"""

import json
import random
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli

VARIETIES = ["gulf_arabic", "english", "arabizi", "mixed"]


def distinct2(texts: list[str]) -> float:
    """Unique word bigrams / all word bigrams: higher means more varied wording."""
    grams = [tuple(t.split()[i : i + 2]) for t in texts for i in range(len(t.split()) - 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def summarize(run_dir: Path, judge_seconds: float) -> dict[str, Any]:
    stats = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))
    verified = (run_dir / "verified.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in verified]
    parsed = stats["examples_parsed"]
    usable = [r for r in rows if r["agree"]]
    s: dict[str, Any] = {
        "requests": stats["requests"],
        "parsed": parsed,
        "kept": len(rows),
        "usable": len(usable),
        "usable_rate": len(usable) / parsed,
        "judge_agreement": len(usable) / max(1, len(rows)),
        "gen_seconds": stats["seconds"],
        "judge_seconds": judge_seconds,
        "tokens_per_s": stats["tokens_per_s"],
        "reasons": {k: v for k, v in stats.items() if "/" not in k and k not in (
            "requests", "examples_parsed", "short_answers", "completion_tokens", "seconds",
            "tokens_per_s", "kept_per_min")},  # fmt: skip
        "by_variety": {},
    }
    for v in VARIETIES:
        vrows = [r for r in rows if r["variety"] == v]
        v_parsed = sum(n for k, n in stats.items() if k.startswith(f"{v}/"))
        # the dialect filter only sees rows that passed both language checks
        checked = sum(stats.get(f"{v}/{k}", 0) for k in ("kept", "non_gulf_dialect", "duplicate", "near_duplicate"))  # fmt: skip
        s["by_variety"][v] = {
            "parsed": v_parsed,
            "kept": len(vrows),
            "usable": sum(r["agree"] for r in vrows),
            "non_gulf_rate": stats.get(f"{v}/non_gulf_dialect", 0) / checked if checked else None,
            "distinct2": distinct2([r["message"] for r in vrows]),
            "avg_words": sum(len(r["message"].split()) for r in vrows) / max(1, len(vrows)),
        }
    gulf = [v for v in ("gulf_arabic", "mixed") if s["by_variety"][v]["non_gulf_rate"] is not None]
    s["non_gulf_rate"] = sum(s["by_variety"][v]["non_gulf_rate"] for v in gulf) / max(1, len(gulf))
    s["seconds_per_usable"] = (s["gen_seconds"] + judge_seconds) / max(1, len(usable))
    s["confusions"] = Counter(
        f"{r['intent']} → {r['judge_label']}" for r in rows if not r["agree"]
    ).most_common(8)
    s["rows"] = rows
    return s


def decide(base: dict, cand: dict, r: dict) -> dict[str, Any]:
    d = r["decision"]
    quality_gain = cand["usable_rate"] / base["usable_rate"] - 1
    non_gulf_drop = (
        1 - cand["non_gulf_rate"] / base["non_gulf_rate"] if base["non_gulf_rate"] else 0.0
    )
    hours = {k: s["seconds_per_usable"] * r["full_run_target"] / 3600 for k, s in (("baseline", base), ("candidate", cand))}  # fmt: skip
    better = quality_gain >= d["min_quality_gain"] or non_gulf_drop >= d["or_non_gulf_drop"]
    fits = hours["candidate"] <= d["max_full_run_hours"]
    return {
        "quality_gain": quality_gain,
        "non_gulf_drop": non_gulf_drop,
        "full_run_hours": hours,
        "winner": "candidate" if better and fits else "baseline",
        "reason": ("better and fits the time budget" if better and fits
                   else "not better enough" if not better else "too slow for the full run"),  # fmt: skip
    }


def main() -> None:
    cfg = config_from_cli(__doc__)
    r = cfg["report"]
    judge = json.loads(sorted(Path(cfg["paths"]["runs_dir"]).glob("sft-pilot-verify-*/verify_summary.json"))[-1].read_text(encoding="utf-8"))  # fmt: skip
    judge_s = {Path(p).parent.name: f["seconds"] for p, f in judge["files"].items()}
    runs = {}
    for key in ("baseline", "candidate"):
        run_dir = Path(r[key]["dir"])
        runs[key] = summarize(run_dir, judge_s[run_dir.name])
    d = decide(runs["baseline"], runs["candidate"], r)

    names = {k: r[k]["name"] for k in runs}
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    lines = [
        "# Phase 2b results: teacher pilot",
        "",
        f"Measured on {date.today().isoformat()}: the same {runs['baseline']['requests']} "
        "requests (21 intents × 4 varieties, 8 examples each) sent to both teachers; every kept "
        f"example re-labelled blindly by the judge ({r['judge']}). The decision rule was fixed in "
        "`configs/sft/pilot_report.yaml` before the run.",
        "",
        "Caveat: the judge is the candidate model, so it may slightly favour its own phrasing. "
        "No native speaker has reviewed any of this yet.",
        "",
        "| | " + " | ".join(names.values()) + " |",
        "|---|---:|---:|",
    ]

    def row(label: str, fn) -> None:  # noqa: E306
        lines.append(f"| {label} | " + " | ".join(fn(runs[k]) for k in runs) + " |")

    row("Examples parsed", lambda s: f"{s['parsed']:,}")
    row("Kept after filters", lambda s: f"{s['kept']:,} ({pct(s['kept'] / s['parsed'])})")
    row("Judge agrees with the intended label", lambda s: pct(s["judge_agreement"]))
    row(
        "**Usable** (kept and judge agrees)",
        lambda s: f"**{s['usable']:,} ({pct(s['usable_rate'])})**",
    )
    row("Non-Gulf dialect rate (Gulf Arabic + mixed)", lambda s: pct(s["non_gulf_rate"]))
    row("Generation speed", lambda s: f"{s['tokens_per_s']:.0f} tok/s")
    row("Seconds per usable example (generate + judge)", lambda s: f"{s['seconds_per_usable']:.2f}")
    row(f"Estimated full run ({r['full_run_target']:,} usable)",
        lambda s: f"{s['seconds_per_usable'] * r['full_run_target'] / 3600:.1f} h")  # fmt: skip
    lines += [
        "",
        "## Per variety",
        "",
        "| Variety | Metric | " + " | ".join(names.values()) + " |",
        "|---|---|---:|---:|",
    ]
    for v in VARIETIES:
        for label, key, fmt in (("usable / parsed", None, None), ("non-Gulf rate", "non_gulf_rate", pct),
                                ("distinct-2 (variety)", "distinct2", lambda x: f"{x:.2f}"),
                                ("avg words per message", "avg_words", lambda x: f"{x:.1f}")):  # fmt: skip
            cells = []
            for k in runs:
                bv = runs[k]["by_variety"][v]
                if key is None:
                    cells.append(f"{bv['usable']}/{bv['parsed']}")
                else:
                    cells.append("—" if bv[key] is None else fmt(bv[key]))
            lines.append(f"| {v} | {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Rejection reasons",
        "",
        "| Reason | " + " | ".join(names.values()) + " |",
        "|---|---:|---:|",
    ]
    reasons = sorted({k for s in runs.values() for k in s["reasons"]} - {"kept"})
    for reason in reasons:
        lines.append(
            f"| {reason} | "
            + " | ".join(str(runs[k]["reasons"].get(reason, 0)) for k in runs)
            + " |"
        )
    lines += ["", "## Most common judge disagreements", ""]
    for k in runs:
        lines.append(
            f"- **{names[k]}:** " + ", ".join(f"{c} ({n})" for c, n in runs[k]["confusions"])
        )
    dec = r["decision"]
    lines += [
        "",
        "## Decision",
        "",
        f"Rule: the candidate wins if usable/parsed is at least {dec['min_quality_gain']:.0%} "
        f"higher **or** its non-Gulf rate is at least {dec['or_non_gulf_drop']:.0%} lower, **and** "
        f"the full run fits in {dec['max_full_run_hours']} h.",
        "",
        f"- Quality gain: {d['quality_gain']:+.1%}; non-Gulf drop: {d['non_gulf_drop']:+.1%}; "
        f"full run: {d['full_run_hours']['candidate']:.1f} h (candidate) vs "
        f"{d['full_run_hours']['baseline']:.1f} h (baseline)",
        f"- → **{names[d['winner']]}** ({d['reason']})",
        "",
        "## Random samples (usable examples)",
        "",
    ]
    rng = random.Random(cfg["seed"])
    for k in runs:
        lines.append(f"### {names[k]}\n")
        for v in VARIETIES:
            pool = [x for x in runs[k]["rows"] if x["variety"] == v and x["agree"]]
            for x in rng.sample(pool, min(r["samples_per_variety"], len(pool))):
                lines.append(f"- `{v}` **{x['intent']}**: {x['message']}  \n  → {x['reply']}")
        lines.append("")
    Path(r["results_md"]).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(d, indent=2))
    print(f"Results page: {r['results_md']}")


if __name__ == "__main__":
    main()
