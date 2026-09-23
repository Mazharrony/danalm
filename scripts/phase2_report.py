"""Write the Phase 2 results page: the pretraining corpus and the synthetic SFT set.

Usage: uv run python scripts/phase2_report.py --config configs/data/phase2_report.yaml
Every number comes from the run outputs (ledger, cleaning stats, token index, shard check, SFT
generation stats, judge labels, final SFT stats). Sections whose outputs do not exist yet are
marked as pending. Writes report.results_md.
"""

import json
import random
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli
from danalm.data import ledger

VARIETIES = ["gulf_arabic", "english", "arabizi", "mixed"]


def load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def pct(part: float, whole: float) -> str:
    if not whole:
        return "–"
    return "<0.1%" if 0 < part / whole < 0.0005 else f"{part / whole:.1%}"


def distinct2(texts: list[str]) -> float:
    """Unique word bigrams / all word bigrams: higher means more varied wording."""
    grams = [tuple(t.split()[i : i + 2]) for t in texts for i in range(len(t.split()) - 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def cell(text: str) -> str:
    """Make text safe inside a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def corpus_section(r: dict[str, Any], ledger_jsonl: str) -> list[str]:
    index, clean = load(r["token_index"]), load(r["clean_stats"])
    entries = [e for e in ledger.read(ledger_jsonl) if e["id"].startswith(r["ledger_prefix"])]
    tok_name = index["tokenizer"]
    chars: Counter = Counter()
    for split in index["splits"].values():
        for source, c in split["per_source"].items():
            chars[source] += c["chars"]
    total = sum(e["tokens"][tok_name] for e in entries)
    out = [
        "## Pretraining corpus",
        "",
        f"Tokenizer {tok_name} (sha256 `{index['tokenizer_sha256'][:12]}…`). Tokens are text tokens;"
        " the shards add one EOS per document.",
        "",
        "| Source | Licence | Docs (raw → kept) | Characters | Tokens | Share | Chars/token |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for e in sorted(entries, key=lambda e: -e["tokens"][tok_name]):
        name = e["id"].removeprefix(r["ledger_prefix"])
        n = e["tokens"][tok_name]
        out.append(
            f"| {name} | {cell(e['licence'])} | {e['docs_raw']:,} → {e['docs_kept']:,} | "
            f"{chars[name]:,} | {n:,} | {pct(n, total)} | {chars[name] / n:.2f} |"
        )
    docs_raw, docs_kept = sum(e["docs_raw"] for e in entries), sum(e["docs_kept"] for e in entries)
    out.append(
        f"| **Total** | | {docs_raw:,} → {docs_kept:,} | {sum(chars.values()):,} | {total:,} | "
        f"100% | {sum(chars.values()) / total:.2f} |"
    )
    splits = index["splits"]
    out += [
        "",
        "| Split | Tokens (with EOS) | Shards |",
        "|---|---:|---:|",
        *(
            f"| {name} | {s['tokens_with_eos']:,} | {len(s['files'])} |"
            for name, s in splits.items()
        ),
        "",
        "**Cleaning** (`scripts/prepare_data.py`): "
        + ", ".join(
            f"{k.removeprefix('drop_').replace('_', ' ')} {v:,}"
            for k, v in sorted(clean.items(), key=lambda kv: -kv[1])
            if k.startswith("drop_")
        )
        + f" dropped, of {clean['read']:,} documents read.",
        "",
        "**Language tags** (heuristic, D-011; the Arabizi tag has known false positives on"
        " English text such as model numbers): "
        + ", ".join(
            f"{k.removeprefix('lang_')} {v:,}"
            for k, v in sorted(clean.items(), key=lambda kv: -kv[1])
            if k.startswith("lang_")
        )
        + ".",
    ]
    if Path(r["shard_verify"]).exists():
        v = load(r["shard_verify"])
        out += [
            "",
            "**Shard check** (`scripts/verify_shards.py`): "
            + "; ".join(
                f"{name} {'OK' if s['ok'] else 'FAILED'} ({s['eos']:,} EOS for {s['docs']:,} docs,"
                f" max id {s['max_id']} < {s['vocab']}, {s['checked_docs']} docs decoded exactly"
                f" of which {s['boundary_docs']} cross a shard boundary"
                f"{', mismatches: ' + str(s['mismatched_docs']) if s['mismatched_docs'] else ''})"
                for name, s in v.items()
            )
            + ".",
        ]
    return out


def generation_section(r: dict[str, Any]) -> list[str]:
    d = Path(r["sft_dir"])
    stats, manifest = load(d / "stats.json"), load(d / "manifest.json")
    parsed = stats["examples_parsed"]
    out = [
        "## Synthetic SFT data",
        "",
        "### Generation",
        "",
        f"Teacher: {manifest['teacher']} (revision `{manifest['revision'][:8]}`), "
        f"{manifest['requests']:,} requests of {r['examples_per_request']} examples each. "
        f"Answered: {stats['requests']:,} (failed after retries: {stats.get('failed_requests', 0)};"
        f" reused from a crashed attempt: {stats.get('resumed_requests', 0)}). "
        f"{stats['completion_tokens']:,} teacher tokens, "
        f"{stats['seconds'] / 3600:.1f} h, {stats['tokens_per_s']:.0f} tok/s.",
        "",
        "| Variety | Parsed | Kept by the filters | Rejected (reason: count) |",
        "|---|---:|---:|---|",
    ]
    for v in VARIETIES:
        c = {k.split("/", 1)[1]: n for k, n in stats.items() if k.startswith(f"{v}/")}
        total = sum(c.values())
        reasons = ", ".join(f"{k} {n:,}" for k, n in sorted(c.items(), key=lambda kv: -kv[1]) if k != "kept")  # fmt: skip
        out.append(f"| {v} | {total:,} | {c.get('kept', 0):,} ({pct(c.get('kept', 0), total)}) | {reasons} |")  # fmt: skip
    out.append(f"| **all** | {parsed:,} | {stats['kept']:,} ({pct(stats['kept'], parsed)}) | |")
    return out


def judge_section(r: dict[str, Any]) -> list[str]:
    rows = jsonl(Path(r["sft_dir"]) / "verified.jsonl")
    out = [
        "",
        "### Judge (blind re-labelling by the same model)",
        "",
        "The judge sees only the message and the 21 intent descriptions. Rows where it disagrees"
        " with the intended label are dropped from the final set.",
        "",
        "| Variety | Kept rows | Judge agrees | No valid label |",
        "|---|---:|---:|---:|",
    ]
    for v in [*VARIETIES, None]:
        vr = [x for x in rows if v is None or x["variety"] == v]
        agree = sum(x["agree"] for x in vr)
        none = sum(x["judge_label"] is None for x in vr)
        name = v or "**all**"
        out.append(f"| {name} | {len(vr):,} | {agree:,} ({pct(agree, len(vr))}) | {none:,} |")
    by_intent: dict[str, list[bool]] = {}
    for x in rows:
        by_intent.setdefault(x["intent"], []).append(x["agree"])
    ranked = sorted(by_intent.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    out += [
        "",
        "Agreement per intent, lowest first:",
        "",
        "| Intent | Rows | Agrees |",
        "|---|---:|---:|",
        *(f"| {i} | {len(a):,} | {pct(sum(a), len(a))} |" for i, a in ranked),
        "",
        "Most frequent disagreements (intended → judge):",
        "",
        *(
            f"- {pair}: {n:,}"
            for pair, n in Counter(
                f"{x['intent']} → {x['judge_label']}" for x in rows if not x["agree"]
            ).most_common(r["top_confusions"])
        ),
    ]
    return out


def final_section(r: dict[str, Any]) -> list[str]:
    d = Path(r["sft_final_dir"])
    stats, train = load(d / "stats.json"), jsonl(d / "train.jsonl")
    out = [
        "",
        "### Final SFT set",
        "",
        f"{stats['kept']:,} examples (train {stats['train']:,}, val {stats['val']:,}),"
        f" {stats['tokens']:,} tokens (message + JSON target, {r['tokenizer_name']}).",
        "",
        "| Variety | Examples | Avg words per message | Distinct word bigrams (messages) |",
        "|---|---:|---:|---:|",
    ]
    for v in VARIETIES:
        msgs = [x["message"] for x in train if x["variety"] == v]
        avg = sum(len(m.split()) for m in msgs) / max(1, len(msgs))
        out.append(f"| {v} | {stats['by_variety'].get(v, 0):,} | {avg:.1f} | {distinct2(msgs):.2f} |")  # fmt: skip
    counts = sorted(stats["by_intent"].items(), key=lambda kv: kv[1])
    out += [
        "",
        f"Examples per intent: min {counts[0][1]:,} ({counts[0][0]}), max {counts[-1][1]:,}"
        f" ({counts[-1][0]}). Smallest intent × variety cells: "
        + ", ".join(f"{name} {n}" for name, n in stats["smallest_cells"])
        + ".",
        "",
        f"Random training examples (seed {r['sample_seed']}):",
        "",
        "| Variety | Intent | Message | Reply |",
        "|---|---|---|---|",
    ]
    rng = random.Random(r["sample_seed"])
    for v in VARIETIES:
        pool = [x for x in train if x["variety"] == v]
        for x in rng.sample(pool, min(r["samples_per_variety"], len(pool))):
            out.append(f"| {v} | {x['intent']} | {cell(x['message'])} | {cell(x['reply'])} |")
    return out


def main() -> None:
    cfg = config_from_cli(__doc__)
    r = cfg["report"]
    lines = [
        "# Phase 2 results: data",
        "",
        f"Generated on {date.today().isoformat()} by `scripts/phase2_report.py` from the run"
        " outputs; do not edit by hand. Decisions: D-019 to D-022 in"
        " [DECISIONS.md](../DECISIONS.md). No native speaker has reviewed the synthetic data yet.",
        "",
        *corpus_section(r, cfg["ledger"]["jsonl"]),
    ]
    sft = Path(r["sft_dir"])
    steps = [
        (sft / "stats.json", generation_section),
        (sft / "verified.jsonl", judge_section),
        (Path(r["sft_final_dir"]) / "stats.json", final_section),
    ]
    for needed, section in steps:
        if not needed.exists():
            lines += ["", f"*{section.__name__.removesuffix('_section')}: pending ({needed.as_posix()} not found).*"]  # fmt: skip
            break
        lines += section(r)
    out = Path(r["results_md"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
