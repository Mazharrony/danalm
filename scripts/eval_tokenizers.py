"""Measure tokenizer fertility on held-out text and pick DanaLM's tokenizer by pre-declared rules.

Usage: uv run python scripts/eval_tokenizers.py --config configs/tokenizer/eval.yaml
Writes fertility.json to the run folder and the results page given by eval.results_md.
"""

import json
import random
import re
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from danalm.config import config_from_cli
from danalm.data.pipeline import PLACEHOLDER, detect_lang, normalize
from danalm.tokenizer.fertility import fertility, train_flops_per_token
from danalm.utils.run import start_run
from danalm.utils.seed import set_seed

SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")
VARIETIES = ["MSA", "Gulf Arabic", "English", "Arabizi", "Mixed"]


def load_set(spec: dict[str, Any], e: dict[str, Any], seed: int) -> list[str]:
    """Normalized texts of one eval set, filtered as its spec says, seeded sample of max_texts."""
    with open(spec["path"], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    if "sources" in spec:
        rows = [r for r in rows if r.get("source") in spec["sources"]]
    texts = [normalize(r["text"], **e["normalize"]) for r in rows]
    if "mine_lang" in spec:
        low, high = e["mine_words"]
        texts = [
            s
            for t in texts
            for s in SENTENCE_END.split(t)
            if low <= len(s.split()) <= high and detect_lang(s) == spec["mine_lang"]
        ]
    if "keep_langs" in spec:
        texts = [t for t in texts if detect_lang(t) in spec["keep_langs"]]
    # Drop PII placeholders: ours encode them as one token, other tokenizers as ~3, which
    # would bias the comparison. Fertility is measured on natural text only.
    texts = [" ".join(PLACEHOLDER.sub(" ", t).split()) for t in texts]
    texts = [t for t in texts if t]
    random.Random(seed).shuffle(texts)
    return texts[: e["max_texts"]]


def load_tokenizer(spec: dict[str, Any], external_dir: str) -> Tokenizer:
    """One of ours from its file, or an external one's tokenizer.json at a pinned revision."""
    if "file" in spec:
        return Tokenizer.from_file(spec["file"])
    path = hf_hub_download(
        spec["repo_id"],
        "tokenizer.json",
        revision=spec["revision"],
        local_dir=Path(external_dir) / spec["name"],
    )
    return Tokenizer.from_file(path)


def decide(e: dict[str, Any], by_variety: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Apply the config's decision rules: first the pre-tokenizer, then the vocabulary size."""
    rules, ref = e["decision"], e["reference_model"]
    ours = [t for t in e["tokenizers"] if "vocab_size" in t]
    name = {(t["pretokenizer"], t["vocab_size"]): t["name"] for t in ours}
    sizes = sorted({t["vocab_size"] for t in ours})

    per_size = {}
    for v in sizes:
        std, az = by_variety[name["standard", v]], by_variety[name["arabizi", v]]
        per_size[v] = {
            "arabizi_gain": 1 - az["Arabizi"] / std["Arabizi"],
            "worst_other_loss": max(az[k] / std[k] - 1 for k in VARIETIES if k != "Arabizi"),
        }
    use_arabizi = all(
        s["arabizi_gain"] >= rules["arabizi_min_gain"]
        and s["worst_other_loss"] <= rules["other_max_loss"]
        for s in per_size.values()
    )
    pretokenizer = "arabizi" if use_arabizi else "standard"

    flops_per_word = {}
    for v in sizes:
        overall = mean(by_variety[name[pretokenizer, v]].values())
        flops = train_flops_per_token(v, ref["d_model"], ref["n_layers"])
        flops_per_word[v] = overall * flops
    best = min(flops_per_word.values())
    chosen = min(v for v in sizes if flops_per_word[v] <= best * (1 + rules["vocab_tie"]))
    return {  # string keys: W&B summaries and JSON need them
        "pretokenizer": pretokenizer,
        "pretokenizer_check": {str(v): c for v, c in per_size.items()},
        "flops_per_word": {str(v): f for v, f in flops_per_word.items()},
        "vocab_size": chosen,
        "tokenizer": name[pretokenizer, chosen],
    }


def to_markdown(cfg: dict[str, Any], sets: list[dict], results: dict, summary: dict) -> str:
    """The results page: eval sets, fertility per set and per variety, decision."""
    e = cfg["eval"]
    toks = [t["name"] for t in e["tokenizers"]]
    vocab = summary["vocab_sizes"]
    d = summary["decision"]
    lines = [
        "# Phase 1 results: tokenizer fertility",
        "",
        f"Measured on {date.today().isoformat()} with `scripts/eval_tokenizers.py` "
        f"(`configs/tokenizer/eval.yaml`, git commit `{summary['commit']}`).",
        "",
        "**Fertility** = tokens per word; lower is better. Words are whitespace-separated units of "
        "the normalized text (`danalm.data.pipeline.normalize`, diacritics stripped, alef forms "
        "kept), identical for every tokenizer. No special tokens are added, and PII placeholders "
        "are removed so no tokenizer gets credit for them.",
        "",
        "## Evaluation sets (held out: never used to train a DanaLM tokenizer)",
        "",
        "| Set | Variety | Kind | Texts | Words |",
        "|---|---|---|---:|---:|",
    ]
    for s in sets:
        lines.append(
            f"| {s['name']} | {s['variety']} | {s['kind']} | {s['texts']:,} | {s['words']:,} |"
        )
    lines += ["", "## Fertility per set", ""]
    lines.append("| Set | " + " | ".join(f"{t}<br>({vocab[t]:,})" for t in toks) + " |")
    lines.append("|---|" + "---:|" * len(toks))
    for s in sets:
        row = [f"{results[t][s['name']]['fertility']:.3f}" for t in toks]
        lines.append(f"| {s['name']} | " + " | ".join(row) + " |")
    lines += ["", "## Fertility per variety (mean of its sets)", ""]
    lines.append("| Variety | " + " | ".join(toks) + " |")
    lines.append("|---|" + "---:|" * len(toks))
    for var in [*VARIETIES, "Overall"]:
        row = [f"{summary['by_variety'][t][var]:.3f}" for t in toks]
        lines.append(f"| {var} | " + " | ".join(row) + " |")
    ref = e["reference_model"]
    rules = e["decision"]
    lines += [
        "",
        "## Decision (rules fixed in the config before the run)",
        "",
        f"1. **Pre-tokenizer:** use `arabizi` if, at every vocab size, Arabizi fertility drops by "
        f"at least {rules['arabizi_min_gain']:.0%} and no other variety gets more than "
        f"{rules['other_max_loss']:.0%} worse than `standard`.",
        "",
        "| Vocab | Arabizi gain | Worst change elsewhere |",
        "|---:|---:|---:|",
    ]
    for v, c in d["pretokenizer_check"].items():
        lines.append(f"| {int(v):,} | {c['arabizi_gain']:+.1%} | {c['worst_other_loss']:+.1%} |")
    lines += [
        "",
        f"   → **{d['pretokenizer']}**",
        "",
        f"2. **Vocabulary size:** lowest training FLOPs per word (overall fertility × FLOPs per "
        f"token of a reference model, d_model={ref['d_model']}, {ref['n_layers']} layers, tied "
        f"embeddings); the smaller size wins within {rules['vocab_tie']:.0%}.",
        "",
        "| Vocab | FLOPs per word (relative) | Embedding params at d_model=512 |",
        "|---:|---:|---:|",
    ]
    best = min(d["flops_per_word"].values())
    for v, f in d["flops_per_word"].items():
        lines.append(f"| {int(v):,} | {f / best:.3f} | {int(v) * ref['d_model'] / 1e6:.1f}M |")
    lines += ["", f"   → **{d['tokenizer']}** (vocab {d['vocab_size']:,})", ""]
    return "\n".join(lines)


def main() -> None:
    cfg = config_from_cli(__doc__)
    set_seed(cfg["seed"], cfg["deterministic"])
    run = start_run(cfg, job_type="tokenizer-eval")
    e = cfg["eval"]

    sets, texts = [], {}
    for spec in e["sets"]:
        texts[spec["name"]] = load_set(spec, e, cfg["seed"])
        words = sum(len(t.split()) for t in texts[spec["name"]])
        sets.append({**spec, "texts": len(texts[spec["name"]]), "words": words})
        print(f"{spec['name']:24s} {len(texts[spec['name']]):>6,} texts {words:>10,} words")

    results: dict[str, dict[str, dict]] = {}
    vocab_sizes = {}
    for spec in e["tokenizers"]:
        tok = load_tokenizer(spec, e["external_dir"])
        vocab_sizes[spec["name"]] = tok.get_vocab_size()
        results[spec["name"]] = {s["name"]: fertility(tok, texts[s["name"]]) for s in sets}

    by_variety = {}
    for t, per_set in results.items():
        by_variety[t] = {
            var: mean(per_set[s["name"]]["fertility"] for s in sets if s["variety"] == var)
            for var in VARIETIES
        }
    decision = decide(e, by_variety)
    for t in by_variety:
        by_variety[t]["Overall"] = mean(by_variety[t][v] for v in VARIETIES)

    summary = {
        "commit": (run.meta["git"]["commit"] or "unknown")[:8],
        "vocab_sizes": vocab_sizes,
        "by_variety": by_variety,
        "decision": decision,
    }
    (run.dir / "fertility.json").write_text(
        json.dumps({"sets": sets, "results": results, **summary}, indent=2), encoding="utf-8"
    )
    Path(e["results_md"]).parent.mkdir(parents=True, exist_ok=True)
    Path(e["results_md"]).write_text(
        to_markdown(cfg, sets, results, summary), "utf-8", newline="\n"
    )
    run.wandb.summary.update({"decision": decision, "by_variety": by_variety})
    run.wandb.finish()
    print(json.dumps({"by_variety": by_variety, "decision": decision}, indent=2))
    print(f"\nResults page: {e['results_md']}; details in {run.dir}")


if __name__ == "__main__":
    main()
