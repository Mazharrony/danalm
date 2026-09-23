"""Build the final SFT dataset from judge-verified examples.

Usage: uv run python scripts/build_sft_dataset.py --config configs/sft/build.yaml
Keeps rows the judge agreed with, adds the exact training target {"intent": ..., "reply": ...},
splits train/val by a seeded hash of the message, and records the dataset in the ledger.
Writes <build.out_dir>/{train,val}.jsonl and stats.json.
"""

import json
from collections import Counter
from datetime import date
from pathlib import Path

from tokenizers import Tokenizer

from danalm.config import config_from_cli
from danalm.data import ledger
from danalm.data.hub import text_stats
from danalm.data.pipeline import in_val


def target(row: dict) -> str:
    """The exact JSON string the model must produce."""
    return json.dumps({"intent": row["intent"], "reply": row["reply"]}, ensure_ascii=False)


def main() -> None:
    cfg = config_from_cli(__doc__)
    b = cfg["build"]
    out = Path(b["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in b["inputs"]:
        with open(path, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    kept = [r for r in rows if r["agree"]]
    splits: dict[str, list[dict]] = {"train": [], "val": []}
    for r in kept:
        example = {k: r[k] for k in ("message", "intent", "reply", "variety", "lang", "domain")}
        example["target"] = target(r)
        splits["val" if in_val(r["message"], cfg["seed"], b["val_frac"]) else "train"].append(
            example
        )
    for name, examples in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)

    tok = Tokenizer.from_file(b["tokenizer_file"])
    texts = [f"{r['message']}\n{target(r)}" for r in kept]
    tokens = sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))
    cells = Counter((r["intent"], r["variety"]) for r in kept)
    stats = {
        "verified_rows": len(rows),
        "kept": len(kept),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "tokens": tokens,
        "by_variety": dict(Counter(r["variety"] for r in kept)),
        "by_intent": dict(Counter(r["intent"] for r in kept)),
        "smallest_cells": [
            [f"{i}/{v}", n] for (i, v), n in sorted(cells.items(), key=lambda x: x[1])[:5]
        ],
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    text = text_stats(texts)
    entry = {
        "id": cfg["run_name"],
        "source": "judge-verified examples from " + ", ".join(b["inputs"]),
        "licence": "Apache-2.0 (teacher outputs)",
        "kind": "synthetic",
        "collected_on": date.today().isoformat(),
        "used_for": b["used_for"],
        "docs_raw": len(rows),
        "docs_kept": len(kept),
        "utf8_bytes": text["utf8_bytes"],
        "words": text["words"],
        "tokens": {b["tokenizer_name"]: tokens},
        "notes": f"train {len(splits['train']):,} / val {len(splits['val']):,}; "
        "message + JSON target; not reviewed by a native speaker",
    }
    ledger.write_markdown(cfg["ledger"]["markdown"], ledger.upsert(cfg["ledger"]["jsonl"], [entry]))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
