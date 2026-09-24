"""Build the SFT data for the overnight run (D-031 A and B): data/sft/final-v3.

Usage: uv run python scripts/assemble_sft_v3.py --config configs/sft/v3.yaml
1. Every example of the original set keeps its reply unless the full reply check marked it
   BROKEN. A broken reply is replaced by the teacher's rewrite when the rewrite passed the reply
   filters and was not marked BROKEN again; otherwise the example is dropped.
2. Top-up examples the label judge agreed with, and whose reply was not marked BROKEN, are added
   to the training split only. New messages that repeat a training or validation message
   exactly, or nearly repeat a validation message (MinHash), are dropped, so the validation split
   stays the original examples and every comparison uses the same development set.
Writes <out_dir>/train.jsonl, val.jsonl and stats.json; `target` is rebuilt from intent and reply.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasketch import MinHashLSH

from danalm.config import config_from_cli
from danalm.data.pipeline import minhash, normalize
from danalm.utils.run import write_provenance


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def target(intent: str, reply: str) -> str:
    """The exact JSON string the model must produce (as in scripts/build_sft_dataset.py)."""
    return json.dumps({"intent": intent, "reply": reply}, ensure_ascii=False)


def verdicts(judged_path: str) -> dict[tuple[str, int], str | None]:
    """{(split, index): verdict} for the counted replies of a check_replies.py run."""
    return {(j["split"], j["index"]): j["verdict"] for j in read_jsonl(judged_path) if j["counted"]}


def main() -> None:
    cfg = config_from_cli(__doc__)
    a = cfg["assemble"]
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_provenance(out, cfg)
    stats: Counter = Counter()
    keys = ["message", "intent", "reply", "variety", "lang", "domain"]

    original = {s: read_jsonl(Path(a["original_dir"]) / f"{s}.jsonl") for s in ("train", "val")}
    judged = verdicts(a["judged_all"])
    rewrites = read_jsonl(a["rewrites"])
    rewrite_ok = verdicts(a["rewrites_judged"])
    fixed = {(r["split"], r["index"]): r["reply"] for n, r in enumerate(rewrites)
             if rewrite_ok.get(("generated", n)) != "BROKEN"}  # fmt: skip
    splits: dict[str, list[dict]] = {}
    for s, rows in original.items():
        kept = []
        for i, row in enumerate(rows):
            if judged.get((s, i)) != "BROKEN":
                kept.append({k: row[k] for k in keys} | {"reply_rewritten": False})
                stats[f"{s}/kept_as_is"] += 1
            elif (s, i) in fixed:
                kept.append(
                    {k: row[k] for k in keys} | {"reply": fixed[(s, i)], "reply_rewritten": True}
                )
                stats[f"{s}/reply_rewritten"] += 1
            else:
                stats[f"{s}/dropped_broken_reply"] += 1
        splits[s] = kept

    norm = a["normalize"]
    seen = {normalize(r["message"], **norm) for rows in splits.values() for r in rows}
    lsh = MinHashLSH(threshold=a["near_dup_threshold"], num_perm=a["minhash_num_perm"])
    for n, r in enumerate(splits["val"]):
        lsh.insert(f"v{n}", minhash(normalize(r["message"], **norm), a["minhash_num_perm"], a["shingle_words"]))  # fmt: skip
    topup_ok = verdicts(a["topup_judged"])
    added: Counter = Counter()
    for n, r in enumerate(read_jsonl(a["topup"])):
        if not r.get("agree"):
            stats["topup/judge_disagreed"] += 1
            continue
        if topup_ok.get(("verified", n)) == "BROKEN":
            stats["topup/broken_reply"] += 1
            continue
        text = normalize(r["message"], **norm)
        if text in seen:
            stats["topup/duplicate"] += 1
            continue
        if lsh.query(minhash(text, a["minhash_num_perm"], a["shingle_words"])):
            stats["topup/near_copy_of_val"] += 1
            continue
        seen.add(text)
        splits["train"].append({k: r[k] for k in keys} | {"reply_rewritten": False, "topup": True})
        added[f"{r['intent']}/{r['variety']}"] += 1
        stats["topup/added"] += 1

    for s, rows in splits.items():
        for r in rows:
            r["target"] = target(r["intent"], r["reply"])
        with open(out / f"{s}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        stats[f"{s}/examples"] = len(rows)
    cells = Counter((r["intent"], r["variety"]) for rows in splits.values() for r in rows)
    summary = {
        **dict(sorted(stats.items())),
        "topup_added_per_cell": dict(sorted(added.items())),
        "smallest_cells": [[f"{i}/{v}", n] for (i, v), n in sorted(cells.items(), key=lambda kv: kv[1])[:10]],
    }  # fmt: skip
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    (out / "stats.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
