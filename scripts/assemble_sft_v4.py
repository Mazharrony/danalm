"""Build the SFT data of the real-message round (D-033): data/sft/final-v4.

Usage: uv run python scripts/assemble_sft_v4.py --config configs/sft/v4.yaml
Starts from assemble.base_dir (final-v3: train and the unchanged validation split) and adds, to
the training split only, every example of assemble.sources that the blind label judge agreed with
and whose reply the reply check did not mark BROKEN. New messages that repeat an existing message,
or equal or nearly copy a human test message or a real-dev message (the matching of
scripts/check_overlap.py), are dropped. `target` is built from intent and reply.
Writes <out_dir>/train.jsonl, val.jsonl and stats.json.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasketch import MinHashLSH

from danalm.config import config_from_cli
from danalm.data.overlap import canonical, char_minhash
from danalm.data.pipeline import normalize
from danalm.utils.run import write_provenance


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def target(intent: str, reply: str) -> str:
    """The exact JSON string the model must produce (as in scripts/build_sft_dataset.py)."""
    return json.dumps({"intent": intent, "reply": reply}, ensure_ascii=False)


def main() -> None:
    cfg = config_from_cli(__doc__)
    a = cfg["assemble"]
    o = a["overlap"]
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_provenance(out, cfg)
    keys = ["message", "intent", "reply", "variety", "lang", "domain"]
    base = {s: read_jsonl(Path(a["base_dir"]) / f"{s}.jsonl") for s in ("train", "val")}

    guarded = [t["text"] for t in read_jsonl(a["test_set"])] + [r["message"] for r in read_jsonl(a["real_dev"])]  # fmt: skip
    exact = {canonical(t, o["normalize"]) for t in guarded}
    lsh = MinHashLSH(threshold=o["near_dup_threshold"], num_perm=o["num_perm"])
    for n, t in enumerate(guarded):
        lsh.insert(str(n), char_minhash(canonical(t, o["normalize"]), o["char_ngram"], o["num_perm"]))  # fmt: skip
    seen = {normalize(r["message"], **a["normalize"]) for rows in base.values() for r in rows}

    stats: Counter = Counter()
    added: Counter = Counter()
    train = list(base["train"])
    for src in a["sources"]:
        verdict = {j["index"]: j["verdict"] for j in read_jsonl(src["reply_check"]) if j["counted"]}
        for n, r in enumerate(read_jsonl(src["verified"])):
            if not r.get("agree"):
                stats[f"{src['name']}/judge_disagreed"] += 1
                continue
            if verdict.get(n) == "BROKEN":
                stats[f"{src['name']}/broken_reply"] += 1
                continue
            text = normalize(r["message"], **a["normalize"])
            c = canonical(r["message"], o["normalize"])
            if text in seen:
                stats[f"{src['name']}/duplicate"] += 1
                continue
            if c in exact or lsh.query(char_minhash(c, o["char_ngram"], o["num_perm"])):
                stats[f"{src['name']}/near_copy_of_test_or_real_dev"] += 1
                continue
            seen.add(text)
            train.append({k: r.get(k) for k in keys} | {"source": src["name"]})
            added[f"{src['name']}/{r['intent']}"] += 1
            stats[f"{src['name']}/added"] += 1

    for s, rows in (("train", train), ("val", base["val"])):
        for r in rows:
            r["target"] = target(r["intent"], r["reply"])
        with open(out / f"{s}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        stats[f"{s}/examples"] = len(rows)
    summary = {
        **dict(sorted(stats.items())),
        "added_per_source_and_intent": dict(sorted(added.items())),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    (out / "stats.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
