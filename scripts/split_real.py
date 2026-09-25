"""Split the real CC-BY messages into the real dev set and SFT training candidates (D-033).

Usage: uv run python scripts/split_real.py --config configs/sft/real.yaml
Reads <fetch.out_dir>/messages.jsonl (written by scripts/fetch_labelled.py) and:
1. drops every message that equals or nearly copies a human test message (the matching of
   scripts/check_overlap.py);
2. holds out split.dev_share of each intent (seeded) as the real dev set, labels unfiltered;
3. drops training candidates that nearly copy a dev message, then keeps at most
   split.max_per_intent per intent, spread over the source labels.
Writes split.dev, split.train and <fetch.out_dir>/split_stats.json.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasketch import MinHashLSH

from danalm.config import config_from_cli
from danalm.data.labelled import sample_by_intent
from danalm.data.overlap import canonical, char_minhash


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


class NearCopies:
    """Exact and near-copy lookup against a fixed set of texts, as in check_overlap.py."""

    def __init__(self, texts: list[str], o: dict[str, Any]) -> None:
        self.o = o
        self.exact: set[str] = set()
        self.lsh = MinHashLSH(threshold=o["near_dup_threshold"], num_perm=o["num_perm"])
        for n, text in enumerate(texts):
            c = canonical(text, o["normalize"])
            self.exact.add(c)
            self.lsh.insert(str(n), char_minhash(c, o["char_ngram"], o["num_perm"]))

    def __contains__(self, text: str) -> bool:
        c = canonical(text, self.o["normalize"])
        return c in self.exact or bool(self.lsh.query(char_minhash(c, self.o["char_ngram"], self.o["num_perm"])))  # fmt: skip


def main() -> None:
    cfg = config_from_cli(__doc__)
    s = cfg["split"]
    rows = read_jsonl(Path(cfg["fetch"]["out_dir"]) / "messages.jsonl")
    tests = NearCopies([t["text"] for t in read_jsonl(s["test_set"])], s["overlap"])
    clean = [r for r in rows if r["message"] not in tests]

    rng = random.Random(cfg["seed"])
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in clean:
        by_intent[r["intent"]].append(r)
    dev, rest = [], []
    for intent in sorted(by_intent):
        group = by_intent[intent][:]
        rng.shuffle(group)
        k = round(s["dev_share"] * len(group))
        dev += group[:k]
        rest += group[k:]
    devs = NearCopies([r["message"] for r in dev], s["overlap"])
    candidates = [r for r in rest if r["message"] not in devs]
    train = sample_by_intent(candidates, s["max_per_intent"], cfg["seed"])

    write_jsonl(s["dev"], dev)
    write_jsonl(s["train"], train)
    stats = {
        "messages": len(rows),
        "dropped_near_copy_of_test": len(rows) - len(clean),
        "dev": len(dev),
        "dropped_near_copy_of_dev": len(rest) - len(candidates),
        "train_candidates": len(train),
        "dev_per_intent": dict(sorted(Counter(r["intent"] for r in dev).items())),
        "train_per_intent": dict(sorted(Counter(r["intent"] for r in train).items())),
    }
    text = json.dumps(stats, indent=2)
    (Path(cfg["fetch"]["out_dir"]) / "split_stats.json").write_text(text + "\n", encoding="utf-8", newline="\n")  # fmt: skip
    print(text)


if __name__ == "__main__":
    main()
