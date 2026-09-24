"""Check that the test set does not overlap with any training data (brief, Phase 2c).

Usage: uv run python scripts/check_overlap.py --config configs/data/overlap.yaml
Texts are canonicalized first (pipeline normalization, lower case, ة→ه, ى→ي, alef forms→ا,
punctuation removed). A test text overlaps if it then equals a training text, or is a
near-duplicate of a short training text (MinHash over character n-grams, which suits messages of
a few words). Long pretraining documents are checked for exact equality only. Training files are
streamed, and texts longer than `compare_up_to` x the longest test text are skipped (they cannot
be a copy of one), so memory stays small even with the 1.5B-token corpus.
The check is deliberately strict: a flagged test message is rewritten or removed by a person.
Writes overlap_report.json next to the first test file and exits with code 1 on any overlap.
"""

import glob
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasketch import MinHashLSH

from danalm.config import config_from_cli
from danalm.data.overlap import canonical, char_minhash


def iter_texts(patterns: list[str], field: str) -> Iterator[tuple[str, str]]:
    """(file, text) for every row of every file matching the glob patterns, one line at a time.
    Rows without a text in `field` (e.g. rejected teacher output with broken fields) are skipped."""
    for path in sorted({p for pattern in patterns for p in glob.glob(pattern, recursive=True)}):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                text = json.loads(line).get(field) if line.strip() else None
                if isinstance(text, str) and text.strip():
                    yield path, text


def main() -> None:
    cfg = config_from_cli(__doc__)
    o = cfg["overlap"]
    norm = lambda t: canonical(t, o["normalize"])  # noqa: E731
    test = list(iter_texts(o["test_files"], o["test_field"]))
    if not test:
        sys.exit(f"no test texts found in {o['test_files']}")
    max_len = o["compare_up_to"] * max(len(text) for _, text in test)
    skipped = 0

    lsh = MinHashLSH(threshold=o["near_dup_threshold"], num_perm=o["num_perm"])
    short: dict[str, str] = {}
    exact: dict[str, str] = {}
    for spec in o["train_sets"]:
        for i, (path, text) in enumerate(iter_texts(spec["files"], spec["field"])):
            if len(text) > max_len:
                skipped += 1
                continue
            key = norm(text)
            exact.setdefault(key, path)
            if spec["near_dup"] and key not in short:
                short[key] = path
                lsh.insert(f"{path}:{i}", char_minhash(key, o["char_ngram"], o["num_perm"]))

    hits: list[dict[str, Any]] = []
    for path, text in test:
        key = norm(text)
        if key in exact:
            hits.append(
                {"test": text, "test_file": path, "kind": "exact", "train_file": exact[key]}
            )
            continue
        near = lsh.query(char_minhash(key, o["char_ngram"], o["num_perm"]))
        if near:
            hits.append(
                {"test": text, "test_file": path, "kind": "near", "train_matches": near[:3]}
            )
    report = {
        "test_texts": len(test),
        "train_texts": len(exact),
        "train_texts_skipped_as_too_long": skipped,
        "overlaps": len(hits),
        "hits": hits,
    }
    out = Path(o["test_files"][0]).parent / "overlap_report.json"
    if not any(ch in str(out) for ch in "*?["):
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(test)} test texts vs {len(exact):,} training texts: {len(hits)} overlap(s)")
    for h in hits[:20]:
        print(f"  [{h['kind']}] {h['test']}")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
