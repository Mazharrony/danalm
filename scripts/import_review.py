"""Add the rows a person kept in a review sheet to the human test set (docs/TEST_SET.md).

Usage: uv run python scripts/import_review.py --config configs/data/test_candidates_en.yaml \
           review.verified_by=<initials>
Reads review.sheet (from make_review_sheet.py, filled in by the reviewer). Rows with keep = y are
appended to review.test_set with the next free ids (t0001, t0002, ...); correct_intent replaces the
proposed intent when filled in. Texts already in the test set are skipped. Run check_overlap.py
afterwards; it must report 0 overlaps.
"""

import csv
import json
from pathlib import Path

from danalm.config import config_from_cli
from danalm.data.intents import load_intents


def main() -> None:
    cfg = config_from_cli(__doc__)
    r = cfg["review"]
    if not r["verified_by"]:
        raise ValueError("set review.verified_by=<your initials>")
    intents = {i["name"] for i in load_intents(cfg["sft"]["intents_file"])}
    variety = {s["name"]: s["variety"] for s in cfg["fetch"]["sources"]}
    test_set = Path(r["test_set"])
    existing = []
    if test_set.exists():
        with open(test_set, encoding="utf-8") as fh:
            existing = [json.loads(line) for line in fh]
    texts = {x["text"] for x in existing}
    next_id = 1 + max((int(x["id"][1:]) for x in existing), default=0)
    added, skipped = [], 0
    with open(r["sheet"], encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["keep"].strip().lower() not in ("y", "yes"):
                continue
            intent = row["correct_intent"].strip() or row["intent"]
            if intent not in intents:
                raise ValueError(f"row {row['row']}: unknown intent {intent!r}")
            if row["text"] in texts:
                skipped += 1
                continue
            texts.add(row["text"])
            added.append({"id": f"t{next_id:04d}", "text": row["text"], "intent": intent,
                          "variety": variety[row["source"]], "author": row["source"],
                          "verified_by": r["verified_by"], "source": row["source"],
                          "notes": row["notes"].strip()})  # fmt: skip
            next_id += 1
    test_set.parent.mkdir(parents=True, exist_ok=True)
    with open(test_set, "a", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in added)
    print(f"added {len(added)} rows ({skipped} already present) -> {test_set}")


if __name__ == "__main__":
    main()
