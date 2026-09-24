"""Turn test-set candidates into a sheet a person reviews (docs/TEST_SET.md).

Usage: uv run python scripts/make_review_sheet.py --config configs/data/test_candidates_en.yaml
Reads <fetch.out_dir>/messages.jsonl and the overlap_report.json written by check_overlap.py,
drops every candidate that overlaps training data, keeps up to review.per_intent per intent
(spread over the sources), and writes review.sheet: a UTF-8 CSV that opens in Excel. The reviewer
fills `keep` (y/n) and, if the proposed intent is wrong, `correct_intent`; import_review.py then
adds the kept rows to the test set.
"""

import csv
import json
from collections import Counter
from pathlib import Path

from danalm.config import config_from_cli
from danalm.data.labelled import sample_by_intent

COLUMNS = ["row", "intent", "text", "source", "source_label", "keep", "correct_intent", "notes"]


def main() -> None:
    cfg = config_from_cli(__doc__)
    f, r = cfg["fetch"], cfg["review"]
    folder = Path(f["out_dir"])
    with open(folder / "messages.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    report = json.loads((folder / "overlap_report.json").read_text(encoding="utf-8"))
    if report["test_texts"] != len(rows):
        raise ValueError("overlap_report.json is for another candidate file; rerun check_overlap")
    flagged = {h["test"] for h in report["hits"]}
    clean = [x for x in rows if x["message"] not in flagged]
    # sample_by_intent spreads over `source_label`; spread over the datasets instead
    picked = sample_by_intent(
        [{**x, "source_label": x["source"], "label": x["source_label"]} for x in clean],
        r["per_intent"],
        cfg["seed"],
    )
    picked.sort(key=lambda x: (x["intent"], x["source"]))
    with open(r["sheet"], "w", encoding="utf-8-sig", newline="") as fh:  # BOM: Excel reads UTF-8
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for n, x in enumerate(picked, start=1):
            writer.writerow({"row": n, "intent": x["intent"], "text": x["message"],
                             "source": x["source"], "source_label": x["label"], "keep": "",
                             "correct_intent": "", "notes": ""})  # fmt: skip
    counts = Counter(x["intent"] for x in picked)
    print(
        f"{len(rows)} candidates, {len(flagged)} overlap training data, {len(picked)} in the sheet"
    )
    print(dict(sorted(counts.items())))
    print(f"-> {r['sheet']}")


if __name__ == "__main__":
    main()
