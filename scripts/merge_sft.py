"""Combine SFT sources into one candidate file for a single judge pass.

Usage: uv run python scripts/merge_sft.py --config configs/sft/final.yaml
Reads merge.sources in order (each with `exclude_varieties`), re-tags every message with the
current language tagger and keeps it only if the tag is in merge.keep_langs for its variety,
drops messages already seen (case-insensitive, first source wins), and writes merge.out plus
merge_stats.json next to it. verify_sft.py then judges the file and build_sft_dataset.py builds
the final set from the verified rows.
"""

import json
from collections import Counter
from pathlib import Path

from danalm.config import config_from_cli
from danalm.data.pipeline import detect_lang
from danalm.utils.run import write_provenance


def main() -> None:
    cfg = config_from_cli(__doc__)
    m = cfg["merge"]
    out = Path(m["out"])
    write_provenance(out.parent, cfg)
    seen: set[str] = set()
    rows, stats = [], Counter()
    for src in m["sources"]:
        name = Path(src["path"]).parent.name
        with open(src["path"], encoding="utf-8") as fh:
            for row in map(json.loads, fh):
                if row["variety"] in src["exclude_varieties"]:
                    continue
                lang = detect_lang(row["message"])
                key = row["message"].lower()
                if lang not in m["keep_langs"][row["variety"]]:
                    reason = f"message_lang_{lang}"
                elif key in seen:
                    reason = "duplicate"
                else:
                    seen.add(key)
                    rows.append({**row, "lang": lang, "sft_source": name})
                    reason = "kept"
                stats[f"{name}/{row['variety']}/{reason}"] += 1
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out.parent / "merge_stats.json").write_text(
        json.dumps(dict(stats), indent=2), encoding="utf-8"
    )
    for key, n in sorted(stats.items()):
        print(f"{key:55s} {n:>7,}")
    print(f"{len(rows):,} candidates -> {out}")


if __name__ == "__main__":
    main()
