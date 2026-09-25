"""Fetch labelled utterances from Hugging Face parquet datasets and map their labels to ours.

Usage: uv run python scripts/fetch_labelled.py --config configs/<name>.yaml
For each entry in fetch.sources: reads Hugging Face parquet files (label names from the file's
metadata) or a CSV at a URL, always at a pinned revision; maps the labels to DanaLM intents with
`label_map` ("*" = every label not listed; labels mapped to null or starting with one of
`exclude_label_prefixes` are skipped); skips texts containing any of `drop_substrings`
(optional); normalizes the text like SFT data (PII masked); drops duplicates; and takes a seeded sample: up to `max_per_intent` per intent if set, otherwise
spread evenly over the source labels (`max_per_label`, `max_rows`). Writes
<fetch.out_dir>/messages.jsonl, manifest.json and one ledger entry per source.
"""

import json
from datetime import date
from pathlib import Path

from huggingface_hub import HfFileSystem

from danalm.config import config_from_cli
from danalm.data import ledger
from danalm.data.hub import text_stats
from danalm.data.intents import load_intents
from danalm.data.labelled import iter_labelled, map_label, sample_by_intent, spread_sample
from danalm.data.pipeline import detect_lang, normalize
from danalm.utils.run import write_provenance


def main() -> None:
    cfg = config_from_cli(__doc__)
    f = cfg["fetch"]
    out = Path(f["out_dir"])
    write_provenance(out, cfg)
    intents = {i["name"]: i for i in load_intents(cfg["sft"]["intents_file"])}
    fs = HfFileSystem()
    kept_all, manifest, entries, seen = [], [], [], set()
    for src in f["sources"]:
        raw, rows, dropped = 0, [], 0
        for text, label in iter_labelled(src, fs):
            raw += 1
            # e.g. Bitext's template slots ("{{Order Number}}"), which no customer would type
            if any(s in text for s in src.get("drop_substrings", [])):
                dropped += 1
                continue
            intent = map_label(label, src)
            message = normalize(text, **cfg["sft"]["normalize"])
            if intent is None or not message or message.lower() in seen:
                continue
            if intent not in intents:
                raise ValueError(f"{src['name']}: label_map gives unknown intent {intent!r}")
            seen.add(message.lower())
            rows.append({"message": message, "intent": intent, "variety": src["variety"],
                         "lang": detect_lang(message), "domain": intents[intent]["domain"],
                         "source": src["name"], "source_label": label})  # fmt: skip
        if src["max_per_intent"]:
            sample = sample_by_intent(rows, src["max_per_intent"], cfg["seed"])
        else:
            sample = spread_sample(rows, src["max_per_label"], src["max_rows"], cfg["seed"])
        kept_all += sample
        stats = text_stats([r["message"] for r in sample])
        manifest.append({**src, "rows_read": raw, "rows_dropped_by_substring": dropped,
                         "rows_mapped": len(rows), "rows_kept": len(sample)})  # fmt: skip
        entries.append({
            "id": f"{cfg['run_name']}/{src['name']}",
            "source": src.get("url") or f"{src['repo_id']} `{src['files']}`",
            "revision": src["revision"],
            "licence": src["licence"],
            "kind": src["kind"],
            "collected_on": date.today().isoformat(),
            "used_for": f["used_for"],
            "docs_raw": raw,
            "docs_kept": len(sample),
            "utf8_bytes": stats["utf8_bytes"],
            "words": stats["words"],
            "notes": src["description"],
        })  # fmt: skip
        print(f"{src['name']}: {raw:,} rows read, {dropped:,} dropped (drop_substrings), "
              f"{len(rows):,} mapped, {len(sample):,} kept")  # fmt: skip
    with open(out / "messages.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept_all)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ledger.write_markdown(cfg["ledger"]["markdown"], ledger.upsert(cfg["ledger"]["jsonl"], entries))
    print(f"{len(kept_all):,} messages -> {out / 'messages.jsonl'}; ledger updated")


if __name__ == "__main__":
    main()
