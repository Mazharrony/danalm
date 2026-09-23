"""Fill kept-document, byte, word and token counts into the data ledger.

Usage: uv run python scripts/update_ledger.py --config configs/ledger/<name>.yaml
Tokens are counted with the configured tokenizer on the text exactly as the model will see it.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from danalm.config import config_from_cli
from danalm.data import ledger
from danalm.data.hub import text_stats
from danalm.data.pipeline import detect_lang, normalize

CHUNK = 2000  # documents per encode_batch call; whole-corpus batches would need many GB of RAM


def count(texts: list[str], tok: Tokenizer) -> dict[str, Any]:
    """Docs, chars, UTF-8 bytes, words and tokens of `texts`."""
    tokens = 0
    for i in range(0, len(texts), CHUNK):
        batch = tok.encode_batch(texts[i : i + CHUNK], add_special_tokens=False)
        tokens += sum(len(e.ids) for e in batch)
    return {**text_stats(texts), "tokens": tokens}


def entry(entry_id: str, c: dict[str, Any], tokenizer_name: str) -> dict[str, Any]:
    return {
        "id": entry_id,
        "docs_kept": c["docs"],
        "utf8_bytes": c["utf8_bytes"],
        "words": c["words"],
        "tokens": {tokenizer_name: c["tokens"]},
    }


def main() -> None:
    cfg = config_from_cli(__doc__)
    u = cfg["ledger_update"]
    tok = Tokenizer.from_file(u["tokenizer_file"])
    entries = []
    for corpus in u["cleaned_corpora"]:
        by_source: dict[str, list[str]] = defaultdict(list)
        for split in ("train", "val"):
            with open(Path(corpus["dir"]) / f"{split}.jsonl", encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    by_source[row["source"]].append(row["text"])
        for source, texts in by_source.items():
            c = count(texts, tok)
            entries.append(entry(f"{corpus['id_prefix']}/{source}", c, u["tokenizer_name"]))
            print(f"{source:18s} {c['docs']:>9,} docs {c['tokens']:>13,} tokens")
    for spec in u["file_sets"]:
        with open(spec["path"], encoding="utf-8") as fh:
            texts = [normalize(json.loads(line)["text"], **u["normalize"]) for line in fh]
        if "keep_langs" in spec:
            texts = [t for t in texts if detect_lang(t) in spec["keep_langs"]]
        c = count(texts, tok)
        entries.append(entry(spec["id"], c, u["tokenizer_name"]))
        print(f"{spec['id']:40s} {c['docs']:>7,} docs {c['tokens']:>10,} tokens")
    ledger.write_markdown(cfg["ledger"]["markdown"], ledger.upsert(cfg["ledger"]["jsonl"], entries))
    print(f"\nLedger updated: {cfg['ledger']['markdown']}")


if __name__ == "__main__":
    main()
