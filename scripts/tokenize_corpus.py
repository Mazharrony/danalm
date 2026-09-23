"""Tokenize a cleaned corpus (train/val.jsonl from prepare_data.py) into uint16 token shards,
write index.json, and record kept-document and token counts per source in the data ledger.

Usage: uv run python scripts/tokenize_corpus.py --config configs/data/<name>.yaml
"""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from danalm.config import config_from_cli
from danalm.data import ledger
from danalm.data.shards import ShardWriter, tokenize_split
from danalm.utils.run import write_provenance


def main() -> None:
    cfg = config_from_cli(__doc__)
    t = cfg["tokenize"]
    src_dir, out = Path(cfg["data"]["out_dir"]), Path(t["out_dir"])
    write_provenance(out, cfg)
    tok = Tokenizer.from_file(t["tokenizer_file"])
    eos_id = tok.token_to_id(t["eos_token"])
    if eos_id is None:
        raise ValueError(f"{t['eos_token']} is not in the tokenizer")

    start = time.time()
    index: dict = {
        "tokenizer": t["tokenizer_name"],
        "tokenizer_sha256": hashlib.sha256(Path(t["tokenizer_file"]).read_bytes()).hexdigest(),
        "dtype": "uint16",
        "eos_id": eos_id,
        "splits": {},
    }
    per_source: dict[str, Counter] = {}
    for split in ("train", "val"):
        writer = ShardWriter(out, split, t["shard_tokens"])
        counts = tokenize_split(src_dir / f"{split}.jsonl", tok, eos_id, writer, t["batch_docs"])
        files = writer.close()
        index["splits"][split] = {
            "files": files,
            "tokens_with_eos": sum(f["tokens"] for f in files),
            "per_source": {s: dict(c) for s, c in counts.items()},
        }
        for source, c in counts.items():
            per_source.setdefault(source, Counter()).update(c)
        print(
            f"{split}: {index['splits'][split]['tokens_with_eos']:,} tokens in {len(files)} shards"
        )
    index["seconds"] = round(time.time() - start, 1)
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    entries = [
        {
            "id": f"{cfg['run_name']}/{source}",
            "docs_kept": c["docs"],
            "utf8_bytes": c["utf8_bytes"],
            "words": c["words"],
            "tokens": {t["tokenizer_name"]: c["tokens"]},
        }
        for source, c in per_source.items()
    ]
    ledger.write_markdown(cfg["ledger"]["markdown"], ledger.upsert(cfg["ledger"]["jsonl"], entries))
    for e in entries:
        print(
            f"{e['id']:36s} {e['docs_kept']:>9,} docs {e['tokens'][t['tokenizer_name']]:>14,} tokens"
        )
    print(f"\nShards and index.json in {out}; ledger updated ({index['seconds']}s)")


if __name__ == "__main__":
    main()
