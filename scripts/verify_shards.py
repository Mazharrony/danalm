"""Check the token shards from tokenize_corpus.py against the cleaned corpus they came from.

Usage: uv run python scripts/verify_shards.py --config configs/data/<name>.yaml
Per split: the EOS count must equal the number of documents, every id must be below the
vocabulary size, and a seeded sample of documents (plus every document that crosses a shard
boundary) must decode back to its cleaned text exactly. Writes <tokenize.out_dir>/verify.json
and exits with status 1 if any check fails.
"""

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

from danalm.config import config_from_cli
from danalm.data.shards import verify_split


def main() -> None:
    cfg = config_from_cli(__doc__)
    t = cfg["tokenize"]
    src_dir, out = Path(cfg["data"]["out_dir"]), Path(t["out_dir"])
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(t["tokenizer_file"])
    results = {}
    for split, info in index["splits"].items():
        files = [out / f["file"] for f in info["files"]]
        r = verify_split(
            files, src_dir / f"{split}.jsonl", tok, index["eos_id"], t["verify_docs"], cfg["seed"]
        )
        results[split] = r
        print(
            f"{split}: {'OK' if r['ok'] else 'FAILED'} - {r['tokens']:,} tokens in {r['shards']} "
            f"shards, {r['eos']:,} EOS for {r['docs']:,} docs, max id {r['max_id']} "
            f"(vocab {r['vocab']}), {r['checked_docs']} docs decoded ({r['boundary_docs']} cross "
            f"a shard boundary), mismatches: {r['mismatched_docs'] or 'none'}"
        )
    (out / "verify.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not all(r["ok"] for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
