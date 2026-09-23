"""Train DanaLM's byte-level BPE tokenizer.

Usage: uv run python scripts/train_tokenizer.py --config configs/tokenizer/bpe.yaml [key=value ...]
Writes <tokenizer.out_dir>/ in Hugging Face format plus config.yaml, meta.json, train_stats.json.
"""

import hashlib
import json
import time
from collections.abc import Iterator
from pathlib import Path

from danalm.config import config_from_cli
from danalm.tokenizer.bpe import TokenizerConfig, save, train
from danalm.utils.run import write_provenance
from danalm.utils.seed import set_seed


def iter_texts(path: str, counter: dict[str, int]) -> Iterator[str]:
    """Yield the "text" field of every JSONL row, counting documents and characters."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            text = json.loads(line)["text"]
            counter["docs"] += 1
            counter["chars"] += len(text)
            yield text


def main() -> None:
    cfg = config_from_cli(__doc__)
    set_seed(cfg["seed"], cfg["deterministic"])
    tcfg = TokenizerConfig(**cfg["tokenizer"])
    out = Path(tcfg.out_dir)
    write_provenance(out, cfg)

    start = time.time()
    counter = {"docs": 0, "chars": 0}
    tok = train(tcfg, iter_texts(tcfg.corpus, counter))
    save(tok, tcfg, out)
    tokenizer_json = (out / "tokenizer.json").read_bytes()
    stats = {
        "vocab_size": tok.get_vocab_size(),
        "corpus": tcfg.corpus,
        "corpus_docs": counter["docs"],
        "corpus_chars": counter["chars"],
        "seconds": round(time.time() - start, 1),
        "tokenizer_json_sha256": hashlib.sha256(tokenizer_json).hexdigest(),
    }
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
