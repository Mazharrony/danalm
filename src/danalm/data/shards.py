"""Token shards for pretraining: flat uint16 arrays of token ids, with EOS after every document.

A split is stored as <out_dir>/<split>_0000.bin, <split>_0001.bin, ... of `shard_tokens` tokens
each (the last shard is shorter). Training memory-maps the shards and samples windows from them.
Memory use is one shard buffer plus one encode batch, whatever the corpus size.
"""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

DTYPE = np.uint16  # vocabularies up to 65,535 tokens (danalm-v1 has 16,384)


class ShardWriter:
    """Collects token ids and writes them as fixed-size uint16 shard files."""

    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int) -> None:
        self.out_dir, self.prefix, self.shard_tokens = Path(out_dir), prefix, shard_tokens
        self.buf = np.empty(shard_tokens, dtype=DTYPE)
        self.fill = 0
        self.files: list[dict[str, Any]] = []

    def add(self, ids: np.ndarray) -> None:
        """Append token ids, writing out every shard that fills up."""
        pos = 0
        while pos < len(ids):
            take = min(len(ids) - pos, self.shard_tokens - self.fill)
            self.buf[self.fill : self.fill + take] = ids[pos : pos + take]
            self.fill += take
            pos += take
            if self.fill == self.shard_tokens:
                self._flush()

    def close(self) -> list[dict[str, Any]]:
        """Write the last, partial shard and return [{file, tokens}, ...]."""
        if self.fill:
            self._flush()
        return self.files

    def _flush(self) -> None:
        name = f"{self.prefix}_{len(self.files):04d}.bin"
        self.buf[: self.fill].tofile(self.out_dir / name)
        self.files.append({"file": name, "tokens": int(self.fill)})
        self.fill = 0


def iter_batches(jsonl_path: Path, batch_docs: int) -> Iterator[list[dict[str, Any]]]:
    """Rows of a JSONL file in lists of `batch_docs`."""
    batch: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            batch.append(json.loads(line))
            if len(batch) == batch_docs:
                yield batch
                batch = []
    if batch:
        yield batch


def tokenize_split(
    jsonl_path: Path, tok: Tokenizer, eos_id: int, writer: ShardWriter, batch_docs: int
) -> dict[str, Counter]:
    """Encode every row's "text" (EOS appended) into `writer`.

    Returns per-source counts: docs, chars, utf8_bytes, words, and tokens (text only, without the
    EOS, matching the ledger's convention).
    """
    if tok.get_vocab_size() > np.iinfo(DTYPE).max:
        raise ValueError(f"vocabulary of {tok.get_vocab_size()} does not fit {DTYPE.__name__}")
    counts: dict[str, Counter] = {}
    for batch in iter_batches(jsonl_path, batch_docs):
        encodings = tok.encode_batch([row["text"] for row in batch], add_special_tokens=False)
        flat: list[int] = []
        for row, enc in zip(batch, encodings, strict=True):
            flat.extend(enc.ids)
            flat.append(eos_id)
            c = counts.setdefault(row["source"], Counter())
            c["docs"] += 1
            c["chars"] += len(row["text"])
            c["utf8_bytes"] += len(row["text"].encode("utf-8"))
            c["words"] += len(row["text"].split())
            c["tokens"] += len(enc.ids)
        writer.add(np.asarray(flat, dtype=DTYPE))
    return counts


def read_shard(path: str | Path) -> np.memmap:
    """Memory-map one shard (read-only)."""
    return np.memmap(path, dtype=DTYPE, mode="r")
