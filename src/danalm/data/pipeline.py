"""Raw Arabic / English / Arabizi text -> clean, deduplicated, language-tagged train/val JSONL.
Token shards for training are written by a separate step (danalm.data.shards).

Adapted from the standalone `data_pipeline.py` (GulfLite data pipeline, step 1). Runs on CPU.
Inputs: .txt (one sample per line), .jsonl (text in `text_field`), .csv (column `text_field`).
CLI:    uv run python scripts/prepare_data.py --config configs/data/<name>.yaml
"""

import csv
import glob
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from datasketch import MinHash, MinHashLSH

# normalization, PII masking and language tags live in danalm.text; the data scripts import them
# from here too, so PLACEHOLDER and mask_pii are re-exported
from danalm.text import (  # noqa: F401
    AR_CHAR,
    LAT_CHAR,
    PLACEHOLDER,
    detect_lang,
    mask_pii,
    normalize,
)


@dataclass(frozen=True)
class PipelineConfig:
    """The `data:` section of a data config. No defaults: every value comes from YAML."""

    inputs: list[str]  # glob patterns
    out_dir: str
    text_field: str  # JSONL key / CSV column holding the text
    strip_diacritics: bool
    unify_alef: bool  # إ أ آ ٱ -> ا and ى -> ي
    min_chars: int
    max_chars: int
    min_letter_ratio: float  # letters / all characters
    repetitive_min_words: int  # repetition is only checked for texts with this many words
    min_unique_word_ratio: float
    near_dup_threshold: float  # MinHash Jaccard threshold; 0 disables near-duplicate removal
    minhash_num_perm: int
    shingle_words: int  # word n-gram size of the MinHash shingles
    val_frac: float  # share of documents sent to val.jsonl (seeded hash split)


def quality_ok(text: str, cfg: PipelineConfig) -> tuple[bool, str]:
    """Reject texts that are too short/long, mostly non-letters, or highly repetitive."""
    n = len(text)
    if n < cfg.min_chars:
        return False, "too_short"
    if n > cfg.max_chars:
        return False, "too_long"
    letters = len(AR_CHAR.findall(text)) + len(LAT_CHAR.findall(text))
    if letters / n < cfg.min_letter_ratio:
        return False, "low_letter_ratio"
    words = text.split()
    if (
        len(words) >= cfg.repetitive_min_words
        and len(set(words)) / len(words) < cfg.min_unique_word_ratio
    ):
        return False, "repetitive"
    return True, "ok"


# ---------------------------------------------------------------- reading
def read_inputs(patterns: list[str], text_field: str) -> Iterator[tuple[str, str]]:
    """Yield (text, source) for every sample in the files matching `patterns` (sorted order)."""
    files = sorted({f for p in patterns for f in glob.glob(p, recursive=True)})
    if not files:
        raise FileNotFoundError(f"No input files matched: {patterns}")
    for f in files:
        src = Path(f).stem
        suffix = Path(f).suffix.lower()
        # utf-8-sig drops a leading BOM, which would otherwise break the first JSON line
        with open(f, encoding="utf-8-sig", errors="replace") as fh:
            if suffix == ".jsonl":
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and isinstance(obj.get(text_field), str):
                        yield obj[text_field], obj.get("source", src)
            elif suffix == ".csv":
                for row in csv.DictReader(fh):
                    if row.get(text_field):
                        yield row[text_field], src
            else:
                for line in fh:
                    if line.strip():
                        yield line, src


# ---------------------------------------------------------------- dedup + split
def minhash(text: str, num_perm: int, shingle_words: int) -> MinHash:
    """MinHash signature over the lower-cased word n-gram shingles of `text`."""
    mh = MinHash(num_perm=num_perm)
    toks = text.lower().split()
    n_shingles = max(1, len(toks) - shingle_words + 1)
    for g in {" ".join(toks[i : i + shingle_words]) for i in range(n_shingles)}:
        mh.update(g.encode())
    return mh


def in_val(text: str, seed: int, val_frac: float) -> bool:
    """Seeded hash split: the same text always lands in the same split, with no shuffling."""
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < val_frac


# ---------------------------------------------------------------- main
def run_pipeline(cfg: PipelineConfig, seed: int) -> Counter:
    """Clean, deduplicate and tag every input document, streaming them into train/val.jsonl
    (input order kept) and writing stats.json to `cfg.out_dir`.

    Memory stays flat whatever the corpus size: only 16-byte hashes are kept for exact
    deduplication (plus MinHash signatures when near-duplicate removal is on, which is meant for
    small sets). Tokenization is a separate step (danalm.data.shards).
    """
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    seen: set[bytes] = set()

    lsh = None
    if cfg.near_dup_threshold > 0:
        lsh = MinHashLSH(threshold=cfg.near_dup_threshold, num_perm=cfg.minhash_num_perm)

    with (
        open(out / "train.jsonl", "w", encoding="utf-8", newline="\n") as train,
        open(out / "val.jsonl", "w", encoding="utf-8", newline="\n") as val,
    ):
        for raw, src in read_inputs(cfg.inputs, cfg.text_field):
            stats["read"] += 1
            text = normalize(raw, cfg.strip_diacritics, cfg.unify_alef)
            ok, reason = quality_ok(text, cfg)
            if not ok:
                stats[f"drop_{reason}"] += 1
                continue
            digest = hashlib.md5(text.lower().encode(), usedforsecurity=False).digest()
            if digest in seen:
                stats["drop_exact_dup"] += 1
                continue
            seen.add(digest)
            if lsh is not None:
                mh = minhash(text, cfg.minhash_num_perm, cfg.shingle_words)
                if lsh.query(mh):
                    stats["drop_near_dup"] += 1
                    continue
                lsh.insert(digest.hex(), mh)
            lang = detect_lang(text)
            stats[f"lang_{lang}"] += 1
            split = "val" if in_val(text, seed, cfg.val_frac) else "train"
            row = {"text": text, "lang": lang, "source": src}
            (val if split == "val" else train).write(json.dumps(row, ensure_ascii=False) + "\n")
            stats[f"{split}_samples"] += 1
            stats[f"{split}_chars"] += len(text)

    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), "utf-8")
    return stats
