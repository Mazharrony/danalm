"""Raw Arabic / English / Arabizi text -> clean, deduplicated, language-tagged train/val JSONL
-> (optional) tokenized uint16/uint32 .bin files for training.

Adapted from the standalone `data_pipeline.py` (GulfLite data pipeline, step 1). Runs on CPU.
Inputs: .txt (one sample per line), .jsonl (text in `text_field`), .csv (column `text_field`).
CLI:    uv run python scripts/prepare_data.py --config configs/data/<name>.yaml
"""

import csv
import glob
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from datasketch import MinHash, MinHashLSH


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
    val_frac: float
    tokenizer: str | None  # HF tokenizer path/name -> also write train.bin / val.bin


# ---------------------------------------------------------------- normalization
AR_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ALEF_VARIANTS = re.compile("[إأآٱ]")  # إ أ آ ٱ
ALEF, ALEF_MAKSURA, YEH = "ا", "ى", "ي"  # ا ى ي
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# UAE numbers: +971 / 00971 / 05x...
PHONE = re.compile(
    r"(?:\+|00)?971[\s-]?\d{1,2}[\s-]?\d{3}[\s-]?\d{4}|\b0?5\d[\s-]?\d{3}[\s-]?\d{4}\b"
)
EMIRATES_ID = re.compile(r"\b784[-\s]?\d{4}[-\s]?\d{7}[-\s]?\d\b")
REPEAT = re.compile(r"(.)\1{4,}")  # "هههههههه" / "sooooo" -> capped at 3
WS = re.compile(r"\s+")
AR_CHAR = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LAT_CHAR = re.compile(r"[A-Za-z]")
ARABIZI_HINT = re.compile(r"\b\w*[a-zA-Z][2356789][a-zA-Z]\w*\b|\b[2356789][a-zA-Z]{2,}\b")


def normalize(text: str, strip_diacritics: bool, unify_alef: bool) -> str:
    """NFKC, drop tatweel (and optionally diacritics), unify alef forms, mask PII,
    cap repeated characters at 3 and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(TATWEEL, "")
    if strip_diacritics:
        text = AR_DIACRITICS.sub("", text)
    if unify_alef:
        text = ALEF_VARIANTS.sub(ALEF, text).replace(ALEF_MAKSURA, YEH)
    text = mask_pii(text)
    text = REPEAT.sub(lambda m: m.group(1) * 3, text)
    return WS.sub(" ", text).strip()


def mask_pii(text: str) -> str:
    """Replace URLs, emails, Emirates IDs and UAE phone numbers with placeholder tokens.

    Keeps PII out of the model and out of the public repo.
    """
    text = URL.sub("<URL>", text)
    text = EMAIL.sub("<EMAIL>", text)
    text = EMIRATES_ID.sub("<EID>", text)
    return PHONE.sub("<PHONE>", text)


def detect_lang(text: str) -> str:
    """Script-based tag: ar | en | mixed | arabizi | other. No model download needed."""
    ar, lat = len(AR_CHAR.findall(text)), len(LAT_CHAR.findall(text))
    letters = ar + lat
    if letters == 0:
        return "other"
    ar_ratio = ar / letters
    if ar_ratio > 0.85:
        return "ar"
    if ar_ratio < 0.15:
        return "arabizi" if len(ARABIZI_HINT.findall(text)) >= 1 else "en"
    return "mixed"


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
        with open(f, encoding="utf-8", errors="replace") as fh:
            if suffix == ".jsonl":
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj.get(text_field), str):
                        yield obj[text_field], obj.get("source", src)
            elif suffix == ".csv":
                for row in csv.DictReader(fh):
                    if row.get(text_field):
                        yield row[text_field], src
            else:
                for line in fh:
                    if line.strip():
                        yield line, src


# ---------------------------------------------------------------- dedup + tokenizing
def minhash(text: str, num_perm: int, shingle_words: int) -> MinHash:
    """MinHash signature over the lower-cased word n-gram shingles of `text`."""
    mh = MinHash(num_perm=num_perm)
    toks = text.lower().split()
    n_shingles = max(1, len(toks) - shingle_words + 1)
    for g in {" ".join(toks[i : i + shingle_words]) for i in range(n_shingles)}:
        mh.update(g.encode())
    return mh


def write_bin(samples: list[dict], tokenizer_path: str, out_path: Path) -> tuple[int, str]:
    """Tokenize samples (EOS after each) into one flat token-id file; return (n_tokens, dtype)."""
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    dtype = np.uint16 if len(tok) < 65535 else np.uint32
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    ids: list[int] = []
    for s in samples:
        ids.extend(tok.encode(s["text"], add_special_tokens=False))
        ids.append(eos)
    arr = np.array(ids, dtype=dtype)
    arr.tofile(out_path)
    return len(arr), dtype.__name__


# ---------------------------------------------------------------- main
def run_pipeline(cfg: PipelineConfig, seed: int) -> Counter:
    """Run every step and write train/val.jsonl (+ .bin) and stats.json to `cfg.out_dir`."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    seen: set[str] = set()
    kept: list[dict] = []

    lsh = None
    if cfg.near_dup_threshold > 0:
        lsh = MinHashLSH(threshold=cfg.near_dup_threshold, num_perm=cfg.minhash_num_perm)

    for raw, src in read_inputs(cfg.inputs, cfg.text_field):
        stats["read"] += 1
        text = normalize(raw, cfg.strip_diacritics, cfg.unify_alef)
        ok, reason = quality_ok(text, cfg)
        if not ok:
            stats[f"drop_{reason}"] += 1
            continue
        h = hashlib.md5(text.lower().encode(), usedforsecurity=False).hexdigest()
        if h in seen:
            stats["drop_exact_dup"] += 1
            continue
        seen.add(h)
        if lsh is not None:
            mh = minhash(text, cfg.minhash_num_perm, cfg.shingle_words)
            if lsh.query(mh):
                stats["drop_near_dup"] += 1
                continue
            lsh.insert(h, mh)
        lang = detect_lang(text)
        stats[f"lang_{lang}"] += 1
        kept.append({"text": text, "lang": lang, "source": src})

    random.Random(seed).shuffle(kept)
    n_val = max(1, int(len(kept) * cfg.val_frac)) if kept else 0
    splits = {"val": kept[:n_val], "train": kept[n_val:]}
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"{name}_samples"] = len(rows)
        stats[f"{name}_chars"] = sum(len(r["text"]) for r in rows)

    if cfg.tokenizer:
        for name, rows in splits.items():
            n_tok, dt = write_bin(rows, cfg.tokenizer, out / f"{name}.bin")
            stats[f"{name}_tokens"] = n_tok
            stats["token_dtype"] = dt

    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), "utf-8")
    return stats
