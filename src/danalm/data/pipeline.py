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
    val_frac: float  # share of documents sent to val.jsonl (seeded hash split)


# ---------------------------------------------------------------- normalization
AR_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ALEF_VARIANTS = re.compile("[إأآٱ]")  # إ أ آ ٱ
ALEF, ALEF_MAKSURA, YEH = "ا", "ى", "ي"  # ا ى ي
# Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digits -> ASCII, so PII patterns see them.
ASCII_DIGITS = str.maketrans(
    {chr(0x0660 + i): str(i) for i in range(10)} | {chr(0x06F0 + i): str(i) for i in range(10)}
)
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Number patterns use digit boundaries (?<!\d)/(?!\d) rather than \b, which fails when a number
# is glued to Arabic letters ("رقمي0501234567").
EMIRATES_ID = re.compile(r"(?<!\d)784[-\s]?\d{4}[-\s]?\d{7}[-\s]?\d(?!\d)")
IBAN = re.compile(r"(?<![A-Za-z])AE\d{2}(?:[ -]?\d){19}(?!\d)", re.IGNORECASE)  # AE + 21 digits
CARD = re.compile(
    r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)"  # 4-4-4-4
    r"|(?<!\d)\d{4}[ -]?\d{6}[ -]?\d{5}(?!\d)"  # Amex 4-6-5
    r"|(?<!\d)(?!(?:00)?971)\d{13,19}(?!\d)"  # any other 13-19 digit run (not a +971 phone)
)
PHONE = re.compile(  # UAE numbers
    r"(?<!\d)(?:\+|00)?971[\s-]?\d{1,2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)"  # +971 / 00971 / 971
    r"|(?<!\d)0?5\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)"  # mobile 05X XXX XXXX
    r"|(?<!\d)0[2-4679][\s-]?\d{3}[\s-]?\d{4}(?!\d)"  # landline 0X XXX XXXX
)
LONG_NUMBER = re.compile(r"(?<!\d)\d{9,}(?!\d)")  # account / reference numbers left over
PLACEHOLDER = re.compile(r"<(?:URL|EMAIL|EID|IBAN|CARD|PHONE|NUM)>")
REPEAT = re.compile(r"([^\d\s])\1{4,}")  # "هههههههه" / "sooooo" -> capped at 3; digits untouched
WS = re.compile(r"\s+")
AR_CHAR = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LAT_CHAR = re.compile(r"[A-Za-z]")
ARABIZI_HINT = re.compile(r"\b\w*[a-zA-Z][2356789][a-zA-Z]\w*\b|\b[2356789][a-zA-Z]{2,}\b")
LATIN_WORD = re.compile(r"[A-Za-z0-9']+")
# Common Gulf Arabizi words without digits. Each is nearly absent from English web text: at most
# 1.5 per million words (shu) in 23.5M words of FineWeb-Edu + English Wikipedia (Phase 2, D-011).
_ARABIZI_WORD_LIST = """
abi abgha abga laish leish shlon shlonak shlonich shu wain wayed waayed zain zein wala walla wallah
yalla inshallah mashallah habibi khalas shukran mafi hada hatha haka liya jdid jadid waqt kthir
jiddan fawran ghalat ykoun ykoon mashi alhin alheen
"""
ARABIZI_WORDS = frozenset(_ARABIZI_WORD_LIST.split())
ARABIZI_MIN_SHARE = 0.1  # share of Latin words that must be Arabizi evidence
# English number+suffix tokens that look like Arabizi digit-letters: 2nd, 5pm, 2FA, 3DS, 10GB ...
EN_NUMERIC = re.compile(
    r"\b\d+(?:st|nd|rd|th|am|pm|fa|ds|gb|mb|kg|km|min|mins|hr|hrs|aed|dhs|usd)\b", re.IGNORECASE
)


def normalize(text: str, strip_diacritics: bool, unify_alef: bool) -> str:
    """NFKC, drop tatweel (and optionally diacritics), unify alef forms, mask PII (which also
    turns Arabic-Indic digits into ASCII), cap repeated characters at 3, collapse whitespace."""
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
    """Replace URLs, emails, Emirates IDs, IBANs, card numbers, UAE phone numbers and other long
    digit runs with placeholder tokens. Keeps PII out of the model and out of the public repo.

    Arabic-Indic digits are converted to ASCII first so they cannot slip past the patterns.
    Order matters: the most specific number patterns run first.
    """
    text = text.translate(ASCII_DIGITS)
    text = URL.sub("<URL>", text)
    text = EMAIL.sub("<EMAIL>", text)
    text = EMIRATES_ID.sub("<EID>", text)
    text = IBAN.sub("<IBAN>", text)
    text = CARD.sub("<CARD>", text)
    text = PHONE.sub("<PHONE>", text)
    return LONG_NUMBER.sub("<NUM>", text)


def detect_lang(text: str) -> str:
    """Script-based tag: ar | en | mixed | arabizi | other. No model download needed.

    Latin-script text is Arabizi when at least ARABIZI_MIN_SHARE of its words are evidence: a
    digit used as a letter (3andi, al7een) or a common Gulf Arabizi word (laish, shlon). So one
    model number in a long English text ("7up", "V2O5") no longer makes it Arabizi. PII
    placeholders are ignored, and English number+suffix tokens (2nd, 5pm, 2FA) are not evidence.
    """
    text = PLACEHOLDER.sub(" ", text)
    ar, lat = len(AR_CHAR.findall(text)), len(LAT_CHAR.findall(text))
    letters = ar + lat
    if letters == 0:
        return "other"
    ar_ratio = ar / letters
    if ar_ratio > 0.85:
        return "ar"
    if ar_ratio < 0.15:
        words = LATIN_WORD.findall(EN_NUMERIC.sub(" ", text))
        evidence = sum(1 for w in words if ARABIZI_HINT.fullmatch(w) or w.lower() in ARABIZI_WORDS)
        return "arabizi" if words and evidence / len(words) >= ARABIZI_MIN_SHARE else "en"
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
