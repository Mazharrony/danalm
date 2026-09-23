#!/usr/bin/env python3
"""
GulfLite data pipeline (Step 1)
-------------------------------
Raw Arabic / English / Arabizi text  ->  clean, deduplicated, language-tagged
train/val JSONL  ->  (optional) tokenized uint16/uint32 .bin shards for training.

Designed for a single machine (12 GB GPU, 64 GB RAM). Everything runs on CPU.

Input formats: .txt (one sample per line), .jsonl (field set by --text-field), .csv
Usage:
    python data_pipeline.py --inputs "raw/**/*.txt" "raw/**/*.jsonl" --out data/
    python data_pipeline.py --inputs "raw/*.jsonl" --out data/ --tokenizer path/to/tokenizer
Optional: pip install datasketch   (near-duplicate removal with MinHash)
          pip install transformers (tokenizing into .bin shards)
"""
import argparse, csv, glob, hashlib, json, random, re, sys, unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- normalization
AR_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# UAE numbers: +971 / 00971 / 05x..., plus generic long digit runs
PHONE = re.compile(r"(?:\+|00)?971[\s-]?\d{1,2}[\s-]?\d{3}[\s-]?\d{4}|\b0?5\d[\s-]?\d{3}[\s-]?\d{4}\b")
EMIRATES_ID = re.compile(r"\b784[-\s]?\d{4}[-\s]?\d{7}[-\s]?\d\b")
REPEAT = re.compile(r"(.)\1{4,}")  # "هههههههه" / "sooooo" -> capped at 3
WS = re.compile(r"\s+")
AR_CHAR = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LAT_CHAR = re.compile(r"[A-Za-z]")
ARABIZI_HINT = re.compile(r"\b\w*[a-zA-Z][2356789][a-zA-Z]\w*\b|\b[2356789][a-zA-Z]{2,}\b")


def normalize(text: str, strip_diacritics: bool, unify_alef: bool) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(TATWEEL, "")
    if strip_diacritics:
        text = AR_DIACRITICS.sub("", text)
    if unify_alef:
        text = re.sub("[إأآٱ]", "ا", text).replace("ى", "ي")
    # PII masking: keep it out of the model and out of your public repo
    text = URL.sub("<URL>", text)
    text = EMAIL.sub("<EMAIL>", text)
    text = EMIRATES_ID.sub("<EID>", text)
    text = PHONE.sub("<PHONE>", text)
    text = REPEAT.sub(lambda m: m.group(1) * 3, text)
    return WS.sub(" ", text).strip()


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


def quality_ok(text: str, min_chars: int, max_chars: int) -> tuple[bool, str]:
    n = len(text)
    if n < min_chars:
        return False, "too_short"
    if n > max_chars:
        return False, "too_long"
    letters = len(AR_CHAR.findall(text)) + len(LAT_CHAR.findall(text))
    if letters / n < 0.5:
        return False, "low_letter_ratio"
    words = text.split()
    if len(words) >= 6 and len(set(words)) / len(words) < 0.3:
        return False, "repetitive"
    return True, "ok"


# ---------------------------------------------------------------- reading
def read_inputs(patterns, text_field):
    files = sorted({f for p in patterns for f in glob.glob(p, recursive=True)})
    if not files:
        sys.exit(f"No input files matched: {patterns}")
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


# ---------------------------------------------------------------- tokenizing
def write_bin(samples, tokenizer_path, out_path):
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    dtype = np.uint16 if len(tok) < 65535 else np.uint32
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    ids = []
    for s in samples:
        ids.extend(tok.encode(s["text"], add_special_tokens=False))
        ids.append(eos)
    arr = np.array(ids, dtype=dtype)
    arr.tofile(out_path)
    return len(arr), dtype.__name__


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="glob patterns")
    ap.add_argument("--out", default="data")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--min-chars", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--keep-diacritics", action="store_true")
    ap.add_argument("--no-unify-alef", action="store_true")
    ap.add_argument("--near-dup", type=float, default=0.8, help="MinHash Jaccard threshold (needs datasketch); 0 disables")
    ap.add_argument("--tokenizer", help="HF tokenizer path/name -> also writes train.bin / val.bin")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    seen, kept = set(), []

    lsh = None
    if args.near_dup > 0:
        try:
            from datasketch import MinHash, MinHashLSH
            lsh = MinHashLSH(threshold=args.near_dup, num_perm=64)
        except ImportError:
            print("[info] datasketch not installed -> exact dedup only")

    for raw, src in read_inputs(args.inputs, args.text_field):
        stats["read"] += 1
        text = normalize(raw, not args.keep_diacritics, not args.no_unify_alef)
        ok, reason = quality_ok(text, args.min_chars, args.max_chars)
        if not ok:
            stats[f"drop_{reason}"] += 1
            continue
        h = hashlib.md5(text.lower().encode()).hexdigest()
        if h in seen:
            stats["drop_exact_dup"] += 1
            continue
        seen.add(h)
        if lsh is not None:
            mh = MinHash(num_perm=64)
            toks = text.lower().split()
            for g in {" ".join(toks[i:i + 3]) for i in range(max(1, len(toks) - 2))}:
                mh.update(g.encode())
            if lsh.query(mh):
                stats["drop_near_dup"] += 1
                continue
            lsh.insert(h, mh)
        lang = detect_lang(text)
        stats[f"lang_{lang}"] += 1
        kept.append({"text": text, "lang": lang, "source": src})

    random.Random(args.seed).shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_frac)) if kept else 0
    splits = {"val": kept[:n_val], "train": kept[n_val:]}
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"{name}_samples"] = len(rows)
        stats[f"{name}_chars"] = sum(len(r["text"]) for r in rows)

    if args.tokenizer:
        for name, rows in splits.items():
            n_tok, dt = write_bin(rows, args.tokenizer, out / f"{name}.bin")
            stats[f"{name}_tokens"] = n_tok
            stats["token_dtype"] = dt

    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
