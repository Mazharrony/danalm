"""Labelled utterances from public intent datasets, mapped onto DanaLM's intents.

Used for real `other` examples (train splits) and for test-set candidates (test splits).
"""

import csv
import io
import json
import random
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def label_names(pf: pq.ParquetFile, column: str) -> list[str]:
    """ClassLabel names stored in a Hugging Face parquet file's schema metadata."""
    meta = json.loads((pf.schema_arrow.metadata or {}).get(b"huggingface", b"{}"))
    feature = meta.get("info", {}).get("features", {}).get(column, {})
    if feature.get("_type") != "ClassLabel":
        raise ValueError(f"column {column!r} has no ClassLabel names in the file metadata")
    return feature["names"]


def map_label(label: str, src: dict[str, Any]) -> str | None:
    """Our intent for a source label, or None to skip it."""
    if any(label.startswith(p) for p in src["exclude_label_prefixes"]):
        return None
    return src["label_map"].get(label, src["label_map"].get("*"))


def spread_sample(rows: list[dict], per_label: int | None, total: int | None, seed: int) -> list:
    """Seeded sample that takes labels in turn, so frequent labels do not crowd out rare ones."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["source_label"]].append(row)
    rng = random.Random(seed)
    queues = []
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        queues.append(group[:per_label] if per_label else group)
    out: list[dict] = []
    while any(queues) and (total is None or len(out) < total):
        for q in queues:
            if q and (total is None or len(out) < total):
                out.append(q.pop(0))
    return out


def sample_by_intent(rows: list[dict], per_intent: int, seed: int) -> list[dict]:
    """Up to `per_intent` rows for each of our intents, spread over the source labels."""
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_intent[row["intent"]].append(row)
    out: list[dict] = []
    for intent in sorted(by_intent):
        out += spread_sample(by_intent[intent], per_label=None, total=per_intent, seed=seed)
    return out


def iter_labelled(src: dict[str, Any], fs: Any) -> Iterator[tuple[str, str]]:
    """(text, label name) pairs of one source: Hugging Face parquet files (`repo_id`, `files`) or
    a CSV file at a URL (`url`, where {revision} is filled in), always at a pinned revision."""
    if "url" in src:
        url = src["url"].format(revision=src["revision"])
        with urllib.request.urlopen(url, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        for row in csv.DictReader(io.StringIO(text)):
            yield row[src["text_column"]], row[src["label_column"]]
        return
    paths = sorted(fs.glob(f"datasets/{src['repo_id']}@{src['revision']}/{src['files']}"))
    if not paths:
        raise FileNotFoundError(f"{src['name']}: no files match {src['files']}")
    for path in paths:
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            kind = pf.schema_arrow.field(src["label_column"]).type
            # a text column holds the label name itself (e.g. Bitext); integers need ClassLabel names
            names = None if pa.types.is_string(kind) or pa.types.is_large_string(kind) else label_names(pf, src["label_column"])  # fmt: skip
            table = pf.read(columns=[src["text_column"], src["label_column"]]).to_pydict()
        for text, label in zip(table[src["text_column"]], table[src["label_column"]], strict=True):
            yield text, label if names is None else names[label]
