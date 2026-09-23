"""Sample text from Hugging Face parquet datasets without downloading whole files.

Parquet files are split into row groups (1,000 rows each in FineWeb and Wikipedia). We shuffle
every row group of the selected files with a seed and read them one at a time, text columns
only, until the character target is reached. Files are read at a pinned commit, so the manifest
is enough to reproduce a sample exactly.
"""

import random
from dataclasses import dataclass
from datetime import date
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


@dataclass(frozen=True)
class HubSource:
    """One entry of a corpus config's `sources:` list. No defaults: everything comes from YAML."""

    name: str  # short id used in file names and the ledger, e.g. fineweb2-arb
    repo_id: str
    revision: str  # commit sha (pinned)
    files: str  # glob inside the repo, e.g. data/arb_Arab/train/*.parquet
    text_columns: list[str]  # joined with a newline when there are several
    target_chars: int  # stop once this many characters are collected; 0 = read everything
    licence: str
    kind: str  # "real" or "synthetic"
    description: str


def sample_source(src: HubSource, seed: int, fs: Any = None) -> tuple[list[str], dict[str, Any]]:
    """Sample one source from the Hugging Face Hub. Returns (texts, manifest)."""
    fs = fs or HfFileSystem()
    root = f"datasets/{src.repo_id}@{src.revision}"
    paths = sorted(fs.glob(f"{root}/{src.files}"))
    if not paths:
        raise FileNotFoundError(f"{src.name}: no files match {root}/{src.files}")
    texts, units = sample_parquet(fs, paths, src.text_columns, src.target_chars, seed)
    manifest = {
        "name": src.name,
        "repo_id": src.repo_id,
        "revision": src.revision,
        "files": src.files,
        "licence": src.licence,
        "kind": src.kind,
        "description": src.description,
        "collected_on": date.today().isoformat(),
        "seed": seed,
        "row_groups_read": [[p.removeprefix(f"{root}/"), i] for p, i in units],
        **text_stats(texts),
    }
    return texts, manifest


def sample_parquet(
    fs: Any, paths: list[str], columns: list[str], target_chars: int, seed: int
) -> tuple[list[str], list[tuple[str, int]]]:
    """Read seeded-random row groups from parquet `paths` until `target_chars` is reached.

    Returns the texts and the (path, row group) units that were read. Each file is opened once:
    big files have multi-MB footers, and re-reading them per row group dominated the runtime.
    """
    handles = {path: fs.open(path) for path in paths}
    files = {path: pq.ParquetFile(fh) for path, fh in handles.items()}
    try:
        units = [(path, i) for path, pf in files.items() for i in range(pf.num_row_groups)]
        random.Random(seed).shuffle(units)
        texts: list[str] = []
        chars = 0
        read: list[tuple[str, int]] = []
        for path, group in units:
            rows = files[path].read_row_group(group, columns=columns).to_pylist()
            read.append((path, group))
            for row in rows:
                text = "\n".join(str(row[c]) for c in columns if row.get(c))
                if not text.strip():
                    continue
                texts.append(text)
                chars += len(text)
                if target_chars and chars >= target_chars:
                    return texts, read
        return texts, read
    finally:
        for fh in handles.values():
            fh.close()


def text_stats(texts: list[str]) -> dict[str, int]:
    """Document, character, UTF-8 byte and whitespace-word counts for the ledger."""
    return {
        "docs": len(texts),
        "chars": sum(len(t) for t in texts),
        "utf8_bytes": sum(len(t.encode("utf-8")) for t in texts),
        "words": sum(len(t.split()) for t in texts),
    }
