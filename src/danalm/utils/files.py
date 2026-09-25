"""File helpers without heavy imports (usable by the torch-free inference package)."""

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, read in 1 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
