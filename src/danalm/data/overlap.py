"""Matching used by the test-set overlap check (scripts/check_overlap.py) and by the training-data
decontamination in scripts/assemble_sft_v3.py: a spelling- and punctuation-insensitive canonical
form, and MinHash over character n-grams, which suits messages of a few words."""

import re

from datasketch import MinHash

from danalm.data.pipeline import normalize

_FOLD = str.maketrans({"ة": "ه", "ى": "ي", "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_PUNCT = re.compile(r"[^\w\s<>]")


def canonical(text: str, norm_cfg: dict[str, bool]) -> str:
    """Spelling- and punctuation-insensitive form used only for overlap matching."""
    folded = normalize(text, **norm_cfg).lower().translate(_FOLD)
    return " ".join(_PUNCT.sub(" ", folded).split())


def char_minhash(text: str, n: int, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    padded = f" {text.lower()} "
    for gram in {padded[i : i + n] for i in range(max(1, len(padded) - n + 1))}:
        mh.update(gram.encode("utf-8"))
    return mh
