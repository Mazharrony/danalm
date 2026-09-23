"""Clear non-Gulf dialect markers in Arabic-script text (Egyptian, Levantine, Maghrebi).

The teacher drifts into Egyptian-style Arabic when asked for Emirati (D-016), so generated Gulf
Arabic is filtered with these lists. They hold only unambiguous whole words: forms that Gulf
Arabic also uses, such as عشان or the future prefix ب, are deliberately left out. A match means
"not Gulf"; no match does not prove the text is natural Emirati Arabic.
"""

import re

NON_GULF_MARKERS = {
    "egyptian": ["عايز", "عاوز", "عايزة", "محتاج", "ليه", "ايه", "إيه", "حصل", "مش", "مفيش",
                 "دي", "ده", "ازاي", "إزاي", "امتى", "اوي", "اوى", "بالظبط", "خالص"],
    "levantine": ["بدي", "بدك", "هلق", "هلأ", "كتير", "هيك", "شو بدك", "منيح"],
    "maghrebi": ["واش", "شكون", "هانيك", "بزاف", "دابا", "كيفاش", "ديالي", "ديال", "بغيت"],
}  # fmt: skip
_ARABIC_WORD = re.compile(r"[؀-ۿ]+")


def non_gulf_markers(text: str) -> list[str]:
    """The non-Gulf marker words found in `text`, as "word(dialect)"."""
    words = set(_ARABIC_WORD.findall(text))
    found = []
    for dialect, markers in NON_GULF_MARKERS.items():
        for marker in markers:
            parts = marker.split()
            if (len(parts) == 1 and marker in words) or (len(parts) > 1 and marker in text):
                found.append(f"{marker}({dialect})")
    return found
