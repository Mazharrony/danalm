"""Clear non-Gulf markers in Arabic-script text: other dialects (Egyptian, Levantine, Maghrebi),
the -ش negation, and formal Standard Arabic (MSA) words in customer messages.

The teacher drifts into Egyptian-style Arabic when asked for Emirati (D-016), so generated Gulf
Arabic is filtered with these lists. They hold only unambiguous words: forms that Gulf Arabic
also uses, such as عشان, محتاج or the future prefix ب, are deliberately left out. A match means
"not natural Gulf"; no match does not prove the text is natural Emirati Arabic.
"""

import re

NON_GULF_MARKERS = {
    "egyptian": ["عايز", "عاوز", "عايزة", "ليه", "ايه", "إيه", "حصل", "مش", "مفيش", "دي", "ده",
                 "ازاي", "إزاي", "امتى", "اوي", "اوى", "بالظبط", "خالص"],
    "levantine": ["بدي", "بدك", "هلق", "هلأ", "كتير", "هيك", "منيح"],
    "maghrebi": ["واش", "شكون", "هانيك", "بزاف", "دابا", "كيفاش", "ديالي", "ديال", "بغيت"],
}  # fmt: skip
# Formal Standard Arabic that a Gulf customer would not type in a chat message
MSA_MARKERS = ["لماذا", "ماذا", "سوف", "اريد", "أريد", "ليس", "لست"]
_ARABIC_WORD = re.compile(r"[؀-ۿ]+")
_CONJUNCTIONS = ("و", "ف")  # attached "and" / "so": ومفيش -> مفيش
# ما + a word ending in ش (ما وصلتش, ما اعرفش): Egyptian/Levantine/Maghrebi, not Gulf
_SH_NEGATION = re.compile(r"(?:^|\s)ما\s+[؀-ۿ]{2,}ش(?=$|\s|[؟?!.,،])")


def _words(text: str) -> set[str]:
    words = set(_ARABIC_WORD.findall(text))
    return words | {w[1:] for w in words if w.startswith(_CONJUNCTIONS) and len(w) > 2}


def non_gulf_markers(text: str) -> list[str]:
    """The other-dialect markers found in `text`, as "word(dialect)"."""
    words = _words(text)
    found = [f"{m}({d})" for d, markers in NON_GULF_MARKERS.items() for m in markers if m in words]
    if _SH_NEGATION.search(text):
        found.append("ما...ش(negation)")
    return found


def msa_markers(text: str) -> list[str]:
    """Formal Standard Arabic words found in `text` (for customer messages, not replies)."""
    words = _words(text)
    return [f"{m}(msa)" for m in MSA_MARKERS if m in words]
