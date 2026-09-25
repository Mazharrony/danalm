"""Text normalization, PII masking and script-based language tags.

Shared by the data pipeline (danalm.data.pipeline re-exports these names), the SFT data and the
torch-free inference package (danalm.infer), so this module imports nothing heavy.
"""

import re
import unicodedata

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
