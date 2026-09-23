"""Checks on assistant replies in synthetic SFT data.

DanaLM runs on-device with no access to accounts or back-office systems, so a reply must never
claim an action was done ("your card has been blocked", "تم تجميد البطاقة"). The customer might
believe it. Such replies are dropped; the reply should give the next step instead.
"""

import re

_CLAIM_EN = re.compile(
    r"\b(?:has been|have been|is now|are now|was sent|were sent|is frozen|is blocked|"
    r"emailed you|we(?: have|'ve)? (?:frozen|blocked|sent|cancelled|canceled|refunded|updated|"
    r"activated|changed|reset|unblocked|credited|reversed))\b",
    re.IGNORECASE,
)
# standalone تم ("was done") and قمنا ب ("we did"); يتم ("will be done") is fine
_CLAIM_AR = re.compile(r"(?:^|\s)(?:تم|قمنا\s*ب)")


def claims_done_action(reply: str) -> bool:
    """True if the reply says an action was already carried out."""
    return bool(_CLAIM_EN.search(reply) or _CLAIM_AR.search(reply))
