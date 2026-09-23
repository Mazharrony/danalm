"""Minimal client for the teacher's OpenAI-compatible chat API (stdlib only)."""

import json
import re
import urllib.request
from typing import Any

# "1. ", "2) ", "- ", "* ", "• ", "١. " at the start of a generated line. The marker must be
# followed by whitespace, so messages like "2-3 days late" or "3andi" are left alone.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\(?[0-9٠-٩]{1,3}[.)\-:])\s+")
_QUOTES = "\"'“”«»`"


def chat(
    base_url: str, messages: list[dict[str, str]], params: dict[str, Any], timeout: float
) -> tuple[str, dict[str, int]]:
    """Send one chat request. Returns (answer text, token usage)."""
    body = json.dumps({"messages": messages, **params}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"] or "", data.get("usage", {})


def parse_lines(answer: str) -> list[str]:
    """Split a generated list into one message per line, removing numbering, bullets and quotes."""
    lines = []
    for line in answer.splitlines():
        line = _LIST_MARKER.sub("", line).strip().strip(_QUOTES).strip()
        if line:
            lines.append(line)
    return lines
