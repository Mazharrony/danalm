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


_LABEL_LINE = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*([a-z_]+)\s*$")


def parse_json_objects(answer: str) -> list[dict[str, Any]]:
    """JSON objects from an answer that should be one object per line (JSON-lines).

    Also accepts a single JSON array, code fences and list markers; unparsable lines are skipped.
    """
    text = answer.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        whole = json.loads(text)
        if isinstance(whole, list):
            return [item for item in whole if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    objects = []
    for line in text.splitlines():
        line = _LIST_MARKER.sub("", line).strip()
        start, end = line.find("{"), line.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            obj = json.loads(line[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    return objects


def parse_labels(answer: str) -> dict[int, str]:
    """{item number: label} from answer lines such as "3: order_status"."""
    labels = {}
    for line in answer.splitlines():
        m = _LABEL_LINE.match(line)
        if m:
            labels[int(m.group(1))] = m.group(2)
    return labels
