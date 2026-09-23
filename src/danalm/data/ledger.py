"""The data ledger: every collection of data with its source, licence and counts.

`docs/data_ledger.jsonl` is the machine-readable record (one entry per collection, upserted by
id); section 1 of `docs/DATA_LEDGER.md` is rendered from it between two marker comments.
"""

import json
from pathlib import Path
from typing import Any

START, END = "<!-- ledger:start -->", "<!-- ledger:end -->"
COLUMNS = [
    ("id", "ID"),
    ("source", "Source"),
    ("revision", "Revision"),
    ("licence", "Licence"),
    ("kind", "Kind"),
    ("collected_on", "Collected on"),
    ("used_for", "Used for"),
    ("docs", "Docs (raw → kept)"),
    ("utf8_bytes", "UTF-8 bytes (kept)"),
    ("words", "Words (kept)"),
    ("tokens", "Tokens (kept)"),
    ("notes", "Notes"),
]


def upsert(jsonl_path: str | Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add entries, or merge their fields into existing ones with the same "id" (so kept and
    token counts can be filled in later). Returns the full ledger, oldest first."""
    path = Path(jsonl_path)
    ledger = read(path)
    by_id = {e["id"]: e for e in ledger}
    for entry in entries:
        if entry["id"] not in by_id:
            ledger.append(entry)
        by_id[entry["id"]] = {**by_id.get(entry["id"], {}), **entry}
    ledger = [by_id[e["id"]] for e in ledger]
    text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ledger)
    path.write_text(text, encoding="utf-8", newline="\n")
    return ledger


def read(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """All ledger entries, or [] if the ledger does not exist yet."""
    path = Path(jsonl_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_table(ledger: list[dict[str, Any]]) -> str:
    """Markdown table of the ledger plus a total-tokens line per tokenizer."""
    lines = [
        "| " + " | ".join(title for _, title in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    totals: dict[str, int] = {}
    for e in ledger:
        lines.append("| " + " | ".join(_cell(e, key) for key, _ in COLUMNS) + " |")
        for tokenizer, n in (e.get("tokens") or {}).items():
            totals[tokenizer] = totals.get(tokenizer, 0) + n
    total = "; ".join(f"{n:,} tokens ({tok})" for tok, n in totals.items()) or "0 tokens"
    return "\n".join(lines) + f"\n\n**Total kept so far:** {total}.\n"


def write_markdown(md_path: str | Path, ledger: list[dict[str, Any]]) -> None:
    """Replace the text between the ledger markers in DATA_LEDGER.md with a fresh table."""
    path = Path(md_path)
    doc = path.read_text(encoding="utf-8")
    head, _, rest = doc.partition(START)
    _, _, tail = rest.partition(END)
    if not rest:
        raise ValueError(f"{path} has no {START} marker")
    body = f"{START}\n{render_table(ledger)}{END}"
    path.write_text(head + body + tail, encoding="utf-8", newline="\n")


def _cell(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if key == "docs":
        kept = entry.get("docs_kept")
        return f"{entry.get('docs_raw', 0):,} → {'—' if kept is None else f'{kept:,}'}"
    if key == "tokens":
        return "<br>".join(f"{n:,} ({tok})" for tok, n in (value or {}).items()) or "—"
    if isinstance(value, int):
        return f"{value:,}"
    if key == "revision" and value:
        return f"`{str(value)[:8]}`"
    return str(value).replace("|", "\\|") if value not in (None, "") else "—"
