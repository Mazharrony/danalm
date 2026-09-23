"""Crash-safe progress for long teacher jobs.

Every answer is appended to a JSONL file and forced to disk, so a run that crashes, is killed
or loses power resumes where it stopped instead of starting over.
"""

import json
import os
from pathlib import Path
from typing import Any


class AnswerLog:
    """Append-only log of teacher answers, one row per finished request.

    `keys[i]` identifies request i of the plan (e.g. its intent, variety and seed). Opening an
    existing log keeps its rows in `done` (index -> row), drops a torn tail left by a crash, and
    raises if a row does not match the plan (the config changed; delete the file to start over).
    """

    def __init__(self, path: Path, keys: list[str]) -> None:
        self.path = path
        self.done: dict[int, dict[str, Any]] = {}
        if path.exists():
            good = []
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        break  # a torn write (or NUL bytes) from a crash: keep what came before
                    i = row["i"]
                    if i >= len(keys) or row["key"] != keys[i]:
                        raise ValueError(
                            f"{path} does not match this run's plan at request {i}; "
                            "delete it to start over"
                        )
                    self.done[i] = row
                    good.append(line if line.endswith("\n") else line + "\n")
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(good), encoding="utf-8", newline="\n")
            os.replace(tmp, path)
        self._fh = open(path, "a", encoding="utf-8", newline="\n")  # noqa: SIM115 - see close()

    def add(self, i: int, key: str, **data: Any) -> None:
        """Record request i and force it to disk."""
        row = {"i": i, "key": key, **data}
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.done[i] = row

    def close(self) -> None:
        self._fh.close()
