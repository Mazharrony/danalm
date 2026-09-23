"""Generate small synthetic text sets with the local teacher model and record them in the ledger.

Usage: uv run python scripts/generate_synthetic.py --config configs/synthetic/<name>.yaml
Starts the llama.cpp server (or reuses one already running), runs every task, stops the server.
Writes <generation.out_dir>/<task>.jsonl, manifest.json and provenance files.
"""

import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli
from danalm.data import ledger
from danalm.data.hub import text_stats
from danalm.data.pipeline import detect_lang, normalize
from danalm.teacher.client import chat, parse_lines
from danalm.teacher.server import TeacherServer
from danalm.utils.run import start_run, write_provenance
from danalm.utils.seed import set_seed


def _words(text: str) -> set[str]:
    """Lower-cased words without punctuation, for near-copy detection."""
    return set(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def run_task(
    task: dict[str, Any], task_index: int, cfg: dict[str, Any], base_url: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate one task's messages. Returns (rows, task manifest)."""
    t, g = cfg["teacher"], cfg["generation"]
    n_requests = math.ceil(task["target"] / g["per_request"] * g["oversample"])
    examples = " / ".join(f'"{e}"' for e in task["examples"])
    example_words = [_words(e) for e in task["examples"]]

    def request(i: int) -> tuple[str, str, dict[str, int]]:
        domain = g["domains"][i % len(g["domains"])]
        prompt = task["prompt"].format(n=g["per_request"], domain=domain, examples=examples)
        messages = [
            {"role": "system", "content": g["system"]},
            {"role": "user", "content": prompt},
        ]
        params = {**g["params"], "seed": cfg["seed"] + 1000 * task_index + i}
        answer, usage = chat(base_url, messages, params, t["request_timeout_s"])
        return domain, answer, usage

    start = time.time()
    with ThreadPoolExecutor(t["parallel"]) as pool:
        answers = list(pool.map(request, range(n_requests)))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    tags: Counter = Counter()
    dropped: Counter = Counter()
    for i, (domain, answer, _) in enumerate(answers):
        for line in parse_lines(answer):
            lang = detect_lang(normalize(line, **g["normalize"]))
            tags[lang] += 1
            words = _words(line)
            if line in seen:
                dropped["duplicate"] += 1
            elif lang not in task["keep_langs"]:
                dropped[f"lang_{lang}"] += 1
            elif max(_jaccard(words, ex) for ex in example_words) >= g["example_max_jaccard"]:
                dropped["copies_example"] += 1
            else:
                seen.add(line)
                rows.append(
                    {
                        "text": line,
                        "task": task["name"],
                        "lang": lang,
                        "domain": domain,
                        "request": i,
                    }
                )
    rows = rows[: task["target"]]
    manifest = {
        "requests": n_requests,
        "lines_generated": sum(tags.values()),
        "lang_tags_generated": dict(tags),
        "dropped": dict(dropped),
        "kept": len(rows),
        "completion_tokens": sum(u.get("completion_tokens", 0) for *_, u in answers),
        "seconds": round(time.time() - start, 1),
        "prompt": task["prompt"],
    }
    return rows, manifest


def main() -> None:
    cfg = config_from_cli(__doc__)
    set_seed(cfg["seed"], cfg["deterministic"])
    run = start_run(cfg, job_type="synthetic")
    g = cfg["generation"]
    out = Path(g["out_dir"])
    write_provenance(out, cfg)
    manifest: dict[str, Any] = {"teacher": cfg["teacher"]["model_name"], "params": g["params"]}
    entries = []
    with TeacherServer(cfg["teacher"], run.dir / "llama-server.log") as server:
        for i, task in enumerate(g["tasks"]):
            rows, task_manifest = run_task(task, i, cfg, server.base_url)
            with open(out / f"{task['name']}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest[task["name"]] = task_manifest
            stats = text_stats([r["text"] for r in rows])
            entries.append(
                {
                    "id": f"{cfg['run_name']}/{task['name']}",
                    "source": f"teacher: {cfg['teacher']['model_name']} via llama.cpp",
                    "revision": cfg["teacher"]["revision"],
                    "licence": "Apache-2.0 (teacher outputs)",
                    "kind": "synthetic",
                    "collected_on": date.today().isoformat(),
                    "used_for": g["used_for"],
                    "docs_raw": task_manifest["lines_generated"],
                    "docs_kept": len(rows),
                    "utf8_bytes": stats["utf8_bytes"],
                    "words": stats["words"],
                    "notes": (
                        f"{task_manifest['requests']} requests, "
                        f"{task_manifest['completion_tokens']:,} teacher tokens; "
                        "not yet checked by a native speaker"
                    ),
                }
            )
            run.wandb.log(
                {
                    f"{task['name']}/kept": len(rows),
                    **{
                        f"{task['name']}/{k}": v
                        for k, v in task_manifest.items()
                        if isinstance(v, int | float)
                    },
                }
            )
            print(
                f"{task['name']:14s} kept {len(rows):>4} of {task_manifest['lines_generated']:>4} lines "
                f"({task_manifest['completion_tokens']:,} tokens, {task_manifest['seconds']}s) "
                f"tags: {task_manifest['lang_tags_generated']}"
            )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    ledger.write_markdown(cfg["ledger"]["markdown"], ledger.upsert(cfg["ledger"]["jsonl"], entries))
    run.wandb.finish()
    print(f"\nSaved to {out}; ledger updated")


if __name__ == "__main__":
    main()
