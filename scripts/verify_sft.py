"""Blindly re-label generated SFT examples with a judge model and measure label agreement.

Usage: uv run python scripts/verify_sft.py --config configs/sft/<name>.yaml
For every file in verify.inputs, writes <same dir>/verified.jsonl: each row gains "judge_label"
and "agree". Rows with agree=false are dropped when the final SFT set is built. The judge sees only
the message, never the intended label.
"""

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli
from danalm.data.intents import load_intents
from danalm.teacher.client import chat, parse_labels
from danalm.teacher.server import TeacherServer
from danalm.utils.run import start_run


def build_prompt(intents: list[dict], messages: list[str]) -> str:
    labels = "\n".join(f"- {i['name']}: {i['description']}" for i in intents)
    numbered = "\n".join(f"{n}. {m}" for n, m in enumerate(messages, start=1))
    return (
        f"Intents:\n{labels}\n\nMessages:\n{numbered}\n\n"
        "For each message, answer with its number and the single best intent name, "
        'like "3: order_status".'
    )


def main() -> None:
    cfg = config_from_cli(__doc__)
    run = start_run(cfg, job_type="sft-verify")
    v, t = cfg["verify"], cfg["teacher"]
    intents = load_intents(cfg["sft"]["intents_file"])
    names = {i["name"] for i in intents}
    summary: dict[str, Any] = {"judge": t["model_name"], "files": {}}
    with TeacherServer(t, run.dir / "llama-server.log") as server:
        for path in map(Path, v["inputs"]):
            with open(path, encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh]
            batches = [rows[i : i + v["batch"]] for i in range(0, len(rows), v["batch"])]

            def judge(batch: list[dict]) -> dict[int, str]:
                messages = [
                    {"role": "system", "content": v["system"]},
                    {
                        "role": "user",
                        "content": build_prompt(intents, [r["message"] for r in batch]),
                    },
                ]
                answer, _ = chat(server.base_url, messages, v["params"], t["request_timeout_s"])
                return parse_labels(answer)

            start = time.time()
            with ThreadPoolExecutor(t["parallel"]) as pool:
                answers = list(pool.map(judge, batches))
            counts: Counter = Counter()
            for batch, labels in zip(batches, answers, strict=True):
                for n, row in enumerate(batch, start=1):
                    label = labels.get(n)
                    row["judge_label"] = label if label in names else None
                    row["agree"] = row["judge_label"] == row["intent"]
                    counts["rows"] += 1
                    counts["agree"] += row["agree"]
                    counts["no_label"] += row["judge_label"] is None
                    counts[f"{row['variety']}/rows"] += 1
                    counts[f"{row['variety']}/agree"] += row["agree"]
            out = path.with_name("verified.jsonl")
            with open(out, "w", encoding="utf-8", newline="\n") as fh:
                fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            counts["seconds"] = round(time.time() - start, 1)
            summary["files"][str(path)] = dict(counts)
            rate = counts["agree"] / max(1, counts["rows"])
            print(f"{path}: {counts['agree']}/{counts['rows']} agree ({rate:.1%}) -> {out}")
    (run.dir / "verify_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run.wandb.summary.update(summary)
    run.wandb.finish()


if __name__ == "__main__":
    main()
