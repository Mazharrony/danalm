"""The teacher parts of the Phase 6 evaluation (D-032): Qwen's zero-shot intents on the human test
set, and Qwen's judgement of every valid DanaLM reply.

Usage: uv run python scripts/evaluate_teacher.py --config configs/eval/phase6.yaml
Run scripts/evaluate_test.py first (it writes danalm_predictions.jsonl). Every system sees the same
normalized message. The zero-shot baseline uses the Phase 2 label-judge prompt and settings. The
reply judge answers four yes/no questions per reply; a reply is "good" when all four are yes.
Writes teacher_predictions.jsonl and reply_judge.jsonl to phase6.out_dir. Answers are saved as they
arrive (teacher_answers.jsonl), so a re-run resumes.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli
from danalm.data.intents import load_intents
from danalm.data.pipeline import normalize
from danalm.teacher.checkpoint import AnswerLog
from danalm.teacher.client import (
    TEACHER_ERRORS,
    chat_with_retries,
    parse_json_objects,
    parse_labels,
)
from danalm.teacher.prompts import intent_judge_prompt
from danalm.teacher.server import TeacherServer

QUESTIONS = ("answers", "polite_clear", "language", "safe")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    cfg = config_from_cli(__doc__)
    p, t, d = cfg["phase6"], cfg["teacher"], cfg["sft_data"]
    out = Path(p["out_dir"])
    preds = read_jsonl(out / "danalm_predictions.jsonl")  # one per test message, in test order
    intents = load_intents(d["intents_file"])
    names = {i["name"] for i in intents}
    text = [normalize(x["message"], **d["normalize"]) for x in preds]

    msg_batches = [list(range(i, min(i + p["teacher_batch"], len(preds))))
                   for i in range(0, len(preds), p["teacher_batch"])]  # fmt: skip
    valid = [i for i, x in enumerate(preds) if x["valid"]]
    reply_batches = [
        valid[i : i + p["judge_batch"]] for i in range(0, len(valid), p["judge_batch"])
    ]
    requests = [(p["intent_system"], intent_judge_prompt(intents, [text[i] for i in b]))
                for b in msg_batches]  # fmt: skip
    for b in reply_batches:
        items = "\n".join(f"{k}. Customer: {text[i]}\n   Reply: {preds[i]['reply']}"
                          for k, i in enumerate(b, start=1))  # fmt: skip
        requests.append((p["judge_system"], p["judge_prompt"].format(n=len(b), items=items)))
    keys = [hashlib.md5((s + "\n" + u).encode("utf-8")).hexdigest() for s, u in requests]
    log = AnswerLog(out / "teacher_answers.jsonl", keys)
    todo = [i for i in range(len(requests)) if i not in log.done]
    print(f"{len(requests)} requests; {len(log.done)} already answered", flush=True)
    try:
        if todo:
            with TeacherServer(t, out / "llama-server.log") as server:

                def ask(i: int) -> str | None:
                    system, user = requests[i]
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                    try:
                        answer, _ = chat_with_retries(
                            server.base_url, messages, p["teacher_params"], t["request_timeout_s"],
                            t["max_retries"], t["retry_backoff_s"],
                        )  # fmt: skip
                    except TEACHER_ERRORS as err:
                        print(f"[teacher] request {i} failed: {err!r}", flush=True)
                        return None
                    return answer

                with ThreadPoolExecutor(t["parallel"]) as pool:
                    for i, answer in zip(todo, pool.map(ask, todo), strict=True):
                        if answer is not None:
                            log.add(i, keys[i], answer=answer)
    finally:
        log.close()
    if len(log.done) < len(requests):
        raise RuntimeError(f"{len(requests) - len(log.done)} requests failed; re-run to resume")

    teacher = []
    for n, b in enumerate(msg_batches):
        labels = parse_labels(log.done[n]["answer"])
        for k, i in enumerate(b, start=1):
            label = labels.get(k)
            teacher.append({"id": preds[i]["id"], "message": preds[i]["message"],
                            "intent": preds[i]["intent"],
                            "pred_intent": label if label in names else None})  # fmt: skip
    write_jsonl(out / "teacher_predictions.jsonl", teacher)

    judged = []
    for n, b in enumerate(reply_batches, start=len(msg_batches)):
        verdicts = {o.get("n"): o for o in parse_json_objects(log.done[n]["answer"])}
        for k, i in enumerate(b, start=1):
            v = verdicts.get(k, {})
            row = {q: v.get(q) if isinstance(v.get(q), bool) else None for q in QUESTIONS}
            row["good"] = all(row[q] is True for q in QUESTIONS) if all(
                row[q] is not None for q in QUESTIONS) else None  # fmt: skip
            judged.append({"id": preds[i]["id"], "message": preds[i]["message"],
                           "reply": preds[i]["reply"], **row, "note": str(v.get("note") or "")})  # fmt: skip
    write_jsonl(out / "reply_judge.jsonl", judged)
    print(f"teacher intents for {len(teacher)} messages, judged {len(judged)} replies")


if __name__ == "__main__":
    main()
