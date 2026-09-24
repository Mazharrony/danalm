"""Judge the wording of SFT replies with the teacher model before training (D-029 reply check).

Usage: uv run python scripts/check_replies.py --config configs/sft/reply_check.yaml
With check.scope=sample, the teacher judges check.sample_per_variety training replies per
variety (seeded) as OK or BROKEN, plus the canary replies, which are judged on their own and never
counted. The broken share of the whole training set is estimated from the sample, with each variety
weighted by its share of the set. With check.scope=all, every reply of the files in check.splits
is judged; copies of those files without the BROKEN rows, and broken.jsonl (the BROKEN rows with
their split, index and reason, the input for rewriting them), are written to check.out_dir.
Writes judged.jsonl (every reply with its verdict and reason) and summary.json to check.out_dir.
Answers are saved as they arrive (answers.jsonl), so running the same config again resumes, and
a finished run re-writes its outputs without starting the teacher.
"""

import hashlib
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli
from danalm.data.reply_checks import parse_verdicts, weighted_rate
from danalm.teacher.checkpoint import AnswerLog
from danalm.teacher.client import TEACHER_ERRORS, chat_with_retries
from danalm.teacher.server import TeacherServer
from danalm.utils.run import start_run


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def pick_items(c: dict[str, Any], splits: dict[str, list[dict]], seed: int) -> list[dict]:
    """The replies to judge: a seeded sample per variety or every reply, then the canaries."""
    items = []
    if c["scope"] == "sample":
        rng = random.Random(seed)
        for v in c["varieties"]:
            idx = [i for i, r in enumerate(splits["train"]) if r["variety"] == v]
            items += [{"split": "train", "index": i, "variety": v, "counted": True}
                      for i in sorted(rng.sample(idx, c["sample_per_variety"]))]  # fmt: skip
    elif c["scope"] == "all":
        items += [{"split": s, "index": i, "variety": r["variety"], "counted": True}
                  for s, rows in splits.items() for i, r in enumerate(rows)]  # fmt: skip
    else:
        raise ValueError(f"check.scope must be 'sample' or 'all', not {c['scope']!r}")
    for it in items:
        it["reply"] = splits[it["split"]][it["index"]]["reply"]
    items += [{"split": "canary", "index": n, "variety": x["variety"], "reply": x["reply"],
               "counted": False} for n, x in enumerate(c["canaries"])]  # fmt: skip
    return items


def judge_batches(
    prompts: list[str], keys: list[str], log: AnswerLog, c: dict[str, Any], t: dict[str, Any],
    log_dir: Path,
) -> int:  # fmt: skip
    """Ask the teacher for every prompt not yet in the answer log; returns the failed requests.
    The teacher server is started only when there is something to ask."""
    todo = [i for i in range(len(prompts)) if i not in log.done]
    failed = 0
    if not todo:
        return failed
    with TeacherServer(t, log_dir / "llama-server.log") as server:

        def judge(prompt: str) -> str | None:
            messages = [
                {"role": "system", "content": c["system"]},
                {"role": "user", "content": prompt},
            ]
            try:
                answer, _ = chat_with_retries(
                    server.base_url, messages, c["params"], t["request_timeout_s"],
                    t["max_retries"], t["retry_backoff_s"],
                )  # fmt: skip
            except TEACHER_ERRORS as err:
                print(f"[teacher] a request failed: {err!r}", flush=True)
                return None
            return answer

        with ThreadPoolExecutor(t["parallel"]) as pool:
            try:
                for n, (i, answer) in enumerate(
                    zip(todo, pool.map(judge, [prompts[i] for i in todo]), strict=True), start=1
                ):
                    if answer is None:  # not saved, so a re-run asks again
                        failed += 1
                        if failed > t["max_failed_requests"]:
                            raise RuntimeError(f"{failed} failed requests; re-run to resume")
                        continue
                    log.add(i, keys[i], answer=answer)
                    if n % 50 == 0 or n == len(todo):
                        print(f"{len(log.done)}/{len(prompts)} requests judged", flush=True)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    return failed


def main() -> None:
    cfg = config_from_cli(__doc__)
    c, t = cfg["check"], cfg["teacher"]
    run = start_run(cfg, job_type="sft-reply-check")
    data_dir, out_dir = Path(c["data_dir"]), Path(c["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {s: read_jsonl(data_dir / f"{s}.jsonl") for s in c["splits"]}
    items = pick_items(c, splits, cfg["seed"])

    # one variety per request, so each prompt names one expected language
    groups: dict[tuple[bool, str], list[dict]] = {}
    for it in items:
        groups.setdefault((it["counted"], it["variety"]), []).append(it)
    batches = [(v, g[i : i + c["batch"]]) for (_, v), g in groups.items()
               for i in range(0, len(g), c["batch"])]  # fmt: skip
    prompts = [
        c["prompt"].format(n=len(b), language=c["languages"][v],
                           replies="\n".join(f"{k}. {it['reply']}" for k, it in enumerate(b, start=1)))
        for v, b in batches
    ]  # fmt: skip
    keys = [hashlib.md5(p.encode("utf-8")).hexdigest() for p in prompts]
    log = AnswerLog(out_dir / "answers.jsonl", keys)
    print(
        f"{len(items)} replies in {len(batches)} requests; {len(log.done)} already judged",
        flush=True,
    )
    start = time.time()
    try:
        failed = judge_batches(prompts, keys, log, c, t, run.dir)
    finally:
        log.close()

    judged = []
    for i, (_, b) in enumerate(batches):
        verdicts = parse_verdicts(log.done[i]["answer"], len(b)) if i in log.done else {}
        for k, it in enumerate(b, start=1):
            verdict, reason = verdicts.get(k, (None, ""))
            judged.append({**it, "verdict": verdict, "reason": reason})
    write_jsonl(out_dir / "judged.jsonl", judged)

    counted = [j for j in judged if j["counted"]]
    per: dict[str, Counter] = {}
    for j in counted:
        cnt = per.setdefault(j["variety"], Counter())
        cnt["replies"] += 1
        cnt["judged"] += j["verdict"] is not None
        cnt["broken"] += j["verdict"] == "BROKEN"
    # variety shares of the training split, or of everything checked when there is no such split
    base = splits.get("train") or [r for rows in splits.values() for r in rows]
    share = {v: n / len(base) for v, n in Counter(r["variety"] for r in base).items()}
    estimate = weighted_rate({v: x["broken"] for v, x in per.items()},
                             {v: x["judged"] for v, x in per.items()}, share)  # fmt: skip
    summary: dict[str, Any] = {
        "judge": t["model_name"], "scope": c["scope"], "splits": list(splits),
        "threshold": c["threshold"],
        "per_variety": {v: {**x, "rate": x["broken"] / max(1, x["judged"])} for v, x in per.items()},
        "estimated_broken_share": estimate,
        "canaries": [{"reply": j["reply"], "verdict": j["verdict"], "reason": j["reason"]}
                     for j in judged if not j["counted"]],
        "failed_requests": failed, "seconds": round(time.time() - start, 1),
    }  # fmt: skip
    if c["scope"] == "sample":
        summary["decision"] = ("judge all replies and drop the broken ones" if estimate > c["threshold"]
                               else "use the data as it is")  # fmt: skip
    else:
        verdict = {(j["split"], j["index"]): j for j in counted}
        broken = []
        for s, rows in splits.items():
            ok = [r for i, r in enumerate(rows) if verdict[(s, i)]["verdict"] != "BROKEN"]
            broken += [{**r, "split": s, "index": i, "broken_reason": verdict[(s, i)]["reason"]}
                       for i, r in enumerate(rows) if verdict[(s, i)]["verdict"] == "BROKEN"]  # fmt: skip
            write_jsonl(out_dir / f"{s}.jsonl", ok)
            summary[f"{s}_kept"] = len(ok)
            summary[f"{s}_broken"] = len(rows) - len(ok)
        write_jsonl(out_dir / "broken.jsonl", broken)  # the input for rewriting (D-031 A)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    (out_dir / "summary.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    run.wandb.summary.update({k: v for k, v in summary.items() if k != "canaries"})
    run.wandb.finish()


if __name__ == "__main__":
    main()
