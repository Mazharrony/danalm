"""Write new assistant replies for existing SFT messages with the teacher.

Usage: uv run python scripts/write_replies.py --config configs/sft/<name>.yaml
Reads the JSONL rows in replies.inputs (message, intent, variety, ...), keeps the varieties listed
in replies.varieties, and asks the teacher for one reply per message, replies.batch messages per
request. Each new reply is normalized and checked like a generated one (language, claimed
actions, Gulf dialect). Writes <replies.out_dir>/generated.jsonl (the rows with the new reply;
the old one, if any, is kept as reply_original), rejected.jsonl, stats.json and a ledger entry.
Teacher answers are saved to answers.jsonl as they arrive, so a crashed run resumes.
"""

import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli
from danalm.data import ledger
from danalm.data.hub import text_stats
from danalm.data.intents import load_intents
from danalm.data.pipeline import detect_lang, normalize
from danalm.data.reply_checks import reply_problem
from danalm.teacher.checkpoint import AnswerLog
from danalm.teacher.client import TEACHER_ERRORS, chat_with_retries, parse_numbered
from danalm.teacher.server import TeacherServer
from danalm.utils.run import start_run, write_provenance


def build_prompt(batch: list[dict], r: dict[str, Any], sft: dict[str, Any], intents: dict) -> str:
    """One numbered line per message with its intent (and the intent's reply rule, if any)."""
    lines = []
    for n, row in enumerate(batch, start=1):
        rule = sft["intent_rules"].get(row["intent"], "").strip()
        about = intents[row["intent"]]["description"]
        lines.append(f"{n}. intent {row['intent']} ({about}){' ' + rule if rule else ''}")
        lines.append(f"   message: {row['message']}")
    style = r["varieties"][batch[0]["variety"]]["reply_style"]
    return r["prompt"].format(n=len(batch), reply_style=style, messages="\n".join(lines))


def main() -> None:
    cfg = config_from_cli(__doc__)
    run = start_run(cfg, job_type="sft-replies")
    r, t, sft = cfg["replies"], cfg["teacher"], cfg["sft"]
    out = Path(r["out_dir"])
    write_provenance(out, cfg)
    intents = {i["name"]: i for i in load_intents(sft["intents_file"])}
    rows = []
    for path in r["inputs"]:
        with open(path, encoding="utf-8") as fh:
            rows += [x for x in map(json.loads, fh) if x["variety"] in r["varieties"]]
    if r["max_rows"]:
        rows = rows[: r["max_rows"]]
    # batches never mix varieties, so each prompt has one reply style
    batches = []
    for v in r["varieties"]:
        vrows = [x for x in rows if x["variety"] == v]
        batches += [vrows[i : i + r["batch"]] for i in range(0, len(vrows), r["batch"])]
    prompts = [build_prompt(b, r, sft, intents) for b in batches]
    keys = [hashlib.md5(p.encode("utf-8")).hexdigest() for p in prompts]
    log = AnswerLog(out / "answers.jsonl", keys)
    print(
        f"{len(rows):,} messages in {len(batches):,} requests; {len(log.done):,} already answered"
    )

    def ask(i: int) -> str | None:
        messages = [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": prompts[i]},
        ]
        try:
            answer, _ = chat_with_retries(
                server.base_url, messages, r["params"], t["request_timeout_s"],
                t["max_retries"], t["retry_backoff_s"],
            )  # fmt: skip
        except TEACHER_ERRORS as err:
            print(f"[teacher] request {i} failed: {err!r}", flush=True)
            return None
        return answer

    start, failed = time.time(), 0
    todo = [i for i in range(len(batches)) if i not in log.done]
    with (
        TeacherServer(t, run.dir / "llama-server.log") as server,
        ThreadPoolExecutor(t["parallel"]) as pool,
    ):
        try:
            for n, (i, answer) in enumerate(zip(todo, pool.map(ask, todo), strict=True), 1):
                if answer is None:  # not saved, so a re-run asks again
                    failed += 1
                    if failed > t["max_failed_requests"]:
                        raise RuntimeError(f"{failed} failed requests; re-run to resume")
                    continue
                log.add(i, keys[i], answer=answer)
                if n % 50 == 0 or n == len(todo):
                    print(f"{len(log.done)}/{len(batches)} requests answered", flush=True)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)  # don't send the queued requests
            raise
        finally:
            log.close()

    stats: Counter = Counter()
    kept, rejected = [], []
    for i, batch in enumerate(batches):
        replies = (
            parse_numbered(log.done[i]["answer"], len(batch), "reply") if i in log.done else {}
        )
        for k, row in enumerate(batch, start=1):
            v = r["varieties"][row["variety"]]
            reply = normalize(replies[k], **sft["normalize"]) if k in replies else ""
            reason = "no_reply" if not reply else reply_problem(reply, v["reply_langs"], v["gulf_filter"])  # fmt: skip
            new = {**row, "lang": detect_lang(row["message"]), "reply": reply, "reply_original": row.get("reply")}  # fmt: skip
            stats[f"{row['variety']}/{reason or 'kept'}"] += 1
            if reason:
                rejected.append({**new, "reason": reason})
            else:
                kept.append(new)
    seconds = time.time() - start
    stats |= {"messages": len(rows), "requests": len(batches), "failed_requests": failed}
    for name, data in (("generated", kept), ("rejected", rejected)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in data)
    (out / "stats.json").write_text(json.dumps(dict(stats), indent=2), encoding="utf-8")

    text = text_stats([f"{x['message']}\n{x['reply']}" for x in kept])
    entry = {
        "id": cfg["run_name"],
        "source": f"teacher: {t['model_name']} via llama.cpp; new replies for "
        + ", ".join(r["inputs"]),
        "revision": t["revision"],
        "licence": "Apache-2.0 (teacher outputs)",
        "kind": "synthetic",
        "collected_on": date.today().isoformat(),
        "used_for": sft["used_for"],
        "docs_raw": len(rows),
        "docs_kept": len(kept),
        "utf8_bytes": text["utf8_bytes"],
        "words": text["words"],
        "notes": f"{len(batches)} requests; varieties {', '.join(r['varieties'])}; "
        "before judge verification",
    }
    ledger.write_markdown(cfg["ledger"]["markdown"], ledger.upsert(cfg["ledger"]["jsonl"], [entry]))
    run.wandb.summary.update(dict(stats))
    run.wandb.finish()
    print(json.dumps(dict(stats), indent=2))
    print(f"{len(kept):,} kept, {len(rejected):,} rejected in {seconds / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
