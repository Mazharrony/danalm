"""Generate synthetic SFT examples (customer message -> intent + reply) with the local teacher,
filter them, and record the run in the data ledger.

Usage: uv run python scripts/generate_sft.py --config configs/sft/<name>.yaml
Writes <sft.out_dir>/generated.jsonl (kept), rejected.jsonl (with reasons), stats.json,
manifest.json and provenance files. The intent label is known by construction; verify_sft.py
then re-labels the kept rows blindly with a judge model.
Every teacher answer is saved to <sft.out_dir>/answers.jsonl as it arrives; running the same
config again after a crash resumes from there (saved answers are re-filtered in plan order, so
the result matches an uninterrupted run).
"""

import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

from datasketch import MinHashLSH

from danalm.config import config_from_cli
from danalm.data import ledger
from danalm.data.dialect import msa_markers, non_gulf_markers
from danalm.data.hub import text_stats
from danalm.data.intents import load_intents
from danalm.data.pipeline import detect_lang, minhash, normalize
from danalm.data.reply_checks import reply_problem
from danalm.teacher.checkpoint import AnswerLog
from danalm.teacher.client import TEACHER_ERRORS, chat_with_retries, parse_json_objects
from danalm.teacher.server import TeacherServer
from danalm.utils.run import start_run, write_provenance
from danalm.utils.seed import set_seed


def plan_requests(intents: list[dict], sft: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """One request per (intent, variety, repeat), with a seeded persona, tone and sampling seed.
    `only_intents` / `only_varieties` (lists or null) restrict the plan, e.g. for a top-up run.
    `cell_requests` ({"intent/variety": n}, optional) sets the number of requests per cell
    instead; cells it does not list get none (D-031 top-up)."""
    rng = random.Random(seed)
    cells = sft.get("cell_requests")
    specs = []
    for intent in intents:
        if sft["only_intents"] and intent["name"] not in sft["only_intents"]:
            continue
        for variety, v in sft["varieties"].items():
            if sft["only_varieties"] and variety not in sft["only_varieties"]:
                continue
            # a variety may ask for more requests when its yield is low (e.g. mixed)
            n = v.get("requests_per_cell", sft["requests_per_cell"])
            if cells is not None:
                n = cells.get(f"{intent['name']}/{variety}", 0)
            for _ in range(n):
                specs.append(
                    {
                        **intent,
                        "intent": intent["name"],
                        "variety": variety,
                        "persona": rng.choice(v["personas"]),
                        "tone": rng.choice(sft["tones"]),
                        "seed": rng.randrange(2**31),
                    }
                )
    return specs


def build_prompt(spec: dict[str, Any], sft: dict[str, Any]) -> str:
    v = sft["varieties"][spec["variety"]]
    return sft["prompt"].format(
        n=sft["examples_per_request"],
        intent=spec["intent"],
        domain=spec["domain"],
        description=spec["description"],
        persona=spec["persona"],
        tone=spec["tone"],
        style=v["style"].strip(),
        reply_style=v["reply_style"],
        intent_rule=sft["intent_rules"].get(spec["intent"], ""),
        # optional: makes the message clearly this intent (D-031); unused by older prompts
        message_rule=(sft.get("message_rules") or {}).get(spec["intent"], ""),
    )


class Filter:
    """Normalizes generated examples and rejects bad fields, wrong language, replies that claim
    an action was done, non-Gulf dialect or formal MSA in Gulf messages, and exact or near
    duplicates (across the whole run)."""

    def __init__(self, sft: dict[str, Any]) -> None:
        self.sft = sft
        self.lsh = MinHashLSH(threshold=sft["near_dup_threshold"], num_perm=sft["minhash_num_perm"])
        self.seen: set[str] = set()

    def __call__(self, example: dict, spec: dict[str, Any]) -> tuple[dict | None, str]:
        msg, reply = example.get("message"), example.get("reply")
        if not (isinstance(msg, str) and isinstance(reply, str) and msg.strip() and reply.strip()):
            return None, "bad_fields"
        v = self.sft["varieties"][spec["variety"]]
        msg, reply = normalize(msg, **self.sft["normalize"]), normalize(
            reply, **self.sft["normalize"]
        )
        lang = detect_lang(msg)
        if lang not in v["keep_langs"]:
            return None, f"message_lang_{lang}"
        if problem := reply_problem(reply, v["reply_langs"], v["gulf_filter"]):
            return None, problem
        if v["gulf_filter"] and non_gulf_markers(msg):
            return None, "non_gulf_dialect"
        if v["gulf_filter"] and msa_markers(msg):
            return None, "msa_in_message"
        key = msg.lower()
        if key in self.seen:
            return None, "duplicate"
        mh = minhash(msg, self.sft["minhash_num_perm"], self.sft["shingle_words"])
        if self.lsh.query(mh):
            return None, "near_duplicate"
        self.lsh.insert(key, mh)
        self.seen.add(key)
        return {
            "message": msg,
            "intent": spec["intent"],
            "reply": reply,
            "variety": spec["variety"],
            "lang": lang,
            "domain": spec["domain"],
            "persona": spec["persona"],
            "tone": spec["tone"],
        }, "kept"


def main() -> None:
    cfg = config_from_cli(__doc__)
    set_seed(cfg["seed"], cfg["deterministic"])
    run = start_run(cfg, job_type="sft-generate")
    sft, t = cfg["sft"], cfg["teacher"]
    out = Path(sft["out_dir"])
    write_provenance(out, cfg)
    specs = plan_requests(load_intents(sft["intents_file"]), sft, cfg["seed"])
    keys = [f"{s['intent']}|{s['variety']}|{s['seed']}" for s in specs]
    log = AnswerLog(out / "answers.jsonl", keys)
    resumed = len(log.done)
    if resumed:
        print(f"resuming: {resumed}/{len(specs)} answers already saved in {log.path}", flush=True)
    keep = Filter(sft)
    stats: Counter = Counter(resumed_requests=resumed)
    kept, rejected = [], []
    tokens = fresh_tokens = 0

    start = time.time()
    with TeacherServer(t, run.dir / "llama-server.log") as server:

        def request(i: int) -> tuple[str | None, dict]:
            messages = [
                {"role": "system", "content": sft["system"]},
                {"role": "user", "content": build_prompt(specs[i], sft)},
            ]
            params = {**sft["params"], "seed": specs[i]["seed"]}
            try:
                return chat_with_retries(
                    server.base_url, messages, params, t["request_timeout_s"],
                    t["max_retries"], t["retry_backoff_s"],
                )  # fmt: skip
            except TEACHER_ERRORS as err:
                print(f"[teacher] request {i} failed: {err!r}", flush=True)
                return None, {}

        todo = [i for i in range(len(specs)) if i not in log.done]
        with ThreadPoolExecutor(t["parallel"]) as pool:
            fresh = zip(todo, pool.map(request, todo), strict=True)
            try:
                for i, spec in enumerate(specs):
                    if i in log.done:
                        answer, usage = log.done[i]["answer"], log.done[i]["usage"]
                    else:
                        _, (answer, usage) = next(fresh)
                        if answer is None:  # not saved, so a re-run asks again
                            stats["failed_requests"] += 1
                            if stats["failed_requests"] > t["max_failed_requests"]:
                                raise RuntimeError(
                                    f"{stats['failed_requests']} failed requests; progress is "
                                    f"saved in {log.path}, re-run to resume"
                                )
                            continue
                        log.add(i, keys[i], answer=answer, usage=usage)
                        fresh_tokens += usage.get("completion_tokens", 0)
                    tokens += usage.get("completion_tokens", 0)
                    examples = parse_json_objects(answer)
                    stats["requests"] += 1
                    stats["examples_parsed"] += len(examples)
                    stats["short_answers"] += len(examples) < sft["examples_per_request"]
                    for example in examples:
                        row, reason = keep(example, spec)
                        stats[reason] += 1
                        stats[f"{spec['variety']}/{reason}"] += 1
                        if row:
                            kept.append(row)
                        else:
                            rejected.append(
                                {
                                    **example,
                                    **{k: spec[k] for k in ("intent", "variety")},
                                    "reason": reason,
                                }
                            )
                    if (i + 1) % 20 == 0 or i + 1 == len(specs):
                        rate = fresh_tokens / (time.time() - start)
                        print(
                            f"{i + 1}/{len(specs)} requests, {len(kept)} kept, {rate:.0f} tok/s",
                            flush=True,
                        )
                        run.wandb.log({"requests": i + 1, "kept": len(kept), "tokens_per_s": rate})
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)  # don't send the queued requests
                raise
            finally:
                log.close()

    seconds = time.time() - start
    # speeds cover this process only; kept_per_min is left out when earlier answers were reused
    summary = {"completion_tokens": tokens, "seconds": round(seconds, 1),
               "tokens_per_s": round(fresh_tokens / seconds, 1),
               "kept_per_min": None if resumed else round(len(kept) / seconds * 60, 1)}  # fmt: skip
    stats = dict(stats) | summary  # a plain dict: Counter's |= keeps the larger value per key
    for name, rows in (("generated", kept), ("rejected", rejected)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    manifest = {"teacher": t["model_name"], "revision": t["revision"], "params": sft["params"],
                "prompt": sft["prompt"], "requests": len(specs), "resumed_requests": resumed,
                "failed_requests": stats.get("failed_requests", 0)}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")

    text = text_stats([f"{r['message']}\n{r['reply']}" for r in kept])
    entry = {
        "id": cfg["run_name"],
        "source": f"teacher: {t['model_name']} via llama.cpp",
        "revision": t["revision"],
        "licence": "Apache-2.0 (teacher outputs)",
        "kind": "synthetic",
        "collected_on": date.today().isoformat(),
        "used_for": sft["used_for"],
        "docs_raw": stats["examples_parsed"],
        "docs_kept": len(kept),
        "utf8_bytes": text["utf8_bytes"],
        "words": text["words"],
        "notes": f"{len(specs)} requests, {tokens:,} teacher tokens; before judge verification",
    }
    ledger.write_markdown(cfg["ledger"]["markdown"], ledger.upsert(cfg["ledger"]["jsonl"], [entry]))
    run.wandb.summary.update(dict(stats))
    run.wandb.finish()
    print(json.dumps({k: v for k, v in stats.items() if "/" not in k}, indent=2))
    print(f"\nSaved to {out}; ledger updated")


if __name__ == "__main__":
    main()
