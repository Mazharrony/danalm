"""Generate synthetic SFT examples (customer message -> intent + reply) with the local teacher,
filter them, and record the run in the data ledger.

Usage: uv run python scripts/generate_sft.py --config configs/sft/<name>.yaml
Writes <sft.out_dir>/generated.jsonl (kept), rejected.jsonl (with reasons), stats.json,
manifest.json and provenance files. The intent label is known by construction; verify_sft.py
then re-labels the kept rows blindly with a judge model.
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
from danalm.data.reply_checks import claims_done_action
from danalm.teacher.client import chat, parse_json_objects
from danalm.teacher.server import TeacherServer
from danalm.utils.run import start_run, write_provenance
from danalm.utils.seed import set_seed


def plan_requests(intents: list[dict], sft: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """One request per (intent, variety, repeat), with a seeded persona, tone and sampling seed."""
    rng = random.Random(seed)
    specs = []
    for intent in intents:
        for variety, v in sft["varieties"].items():
            # a variety may ask for more requests when its yield is low (e.g. mixed)
            for _ in range(v.get("requests_per_cell", sft["requests_per_cell"])):
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
        lang, reply_lang = detect_lang(msg), detect_lang(reply)
        if lang not in v["keep_langs"]:
            return None, f"message_lang_{lang}"
        if reply_lang not in v["reply_langs"]:
            return None, f"reply_lang_{reply_lang}"
        if claims_done_action(reply):
            return None, "reply_claims_action"
        if v["gulf_filter"] and (non_gulf_markers(msg) or non_gulf_markers(reply)):
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
    keep = Filter(sft)
    stats: Counter = Counter()
    kept, rejected = [], []
    tokens = 0

    start = time.time()
    with TeacherServer(t, run.dir / "llama-server.log") as server:

        def request(spec: dict[str, Any]) -> tuple[dict, str, dict]:
            messages = [
                {"role": "system", "content": sft["system"]},
                {"role": "user", "content": build_prompt(spec, sft)},
            ]
            params = {**sft["params"], "seed": spec["seed"]}
            answer, usage = chat(server.base_url, messages, params, t["request_timeout_s"])
            return spec, answer, usage

        with ThreadPoolExecutor(t["parallel"]) as pool:
            for i, (spec, answer, usage) in enumerate(pool.map(request, specs), start=1):
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
                if i % 20 == 0 or i == len(specs):
                    rate = tokens / (time.time() - start)
                    print(
                        f"{i}/{len(specs)} requests, {len(kept)} kept, {rate:.0f} tok/s", flush=True
                    )
                    run.wandb.log({"requests": i, "kept": len(kept), "tokens_per_s": rate})

    seconds = time.time() - start
    stats |= {"completion_tokens": tokens, "seconds": round(seconds, 1),
              "tokens_per_s": round(tokens / seconds, 1), "kept_per_min": round(len(kept) / seconds * 60, 1)}  # fmt: skip
    for name, rows in (("generated", kept), ("rejected", rejected)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    manifest = {"teacher": t["model_name"], "revision": t["revision"], "params": sft["params"],
                "prompt": sft["prompt"], "requests": len(specs)}  # fmt: skip
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
