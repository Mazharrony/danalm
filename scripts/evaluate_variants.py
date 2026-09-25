"""Score the ONNX variants against PyTorch float32 and choose the deployed one (Phase 7, D-034).

Usage:
  uv run python scripts/evaluate_variants.py --config configs/deploy/phase7.yaml                   # dev + choice
  uv run python scripts/evaluate_variants.py --config configs/deploy/phase7.yaml phase7.split=test # test, once
Systems:
- pytorch-fp32: the reference, PyTorch float32 on the CPU without the cache (the D-029 decoding
  of danalm.sft.evaluate.evaluate);
- pytorch-fp32-cache (dev only): the same model with the KV cache (danalm.eval.cached);
- onnx-<variant> for every variant in phase7.variants, with the KV cache.
dev: on phase7.dev_sets. Every system gets its threshold on phase7.threshold_set (D-030 target);
the ONNX variants' thresholds are written into their danalm.json. The D-034 rule chooses the
deployed variant and writes phase7.selected.
test: refuses to run before the choice exists, checks the test file's SHA-256, and scores every
system once with its dev threshold. Nothing is chosen from it.
Predictions and metrics go to phase7.out_dir; finished systems are read back instead of re-run
(the reference takes ~30 minutes on the dev sets). The process opts out of Windows power
throttling (danalm.utils.power).
"""

import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from danalm.config import config_from_cli
from danalm.data.intents import load_intents
from danalm.eval.cached import evaluate_cached
from danalm.infer.onnx import OnnxStep
from danalm.model.export import empty_cache, numpy_step
from danalm.model.transformer import DanaLM, ModelConfig
from danalm.sft.evaluate import coverage_at, evaluate, summarize
from danalm.sft.format import ChatTokens
from danalm.utils.power import disable_power_throttling


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def load_test(path: str, expected: str) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise ValueError(f"test set changed: sha256 {digest} != {expected} (D-032)")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return [{"id": r["id"], "message": r["text"], "intent": r["intent"], "variety": r["variety"]}
            for r in rows]  # fmt: skip


def agreement(ref: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, float]:
    n = len(ref)
    return {
        "same_intent": sum(a["pred_intent"] == b["pred_intent"] for a, b in zip(ref, preds, strict=True)) / n,
        "same_answer": sum(a["output"] == b["output"] for a, b in zip(ref, preds, strict=True)) / n,
    }  # fmt: skip


def covered(preds: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    """Coverage and accuracy at a threshold fixed elsewhere (None: nothing is answered)."""
    got = [x for x in preds if threshold is not None and x["conf"] >= threshold]
    return {"threshold": threshold, "answered": len(got), "coverage": len(got) / len(preds),
            "accuracy": sum(x["pred_intent"] == x["intent"] for x in got) / len(got) if got else None}  # fmt: skip


def choose(p: dict[str, Any], passing: list[str], sizes: dict[str, int]) -> tuple[str, str]:
    """D-034: the smallest passing variant; within size_tie of it, the lowest median latency."""
    if not passing:
        return "fp32", "no quantized variant passed the rule, so onnx-fp32 is deployed"
    smallest = min(passing, key=lambda v: sizes[v])
    tied = [v for v in passing if sizes[v] <= sizes[smallest] * (1 + p["rule"]["size_tie"])]
    if len(tied) == 1:
        return smallest, f"the smallest passing variant ({sizes[smallest] / 2**20:.1f} MB)"
    lat_path = Path(p["latency_results"])
    if not lat_path.exists():
        raise SystemExit(f"a size tie between {tied}: run scripts/benchmark_latency.py first")
    lat = json.loads(lat_path.read_text(encoding="utf-8"))
    best = min(tied, key=lambda v: lat[f"onnx-{v}"]["threads_4"]["full_median_s"])
    return (
        best,
        f"within {100 * p['rule']['size_tie']:.0f}% in size of {tied}: the lowest median latency",
    )


def main() -> None:
    cfg = config_from_cli(__doc__)
    p, d, ev = cfg["phase7"], cfg["sft_data"], cfg["eval"]
    throttling_off = disable_power_throttling()
    out = Path(p["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    deploy = Path(cfg["deploy"]["out_dir"])
    split = p["split"]
    if split == "dev":
        sets = {name: read_jsonl(path) for name, path in p["dev_sets"].items()}
    elif split == "test":
        if not Path(p["selected"]).exists():
            raise SystemExit("choose the variant on the development sets first (D-034)")
        sets = {"test": load_test(p["test_set"], p["test_sha256"])}
    else:
        raise ValueError(f"phase7.split must be dev or test, not {split!r}")

    tok = Tokenizer.from_file(d["tokenizer_file"])
    chat = ChatTokens.from_tokenizer(tok, d["special"])
    intents = [i["name"] for i in load_intents(d["intents_file"])]
    checkpoint = json.loads(Path(cfg["deploy"]["selected"]).read_text(encoding="utf-8"))["checkpoint"]  # fmt: skip
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DanaLM(ModelConfig(**state["model_config"])).eval()
    model.load_state_dict(state["model"])
    torch.set_num_threads(p["reference_threads"])
    common = (tok, chat)
    extra = (intents, d["normalize"], d["reply_langs"], ev)

    def run_reference(rows):
        with torch.no_grad():
            return evaluate(model, *common, rows, *extra, contextlib.nullcontext)

    def run_cached(step, empty):
        return lambda rows: evaluate_cached(step, empty, *common, rows, *extra, p["batch"], p["label_batch"])  # fmt: skip

    systems = {"pytorch-fp32": run_reference}
    if split == "dev":
        systems["pytorch-fp32-cache"] = run_cached(
            numpy_step(model), lambda b: empty_cache(model, b)
        )
    for v in p["variants"]:
        step = OnnxStep(deploy / v / "model.onnx", p["onnx_threads"])
        systems[f"onnx-{v}"] = run_cached(step, step.empty)

    preds: dict[str, dict[str, list]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for sys_name, run in systems.items():
        for set_name, rows in sets.items():
            path = out / f"{split}_{set_name}_{sys_name}.jsonl"
            if path.exists():
                got = read_jsonl(path)
                m = summarize(got, ev["target_accuracy"])
            else:
                m, got = run(rows)
                for r, g in zip(rows, got, strict=True):
                    if "id" in r:
                        g["id"] = r["id"]
                write_jsonl(path, got)
            preds.setdefault(sys_name, {})[set_name] = got
            metrics.setdefault(sys_name, {})[set_name] = m
            print(f"{split} {set_name:<9} {sys_name:<20} acc {m['intent_accuracy']:.4f} "
                  f"json {m['valid_json']:.4f} lang {m['reply_lang']:.4f}", flush=True)  # fmt: skip

    ref = preds["pytorch-fp32"]
    agree = {s: {k: agreement(ref[k], preds[s][k]) for k in sets} for s in systems if s != "pytorch-fp32"}  # fmt: skip
    if split == "dev":
        target = ev["target_accuracy"]
        thr_set = p["threshold_set"]
        thresholds = {}
        for s in systems:
            ps = preds[s][thr_set]
            thresholds[s] = coverage_at([x["conf"] for x in ps],
                                        [x["pred_intent"] == x["intent"] for x in ps], target)["threshold"]  # fmt: skip
        cover = {s: {k: covered(preds[s][k], thresholds[s]) for k in sets} for s in systems}
        for v in p["variants"]:
            meta_path = deploy / v / "danalm.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["threshold"] = thresholds[f"onnx-{v}"]
            meta["threshold_source"] = f"{thr_set}: the lowest confidence with >= {target:.0%} intent accuracy above it (D-030, D-033)"  # fmt: skip
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        r = p["rule"]
        exact = {s: all(agree[s][k]["same_answer"] >= r["min_same_answer"] for k in sets)
                 for s in ("pytorch-fp32-cache", "onnx-fp32")}  # fmt: skip
        base = metrics["onnx-fp32"]
        checks = {}
        for v in p["variants"]:
            if v == "fp32":
                continue
            m = metrics[f"onnx-{v}"]
            checks[v] = {k: {
                "accuracy_ok": m[k]["intent_accuracy"] >= base[k]["intent_accuracy"] - r["max_accuracy_drop"],
                "valid_json_ok": m[k]["valid_json"] >= r["min_valid_json"],
                "reply_lang_ok": m[k]["reply_lang"] >= r["min_reply_lang"],
            } for k in sets}  # fmt: skip
        passing = [v for v, c in checks.items() if all(all(x.values()) for x in c.values())]
        sizes = {v: (deploy / v / "model.onnx").stat().st_size for v in p["variants"]}
        variant, reason = choose(p, passing, sizes)
        selected = {"variant": variant, "model_dir": str(deploy / variant), "reason": reason,
                    "threshold": thresholds[f"onnx-{variant}"], "bytes": sizes[variant],
                    "passing": passing, "checks": checks, "exactness_ok": exact}  # fmt: skip
        Path(p["selected"]).write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        result = {"split": split, "power_throttling_off": throttling_off, "checkpoint": checkpoint,
                  "metrics": metrics, "agreement": agree, "thresholds": thresholds,
                  "coverage_at_threshold": cover, "sizes": sizes, "selected": selected}  # fmt: skip
    else:
        dev = json.loads((out / "dev_metrics.json").read_text(encoding="utf-8"))
        cover = {s: {k: covered(preds[s][k], dev["thresholds"][s]) for k in sets} for s in systems}
        result = {"split": split, "power_throttling_off": throttling_off, "checkpoint": checkpoint,
                  "test_sha256": p["test_sha256"], "metrics": metrics, "agreement": agree,
                  "coverage_at_dev_threshold": cover,
                  "selected": json.loads(Path(p["selected"]).read_text(encoding="utf-8"))}  # fmt: skip
    text = json.dumps(result, indent=2)
    (out / f"{split}_metrics.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("agreement",)}, indent=2))
    if split == "dev":
        print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
