"""Fine-tune the CAMeLBERT-mix intent classifier, the Phase 6 baseline (D-027, D-031 G).

Usage: uv run python scripts/baseline_camelbert.py --config configs/eval/baseline_camelbert.yaml
Downloads baseline.repo_id at the pinned revision (the PyTorch weights and tokenizer files only),
fine-tunes it as a 21-intent classifier on the SFT training messages (normalized as in the SFT
data), evaluates accuracy and macro-F1 on the SFT validation split after every epoch, and keeps
the best epoch (by validation accuracy) in <out_dir>/best/. The human test set is not used
(Phase 6 scores it). Refuses to run while the teacher server is up.
"""

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from danalm.config import config_from_cli
from danalm.data.intents import load_intents
from danalm.data.pipeline import normalize
from danalm.sft.evaluate import intent_metrics
from danalm.teacher.server import assert_teacher_stopped
from danalm.utils.run import start_run
from danalm.utils.seed import set_seed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def main() -> None:
    cfg = config_from_cli(__doc__)
    assert_teacher_stopped(cfg["teacher"]["host"], cfg["teacher"]["port"])
    set_seed(cfg["seed"], cfg["deterministic"])
    b = cfg["baseline"]
    out = Path(b["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    local = snapshot_download(b["repo_id"], revision=b["revision"], allow_patterns=b["files"])
    names = [i["name"] for i in load_intents(b["intents_file"])]
    label = {n: k for k, n in enumerate(names)}
    tok = AutoTokenizer.from_pretrained(local)
    model = AutoModelForSequenceClassification.from_pretrained(
        local, num_labels=len(names), id2label=dict(enumerate(names)), label2id=label
    ).to(device)
    rows = {s: read_jsonl(Path(b["data_dir"]) / f"{s}.jsonl") for s in ("train", "val")}
    texts = {s: [normalize(r["message"], **b["normalize"]) for r in rs] for s, rs in rows.items()}
    gold = {s: [label[r["intent"]] for r in rs] for s, rs in rows.items()}

    def encode(batch: list[str]) -> dict[str, torch.Tensor]:
        enc = tok(
            batch, truncation=True, max_length=b["max_length"], padding=True, return_tensors="pt"
        )
        return {k: v.to(device) for k, v in enc.items()}

    @torch.no_grad()
    def predict(split: str) -> list[int]:
        model.eval()
        preds = []
        for i in range(0, len(texts[split]), b["eval_batch"]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**encode(texts[split][i : i + b["eval_batch"]])).logits
            preds += logits.argmax(-1).tolist()
        return preds

    steps = b["epochs"] * -(-len(texts["train"]) // b["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=b["lr"], weight_decay=b["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, b["warmup_ratio"] * steps)) * max(0.0, 1 - s / steps),
    )
    run = start_run(cfg, job_type="baseline")
    rng, best = random.Random(cfg["seed"]), -1.0
    history = []
    for epoch in range(1, b["epochs"] + 1):
        model.train()
        order = list(range(len(texts["train"])))
        rng.shuffle(order)
        start, losses = time.perf_counter(), []
        for i in range(0, len(order), b["batch_size"]):
            idx = order[i : i + b["batch_size"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**encode([texts["train"][k] for k in idx]),
                             labels=torch.tensor([gold["train"][k] for k in idx], device=device)).loss  # fmt: skip
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        preds = predict("val")
        m = intent_metrics([names[g] for g in gold["val"]], [names[p] for p in preds])
        record = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_accuracy": m["accuracy"],
                  "val_macro_f1": m["macro_f1"], "seconds": round(time.perf_counter() - start, 1)}  # fmt: skip
        history.append(record)
        run.wandb.log(record, step=epoch)
        print(json.dumps(record), flush=True)
        if m["accuracy"] > best:
            best = m["accuracy"]
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")
            (out / "best" / "epoch.json").write_text(
                json.dumps(record) + "\n", encoding="utf-8", newline="\n"
            )
    summary = {"repo_id": b["repo_id"], "revision": b["revision"], "epochs": history,
               "best_epoch": max(history, key=lambda h: h["val_accuracy"])["epoch"]}  # fmt: skip
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    run.wandb.finish()
    print(f"best epoch {summary['best_epoch']} -> {out / 'best'}")


if __name__ == "__main__":
    main()
