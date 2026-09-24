"""Fine-tune the pretrained model on the SFT data (Phase 5, D-029).

Usage: uv run python scripts/sft.py --config configs/sft/train.yaml [train.lr=... run_name=...]
Starts from train.init (a pretrained model.pt) and trains train.epochs epochs on
<sft_data.data_dir>/train.jsonl, with the loss on the answer tokens only. After every epoch:
- saves epoch_<k>.pt (the model weights and their config);
- measures the development metrics on val.jsonl: the answer loss, then with greedy decoding the
  valid-JSON rate, intent accuracy and macro-F1 and the reply-language rate, intent by
  likelihood, and the D-030 coverage;
- writes the validation predictions to val_predictions_epoch<k>.jsonl.
Metrics go to <out_dir>/metrics.jsonl and W&B. A finished run exits at once; an unfinished one
starts over (a run takes minutes). Refuses to run while the teacher server is up.
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from danalm.config import config_from_cli
from danalm.data.intents import load_intents
from danalm.model.transformer import DanaLM, ModelConfig
from danalm.sft.data import ChatTokens, collate, example_ids, label_continuations, length_batches
from danalm.sft.evaluate import answer_loss, evaluate
from danalm.teacher.server import assert_teacher_stopped
from danalm.train.loop import lr_at, make_optimizer, save_checkpoint, set_lr, train_step
from danalm.utils.run import start_run
from danalm.utils.seed import set_seed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def check_targets(rows: list[dict], enc: list[tuple[list[int], list[int]]], prefix: list[int],
                  conts: dict[str, list[int]]) -> None:  # fmt: skip
    """Every answer must start with the tokens that intent scoring uses, or the likelihood
    metrics would score something the model never learned."""
    bad = 0
    for r, (_, labels) in zip(rows, enc, strict=True):
        answer = [t for t in labels if t >= 0]
        want = prefix + conts[r["intent"]]
        bad += answer[: len(want)] != want
    if bad:
        raise ValueError(f"{bad} targets do not start with the scored intent tokens")


def main() -> None:
    cfg = config_from_cli(__doc__)
    assert_teacher_stopped(cfg["teacher"]["host"], cfg["teacher"]["port"])
    set_seed(cfg["seed"], cfg["deterministic"])
    d, t, ev = cfg["sft_data"], cfg["train"], cfg["eval"]
    out = Path(t["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    finished = (out / f"epoch_{t['epochs']}.pt").exists() and metrics_path.exists()
    if finished and len(metrics_path.read_text(encoding="utf-8").splitlines()) >= t["epochs"]:
        print(f"already finished: {out}")
        return

    device = "cuda"
    tok = Tokenizer.from_file(d["tokenizer_file"])
    chat = ChatTokens.from_tokenizer(tok, d["special"])
    intents = [i["name"] for i in load_intents(d["intents_file"])]
    rows = {s: read_jsonl(Path(d["data_dir"]) / f"{s}.jsonl") for s in ("train", "val")}
    enc = {s: [example_ids(tok, chat, r["message"], r["target"], d["normalize"]) for r in rs]
           for s, rs in rows.items()}  # fmt: skip
    prefix, conts = label_continuations(tok, intents)
    for s in rows:
        check_targets(rows[s], enc[s], prefix, dict(zip(intents, conts, strict=True)))
    longest = max(sum(i >= 0 for i in labels) for _, labels in enc["train"])
    if longest > ev["max_new_tokens"]:
        raise ValueError(f"eval.max_new_tokens {ev['max_new_tokens']} < longest answer {longest}")

    state = torch.load(t["init"], map_location="cpu", weights_only=True)
    model = DanaLM(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    opt = make_optimizer(model, t["lr"], tuple(t["betas"]), t["weight_decay"], fused=True)
    lengths = [len(ids) for ids, _ in enc["train"]]
    per_epoch = math.ceil(len(lengths) / t["batch_size"])
    total = per_epoch * t["epochs"]
    warmup = max(1, round(t["warmup_ratio"] * total))
    rng = random.Random(cfg["seed"])
    run = start_run(cfg, job_type="sft")
    print(f"{len(lengths):,} training examples, {per_epoch} steps per epoch, {total} steps; "
          f"longest answer {longest} tokens", flush=True)  # fmt: skip

    def autocast() -> torch.autocast:
        return torch.autocast("cuda", dtype=torch.bfloat16)

    metrics_path.write_text("", encoding="utf-8")  # an unfinished run starts over
    step = 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        start, losses = time.perf_counter(), []
        for batch in length_batches(lengths, t["batch_size"], t["bucket_chunks"], rng):
            lr = lr_at(step, total, warmup, t["lr"], t["lr"] * t["min_lr_ratio"])
            set_lr(opt, lr)
            x, y = collate([enc["train"][i] for i in batch], chat.pad)
            loss, grad_norm = train_step(model, opt, [(x.to(device), y.to(device))],
                                         t["grad_clip"], autocast)  # fmt: skip
            losses.append(loss)
            step += 1
            if step % t["log_every"] == 0:
                run.wandb.log({"train_loss": loss, "lr": lr, "grad_norm": grad_norm}, step=step)
        train_seconds = time.perf_counter() - start
        save_checkpoint(out / f"epoch_{epoch}.pt", {
            "model": model.state_dict(), "model_config": state["model_config"], "epoch": epoch,
            "init": t["init"], "data_dir": d["data_dir"], "lr": t["lr"],
            "tokenizer_sha256": state.get("tokenizer_sha256"),
        })  # fmt: skip
        start = time.perf_counter()
        model.eval()
        val_loss = answer_loss(model, enc["val"], chat.pad, ev["gen_batch"], autocast)
        m, preds = evaluate(model, tok, chat, rows["val"], intents, d["normalize"],
                            d["reply_langs"], ev, autocast)  # fmt: skip
        record = {"epoch": epoch, "step": step, "lr_peak": t["lr"],
                  "train_loss": sum(losses) / len(losses), "val_loss": val_loss, **m,
                  "train_seconds": round(train_seconds, 1),
                  "eval_seconds": round(time.perf_counter() - start, 1)}  # fmt: skip
        with open(metrics_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record) + "\n")
        with open(
            out / f"val_predictions_epoch{epoch}.jsonl", "w", encoding="utf-8", newline="\n"
        ) as fh:
            fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in preds)  # fmt: skip
        flat = {k: v for k, v in record.items() if not isinstance(v, dict)}
        run.wandb.log(flat, step=step)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in flat.items()}),
              flush=True)  # fmt: skip
    run.wandb.finish()
    print(f"done: {t['epochs']} epochs -> {out}")


if __name__ == "__main__":
    main()
