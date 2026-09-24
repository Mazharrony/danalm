"""Pretrain DanaLM on the token shards (Phase 4, D-028).

Usage: uv run python scripts/pretrain.py --config configs/pretrain/<name>.yaml
Trains on one pass over the train shards (or train.total_tokens) in a seeded order of windows
that make every token a target exactly once. bf16 autocast, gradient accumulation, AdamW, linear
warmup then cosine decay, gradient clipping. Validation loss on a fixed set of validation windows
every train.eval_every steps. Checkpoints (model, optimizer, step) go to train.out_dir every
train.ckpt_every steps; if one is there, the run resumes from the latest. The data order depends
only on the step, so a resumed run sees exactly the data it would have seen. Metrics go to W&B
and to <out_dir>/metrics.jsonl (appended across resumes). The finished model is <out_dir>/model.pt.
Refuses to run while the teacher server is up.
"""

import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from danalm.config import config_from_cli
from danalm.data.shards import gather_windows, read_shard, window_index
from danalm.model.transformer import DanaLM, ModelConfig, flops_per_token
from danalm.teacher.server import assert_teacher_stopped
from danalm.train.loop import (
    checkpoints,
    lr_at,
    make_optimizer,
    prune_checkpoints,
    save_checkpoint,
    set_lr,
    train_step,
)
from danalm.utils.run import start_run
from danalm.utils.seed import set_seed


def main() -> None:
    cfg = config_from_cli(__doc__)
    assert_teacher_stopped(cfg["teacher"]["host"], cfg["teacher"]["port"])
    set_seed(cfg["seed"], cfg["deterministic"])
    t, device = cfg["train"], "cuda"
    out = Path(t["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    mcfg = ModelConfig(**cfg["model"])
    index = json.loads((Path(t["tokens_dir"]) / "index.json").read_text(encoding="utf-8"))
    shards = {
        split: [read_shard(Path(t["tokens_dir"]) / f["file"]) for f in info["files"]]
        for split, info in index["splits"].items()
    }
    seq, mb = t["seq_len"], t["micro_batch"]
    windows = window_index([len(s) for s in shards["train"]], seq)
    order = np.random.default_rng(cfg["seed"]).permutation(len(windows))
    per_step = mb * t["grad_accum"]
    tokens_per_step = per_step * seq
    total_steps = len(windows) // per_step
    if t["total_tokens"]:
        total_steps = min(total_steps, t["total_tokens"] // tokens_per_step)
    val_all = window_index([len(s) for s in shards["val"]], seq)
    val_rows = np.random.default_rng(cfg["seed"]).permutation(len(val_all))
    val_windows = val_all[val_rows[: t["eval_batches"] * mb]]

    model = DanaLM(mcfg).to(device)
    opt = make_optimizer(model, t["lr"], tuple(t["betas"]), t["weight_decay"], fused=True)
    step = 0
    if saved := checkpoints(out):
        state = torch.load(saved[-1], map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
        print(f"resuming from {saved[-1]} at step {step}", flush=True)
    run = start_run(cfg, job_type="pretrain")
    net = torch.compile(model) if t["compile"] else model
    fpt = flops_per_token(mcfg, seq)
    print(
        f"{total_steps:,} steps of {tokens_per_step:,} tokens; {len(windows):,} windows", flush=True
    )

    def autocast() -> torch.autocast:
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def to_device(w: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(w).pin_memory().to(device, non_blocking=True)
        return x[:, :-1], x[:, 1:]

    def micro_batches(step: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for m in range(t["grad_accum"]):
            rows = order[step * per_step + m * mb : step * per_step + (m + 1) * mb]
            yield to_device(gather_windows(shards["train"], windows[rows], seq))

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        losses = []
        for b in range(t["eval_batches"]):
            x, y = to_device(gather_windows(shards["val"], val_windows[b * mb : (b + 1) * mb], seq))
            with autocast():
                losses.append(net(x, y)[1])
        model.train()
        return torch.stack(losses).mean().item()

    def log(record: dict) -> None:
        with open(out / "metrics.jsonl", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record) + "\n")
        run.wandb.log(record, step=record["step"])

    if step == 0:
        log({"step": 0, "tokens": 0, "val_loss": val_loss()})
    losses: list[float] = []
    start, tokens_timed = time.perf_counter(), 0
    while step < total_steps:
        lr = lr_at(step, total_steps, t["warmup_steps"], t["lr"], t["min_lr"])
        set_lr(opt, lr)
        loss, grad_norm = train_step(net, opt, micro_batches(step), t["grad_clip"], autocast)
        step += 1
        losses.append(loss)
        tokens_timed += tokens_per_step
        record = {}
        if step % t["log_every"] == 0 or step == total_steps:
            tps = tokens_timed / (time.perf_counter() - start)
            record = {"step": step, "tokens": step * tokens_per_step, "lr": lr,
                      "train_loss": sum(losses) / len(losses), "grad_norm": grad_norm,
                      "tokens_per_s": round(tps), "mfu": round(tps * fpt / (t["peak_tflops"] * 1e12), 3)}  # fmt: skip
            losses = []
        if step % t["eval_every"] == 0 or step == total_steps:
            record = {"step": step, "tokens": step * tokens_per_step, **record, "val_loss": val_loss()}  # fmt: skip
        if record:
            log(record)
            print(json.dumps(record), flush=True)
        stopping = bool(t["stop_after_steps"]) and t["stop_after_steps"] <= step < total_steps
        if step % t["ckpt_every"] == 0 or step == total_steps or stopping:
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}
            save_checkpoint(out / f"step_{step}.pt", state)
            prune_checkpoints(out, t["keep_ckpts"])
        if record or step % t["ckpt_every"] == 0:  # time only training, not evals or saves
            start, tokens_timed = time.perf_counter(), 0
        if stopping:
            print(f"stopped at step {step} (stop_after_steps), checkpoint saved", flush=True)
            break
    if step == total_steps:
        final = {"model": model.state_dict(), "model_config": cfg["model"], "steps": step,
                 "tokens": step * tokens_per_step, "tokenizer": index["tokenizer"],
                 "tokenizer_sha256": index["tokenizer_sha256"]}  # fmt: skip
        save_checkpoint(out / "model.pt", final)
        print(f"done: {step:,} steps, {step * tokens_per_step:,} tokens -> {out / 'model.pt'}")
    run.wandb.finish()


if __name__ == "__main__":
    main()
