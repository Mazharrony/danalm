"""The brief's sanity checks for a model size, before any long run (Phase 3).

Usage: uv run python scripts/model_sanity.py --config configs/model/<size>.yaml
1. Initial loss on validation windows of the pretraining shards, against ln(vocab_size).
2. Overfit one small batch: the loss must fall below sanity.overfit.target_loss.
3. Speed: training tokens per second (bf16 autocast, AdamW) at the speed batch size, peak VRAM,
   and MFU against the measured bf16 peak; plus the time one pass over the train shards would
   take at that speed.
Refuses to run while the teacher server is up (teacher and student never share the GPU).
Writes sanity.json to the run folder.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from danalm.config import config_from_cli
from danalm.data.shards import read_shard, sample_windows
from danalm.model.transformer import DanaLM, ModelConfig, count_params, flops_per_token
from danalm.teacher.server import assert_teacher_stopped
from danalm.utils.run import start_run
from danalm.utils.seed import set_seed


def batch(shards: list, n: int, seq_len: int, rng: np.random.Generator, device: str) -> tuple:
    w = torch.from_numpy(sample_windows(shards, n, seq_len, rng)).to(device, non_blocking=True)
    return w[:, :-1], w[:, 1:]


def main() -> None:
    cfg = config_from_cli(__doc__)
    assert_teacher_stopped(cfg["teacher"]["host"], cfg["teacher"]["port"])
    set_seed(cfg["seed"], cfg["deterministic"])
    run = start_run(cfg, job_type="model-sanity")
    s, device = cfg["sanity"], "cuda"
    mcfg = ModelConfig(**cfg["model"])
    index = json.loads((Path(s["tokens_dir"]) / "index.json").read_text(encoding="utf-8"))
    shards = {
        split: [read_shard(f"{s['tokens_dir']}/{f['file']}") for f in info["files"]]
        for split, info in index["splits"].items()
    }
    rng = np.random.default_rng(cfg["seed"])
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    # 1. initial loss
    model = DanaLM(mcfg).to(device).eval()
    result: dict = {"params": count_params(model), "ln_vocab": math.log(mcfg.vocab_size)}
    with torch.no_grad(), autocast:
        losses = [
            model(*batch(shards["val"], 8, s["speed"]["seq_len"], rng, device))[1].item()
            for _ in range(s["init_loss_batches"])
        ]
    result["initial_loss"] = sum(losses) / len(losses)
    print(f"initial loss {result['initial_loss']:.3f} (ln vocab = {result['ln_vocab']:.3f})")

    # 2. overfit one batch
    o = s["overfit"]
    model = DanaLM(mcfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"])
    x, y = batch(shards["val"], o["batch_size"], o["seq_len"], rng, device)
    curve = []
    for step in range(o["steps"]):
        with autocast:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(loss.item())
        if step % 50 == 0:
            run.wandb.log({"overfit/loss": curve[-1]}, step=step)
    result["overfit"] = {"first": curve[0], "last": curve[-1], "passed": curve[-1] < o["target_loss"]}  # fmt: skip
    print(f"overfit one batch: {curve[0]:.3f} -> {curve[-1]:.4f} in {o['steps']} steps")

    # 3. speed and memory
    sp = s["speed"]
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = DanaLM(mcfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=sp["lr"], fused=True)
    for step in range(sp["warmup_steps"] + sp["steps"]):
        if step == sp["warmup_steps"]:
            torch.cuda.synchronize()
            start = time.perf_counter()
        x, y = batch(shards["train"], sp["batch_size"], sp["seq_len"], rng, device)
        with autocast:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    tokens_per_s = sp["steps"] * sp["batch_size"] * sp["seq_len"] / seconds
    fpt = flops_per_token(mcfg, sp["seq_len"])
    train_tokens = index["splits"]["train"]["tokens_with_eos"]
    result["speed"] = {
        "tokens_per_s": round(tokens_per_s),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "mfu": round(tokens_per_s * fpt / (s["peak_tflops"] * 1e12), 3),
        "flops_per_token": fpt,
        "hours_per_train_pass": round(train_tokens / tokens_per_s / 3600, 2),
    }
    print(json.dumps(result, indent=2))
    (run.dir / "sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    run.wandb.summary.update(result)
    run.wandb.finish()


if __name__ == "__main__":
    main()
