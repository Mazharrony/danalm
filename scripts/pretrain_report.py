"""Phase 4 results page for a finished pretraining run.

Usage: uv run python scripts/pretrain_report.py --config configs/pretrain/report.yaml
Reads <run>/metrics.jsonl and <run>/model.pt and writes report.results_md and report.plot:
1. train and validation loss, learning rate, gradient norm and speed over the run;
2. validation loss per corpus source, on up to report.docs_per_source held-out documents each
   (from the cleaned validation split, which training never saw), next to each source's share
   of the train and validation tokens;
3. the final model's loss on a seeded sample of train windows and on the run's own validation
   windows (the same draw as scripts/pretrain.py), to explain the gap between the two curves;
4. sample continuations of fixed prompts (seeded), to see what the model learned.
The config is based on the run's config, so train.* and seed are the run's own.
"""

import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from danalm.config import config_from_cli  # noqa: E402
from danalm.data.shards import gather_windows, read_shard, window_index  # noqa: E402
from danalm.model.transformer import DanaLM, ModelConfig  # noqa: E402


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ⏎ ")


def main() -> None:
    cfg = config_from_cli(__doc__)
    r = cfg["report"]
    run_dir = Path(r["run_dir"])
    lines_in = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines_in if line]
    train = [x for x in rows if "train_loss" in x]
    val = [x for x in rows if "val_loss" in x]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(
        [x["tokens"] / 1e9 for x in train], [x["train_loss"] for x in train], label="train"
    )
    axes[0].plot([x["tokens"] / 1e9 for x in val], [x["val_loss"] for x in val], "o-", label="validation")  # fmt: skip
    axes[0].set(title="Loss", xlabel="tokens (billions)", ylabel="loss", ylim=(None, min(10, max(x["val_loss"] for x in val) + 0.2)))  # fmt: skip
    axes[1].plot([x["step"] for x in train], [x["lr"] for x in train])
    axes[1].set(title="Learning rate", xlabel="step")
    axes[2].plot([x["step"] for x in train], [x["grad_norm"] for x in train], lw=0.6)
    axes[2].set(title="Gradient norm (before clipping at 1.0)", xlabel="step", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(r["plot"], dpi=110)

    state = torch.load(run_dir / "model.pt", map_location=r["device"], weights_only=True)
    model = DanaLM(ModelConfig(**state["model_config"])).to(r["device"]).eval()
    model.load_state_dict(state["model"])
    tok = Tokenizer.from_file(r["tokenizer_file"])
    eos = tok.token_to_id(r["eos_token"])
    seq = state["model_config"]["max_seq_len"]

    docs: dict[str, list[str]] = defaultdict(list)
    with open(r["val_jsonl"], encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if len(docs[row["source"]]) < r["docs_per_source"]:
                docs[row["source"]].append(row["text"])
    per_source = {}
    with torch.no_grad():
        for source, texts in sorted(docs.items()):
            total, count = 0.0, 0
            for enc in tok.encode_batch(texts, add_special_tokens=False):
                ids = torch.tensor(enc.ids[: seq - 1] + [eos], device=r["device"])[None]
                if ids.shape[1] < 2:
                    continue
                logits, _ = model(ids[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    logits[0].float(), ids[0, 1:], reduction="sum"
                )
                total, count = total + loss.item(), count + ids.shape[1] - 1
            per_source[source] = (len(texts), count, total / count)

    t = cfg["train"]
    tokens_dir = Path(t["tokens_dir"])
    index = json.loads((tokens_dir / "index.json").read_text(encoding="utf-8"))
    mb, n_windows = t["micro_batch"], t["eval_batches"] * t["micro_batch"]
    window_loss, share = {}, {}
    for split in ("train", "val"):
        shards = [read_shard(tokens_dir / f["file"]) for f in index["splits"][split]["files"]]
        windows = window_index([len(s) for s in shards], t["seq_len"])
        rows = np.random.default_rng(cfg["seed"]).permutation(len(windows))[:n_windows]
        losses = []
        with torch.no_grad():
            for b in range(0, n_windows, mb):
                w = gather_windows(shards, windows[rows[b : b + mb]], t["seq_len"])
                x = torch.from_numpy(w).to(r["device"])
                losses.append(model(x[:, :-1], x[:, 1:])[1].item())
        window_loss[split] = sum(losses) / len(losses)
        counts = {s: v["tokens"] for s, v in index["splits"][split]["per_source"].items()}
        share[split] = {s: n / sum(counts.values()) for s, n in counts.items()}

    torch.manual_seed(cfg["seed"])
    samples = []
    for prompt in r["prompts"]:
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids], device=r["device"])
        out = model.generate(ids, r["sample_tokens"], eos, r["temperature"])
        samples.append((prompt, tok.decode(out[0, ids.shape[1] :].tolist())))

    last_val = val[-1]
    lines = [
        "# Phase 4 results: pretraining",
        "",
        f"Generated on {date.today().isoformat()} by `scripts/pretrain_report.py`; do not edit by"
        " hand. Setup: D-028 in [DECISIONS.md](../DECISIONS.md).",
        "",
        f"{state['steps']:,} steps, {state['tokens'] / 1e9:.3f}B tokens. Final validation loss"
        f" **{last_val['val_loss']:.3f}** (perplexity {math.exp(last_val['val_loss']):.1f}), from"
        f" {val[0]['val_loss']:.3f} at step 0. Median speed"
        f" {sorted(x['tokens_per_s'] for x in train)[len(train) // 2]:,} tokens/s.",
        "",
        f"![pretraining curves]({Path(r['plot']).name})",
        "",
        f"Validation loss per source (up to {r['docs_per_source']} held-out documents each, first"
        f" {seq:,} tokens of each), with each source's share of the train and validation tokens:",
        "",
        "| Source | Documents | Tokens | Loss | Perplexity | Share of train | Share of val |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {s} | {n:,} | {c:,} | {loss:.3f} | {math.exp(loss):.1f} |"
            f" {share['train'].get(s, 0):.2%} | {share['val'].get(s, 0):.2%} |"
            for s, (n, c, loss) in sorted(per_source.items(), key=lambda kv: kv[1][2])
        ),
        "",
        "Train vs validation: each point of the train curve is measured on a batch before the"
        " model learns from it, so both curves are losses on text the model has not seen yet. The"
        f" final model scores **{window_loss['train']:.3f}** on {n_windows:,} train windows of"
        f" {t['seq_len']:,} tokens (the first ones in the run's seeded order, so a random sample"
        " that it trained on in its first steps) and"
        f" **{window_loss['val']:.3f}** on the run's {n_windows:,} validation windows.",
        "",
        f"Sample continuations (temperature {r['temperature']}, {r['sample_tokens']} tokens, seed"
        f" {cfg['seed']}). This is a base model: it continues text, it does not answer yet.",
        "",
        "| Prompt | Continuation |",
        "|---|---|",
        *(f"| {cell(p)} | {cell(c)} |" for p, c in samples),
    ]
    Path(r["results_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
