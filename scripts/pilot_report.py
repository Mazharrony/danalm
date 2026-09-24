"""Phase 4 pilot report: loss curves, the learning-rate choice and the resume check.

Usage: uv run python scripts/pilot_report.py --config configs/pretrain/pilot_report.yaml
Reads <checkpoints_dir>/<run>/metrics.jsonl of every pilot run, plots train and validation loss,
applies the learning-rate rule fixed in the config before the runs, compares the resumed run
with the straight one, estimates the full run's time from the measured speed, and writes the
plot and report.results_md.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from danalm.config import config_from_cli, load_config  # noqa: E402
from danalm.data.shards import window_index  # noqa: E402


def metrics(folder: Path) -> list[dict[str, Any]]:
    path = folder / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the pilot first")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    points = [(r["tokens"] / 1e6, r[key]) for r in rows if key in r]
    return [p[0] for p in points], [p[1] for p in points]


def spike(rows: list[dict], warmup_steps: int) -> float:
    """Largest rise of the logged train loss between consecutive logs after warmup."""
    losses = [r["train_loss"] for r in rows if "train_loss" in r and r["step"] > warmup_steps]
    return max((b - a for a, b in zip(losses, losses[1:], strict=False)), default=0.0)


def main() -> None:
    cfg = config_from_cli(__doc__)
    r = cfg["report"]
    pilot = load_config(r["pilot_config"])["train"]
    full = load_config(r["full_config"])["train"]
    root = Path(r["checkpoints_dir"])
    runs = {lr: metrics(root / name) for lr, name in r["runs"].items()}
    resumed = metrics(root / r["resume_run"])
    straight = runs[r["resume_lr"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for lr, rows in runs.items():
        axes[0].plot(*series(rows, "train_loss"), label=f"peak LR {lr}")
        axes[1].plot(*series(rows, "val_loss"), marker="o", label=f"peak LR {lr}")
    axes[2].plot(*series(straight, "train_loss"), label=f"straight (LR {r['resume_lr']})")
    axes[2].plot(*series(resumed, "train_loss"), "--", label="stopped at step 24, resumed")
    titles = ("Train loss", "Validation loss", "Resume check (train loss)")
    for ax, title in zip(axes, titles, strict=True):
        ax.set(title=title, xlabel="tokens (millions)", ylabel="loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(r["plot"], dpi=110)

    rule = r["rule"]
    final = {lr: [x for x in rows if "val_loss" in x][-1]["val_loss"] for lr, rows in runs.items()}
    spikes = {lr: spike(rows, pilot["warmup_steps"]) for lr, rows in runs.items()}
    stable = [lr for lr in runs if spikes[lr] <= rule["max_spike"]]
    best = min(stable, key=lambda lr: final[lr])
    # prefer the lowest learning rate that is within tie_margin of the best (safer when 100x longer)
    chosen = min((lr for lr in stable if final[lr] - final[best] <= rule["tie_margin"]), key=float)
    both = {x["step"]: x["train_loss"] for x in straight if "train_loss" in x}
    diffs = [abs(both[x["step"]] - x["train_loss"]) for x in resumed if x.get("step") in both and "train_loss" in x]  # fmt: skip
    speed = [x["tokens_per_s"] for x in straight if "tokens_per_s" in x][
        1:
    ]  # skip the compile step
    tps = sorted(speed)[len(speed) // 2]
    index = json.loads((Path(full["tokens_dir"]) / "index.json").read_text(encoding="utf-8"))
    lengths = [f["tokens"] for f in index["splits"]["train"]["files"]]
    steps_full = len(window_index(lengths, full["seq_len"])) // (
        full["micro_batch"] * full["grad_accum"]
    )
    hours = steps_full * full["micro_batch"] * full["grad_accum"] * full["seq_len"] / tps / 3600

    lines = [
        "# Phase 4 results: pretraining pilot",
        "",
        f"Generated on {date.today().isoformat()} by `scripts/pilot_report.py` from the pilot runs;"
        " do not edit by hand. Decision: D-028 in [DECISIONS.md](../DECISIONS.md).",
        "",
        f"Each pilot trains the chosen 62.1M model (D-026) on ~1% of the full run's tokens"
        f" ({pilot['total_tokens'] / 1e6:.1f}M), with the full run's settings and the schedule"
        f" compressed to the pilot's length (warmup {pilot['warmup_steps']} steps, cosine decay to"
        " 10% of the peak).",
        "",
        f"![pilot curves]({Path(r['plot']).name})",
        "",
        "| Peak LR | Final val loss | Largest train-loss rise after warmup | Max grad norm |",
        "|---|---:|---:|---:|",
        *(
            f"| {lr} | {final[lr]:.3f} | {spikes[lr]:+.3f} |"
            f" {max(x['grad_norm'] for x in rows if 'grad_norm' in x):.2f} |"
            for lr, rows in runs.items()
        ),
        "",
        f"**Rule (fixed before the runs, `configs/pretrain/pilot_report.yaml`):** among the learning"
        f" rates whose train loss never rises by more than {rule['max_spike']} after warmup, take"
        f" the lowest one whose final validation loss is within {rule['tie_margin']} of the best."
        f" **Chosen: {chosen}.**",
        "",
        f"**Resume check:** the run stopped at step 24 and resumed from its checkpoint; over the"
        f" {len(diffs)} steps logged by both, its train loss differs from the straight run's by at"
        f" most {max(diffs):.3f} (GPU kernels are not bit-exact; on the CPU the resume test is"
        " bit-identical).",
        "",
        f"**Speed:** {tps:,} tokens/s (median, torch.compile). The full run ({steps_full:,} steps)"
        f" would take about {hours:.1f} h plus evaluations and checkpoints.",
    ]
    Path(r["results_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
