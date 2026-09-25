"""Development replies of two models on the same messages, for the D-032 reply judge (D-037).

Usage: uv run python scripts/faithfulness_sample.py --config configs/sft/report_qtypes.yaml
Takes a seeded sample of report.faithfulness.per_set messages from the question-type dev set and
as many from the real dev set, and writes danalm_predictions.jsonl (id, message, intent, variety,
valid, reply, pred_intent: the format scripts/evaluate_teacher.py reads) twice:
- report.faithfulness.before_dir: the earlier model (report.previous_selected, with its
  question-type-dev answers from report.baseline_out);
- report.faithfulness.after_dir: the chosen model (report.selected).
Nothing is scored here and the human test set is not used.
"""

import json
import random
from pathlib import Path
from typing import Any

from danalm.config import config_from_cli


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run_preds(ckpt: Path, selected: dict[str, Any], split: str) -> list[dict[str, Any]]:
    run = Path(selected["checkpoint"]).parent.name
    return read_jsonl(ckpt / run / f"dev_{split}_predictions_epoch{selected['epoch']}.jsonl")


def main() -> None:
    cfg = config_from_cli(__doc__)
    r = cfg["report"]
    f = r["faithfulness"]
    ckpt = Path(r["checkpoints_dir"])
    chosen = json.loads(Path(r["selected"]).read_text(encoding="utf-8"))
    earlier = json.loads(Path(r["previous_selected"]).read_text(encoding="utf-8"))
    base_q = json.loads(Path(r["baseline_out"]).read_text(encoding="utf-8"))["predictions"]
    models = {
        f["before_dir"]: {"qtype": base_q, "real": run_preds(ckpt, earlier, "real")},
        f["after_dir"]: {"qtype": run_preds(ckpt, chosen, "qtype"), "real": run_preds(ckpt, chosen, "real")},
    }  # fmt: skip
    rng = random.Random(f["seed"])
    picks = {}
    for split in ("qtype", "real"):
        a, b = (m[split] for m in models.values())
        if [x["message"] for x in a] != [x["message"] for x in b]:
            raise ValueError(f"the two models' {split} predictions are not on the same messages")
        picks[split] = sorted(rng.sample(range(len(a)), f["per_set"]))
    for out_dir, by_split in models.items():
        rows = [{"id": f"{split[0]}{i:04d}", **{k: by_split[split][i][k] for k in
                 ("message", "intent", "variety", "valid", "reply", "pred_intent")}}
                for split in ("qtype", "real") for i in picks[split]]  # fmt: skip
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "danalm_predictions.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        print(f"{out}: {len(rows)} replies ({sum(x['valid'] for x in rows)} valid)")


if __name__ == "__main__":
    main()
