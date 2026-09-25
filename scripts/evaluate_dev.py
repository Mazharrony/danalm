"""Score one SFT checkpoint on a development JSONL (message/intent/variety) with the D-029 metrics.

Usage: uv run python scripts/evaluate_dev.py --config configs/sft/report_real.yaml
Scores report.baseline_checkpoint (the Phase 5 model) on report.real_dev, so the real-message round
(D-033) can be compared with it on the same messages. Writes report.baseline_out (metrics and
predictions). The human test set is not used. Refuses to run while the teacher server is up.
"""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from danalm.config import config_from_cli
from danalm.data.intents import load_intents
from danalm.model.transformer import DanaLM, ModelConfig
from danalm.sft.data import ChatTokens
from danalm.sft.evaluate import evaluate
from danalm.teacher.server import assert_teacher_stopped


def main() -> None:
    cfg = config_from_cli(__doc__)
    assert_teacher_stopped(cfg["teacher"]["host"], cfg["teacher"]["port"])
    r, d, ev = cfg["report"], cfg["sft_data"], cfg["eval"]
    with open(r["real_dev"], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    state = torch.load(r["baseline_checkpoint"], map_location="cpu", weights_only=True)
    model = DanaLM(ModelConfig(**state["model_config"])).to("cuda").eval()
    model.load_state_dict(state["model"])
    tok = Tokenizer.from_file(d["tokenizer_file"])
    chat = ChatTokens.from_tokenizer(tok, d["special"])
    intents = [i["name"] for i in load_intents(d["intents_file"])]

    def autocast() -> torch.autocast:
        return torch.autocast("cuda", dtype=torch.bfloat16)

    metrics, preds = evaluate(model, tok, chat, rows, intents, d["normalize"], d["reply_langs"],
                              ev, autocast)  # fmt: skip
    out = Path(r["baseline_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": r["baseline_checkpoint"], "metrics": metrics,
                               "predictions": preds}, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")  # fmt: skip
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
