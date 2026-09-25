"""Export the selected SFT model to an ONNX step graph with a KV cache (Phase 7, D-034).

Usage: uv run python scripts/export_onnx.py --config configs/deploy/export.yaml
Writes <deploy.out_dir>/fp32/: model.onnx, tokenizer.json and danalm.json (the intents, special
tokens, normalization and answer length; the threshold stays empty until
scripts/evaluate_variants.py fixes it on the real dev set). Then checks PyTorch against ONNX
Runtime on deploy.parity.n real-dev messages: the largest logit difference after the prompt and
the share of identical greedy answers. Runs on the CPU in float32.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from danalm.config import config_from_cli
from danalm.data.intents import load_intents
from danalm.infer.decode import greedy_from, prefill
from danalm.infer.onnx import OnnxStep
from danalm.model.export import empty_cache, export_onnx, numpy_step
from danalm.model.transformer import DanaLM, ModelConfig
from danalm.sft.format import ChatTokens, prompt_ids
from danalm.utils.files import file_sha256
from danalm.utils.run import write_provenance


def manifest(cfg: dict[str, Any], variant: str, checkpoint: str, model_file: Path) -> dict:
    d = cfg["sft_data"]
    return {
        "variant": variant,
        "model_file": model_file.name,
        "model_sha256": file_sha256(model_file),
        "model_bytes": model_file.stat().st_size,
        "tokenizer_file": "tokenizer.json",
        "special": d["special"],
        "intents": [i["name"] for i in load_intents(d["intents_file"])],
        "normalize": d["normalize"],
        "reply_langs": d["reply_langs"],
        "max_new_tokens": cfg["eval"]["max_new_tokens"],
        "threshold": None,
        "source_checkpoint": checkpoint,
    }


def main() -> None:
    cfg = config_from_cli(__doc__)
    p, d = cfg["deploy"], cfg["sft_data"]
    root = Path(p["out_dir"])
    out = root / "fp32"
    out.mkdir(parents=True, exist_ok=True)
    write_provenance(root, cfg)
    checkpoint = json.loads(Path(p["selected"]).read_text(encoding="utf-8"))["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DanaLM(ModelConfig(**state["model_config"])).eval()
    model.load_state_dict(state["model"])
    export_onnx(model, out / "model.onnx", p["opset"])
    shutil.copyfile(d["tokenizer_file"], out / "tokenizer.json")
    meta = manifest(cfg, "fp32", checkpoint, out / "model.onnx")
    meta["model_config"] = state["model_config"]
    (out / "danalm.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # parity: the same prompts through PyTorch and ONNX Runtime, float32
    tok = Tokenizer.from_file(d["tokenizer_file"])
    chat = ChatTokens.from_tokenizer(tok, d["special"])
    with open(p["parity"]["messages"], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh][: p["parity"]["n"]]
    torch.set_num_threads(p["parity"]["threads"])
    backends = {"pytorch": (numpy_step(model), lambda b: empty_cache(model, b))}
    ort_step = OnnxStep(out / "model.onnx", p["parity"]["threads"])
    backends["onnx"] = (ort_step, ort_step.empty)
    diffs, same = [], 0
    for r in rows:
        prompt = prompt_ids(tok, chat, r["message"], d["normalize"])
        answers, lasts = {}, {}
        for name, (step, empty) in backends.items():
            last, past = prefill(step, [prompt], empty(1))
            lasts[name] = last
            answers[name] = greedy_from(
                step, last, past, len(prompt), chat.eos, cfg["eval"]["max_new_tokens"]
            )
        diffs.append(float(np.abs(lasts["pytorch"] - lasts["onnx"]).max()))
        same += answers["pytorch"] == answers["onnx"]
    report = {
        "model_mb": round(meta["model_bytes"] / 2**20, 1),
        "parity_messages": len(rows),
        "max_logit_diff_after_prompt": max(diffs),
        "identical_greedy_answers": same,
    }
    (root / "export_parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
