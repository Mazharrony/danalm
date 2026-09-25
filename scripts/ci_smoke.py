"""CI smoke evaluation (Phase 7, D-034): the whole deployment path on a tiny model, on the CPU.

Usage: uv run python scripts/ci_smoke.py
Trains a tiny tokenizer and model until they answer a handful of examples, exports the ONNX step
graph, builds the INT8 and INT4 variants, and evaluates each through danalm.eval.cached and the
service's predictor. It fails if ONNX float32 answers differ from PyTorch, or if float32 or INT8
answers fewer than 90% of the examples with valid JSON and the right intent. The real model is
not in the repository (D-034), so this checks the machinery, not DanaLM's accuracy. ~1 minute.
"""

import contextlib
import json
import shutil
import tempfile
from pathlib import Path

import torch

from danalm.eval.cached import evaluate_cached
from danalm.infer.onnx import OnnxStep
from danalm.infer.predictor import Predictor
from danalm.model.export import export_onnx
from danalm.model.quantize import quantize
from danalm.model.transformer import DanaLM, ModelConfig
from danalm.sft.data import collate, example_ids
from danalm.sft.evaluate import evaluate
from danalm.sft.format import ChatTokens
from danalm.tokenizer.bpe import TokenizerConfig, train
from danalm.utils.seed import set_seed

EXAMPLES = [  # (message, intent, reply): written for this check, not from any dataset
    (
        "my card is not working at the atm",
        "card_not_working",
        "Please check that the card is active in the app.",
    ),
    ("where is my order", "order_status", "Please share your order number so we can check it."),
    ("i want to cancel my order", "cancel_order", "Please share your order number to cancel it."),
    ("وين طلبي؟ صار له ساعتين", "order_status", "ممكن ترسل رقم الطلب عشان نتابعه؟"),
    ("البطاقة ما تشتغل", "card_not_working", "تأكد إن البطاقة مفعلة في التطبيق."),
    ("3andi mushkila fil card", "card_not_working", "تأكد إن البطاقة مفعلة في التطبيق."),
    ("hello", "other", "Hello! How can I help you today?"),
    ("i want to talk to a person", "handoff_to_human", "I will connect you with a team member."),
]
INTENTS = ["cancel_order", "card_not_working", "handoff_to_human", "order_status", "other"]
SPECIAL = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "eos": "<|endoftext|>",
    "pad": "<|pad|>",
}
NORM = {"strip_diacritics": True, "unify_alef": False}
LANGS = {"english": ["en"], "gulf_arabic": ["ar"], "arabizi": ["ar"]}
EV = {"max_new_tokens": 48, "gen_batch": 4, "score_batch": 4, "target_accuracy": 0.95}
VARIANTS = {
    "int8": {"method": "dynamic_int8", "keep_output_float": False},
    "int4": {"method": "matmul_nbits", "bits": 4, "block_size": 32, "symmetric": True,
             "accuracy_level": 4, "keep_output_float": False},
}  # fmt: skip


def target(intent: str, reply: str) -> str:
    return json.dumps({"intent": intent, "reply": reply}, ensure_ascii=False)


def main() -> None:
    set_seed(0, deterministic=True)
    torch.set_num_threads(2)  # CI runners are small; the result does not depend on it
    rows = [{"message": m, "intent": i, "variety": v}
            for (m, i, _), v in zip(EXAMPLES, ["english"] * 3 + ["gulf_arabic"] * 2 + ["arabizi"]
                                    + ["english"] * 2, strict=True)]  # fmt: skip
    corpus = [m for m, _, _ in EXAMPLES] + [target(i, r) for _, i, r in EXAMPLES]
    tok = train(TokenizerConfig(
        vocab_size=512, min_frequency=1, pretokenizer="standard", eos_token="<|endoftext|>",
        pad_token="<|pad|>", extra_special_tokens=["<|user|>", "<|assistant|>"], reserved_tokens=2,
        placeholder_tokens=["<PHONE>"], corpus="unused", out_dir="unused",
    ), corpus * 20)  # fmt: skip
    chat = ChatTokens.from_tokenizer(tok, SPECIAL)
    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), d_model=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, ffn_hidden=128, max_seq_len=128, rope_theta=10000.0,
                      norm_eps=1e-5, init_std=0.02, embed_init_std=1 / 64)  # fmt: skip
    model = DanaLM(cfg)
    x, y = collate(
        [example_ids(tok, chat, m, target(i, r), NORM) for m, i, r in EXAMPLES], chat.pad
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(400):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    print(f"tiny model trained: final loss {loss.item():.4f}")
    ref_m, ref_p = evaluate(
        model, tok, chat, rows, INTENTS, NORM, LANGS, EV, contextlib.nullcontext
    )
    report = {"pytorch-fp32": ref_m}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "fp32").mkdir()
        export_onnx(model, root / "fp32" / "model.onnx", opset=18)
        for name, spec in VARIANTS.items():
            (root / name).mkdir()
            quantize(root / "fp32" / "model.onnx", root / name / "model.onnx", spec, cfg.d_model,
                     cfg.vocab_size)  # fmt: skip
        failures = []
        for name in ["fp32", *VARIANTS]:
            step = OnnxStep(root / name / "model.onnx", threads=2)
            m, p = evaluate_cached(
                step, step.empty, tok, chat, rows, INTENTS, NORM, LANGS, EV, 4, 4
            )
            report[f"onnx-{name}"] = m
            ok = sum(x["pred_intent"] == x["intent"] for x in p) / len(p)
            if name == "fp32" and [x["output"] for x in p] != [x["output"] for x in ref_p]:
                failures.append("onnx-fp32 answers differ from PyTorch")
            if name in ("fp32", "int8") and (m["valid_json"] < 0.9 or ok < 0.9):
                failures.append(f"onnx-{name}: valid JSON {m['valid_json']:.2f}, right {ok:.2f}")
            # the service path: a model directory and the predictor
            tok.save(str(root / name / "tokenizer.json"))
            meta = {"variant": name, "model_file": "model.onnx", "tokenizer_file": "tokenizer.json",
                    "threshold": 0.5, "intents": INTENTS, "max_new_tokens": EV["max_new_tokens"],
                    "model_config": {"max_seq_len": cfg.max_seq_len}, "special": SPECIAL,
                    "normalize": NORM, "reply_langs": LANGS}  # fmt: skip
            (root / name / "danalm.json").write_text(json.dumps(meta), encoding="utf-8")
            out = Predictor(root / name, threads=2).predict("my card is not working at the atm")
            print(
                f"onnx-{name:<5} predictor: {out['intent']} ({out['confidence']:.2f}, {out['route']})"
            )
        shutil.rmtree(root, ignore_errors=True)
    for name, m in report.items():
        print(f"{name:<13} valid JSON {m['valid_json']:.2f}  intent accuracy {m['intent_accuracy']:.2f}"
              f"  reply language {m['reply_lang']:.2f}")  # fmt: skip
    if failures:
        raise SystemExit("smoke evaluation failed: " + "; ".join(failures))
    print("smoke evaluation passed")


if __name__ == "__main__":
    main()
