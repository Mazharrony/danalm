"""Reference results for the browser port (web/danalm.js), from the Python inference path.

Usage: uv run python scripts/web_parity.py --config configs/deploy/phase7.yaml
Writes web_parity.out: for a seeded sample of development messages (never the test set) and a
set of crafted strings (PII in every form, Arabic-Indic digits, repeats, unusual spaces), the
normalized text, the detected language, the token ids and the decoded text; and, for
web_parity.predict of the messages, the deployed variant's full prediction with its generated
token ids. web/parity.html runs the port on the same inputs and lists every difference.
"""

import json
import random
from pathlib import Path

from tokenizers import Tokenizer

from danalm.config import config_from_cli
from danalm.infer.decode import greedy_from, prefill
from danalm.infer.predictor import Predictor
from danalm.sft.format import prompt_ids
from danalm.text import detect_lang, normalize

CRAFTED = [
    "call me on 0501234567 or +971 50 123 4567 or 04 123 4567",
    "my emirates id is 784-1990-1234567-1 and iban AE070331234567890123456",
    "card 4111 1111 1111 1111, amex 3782 822463 10005, ref 1234567890123",
    "email me at test.user+x@mail.example.com or see https://example.com/a?b=c www.x.ae",
    "رقمي ٠٥٠١٢٣٤٥٦٧ والبطاقة ۴۱۱۱۱۱۱۱۱۱۱۱۱۱۱۱",
    "ههههههههههه sooooooo gooood!!!!!!",
    "وين الطلب صار له　ساعتين",
    "مَرْحَبًا بِكُم، ـــ أريد إلغاء الطلب",
    "3andi mushkila fil card, laish ma yishtaghil?",
    "I'll call you at 5pm about the 2FA code for 10GB",
    "ABC 123 it's  can't   won't   they're",
    "emoji test 🙂🙂🙂🙂🙂🙂 and café naïve",
    "   leading and trailing spaces   ",
    "",
]


def main() -> None:
    cfg = config_from_cli(__doc__)
    w, d = cfg["web_parity"], cfg["sft_data"]
    rng = random.Random(w["seed"])
    messages = []
    for path in cfg["phase7"]["dev_sets"].values():
        rows = []
        for f in [path] if isinstance(path, str) else path:  # one file or a list of files
            with open(f, encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh]
        messages += [r["message"] for r in rng.sample(rows, w["per_set"])]
    model_dir = Path(cfg["deploy"]["out_dir"]) / w["variant"]
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    texts = []
    for text in messages + CRAFTED:
        norm = normalize(text, **d["normalize"])
        ids = tok.encode(norm, add_special_tokens=False).ids
        texts.append({"text": text, "normalized": norm, "lang": detect_lang(norm), "ids": ids,
                      "decoded": tok.decode(ids, skip_special_tokens=False)})  # fmt: skip
    pred = Predictor(model_dir, threads=4)
    predictions = []
    for text in messages[: w["predict"]]:
        out = pred.predict(text)
        prompt = prompt_ids(pred.tok, pred.chat, text, d["normalize"])
        last, past = prefill(pred.step, [prompt], pred.step.empty(1))
        [(ids, _)] = greedy_from(pred.step, last, past, len(prompt), pred.chat.eos, pred.meta["max_new_tokens"])  # fmt: skip
        out.pop("latency_ms")
        predictions.append({"message": text, "generated_ids": ids, **out})
    out_path = Path(w["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # model_dir: where web/parity.html loads the same model from (relative to the repository root)
    out_path.write_text(json.dumps({"variant": w["variant"], "model_dir": model_dir.as_posix(),
                                    "texts": texts, "predictions": predictions},
                                   ensure_ascii=False), encoding="utf-8")  # fmt: skip
    print(f"{out_path}: {len(texts)} texts, {len(predictions)} predictions")


if __name__ == "__main__":
    main()
