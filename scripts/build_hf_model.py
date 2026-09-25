"""Assemble the Hugging Face model repository folder (Phase 8).

Usage: uv run python scripts/build_hf_model.py --config configs/deploy/phase7.yaml
Writes hf_model.dir: README.md (docs/MODEL_CARD.md), one folder per variant in hf_model.variants
(model.onnx, tokenizer.json, danalm.json), and the torch-free danalm inference modules, so that
`from danalm.infer.predictor import Predictor; Predictor("int4", threads=4)` works from that
folder. Publishing it is the owner's step, e.g.
  hf upload <user>/danalm artifacts/hf-model . --repo-type model
Nothing is uploaded here.
"""

import shutil
from pathlib import Path

from danalm.config import config_from_cli

MODULES = [  # the torch-free inference modules (as in the Space)
    "src/danalm/__init__.py", "src/danalm/text.py", "src/danalm/sft/__init__.py",
    "src/danalm/sft/format.py", "src/danalm/infer",
]  # fmt: skip


def main() -> None:
    cfg = config_from_cli(__doc__)
    h = cfg["hf_model"]
    out = Path(h["dir"])
    if out.exists():  # a previous build of this script only: never an unrelated folder
        if (
            not (out / "README.md").exists()
            or not (out / h["variants"][0] / "danalm.json").exists()
        ):
            raise SystemExit(f"{out} exists and is not a model build; choose another hf_model.dir")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(h["model_card"], out / "README.md")
    ignore = shutil.ignore_patterns("__pycache__")
    for v in h["variants"]:
        shutil.copytree(Path(cfg["deploy"]["out_dir"]) / v, out / v, ignore=ignore)
    for m in MODULES:
        src, dst = Path(m), out / Path(m).relative_to("src")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=ignore)
        else:
            shutil.copyfile(src, dst)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: {', '.join(h['variants'])}, {size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
