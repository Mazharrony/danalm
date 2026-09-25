"""Build the quantized variants of the exported step graph (Phase 7, D-034).

Usage: uv run python scripts/quantize_onnx.py --config configs/deploy/export.yaml
For every name in deploy.build, reads <deploy.out_dir>/fp32/model.onnx and writes
<deploy.out_dir>/<name>/ with model.onnx, tokenizer.json and danalm.json. The methods are in
danalm.model.quantize (dynamic INT8; block-wise MatMulNBits; optionally the output projection
in float32). Nothing is scored here.
"""

import json
import shutil
from pathlib import Path

from danalm.config import config_from_cli
from danalm.model.quantize import quantize
from danalm.utils.files import file_sha256


def main() -> None:
    cfg = config_from_cli(__doc__)
    p = cfg["deploy"]
    root = Path(p["out_dir"])
    src = root / "fp32" / "model.onnx"
    meta = json.loads((root / "fp32" / "danalm.json").read_text(encoding="utf-8"))
    mc = meta["model_config"]
    sizes = {"fp32": src.stat().st_size}
    for name in p["build"]:
        spec = p["variants"][name]
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "model.onnx"
        ops = quantize(src, dst, spec, mc["d_model"], mc["vocab_size"])
        shutil.copyfile(root / "fp32" / "tokenizer.json", out / "tokenizer.json")
        vmeta = meta | {"variant": name, "quantization": spec, "threshold": None,
                        "model_sha256": file_sha256(dst), "model_bytes": dst.stat().st_size,
                        "quantized_ops": ops}  # fmt: skip
        (out / "danalm.json").write_text(json.dumps(vmeta, indent=2) + "\n", encoding="utf-8")
        sizes[name] = dst.stat().st_size
        head = "float32" if spec["keep_output_float"] else "quantized"
        print(f"{name}: {sizes[name] / 2**20:.1f} MB, ops {ops}, output projection {head}", flush=True)  # fmt: skip
    print(json.dumps({k: round(v / 2**20, 1) for k, v in sizes.items()}, indent=2))


if __name__ == "__main__":
    main()
