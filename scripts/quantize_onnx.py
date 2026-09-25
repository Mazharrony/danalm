"""Build the quantized variants of the exported step graph (Phase 7, D-034).

Usage: uv run python scripts/quantize_onnx.py --config configs/deploy/export.yaml
For every name in deploy.build, reads <deploy.out_dir>/fp32/model.onnx and writes
<deploy.out_dir>/<name>/ with model.onnx, tokenizer.json and danalm.json:
- dynamic_int8: onnxruntime's quantize_dynamic, signed per-channel INT8 weights, MatMuls with a
  constant weight only (activation x activation products in attention stay float32);
- matmul_nbits: onnxruntime's MatMulNBitsQuantizer (block-wise, e.g. 4-bit weights).
keep_output_float leaves the output projection (the MatMul with the (d_model, vocab) weight) in
float32. The input embedding table stays float32 in every variant. Nothing is scored here.
"""

import json
import shutil
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

from danalm.config import config_from_cli
from danalm.utils.files import file_sha256


def output_projection(model: onnx.ModelProto, d_model: int, vocab: int) -> str:
    """Name of the MatMul whose constant weight is the (d_model, vocab) output projection."""
    weights = {i.name for i in model.graph.initializer if tuple(i.dims) == (d_model, vocab)}
    nodes = [n.name for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in weights]
    if len(nodes) != 1:
        raise ValueError(f"expected one output-projection MatMul, found {nodes}")
    return nodes[0]


def main() -> None:
    cfg = config_from_cli(__doc__)
    p = cfg["deploy"]
    root = Path(p["out_dir"])
    src = root / "fp32" / "model.onnx"
    meta = json.loads((root / "fp32" / "danalm.json").read_text(encoding="utf-8"))
    mc = meta["model_config"]
    head = output_projection(onnx.load(str(src)), mc["d_model"], mc["vocab_size"])
    sizes = {"fp32": src.stat().st_size}
    for name in p["build"]:
        v = p["variants"][name]
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "model.onnx"
        exclude = [head] if v["keep_output_float"] else []
        if v["method"] == "dynamic_int8":
            quantize_dynamic(
                str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True,
                op_types_to_quantize=["MatMul"], nodes_to_exclude=exclude,
                extra_options={"MatMulConstBOnly": True},
            )  # fmt: skip
        elif v["method"] == "matmul_nbits":
            q = MatMulNBitsQuantizer(
                onnx.load(str(src)), bits=v["bits"], block_size=v["block_size"],
                is_symmetric=v["symmetric"], accuracy_level=v["accuracy_level"],
                nodes_to_exclude=exclude, op_types_to_quantize=("MatMul",),
            )  # fmt: skip
            q.process()
            q.model.save_model_to_file(str(dst), use_external_data_format=False)
        else:
            raise ValueError(f"unknown method {v['method']!r}")
        shutil.copyfile(root / "fp32" / "tokenizer.json", out / "tokenizer.json")
        ops = sorted({n.op_type for n in onnx.load(str(dst)).graph.node})
        vmeta = meta | {"variant": name, "quantization": v, "threshold": None,
                        "model_sha256": file_sha256(dst), "model_bytes": dst.stat().st_size,
                        "quantized_ops": [o for o in ops if "Integer" in o or "NBits" in o or "Quant" in o]}  # fmt: skip
        (out / "danalm.json").write_text(json.dumps(vmeta, indent=2) + "\n", encoding="utf-8")
        sizes[name] = dst.stat().st_size
        print(f"{name}: {dst.stat().st_size / 2**20:.1f} MB, ops {vmeta['quantized_ops']}, "
              f"output projection {'float32' if exclude else 'quantized'}", flush=True)  # fmt: skip
    print(json.dumps({k: round(v / 2**20, 1) for k, v in sizes.items()}, indent=2))


if __name__ == "__main__":
    main()
