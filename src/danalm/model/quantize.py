"""Quantized variants of an exported step graph (Phase 7, D-034), with onnxruntime's tools.

- dynamic_int8: quantize_dynamic, signed per-channel INT8 weights, only MatMuls with a constant
  weight (the activation x activation products of attention stay float32);
- matmul_nbits: MatMulNBitsQuantizer, block-wise low-bit weights (e.g. 4-bit, block 32).
keep_output_float leaves the output projection (the MatMul with the (d_model, vocab) weight) in
float32. The input embedding table (a Gather) stays float32 in every variant.
"""

from pathlib import Path
from typing import Any

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def output_projection(model: onnx.ModelProto, d_model: int, vocab: int) -> str:
    """Name of the MatMul whose constant weight is the (d_model, vocab) output projection."""
    weights = {i.name for i in model.graph.initializer if tuple(i.dims) == (d_model, vocab)}
    nodes = [n.name for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in weights]
    if len(nodes) != 1:
        raise ValueError(f"expected one output-projection MatMul, found {nodes}")
    return nodes[0]


def quantize(src: Path, dst: Path, spec: dict[str, Any], d_model: int, vocab: int) -> list[str]:
    """Write the variant described by spec (a deploy.variants entry) of src to dst; returns the
    quantized op types in the result."""
    head = output_projection(onnx.load(str(src)), d_model, vocab)
    exclude = [head] if spec["keep_output_float"] else []
    if spec["method"] == "dynamic_int8":
        quantize_dynamic(
            str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True,
            op_types_to_quantize=["MatMul"], nodes_to_exclude=exclude,
            extra_options={"MatMulConstBOnly": True},
        )  # fmt: skip
    elif spec["method"] == "matmul_nbits":
        q = MatMulNBitsQuantizer(
            onnx.load(str(src)), bits=spec["bits"], block_size=spec["block_size"],
            is_symmetric=spec["symmetric"], accuracy_level=spec["accuracy_level"],
            nodes_to_exclude=exclude, op_types_to_quantize=("MatMul",),
        )  # fmt: skip
        q.process()
        q.model.save_model_to_file(str(dst), use_external_data_format=False)
    else:
        raise ValueError(f"unknown method {spec['method']!r}")
    ops = {n.op_type for n in onnx.load(str(dst)).graph.node}
    return sorted(o for o in ops if "Integer" in o or "NBits" in o or "Quant" in o)
