# Phase 7 results: quantization and deployment

Generated on 2026-09-25 by `scripts/phase7_report.py`; do not edit by hand. Rules: D-034 in [DECISIONS.md](../DECISIONS.md), fixed before any quantized model was scored. The model is the D-033 model (`artifacts/checkpoints/sft-r3-second-lr3e-4/epoch_3.pt`).

## Export and variants

One ONNX step graph with a KV cache serves both the prompt and each new token. Right after the export, PyTorch and ONNX Runtime gave identical greedy answers on 32 of 32 real-dev messages; the largest logit difference after the prompt was 2.1e-05.

| Variant | Method | File | Quantized ops | Threshold (real dev, 95% target) |
|---|---|---:|---|---:|
| fp32 | float32 | 278.0 MB | – | 0.600 |
| int8 | dynamic_int8 | 100.8 MB | DynamicQuantizeLinear, MatMulInteger | 0.589 |
| int4 | matmul_nbits | 78.1 MB | MatMulNBits | 0.564 |

The input embedding table stays float32 in every variant (D-034).

## Development sets: the check and the choice

**SFT validation (957, all varieties)**

| System | Intent accuracy | Macro-F1 | Valid JSON | Reply language | Same intent as the reference | Same answer as the reference | Answered at its threshold (accuracy) |
|---|---:|---:|---:|---:|---:|---:|---|
| pytorch-fp32 | 93.0% | 92.7% | 100.0% | 99.9% | – | – | 96.3% (94.9%) |
| pytorch-fp32-cache | 93.0% | 92.7% | 100.0% | 99.9% | 100.0% | 100.0% | 96.3% (94.9%) |
| onnx-fp32 | 93.0% | 92.7% | 100.0% | 99.9% | 100.0% | 100.0% | 96.3% (94.9%) |
| onnx-int8 | 93.0% | 92.8% | 100.0% | 99.9% | 98.7% | 37.8% | 96.8% (94.8%) |
| onnx-int4 | 93.2% | 92.9% | 100.0% | 99.9% | 98.0% | 21.1% | 96.6% (94.8%) |

**real dev set (804, English)**

| System | Intent accuracy | Macro-F1 | Valid JSON | Reply language | Same intent as the reference | Same answer as the reference | Answered at its threshold (accuracy) |
|---|---:|---:|---:|---:|---:|---:|---|
| pytorch-fp32 | 94.2% | 94.3% | 99.9% | 99.9% | – | – | 97.6% (95.0%) |
| pytorch-fp32-cache | 94.2% | 94.3% | 99.9% | 99.9% | 100.0% | 100.0% | 97.6% (95.0%) |
| onnx-fp32 | 94.2% | 94.3% | 99.9% | 99.9% | 100.0% | 100.0% | 97.6% (95.0%) |
| onnx-int8 | 93.9% | 93.7% | 100.0% | 99.9% | 99.0% | 29.7% | 97.5% (95.0%) |
| onnx-int4 | 93.2% | 93.2% | 100.0% | 100.0% | 98.0% | 16.9% | 97.0% (95.0%) |

The reference is PyTorch float32 on the CPU, recomputing the whole sequence at every step (the D-029 decoding).

- Exactness (same answer as the reference on at least 99.5% of both sets): pytorch-fp32-cache: **met**; onnx-fp32: **met**.
- The D-034 rule for a quantized variant, on both sets: intent accuracy at most 1 point below onnx-fp32, valid JSON ≥ 99.0%, reply language ≥ 99.0%.
  - int8: **passes**. Intent accuracy against onnx-fp32: SFT validation +0.000 points (+0 of 957 messages; the limit is -9.57); real dev set -0.249 points (-2 of 804 messages; the limit is -8.04). Valid JSON and reply language within the limits.
  - int4: **passes**. Intent accuracy against onnx-fp32: SFT validation +0.209 points (+2 of 957 messages; the limit is -9.57); real dev set -0.995 points (-8 of 804 messages; the limit is -8.04). Valid JSON and reply language within the limits.
- **Deployed: int4**, threshold 0.564: the smallest passing variant (78.1 MB).
