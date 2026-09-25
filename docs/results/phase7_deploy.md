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

## Latency and memory

Batch 1 on the CPU (Intel64 Family 6 Model 183 Stepping 1, GenuineIntel), 50 messages from each development set (seed 42), after warm-up; each system in its own process, with Windows power throttling turned off for that process (D-034). "With confidence" adds the 21-intent scores that decide the route; it is timed on its own.

| System | Threads | Answer: median (p95) | With confidence: median (p95) | Answer tokens/s | Peak memory | Load | File |
|---|---:|---:|---:|---:|---:|---:|---:|
| pytorch-fp32 | 4 | 0.788 s (1.101) | 0.972 s (1.300) | 56 | 1190 MB | 1.5 s | 237 MB |
| pytorch-fp32 | 1 | 2.115 s (3.060) | 2.765 s (3.799) | 21 | 1178 MB | 1.6 s | 237 MB |
| pytorch-fp32-cache | 4 | 0.350 s (0.423) | 0.443 s (0.509) | 129 | 1052 MB | 1.5 s | 237 MB |
| pytorch-fp32-cache | 1 | 0.518 s (0.623) | 0.777 s (0.884) | 87 | 1048 MB | 1.5 s | 237 MB |
| onnx-fp32 | 4 | 0.284 s (0.342) | 0.385 s (0.439) | 159 | 435 MB | 0.5 s | 278 MB |
| onnx-fp32 | 1 | 0.484 s (0.602) | 0.734 s (0.839) | 93 | 433 MB | 0.4 s | 278 MB |
| onnx-int8 | 4 | 0.128 s (0.158) | 0.188 s (0.221) | 349 | 260 MB | 0.3 s | 101 MB |
| onnx-int8 | 1 | 0.149 s (0.181) | 0.246 s (0.281) | 300 | 259 MB | 0.3 s | 101 MB |
| onnx-int4 | 4 | 0.118 s (0.140) | 0.251 s (0.282) | 389 | 252 MB | 0.3 s | 78 MB |
| onnx-int4 | 1 | 0.159 s (0.205) | 0.448 s (0.496) | 282 | 251 MB | 0.3 s | 78 MB |

pytorch-fp32 is the setting of D-032 (no cache). Its file is the PyTorch checkpoint. The PyTorch rows include PyTorch's own memory; the ONNX rows never import PyTorch, as in the service.

## Human test set: accuracy before and after quantization

The English part (64 messages, SHA-256 `e6961fd244c5…`), scored once per system after the choice was committed (D-034). The test messages come from the test splits of the datasets whose train splits D-033 trained on (see its caveat).

| System | Intent accuracy (95% interval) | Macro-F1 | Valid JSON | Reply language | Same intent as PyTorch float32 | Answered at the dev threshold (accuracy) |
|---|---|---:|---:|---:|---:|---|
| pytorch-fp32 | 85.9% (55/64; 75–92%) | 87.1% | 100.0% | 100.0% | – | 96.9% (87.1%) |
| onnx-fp32 | 85.9% (55/64; 75–92%) | 87.1% | 100.0% | 100.0% | 100.0% | 96.9% (87.1%) |
| onnx-int8 | 82.8% (53/64; 72–90%) | 83.7% | 100.0% | 100.0% | 93.8% | 93.8% (85.0%) |
| onnx-int4 | 84.4% (54/64; 74–91%) | 85.7% | 98.4% | 96.9% | 93.8% | 93.8% (88.3%) |

Phase 6b scored the same model in bf16 on the GPU: 84.4% ([phase6b_eval.md](phase6b_eval.md)).

Qwen judged the deployed variant's 63 valid test replies with the D-032 prompt: answers 93.7% (59/63; 85–98%), polite clear 95.2% (60/63; 87–98%), language 96.8% (61/63; 89–99%), safe 95.2% (60/63; 87–98%). **All four yes: 87.3% (55/63; 77–93%).** Phase 6b, the same model unquantized: 92.2% (59/64).

Broken answers of the deployed variant on the test set: t0010 (reply: 'I cannot provide details about buyingنوضح لك كيفية الاطلاع على تفاصيل '); t0013 (invalid JSON); t0064 (reply: 'I am not able to answer this albertكلة طريقك، I will pass your request').

## Reply guard (D-035)

Added after the test run: the predictor escalates a reply that does not fit the customer's language. Applied here to the saved predictions; no model was re-run. The development sets give the false-alarm check; the test rows are not an independent estimate, because the test set revealed the problem.

| Set | System | Replies rejected | Answered on the device, before → after | Their intent accuracy |
|---|---|---:|---:|---:|
| sft_val | pytorch-fp32 | 1 of 957 | 922 → 921 | 94.9% → 94.9% |
| sft_val | pytorch-fp32-cache | 1 of 957 | 922 → 921 | 94.9% → 94.9% |
| sft_val | onnx-fp32 | 1 of 957 | 922 → 921 | 94.9% → 94.9% |
| sft_val | onnx-int8 | 3 of 957 | 926 → 923 | 94.8% → 94.8% |
| sft_val | onnx-int4 | 2 of 957 | 924 → 922 | 94.8% → 94.8% |
| real_dev | pytorch-fp32 | 1 of 804 | 785 → 784 | 95.0% → 95.2% |
| real_dev | pytorch-fp32-cache | 1 of 804 | 785 → 784 | 95.0% → 95.2% |
| real_dev | onnx-fp32 | 1 of 804 | 785 → 784 | 95.0% → 95.2% |
| real_dev | onnx-int8 | 3 of 804 | 784 → 782 | 95.0% → 95.0% |
| real_dev | onnx-int4 | 0 of 804 | 780 → 780 | 95.0% → 95.0% |
| test | pytorch-fp32 | 0 of 64 | 62 → 62 | 87.1% → 87.1% |
| test | onnx-fp32 | 0 of 64 | 62 → 62 | 87.1% → 87.1% |
| test | onnx-int8 | 1 of 64 | 60 → 59 | 85.0% → 84.7% |
| test | onnx-int4 | 2 of 64 | 60 → 58 | 88.3% → 87.9% |

Every rejected reply:

| Set | System | Message | Reply |
|---|---|---|---|
| sft_val | pytorch-fp32 | laish ma a3mel qar2 thabtah wala aplik? | سأقوم بتحويل استفسارك حولloans-installmentشكرا لتواصلك معنا. سيتم تحويل طلبك للفريق المختص لمراجعته. |
| sft_val | pytorch-fp32-cache | laish ma a3mel qar2 thabtah wala aplik? | سأقوم بتحويل استفسارك حولloans-installmentشكرا لتواصلك معنا. سيتم تحويل طلبك للفريق المختص لمراجعته. |
| sft_val | onnx-fp32 | laish ma a3mel qar2 thabtah wala aplik? | سأقوم بتحويل استفسارك حولloans-installmentشكرا لتواصلك معنا. سيتم تحويل طلبك للفريق المختص لمراجعته. |
| sft_val | onnx-int8 | when was toyota created | I am unable to answer questions about the history of Warsawبوك. Your inquiry will be forwarded to the appropriate support team. |
| sft_val | onnx-int8 | what cleaning product should i use on wood floors | I cannot provide information on how to get a product to flourندرك استيائك. Your request will be passed to a specialist who can assist you. |
| sft_val | onnx-int8 | should i invest in microsoft stock | I cannot provideفضل شركة فيونكينغ، وسنحيل استفسارك للفريق المختص. |
| sft_val | onnx-int4 | what percentage of species display cold blooded traits | I cannot provide information aboutياتبحرقة حفظملاچ��اچ��ا. Your request will be passed to a team member for further assistance. |
| sft_val | onnx-int4 | what kind of gas do i use | I am unable to provide gasy فقط. Your request will be passed to a team member who can help. |
| real_dev | pytorch-fp32 | Hello. I just noticed I was charged two times for a restaurant I visited this week. Can you reverse the additional charge? | I understand your concern about the reverse of the lastدرءrestaurant. Our team will review your transaction details and get back to you. |
| real_dev | pytorch-fp32-cache | Hello. I just noticed I was charged two times for a restaurant I visited this week. Can you reverse the additional charge? | I understand your concern about the reverse of the lastدرءrestaurant. Our team will review your transaction details and get back to you. |
| real_dev | onnx-fp32 | Hello. I just noticed I was charged two times for a restaurant I visited this week. Can you reverse the additional charge? | I understand your concern about the reverse of the lastدرءrestaurant. Our team will review your transaction details and get back to you. |
| real_dev | onnx-int8 | my card didn't go through when i was buying a case of water at walmart | I am sorry to hear about the issue at the time of theايت. Please contact our support team so they can investigate this for you. |
| real_dev | onnx-int8 | should i invest in microsoft stock | I cannot provideفضل شركة فيونكينغ، وسنحيل استفسارك للفريق المختص. |
| real_dev | onnx-int8 | what did the president say todayu | I am unable to provide information on the president's إعلاميا. Your request will be passed to a team member who can help. |
| test | onnx-int8 | what is albert einstein famous for | I am not able to provide information about albertىطون. Your question will be forwarded to the appropriate support team. |
| test | onnx-int4 | tell me how much i have been buying as of late | I cannot provide details about buyingنوضح لك كيفية الاطلاع على تفاصيل الرصيد بدقة. Please log in to your online banking or mobile app to see your current balance. |
| test | onnx-int4 | what is albert einstein famous for | I am not able to answer this albertكلة طريقك، I will pass your request to a team member for assistance. |
