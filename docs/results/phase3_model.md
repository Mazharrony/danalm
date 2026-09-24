# Phase 3 results: model sanity checks and size

Generated on 2026-09-24 by `scripts/phase3_report.py` from the sanity runs; do not edit by hand. Decision: D-026 in [DECISIONS.md](../DECISIONS.md).

| | small | medium | large |
|---|---:|---:|---:|
| Parameters (total) | 25.2M | 42.2M | 62.1M |
| Non-embedding parameters | 18.9M | 33.8M | 51.6M |
| Initial loss (ln vocab = 9.70) | 9.705 | 9.705 | 9.705 |
| Overfit one batch (first → last loss) | 9.71 → 0.019 | 9.71 → 0.007 | 9.71 → 0.005 |
| Training speed | 90,208 tok/s | 62,917 tok/s | 46,157 tok/s |
| Peak VRAM | 7.53 GB | 9.04 GB | 10.49 GB |
| MFU (bf16 peak 61 TFLOPS) | 26% | 30% | 32% |
| Hours per pass over the train shards | 4.6 | 6.6 | 8.9 |

**Rule (fixed before the runs, `configs/model/report.yaml`):** the largest size whose pass over the train shards takes at most 12 h, whose initial loss is within 0.5 of ln(vocab) and that overfits one batch.

- model-small: passes
- model-medium: passes
- model-large: passes

**Chosen: large**
