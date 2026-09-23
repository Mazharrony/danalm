# Phase 1 results: tokenizer fertility

Measured on 2026-09-23 with `scripts/eval_tokenizers.py` (`configs/tokenizer/eval.yaml`, git commit `8f60b2c9`).

**Fertility** = tokens per word; lower is better. Words are whitespace-separated units of the normalized text (`danalm.data.pipeline.normalize`, diacritics stripped, alef forms kept), identical for every tokenizer. No special tokens are added, and PII placeholders are removed so no tokenizer gets credit for them.

## Evaluation sets (held out: never used to train a DanaLM tokenizer)

| Set | Variety | Kind | Texts | Words |
|---|---|---|---:|---:|
| msa-web | MSA | real | 2,058 | 999,922 |
| msa-wikipedia | MSA | real | 864 | 245,035 |
| arabic-short-messages | MSA | real (MASSIVE ar-SA test) | 2,691 | 14,214 |
| najdi-web | Gulf Arabic | real | 439 | 387,297 |
| emirati-cs-messages | Gulf Arabic | synthetic (teacher) | 300 | 2,048 |
| english-web | English | real | 1,384 | 967,397 |
| english-short-messages | English | real (CLINC150 test) | 3,000 | 24,850 |
| arabizi-cs-messages | Arabizi | synthetic (teacher) | 300 | 2,541 |
| mixed-cs-messages | Mixed | synthetic (teacher) | 248 | 2,170 |
| mixed-web-sentences | Mixed | real (mined from web) | 1,383 | 23,706 |

## Fertility per set

| Set | danalm-standard-16k<br>(16,384) | danalm-arabizi-16k<br>(16,384) | danalm-standard-24k<br>(24,576) | danalm-arabizi-24k<br>(24,576) | danalm-standard-32k<br>(32,768) | danalm-arabizi-32k<br>(32,768) | jais-family-590m<br>(84,992) | qwen3.5<br>(248,070) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| msa-web | 1.640 | 1.640 | 1.531 | 1.531 | 1.463 | 1.463 | 1.291 | 1.726 |
| msa-wikipedia | 1.733 | 1.733 | 1.621 | 1.620 | 1.552 | 1.552 | 1.374 | 1.924 |
| arabic-short-messages | 1.500 | 1.501 | 1.382 | 1.390 | 1.315 | 1.315 | 1.236 | 1.638 |
| najdi-web | 1.756 | 1.756 | 1.639 | 1.639 | 1.565 | 1.564 | 1.563 | 2.028 |
| emirati-cs-messages | 1.576 | 1.575 | 1.497 | 1.497 | 1.440 | 1.440 | 1.373 | 1.843 |
| english-web | 1.664 | 1.663 | 1.563 | 1.562 | 1.504 | 1.503 | 1.350 | 1.353 |
| english-short-messages | 1.307 | 1.304 | 1.255 | 1.251 | 1.217 | 1.214 | 1.116 | 1.090 |
| arabizi-cs-messages | 2.593 | 2.561 | 2.492 | 2.460 | 2.420 | 2.388 | 2.195 | 2.163 |
| mixed-cs-messages | 1.448 | 1.448 | 1.360 | 1.361 | 1.308 | 1.308 | 1.229 | 1.403 |
| mixed-web-sentences | 1.971 | 1.969 | 1.841 | 1.838 | 1.764 | 1.761 | 1.537 | 1.861 |

## Fertility per variety (mean of its sets)

| Variety | danalm-standard-16k | danalm-arabizi-16k | danalm-standard-24k | danalm-arabizi-24k | danalm-standard-32k | danalm-arabizi-32k | jais-family-590m | qwen3.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSA | 1.624 | 1.625 | 1.511 | 1.514 | 1.443 | 1.443 | 1.301 | 1.763 |
| Gulf Arabic | 1.666 | 1.666 | 1.568 | 1.568 | 1.503 | 1.502 | 1.468 | 1.936 |
| English | 1.485 | 1.484 | 1.409 | 1.407 | 1.360 | 1.358 | 1.233 | 1.221 |
| Arabizi | 2.593 | 2.561 | 2.492 | 2.460 | 2.420 | 2.388 | 2.195 | 2.163 |
| Mixed | 1.709 | 1.708 | 1.600 | 1.600 | 1.536 | 1.535 | 1.383 | 1.632 |
| Overall | 1.816 | 1.809 | 1.716 | 1.710 | 1.653 | 1.645 | 1.516 | 1.743 |

## Decision (rules fixed in the config before the run)

1. **Pre-tokenizer:** use `arabizi` if, at every vocab size, Arabizi fertility drops by at least 5% and no other variety gets more than 1% worse than `standard`.

| Vocab | Arabizi gain | Worst change elsewhere |
|---:|---:|---:|
| 16,384 | +1.2% | +0.0% |
| 24,576 | +1.3% | +0.2% |
| 32,768 | +1.3% | -0.0% |

   → **standard**

2. **Vocabulary size:** lowest training FLOPs per word (overall fertility × FLOPs per token of a reference model, d_model=512, 12 layers, tied embeddings); the smaller size wins within 2%.

| Vocab | FLOPs per word (relative) | Embedding params at d_model=512 |
|---:|---:|---:|
| 16,384 | 1.000 | 8.4M |
| 24,576 | 1.031 | 12.6M |
| 32,768 | 1.076 | 16.8M |

   → **danalm-standard-16k** (vocab 16,384)
