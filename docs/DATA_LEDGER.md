# Data ledger

Every piece of data DanaLM uses: where it came from, under which licence, and how much we took.

**Rules** (set by the project owner, see [DECISIONS.md](DECISIONS.md) D-012):

1. Use only free data that is legal to use. Check the licence on the primary source (dataset or
   model card, licence file) *before* downloading anything.
2. Record every collection in section 1, with counts. Counts are **measured by us** unless marked
   "per card". Token counts use the DanaLM tokenizer once it exists (Phase 1); until then we
   record documents, UTF-8 bytes and words.

## 1. Collected data

Generated from [`data_ledger.jsonl`](data_ledger.jsonl) by the data scripts; do not edit by hand.
"raw" is what was downloaded or generated; "kept" is what survived cleaning, deduplication and
filtering. Tokens are counted on the kept text with the tokenizer named in brackets.

<!-- ledger:start -->
| ID | Source | Revision | Licence | Kind | Collected on | Used for | Docs (raw → kept) | UTF-8 bytes (kept) | Words (kept) | Tokens (kept) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tokenizer-eval-data/massive-ar-test | AmazonScience/massive `ar-SA/test/*.parquet` | `ed58ac42` | CC-BY-4.0 | real (localized by native speakers) | 2026-09-23 | tokenizer fertility evaluation only (Phase 1) | 2,974 → 2,691 | 134,274 | 14,214 | 21,325 (danalm-v1) | 3 row groups; Arabic (ar-SA) assistant utterances, test split |
| tokenizer-eval-data/clinc-test | clinc/clinc_oos `plus/test-*.parquet` | `155b9c71` | CC-BY-3.0 | real (crowdsourced) | 2026-09-23 | tokenizer fertility evaluation only (Phase 1) | 5,500 → 5,498 | 220,917 | 45,596 | 59,603 (danalm-v1) | 6 row groups; English assistant queries incl. out-of-scope, test split |
| synth-fertility-eval/gulf_emirati | teacher: Qwen3.5-9B Q4_K_M (unsloth/Qwen3.5-9B-GGUF) via llama.cpp; 57/300 lines then corrected by an external AI tool (file supplied by the owner, 2026-09-23) | `3885219b` | Apache-2.0 (teacher outputs); the external AI tool's terms apply to its corrections | synthetic (teacher + AI-corrected) | 2026-09-23 | tokenizer fertility evaluation only; NEVER for training (contains external-AI edits) | 376 → 300 | 19,183 | 2,058 | 3,183 (danalm-v1) | replaces Qwen's raw set (kept as archive/gulf_emirati.qwen_raw.jsonl); lines with clear non-Gulf dialect markers: 43/300 raw -> 10/300 corrected. A second AI-edited variant (native_emirati_uae_dubai_replacement.jsonl: 14/300 markers plus apparently invented forms) was reviewed and not used (archived). Not checked by a native speaker |
| synth-fertility-eval/arabizi | teacher: Qwen3.5-9B Q4_K_M (unsloth/Qwen3.5-9B-GGUF) via llama.cpp | `3885219b` | Apache-2.0 (teacher outputs) | synthetic | 2026-09-23 | tokenizer fertility evaluation only (Phase 1); teacher-quality pilot | 367 → 300 | 13,436 | 2,541 | 6,589 (danalm-v1) | 19 requests, 7,174 teacher tokens; not yet checked by a native speaker |
| synth-fertility-eval/mixed | teacher: Qwen3.5-9B Q4_K_M (unsloth/Qwen3.5-9B-GGUF) via llama.cpp | `3885219b` | Apache-2.0 (teacher outputs) | synthetic | 2026-09-23 | tokenizer fertility evaluation only (Phase 1); teacher-quality pilot | 383 → 248 | 16,417 | 2,170 | 3,142 (danalm-v1) | 19 requests, 5,198 teacher tokens; not yet checked by a native speaker |
| tokenizer-corpus/fineweb2-arb | HuggingFaceFW/fineweb-2 `data/arb_Arab/train/*.parquet` | `af9c1333` | ODC-By-1.0 (+ Common Crawl ToU) | real | 2026-09-23 | tokenizer training (Phase 1) | 101,702 → 101,700 | 512,846,277 | 49,119,496 | 80,377,186 (danalm-v1) | 102 row groups; Standard Arabic web text |
| tokenizer-corpus/fineweb2-ars | HuggingFaceFW/fineweb-2 `data/ars_Arab/train/*.parquet` | `af9c1333` | ODC-By-1.0 (+ Common Crawl ToU) | real | 2026-09-23 | tokenizer training (Phase 1) | 23,488 → 23,488 | 191,756,099 | 20,355,907 | 35,036,735 (danalm-v1) | 24 row groups; Najdi (Saudi) dialect web text, the closest open Gulf-like dialect |
| tokenizer-corpus/wikipedia-ar | wikimedia/wikipedia `20231101.ar/*.parquet` | `b04c8d1c` | CC-BY-SA-3.0 + GFDL | real | 2026-09-23 | tokenizer training (Phase 1) | 42,471 → 42,451 | 111,662,024 | 10,600,741 | 18,311,659 (danalm-v1) | 44 row groups; Arabic Wikipedia |
| tokenizer-corpus/fineweb-edu | HuggingFaceFW/fineweb-edu `sample/10BT/*.parquet` | `87f09149` | ODC-By-1.0 (+ Common Crawl ToU) | real | 2026-09-23 | tokenizer training (Phase 1) | 54,137 → 54,122 | 256,722,822 | 41,532,739 | 66,489,977 (danalm-v1) | 55 row groups; Educational English web text |
| tokenizer-corpus/wikipedia-en | wikimedia/wikipedia `20231101.en/*.parquet` | `b04c8d1c` | CC-BY-SA-3.0 + GFDL | real | 2026-09-23 | tokenizer training (Phase 1) | 16,520 → 16,509 | 47,750,961 | 7,580,350 | 14,013,159 (danalm-v1) | 17 row groups; English Wikipedia |
| tokenizer-corpus/bitext-support | bitext/Bitext-customer-support-llm-chatbot-training-dataset `default/train/*.parquet` | `f93dfd84` | CDLA-Sharing-1.0 | synthetic (by Bitext) | 2026-09-23 | tokenizer training (Phase 1) | 26,872 → 26,872 | 18,305,246 | 3,049,435 | 3,937,071 (danalm-v1) | 27 row groups; English customer-support requests and answers |
| tokenizer-corpus/bitext-banking | bitext/Bitext-retail-banking-llm-chatbot-training-dataset `*.parquet` | `3e362109` | CDLA-Sharing-1.0 | synthetic (by Bitext) | 2026-09-23 | tokenizer training (Phase 1) | 25,545 → 25,545 | 24,525,123 | 4,177,867 | 5,401,059 (danalm-v1) | 1 row groups; English retail-banking requests and answers |
| tokenizer-corpus/bitext-telco | bitext/Bitext-telco-llm-chatbot-training-dataset `default/train/*.parquet` | `45588345` | CDLA-Sharing-1.0 | synthetic (by Bitext) | 2026-09-23 | tokenizer training (Phase 1) | 26,000 → 25,961 | 14,068,652 | 2,180,712 | 3,305,755 (danalm-v1) | 26 row groups; English telecom requests and answers |
| tokenizer-corpus/clinc-train | clinc/clinc_oos `plus/train-*.parquet` | `155b9c71` | CC-BY-3.0 | real (crowdsourced) | 2026-09-23 | tokenizer training (Phase 1) | 15,250 → 15,120 | 607,188 | 126,663 | 164,838 (danalm-v1) | 16 row groups; English assistant queries incl. out-of-scope (train split only) |
| tokenizer-corpus/massive-ar-train | AmazonScience/massive `ar-SA/train/*.parquet` | `ed58ac42` | CC-BY-4.0 | real (localized by native speakers) | 2026-09-23 | tokenizer training (Phase 1) | 11,514 → 10,403 | 539,448 | 58,251 | 87,184 (danalm-v1) | 12 row groups; Arabic (ar-SA) assistant utterances (train split only) |

**Total kept so far:** 227,218,465 tokens (danalm-v1).
<!-- ledger:end -->

## 2. Licence checks

Checked on 2026-09-23 via the Hugging Face Hub API and the dataset/model cards.

### Approved: clear licence and clear provenance

| Source | What it is (per card) | Licence | Planned use | Notes |
|---|---|---|---|---|
| `HuggingFaceFW/fineweb-2`, config `arb_Arab` | Standard Arabic web text: 32.8B words, 62M docs, 25 parquet shards of ~4.5 GB | ODC-By 1.0, plus Common Crawl terms of use | pretraining, tokenizer | we sample a small fraction |
| `HuggingFaceFW/fineweb-2`, config `ars_Arab` | Najdi (Saudi) dialect web text: 301k docs, 0.78 GB parquet | ODC-By 1.0, plus Common Crawl terms of use | pretraining, tokenizer | closest open Gulf-like dialect; lots of forum boilerplate, so filter it |
| `HuggingFaceFW/fineweb-edu` | educational English web text | ODC-By 1.0, plus Common Crawl terms of use | pretraining, tokenizer | |
| `wikimedia/wikipedia`, `20231101.ar` and `20231101.en` | Wikipedia articles | CC-BY-SA 3.0 + GFDL | pretraining, tokenizer | attribution in the model card; share-alike applies only if we republish the text (not planned) |
| `Qwen/Qwen3.5-9B`, run locally as `unsloth/Qwen3.5-9B-GGUF` Q4_K_M (HF commit `3885219b`) | teacher LLM, 201 languages and dialects | Apache-2.0 | synthetic SFT data; teacher baseline in Phase 6 | Apache-2.0 places no limits on training with the outputs |
| `PolyAI/banking77` | English banking queries, 77 intents | CC-BY-4.0 | intent-taxonomy reference, teacher seed prompts | attribution |
| `clinc/clinc_oos` | English intents, including out-of-scope queries | CC-BY-3.0 | examples for `other` / out-of-scope | attribution |
| `AmazonScience/massive` | virtual-assistant utterances in 51 languages, incl. Arabic (ar-SA) | CC-BY-4.0 | Arabic intent-style utterances | a different domain (assistant, not customer service) |
| `bitext/Bitext-customer-support-…`, `…-retail-banking-…`, `…-telco-…` | synthetic English customer-service conversations with intent labels | CDLA-Sharing-1.0 | taxonomy and teacher seeds | if we publish modified data it must stay CDLA-Sharing; trained models are not restricted |
| `CohereForAI/aya_dataset` | human-written prompts and answers in many languages, incl. Arabic | Apache-2.0 | possible Arabic instruction data | |
| `CAMeL-Lab/bert-base-arabic-camelbert-mix` | Arabic BERT (MSA, dialectal, classical) | Apache-2.0 | Phase 6 classifier baseline | replaces MARBERT/AraBERT (see Rejected) |
| `inceptionai/jais-family-590m` (tokenizer files only) | Arabic–English tokenizer | Apache-2.0 | Phase 1 fertility comparison | |

About Common Crawl-based data: ODC-By covers the *dataset*, while the web text itself stays with
its authors. Training on it is standard practice for open LLMs, but it is not zero-risk. We never
republish the text, and we mask PII ourselves.

### Needs a provenance check before use

Small dialect and Arabizi sets on the Hub carry permissive licence tags (e.g.
`HeshamHaroon/saudi-dialect-conversations`, `ebubekr53/organic-gulf-arabic-dialect-dataset`,
`rabeeeehh/arabizi-kit-corpus`, `drelhaj/Arabic-Dialects`). A tag alone is not proof of rights.
Before using any of them we check how the text was obtained: scraped social media, or generated by
a closed model whose terms forbid training competing models, are both excluded.

### Rejected

| Source | Reason |
|---|---|
| `UBC-NLP/MARBERT`, `UBC-NLP/MARBERTv2`, `aubmindlab/bert-base-arabertv02` | no licence on the model card, so the rights are unclear |
| Tweet collections, even when tagged MIT or CC-BY | X/Twitter terms do not allow redistributing tweet text, so the uploader cannot license it |
| Datasets tagged `cc-by-nc*`, `other`, or with no licence (43 of the 79 found in the dialect search) | non-commercial or unclear rights |
| `HuggingFaceFW/fineweb-2`, config `acm_Arab` | tiny (7.4k docs) and mostly song lyrics |
