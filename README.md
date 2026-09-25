# DanaLM (دانة)

**A 62M-parameter language model, trained from scratch on one RTX 4070, for UAE customer
service.** It reads a customer's message in Gulf Arabic, English, Arabizi or a mix of them, and
answers with strict JSON: `{"intent": "...", "reply": "..."}`.

It is built to run quantized on a CPU or a phone. It handles the easy majority of messages on the
device and passes the rest to a bigger model or a person. The name comes from *dana* (دانة), the
Gulf pearl: small but valuable.

How it is built:

- Every run can be reproduced from its config, seed and git commit.
- Every data source is licence-checked and counted in the [data ledger](docs/DATA_LEDGER.md).
- Every decision is recorded in [DECISIONS.md](docs/DECISIONS.md), together with the rule that
  settled it, written down before the result was known.

## Where it stands (2026-09-25)

| | DanaLM | For comparison |
|---|---|---|
| Size | 62.1M parameters; deployed as a **78 MB INT4 ONNX model**, no PyTorch needed | CAMeLBERT-mix classifier: 110M; Qwen3.5-35B-A3B |
| Intent accuracy, 64 real English test messages | **84%** with the deployed INT4, 86% in float32 (56% before real training data) | CAMeLBERT 47%, Qwen zero-shot 89% |
| Intent accuracy, 804 held-out real messages (dev set) | 93% INT4, 94% float32 | the model before real training data: 66% |
| Intent accuracy, synthetic validation set (957) | 93% | CAMeLBERT 93% |
| Valid JSON / reply in the customer's language, test | 98% / 97% INT4; 100% / 100% float32 | — |
| CPU time per message, 4 threads | **0.12 s** for the answer, 0.25 s with the confidence (was 0.79 s and 0.97 s) | — |
| Peak memory | **252 MB** (was 1,190 MB with PyTorch) | — |

The first model, trained only on teacher-written messages, fell from 93% on synthetic data to
56% on real messages. Training on 2,921 real, openly licensed customer messages (D-033) raised it
to 84%, close to the 35B teacher's 89%. Part of that gain is style the test messages share with
the new training data, since both come from the same two public datasets.

Phase 7 made it deployable:
- A KV cache and ONNX Runtime, then quantization, made it 6.7× faster per answer and 4.7× lighter
  in memory.
- It runs as a FastAPI service, a Docker image and a Gradio demo.

What is still open:

- **Gulf Arabic, Arabizi and mixed test messages.** Those parts of the test set need a native
  speaker, and they are the real check.
- **Confidence.** At the chosen threshold, 88% of the answered test messages are right, short of
  the 95% bar.
- **Garbled replies.** A few replies mix scripts, e.g. "…the lastدرءrestaurant…". That happened in 2 of
  1,761 development replies, in float32 and INT4 alike, and in 2 of INT4's 64 test replies. A
  guard in the service escalates them instead of sending them (D-035).

Next steps:

1. Presentation (Phase 8): a README with an architecture diagram, a model card, and publishing on
   Hugging Face.
2. Complete the Gulf Arabic, Arabizi and mixed parts of the human test set.
3. Make the confidence trustworthy on real messages.

## Progress

| Phase | Status | Headline |
|---|---|---|
| 0. Setup | done | Every run records its config, seed, git commit and library versions |
| 1. Tokenizer | done | 16,384-token byte-level BPE; 14% fewer tokens than Qwen3.5 on Gulf Arabic |
| 2. Data | done; the Arabic parts of the test set are open | 1.495B pretraining tokens; 18,547 SFT examples (20,572 after the Phase 5 clean-up and top-up, 24,263 with the real messages of D-033) |
| 3. Model | done | Llama-style decoder, 62.1M parameters; passes the sanity checks |
| 4. Pretraining | done | Validation loss 9.705 → 3.330 in 4 h 49 min; a second pass reached 3.232 |
| 5. SFT | done; a second round added real messages (D-033) | Valid JSON 100%; intent accuracy 93.1% on the SFT validation split and 94.4% on 804 held-out real messages |
| 6. Evaluation | English part done; the Arabic parts need a native speaker | On 64 real English messages: intent accuracy 84% after D-033 (56% before; CAMeLBERT 47%, Qwen 89%); valid JSON 100% |
| 7. Quantization and deployment | done | INT4 ONNX, 78 MB: 0.12 s per answer on 4 CPU threads (6.7× faster), 84% on the English test; FastAPI, Docker, CI, Gradio demo |
| 8. Presentation | later | |

Results by phase:

- **Tokenizer (Phase 1):** `danalm-v1` is a 16,384-token byte-level BPE for Arabic, English and
  Arabizi. Against Qwen3.5's 248k-token tokenizer, it needs 14% fewer tokens on Gulf Arabic and
  8% fewer on MSA ([results](docs/results/phase1_tokenizer.md)).
- **Pretraining corpus (Phase 2a):** 1.495B tokens from 10 openly licensed sources: 64% Arabic
  (including 300M tokens of Najdi dialect) and 36% English. It is cleaned, deduplicated,
  tokenized, and checked by decoding sample documents back to their text
  ([results](docs/results/phase2_data.md)).
- **SFT data (Phase 2b):** 18,547 customer-service examples covering 21 intents, in Gulf Arabic,
  English, Arabizi and mixed text. Phase 5 cleaned and extended them to 20,572 (see below).
  - A local Qwen3.5-35B-A3B teacher wrote them. Arabizi messages are answered in Gulf Arabic
    script (D-023).
  - The `other` intent adds 491 real out-of-scope questions from CLINC150 and MASSIVE (D-024).
  - Filters removed non-Gulf dialect, formal Arabic, and replies that claim actions the
    assistant cannot take.
  - A blind judge, working from the final intent descriptions, agreed with 87.9% of the labels.
  ([teacher pilot](docs/results/phase2_teacher_pilot.md), [results](docs/results/phase2_data.md))
- **Model (Phase 3):** a Llama-style decoder with RMSNorm, RoPE, grouped-query attention,
  SwiGLU and tied embeddings, at 62.1M parameters. It passes the pre-training sanity checks:
  initial loss 9.705 against ln(16,384) = 9.704, and it overfits one batch. On the RTX 4070 it
  trains at 46k tokens/s, or 9.0 h per pass over the corpus
  ([results](docs/results/phase3_model.md)).
- **Pretraining (Phase 4):** one pass over the corpus's train split (1.488B tokens) in 4 h 49 min
  on the RTX 4070.
  With `torch.compile` (through `triton-windows`) it runs at 87k tokens/s, about 60% MFU.
  - A pilot at ~1% of the budget chose the peak learning rate of 2e-3 by a rule fixed in advance
    ([pilot](docs/results/phase4_pilot.md)).
  - Validation loss fell from 9.705 to 3.330, with no loss spikes and no restarts
    ([results](docs/results/phase4_pretrain.md)).
  - A second pass over the same data brought it to 3.232 (D-031,
    [results](docs/results/phase4_second_pass.md)).
- **SFT (Phase 5):** the model answers `<|user|>message<|assistant|>{"intent": …, "reply": …}`.
  The loss counts only the answer.
  - Data: 20,572 training examples.
    - Qwen checked every reply for broken words. The 763 broken ones were rewritten, and 679
      of the rewrites passed.
    - A top-up added 3,066 examples, mostly for the thin mixed and Arabizi cells and the
      intents that get confused.
  - Two sweeps (3 learning rates × 5 epochs each; the second on the second-pass base) ran,
    with the choosing rule fixed in advance (D-029).
  - The chosen model scores, on the SFT validation split: valid JSON 100%, intent accuracy
    92.9% (macro-F1 92.5%), and replies in the right language 100%.
  - The CAMeLBERT-mix classifier baseline scores 93.3% on the same split. These numbers
    come from synthetic validation data. The honest score comes from the human test set in
    Phase 6 ([results](docs/results/phase5_sft.md)).
- **SFT with real messages (Phase 5b, D-033):** Phase 6 showed the synthetic data was the
  weakness, so this round added real customer messages. Every rule was fixed before any data
  was fetched ([results](docs/results/phase5b_real.md)).
  - The messages come from the *train* splits of Banking77 and CLINC150 (CC-BY), mapped to our
    intents. Near copies of test messages were dropped first.
  - 804 of them are held out as a real dev set. It is used only to choose the model and its
    confidence threshold.
  - Qwen wrote the replies. The blind label judge and the reply check filtered them, which left
    2,921 real examples.
  - A style top-up added 770 short, lowercase, typo-laden English messages.
  - The D-029 recipe ran on both pretraining bases × two learning rates.
  - The chosen model scores 94.4% on the real dev set, where the Phase 5 model scores 66.4%. On
    the synthetic validation split it scores 93.1%, and every language variety holds.
- **Evaluation (Phase 6, English part):** the 64 human test messages, scored once per model with
  the same protocol.

  | | Intent accuracy | Macro-F1 | Valid JSON | Reply language |
  |---|---:|---:|---:|---:|
  | DanaLM, Phase 5 model ([results](docs/results/phase6_eval.md)) | 56.2% | 57.8% | 100% | 100% |
  | DanaLM after real messages ([results](docs/results/phase6b_eval.md)) | **84.4%** | 85.5% | 100% | 100% |
  | CAMeLBERT-mix classifier (110M) | 46.9% | 46.4% | – | – |
  | Qwen3.5-35B-A3B, zero-shot | 89.1% | 90.3% | – | – |

  - Real messages fixed 19 of the first model's errors and added 1 (McNemar p = 0.00004). The
    macro-F1 gap to Qwen, −4.7 points, is inside its 95% interval (−16 to +6).
  - Caveat: the test messages come from the test splits of the same two datasets as the new
    training messages. Part of the gain is their shared style.
  - Three of the four D-030 bars are met. Coverage is still missed: at the threshold fixed on
    the real dev set, 87% of the answered test messages are right, not 95%.
  - Latency: 0.63 s per message on the CPU (float32, 4 threads) for the Phase 5 model, before
    any optimisation. The second measurement ran while Windows was throttling background
    processes, so it is not comparable (see D-033). Phase 7 measures it properly.
- **Quantization and deployment (Phase 7, D-034):** the rules were fixed before any quantized
  model was scored ([results](docs/results/phase7_deploy.md)).
  - **Exactness:** a KV-cache step graph in ONNX gives answers identical to PyTorch on all
    1,761 development messages.
  - **Choice:** INT8 (101 MB) and INT4 (78 MB) both kept intent accuracy within 1 point on the
    development sets. The rule deploys the smallest, INT4, which passed by less than one message.
  - **Test set, once per system:** float32 85.9%, INT8 82.8%, INT4 84.4%.
    - INT4 gave three broken answers on the test set: two mixed Arabic words into English
      replies, and one looped until the length limit.
    - Qwen judged 87% of INT4's replies good, against 92% unquantized.
    - A reply-language guard now escalates such replies (D-035). On the development sets it
      rejected only garbled replies (2–6 of 1,761 per system).
  - **Speed** (batch 1, 4 CPU threads): 0.118 s per answer against 0.788 s before (6.7×), and
    0.251 s with the confidence. Peak memory is 252 MB against 1,190 MB. With 1 thread: 0.159 s
    per answer.
  - **Trade-off:** INT8 is faster than INT4 for the full prediction (0.188 s). The size-first rule
    did not weigh that.
  - **Serving:** a torch-free FastAPI service, a Docker image, a Gradio demo for Hugging Face
    Spaces, and GitHub Actions CI with a smoke evaluation.

Open: **the Arabic, Arabizi and mixed parts of the human test set** need a native Gulf Arabic
speaker (see [Test data](#test-data)).

The plan is in [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md), and the reasons behind every
choice are in [docs/DECISIONS.md](docs/DECISIONS.md). Every data source, with its licence and
token count, is in [docs/DATA_LEDGER.md](docs/DATA_LEDGER.md).

## Test data

DanaLM is scored only on data it never trained on. Like everything in `data/`, the sets are
gitignored: the repository shows how each set is built and checked, not the messages themselves.

| Set | Size | Used for | Status |
|---|---|---|---|
| **Human test set** (`data/test/test_set.jsonl`) | ~420 messages planned, 64 so far | Phase 6: intent accuracy, macro-F1 and valid-JSON rate, per language | English part ready |
| Pretraining validation split | 7.64M tokens, ~0.5% of each source's documents | Validation loss during pretraining | done |
| SFT validation split | 957 examples (11 of 968 dropped for broken replies) | Choosing the SFT model (D-029). The same teacher wrote it as the training data, so it cannot replace the test set | done |
| Tokenizer evaluation sets | 10 sets of web text and customer-service messages | Comparing tokenizers in Phase 1 only | done |

### Human test set

It is the only honest measure of the model, so it is never used for training, tuning, prompt
design or choosing between models. The full guide is [docs/TEST_SET.md](docs/TEST_SET.md).

- **Size:** 21 intents × 4 varieties (Gulf Arabic, English, Arabizi, mixed) × 5 messages = 420,
  and never fewer than 3 per cell.
- **Who writes it:**
  - People write it, not AI tools, and without looking at the training data. No real personal
    data goes in.
  - A native Gulf Arabic speaker writes or checks the Gulf Arabic, Arabizi and mixed messages.
- **Leak check:** `scripts/check_overlap.py` compares every test message with all the training
  text and finds exact and near copies (character 3-gram MinHash). It must report 0 overlaps
  before every evaluation.

| Part | Needed | Now |
|---|---:|---|
| English, the 10 intents covered by public human-written test data | 50 | **64 messages**, checked by a person (MR); 0 overlaps with training data |
| English, the other 11 intents (telecom and most delivery intents) | 55 | a person writes them |
| Gulf Arabic, all 21 intents | 105 | needs a native speaker |
| Arabizi, all 21 intents | 105 | needs a native speaker |
| Mixed, all 21 intents | 105 | needs a native speaker |

**The English candidates** (D-025), reviewed on 2026-09-24:

- **Result:** MR checked 67 candidates on a click-through page: the 65 below plus 2 more
  CLINC150 test sentences. They kept 64, dropped 3 and corrected 1 intent. The sheet is
  `data/test/candidates-en/review-human.csv`, and `import_review.py` wrote `test_set.jsonl`.

- **Source:** the test splits of Banking77 (CC-BY-4.0) and CLINC150 (CC-BY-3.0), each pinned to
  a fixed revision. Their labels are mapped to our intents by rule. The train splits are never
  used.
- **Selection:**
  - 160 candidates were fetched: 70 from Banking77 and 90 from CLINC150, at most 10 per intent
    from each.
  - The overlap check dropped 48 of them, mostly CLINC150 test sentences that nearly repeat its
    train split.
  - The sheet keeps up to 7 per intent, which gives 65 rows: 31 from Banking77 and 34 from
    CLINC150.
- **Per intent:**
  - 7 each: `account_access`, `card_not_working`, `fees_and_charges`, `lost_or_stolen_card`,
    `other`, `transfer_issue`, `unrecognized_transaction`.
  - 6 each: `balance_or_statement`, `order_status`.
  - 4: `loans_and_credit`.
- **To review:**
  1. In each row, write `y` or `n` in `keep`.
  2. If the mapped intent is wrong, write the right one in `correct_intent`. Anything unusual
     goes in `notes`.
  3. Import the kept rows with your initials, then re-run the overlap check:

```bash
uv run python scripts/import_review.py --config configs/data/test_candidates_en.yaml review.verified_by=XX
uv run python scripts/check_overlap.py --config configs/data/overlap.yaml
```

The 169 unit tests (`uv run pytest`) run on tiny synthetic fixtures in `tests/`, not on any of
these sets.

## Setup (Windows 11, native)

You need [uv](https://docs.astral.sh/uv/) and an NVIDIA driver that supports CUDA 13.
`make` is optional (`winget install ezwinports.make`), because every target is a plain `uv run` command.

```bash
uv sync --locked            # .venv: Python 3.12 + PyTorch 2.14 (CUDA 13.0), ~2.2 GB download
uv run pre-commit install   # ruff, black and data guards run on every commit
uv run wandb login          # once; or add tracking.mode=offline to any command
```

On Windows, set `PYTHONUTF8=1` when running scripts directly; the Makefile already does this.
Without it, log files use cp1252 and fail on Arabic text.

## Everyday commands

| make | Plain command | What it does |
|---|---|---|
| `make test` | `uv run pytest` | Unit tests |
| `make lint` | `uv run pre-commit run --all-files` | ruff, black, whitespace, lock check, data guards |
| `make check-env` | `uv run python scripts/check_env.py --config configs/check_env.yaml` | Records the environment and measures GPU throughput |
| `make data-smoke` | `uv run python scripts/prepare_data.py --config configs/data/smoke.yaml` | Runs the data pipeline on the test fixture |

Every script takes `--config <file.yaml>` plus `key=value` overrides, e.g. `seed=1 tracking.mode=offline`.

### Phase 1: tokenizer

Run these one at a time: each target is CPU- or GPU-heavy.

| make | What it does | Time on the dev machine |
|---|---|---|
| `make tokenizer-data` | Downloads a licence-checked 828M-character sample (~0.5 GB of text columns), then cleans it | ~9 + 7 min |
| `make synthetic-eval` | Generates Emirati, Arabizi and mixed eval messages with the local Qwen3.5 teacher (GPU) | ~2 min |
| `make tokenizers` | Trains the 6 candidates (16k / 24k / 32k × two pre-tokenizers), about 45 s and 3.5 GB RAM each | ~5 min |
| `make tokenizer-eval` | Fertility on 10 held-out sets vs Jais and Qwen3.5, applying the decision rules fixed in advance | ~1 min |
| `make tokenizer-final` | Trains `danalm-v1` and writes token counts into the data ledger | ~2 min |

### Phase 2: data

Run these one at a time as well.

- **Disk:** the corpus needs ~19 GB (8.1 GB raw, 7.9 GB cleaned, 3.0 GB of token shards).
- **Teacher:** the SFT scripts need a local llama.cpp server and the Qwen3.5-35B-A3B GGUF named
  in `configs/teacher/qwen35_35b_a3b.yaml`. It uses about 10 GB of VRAM and up to 26 GB of RAM.
- **Crash safety:** teacher answers are saved as they arrive, so rerunning a crashed job
  resumes it.

| Command | What it does | Time on the dev machine |
|---|---|---|
| `make pretrain-data` | Samples the 10 corpus sources at pinned revisions, cleans them, writes 100M-token uint16 shards and checks them against the text | ~40 + 70 + 16 min |
| `uv run python scripts/generate_sft.py --config configs/sft/full.yaml` | Generates ~30k SFT examples with the teacher and filters them | ~4.6 h |
| `uv run python scripts/write_replies.py --config configs/sft/arabizi_replies.yaml` | New Gulf Arabic replies for the Arabizi messages (D-023) | ~55 min |
| `uv run python scripts/fetch_labelled.py --config configs/sft/other_real.yaml`, then `write_replies.py` with the same config | Real out-of-scope messages for `other`, with teacher replies (D-024) | ~7 min |
| `uv run python scripts/generate_sft.py --config configs/sft/other_mixed.yaml` | Code-mixed `other` top-up | ~4 min |
| `uv run python scripts/merge_sft.py --config configs/sft/final.yaml` | Combines all sources into one candidate file | seconds |
| `uv run python scripts/verify_sft.py --config configs/sft/final.yaml` | The judge re-labels every candidate without seeing the intended label | ~32 min |
| `uv run python scripts/build_sft_dataset.py --config configs/sft/final.yaml` | Keeps the examples the judge agreed with and splits train/val | seconds |
| `uv run python scripts/phase2_report.py --config configs/data/phase2_report.yaml` | Writes `docs/results/phase2_data.md` from the outputs | seconds |
| `uv run python scripts/check_overlap.py --config configs/data/overlap.yaml` | Fails if a test message is too close to any training text (for the human test set) | ~7 min |

The English test-set candidates and their review are described in [docs/TEST_SET.md](docs/TEST_SET.md).

### Phase 3: model

| Command | What it does | Time on the dev machine |
|---|---|---|
| `uv run pytest tests/test_model.py` | Shapes, causality, tied weights, RoPE, masked loss, parameter and FLOP formulas, overfitting a batch | seconds |
| `uv run python scripts/model_sanity.py --config configs/model/large.yaml` | The brief's checks on the GPU: initial loss vs ln(vocab), overfit one batch, tokens/s, peak VRAM, MFU | ~1 min per size |
| `uv run python scripts/phase3_report.py --config configs/model/report.yaml` | Compares the sizes and applies the size rule that was fixed in advance | seconds |

### Phase 4: pretraining

Stop the teacher server first: the training scripts refuse to run while it is up.

| Command | What it does | Time on the dev machine |
|---|---|---|
| `uv run pytest tests/test_train.py` | Learning-rate schedule, weight-decay groups, windows that make every token a target once, checkpoint pruning, bit-identical resume on the CPU | seconds |
| `uv run python scripts/pretrain.py --config configs/pretrain/pilot.yaml` | One pilot run (~1% of the budget); the other learning rates and the resume check are listed in `configs/pretrain/pilot_report.yaml` | ~3 min each |
| `uv run python scripts/pilot_report.py --config configs/pretrain/pilot_report.yaml` | Pilot curves, and the learning-rate rule that was fixed in advance | seconds |
| `uv run python scripts/pretrain.py --config configs/pretrain/full.yaml` | The full run: checkpoints every 250 steps, and rerunning the command resumes from the latest (a finished run exits at once) | ~4 h 50 min |
| `uv run python scripts/pretrain_report.py --config configs/pretrain/report.yaml` | Curves, validation loss per source, train vs validation windows, sample continuations | ~1 min |

### Phase 5: SFT

Stop the teacher server before the training steps; the Qwen steps start it themselves.

| Command | What it does | Time on the dev machine |
|---|---|---|
| `uv run pytest tests/test_sft_train.py tests/test_reply_check.py` | Chat format and loss masking, batching, batched greedy decoding, intent likelihoods, metrics, the selection rule | seconds |
| `uv run python scripts/check_replies.py --config configs/sft/reply_check.yaml` | Qwen proofreads a sample of 500 replies (`check.scope=all`: every reply) | ~3 min (all: ~80 min) |
| `uv run python scripts/write_replies.py --config configs/sft/fix_replies.yaml` | New replies for the broken ones | ~7 min |
| `uv run python scripts/generate_sft.py --config configs/sft/topup.yaml` | Top-up and contrast examples (then `verify_sft.py` with the same config) | ~1 h 45 min |
| `uv run python scripts/assemble_sft_v3.py --config configs/sft/v3.yaml` | Builds `data/sft/final-v3`: rewritten replies, top-up in training only, no copies of validation or test messages | seconds |
| `uv run python scripts/sft.py --config configs/sft/train.yaml train.lr=3.0e-4 run_name=sft-r1-lr3e-4` | One SFT run (5 epochs, evaluated after each) | ~6 min |
| `uv run python scripts/sft_report.py --config configs/sft/report.yaml` | Applies the D-029 rule to all runs, writes the results page and `artifacts/checkpoints/sft-selected.json` | seconds |
| `uv run python scripts/baseline_camelbert.py --config configs/eval/baseline_camelbert.yaml` | The CAMeLBERT-mix intent classifier baseline (Phase 6, D-027) | ~2 min |

### Phase 5b: real messages (D-033)

| Command | What it does | Time on the dev machine |
|---|---|---|
| `uv run python scripts/fetch_labelled.py --config configs/sft/real.yaml` | Fetches the Banking77 and CLINC150 train splits and maps them to our intents (and writes the ledger) | seconds |
| `uv run python scripts/split_real.py --config configs/sft/real.yaml` | Drops near copies of test messages, holds out the real dev set, drops near copies of dev messages, caps each intent at 500 | seconds |
| `uv run python scripts/write_replies.py --config configs/sft/real.yaml`, then `verify_sft.py` with the same config | Qwen replies, then the blind label judge | ~30 min, ~6 min |
| `uv run python scripts/check_replies.py --config configs/sft/reply_check.yaml check.scope=all check.data_dir=data/sft/sft-real-replies "check.splits=[verified]" check.out_dir=data/sft/reply-check-real` | Qwen proofreads every reply | ~12 min |
| `uv run python scripts/generate_sft.py --config configs/sft/style_en.yaml`, then `verify_sft.py` and `check_replies.py` the same way (`check.data_dir=data/sft/sft-style-en`, `check.out_dir=data/sft/reply-check-style`) | English style top-up | ~8 min, ~2 min, ~4 min |
| `uv run python scripts/assemble_sft_v4.py --config configs/sft/v4.yaml` | Builds `data/sft/final-v4` | seconds |
| `uv run python scripts/sft.py --config configs/sft/train_real.yaml train.lr=3.0e-4 run_name=sft-r3-second-lr3e-4 train.init=artifacts/checkpoints/pretrain-second/model.pt` | One SFT run, evaluated on the SFT validation split and the real dev set after each epoch (four runs: two bases × lr 3e-4 and 1e-3) | ~10 min |
| `uv run python scripts/evaluate_dev.py --config configs/sft/report_real.yaml` | The Phase 5 model on the real dev set, for comparison | ~1 min |
| `uv run python scripts/real_round_report.py --config configs/sft/report_real.yaml` | Applies the D-033 rule, writes the results page and `artifacts/checkpoints/sft-selected-5b.json` | seconds |
| the three Phase 6 commands with `--config configs/eval/phase6b.yaml` | Scores the chosen model on the human test set with the D-032 protocol | ~5 min |

### Phase 6: evaluation

The protocol (D-032) was committed before the test set was scored. The scripts refuse a test
file whose SHA-256 has changed.

| Command | What it does | Time on the dev machine |
|---|---|---|
| `uv run python scripts/evaluate_test.py --config configs/eval/phase6.yaml` | Scores DanaLM and CAMeLBERT on the human test set; latency and memory on CPU and GPU | ~2 min |
| `uv run python scripts/evaluate_teacher.py --config configs/eval/phase6.yaml` | Qwen zero-shot intents and Qwen's judgement of DanaLM's replies (starts the teacher) | ~4 min |
| `uv run python scripts/phase6_report.py --config configs/eval/phase6.yaml` | Results page: intervals, the D-030 bars, errors, all replies | seconds |

### Phase 7: quantization and deployment

The rules (D-034) were committed before any quantized model was scored. Every measuring script
turns off Windows power throttling for its own process, since throttling otherwise distorts CPU
timings. No system setting is changed.

| Command | What it does | Time on the dev machine |
|---|---|---|
| `uv run python scripts/export_onnx.py --config configs/deploy/export.yaml` | One ONNX step graph with a KV cache (float32), plus a parity check against PyTorch | ~1 min |
| `uv run python scripts/quantize_onnx.py --config configs/deploy/export.yaml` | The INT8 and INT4 variants in `artifacts/deploy/` | ~1 min |
| `uv run python scripts/evaluate_variants.py --config configs/deploy/phase7.yaml` | Every system on the development sets; thresholds; the D-034 choice (`artifacts/deploy/selected.json`) | ~30 min |
| `uv run python scripts/evaluate_variants.py --config configs/deploy/phase7.yaml phase7.split=test` | The human test set, once per system, after the choice | ~3 min |
| `uv run python scripts/evaluate_teacher.py --config configs/deploy/phase7_judge.yaml` | Qwen judges the deployed variant's test replies (starts the teacher) | ~3 min |
| `uv run python scripts/benchmark_latency.py --config configs/deploy/phase7.yaml` | Latency and memory, batch 1 on the CPU, each system in its own process | ~20 min |
| `uv run python scripts/phase7_report.py --config configs/deploy/phase7.yaml` | Results page | seconds |
| `uv run python scripts/ci_smoke.py` | The CI check: a tiny model through export, INT8/INT4, evaluation and the predictor | ~1 min |

Serving, without PyTorch:

```bash
DANALM_MODEL_DIR=artifacts/deploy/int4 uv run uvicorn danalm.serve.app:app --port 8000
curl -s localhost:8000/predict -H "content-type: application/json" -d '{"message": "my card got stuck in the ATM"}'
docker build -t danalm-serve .
docker run --rm -p 8000:8000 -v "$PWD/artifacts/deploy/int4:/model:ro" danalm-serve
uv run --group demo python scripts/build_space.py --config configs/deploy/phase7.yaml
```

`POST /predict` returns the intent, the reply, the confidence and a route: `on_device` or
`escalate`. It also returns the PII-masked message, which is the only text that should leave the
device. `build_space.py` assembles the Hugging Face Space in `artifacts/space`, where
`python app.py` runs the demo locally.

## Repository layout

```text
configs/       YAML configs; base.yaml is inherited by all of them
docs/          project brief, decision log, data ledger, test-set guide,
               results/ (measured results per phase)
scripts/       command-line entry points
src/danalm/    library: config, seeding, run tracking, data pipeline and token shards,
               HF sampler, labelled public datasets, ledger, dialect and reply checks,
               teacher server and client (with resumable answer logs), tokenizer, model
               (with its KV-cache step, ONNX export and quantization), training loop, SFT
               format and evaluation; infer/ and serve/ run without PyTorch
deploy/        pinned requirements of the serving image and the demo
space/         the Gradio demo for Hugging Face Spaces
tests/         unit tests and tiny synthetic fixtures
.github/       CI: lint, tests, the smoke evaluation and the service image
data/ runs/ artifacts/   gitignored: datasets, run records, tokenizers and checkpoints
```

## Reproducibility

Each run writes `runs/<run_name>-<timestamp>/` containing:

- `config.yaml`: the fully resolved config
- `meta.json`: git commit, dirty flag, seed, and Python/PyTorch/CUDA/GPU versions
- `git_diff.patch`: only if there were uncommitted changes

To reproduce a run, check out its commit, apply the patch if there is one, and rerun the
recorded command with `--config runs/<run>/config.yaml`.

## License

- **Code:** [Apache-2.0](LICENSE).
- **Data:** each source keeps its own licence; see [docs/DATA_LEDGER.md](docs/DATA_LEDGER.md).
  No data is stored in this repository.
- **Model weights:** not published yet. The licence will be chosen before release, in line with
  the training-data licences in [docs/DATA_LEDGER.md](docs/DATA_LEDGER.md).
