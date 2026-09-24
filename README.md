# DanaLM (دانة)

A small (~30–60M parameter) decoder-only language model trained from scratch for UAE customer
service in Gulf Arabic, English, Arabizi and mixed text. For each customer message it returns
strict JSON, `{"intent": "...", "reply": "..."}`. It is meant to run quantized on a CPU or phone:
it answers the easy majority of messages on-device and hands the rest to a bigger model or a human.

**Status:** Phase 3 (model) is done, and Phase 4 (pretraining) is next.

- **Tokenizer (Phase 1):** `danalm-v1` is a 16,384-token byte-level BPE for Arabic, English and
  Arabizi. Against Qwen3.5's 248k-token tokenizer, it needs 14% fewer tokens on Gulf Arabic and
  8% fewer on MSA ([results](docs/results/phase1_tokenizer.md)).
- **Pretraining corpus (Phase 2a):** 1.495B tokens from 10 openly licensed sources: 64% Arabic
  (including 300M tokens of Najdi dialect) and 36% English. It is cleaned, deduplicated,
  tokenized, and checked by decoding sample documents back to their text
  ([results](docs/results/phase2_data.md)).
- **SFT data (Phase 2b):** 18,547 customer-service examples covering 21 intents, in Gulf Arabic,
  English, Arabizi and mixed text.
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
- **Open:**
  - The human test set: 65 English candidates from the Banking77 and CLINC150 test splits wait
    for review. The Arabic parts need a native Gulf Arabic speaker (D-025).

The plan is in [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md), and the reasons behind every
choice are in [docs/DECISIONS.md](docs/DECISIONS.md). Every data source, with its licence and
token count, is in [docs/DATA_LEDGER.md](docs/DATA_LEDGER.md).

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

## Repository layout

```text
configs/       YAML configs; base.yaml is inherited by all of them
docs/          project brief, decision log, data ledger, test-set guide,
               results/ (measured results per phase)
scripts/       command-line entry points
src/danalm/    library: config, seeding, run tracking, data pipeline and token shards,
               HF sampler, labelled public datasets, ledger, dialect and reply checks,
               teacher server and client (with resumable answer logs), tokenizer, model
tests/         unit tests and tiny synthetic fixtures
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
- **Model weights:** licence to be decided after the Phase 2 review of the training-data licences.
