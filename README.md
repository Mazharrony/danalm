# DanaLM (دانة)

A small (~30–60M parameter) decoder-only language model trained from scratch for UAE customer
service in Gulf Arabic, English, Arabizi and mixed text. For each customer message it returns
strict JSON, `{"intent": "...", "reply": "..."}`. It is meant to run quantized on a CPU or phone:
it answers the easy majority of messages on-device and hands the rest to a bigger model or a human.

**Status:** Phase 1 (tokenizer) is done. `danalm-v1` is a 16,384-token byte-level BPE for
Arabic, English and Arabizi. Against Qwen3.5's 248k-token tokenizer, it needs 14% fewer tokens
on Gulf Arabic and 8% fewer on MSA (see
[docs/results/phase1_tokenizer.md](docs/results/phase1_tokenizer.md)).
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

## Repository layout

```text
configs/       YAML configs; base.yaml is inherited by all of them
docs/          project brief, decision log, data ledger, results/ (measured results per phase)
scripts/       command-line entry points
src/danalm/    library: config, seeding, run tracking, data pipeline, HF sampler, ledger,
               teacher client, tokenizer
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
