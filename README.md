# DanaLM (دانة)

A small (~30–60M parameter) decoder-only language model trained from scratch for UAE customer
service in Gulf Arabic, English, Arabizi and mixed text. For each customer message it returns
strict JSON, `{"intent": "...", "reply": "..."}`. It is meant to run quantized on a CPU or phone:
it answers the easy majority of messages on-device and hands the rest to a bigger model or a human.

**Status:** Phase 0 (setup and guardrails) is done.
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

## Everyday commands

| make | Plain command | What it does |
|---|---|---|
| `make test` | `uv run pytest` | Unit tests |
| `make lint` | `uv run pre-commit run --all-files` | ruff, black, whitespace, lock check, data guards |
| `make check-env` | `uv run python scripts/check_env.py --config configs/check_env.yaml` | Records the environment and measures GPU throughput |
| `make data-smoke` | `uv run python scripts/prepare_data.py --config configs/data/smoke.yaml` | Runs the data pipeline on the test fixture |

Every script takes `--config <file.yaml>` plus `key=value` overrides, e.g. `seed=1 tracking.mode=offline`.

## Repository layout

```text
configs/       YAML configs; base.yaml is inherited by all of them
docs/          project brief and decision log
scripts/       command-line entry points
src/danalm/    library: config loader, seeding, run tracking, data pipeline
tests/         unit tests and tiny synthetic fixtures
data/ runs/ checkpoints/   gitignored: datasets, run records, model weights
```

## Reproducibility

Each run writes `runs/<run_name>-<timestamp>/` containing:

- `config.yaml`: the fully resolved config
- `meta.json`: git commit, dirty flag, seed, and Python/PyTorch/CUDA/GPU versions
- `git_diff.patch`: only if there were uncommitted changes

To reproduce a run, check out its commit, apply the patch if there is one, and rerun the
recorded command with `--config runs/<run>/config.yaml`.
