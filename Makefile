# DanaLM task runner. Each target is a thin wrapper around a `uv run ...` command, so you can
# also copy the command and run it directly (make is a convenience, not a requirement).
# Windows: install GNU make once with `winget install ezwinports.make`.
# Extra config overrides go in ARGS, e.g.  make check-env ARGS="tracking.mode=offline"

.PHONY: setup lint test check check-env data-smoke

setup:  ## create .venv exactly from uv.lock and install the git hooks
	uv sync --locked
	uv run pre-commit install

lint:  ## every pre-commit hook on every file (ruff, black, whitespace, lock check, data guard)
	uv run pre-commit run --all-files

test:  ## unit tests
	uv run pytest

check: lint test  ## what CI will run

check-env:  ## record Python/torch/CUDA/GPU info; measure bf16 TFLOPS and attention kernels
	uv run python scripts/check_env.py --config configs/check_env.yaml $(ARGS)

data-smoke:  ## run the data pipeline on the tiny fixture in tests/fixtures -> data/smoke/
	uv run python scripts/prepare_data.py --config configs/data/smoke.yaml $(ARGS)
