# DanaLM task runner. Each target is a thin wrapper around a `uv run ...` command, so you can
# also copy the command and run it directly (make is a convenience, not a requirement).
# Windows: install GNU make once with `winget install ezwinports.make`.
# Extra config overrides go in ARGS, e.g.  make check-env ARGS="tracking.mode=offline"

# UTF-8 mode: Windows would otherwise write logs in cp1252 and choke on Arabic text.
export PYTHONUTF8 = 1

.PHONY: setup lint test check check-env data-smoke tokenizer-data synthetic-eval tokenizers tokenizer-eval tokenizer-final pretrain-data

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

# ---------------------------------------------------------------- Phase 1: tokenizer
tokenizer-data:  ## download (~0.5 GB) and clean the tokenizer corpus and the real eval sets
	uv run python scripts/sample_corpus.py --config configs/data/tokenizer_corpus.yaml $(ARGS)
	uv run python scripts/sample_corpus.py --config configs/data/tokenizer_eval.yaml $(ARGS)
	uv run python scripts/prepare_data.py --config configs/data/tokenizer_corpus.yaml $(ARGS)

synthetic-eval:  ## synthetic Gulf / Arabizi / mixed eval messages from the local teacher (GPU)
	uv run python scripts/generate_synthetic.py --config configs/synthetic/fertility_eval.yaml $(ARGS)

tokenizers:  ## train the 6 candidates: vocab 16k / 24k / 32k x pre-tokenizer standard / arabizi
	uv run python scripts/train_tokenizer.py --config configs/tokenizer/bpe.yaml tokenizer.vocab_size=16384 tokenizer.pretokenizer=standard
	uv run python scripts/train_tokenizer.py --config configs/tokenizer/bpe.yaml tokenizer.vocab_size=16384 tokenizer.pretokenizer=arabizi
	uv run python scripts/train_tokenizer.py --config configs/tokenizer/bpe.yaml tokenizer.vocab_size=24576 tokenizer.pretokenizer=standard
	uv run python scripts/train_tokenizer.py --config configs/tokenizer/bpe.yaml tokenizer.vocab_size=24576 tokenizer.pretokenizer=arabizi
	uv run python scripts/train_tokenizer.py --config configs/tokenizer/bpe.yaml tokenizer.vocab_size=32768 tokenizer.pretokenizer=standard
	uv run python scripts/train_tokenizer.py --config configs/tokenizer/bpe.yaml tokenizer.vocab_size=32768 tokenizer.pretokenizer=arabizi

tokenizer-eval:  ## fertility of all candidates vs Jais and Qwen3.5 -> docs/results/phase1_tokenizer.md
	uv run python scripts/eval_tokenizers.py --config configs/tokenizer/eval.yaml $(ARGS)

tokenizer-final:  ## train the chosen tokenizer (danalm-v1) and fill token counts into the ledger
	uv run python scripts/train_tokenizer.py --config configs/tokenizer/danalm_v1.yaml $(ARGS)
	uv run python scripts/update_ledger.py --config configs/ledger/phase1.yaml $(ARGS)

# ---------------------------------------------------------------- Phase 2: data
pretrain-data:  ## download (~2-3 GB of text columns), clean and tokenize the ~1.5B-token corpus
	uv run python scripts/sample_corpus.py --config configs/data/pretrain_corpus.yaml $(ARGS)
	uv run python scripts/prepare_data.py --config configs/data/pretrain_corpus.yaml $(ARGS)
	uv run python scripts/tokenize_corpus.py --config configs/data/pretrain_corpus.yaml $(ARGS)
