# Decisions

Every decision that affects results or reproducibility: what we chose, why, the alternatives we
considered, and when to revisit it. Newest at the bottom. The project plan is in
[PROJECT_BRIEF.md](PROJECT_BRIEF.md).

| ID | Phase | Decision |
|----|-------|----------|
| D-001 | 0 | Run natively on Windows 11 with uv (not WSL2) |
| D-002 | 0 | Python 3.12, managed by uv |
| D-003 | 0 | `pyproject.toml` + `uv.lock` instead of `requirements.txt`; PyTorch 2.14.0 (CUDA 13.0 on Windows, CPU on Linux) |
| D-004 | 0 | Weights & Biases for experiment tracking |
| D-005 | 0 | ruff + black via pre-commit, with guards against committing data and models |

---

## D-001 · Phase 0 · Run natively on Windows 11 with uv

- **Decision:** Develop and train on native Windows 11 (RTX 4070 12 GB, i9-13900K, 64 GB RAM).
  All code stays cross-platform, and CI and Docker run it on Linux.
- **Why:** It works today with no extra setup, the repo and data stay on the internal NVMe drive,
  and PyTorch CUDA wheels, bitsandbytes and llama.cpp all support Windows.
- **Alternatives:** WSL2 Ubuntu has the best ML tooling (official Triton for `torch.compile`,
  vLLM, make). It needs an Ubuntu install, data moved into the WSL filesystem for I/O speed, and
  every command routed through `wsl`.
- **Revisit if:** Phase 3 shows that `torch.compile` does not work with the community
  `triton-windows` package *and* the speed loss matters for the pretraining budget, or the
  Phase 2 teacher needs vLLM throughput.

## D-002 · Phase 0 · Python 3.12, managed by uv

- **Decision:** Python 3.12, pinned in `.python-version` and `requires-python`. uv installs it
  for this project only; the system Python (3.14) is untouched.
- **Why:** 3.12 has mature wheels for the whole planned stack (PyTorch, bitsandbytes,
  onnxruntime, gradio), including the GPU and quantization libraries that are slowest to support
  new Python versions.
- **Alternatives:** 3.11 (just as safe, slightly older); 3.14 (newest, highest risk of missing
  wheels in Phases 2 and 7).
- **Revisit if:** a library we need drops 3.12.

## D-003 · Phase 0 · `pyproject.toml` + `uv.lock`; PyTorch 2.14.0

- **Decision:** Dependencies are declared in `pyproject.toml` and pinned exactly, with hashes, in
  `uv.lock` (`uv sync --locked` reproduces the environment). PyTorch 2.14.0 (released
  2026-09-02) comes from the PyTorch index: the CUDA 13.0 build on Windows, the CPU build on Linux.
- **Why:** One lock file pins every transitive package for both platforms, which pip
  `requirements.txt` files cannot do across platforms. PyPI's Windows torch wheel is CPU-only, so
  the index is mandatory on Windows. Linux gets the CPU build because it only runs CI, Docker and
  CPU inference, which saves several GB. CUDA 13.0 is the current build line for torch 2.14; the
  driver (591.86) supports up to CUDA 13.1. The matching `triton-windows` 3.8 exists if we want
  `torch.compile`.
- **Alternatives:** `requirements.txt` + pip. A `uv export` does not carry the PyTorch index URL,
  so `pip install` cannot find `torch==2.14.0+cu130`. Also conda, and poetry.
- **Revisit if:** Phase 7 needs a pip-only install (e.g. Hugging Face Spaces). We will generate a
  small CPU-only requirements file for that target.

## D-004 · Phase 0 · Weights & Biases for experiment tracking

- **Decision:** W&B (free personal tier). `tracking.mode` in the config is `online`, `offline`
  (upload later with `wandb sync`) or `disabled` (tests and CI).
- **Why:** It has the best UI for comparing LLM training curves, and it logs GPU and VRAM system
  metrics automatically, which the brief asks us to report. Public report links can go in the
  README and CV.
- **Alternatives:** MLflow (fully local, no account, common in enterprise; sharing needs
  screenshots or a hosted UI). TensorBoard (local only, weak run comparison).
- **Revisit if:** we need to work fully offline for a long time. MLflow would slot into
  `danalm.utils.run.start_run`, which is the only place that touches W&B.

## D-005 · Phase 0 · ruff + black via pre-commit, with data guards

- **Decision:** ruff for linting and import order, black for formatting, line length 100. Both
  run in pre-commit through `uv run`, so hook versions always match `uv.lock`. Other hooks block
  files over 500 KB, model and data formats (`.pt`, `.safetensors`, `.bin`, `.gguf`, `.onnx`,
  `.npy`, `.parquet`, ...) and private keys, and check that `uv.lock` matches `pyproject.toml`.
  `docs/PROJECT_BRIEF.md` is excluded from the fixers so it stays verbatim.
- **Why:** The brief's rules say never commit data or checkpoints and keep code quality high;
  hooks enforce both on every commit instead of relying on memory.
- **Alternatives:** `ruff format` instead of black (near-identical output, one tool fewer; the
  brief names black). Hooks pinned to their own repos (can drift from the venv's versions).
