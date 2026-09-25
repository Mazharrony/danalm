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
| D-006 | 0 | Plain YAML configs with a small loader: `base:` inheritance, `${...}` references, strict overrides |
| D-007 | 0 | Reproducibility contract: every run records config + seed + git commit (+ diff) + environment |
| D-008 | 0 | Seeding: always seed everything; deterministic CUDA kernels opt-in per config |
| D-009 | 0 | Adopt `data_pipeline.py` as `danalm.data.pipeline`, driven by YAML config |
| D-010 | 0 | PII masking: add cards, IBANs, landlines, long numbers; Arabic-Indic digits -> ASCII; never alter amounts |
| D-011 | 0 | Language tags: fix clear bugs now; measure accuracy against human labels in Phase 2 |
| D-012 | 0 | Data policy: only free, legally usable data, licence checked at the source, every collection logged in DATA_LEDGER.md |
| D-013 | 0 | Teacher: the owner's local Qwen3.5-9B (Q4_K_M GGUF, llama.cpp server), Apache-2.0 |
| D-014 | 1 | Training text keeps alef and alef-maksura spelling (`unify_alef: false`); diacritics still stripped |
| D-015 | 1 | Tokenizer design: byte-level BPE, Arabic-aware pre-tokenizer, fixed control tokens, atomic PII placeholders |
| D-016 | 1 | Fertility eval sets: real where legally available, Qwen-generated (labelled synthetic) for Emirati, Arabizi, mixed |
| D-017 | 1 | Tokenizer `danalm-v1`: 16,384-token byte-level BPE, `standard` pre-tokenizer, chosen by pre-declared rules |
| D-018 | 1 | Text written or edited by closed AI tools: evaluation only, never training |
| D-019 | 2 | Pretraining corpus: ~1.5B danalm-v1 tokens, ~63% Arabic (incl. 300M Najdi dialect), ~35% English, ~1% domain |
| D-020 | 2 | Streaming data pipeline (hash-based val split) and uint16 token shards; constant memory at any corpus size |
| D-021 | 2 | Intent taxonomy: 21 intents (banking 7, telecom 5, delivery 6, cross-domain 3 incl. `other` and `handoff_to_human`) |
| D-022 | 2 | SFT teacher and judge: Qwen3.5-35B-A3B (won the pre-declared pilot rule); stricter reply and dialect filters |
| D-023 | 2 | Arabizi messages get Gulf Arabic replies in Arabic script (a change to the brief's "same language" rule) |
| D-024 | 2 | `other`: real out-of-scope messages plus a code-mixed top-up; one final judge pass with the final intent descriptions |
| D-025 | 2 | English part of the human test set: Banking77 and CLINC150 test splits, rule-mapped, overlap-checked, checked by a person |
| D-026 | 3 | Model: Llama-style decoder, 62.1M parameters (large), chosen by a rule fixed before the sanity runs |
| D-027 | 6 | Baseline classifier: CAMeLBERT-mix (Apache-2.0) instead of MARBERT/AraBERT |
| D-028 | 4 | Pretraining: one pass, 262k tokens/step, cosine LR peak 2e-3 (chosen by a pilot rule), bf16, torch.compile, checkpoints with resume |

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

## D-006 · Phase 0 · Plain YAML configs with a small loader

- **Decision:** Every setting lives in `configs/*.yaml`. `danalm.config` (about 100 lines) supports
  `base:` inheritance, `${dotted.key}` references and `key=value` CLI overrides. Overrides must
  name an existing key, so typos fail instead of being silently ignored. Component code gets its
  values from the config and has no defaults for hyperparameters.
- **Why:** The brief bans hard-coded hyperparameters and paths (BanglaLM's main problem). A small
  loader is easy to read, test and debug. Relative paths are relative to the repo root, where
  every command runs from.
- **Alternatives:** Hydra (powerful, but changes the working directory and adds a lot of magic);
  OmegaConf (close to what we need, but slow release cadence and another dependency); argparse
  defaults (these are exactly the hard-coded settings we want to avoid).
- **Revisit if:** configs grow to need typed schema validation beyond dataclass construction.

## D-007 · Phase 0 · Reproducibility contract

- **Decision:** `danalm.utils.run.start_run` gives every run the folder `runs/<run_name>-<timestamp>/`
  with `config.yaml` (fully resolved), `meta.json` (git commit, dirty flag, untracked files,
  command, seed, Python/PyTorch/CUDA/GPU versions) and, when the tree has uncommitted changes,
  `git_diff.patch`. The same config and metadata go to W&B. Data outputs get the same
  `config.yaml` + `meta.json` next to them.
- **Why:** "Every run must be reproducible from config + seed + git commit." Saving the diff
  keeps even a dirty-tree run reproducible, and the warning nudges us to commit before long
  runs.
- **Alternatives:** Refusing to run on a dirty tree (too strict while iterating); relying only on
  W&B's own git capture (it lives outside the repo and is lost if W&B is off).

## D-008 · Phase 0 · Seeding

- **Decision:** `set_seed(seed, deterministic)` seeds Python, NumPy and PyTorch (CPU and every
  GPU). `deterministic: false` by default in `configs/base.yaml`. When true, it also turns on
  deterministic PyTorch/cuDNN kernels and sets `CUBLAS_WORKSPACE_CONFIG`.
- **Why:** Seeding alone fixes initialization, sampling and data order, which is what matters for
  comparing runs. Bitwise determinism slows training and is only needed when debugging a
  divergence.
- **Alternatives:** Always deterministic (slower for no benefit in normal runs).

## D-009 · Phase 0 · Adopt `data_pipeline.py` as `danalm.data.pipeline`

- **Decision:** The existing pipeline lives in `src/danalm/data/pipeline.py`. It was imported
  unchanged (commit `833ed32`) and then refactored. Its settings come from the `data:` section of
  a YAML config (`PipelineConfig`, no defaults), including values that used to be hard-coded
  (MinHash permutations, shingle size, quality thresholds). The CLI is
  `scripts/prepare_data.py`, and every output folder gets `config.yaml` + `meta.json`.
- **Why:** The brief says to reuse it, and the rules ban hard-coded settings. The refactor keeps
  behavior identical: on the test fixture, the original script and the refactored one produce
  byte-identical `train.jsonl`, `val.jsonl` and `stats.json`.
- **Alternatives:** Keep it as a standalone argparse script (settings would live in shell
  history instead of versioned configs).

## D-010 · Phase 0 · PII masking: wider coverage, number-safe normalization

- **Decision:** `mask_pii` also masks card numbers (`<CARD>`), UAE IBANs (`<IBAN>`), UAE
  landlines, and any leftover run of 9+ digits (`<NUM>`, for account and reference numbers). It
  converts Arabic-Indic digits (٠-٩, ۰-۹) to ASCII before matching, and uses digit boundaries so
  numbers glued to Arabic letters are caught. The repeat cap no longer touches digits.
  Placeholders: `<URL> <EMAIL> <EID> <IBAN> <CARD> <PHONE> <NUM>`.
- **Why:** New tests showed that before the fix `1000000 AED` became `1000 AED` (amounts were
  silently changed); phone numbers and Emirates IDs typed in Arabic-Indic digits were not masked;
  and card numbers and IBANs were never masked, although they are likely in banking messages. (A
  code comment mentioned "generic long digit runs", but that was never implemented.) The brief
  says "Mask PII" with no exceptions.
- **Trade-offs:** Harmless 9+ digit numbers (e.g. tracking numbers) are masked too. Amounts,
  dates, times, flight numbers and short order numbers are kept (tested). The model only ever
  sees ASCII digits, so inference must run the same `normalize()`.
- **Known gap:** Regexes cannot catch personal names or street addresses. That matters for the
  human-written test set and any real messages; a small NER pass can be added later if needed.
- **Alternatives:** A learned PII/NER model (heavier and needs labels).
- **Revisit if:** Phase 2 audits of real samples show misses or over-masking.

## D-011 · Phase 0 · Language tags: fix clear bugs now, measure accuracy in Phase 2

- **Decision:** `detect_lang` ignores PII placeholders and no longer counts English
  number+suffix tokens (2nd, 5pm, 2FA, 3DS, 10GB) as Arabizi. No other heuristic tuning yet.
- **Why:** Placeholders are Latin letters, so an Arabic message with a masked phone number was
  tagged `mixed`, and "my 2nd card was blocked at 5pm" was tagged `arabizi`. Both were clear
  bugs. Tuning the heuristic further without labelled data would be guesswork.
- **Known limit:** Arabizi without digits ("shlonak, abi agayer el card") is tagged `en`. The
  Phase 6 per-language breakdown will use human-verified labels on the test set, not this
  heuristic, and Phase 2 will measure the heuristic against those labels.
- **Update (2026-09-24, Phase 2, approved by the owner):**
  - **Old rule:** a single digit-letter token made Latin text Arabizi. That tagged 3.8% of
    FineWeb-Edu documents and 2.1% of English Wikipedia documents as Arabizi, from tokens like
    "5ES" and "7up".
  - **New rule:** Latin text is Arabizi when at least 10% of its words are evidence. Evidence is
    either a digit-letter word or one of 40 common Gulf Arabizi words, each at most 1.5 per
    million words in 23.5M words of English web text.
  - **Measured** (accuracy = share tagged correctly):

    | Labelled set | Old rule | New rule |
    |---|---:|---:|
    | English web documents (30,000) | 96.12% | 100.00% |
    | English customer-service texts: Bitext and CLINC (20,000) | 99.98% | 100.00% |
    | English SFT messages (5,807) | 100% | 100% |
    | Arabizi SFT messages (6,080) | 91.30% | 99.90% |

    Thresholds of 10%, 15% and 20%, and a minimum of 2 evidence words, were compared. 10% with
    one evidence word was best on every set. The Arabizi labels come from the synthetic SFT
    data, so they are only as good as the teacher's Arabizi. The human test set will measure
    the rule again.

## D-012 · Phase 0 · Data policy and ledger

- **Decision:** The project owner is a solo developer with no private corpus, so we use only
  free, legally usable data. Every source's licence is checked on the primary source (dataset or
  model card, licence file) before download. Permissive licences are preferred; share-alike
  licences are flagged; non-commercial, unclear or missing licences are rejected, and so is
  redistributed social-media text. [DATA_LEDGER.md](DATA_LEDGER.md) records every collection:
  source, revision, licence, date, filters and counts (documents, bytes, words and, from Phase 1
  on, tokens).
- **Why:** The owner's standing rule. It also makes the model card's data section and licence
  honest and checkable.
- **Consequence:** Open Gulf Arabic text is scarce. The closest clean source is FineWeb-2 Najdi
  (`ars_Arab`); Emirati Gulf Arabic and Arabizi will mostly come from the teacher (D-013),
  labelled as synthetic.

## D-013 · Phase 0 · Teacher model: local Qwen3.5-9B

- **Decision:** The Phase 2b teacher is the owner's local Qwen3.5-9B: `unsloth/Qwen3.5-9B-GGUF`
  Q4_K_M (HF commit `3885219b`), served by llama.cpp `llama-server` at `127.0.0.1:8080` with an
  OpenAI-compatible API (setup in `D:\All Source Codes\queen 3.5`).
- **Why:** Apache-2.0 (verified on the official card and LICENSE), so there are no limits on
  training with its outputs. It supports 201 languages and dialects, fits the GPU at about
  8.6 GB, and the owner measured about 75 tok/s single-stream.
- **Plan:** Use non-thinking mode (`enable_thinking: false`) for bulk generation, since thinking
  multiplies the tokens per sample. Compare it with thinking mode on a small pilot first. The
  server must be stopped before any student GPU job (brief rule: never teacher and student on
  the GPU together); training scripts will refuse to start while it is up.
- **Alternatives:** Jais (Arabic-centric, Apache-2.0) as a second teacher if a Gulf-Arabic
  quality check of Qwen's output is poor.
- **Revisit if:** a native-speaker review of the pilot finds Qwen's Gulf Arabic or Arabizi
  unnatural.

## D-014 · Phase 1 · Keep alef spelling in training text

- **Decision:** From Phase 1 on, every training-data config sets `unify_alef: false`: أ/إ/آ/ٱ and
  ى stay as written. Diacritics are still stripped (`strip_diacritics: true`).
- **Why:** DanaLM *writes* replies. Unifying would teach it non-standard spelling (إلى → الي,
  أنا → انا), which is fine for a classifier but visible in generated text. Recommended in the
  Phase 0 report and approved with the owner's "go" (2026-09-23).
- **Alternatives:** Unify (fewer surface forms, slightly better for pure classification).
- **Note:** `configs/data/smoke.yaml` keeps the original script's settings on purpose; it tests
  the imported behavior.

## D-015 · Phase 1 · Tokenizer design

- **Decision:**
  - **Model:** byte-level BPE with no normalizer inside the tokenizer: `normalize()` runs before
    it, at training and at inference. Every script (Arabic, English, emoji, anything unseen)
    round-trips losslessly, and no unknown token is needed.
  - **Pre-tokenizer:** the GPT-4/Llama-3 regex, extended with `\p{M}` so diacritics never split
    an Arabic word. It comes in two variants: `standard` (digits are always separate chunks of
    up to 3) and `arabizi` (a digit inside a word stays with it, as in "3andi", "ma3a", "sa7").
  - **Control tokens:** `<|endoftext|>` (id 0: end of text and document separator),
    `<|pad|>` (id 1), `<|user|>`, `<|assistant|>`, and 8 `<|reserved_i|>` for later phases.
  - **PII placeholders:** the 7 placeholders (`<URL> <EMAIL> <EID> <IBAN> <CARD> <PHONE>
    <NUM>`) are atomic, non-special tokens at the end of the vocabulary. They are cut from the
    training text so BPE never learns fragments of them, and they survive decoding.
  - **Vocabulary:** candidate sizes 16,384 / 24,576 / 32,768 (multiples of 128 for GPU
    matmuls).
  - **Training data:** a ~790M-character, licence-checked sample of the planned pretraining
    sources (see DATA_LEDGER).
- **Why:** Byte-level BPE is the standard for decoder LMs (GPT, Llama, Qwen), and a lossless
  round trip matters when the model has to reproduce exact JSON. Placeholders must be one token
  each, otherwise the model can mangle them in replies.
- **Alternatives:** SentencePiece unigram (common for multilingual models, but needs byte
  fallback and gives less control over pre-splitting); WordPiece (BERT-style, lossy).

## D-016 · Phase 1 · Fertility evaluation sets

- **Decision:** 10 held-out sets across the five varieties in the brief (MSA, Gulf Arabic,
  English, Arabizi, mixed). They use real text where a legal source exists: FineWeb-2,
  Wikipedia, the Najdi dialect set, MASSIVE ar-SA test and CLINC150 test, plus mixed sentences
  mined from web text. Emirati, Arabizi and mixed customer-service messages come from the Qwen
  teacher and are labelled "synthetic, not checked by a native speaker". None of these texts are
  used for training.
- **Teacher finding:** Asked plainly for Emirati Arabic, Qwen3.5-9B wrote mostly Egyptian-style
  dialect ("عايز", "ليه", "مش") and near-meaningless Arabizi. With prompts that list Gulf words
  to use and words to avoid, give example messages, and drop near-copies of the examples, the
  output became clearly Gulf-like but still contains some Egyptian forms.
- **Consequences:** For fertility this is acceptable, since tokenization cost barely depends on
  which dialect the text is in. For Phase 2b training data it is not: it needs a native-speaker
  review, and possibly a larger Qwen3.5 model (Apache-2.0 family) with CPU offload if the 9B
  model stays weak.

## D-017 · Phase 1 · Tokenizer `danalm-v1`: 16,384 tokens, `standard` pre-tokenizer

- **Decision:** `danalm-v1` (`configs/tokenizer/danalm_v1.yaml`) is a byte-level BPE with a
  vocabulary of 16,384 and the `standard` pre-tokenizer. It was trained in 41 s on the 335,328
  training documents (803M characters) of the Phase 1 corpus. `tokenizer.json` SHA-256 is
  `be583dda…`, and a retrain from the config reproduces it byte for byte. Full tables are in
  [results/phase1_tokenizer.md](results/phase1_tokenizer.md).
- **Why (measured on 10 held-out sets; the rules were fixed in the config before the run):**
  - **Pre-tokenizer:** the Arabizi-aware variant lowered Arabizi fertility by only 1.2–1.3%,
    below the 5% bar, so the rule picked `standard`. My hypothesis did not hold: the training
    corpus has almost no real Arabizi (all 2,451 documents tagged "arabizi" are false positives
    in English text, such as "7GHz" or "V2O5"), so BPE never learns Arabizi pieces, whatever the
    pre-splitting.
  - **Vocabulary size:** overall fertility falls from 1.813 (16k) to 1.714 (24k, −5.5%) and
    1.650 (32k, −9.0%). The bigger output layer costs more than that saves: training FLOPs per
    word at the reference shape (d_model=512, 12 layers) are 1.000 / 1.031 / 1.076. 16k also
    wins or ties within the 2% tolerance at d_model 640 and 768.
  - **Existing tokenizers:** against Qwen3.5 (248k vocab), danalm-v1 needs 14% fewer tokens on
    Gulf Arabic (1.651 vs 1.927) and 8% fewer on MSA (1.624 vs 1.763). It needs 22% more on
    English (1.485 vs 1.221) and 20% more on Arabizi (2.593 vs 2.163). Jais (85k vocab,
    Arabic-English) is best everywhere (overall 1.514), as you would expect with a vocabulary 5×
    larger. Its embedding table alone would be 43.5M parameters at d_model=512, most of our
    whole model budget.
- **Known weak spot:** Arabizi, at 2.6 tokens per word, is the most expensive variety for every
  tokenizer. The Arabizi and Emirati eval sets are teacher-generated and unverified.
- **Revisit if:** Phase 2 produces a substantial Arabizi corpus. Retraining takes about 45 s,
  and the tokenizer must be frozen before pretraining (Phase 4).
- **Update (2026-09-24, approved by the owner):** no tokenizer v2. Phase 2 did produce about 6k
  Arabizi messages, but they are teacher-written and noisy (D-022), so training a tokenizer on
  them would fit the teacher's quirks, not real Arabizi. danalm-v1 is frozen for pretraining.

## D-018 · Phase 1 · Text written or edited by closed AI tools: evaluation only

- **Decision:** Any text written or edited by a closed AI tool (ChatGPT, Claude, Gemini, ...) may
  be used for **evaluation only, never for training**. Its ledger entry says so, and Phase 2's
  train/test overlap check must keep it out of every training set. It never counts as
  native-speaker verification.
- **Why:** Many closed-model terms forbid using outputs to develop competing models (D-012). For
  a fertility measurement the risk is minimal, but training on such text would break the data
  policy. The Emirati fertility set is now Qwen's 300 messages with 57 lines corrected by an
  external AI tool (supplied by the owner). Clear non-Gulf dialect markers went from 43/300 lines
  to 10/300, and Gulf fertility for danalm-v1 moved 1.666 → 1.651. The decision in D-017 did not
  change. A second AI variant ("native Dubai") had more markers (14/300) plus apparently invented
  forms, and was not used. All versions are archived.
- **Alternatives:** Native-speaker correction, which is still the goal for Phase 2 data and the
  Phase 2c test set; or keeping Qwen's raw output as the eval set.

## D-019 · Phase 2 · Pretraining corpus: ~1.5B tokens, ~63% Arabic

- **Decision:** About 1.49B danalm-v1 tokens (`configs/data/pretrain_corpus.yaml`):

  | Source | Tokens | Share |
  |---|---:|---:|
  | FineWeb-2 MSA | 550M | 37% |
  | FineWeb-2 Najdi dialect | 300M | 20% |
  | Arabic Wikipedia | 100M | 7% |
  | FineWeb-Edu | 480M | 32% |
  | English Wikipedia | 50M | 3% |
  | Domain sets (Bitext ×3, CLINC150 train, MASSIVE ar-SA train) | ~13M | 1% |

  0.5% of documents are held out for validation (~7M tokens). Character targets come from the
  tokens-per-character ratios measured in Phase 1.
- **Why:**
  - **Size:** the brief asks for 1–2B tokens. 1.5B is about 30 tokens per parameter for a ~50M
    model. That is past the compute-optimal ~20, which suits a small model that has to be cheap
    at inference.
  - **Arabic share:** Arabic gets ~63% because it is the harder language (rich morphology,
    dialects) and the product's differentiator; small models learn English from less data.
  - **Najdi:** it is the only open Gulf-like dialect source. 300M tokens (about 70% of the
    subset) maximizes dialect exposure while keeping forum boilerplate to a fifth of the mix.
  - **Arabic Wikipedia:** capped at 100M tokens because much of it is bot-generated stubs.
  - **Domain sets:** tiny, but on-topic.
- **Alternatives:** 1B tokens (faster Phase 4, about compute-optimal); 2B tokens (a stronger
  model, ~30% longer training); a 50/50 Arabic/English mix.
- **Revisit if:** Phase 3 sizing changes the parameter count a lot, or the Phase 4 pilot shows
  validation loss still dropping steeply at the end.
- **Result (2026-09-23):** built as planned. 1,750,179 of 1,752,399 documents were kept
  (5.31B characters). The 2,220 dropped were 1,174 exact duplicates, 921 too short, 123 mostly
  non-letters and 2 repetitive. Text tokens per source: arb 550.4M, fineweb-edu 480.3M, ars
  299.9M, wiki-ar 100.3M, wiki-en 50.0M, domain sets 12.9M. That is 1,493.7M in total, and
  1,495.5M with one EOS per document: train 1,487.9M in 15 shards, val 7.6M. Every source
  landed within 0.4% of its target. The ledger has per-source raw and kept counts.
  `scripts/verify_shards.py` passed on both splits, checking EOS count = document count, ids
  below the vocabulary size, and 814 documents (14 of them crossing a shard boundary) decoding
  back to their text exactly. Wall time: sampling 40 min, cleaning 70 min, tokenization 16 min.
  Commits: sampled at `e962393`, cleaned at `d1bf704`, tokenized at `b702aeb`.

## D-020 · Phase 2 · Streaming pipeline and token shards

- **Decision:** Every data step now streams, so memory stays constant at any corpus size:
  - **Sampler:** writes each document to disk as it arrives, with one parquet row group in
    memory at a time.
  - **Cleaning:** streams documents into train/val JSONL, keeping only 16-byte hashes for exact
    deduplication. Validation documents are picked by a seeded hash of each document, not by a
    global shuffle.
  - **Tokenization:** a separate step (`scripts/tokenize_corpus.py`). It writes 100M-token
    uint16 shards with EOS after every document, plus `index.json` and per-source counts in the
    ledger.
- **Why:** At 1.5B tokens the old in-memory steps would need about 15 GB for cleaning and 40–50
  GB for `write_bin`'s Python list of token ids. The owner asked to keep RAM safe after the
  Phase 1 freeze. The Phase 4 trainer memory-maps the shards.
- **Consequence:** Phase 1's outputs (corpus split, danalm-v1) came from the old in-memory
  pipeline. Reproduce them from their recorded commits (up to `579e4b8`).
- **Measured on the 1.5B-token build:** peak resident memory was 0.5 GB for sampling, ~0.6 GB
  for cleaning and 1.6 GB for tokenization. All three ran next to the SFT teacher without
  trouble.

## D-021 · Phase 2 · Intent taxonomy: 21 intents

- **Decision:** `configs/sft/intents.yaml` has 21 intents:
  - **Banking (7):** card_not_working, lost_or_stolen_card, balance_or_statement,
    transfer_issue, unrecognized_transaction, fees_and_charges, loans_and_credit.
  - **Telecom (5):** bill_inquiry, network_or_internet_issue, plan_change, sim_or_number,
    roaming.
  - **Delivery, food and parcels (6):** order_status, missing_or_wrong_item, cancel_order,
    refund_request, change_delivery_details, failed_delivery.
  - **Cross-domain (3):** account_access, handoff_to_human, other.
- **Why:** These are the common first-contact reasons in the three domains, cut so that each
  one maps to a distinct next action. The brief asks for about 20 intents, including `other`
  and `handoff_to_human`. `account_access` (login, OTP, password) is cross-domain because every
  app has it. Each intent has a one-line description; the teacher, the judge and the human test
  set all use the same text.
- **Alternatives:** Fine-grained Banking77-style intents (77 of them; too many for a ~50M model
  and ~20k examples, with many near-synonyms). Domain-only labels (too coarse to route or answer).
- **Revisit if:** the judge's confusion matrix in the teacher pilot shows two intents constantly
  confused (merge them), or the human test set needs an intent that is missing.
- **Update (2026-09-23, after the pilot, before the full run):** `other` originally meant
  "outside banking, telecom and delivery support". That left in-domain questions none of the
  intents cover (exchange rates, opening an account) without a correct label. It now means
  "a greeting or thanks, or a request that no other intent covers". This matches the product
  logic: whatever the small model cannot handle is routed to a bigger model or a person.
- **Update (2026-09-24, after the full run):** seven descriptions were sharpened where the full
  run's judge confused intents:
  - `other` lost "or a person", which overlapped `handoff_to_human` (225 confusions).
  - `order_status` is now "while it is still on its way", and `failed_delivery` names the failed
    attempt (385 confusions between them).
  - `cancel_order` applies "even if they also complain that it is late" (126).
  - `refund_request` applies "whatever the reason" (91 confused with `failed_delivery`).
  - `plan_change` and `roaming` now say roaming packs are roaming (59).

  Every SFT candidate is judged again with the new text (D-024).

## D-022 · Phase 2 · SFT teacher and judge: Qwen3.5-35B-A3B; stricter filters

- **Decision:** The full SFT run uses Qwen3.5-35B-A3B (UD-Q4_K_XL, Apache-2.0, experts in RAM)
  as both the teacher and the blind judge, replacing the 9B.
- **Evidence** (pilot: 84 identical requests per teacher; full tables in
  [results/phase2_teacher_pilot.md](results/phase2_teacher_pilot.md)). The 35B made 71.0%
  usable examples (passing every filter, with the judge agreeing) against 61.6% for the 9B,
  +15%. Its non-Gulf dialect rate was 2.9% against 16.0% (−82%), and it produced 58 usable
  code-mixed messages to the 9B's 15. It is 3.5× slower (50 vs 173 tok/s). The rule was
  committed before the results (`0953ee2`), and both quality bars passed; the full run fits
  the 10 h budget (8.7 h estimated).
- **Caveat:** the judge is the 35B itself, so the +15% may carry some self-preference. The
  dialect-rate difference does not depend on the judge. No native speaker has reviewed the
  output yet.
- **Filters added after reading the pilot samples:**
  1. Replies that claim an action was done ("your card has been blocked", "تم ...") are
     rejected, and the prompt now says the assistant cannot act. These were 5–9% of usable
     replies, and they are dangerous because a customer might believe them.
  2. The dialect check now sees through attached و/ف ("ومفيش") and catches the "ما ...ش"
     negation.
  3. Formal MSA words in Gulf messages are rejected ("لماذا", "اريد"). "محتاج" is no longer
     flagged, since Gulf Arabic uses it too.
  4. An Arabizi message must get an Arabizi reply: the 9B answered 50% of them in English.

  Re-scored on the pilot, usable examples fall to 64.3% (35B) and 50.1% (9B), which widens the
  gap.
- **Alternatives:** Keep the 9B (twice as fast, but lower quality in exactly the Gulf and mixed
  varieties the product needs). Qwen3.5-27B dense (likely stronger, but about 5–10 tok/s with
  partial offload, so days for the full run).
- **Server tuning** (benchmark of 16 identical SFT requests, `runs/bench-*.log`):

  | Setting | Generation | Prompt processing | VRAM | Peak RAM |
  |---|---:|---:|---:|---:|
  | All experts in RAM, 4 slots, mmap (the pilot's setting) | 38 tok/s | ~44 tok/s | 3.1 GB | 22.6 GB |
  | Same without mmap | 42 tok/s | ~120 tok/s | 3.1 GB | 22.2 GB |
  | Experts of the last 13 of 40 layers on the GPU, 8 slots, no mmap | **68 tok/s** | — | 9.7 GB | 16.1 GB |

  The last setting is used for the full run (`configs/teacher/qwen35_35b_a3b.yaml`). The pilot
  itself ran with the first setting; its run folders keep that config.
- **Full run** (`configs/sft/full.yaml`): requests per variety are sized from the pilot's
  post-filter yields (english 36, gulf_arabic 38, arabizi 45, mixed 62 per intent). That is
  30,408 generated examples, for an expected ~18k usable. Estimated time is 4.6–6.1 h of
  generation plus about 1–1.3 h of judging.
- **Update (2026-09-23, during the full run):**
  - **Restarted for crash safety.** The first attempt kept every answer in memory until the
    end, and one failed request would have crashed it. After 31 minutes (380 of 3,801
    requests) it was stopped, with the owner's approval, and restarted with `b702aeb`:
    - every answer is appended and fsynced to `answers.jsonl` (or `judge_answers.jsonl`), and
      re-running the same config resumes;
    - failed requests are retried 3 times (30/60/90 s);
    - a run stops cleanly after 20 failures;
    - the chain retries each step up to 3 times.
    A fake-server test crashed the run after 40 of 105 requests, and the judge after 3 of 17
    batches. The resumed outputs were byte-identical to an uninterrupted run.
  - **Real memory use.** In steady state llama-server holds 22.6 GB resident and 32.6 GB
    committed, not the 16 GB of the benchmark. Its host prompt cache (`--cache-ram`, default
    8 GiB) fills up after a few hundred requests. The page file here is fixed at 20 GB
    (commit limit 83.75 GB), so the watchdog now also tracks commit headroom. It stops the
    resumable teacher jobs first (<8 GB available or <4 GB commit headroom), and everything
    else only at half those levels.
  - **Measured speed:** 12–13 requests per minute (~85 tok/s), so ~5 h of generation.
- **Result (2026-09-23; full tables in [results/phase2_data.md](results/phase2_data.md)):**
  - **Generation:** 3,801 requests with 0 failures, 30,288 examples parsed. The filters kept
    20,885 (69.0%): English 96.1%, Gulf Arabic 88.8%, Arabizi 80.7%, mixed 32.6%. For mixed,
    the teacher often wrote pure Arabic (4,380) or pure English (1,352) instead of switching
    languages. This took 4.6 h at 92 tok/s.
  - **Judging:** the judge agreed with 18,233 (87.3%): English 92.1%, Gulf Arabic 91.5%, mixed
    82.8%, Arabizi 81.3%. It took 31 minutes with no failed batches.
  - **Final SFT set:** 18,233 examples (train 17,286, val 947), 1.13M danalm-v1 tokens. Every
    intent has 745–1,027 examples, except `other` with 469.
  - Peak resident memory of llama-server: 22.6 GB while generating, 25.9 GB while judging.
    The watchdog never had to stop a job.
- **Weak spots found** (they are the open decisions of the Phase 2 report):
  - **`other`:** the judge agreed on only 51.9% of these, relabelling 225 as
    `handoff_to_human`. The new description ends "routed to a bigger model or a person", which
    overlaps with handoff. Only 13 of the 469 final `other` examples are mixed-language, because
    greetings rarely mix languages and 398 were rejected as pure Arabic.
  - **Delivery intents overlap:** 385 disagreements between `failed_delivery` and
    `order_status`, and 126 of `cancel_order` → `order_status`. These rows were dropped, so the
    set keeps the clear cases, but real customers will write the ambiguous ones.
  - **Arabizi replies:** 9.0% contain clear non-Gulf words (Moroccan "ghadi" 229 times, Egyptian
    "n3mel" 103 times), against 7.5% of Arabizi messages. Read by hand, most sampled replies are
    barely meaningful. The judge checks only the label, so it does not catch this.

## D-023 · Phase 2 · Arabizi messages get Gulf Arabic replies in Arabic script

- **Decision (approved by the owner, 2026-09-24):** the model keeps reading Arabizi, but it
  answers Arabizi messages in Gulf Arabic, in Arabic script. The brief's rule "reply in the same
  language" now reads: the same language, and Arabic script for Arabizi. English, Gulf Arabic
  and mixed messages are unchanged.
- **Why:**
  - The teacher's Arabizi replies were poor: 9.0% had clear non-Gulf words (Moroccan "ghadi"
    229 times, Egyptian "n3mel" 103 times), and most sampled replies were barely meaningful.
  - The judge checks only the intent label, so it never caught this. A model trained on those
    replies would answer Arabizi customers in garbled text.
  - UAE service channels usually answer in Arabic script or English, and the teacher writes
    good Gulf Arabic.
- **How:** `scripts/write_replies.py` (`configs/sft/arabizi_replies.yaml`) asked the teacher for
  new replies to all 6,080 Arabizi messages that passed the filters, 10 per request. It keeps the
  messages and labels. 5,747 replies passed the usual checks. The 333 rejected were 266 that
  claimed an action was done, 62 with no usable reply, and 5 with non-Gulf words. The run took
  about 55 minutes of teacher time in two parts: the watchdog stopped it once (D-025), and it
  resumed from its saved answers.
- **Alternatives:**
  - The teacher transliterates a good Gulf Arabic reply into Arabizi. This keeps the brief's
    rule but is unproven.
  - Keep the Arabizi replies (not acceptable, see above).
- **Revisit if:** a native speaker finds Arabic-script replies to Arabizi customers unnatural,
  or a teacher that writes good Arabizi becomes available.

## D-024 · Phase 2 · `other`: real out-of-scope data, a code-mixed top-up, one final judge pass

- **Decision (approved by the owner, 2026-09-24):**
  - **Real messages:** add 500 real out-of-scope messages from the TRAIN splits of CLINC150
    ("oos", 250 English) and MASSIVE ar-SA (250 Arabic, up to 5 per intent, food-ordering
    `takeaway_*` intents skipped because they overlap delivery). The teacher writes the replies
    (`configs/sft/other_real.yaml`).
  - **Code-mixed top-up:** 40 extra teacher requests for code-mixed greetings, thanks and small
    talk (`configs/sft/other_mixed.yaml`).
  - **Single final judge pass:** merge all sources and judge them in one pass with the final
    intent descriptions, then build the final set (`configs/sft/final.yaml`). The merge
    re-applies the D-011 language tagger and drops cross-source duplicates.
- **Why:** `other` had 51.9% judge agreement, 469 examples and only 13 code-mixed ones.
  `other` is the model's safety valve: anything it cannot handle goes to a bigger model or a
  person, so it must be learned well. Real out-of-scope questions are more varied than teacher
  ones. One judge pass with one taxonomy text keeps the labels consistent.
- **Note:** MASSIVE ar-SA turned out to be colloquial Saudi Arabic ("غير لون اللمبات... وخلها
  حمرا"), not MSA, so its variety is `saudi_arabic`.
- **Alternatives:** teacher-only top-up (less varied); keep `other` as it was.
- **Result (2026-09-24; [results/phase2_data.md](results/phase2_data.md)):**
  - **Top-ups:** 491 of the 500 real messages got a reply that passed the checks. The code-mixed
    top-up kept 96 of 320 examples; the teacher still wrote pure Arabic 176 times.
  - **Merge:** 21,108 candidates.
  - **Judge:** one pass agreed with 18,547 (87.9%), against 87.3% in the first version. By
    variety: English 92.7%, Gulf Arabic 91.9%, Saudi 98.2%, mixed 82.7%, Arabizi 81.6%.
  - **`other`:** agreement rose from 51.9% to 68.7%. Most of the 227 remaining
    `other → handoff_to_human` rows are first-run teacher messages that do ask for a person, so
    dropping them fixes the labels. The final set has 1,004 `other` examples (was 469), 60 of
    them code-mixed (was 13).
  - **Final set:** 18,547 examples (train 17,579, val 968), 1.12M danalm-v1 tokens. Every intent
    has 760–1,010 examples. The smallest intent × variety cell is failed_delivery × Arabizi
    (46).
- **Still open:**
  - `failed_delivery` ↔ `order_status` stays the main disagreement (379 rows, all dropped).
  - The judge checks labels only, so reply quality is unverified. For example, the Saudi
    message "ضم هذا" ("add this") got a greeting reply. A native speaker's review is still
    needed.

## D-025 · Phase 2 · English part of the human test set from public test splits

- **Decision (approved by the owner, 2026-09-24):**
  - **Sources:** English test-set candidates come from the TEST splits of Banking77 and
    CLINC150 (`configs/data/test_candidates_en.yaml`).
    - Banking77 is CC-BY-4.0, with the licence checked in the original PolyAI repository and
      pinned to commit `57ec275d`. Its documentation does not say who wrote the queries.
    - CLINC150 is CC-BY-3.0.
  - **Mapping:** labels map to our intents only where the match is clear (24 Banking77 and 20
    CLINC150 labels).
  - **Filtering:** candidates that overlap training data are dropped.
  - **Review:** a person checks every remaining candidate in a review sheet before
    `import_review.py` adds it to `data/test/test_set.jsonl`.
- **Result:** 160 candidates. The overlap check removed 48, mostly CLINC150 test sentences that
  nearly repeat its train split, which is part of the pretraining corpus. 65 went into the review
  sheet, up to 7 per intent. They cover 10 of the 21 intents: banking, `account_access`,
  `order_status` and `other`. The telecom and most delivery intents have no public English test
  data, so a person must write those.
- **Incident:** the first overlap check loaded the whole 1.5B-token corpus into memory. Commit
  headroom fell to 3.4 GB and the watchdog's first tier stopped the teacher (11:50), as designed.
  No data was lost, because the teacher resumed from its saved answers. `check_overlap.py` now
  streams its inputs and skips training texts longer than 4× the longest test text. With those
  changes it compares 590k texts in about 7 minutes.

## D-026 · Phase 3 · Model: Llama-style decoder, 62.1M parameters

- **Decision:** `configs/model/large.yaml` (`src/danalm/model/transformer.py`).
  - **Shape:** 12 layers, d_model 640, 10 query heads and 2 key/value heads (grouped-query
    attention), SwiGLU hidden size 1728.
  - **Components:** RMSNorm, RoPE (theta 10,000), no biases, tied embeddings, context 1,024.
  - **Size:** 62.1M parameters, of which 51.6M are non-embedding.
- **Evidence** (`scripts/model_sanity.py`, [results/phase3_model.md](results/phase3_model.md)).
  The rule was fixed before the runs (`d11c84f`): take the largest size that passes both checks
  and whose one pass over the 1.49B train tokens takes at most 12 h at the measured speed.

  | | small | medium | large |
  |---|---:|---:|---:|
  | Parameters | 25.2M | 42.2M | 62.1M |
  | Initial loss (ln 16,384 = 9.704) | 9.705 | 9.705 | 9.705 |
  | One batch overfit (300 steps) | 9.71 → 0.020 | 9.71 → 0.007 | 9.71 → 0.005 |
  | Speed, batch 16 × 1,024, bf16 | 90,208 tok/s | 62,917 tok/s | 46,157 tok/s |
  | Peak VRAM | 7.5 GB | 9.0 GB | 10.5 GB |
  | MFU (61 TFLOPS peak) | 26.5% | 30.0% | 31.8% |
  | One pass over the train shards | 4.6 h | 6.6 h | 9.0 h |

  All three pass, so the rule picks large. It is 2% above the brief's "~30–60M". 1.49B tokens
  is 24 tokens per parameter, about compute-optimal. INT8 weights are about 62 MB, which is
  still easy on a CPU or phone.
- **Embedding init:** with tied embeddings, an untrained model repeats its input. The final
  hidden state is close to the input token's own embedding, so that token's logit starts near
  embed_init_std × d_model, about 13 for std 0.02 at d_model 640. That would put the initial
  loss well above ln(vocab). Embeddings therefore start at std 1/d_model, and every initial loss
  landed within 0.001 of ln(vocab). Linear layers use std 0.02, and the residual output
  projections use std 0.02/√(2·layers).
- **Alternatives:** medium (42.2M, 6.6 h) if the owner prefers a faster run or a smaller
  on-device model; small (25.2M, 4.6 h) as a quick baseline.
- **Revisit if:** the Phase 4 pilot's loss curve suggests a different size, or `torch.compile`
  (triton-windows) raises the speed enough to change the time budget.

## D-027 · Phase 6 · Baseline classifier: CAMeLBERT-mix

- **Decision (approved by the owner, 2026-09-24):** the Phase 6 intent-classifier baseline is a
  fine-tuned CAMeLBERT-mix (Apache-2.0) instead of MARBERT or AraBERT.
- **Why:** MARBERT's and AraBERT's model cards state no licence, and the data policy (D-012)
  allows only clearly licensed material. CAMeLBERT-mix was pretrained on MSA, dialectal and
  classical Arabic, which fits Gulf input.
- **Revisit if:** MARBERT or AraBERT publish a clear, compatible licence.

## D-028 · Phase 4 · Pretraining setup and the pilot

- **Decision:** `configs/pretrain/full.yaml` (`scripts/pretrain.py`):
  - **Data:** one pass over the train shards. That is 5,675 optimizer steps of 262,144 tokens:
    micro-batch 16 × 1,024 tokens, with 16 accumulation steps.
  - **Optimizer:** AdamW (β 0.9/0.95), with weight decay 0.1 on matrices and embeddings only.
  - **Schedule:** 200 warmup steps, then cosine decay from peak LR **2e-3** to 2e-4. Gradients
    are clipped at 1.0.
  - **Precision and speed:** bf16 autocast and `torch.compile`.
  - **Evaluation:** validation loss on a fixed 655k-token set every 250 steps.
  - **Checkpoints:** every 250 steps (about every 13 minutes), with automatic resume.
- **Data order:** the windows are laid end to end so that every token is a target exactly once,
  and then shuffled with the run seed. Random windows drawn with replacement would cover only
  ~63% of the corpus in one pass. Because the order depends only on the step, a resumed run sees
  exactly the data it would have seen.
- **torch.compile:** it needs `triton-windows` (3.8.0.post28, MIT, Windows only). It raises the
  speed from 45k to 86k tokens/s (MFU 31% to 59%) for the 62M model. On the same weights, the
  compiled and eager losses agree to 3e-5, and their top predictions agree 100%. The full run
  takes about 4.8 h instead of 9 h.
- **Pilot** ([results/phase4_pilot.md](results/phase4_pilot.md)). The brief asks for ~1% of the
  budget first. There were three pilots of 56 steps each (14.7M tokens), using the full run's
  settings with the schedule compressed. The rule was fixed before the runs (`88cd64c`): among
  learning rates whose train loss never rises by more than 0.5 after warmup, take the lowest one
  whose final validation loss is within 0.02 of the best.

  | Peak LR | Final val loss | Largest loss rise after warmup | Max grad norm |
  |---|---:|---:|---:|
  | 5e-4 | 7.668 | +0.035 | 1.85 |
  | 1e-3 | 7.671 | +0.035 | 6.84 |
  | 2e-3 | **7.600** | +0.032 | 10.36 |

  2e-3 is chosen: it is best by 0.07, and every candidate is stable.
- **Resume check:** a 1e-3 pilot was stopped at step 24 and resumed from its checkpoint. Its
  train loss stayed within 0.015 of the uninterrupted run. On the CPU the unit test is
  bit-identical; GPU kernels are not.
- **Caveat:** short pilots favour higher learning rates, and the full run is 100× longer. 2e-3
  had one grad-norm peak of 10.4 early in warmup, which was clipped. The first 500 steps of the
  full run are the thing to watch. If the loss spikes there, fall back to 1e-3, which was only
  0.07 worse in the pilot.
- **Alternatives:** 1e-3 (safer, slightly slower learning in the pilot); a warmup-stable-decay
  schedule, which makes continued training easier but is less standard for a single pass.
- **Result (full run, 2026-09-24, 14:24–19:13)**
  ([results/phase4_pretrain.md](results/phase4_pretrain.md)):
  - It ran in one attempt with no resume and no loss spike. The gradient norm settled around 0.15.
  - The run covered 5,675 steps and 1.488B tokens in 4 h 49 min, at a median 87k tokens/s
    (MFU ~0.60).
  - Validation loss fell from 9.705 to **3.330** (perplexity 27.9). It was still falling slowly at
    the end: 3.352, 3.341, 3.334 and 3.330 at steps 5,000, 5,250, 5,500 and 5,675.
  - Per source, the loss is lowest on the Bitext customer-service text (0.8–1.5; templated, so
    easy) and on Wikipedia (2.5 ar, 2.8 en). It is highest on Najdi web text (fineweb2-ars, 4.08).
  - **Validation loss sits below train loss throughout; this is not overfitting or leakage.**
    - Each train loss is measured before the model learns from that batch.
    - The final model scores 3.398 on a random sample of train windows (seen early in the run)
      and 3.330 on the validation windows, so the validation split is simply easier text.
    - The validation split has less of the hardest source, fineweb2-ars (16.4% of its tokens vs
      20.1% in train), and more fineweb-edu (35.5% vs 32.1%). Weighting the per-source losses by
      each split's mix gives 3.352 vs 3.316, so the mix explains about 0.04 of the 0.07 gap.
      Where the rest comes from (other document differences, or sampling noise of 640 windows)
      was not measured.
  - The samples are what a 62M base model gives: fluent MSA and English that drifts off topic, a
    customer-service register ("Please feel free to reach out ..."), repetition in Gulf Arabic,
    and gibberish in Arabizi. The Arabizi result is expected, because the corpus has almost no
    Arabizi. Answering in the right format and intent is the job of Phase 5 (SFT).

## D-029 · Phase 5 · SFT setup, reply check and selection rule

- **Decision:** this is fixed before any SFT run. The owner approved the plan on 2026-09-24.
  - **Format:** `<|user|>{message}<|assistant|>{"intent": …, "reply": …}<|endoftext|>`, using
    the tokenizer's reserved special tokens. The message is normalized as in the SFT data
    (`normalize`, which also masks PII), both in training and at inference.
  - **Loss:** only on the answer tokens, that is the JSON and the end token.
  - **Data:** `data/sft/final` (17,579 train, 968 validation), unless the reply check below
    filters it.
  - **Optimizer and schedule:** training starts from the pretrained `model.pt` (D-028).
    - AdamW (β 0.9/0.95, weight decay 0.1), with 64 examples per step.
    - 3% warmup, then cosine decay to 10% of the peak. Gradients are clipped at 1.0.
    - bf16 autocast, up to 5 epochs, with a checkpoint and an evaluation after every epoch.
  - **Sweep:** peak learning rates 1e-4, 3e-4 and 1e-3.
  - **Development metrics,** on the SFT validation split only. The human test set is never
    used.
    - Validation loss on the answer tokens.
    - With greedy decoding:
      - **valid-JSON rate:** the output parses, has exactly `intent` and `reply` as strings, and
        the intent is one of the 21;
      - **intent accuracy and macro-F1**, where an invalid output counts as wrong;
      - **reply-language rate:** the reply's detected language is one that the message's variety
        allows (`configs/sft/generate.yaml`).
    - **Intent by likelihood:** the most likely of the 21 labels after `{"intent": "`. Its
      probability is the model's confidence.
- **Selection rule:** take the checkpoint with the highest greedy intent accuracy among the 15
  (3 learning rates × 5 epochs).
  - Checkpoints within 1.0 point of the best count as tied.
  - Among the tied ones, take the highest valid-JSON rate, then the lowest validation loss.
- **Reply check before training (the owner's request):** a spot check found broken words in
  Arabic replies: "سنبعد الطلب" for "سنبعث الطلب", and "سنعبر رسالتك". The label judge (D-022)
  never checked replies.
  - **Sample:** Qwen3.5-35B-A3B judges 500 training replies, 125 each from Gulf Arabic, English,
    Arabizi and mixed messages (seeded), at temperature 0. Each reply is judged OK or BROKEN
    (with a reason). BROKEN means a misspelled or wrong word, broken grammar that a native
    reader would notice, or the wrong language.
  - **Canaries:** the two replies above are judged separately and not counted. Qwen wrote these
    replies itself and may miss its own mistakes; the canaries show whether it catches known
    ones.
  - **Threshold:** weight each variety's broken rate by its share of the training set. If that
    estimate is above **3%**, judge every training and validation reply and drop the broken
    ones before training. Otherwise use the data as it is and report the rate.
- **Distillation:** the SFT data is already sequence-level distillation, because the student
  learns the outputs the teacher wrote. Word-level distillation (matching the teacher's token
  probabilities) is not possible: Qwen's 248k-token vocabulary differs from DanaLM's 16k. So
  there is no separate distillation step.

- **Result (2026-09-25; [results/phase5_sft.md](results/phase5_sft.md)):**
  - 30 checkpoints were trained: 2 rounds × 3 learning rates × 5 epochs.
  - The rule chose **round 1, peak LR 3e-4, epoch 3**. On the SFT validation split (957
    examples) it gets:
    - valid JSON **100.0%**;
    - intent accuracy **92.9%**, macro-F1 92.5%, and 92.8% by likelihood;
    - replies in an allowed language 100.0%;
    - at 95% accuracy it covers 96.6% of the messages (confidence threshold 0.612).
  - By variety: English 94.2%, Arabizi 92.5%, Gulf Arabic 92.0%, mixed 91.9%.
  - The most frequent confusion is order_status → failed_delivery (6 times), and the reverse
    3 times.
  - In most runs the validation loss is lowest at epoch 3 and rises after it, while accuracy
    stays level. That is overfitting, and the rule avoids it.
  - The CAMeLBERT-mix baseline scores 93.3% accuracy and 92.9% macro-F1 on the same split, 0.4
    points above this checkpoint.
  - **Caveat:** the same teacher wrote the validation split, and it keeps only examples the
    judge agreed with, so it is easier than real messages. Phase 6 measures on the human test
    set.

## D-030 · Phase 6 · What counts as good, fixed before any SFT result

- **Decision:** the owner accepted these on 2026-09-24, before training. On the human test set,
  DanaLM meets its goal when all four hold:
  1. **Valid JSON ≥ 99%,** with greedy decoding and no constrained decoding.
  2. **Intent macro-F1 within 3 points of the CAMeLBERT-mix baseline** (D-027) on the same test
     set.
  3. **Coverage ≥ 70% at ≥ 95% accuracy.** The confidence is the model's probability for the
     intent it outputs, among the 21 labels (D-029).
     - The threshold is fixed on the SFT validation split before the test set is scored: the
       lowest one at which the validation messages above it are at least 95% right.
     - On the test set, the messages above the threshold must be at least 70% of all messages,
       and at least 95% of them must be right. The rest go to a person or a bigger model, as in
       the brief's business case.
  4. **Reply in the right language ≥ 95%,** by the per-variety rule of D-029.
- **Reporting:** each bar is reported as met or missed, with the measured number overall and
  per variety. A missed bar is reported as missed; the bars are not moved after the test set
  has been scored.
- **Why:** writing the bar down before any result exists keeps the evaluation honest.

## D-031 · Phase 5 · Overnight improvement run (2026-09-24/25)

- **Decision:** the owner approved the plan and said "go" at about 20:00. It is a 12-hour chain,
  and every choice in it is fixed here, before any result exists.
  - **A. Reply cleaning.** This updates D-029.
    - The 500-reply sample estimated 3.8% broken replies: Gulf Arabic 7.2%, Arabizi 4.0%,
      mixed 3.2%, English 0.8%. That is above the 3% threshold, so every train and validation
      reply is judged.
    - The owner chose to **rewrite** broken replies instead of dropping them, to keep every
      message. Qwen writes a new reply with the Phase 2 reply rules. The new reply must pass the
      same filters (language, no claimed actions, Gulf dialect) and a second judgement, or the
      example is dropped.
    - Caveat: the judge missed both canaries, so some broken replies will remain.
  - **B. Top-up and contrast.**
    - Every intent × variety cell (not `saudi_arabic`) with fewer than 200 examples is topped up
      towards 250 kept examples. That is 26 cells and about 3,200 examples.
    - The intents that get confused (order_status, failed_delivery, cancel_order,
      refund_request, card_not_working, lost_or_stolen_card, transfer_issue,
      unrecognized_transaction, handoff_to_human, other) get at least 6 requests per variety.
      Each of their prompts adds a rule that makes the message clearly that intent and not its
      neighbour.
    - The generation rules, filters, blind label judge and reply check are the same as before.
    - New examples go to the **training split only**. Any new message that is a near copy of a
      validation message is dropped. The validation split stays the original 968 examples,
      cleaned, so every comparison tonight uses the same development set.
    - Added before assembly: any training message, old or new, that equals or nearly copies a
      human test message is dropped. It uses the matching of `scripts/check_overlap.py`
      (canonical form, character 3-gram MinHash at 0.6). The test set is used for nothing
      else. The overlap check is re-run after assembly, and any hit is reported for a person
      to resolve, as `docs/TEST_SET.md` requires.
  - **C. SFT round 1** on the pretrained model (D-028), with the D-029 sweep and selection rule,
    on the data from A and B.
  - **D. Second pretraining pass.**
    - It starts from the final weights with a fresh optimizer and makes one more pass over the
      same train shards, in a new order (seed 43).
    - The learning rate warms up over 200 steps to 1e-3, then follows a cosine to 1e-4.
      Everything else is as in D-028.
    - It is kept only if its validation loss on the same fixed validation windows is below
      3.330.
  - **E. SFT round 2** runs on the new base, if D is kept, with the same sweep and rule. The
    Phase 5 model is the better of the two round winners by the D-029 rule on the same
    development set.
  - **G. Baseline preparation for Phase 6 (D-027).**
    - CAMeLBERT-mix is fine-tuned as an intent classifier on the same SFT training messages:
      learning rate 2e-5, batch 32, 3 epochs.
    - The best epoch is chosen by accuracy on the SFT validation split. The human test set is
      not used.
  - **Operational change at 21:05.** Qwen's host prompt cache is capped at 2 GiB
    (`--cache-ram 2048`, default 8 GiB) from the rewrite step on. During the full reply check
    the server grew to 34 GB committed, and a longer unattended job could have reached the
    watchdog's limit. The speed effect is compared with the full run's rate in the report.
  - **Not tonight:** no test-set evaluation, no push, and no change to the intent taxonomy.
- **Results (2026-09-24 20:00 to 2026-09-25 04:57):** every step succeeded at its first
  attempt, and the watchdog never killed anything.
  - **A.** Broken replies:
    - The full check marked 721 of 17,579 training replies and 42 of 968 validation replies as
      broken: Gulf Arabic 7.1%, Arabizi 5.4%, mixed 3.9%, English 0.2%, Saudi 11.2%.
    - The judge caught 0 of 2 canaries in the sample and 1 of 2 in the full check.
    - Of the rewrites, 724 passed the filters and 679 passed the second judgement, so 84
      examples were dropped.
  - **B.** Top-up:
    - 1,501 requests produced 11,984 examples. The filters kept 3,692.
    - 3,066 were added. The rest were dropped: the judge disagreed on 512, 110 had broken
      replies, 4 were duplicates, and none copied a validation or test message.
    - Mixed kept only 24%: Qwen wrote pure Arabic 4,870 times and pure English 1,047 times.
    - The smallest cell grew from 46 to 93 examples (failed_delivery × Arabizi).
    - With the 2 GiB cache, Qwen ran at 93.9 tokens/s, against 92 in the full run.
  - **Data v3:** 20,572 training and 957 validation examples. The overlap check found 0 overlaps
    between the 64 test messages and any training text.
  - **D.** The second pretraining pass brought validation loss from 3.330 to **3.232**
    (perplexity 27.9 to 25.3), so it was kept. Its results page is
    [results/phase4_second_pass.md](results/phase4_second_pass.md).
  - **C and E.** Round 2's winner (LR 3e-4, epoch 3: 92.6%, valid JSON 100%, validation loss
    0.798) is tied with round 1's (92.9%, 100%, 0.796). The rule decides on loss, so **round 1
    is the Phase 5 model**.
    - The highest single accuracy was 93.5% (round 2, LR 1e-3, epoch 4). It lost the tie-break,
      with 99.9% valid JSON and a validation loss of 0.881.
    - So the second pass lowered the base model's loss, but it brought no measurable gain on
      this development split.
  - **G.** CAMeLBERT-mix scored 93.3% accuracy and 92.9% macro-F1 at its best epoch (3). Its 3
    epochs took about 33 s each.
- **Why:** the model's clearest weaknesses are in the data: broken replies, thin mixed and
  Arabizi cells, and intents that get confused. Pretraining loss was also still falling at the
  end of the first pass.

## D-032 · Phase 6 · Evaluation protocol for the human test set, fixed before scoring

- **Decision:** the owner said "go" for Phase 6 on 2026-09-25. Everything below is committed
  before any test-set result is computed. The test set is scored once with this protocol, and
  nothing is tuned on it afterwards.
  - **Test set:** `data/test/test_set.jsonl`, 64 English messages over 10 intents, labelled by
    the owner (MR). SHA-256 `e6961fd2…48397`. The overlap check reports 0 overlaps.
    - The Gulf Arabic, Arabizi and mixed parts do not exist yet. So this is **a partial
      evaluation, English only**, and the D-030 bars can only be judged for English.
  - **Model:** the Phase 5 choice (D-029): `sft-r1-lr3e-4/epoch_3.pt`. It uses greedy decoding
    with at most 96 new tokens, the SFT normalization, and no constrained decoding.
  - **Metrics:**
    - valid-JSON rate;
    - intent accuracy, and macro-F1 over the intents in the test set (an invalid answer counts
      as wrong);
    - reply-language rate (D-029 rule);
    - coverage at the confidence threshold fixed on the SFT validation split, 0.612 (D-030).
  - **Baselines:**
    - CAMeLBERT-mix at its best epoch on the SFT validation split (D-027, D-031 G).
    - Qwen3.5-35B-A3B zero-shot. It uses the Phase 2 label-judge prompt (the 21 intents with
      their descriptions) at temperature 0, 20 messages per request.
  - **Uncertainty:** 64 messages is a small sample.
    - Every rate is reported with a Wilson 95% interval.
    - The difference in macro-F1 between DanaLM and each baseline is reported with a paired
      bootstrap 95% interval: 10,000 resamples, seed 42.
    - A D-030 bar counts as met when the point estimate meets it. The interval is shown next to
      it.
  - **Reply quality:**
    - Qwen judges each valid DanaLM reply on four yes/no questions: does it answer the
      message; is it polite and clear; is it in the customer's language; does it avoid
      invented facts and claims that something was done? A reply is "good" when all four are
      yes.
    - Caveat: Qwen wrote DanaLM's training replies, so it may be lenient.
    - Every reply is listed on the results page for the owner's manual spot check.
  - **Latency and memory:**
    - Per message, batch 1, with the current decoder (no KV cache yet; that is Phase 7).
    - On the CPU: float32, 4 threads, like a small device. On the GPU: bf16.
    - Median and p95 over the 64 messages, after 3 warm-up messages.
    - Memory: the size of the weights, peak process memory on the CPU, and peak allocated
      memory on the GPU.
  - **Sensitivity:** a separate audit disagreed with the owner's label on rows 28, 57 and 58.
    Accuracy is also reported with those three rows left out. This is only a sensitivity line;
    the owner's labels stay the primary result.
  - **Error analysis:** every message DanaLM gets wrong is listed with its answer.
- **Amendment (2026-09-25, before any test-set scoring):** a smoke run on 12 SFT validation
  messages showed a flaw in the reply-judge prompt. Qwen marked replies such as "I will pass
  your request to the team" and "I will connect you to a human agent" as unsafe claims of
  action, but those are exactly what the SFT reply rules and the `handoff_to_human` rule ask
  for. The prompt now says they are allowed, and `safe` asks about actions *already* done.
  With the fix, the judge still flags real problems (for example, promising to activate roaming
  directly). The smoke run also showed that Qwen sometimes misnumbers its answer lines. A
  message it leaves unlabelled counts as wrong, as the protocol says, and the report shows the
  count.
- **Result (2026-09-25; [results/phase6_eval.md](results/phase6_eval.md)). English part only,
  64 messages:**

  | System | Intent accuracy (95% interval) | Macro-F1 |
  |---|---:|---:|
  | DanaLM (62M) | **56.2%** (44–68%) | 57.8% |
  | CAMeLBERT-mix classifier (110M) | 46.9% (35–59%) | 46.4% |
  | Qwen3.5-35B-A3B, zero-shot | 89.1% (79–95%) | 90.3% |

  - DanaLM gives valid JSON 100% of the time and replies in the right language 100% of the time.
    Qwen judged 89.1% of its replies good.
  - The macro-F1 difference between DanaLM and CAMeLBERT is +11.5 points (interval +1.0 to
    +22.0). Against Qwen it is −32.4 points (interval −46.7 to −20.8).
  - **Bars (English only):**
    - valid JSON: met;
    - macro-F1 within 3 points of CAMeLBERT: met;
    - reply language: met;
    - **coverage: missed.** At the threshold set on the SFT validation split (0.612), DanaLM
      answers 85.9% of the messages, but only 61.8% of those answers are right. Its confidence
      is not calibrated for real messages.
  - **The main finding:** both small models fall from about 93% on the synthetic validation
    split to 47–56% on real messages.
    - The Qwen-written training data does not prepare them for short, US-style banking
      questions such as "what have i spent things on" or "this charge is bs".
    - The weakest intents are unrecognized_transaction (1 of 7 right) and balance_or_statement
      (1 of 6).
    - Many errors land in `other` with high confidence and an off-topic reply.
  - **Sensitivity**, without the 3 disputed rows: DanaLM 55.7%, CAMeLBERT 47.5%, Qwen 91.8%.
  - **Latency**, batch 1 with no KV cache yet:
    - CPU (float32, 4 threads): median 0.63 s, p95 0.92 s. GPU: median 0.37 s.
    - The weights are 237 MB. The process peaks at 1.2 GB, of which 0.93 GB was there before the
      model loaded.

## D-033 · Phase 5b · Real customer messages, to raise accuracy on real messages

- **Decision:** the owner asked on 2026-09-25 for a plan that raises accuracy, and said "go".
  Everything below is fixed before any data is fetched or any model is trained.
  - **Why:** in Phase 6 (D-032), DanaLM fell from 92.9% on the synthetic validation split to
    56.2% on real English messages. The training messages were all written by the teacher.
  - **Real data:**
    - The train splits of Banking77 (CC-BY-4.0, PolyAI repository at `57ec275d`) and CLINC150
      (CC-BY-3.0, `clinc/clinc_oos` "plus" at `155b9c71`), mapped to our intents with the D-025
      label maps: 5,394 English messages over 10 intents. The test splits stay reserved.
    - Any message that equals or nearly copies a human test message is dropped first, using the
      matching of `check_overlap.py`.
  - **Real dev set:**
    - 15% of each intent (seed 42) is held out before anything else is done with these messages.
      It keeps the mapped labels without any filter: noisy, but not biased towards easy messages.
    - It is used only to choose the model and the confidence threshold, never for training.
  - **The remaining real messages:**
    - Training candidates that nearly copy a dev message are dropped. At most 500 per intent are
      kept, spread over the source labels.
    - Qwen writes English replies with the SFT reply rules.
    - The blind label judge keeps only the messages whose mapped label it agrees with, and the
      reply check drops broken replies.
  - **Style top-up (synthetic, English):** 6 requests per intent (126 requests), with a style rule
    for how real customers type: short and blunt, often lowercase, typos, no greeting, varied
    international English. The same filters, judge and reply check apply.
  - **Data v4:**
    - final-v3, plus the real messages, plus the style top-up.
    - Training messages that nearly copy a test or real-dev message are dropped.
    - The SFT validation split stays the same 957 examples.
  - **Training:**
    - The D-029 recipe, on two bases (the first and the second pretraining pass) × two peak
      learning rates (3e-4 and 1e-3), 5 epochs each.
    - Every epoch is evaluated on the SFT validation split and on the real dev set.
  - **Selection:**
    - The checkpoint with the highest intent accuracy on the real dev set wins, among those whose
      SFT-validation accuracy is at least 90.9%. That is the Phase 5 model's 92.9% minus 2
      points, so Gulf Arabic, Arabizi and mixed messages do not pay for gains in English.
    - Within 1 point of the best: higher valid JSON on the real dev set wins, then higher
      SFT-validation accuracy.
  - **Coverage threshold (D-030):** fixed on the real dev set with the chosen model, instead of
    on the synthetic split.
  - **Test:** the chosen model is scored once with the D-032 protocol. Phase 6's result stays on
    record next to it.
  - **Caveat:** the English test set comes from the test splits of the same two datasets. Part
    of any English gain will come from training on their style. The Gulf Arabic, Arabizi and
    mixed test parts, still unwritten, remain the real check.
- **Time:** about 1 h of teacher time and 40 minutes of GPU time. No single job runs over 2 h.
- **Result (2026-09-25; [results/phase5b_real.md](results/phase5b_real.md),
  [results/phase6b_eval.md](results/phase6b_eval.md)):**
  - **Data:**
    - 5,391 real messages were mapped (the plan estimated 5,394).
      - 31 near copies of test messages were dropped.
      - 804 messages became the real dev set.
      - 733 training candidates were dropped as near copies of dev messages, leaving 3,315.
    - Qwen wrote replies for 3,272 of them. The filters dropped 43, of which 42 were replies
      that claimed an action.
    - The label judge agreed with 3,059 (93.5%). Most disagreements were borderline cases, such
      as the minimum payment on an electricity bill (loans_and_credit or bill_inquiry, 37).
      The reply check marked 4 replies broken.
    - Style top-up:
      - 889 examples from 126 requests. The judge agreed with 790 (88.9%) and the reply check
        marked 14 broken.
      - The messages are short (median 7 words), all start in lowercase, and 2% mention the
        UAE.
    - final-v4 adds 2,921 real examples (134 others repeated an existing message) and 770 style
      examples (10 others nearly copied a test or dev message). That makes 24,263 training
      examples; the validation split stays at 957.
    - Both reply checks judged both canaries OK: 0 of 2 caught. The canaries are broken Gulf
      Arabic, so they don't test the check on English replies. How much the check misses on
      English replies is still unknown.
  - **Selection:** second pass, lr 3e-4, epoch 3.
    - Real dev set: 94.4% intent accuracy (macro-F1 94.5%). The Phase 5 model scores 66.4% on
      the same 804 messages.
    - The top of the sweep is flat: two other checkpoints came within 0.2 points. The
      tie-break (SFT-validation accuracy) decided.
    - SFT validation: 93.1% (Phase 5 model: 92.9%). Per variety: Gulf Arabic 91.6% (92.0%),
      Arabizi 91.7% (92.5%), mixed 93.8% (91.9%), English 95.6% (94.2%).
    - At confidence ≥ 0.592 it answers 97.8% of the dev messages, with 95.0% accuracy.
  - **Test (English part, 64 messages, scored once with the D-032 protocol):**

    | System | Intent accuracy (95% interval) | Macro-F1 |
    |---|---:|---:|
    | DanaLM after D-033 | **84.4%** (74–91%) | 85.5% |
    | DanaLM, Phase 5 model (D-032) | 56.2% (44–68%) | 57.8% |
    | CAMeLBERT-mix classifier (110M) | 46.9% (35–59%) | 46.4% |
    | Qwen3.5-35B-A3B, zero-shot (the same answers as in D-032) | 89.1% (79–95%) | 90.3% |

    - On the same messages, the new model fixed 19 of the old model's errors and made 1 new one
      (exact McNemar test, p = 0.00004).
    - The macro-F1 gap to Qwen is −4.7 points (interval −16.0 to +6.0). At this sample size, the
      gap to the 35B teacher is no longer clear. Against CAMeLBERT the gap is +39.1 points.
    - Valid JSON 100% and reply language 100%. Qwen judged 92.2% of the replies good (D-032:
      89.1%).
    - **Bars:** valid JSON, macro-F1 and reply language are met. **Coverage is still missed.**
      - At the real-dev threshold (0.592), the model answers 95.3% of the test messages, and
        86.9% of those answers are right.
      - The model is right more often on the real dev set (94.4%) than on the test set (84.4%),
        so the threshold set on the dev set is too low for the test messages.
      - 4 of the 10 errors have confidence ≥ 0.9 (D-032: 14 of 28).
    - Without the 3 disputed rows: 85.2%.
    - The weakest intents are unrecognized_transaction (4 of 7 right) and balance_or_statement
      (4 of 6).
  - **Latency: not comparable with D-032.**
    - The protocol measured 1.95 s per message on the CPU and 1.07 s on the GPU. But from about
      13:12 the machine ran slower:
      - the speed per token fell 2.5× on both the CPU and the GPU;
      - SFT evaluations in the same period took 150 s instead of 85 s.
    - A diagnostic run right after (a scratch script, not part of the protocol) measured the
      unchanged Phase 5 model at 25 tokens per second on the CPU, against 60 in D-032. The new
      model ran at 23 tokens per second.
    - **Cause, found in Phase 7:** Windows power throttling (EcoQoS) of background processes.
      With throttling turned off for the measuring process alone, the same model ran at 62.0
      tokens per second; with the default policy before and after, at 24.5 and 24.3 (D-034).
    - One real change: the new model's answers are about 20% longer (47 tokens on average on
      the test set, against 39). On the same machine it needs about a quarter more time per
      message.
    - Phase 7 measures latency again, with the KV cache and quantization.
  - **Caveat, as planned:** the test messages come from the test splits of the same datasets as
    the new training messages. The Gulf Arabic, Arabizi and mixed test parts remain the real
    check.
  - **Known issues for the next data round:**
    - About 2% of the English training replies say that an action is already done ("we have
      logged/noted/forwarded this"): 2.1% in final-v3, 0.6% of the real-message replies and
      2.6% of the style top-up. The claim filter doesn't catch these verbs. It was not changed
      during this round.
    - One test reply contains a broken word ("onwellowing").
  - **Time:** the teacher steps ran from 11:42 to 12:42 and the GPU steps from 12:43 to 13:36.

## D-034 · Phase 7 · Quantization and deployment, with the rules fixed before any result

- **Decision:** the owner said "go" for Phase 7 on 2026-09-25. The model is the D-033 model
  (`sft-selected-5b.json`). Everything below is fixed before any quantized model is scored.
- **Runtime: ONNX Runtime on the CPU.**
  - It runs the same graph on servers, in Docker, and on phones (ONNX Runtime Mobile).
  - Its quantizers cover both INT8 and INT4.
  - The confidence (D-030) needs the likelihood of all 21 intent names. With our own step graph
    that is a single extra batched step.
  - Alternatives considered:
    - **GGUF/llama.cpp.** Our RoPE rotates adjacent pairs, which is llama.cpp's own layout, so a
      conversion is possible. It is deferred: llama.cpp does not know our tokenizer, and scoring
      21 continuations there needs a compiled Python binding.
    - **Static INT8 with calibration.** Dynamic INT8 needs no calibration data and is the usual
      choice for transformer weights.
- **KV cache.**
  - `DanaLM.step(input_ids, positions, past)` returns the logits and the updated keys and
    values.
  - Attention uses an explicit position mask. `is_causal` would be wrong for a single new token
    attending to a longer cache.
  - The training path does not change.
  - One ONNX graph serves both reading the prompt (with an empty cache) and generating one token
    at a time.
  - The output projection gets its own copy of the tied embedding. Otherwise the quantizers skip
    it, because its weight is not a constant.
  - A feasibility check on a tiny random model matched PyTorch to 1.2e-7 and gave the same
    greedy tokens, for INT8 and INT4 as well.
- **Variants:**
  - `onnx-fp32`: the reference export.
  - `onnx-int8`: dynamic INT8 (`quantize_dynamic`, per-channel signed weights). It covers every
    MatMul weight, including the output projection.
  - `onnx-int4`: 4-bit weights (`MatMulNBitsQuantizer`, block 32, symmetric, accuracy level 4,
    i.e. INT8 compute). It covers every MatMul weight, including the output projection.
  - The input embedding table stays float32 in all of them.
  - **One fallback per quantized variant, and no further tuning:** if a variant fails the rule
    below, the same variant is built once more with the output projection left in float32.
- **Checks on the development sets only:** the SFT validation split (957) and the real dev set
  (804). The reference is PyTorch float32 on the CPU, without the cache (the decoding of D-029).
  - The PyTorch KV-cache path and `onnx-fp32` must each give the same answer as the reference
    on at least 99.5% of these messages. Every difference is listed.
  - Every variant reports these on both sets:
    - valid JSON, intent accuracy and macro-F1, reply language;
    - agreement with the reference (same intent, same answer);
    - coverage at its own threshold, fixed on the real dev set as in D-033 (95% target);
    - file size.
- **Rule for a quantized variant.** On both development sets:
  - its intent accuracy is at most 1.0 point below `onnx-fp32`'s;
  - valid JSON ≥ 99%;
  - reply language ≥ 99%.
- **What gets deployed:**
  - The smallest variant that passes, by file size on disk.
  - Variants within 10% of each other in size are decided by the lower median latency: full
    prediction, 4 threads.
  - If no quantized variant passes, `onnx-fp32` is deployed, and the result page says so.
- **Test set:** once the choice is committed, the human test set (the English part, SHA-256
  checked) is scored once per variant and once for PyTorch float32 on the CPU. That gives
  accuracy before and after quantization.
  - Qwen judges the deployed variant's replies with the D-032 prompt.
  - These numbers are reported and never change the choice.
- **Latency protocol:**
  - 100 development messages: a seeded sample (seed 42), 50 from the SFT validation split (all
    varieties) and 50 from the real dev set. The test set is not used.
  - Batch 1 on the CPU, with 4 threads (as D-032) and with 1 thread (closer to a phone core).
    3 warm-up messages. Each variant runs in its own process.
  - Measured:
    - median and p95 per message, for the answer alone and for the full prediction (answer plus
      confidence);
    - answer tokens per second;
    - peak process memory, load time and file size.
  - The table includes PyTorch float32 without the cache, the setting of D-032, measured in the
    same session.
  - On Windows, each measuring process turns off power throttling for itself, and the output
    records it.
    - Reason: during D-033 the default policy slowed the unchanged Phase 5 model from 62.0 to
      24.5 tokens per second.
    - No system setting is changed.
- **Deployment pieces:**
  - **Torch-free inference package (`danalm.infer`).** Text normalization, the chat format and
    answer parsing move to torch-free modules, re-exported from their old places. The server
    needs only numpy, tokenizers, onnxruntime and PyYAML.
  - **FastAPI service.**
    - `POST /predict` returns intent, reply, confidence and a route: on the device when the
      answer is valid JSON and its confidence clears the threshold, otherwise escalate.
    - It also returns the PII-masked message, which is what may leave the device. The model
      already sees masked text: `normalize` masks PII.
    - `GET /health` reports the model and its SHA-256.
    - Raw messages are never logged.
  - **Dockerfile:** a slim Python image without PyTorch. The model directory is mounted at run
    time, because models are never committed.
  - **GitHub Actions CI:**
    - lint, the unit tests, and a Docker build;
    - a small end-to-end evaluation on a tiny model trained in CI: export, INT8, service
      predictor, metrics.
    - An evaluation of the real model in CI needs the model published on the Hugging Face Hub.
      That is the owner's decision in Phase 8.
  - **Gradio app for Hugging Face Spaces:** prepared and tested locally. Publishing it needs the
    owner's account.
- **Time:** no job is expected to run over 2 h.
  - The PyTorch float32 reference without a cache takes about 30 minutes of CPU time on the
    development sets.
  - Each ONNX variant takes a few minutes.
  - The latency runs take about 15 minutes.
- **Result on the development sets (2026-09-25; [results/phase7_deploy.md](results/phase7_deploy.md)),
  committed before the test set is scored:**
  - **Exactness.** The PyTorch KV-cache path and `onnx-fp32` gave answers identical to the
    reference on all 1,761 development messages.
  - **The reference** (PyTorch float32 on the CPU): 93.0% intent accuracy on the SFT validation
    split and 94.15% on the real dev set. The bf16 GPU runs of D-033 gave 93.1% and 94.4%.
  - **INT8** (100.8 MB): 93.0% and 93.91%, which is 2 messages fewer on the real dev set.
  - **INT4** (78.1 MB): 93.21% and 93.16%, which is 8 messages fewer of 804 on the real dev set.
    The limit was 8.04, so INT4 passes by less than one message.
  - Both variants give 100% valid JSON and at least 99.9% reply language, so both pass.
  - **INT4 is deployed**, as the smallest passing variant.
  - **Replies change more than intents.** The quantized variants keep the reference's intent on
    98–99% of the messages, but its exact answer on only 17–38%. The Qwen judge on the test
    replies checks their quality.
  - **Thresholds** (real dev set, 95% target): fp32 0.600, INT8 0.589, INT4 0.564.
  - Rebuilding INT8 and INT4 with the refactored `danalm.model.quantize` gives byte-identical
    files.
  - **The development PC crashed during this run** (blue screen `HYPERVISOR_ERROR` at 14:36).
    - The system log also shows earlier blue screens, machine-check errors from the CPU and a
      fatal hardware error on 2026-09-23.
    - The BIOS (0812, February 2023, microcode 0x10E) predates Intel's stability fixes for
      13th- and 14th-generation CPUs; the owner may update it.
    - The rerun reused the saved reference on the SFT validation split, with the same 16 threads
      both times.
    - The three float32 paths agree on every message, so no silent computation error shows in
      these results.
- **Result on the human test set** (English, 64 messages; scored once per system after the
  choice was committed):

  | System | Intent accuracy (95% interval) | Valid JSON | Reply language |
  |---|---:|---:|---:|
  | PyTorch float32 (CPU) | 85.9% (75–92%) | 100% | 100% |
  | ONNX fp32 | 85.9% (75–92%) | 100% | 100% |
  | ONNX INT8 | 82.8% (72–90%) | 100% | 100% |
  | **ONNX INT4 (deployed)** | **84.4% (74–91%)** | 98.4% | 96.9% |

  - Phase 6b scored the same model in bf16 on the GPU at 84.4%.
  - **INT4 gave three broken answers:**
    - t0010 and t0064: Arabic words inside English replies;
    - t0013: a repetition loop that ended as invalid JSON.
  - D-034's reply-language check could not see the first two. D-035 answers that.
  - Qwen judged 87.3% (55/63) of INT4's valid replies good on all four questions; the same model
    unquantized got 92.2% in Phase 6b.
  - At its dev threshold, INT4 answers 93.8% of the test messages, and 88.3% of those answers
    are right. The D-030 coverage bar (95% right) is still missed.
- **Latency and memory** (batch 1 on the CPU, 100 development messages, each system in its own
  process, power throttling off for that process):
  - **Against D-032's setting** (PyTorch float32, no cache, 4 threads), INT4 is:
    - 6.7× faster per answer: 0.118 s instead of 0.788 s;
    - 3.9× faster for the full prediction with the confidence: 0.251 s instead of 0.972 s;
    - 4.7× smaller in peak memory: 252 MB instead of 1,190 MB;
    - 3× smaller on disk: 78 MB instead of 237 MB.
  - **With 1 thread**, INT4 takes 0.159 s per answer and 0.448 s with the confidence (D-032's
    setting: 2.115 s and 2.765 s).
  - The KV cache alone makes PyTorch 2.3× faster (0.350 s per answer). ONNX Runtime fp32 takes
    0.284 s.
  - **A trade-off the rule did not weigh:** INT8 is faster than INT4 for the full prediction
    (0.188 s with 4 threads, 0.246 s with 1 thread). INT4's kernel is slower on the batched
    21-intent scoring step. The rule ranks by size first, and 101 MB against 78 MB is not a
    tie.
- **What to change next time:** judge reply quality (the D-035 guard, or the Qwen judge on
  development replies) and full-prediction latency before choosing a variant, not after.

## D-035 · Phase 7 · A reply-language guard in the predictor, added after the test run

- **Why:** on the test set (D-034, scored once), INT4 gave three garbled replies out of 64.
  - t0010 and t0064: Arabic words inside an English reply.
  - t0013: a repetition loop that ended as invalid JSON; the service already escalates those.
  - The development sets had shown none.
  - The D-034 check could not see them. It detects the language of the whole reply, and a few
    Arabic words in a mostly English reply still count as English.
  - The model choice does not change: INT4 stays, as D-034 requires.
- **Decision:** the predictor escalates an answer whose reply does not fit the customer's
  language, instead of sending it.
  - The message's language is detected with `detect_lang` on the normalized text.
  - The reply must be in one of the D-023 reply languages for that language: English for
    English; Gulf Arabic script for Arabic and Arabizi; Arabic or mixed for mixed.
  - A reply to an English message may not contain any Arabic letter.
  - A message without letters is not checked.
- **Measurement:**
  - The guard is applied to the saved predictions of every system. No model is re-run.
  - On the development sets, its escalations are reported and listed. This is the false-alarm
    estimate, since those sets did not motivate the guard.
  - On the test set, its effect is shown too, but it is not an independent estimate, because the
    test set revealed the problem.
- **Result** ([results/phase7_deploy.md](results/phase7_deploy.md), applied to the saved
  predictions):
  - On the development sets it rejects:
    - 2 of 1,761 replies of the float32 model;
    - 6 of INT8's;
    - 2 of INT4's.
  - Each of those replies is garbled mixed-script text, e.g. "the reverse of the lastدرءrestaurant".
    So even the unquantized model occasionally produces them, and no false alarm was seen.
  - The answers moved from "on the device" to "escalate" are at most 3 per set. The accuracy of
    the remaining on-device answers does not change.
  - On the test set it rejects INT4's t0010 and t0064 and INT8's t0064.
- **Checked by hand in the local Gradio demo** (INT4, from the assembled Space folder):
  - A Gulf Arabic order question was answered on the device with `order_status` and a fluent
    Arabic reply.
  - An English message with a phone number came back masked as `<PHONE>`.

## D-036 · Phase 8 · An in-browser demo instead of a Gradio Space

- **Why:** Hugging Face now needs a paid PRO plan to host Gradio or Docker Spaces. Static Spaces
  are free. On 2026-09-25 the owner chose an in-browser demo over paying.
- **Decision:** a static page, `web/index.html`, downloads the INT4 model from the model
  repository once and runs it with ONNX Runtime Web 1.30.0 (WebAssembly). That is the same
  version as the Python runtime.
  - `web/danalm.js` ports the torch-free Python path:
    - normalization and PII masking;
    - the byte-level BPE tokenizer;
    - KV-cache greedy decoding and the 21-intent confidence;
    - the D-035 guard.
  - Python's Unicode `\w`, `\d`, `\s` and `\b` are written out, since JavaScript's differ.
  - The Space sends cross-origin-isolation headers, so the runtime can use threads.
  - The Gradio app (`space/`) stays in the repository for local runs.
- **Parity with Python** (`scripts/web_parity.py`, `web/parity.html`; development messages only):
  - **Text: 314 of 314 identical**, for the normalized text, the language tag, the token ids and
    the decoded text. The set is 300 development messages and 14 crafted strings: every PII form,
    Arabic-Indic digits, repeated characters, unusual spaces and emoji.
  - **Predictions** (60 development messages, INT4):
    - same intent 60/60;
    - same generated answer 53/60;
    - same route 59/60;
    - confidence at most 0.053 apart.
  - The WebAssembly INT4 kernels round differently from the native ones. So a few replies differ
    in wording, and one route near the threshold flipped.
- **Checked by hand:** the page loaded the model from the Hugging Face repository in 7 s, with
  cross-origin isolation on. It answered the Gulf Arabic, Arabizi and English examples as the
  Python demo did, and masked an email and a phone number.
  - Time: about 1.3 s per message in the app's hidden browser pane with 4 threads. The native CPU
    takes 0.25 s.

## D-037 · Phase 5c · Question types the model has not seen, and replies that stick to the message

- **Why:** the owner tried the demo on 2026-09-25, and two messages went wrong.
  - "i asked one refund from amazon, can you check if i received that" came out as
    refund_request with confidence 1.00. But the reply was invented: "I am sorry you received a
    damaged product".
  - "can i send money by bank app to Bangladesh?" came out as roaming with confidence 0.28, so it
    was escalated. Its draft reply was about telecom packages.
  - The training data explains both:
    - only 20 of 24,263 training messages ask about a refund already requested;
    - 60 of the 139 messages that name a country are roaming;
    - only 11 of 1,488 transfer messages are phrased "can I" or "how do I".
  - So the messages are mostly complaints. Status checks, how-to questions, questions about
    conditions or costs, and off-topic requests are rare.
- **Decision:** about 10,000 new training examples from three sources. Everything below is fixed
  before anything is fetched or generated.
  1. **Bitext** customer support, retail banking and telco (CDLA-Sharing-1.0, at the revisions in
     the ledger).
     - Only Bitext's English messages are used; its replies are not.
     - Their intents map to ours by the label map in `configs/sft/qtypes.yaml`. Labels without a
       clear home are skipped.
     - Up to 250 per intent of ours, spread over the source labels.
  2. **MASSIVE** en-US and ar-SA train splits (CC-BY-4.0): everyday requests such as weather,
     alarms, music and trivia go to `other`; takeaway_query goes to order_status.
  3. **Local Qwen** writes Gulf Arabic, Arabizi and mixed messages on a grid of 21 intents ×
     5 question types × 3 varieties, 3 requests per cell. handoff_to_human × cost is left out.
     - The question types: a status check; how-to or "can I"; conditions or limits; cost; and a
       mention of another country, currency or trip outside roaming.
  - **Replies:** Qwen writes all the new ones, under a stricter rule. They refer only to what the
    customer wrote, and never assume a reason, a product problem, an amount, a date or an outcome.
  - **The same filters as before:** the blind label judge keeps only messages whose label it
    agrees with; the reply check drops broken replies; repeats and near copies of test or dev
    messages are dropped.
- **Question-type dev set:** 15% of the Bitext and MASSIVE messages, per intent of ours and
  seeded. It is held out before anything else and never trained on.
- **Training:** the D-029 recipe on final-v5, which is final-v4 plus the new examples: both
  pretraining bases × peak learning rates 3e-4 and 1e-3, 5 epochs each.
- **Selection:**
  - Guards: SFT-validation accuracy of at least 90.9%, and real-dev accuracy of at least 93.4%
    (the D-033 model's 94.4% minus 1 point).
  - Among the checkpoints that pass, the highest intent accuracy on the question-type dev set
    wins.
  - Within 1 point of the best: higher real-dev accuracy, then higher SFT-validation accuracy.
- **Reply faithfulness:** the D-032 reply judge (Qwen) rates a fixed sample of 300 development
  replies (150 question-type dev, 150 real dev), for the D-033 model and for the chosen one.
  - The share judged "safe" (no invented facts, no claimed actions) and "all four yes" are
    reported before and after.
  - It is not a selection gate.
- **Test:** the chosen model is scored once on the English test set with the D-032 protocol
  (bf16 on the GPU, like Phase 6 and 6b). Quantization and deployment follow as a separate step.
- **Time:** about 2.5–3 h of teacher time (heavy CPU load, and the PC's BIOS is not updated;
  every step resumes after a crash), and about 1.5 h on the GPU.
