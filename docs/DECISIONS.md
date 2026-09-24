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
