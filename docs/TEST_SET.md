# Human test set (Phase 2c)

The test set is the only data that measures DanaLM honestly. It is **never** used for training,
tuning, prompt design or choosing between models; it is only for the final evaluation (Phase 6).

## File format

`data/test/test_set.jsonl` (gitignored), one JSON object per line:

```json
{"id": "t0001", "text": "البطاقة مب راضية تشتغل", "intent": "card_not_working", "variety": "gulf_arabic", "author": "AB", "verified_by": "CD", "source": "written", "notes": ""}
```

| Field | Meaning |
|---|---|
| `id` | `t0001`, `t0002`, ... (never reused) |
| `text` | the customer message exactly as a customer would type it |
| `intent` | one of the 21 intents in `configs/sft/intents.yaml` |
| `variety` | `gulf_arabic`, `english`, `arabizi` or `mixed` (a person's label, not the heuristic) |
| `author` | initials of the person who wrote it, or the dataset it comes from |
| `verified_by` | initials of a second person who checked the intent and variety |
| `source` | `written`, `banking77-test` or `clinc150-test` |
| `notes` | anything unusual, e.g. "could also be refund_request" |

## Size and mix

About **420 messages**: 5 per intent per variety (21 × 4 × 5). The minimum is 3 per cell, so
the per-intent and per-variety numbers in Phase 6 mean something.

## Rules for writing messages

1. Write the way real customers type on WhatsApp: short and long, polite and angry, with typos,
   and sometimes two problems in one message.
2. **Do not look at the synthetic training data or any teacher output while writing,** and do not
   copy from them. The test set must be independent of them.
3. **No AI tools** for writing or correcting (D-018). Everything must be written by a person.
4. **No real personal data.** Use fake numbers (e.g. 050 000 0000) and invented names.
5. One intent per message. If a message fits two intents, pick the one that decides the next
   action, and write the other in `notes`.
6. Gulf Arabic, Arabizi and mixed messages must be written, or at least checked, by a
   **native Gulf Arabic speaker**.

## Human-written public data we may use (English only)

The test splits of two licence-compatible datasets were written by people:

- **Banking77** (CC-BY-4.0): online-banking questions labelled by intent. Its documentation does
  not say who wrote them; the licence was checked in the original PolyAI repository.
- **CLINC150** (CC-BY-3.0): includes out-of-scope questions, which map to `other`.

Their labels are mapped to our intents by rule, and then **a person checks every mapped example**
before it enters the test set (`source` records which dataset it came from). Only their test
splits may be used, never the train splits that feed training data.

The mapping covers 10 of the 21 intents (banking, `account_access`, `order_status`, `other`).
The telecom and most delivery intents have no public English test data, so a person writes those.
The steps (`configs/data/test_candidates_en.yaml`):

```bash
uv run python scripts/fetch_labelled.py --config configs/data/test_candidates_en.yaml
uv run python scripts/check_overlap.py --config configs/data/overlap.yaml "overlap.test_files=[\"data/test/candidates-en/messages.jsonl\"]" overlap.test_field=message
uv run python scripts/make_review_sheet.py --config configs/data/test_candidates_en.yaml
```

1. Open `data/test/candidates-en/review.csv` (Excel works).
2. For each row, write `y` or `n` in `keep`.
3. If the proposed intent is wrong, write the right one in `correct_intent`. Anything unusual
   goes in `notes`.
4. Import the kept rows with your initials, then re-run the overlap check on the test set:

```bash
uv run python scripts/import_review.py --config configs/data/test_candidates_en.yaml review.verified_by=XX
uv run python scripts/check_overlap.py --config configs/data/overlap.yaml
```

Candidates that overlap training data never reach the sheet. The first run dropped 48 of 160,
mostly CLINC150 test sentences that nearly repeat its train split.

## Checks before every evaluation

```bash
uv run python scripts/check_overlap.py --config configs/data/overlap.yaml
```

It must report **0 overlaps**: no test message may be an exact or near copy of anything in the
training data. When the test file changes, record its SHA-256 and row count in
`docs/DATA_LEDGER.md`.
