"""Tests for the data pipeline: normalization, PII masking, language tags, filters, dedup, split."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from danalm.config import load_config
from danalm.data.pipeline import (
    PipelineConfig,
    detect_lang,
    in_val,
    normalize,
    quality_ok,
    read_inputs,
    run_pipeline,
)
from danalm.data.shards import ShardWriter, read_shard, tokenize_split, verify_split

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "raw"


def norm(text: str) -> str:
    return normalize(text, strip_diacritics=True, unify_alef=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def cfg(tmp_path) -> PipelineConfig:
    """configs/data/smoke.yaml, pointed at absolute fixture paths and a temp output dir."""
    data = load_config(REPO / "configs" / "data" / "smoke.yaml")["data"]
    data |= {"inputs": [str(FIXTURES / "*")], "out_dir": str(tmp_path / "out")}
    return PipelineConfig(**data)


def make_tiny_tokenizer(path: Path) -> Path:
    """A ~300-token byte-level BPE in Hugging Face format, for tests only."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(["where is my order", "وين الطلب مالي"], trainer)
    PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>").save_pretrained(path)
    return path


# ---------------------------------------------------------------- normalization
def test_diacritics_and_tatweel_are_removed():
    assert norm("مَرْحَبـــًا بِكُم") == "مرحبا بكم"


def test_diacritics_can_be_kept():
    assert normalize("مَرْحَبًا", strip_diacritics=False, unify_alef=False) == "مَرْحَبًا"


def test_alef_forms_and_alef_maksura_are_unified():
    assert norm("أنا إلى آخر") == "انا الي اخر"
    assert normalize("أنا إلى", strip_diacritics=True, unify_alef=False) == "أنا إلى"


def test_nfkc_expands_arabic_presentation_forms():
    assert norm("ﻻ") == "لا"  # lam-alef ligature


def test_long_character_runs_are_capped_at_three():
    assert norm("هههههههه sooooo") == "ههه sooo"


def test_whitespace_is_collapsed():
    assert norm("  hello \n\t world  ") == "hello world"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("call +971 50 123 4567 now", "call <PHONE> now"),
        ("call 00971501234567 now", "call <PHONE> now"),
        ("call 050-123-4567 now", "call <PHONE> now"),
        ("ID 784-1990-1234567-1 ok", "ID <EID> ok"),
        ("mail a.b+c@example.ae now", "mail <EMAIL> now"),
        ("see https://example.ae/x?y=1 and www.example.com", "see <URL> and <URL>"),
    ],
)
def test_pii_is_masked(raw, expected):
    assert norm(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # the repeat cap used to shorten digit runs: 1000000 -> 1000
        ("I was charged 1000000 AED, not 100000", "I was charged 1000000 AED, not 100000"),
        # Arabic-Indic digits used to bypass every PII pattern
        ("رقمي ٠٥٠١٢٣٤٥٦٧", "رقمي <PHONE>"),
        ("هويتي ٧٨٤-١٩٩٠-١٢٣٤٥٦٧-١", "هويتي <EID>"),
        # a number glued to Arabic letters has no \b word boundary
        ("رقمي0501234567", "رقمي<PHONE>"),
        # UAE landlines were not covered
        ("call 04 123 4567", "call <PHONE>"),
        # cards, IBANs and long account numbers were not covered (banking domain)
        ("card 4111 1111 1111 1111 blocked", "card <CARD> blocked"),
        ("card 4111-1111-1111-1111 blocked", "card <CARD> blocked"),
        ("card 4111111111111111 blocked", "card <CARD> blocked"),
        ("amex 3782 822463 10005 ok", "amex <CARD> ok"),
        ("IBAN AE07 0331 2345 6789 0123 456 ok", "IBAN <IBAN> ok"),
        ("iban ae070331234567890123456", "iban <IBAN>"),
        ("account 123456789012 please", "account <NUM> please"),
        # must stay untouched
        ("pay 250 AED on 2026-09-23 at 14:30", "pay 250 AED on 2026-09-23 at 14:30"),
        ("order 12345678 is late", "order 12345678 is late"),
    ],
)
def test_pii_and_number_regressions(raw, expected):
    assert norm(raw) == expected


# ---------------------------------------------------------------- language tags
@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("وين الطلب مالي", "ar"),
        ("where is my order", "en"),
        ("3andi mushkila fil card", "arabizi"),
        ("الطلب ما وصل order رقم 123 من ساعتين", "mixed"),
        ("123 456 !!", "other"),
    ],
)
def test_detect_lang(text, lang):
    assert detect_lang(text) == lang


@pytest.mark.parametrize(
    ("raw", "lang"),
    [
        # PII placeholders are Latin letters and used to push Arabic text to "mixed"
        ("ابي اغير رقمي 0501234567", "ar"),
        ("ابي اغير الايميل الى a.b@example.ae", "ar"),
        # English number+suffix tokens used to count as Arabizi evidence
        ("my 2nd card was blocked at 5pm", "en"),
        ("I did not get the 2FA code", "en"),
        ("3DS verification failed", "en"),
        # one model number in a longer English text used to make the whole text Arabizi
        ("The 7up bottling plant in our town employed many workers during the summer season", "en"),
        ("Vanadium oxide V2O5 is a catalyst used in the contact process for sulfuric acid", "en"),
        # Arabizi without digits used to be tagged English
        ("laish ma yshtaghil el card", "arabizi"),
        ("shlon asawi reset lel password", "arabizi"),
    ],
)
def test_detect_lang_regressions(raw, lang):
    assert detect_lang(norm(raw)) == lang


def test_reader_handles_bom_and_non_object_json_lines(tmp_path):
    lines = ['﻿{"text": "first row after a BOM"}', "[1, 2]", '"just a string"']
    (tmp_path / "bom.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert list(read_inputs([str(tmp_path / "*.jsonl")], "text")) == [
        ("first row after a BOM", "bom")
    ]


# ---------------------------------------------------------------- quality filter
@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("ok", "too_short"),
        ("x" * 5000, "too_long"),
        ("12345 67890 !!!", "low_letter_ratio"),
        ("yes yes yes yes yes yes yes yes", "repetitive"),
        ("Where is my order? It is late.", "ok"),
    ],
)
def test_quality_filter(cfg, text, reason):
    assert quality_ok(text, cfg)[1] == reason


def test_config_rejects_unknown_and_missing_keys(cfg):
    fields = asdict(cfg)
    with pytest.raises(TypeError):
        PipelineConfig(**fields, typo=1)
    fields.pop("val_frac")
    with pytest.raises(TypeError):
        PipelineConfig(**fields)


# ---------------------------------------------------------------- end to end
def test_pipeline_on_fixture(cfg):
    stats = run_pipeline(cfg, seed=0)
    out = Path(cfg.out_dir)
    train, val = read_jsonl(out / "train.jsonl"), read_jsonl(out / "val.jsonl")
    texts = [row["text"] for row in train + val]

    assert stats["read"] == 17  # 11 valid JSONL rows + 4 text lines + 2 CSV rows
    assert stats["drop_too_short"] == 1
    assert stats["drop_repetitive"] == 1
    assert stats["drop_exact_dup"] == 1
    assert stats["drop_near_dup"] == 1
    assert len(train) + len(val) == 13
    assert (stats["train_samples"], stats["val_samples"]) == (len(train), len(val))
    assert all(in_val(row["text"], 0, cfg.val_frac) for row in val)
    assert not any(in_val(row["text"], 0, cfg.val_frac) for row in train)
    assert len(set(texts)) == len(texts)
    assert sum(v for k, v in stats.items() if k.startswith("lang_")) == len(texts)
    assert any("<PHONE>" in t for t in texts)
    assert any("<EID>" in t for t in texts)
    assert not any("4567" in t for t in texts)  # the masked digits never reach the output
    phone_row = next(row for row in train + val if "<PHONE>" in row["text"])
    assert phone_row["lang"] == "ar"  # the placeholder must not make it "mixed"
    assert json.loads((out / "stats.json").read_text(encoding="utf-8")) == dict(stats)


def test_pipeline_is_deterministic_for_a_seed(cfg, tmp_path):
    run_pipeline(cfg, seed=0)
    first = (Path(cfg.out_dir) / "train.jsonl").read_bytes()
    run_pipeline(cfg, seed=0)
    assert (Path(cfg.out_dir) / "train.jsonl").read_bytes() == first
    other = replace(cfg, out_dir=str(tmp_path / "other"))
    run_pipeline(other, seed=1)
    assert (Path(other.out_dir) / "train.jsonl").read_bytes() != first


def test_missing_inputs_raise(cfg):
    with pytest.raises(FileNotFoundError):
        run_pipeline(replace(cfg, inputs=["does/not/exist/*.jsonl"]), seed=0)


def test_hash_split_is_close_to_val_frac_and_seed_dependent():
    texts = [f"message number {i}" for i in range(20000)]
    share = sum(in_val(t, 0, 0.1) for t in texts) / len(texts)
    assert 0.09 < share < 0.11
    assert [in_val(t, 0, 0.1) for t in texts] != [in_val(t, 1, 0.1) for t in texts]


# ---------------------------------------------------------------- token shards
def test_shards_hold_every_token_with_eos_after_each_document(cfg, tmp_path):
    run_pipeline(cfg, seed=0)
    tok = Tokenizer.from_file(str(make_tiny_tokenizer(tmp_path / "tok") / "tokenizer.json"))
    eos_id = tok.token_to_id("<eos>")
    writer = ShardWriter(tmp_path / "shards", "train", shard_tokens=100)
    (tmp_path / "shards").mkdir()
    counts = tokenize_split(Path(cfg.out_dir) / "train.jsonl", tok, eos_id, writer, batch_docs=4)
    files = writer.close()

    ids = np.concatenate([read_shard(tmp_path / "shards" / f["file"]) for f in files])
    rows = read_jsonl(Path(cfg.out_dir) / "train.jsonl")
    assert all(f["tokens"] == 100 for f in files[:-1]) and 0 < files[-1]["tokens"] <= 100
    assert len(ids) == sum(c["tokens"] for c in counts.values()) + len(rows)  # + one EOS each
    assert (ids == eos_id).sum() == len(rows) and ids[-1] == eos_id
    assert sum(c["docs"] for c in counts.values()) == len(rows)
    first_doc = ids[: int(np.argmax(ids == eos_id))]
    assert tok.decode(first_doc.tolist()) == rows[0]["text"]


def test_verify_split_passes_on_good_shards_and_catches_a_corrupted_token(cfg, tmp_path):
    run_pipeline(cfg, seed=0)
    tok = Tokenizer.from_file(str(make_tiny_tokenizer(tmp_path / "tok") / "tokenizer.json"))
    eos_id = tok.token_to_id("<eos>")
    jsonl = Path(cfg.out_dir) / "train.jsonl"
    (tmp_path / "shards").mkdir()
    writer = ShardWriter(tmp_path / "shards", "train", shard_tokens=100)
    tokenize_split(jsonl, tok, eos_id, writer, batch_docs=4)
    files = [tmp_path / "shards" / f["file"] for f in writer.close()]

    good = verify_split(files, jsonl, tok, eos_id, sample_docs=10_000, seed=0)
    assert good["ok"] and good["boundary_docs"] > 0 and good["checked_docs"] == good["docs"]

    ids = np.fromfile(files[0], dtype=np.uint16)  # change the first token of document 0
    z, q = tok.token_to_id("Z"), tok.token_to_id("Q")
    ids[0] = z if ids[0] != z else q
    ids.tofile(files[0])
    bad = verify_split(files, jsonl, tok, eos_id, sample_docs=10_000, seed=0)
    assert not bad["ok"] and bad["mismatched_docs"] == [0]
