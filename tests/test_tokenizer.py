"""Tests for the BPE tokenizer and the fertility metric."""

import pytest
from tokenizers import Regex, pre_tokenizers
from transformers import AutoTokenizer

from danalm.tokenizer.bpe import PRETOKENIZERS, TokenizerConfig, save, train
from danalm.tokenizer.fertility import fertility, train_flops_per_token

PLACEHOLDERS = ["<URL>", "<EMAIL>", "<EID>", "<IBAN>", "<CARD>", "<PHONE>", "<NUM>"]
CORPUS = [
    "call me on <PHONE> please, my card <CARD> is blocked",
    "رقمي <PHONE> والبطاقة موقوفة، ابي اعرف ليش",
    "3andi mushkila ma3a el card, sa7 ya 7abibi",
    "where is my order? it was due at 5pm on 2026-09-23",
    "الطلب ما وصل order رقم 123 من ساعتين",
] * 40


def make_cfg(pretokenizer: str = "arabizi", vocab_size: int = 600) -> TokenizerConfig:
    return TokenizerConfig(
        vocab_size=vocab_size,
        min_frequency=1,
        pretokenizer=pretokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        extra_special_tokens=["<|user|>", "<|assistant|>"],
        reserved_tokens=2,
        placeholder_tokens=PLACEHOLDERS,
        corpus="unused",
        out_dir="unused",
    )


@pytest.fixture(scope="module")
def tok():
    return train(make_cfg(), CORPUS)


def pieces(pretokenizer: str, text: str) -> list[str]:
    split = pre_tokenizers.Split(Regex(PRETOKENIZERS[pretokenizer]), behavior="isolated")
    return [piece for piece, _ in split.pre_tokenize_str(text)]


def test_arabizi_pretokenizer_keeps_digit_letters_inside_words():
    text = "3andi ma3a sa7 covid19 2026"
    assert pieces("standard", text)[:3] == ["3", "andi", " ma"]
    assert pieces("arabizi", text) == ["3andi", " ma3a", " sa7", " covid", "19", " ", "202", "6"]


def test_diacritics_stay_inside_the_word():
    assert pieces("standard", "مَرْحَبًا بِكُم") == ["مَرْحَبًا", " بِكُم"]


def test_control_tokens_have_fixed_ids_and_placeholders_come_last(tok):
    assert tok.token_to_id("<|endoftext|>") == 0
    assert tok.token_to_id("<|pad|>") == 1
    assert tok.token_to_id("<|reserved_1|>") == 5
    ids = [tok.token_to_id(p) for p in PLACEHOLDERS]
    assert ids == list(range(tok.get_vocab_size() - len(PLACEHOLDERS), tok.get_vocab_size()))


def test_placeholders_are_atomic_never_learned_and_survive_decoding(tok):
    enc = tok.encode("رقمي <PHONE> و <CARD>")
    assert "<PHONE>" in enc.tokens and "<CARD>" in enc.tokens
    assert not [t for t in tok.get_vocab() if "PHON" in t and t != "<PHONE>"]
    assert tok.decode(enc.ids, skip_special_tokens=True) == "رقمي <PHONE> و <CARD>"


@pytest.mark.parametrize(
    "text",
    [
        "شلونك؟ ابي اعرف وين الطلب",
        "3andi mushkila",
        "call 5pm, 2026!",
        "😀 漢字 ñ",
        "  two  spaces ",
    ],
)
def test_round_trip_is_lossless_even_for_unseen_scripts(tok, text):
    assert tok.decode(tok.encode(text, add_special_tokens=False).ids) == text


def test_saved_tokenizer_loads_with_autotokenizer(tok, tmp_path):
    save(tok, make_cfg(), tmp_path)
    hf = AutoTokenizer.from_pretrained(tmp_path)
    assert (hf.eos_token_id, hf.pad_token_id) == (0, 1)
    ids = hf.encode("3andi <CARD> <|user|>", add_special_tokens=False)
    assert hf.decode(ids, skip_special_tokens=True).strip() == "3andi <CARD>"


def test_fertility_counts_tokens_per_whitespace_word(tok):
    texts = ["where is my order", "3andi mushkila"]
    result = fertility(tok, texts)
    expected = sum(len(tok.encode(t, add_special_tokens=False).ids) for t in texts)
    assert (result["words"], result["tokens"]) == (6, expected)
    assert result["fertility"] == pytest.approx(expected / 6)


def test_flops_grow_with_vocabulary():
    small, big = train_flops_per_token(16384, 512, 12), train_flops_per_token(32768, 512, 12)
    assert big - small == 6 * 512 * 16384
