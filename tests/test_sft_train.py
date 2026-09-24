"""SFT (Phase 5, D-029): chat format, loss masking, batching, decoding and the metrics."""

import contextlib
import json
import random

import pytest
import torch

from danalm.model.transformer import DanaLM, ModelConfig
from danalm.sft.data import (
    IGNORE,
    INTENT_PREFIX,
    ChatTokens,
    collate,
    example_ids,
    label_continuations,
    length_batches,
    prompt_ids,
)
from danalm.sft.evaluate import (
    coverage_at,
    greedy_answers,
    intent_metrics,
    label_logprobs,
    parse_answer,
)
from danalm.tokenizer.bpe import TokenizerConfig, train

SPECIAL = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "eos": "<|endoftext|>",
    "pad": "<|pad|>",
}
NORM = {"strip_diacritics": True, "unify_alef": False}
INTENTS = ["card_not_working", "order_status", "other"]


def target(intent: str, reply: str) -> str:
    return json.dumps({"intent": intent, "reply": reply}, ensure_ascii=False)


CORPUS = [
    "my card is not working at the ATM",
    "وين الطلب؟ صار له ساعتين",
    target("card_not_working", "Please check the card in the app."),
    target("order_status", "سنتابع طلبك مع فريق التوصيل."),
    target("other", "Hello! How can I help?"),
] * 30


@pytest.fixture(scope="module")
def tok():
    cfg = TokenizerConfig(
        vocab_size=400, min_frequency=1, pretokenizer="standard", eos_token="<|endoftext|>",
        pad_token="<|pad|>", extra_special_tokens=["<|user|>", "<|assistant|>"],
        reserved_tokens=2, placeholder_tokens=["<PHONE>"], corpus="unused", out_dir="unused",
    )  # fmt: skip
    return train(cfg, CORPUS)


@pytest.fixture(scope="module")
def chat(tok):
    return ChatTokens.from_tokenizer(tok, SPECIAL)


def tiny_model(vocab: int) -> DanaLM:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64,
        max_seq_len=128, rope_theta=10000.0, norm_eps=1e-5, init_std=0.02, embed_init_std=1 / 32,
    )  # fmt: skip
    return DanaLM(cfg).eval()


def no_autocast() -> contextlib.AbstractContextManager:
    return contextlib.nullcontext()


def test_example_ids_mask_the_prompt_and_keep_the_answer(tok, chat):
    t = target("card_not_working", "Please check the card in the app.")
    ids, labels = example_ids(tok, chat, "my card is not working at the ATM", t, NORM)
    prompt = prompt_ids(tok, chat, "my card is not working at the ATM", NORM)
    assert ids[0] == chat.user and ids[len(prompt) - 1] == chat.assistant
    assert labels[: len(prompt)] == [IGNORE] * len(prompt)
    assert labels[len(prompt) :] == ids[len(prompt) :]
    assert ids[-1] == chat.eos
    assert tok.decode(ids[len(prompt) : -1]) == t


def test_collate_shifts_targets_and_pads_after_the_tokens(tok, chat):
    a = example_ids(tok, chat, "my card is not working at the ATM", target("other", "Hello!"), NORM)
    b = example_ids(tok, chat, "hi", target("other", "Hello! How can I help?"), NORM)
    x, y = collate([a, b], chat.pad)
    n = max(len(a[0]), len(b[0])) - 1
    assert x.shape == y.shape == (2, n)
    for row, (ids, labels) in enumerate([a, b]):
        k = len(ids) - 1
        assert x[row, :k].tolist() == ids[:-1]
        assert y[row, :k].tolist() == labels[1:]
        assert (x[row, k:] == chat.pad).all() and (y[row, k:] == IGNORE).all()


def test_length_batches_cover_every_example_once_with_similar_lengths():
    lengths = [random.Random(i).randint(5, 60) for i in range(1000)]
    batches = length_batches(lengths, batch_size=16, chunk_batches=8, rng=random.Random(0))
    assert sorted(i for b in batches for i in b) == list(range(1000))
    spread = sum(max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches)
    # a sorted chunk of 128 lengths over 56 values gives batches spanning ~7 values; random ~40
    assert spread / len(batches) < 10


def test_label_continuations_match_the_start_of_training_targets(tok):
    prefix, conts = label_continuations(tok, INTENTS)
    for name, cont in zip(INTENTS, conts, strict=True):
        full = tok.encode(target(name, "Hello!"), add_special_tokens=False).ids
        assert full[: len(prefix) + len(cont)] == prefix + cont
    assert tok.decode(prefix) == INTENT_PREFIX


def test_parse_answer_accepts_only_the_exact_schema_and_known_intents():
    known = set(INTENTS)
    assert parse_answer(target("other", "Hi"), known)["valid"]
    assert not parse_answer(target("fly_to_mars", "Hi"), known)["valid"]
    extra = json.dumps({"intent": "other", "reply": "Hi", "x": 1})
    assert not parse_answer(extra, known)["valid"] and parse_answer(extra, known)["parsed"]
    assert parse_answer('{"intent": "other", "reply": "Hi"', known)["parsed"] is False
    assert parse_answer("[1, 2]", known) == {
        "parsed": True,
        "valid": False,
        "intent": None,
        "reply": None,
    }


def test_intent_metrics_count_invalid_answers_as_wrong():
    m = intent_metrics(["a", "a", "b", "b"], ["a", None, "b", "a"])
    assert m["accuracy"] == 0.5
    # F1(a) = 2*1 / (2*1 + 1 + 1) = 0.5, F1(b) = 2*1 / (2*1 + 0 + 1) = 2/3
    assert m["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_coverage_is_the_largest_confident_share_at_the_target_accuracy():
    conf = [0.9, 0.8, 0.8, 0.7, 0.6, 0.5]
    correct = [True, True, True, False, True, False]
    # >=0.8: 3/3; >=0.7: 3/4; >=0.6: 4/5 = 0.8; >=0.5: 4/6
    assert coverage_at(conf, correct, 0.8) == {"threshold": 0.6, "coverage": 5 / 6, "accuracy": 0.8}
    assert coverage_at(conf, correct, 1.0)["coverage"] == 0.5
    assert coverage_at([0.5], [False], 0.95) == {
        "threshold": None,
        "coverage": 0.0,
        "accuracy": None,
    }


def test_batched_greedy_decoding_equals_one_prompt_at_a_time(tok, chat):
    model = tiny_model(tok.get_vocab_size())
    prompts = [
        prompt_ids(tok, chat, m, NORM) for m in ["hi", "yo", "my card is not working", "وين الطلب"]
    ]
    batched = greedy_answers(model, prompts, 12, chat.eos, batch_size=8, autocast=no_autocast)
    single = [greedy_answers(model, [p], 12, chat.eos, 1, no_autocast)[0] for p in prompts]
    assert batched == single
    reference = model.generate(torch.tensor([prompts[2]]), 12, chat.eos, temperature=0)
    ids, finished = batched[2]
    expected = reference[0, len(prompts[2]) :].tolist()
    assert ids == (expected[:-1] if finished else expected)


def test_label_logprobs_match_a_direct_computation(tok, chat):
    model = tiny_model(tok.get_vocab_size())
    prefix, conts = label_continuations(tok, INTENTS)
    prompt = prompt_ids(tok, chat, "my card is not working", NORM)
    scores = label_logprobs(model, [prompt, prompt], prefix, conts, 1, no_autocast)
    assert scores.shape == (2, len(INTENTS)) and torch.equal(scores[0], scores[1])
    for j, cont in enumerate(conts):
        seq = torch.tensor([prompt + prefix + cont])
        logp = torch.log_softmax(model(seq)[0][0].float(), dim=-1)
        start = len(prompt) + len(prefix)
        direct = sum(logp[t - 1, seq[0, t]].item() for t in range(start, seq.shape[1]))
        assert scores[0, j].item() == pytest.approx(direct, abs=1e-4)
