"""Inference with a KV cache and without PyTorch (Phase 7, D-034), on tiny CPU models."""

import contextlib
import json
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pytest
import torch

from danalm.eval.cached import evaluate_cached
from danalm.infer.decode import greedy_from, label_logprobs_from, prefill
from danalm.infer.onnx import OnnxStep
from danalm.infer.predictor import MessageTooLong, Predictor
from danalm.model.export import empty_cache, export_onnx, numpy_step
from danalm.model.transformer import DanaLM, ModelConfig
from danalm.sft.evaluate import evaluate, greedy_answers, label_logprobs
from danalm.sft.format import ChatTokens
from danalm.tokenizer.bpe import TokenizerConfig, train

BLOCK_HEAVY = """
import sys


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "datasketch"):
            raise ImportError(f"blocked: {name}")


sys.meta_path.insert(0, Block())
"""
TINY = ModelConfig(
    vocab_size=300, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64,
    max_seq_len=96, rope_theta=10000.0, norm_eps=1e-5, init_std=0.02, embed_init_std=1 / 32,
)  # fmt: skip
PREFIX, CONTS = [5, 6], [[7], [8, 9], [10, 11, 12], [13, 14]]


@pytest.fixture(scope="module")
def model() -> DanaLM:
    torch.manual_seed(0)
    return DanaLM(TINY).eval()


@pytest.fixture(scope="module")
def tok():
    return train(TokenizerConfig(
        vocab_size=TINY.vocab_size, min_frequency=1, pretokenizer="standard",
        eos_token="<|endoftext|>", pad_token="<|pad|>", extra_special_tokens=["<|user|>", "<|assistant|>"],
        reserved_tokens=2, placeholder_tokens=["<PHONE>"], corpus="unused", out_dir="unused",
    ), ["my card is not working", "where is my order", "call me on 0501234567"] * 20)  # fmt: skip


@pytest.fixture(scope="module")
def onnx_path(model, tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_onnx(model, path, opset=18)
    return path


def prompts() -> list[list[int]]:
    rng = np.random.default_rng(0)
    return [rng.integers(20, 300, size=n).tolist() for n in (4, 7, 7, 9, 4, 12)]


def cached(step, empty, ps, eos, max_new):
    """Greedy answers and intent scores with the cache, prompts of equal length together."""
    gens, scores = [None] * len(ps), [None] * len(ps)
    by_len = defaultdict(list)
    for i, p in enumerate(ps):
        by_len[len(p)].append(i)
    for n, idxs in by_len.items():
        last, past = prefill(step, [ps[i] for i in idxs], empty(len(idxs)))
        out = greedy_from(step, last, past, n, eos, max_new)
        lp = label_logprobs_from(step, past, n, PREFIX, CONTS, pad=0)
        for k, i in enumerate(idxs):
            gens[i], scores[i] = out[k], lp[k]
    return gens, np.stack(scores)


def test_cached_decoding_matches_the_full_recompute(model):
    ps, eos = prompts(), 3
    ref_gens = greedy_answers(model, ps, 15, eos, 4, contextlib.nullcontext)
    ref_scores = label_logprobs(model, ps, PREFIX, CONTS, 2, contextlib.nullcontext).numpy()
    gens, scores = cached(numpy_step(model), lambda b: empty_cache(model, b), ps, eos, 15)
    assert gens == ref_gens
    assert np.allclose(scores, ref_scores, atol=1e-4)


def test_onnx_step_matches_pytorch(model, onnx_path):
    step = OnnxStep(onnx_path, threads=1)
    ps, eos = prompts(), 3
    gens, scores = cached(step, step.empty, ps, eos, 15)
    ref_gens, ref_scores = cached(numpy_step(model), lambda b: empty_cache(model, b), ps, eos, 15)
    assert gens == ref_gens
    assert np.allclose(scores, ref_scores, atol=1e-4)
    # the output projection is a constant weight, so quantizers can reach it
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    shapes = {i.name: tuple(i.dims) for i in graph.initializer}
    assert (TINY.d_model, TINY.vocab_size) in shapes.values()


@pytest.fixture(scope="module")
def model_dir(tok, onnx_path, tmp_path_factory):
    """A model directory as scripts/export_onnx.py writes it, for the tiny model."""
    d = tmp_path_factory.mktemp("model_dir")
    tok.save(str(d / "tokenizer.json"))
    (d / "model.onnx").write_bytes(onnx_path.read_bytes())
    meta = {
        "variant": "fp32", "model_file": "model.onnx", "tokenizer_file": "tokenizer.json",
        "threshold": 0.5, "intents": ["card_not_working", "order_status", "other"],
        "max_new_tokens": 12, "model_config": {"max_seq_len": TINY.max_seq_len},
        "special": {"user": "<|user|>", "assistant": "<|assistant|>", "eos": "<|endoftext|>", "pad": "<|pad|>"},
        "normalize": {"strip_diacritics": True, "unify_alef": False},
    }  # fmt: skip
    (d / "danalm.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def test_predictor_returns_the_schema_and_masks_pii(model_dir):
    out = Predictor(model_dir, threads=1).predict("my card is not working, call 0501234567")
    assert set(out) == {"intent", "reply", "confidence", "route", "valid_json", "finished",
                        "message_masked", "latency_ms"}  # fmt: skip
    assert out["message_masked"] == "my card is not working, call <PHONE>"
    assert out["route"] in ("on_device", "escalate") and 0.0 <= out["confidence"] <= 1.0
    if not out["valid_json"]:  # an untrained model: nothing valid, so it must escalate
        assert out["route"] == "escalate" and out["intent"] is None


def test_cached_evaluation_equals_the_d029_evaluation(model, tok):
    special = {
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "eos": "<|endoftext|>",
        "pad": "<|pad|>",
    }
    chat = ChatTokens.from_tokenizer(tok, special)
    intents = ["card_not_working", "order_status", "other"]
    rows = [{"message": m, "intent": i, "variety": "english"} for m, i in [
        ("my card is not working", "card_not_working"), ("where is my order", "order_status"),
        ("call me on 0501234567", "other"), ("my order", "order_status"), ("card", "card_not_working"),
    ]]  # fmt: skip
    norm = {"strip_diacritics": True, "unify_alef": False}
    ev = {"max_new_tokens": 10, "gen_batch": 2, "score_batch": 2, "target_accuracy": 0.95}
    langs = {"english": ["en"]}
    ref_m, ref_p = evaluate(
        model, tok, chat, rows, intents, norm, langs, ev, contextlib.nullcontext
    )
    m, p = evaluate_cached(numpy_step(model), lambda b: empty_cache(model, b), tok, chat, rows,
                           intents, norm, langs, ev, batch=2, label_batch=1)  # fmt: skip
    for a, b in zip(ref_p, p, strict=True):
        assert {k: v for k, v in a.items() if "conf" not in k} == {
            k: v for k, v in b.items() if "conf" not in k
        }
        assert abs(a["conf"] - b["conf"]) < 1e-4 and abs(a["lik_conf"] - b["lik_conf"]) < 1e-4
    assert (
        m["intent_accuracy"] == ref_m["intent_accuracy"] and m["valid_json"] == ref_m["valid_json"]
    )


def test_predictor_refuses_a_message_longer_than_the_context(model_dir):
    with pytest.raises(MessageTooLong):
        Predictor(model_dir, threads=1).predict("where is my order " * 40)


def test_service_predicts_and_rejects_bad_requests(model_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from danalm.serve import app as service

    monkeypatch.setenv("DANALM_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("DANALM_THREADS", "1")
    service.predictor.cache_clear()
    client = TestClient(service.app)
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["variant"] == "fp32" and health["threshold"] == 0.5
    r = client.post("/predict", json={"message": "where is my order? call 0501234567"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] in ("on_device", "escalate") and "<PHONE>" in body["message_masked"]
    assert client.post("/predict", json={"message": ""}).status_code == 422
    assert client.post("/predict", json={"message": "where is my order " * 40}).status_code == 422
    service.predictor.cache_clear()


def test_serving_modules_import_without_torch_or_datasketch():
    code = BLOCK_HEAVY + (
        "import danalm.text, danalm.sft.format\n"
        "import danalm.infer.decode, danalm.infer.onnx, danalm.infer.predictor\n"
        "import danalm.serve.app\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_power_throttling_opt_out_only_reports_what_happened():
    from danalm.utils.power import disable_power_throttling

    assert disable_power_throttling() is (sys.platform == "win32")
