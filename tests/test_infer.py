"""Inference with a KV cache and without PyTorch (Phase 7, D-034), on tiny CPU models."""

import contextlib
import json
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pytest
import torch

from danalm.infer.decode import greedy_from, label_logprobs_from, prefill
from danalm.infer.onnx import OnnxStep
from danalm.infer.predictor import Predictor
from danalm.model.export import empty_cache, export_onnx, numpy_step
from danalm.model.transformer import DanaLM, ModelConfig
from danalm.sft.evaluate import greedy_answers, label_logprobs
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


def test_predictor_returns_the_schema_and_masks_pii(model, onnx_path, tmp_path):
    tok = train(TokenizerConfig(
        vocab_size=TINY.vocab_size, min_frequency=1, pretokenizer="standard",
        eos_token="<|endoftext|>", pad_token="<|pad|>", extra_special_tokens=["<|user|>", "<|assistant|>"],
        reserved_tokens=2, placeholder_tokens=["<PHONE>"], corpus="unused", out_dir="unused",
    ), ["my card is not working", "where is my order", "call me on 0501234567"] * 20)  # fmt: skip
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "model.onnx").write_bytes(onnx_path.read_bytes())
    meta = {
        "model_file": "model.onnx", "tokenizer_file": "tokenizer.json", "threshold": 0.5,
        "intents": ["card_not_working", "order_status", "other"], "max_new_tokens": 12,
        "special": {"user": "<|user|>", "assistant": "<|assistant|>", "eos": "<|endoftext|>", "pad": "<|pad|>"},
        "normalize": {"strip_diacritics": True, "unify_alef": False},
    }  # fmt: skip
    (tmp_path / "danalm.json").write_text(json.dumps(meta), encoding="utf-8")
    out = Predictor(tmp_path, threads=1).predict("my card is not working, call 0501234567")
    assert set(out) == {"intent", "reply", "confidence", "route", "valid_json", "finished",
                        "message_masked", "latency_ms"}  # fmt: skip
    assert out["message_masked"] == "my card is not working, call <PHONE>"
    assert out["route"] in ("on_device", "escalate") and 0.0 <= out["confidence"] <= 1.0
    if not out["valid_json"]:  # an untrained model: nothing valid, so it must escalate
        assert out["route"] == "escalate" and out["intent"] is None


def test_serving_modules_import_without_torch_or_datasketch():
    code = BLOCK_HEAVY + (
        "import danalm.text, danalm.sft.format\n"
        "import danalm.infer.decode, danalm.infer.onnx, danalm.infer.predictor\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
