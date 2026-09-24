"""Tests for the DanaLM transformer (Phase 3) on a tiny CPU-sized configuration."""

import math

import pytest
import torch

from danalm.model.transformer import (
    DanaLM,
    ModelConfig,
    apply_rope,
    count_params,
    flops_per_token,
    rope_tables,
)

TINY = ModelConfig(
    vocab_size=97, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64,
    max_seq_len=32, rope_theta=10000.0, norm_eps=1e-5, init_std=0.02, embed_init_std=1 / 32,
)  # fmt: skip


@pytest.fixture
def model() -> DanaLM:
    torch.manual_seed(0)
    return DanaLM(TINY).eval()


def test_forward_shapes_and_initial_loss_near_uniform(model):
    idx = torch.randint(0, TINY.vocab_size, (4, 16))
    targets = torch.randint(0, TINY.vocab_size, (4, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (4, 16, TINY.vocab_size)
    # near-uniform predictions at init, so the loss starts near ln(vocab_size)
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.1


def test_embedding_init_keeps_the_repeat_input_logit_small():
    """Tied embeddings: an input token's own logit starts near embed_init_std * d_model."""
    torch.manual_seed(0)
    for std, low, high in ((1 / 32, 0.5, 1.5), (0.3, 7.0, 12.0)):
        m = DanaLM(ModelConfig(**{**TINY.__dict__, "embed_init_std": std})).eval()
        idx = torch.randint(0, TINY.vocab_size, (1, 16))
        logits, _ = m(idx)
        own = logits[0].gather(-1, idx[0, :, None]).mean().item()
        assert low < own < high


def test_future_tokens_do_not_change_earlier_logits(model):
    idx = torch.randint(0, TINY.vocab_size, (1, 12))
    changed = idx.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % TINY.vocab_size
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-6)
    assert not torch.allclose(a[0, 8:], b[0, 8:])


def test_output_embedding_is_tied_and_param_count_matches(model):
    assert model.embed.weight.data_ptr() == next(model.parameters()).data_ptr()
    kv = TINY.n_kv_heads * TINY.head_dim
    per_layer = 2 * TINY.d_model**2 + 2 * TINY.d_model * kv + 3 * TINY.d_model * TINY.ffn_hidden
    per_layer += 2 * TINY.d_model  # two RMSNorm weights
    expected = TINY.vocab_size * TINY.d_model + TINY.n_layers * per_layer + TINY.d_model
    assert count_params(model)["total"] == expected
    assert count_params(model)["non_embedding"] == expected - TINY.vocab_size * TINY.d_model


def test_rope_scores_depend_only_on_relative_position():
    cos, sin = rope_tables(head_dim=8, seq_len=16, theta=10000.0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

    def score(pos_q: int, pos_k: int) -> float:
        qs = apply_rope(q, cos[pos_q : pos_q + 1], sin[pos_q : pos_q + 1])
        ks = apply_rope(k, cos[pos_k : pos_k + 1], sin[pos_k : pos_k + 1])
        return (qs * ks).sum().item()

    assert score(5, 2) == pytest.approx(score(12, 9), abs=1e-5)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-3)


def test_masked_targets_are_ignored(model):
    idx = torch.randint(0, TINY.vocab_size, (2, 10))
    targets = idx.clone()
    targets[:, :5] = -100  # e.g. the prompt of an SFT example
    _, loss_masked = model(idx, targets)
    logits, _ = model(idx)
    manual = torch.nn.functional.cross_entropy(
        logits[:, 5:].reshape(-1, TINY.vocab_size), idx[:, 5:].reshape(-1)
    )
    assert loss_masked.item() == pytest.approx(manual.item(), rel=1e-5)


def test_can_overfit_a_single_batch():
    torch.manual_seed(0)
    model = DanaLM(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 17))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_is_greedy_and_stops_at_eos(model):
    idx = torch.tensor([[1, 2, 3]])
    out = model.generate(idx, max_new_tokens=5, eos_id=0, temperature=0)
    assert out.shape[1] <= 8 and torch.equal(out[:, :3], idx)
    assert torch.equal(out, model.generate(idx, max_new_tokens=5, eos_id=0, temperature=0))
    logits, _ = model(idx)
    eos_first = model.generate(idx, max_new_tokens=5, eos_id=int(logits[0, -1].argmax()), temperature=0)  # fmt: skip
    assert eos_first.shape[1] == 4  # stopped right after the first (eos) token


def test_config_rejects_bad_head_counts_and_long_input(model):
    with pytest.raises(ValueError):
        ModelConfig(**{**TINY.__dict__, "n_kv_heads": 3})
    with pytest.raises(ValueError):
        model(torch.zeros(1, TINY.max_seq_len + 1, dtype=torch.long))


def test_flops_per_token_counts_weights_and_attention():
    weights = (
        TINY.n_layers
        * (2 * TINY.d_model**2 + 2 * TINY.d_model * 16 + 3 * TINY.d_model * TINY.ffn_hidden)
        + TINY.d_model * TINY.vocab_size
    )
    assert flops_per_token(TINY, seq_len=32) == 6 * weights + 6 * TINY.n_layers * TINY.d_model * 32
