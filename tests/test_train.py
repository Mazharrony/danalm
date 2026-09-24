"""Tests for the training utilities (Phase 4): schedule, optimizer groups, resume, data order."""

import contextlib

import numpy as np
import pytest
import torch

from danalm.data.shards import gather_windows, window_index
from danalm.model.transformer import DanaLM, ModelConfig
from danalm.train.loop import (
    checkpoints,
    lr_at,
    make_optimizer,
    prune_checkpoints,
    save_checkpoint,
    set_lr,
    train_step,
)

TINY = ModelConfig(
    vocab_size=61, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64,
    max_seq_len=16, rope_theta=10000.0, norm_eps=1e-5, init_std=0.02, embed_init_std=1 / 32,
)  # fmt: skip


def test_lr_warms_up_linearly_then_follows_a_cosine_to_the_minimum():
    kw = {"total_steps": 110, "warmup_steps": 10, "peak": 1e-3, "minimum": 1e-4}
    assert lr_at(0, **kw) == pytest.approx(1e-4)
    assert lr_at(9, **kw) == pytest.approx(1e-3)
    assert lr_at(10, **kw) == pytest.approx(1e-3)
    assert lr_at(60, **kw) == pytest.approx((1e-3 + 1e-4) / 2)
    assert lr_at(110, **kw) == pytest.approx(1e-4) and lr_at(500, **kw) == pytest.approx(1e-4)
    after = [lr_at(s, **kw) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(after, after[1:], strict=False))


def test_weight_decay_applies_to_matrices_and_embeddings_only():
    model = DanaLM(TINY)
    opt = make_optimizer(model, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=False)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert any(p is model.embed.weight for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])  # RMSNorm weights
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == sum(p.numel() for p in model.parameters())


def _batches(step: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(step)  # data order depends only on the step
    w = torch.randint(0, TINY.vocab_size, (2, 2, 9), generator=g)
    return [(b[:, :-1], b[:, 1:]) for b in w]


def _train(model: DanaLM, opt: torch.optim.Optimizer, steps: range) -> None:
    for step in steps:
        set_lr(opt, lr_at(step, total_steps=6, warmup_steps=2, peak=1e-2, minimum=1e-3))
        train_step(model, opt, _batches(step), grad_clip=1.0, autocast=contextlib.nullcontext)


def test_resuming_from_a_checkpoint_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(0)
    straight = DanaLM(TINY)
    opt = make_optimizer(straight, 1e-2, (0.9, 0.95), 0.1, fused=False)
    _train(straight, opt, range(6))

    torch.manual_seed(0)
    first = DanaLM(TINY)
    opt = make_optimizer(first, 1e-2, (0.9, 0.95), 0.1, fused=False)
    _train(first, opt, range(3))
    save_checkpoint(tmp_path / "step_3.pt", {"model": first.state_dict(), "optimizer": opt.state_dict()})  # fmt: skip

    resumed = DanaLM(TINY)
    opt = make_optimizer(resumed, 1e-2, (0.9, 0.95), 0.1, fused=False)
    state = torch.load(tmp_path / "step_3.pt", weights_only=True)
    resumed.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    _train(resumed, opt, range(3, 6))
    for a, b in zip(straight.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_checkpoints_are_listed_by_step_and_pruned(tmp_path):
    for step in (5, 40, 300):
        save_checkpoint(tmp_path / f"step_{step}.pt", {"step": step})
    assert [p.name for p in checkpoints(tmp_path)] == ["step_5.pt", "step_40.pt", "step_300.pt"]
    prune_checkpoints(tmp_path, keep=2)
    assert [p.name for p in checkpoints(tmp_path)] == ["step_40.pt", "step_300.pt"]
    assert not list(tmp_path.glob("*.tmp"))


def test_windows_make_every_token_a_target_exactly_once():
    shards = [np.arange(0, 10, dtype=np.uint16), np.arange(100, 107, dtype=np.uint16)]
    idx = window_index([len(s) for s in shards], seq_len=3)
    windows = gather_windows(shards, idx, seq_len=3)
    assert windows.shape == (len(idx), 4) and (np.diff(windows, axis=1) == 1).all()
    targets = windows[:, 1:].ravel().tolist()
    assert len(targets) == len(set(targets))  # no token is a target twice
    assert targets == [1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 103, 104, 105, 106]
    order = np.random.default_rng(0).permutation(len(idx))
    assert (order == np.random.default_rng(0).permutation(len(idx))).all()
