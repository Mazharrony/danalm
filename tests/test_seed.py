"""Tests for seeding."""

import os
import random

import numpy as np
import pytest
import torch

from danalm.utils.seed import set_seed


def draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_gives_same_numbers():
    set_seed(123, deterministic=False)
    first = draw()
    set_seed(123, deterministic=False)
    assert draw() == first


def test_different_seeds_give_different_numbers():
    set_seed(1, deterministic=False)
    first = draw()
    set_seed(2, deterministic=False)
    assert draw() != first


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cuda_generator_is_seeded():
    set_seed(7, deterministic=False)
    first = torch.rand(3, device="cuda").cpu()
    set_seed(7, deterministic=False)
    assert torch.equal(torch.rand(3, device="cuda").cpu(), first)


def test_deterministic_flag_toggles_torch_state(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())  # keep env changes inside this test
    try:
        set_seed(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        set_seed(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
