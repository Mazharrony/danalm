"""Seeding for reproducible runs."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every GPU). Call it first thing in a script.

    Without `deterministic`, the same seed gives the same init, sampling and data order, but some
    GPU kernels may still differ in the last bits between runs. With it, PyTorch uses deterministic
    kernels where they exist (warning where they don't): bitwise-repeatable, but slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CUDA on all devices too
    # Only reaches child processes (e.g. DataLoader workers, which Windows spawns fresh).
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required by cuBLAS for deterministic matmuls; must be set before the first CUDA call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
