"""The pieces of the pretraining and fine-tuning loops (Phase 4, D-028).

The loop itself lives in the scripts; these functions are small enough to test on the CPU:
the learning-rate schedule, the optimizer's parameter groups, one optimizer step with gradient
accumulation, and crash-safe checkpoints (write to a temporary file, then rename).
"""

import math
import os
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
from torch import nn


def lr_at(step: int, total_steps: int, warmup_steps: int, peak: float, minimum: float) -> float:
    """Linear warmup from 0 to `peak` over `warmup_steps`, then cosine decay to `minimum` at
    `total_steps` (and `minimum` after that)."""
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def make_optimizer(
    model: nn.Module, lr: float, betas: tuple[float, float], weight_decay: float, fused: bool
) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices and embeddings only (not on norm weights)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=fused)


def train_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    micro_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    grad_clip: float,
    autocast: Callable[[], AbstractContextManager],
) -> tuple[float, float]:
    """One optimizer step over the given micro-batches (gradients averaged over them).
    Returns (mean loss, gradient norm before clipping)."""
    batches = list(micro_batches)
    total = torch.zeros((), device=next(model.parameters()).device)
    for x, y in batches:
        with autocast():
            _, loss = model(x, y)
        (loss / len(batches)).backward()
        total += loss.detach()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return (total / len(batches)).item(), grad_norm.item()


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Write atomically: a crash mid-write leaves the previous checkpoint intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def checkpoints(folder: Path) -> list[Path]:
    """Checkpoints named step_<n>.pt, oldest first."""
    return sorted(folder.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def prune_checkpoints(folder: Path, keep: int) -> None:
    for old in checkpoints(folder)[:-keep]:
        old.unlink()
