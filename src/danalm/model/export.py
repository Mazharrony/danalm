"""DanaLM's KV-cache step for inference and ONNX export (Phase 7, D-034).

- numpy_step: DanaLM.step as a numpy step function for danalm.infer.decode (PyTorch backend).
- ExportStep: the module that is exported. Its output projection is a separate copy of the tied
  embedding, stored as (d_model, vocab), so the quantizers see a constant weight to quantize.
- export_onnx: one graph with dynamic batch, new-token and cache lengths; an empty cache works,
  so the same graph reads the prompt and generates.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from danalm.infer.decode import Step
from danalm.model.transformer import DanaLM


def numpy_step(model: DanaLM) -> Step:
    """DanaLM.step on CPU tensors, taking and returning numpy arrays."""

    @torch.no_grad()
    def step(
        ids: np.ndarray, positions: np.ndarray, past: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        logits, present = model.step(
            torch.from_numpy(ids), torch.from_numpy(positions), [torch.from_numpy(p) for p in past]
        )
        return logits.float().numpy(), [p.numpy() for p in present]

    return step


def empty_cache(model: DanaLM, batch: int) -> list[np.ndarray]:
    return [p.numpy() for p in model.empty_cache(batch)]


class ExportStep(nn.Module):
    """input_ids, positions, past... -> logits, present... (see DanaLM.step_hidden)."""

    def __init__(self, model: DanaLM) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("out_weight", model.embed.weight.detach().t().contiguous().clone())

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, *past: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        hidden, present = self.model.step_hidden(input_ids, positions, list(past))
        return (hidden @ self.out_weight, *present)


def cache_names(n_layers: int) -> tuple[list[str], list[str]]:
    """Input and output names of the cache: past_k_0, past_v_0, ... and present_k_0, ..."""
    names = [f"{kind}_{i}" for i in range(n_layers) for kind in ("k", "v")]
    return [f"past_{n}" for n in names], [f"present_{n}" for n in names]


def export_onnx(model: DanaLM, path: Path, opset: int) -> None:
    """Write the float32 step graph of `model` (on the CPU, in eval mode) to `path`."""
    cfg = model.cfg
    past_names, present_names = cache_names(cfg.n_layers)
    # example sizes other than 0 and 1, so that torch.export keeps them symbolic
    ids = torch.zeros((2, 3), dtype=torch.long)
    positions = torch.arange(5, 8)
    past = [torch.zeros((2, cfg.n_kv_heads, 5, cfg.head_dim)) for _ in past_names]
    batch, new, cached = (torch.export.Dim(n) for n in ("batch", "new", "past"))
    shapes = {
        "input_ids": {0: batch, 1: new},
        "positions": {0: new},
        "past": tuple({0: batch, 2: cached} for _ in past_names),
    }
    with torch.no_grad():
        torch.onnx.export(
            ExportStep(model).eval(),
            (ids, positions, *past),
            str(path),
            dynamo=True,
            opset_version=opset,
            input_names=["input_ids", "positions", *past_names],
            output_names=["logits", *present_names],
            dynamic_shapes=shapes,
            external_data=False,
        )
