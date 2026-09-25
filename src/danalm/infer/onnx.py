"""The ONNX Runtime backend (Phase 7, D-034): an exported step graph on the CPU."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxStep:
    """A step function (danalm.infer.decode) over a graph from danalm.model.export."""

    def __init__(self, path: str | Path, threads: int) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        self.past_names = [i.name for i in inputs[2:]]
        _, self.kv_heads, _, self.head_dim = inputs[2].shape  # (batch, kv_heads, past, head_dim)

    def empty(self, batch: int) -> list[np.ndarray]:
        """A cache with no positions yet, for the first step."""
        shape = (batch, self.kv_heads, 0, self.head_dim)
        return [np.zeros(shape, dtype=np.float32) for _ in self.past_names]

    def __call__(
        self, ids: np.ndarray, positions: np.ndarray, past: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        feed = {"input_ids": ids, "positions": positions}
        feed.update(zip(self.past_names, past, strict=True))
        logits, *present = self.session.run(None, feed)
        return logits, present
