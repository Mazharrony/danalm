"""Record this machine's environment and measure what later phases depend on.

- bf16/fp16 matmul throughput: the ceiling for training speed, used later for MFU and time estimates
- which fused attention kernels scaled_dot_product_attention can use on this PyTorch build
- whether torch.compile works (it needs Triton, which is not bundled on Windows)

Usage: uv run python scripts/check_env.py --config configs/check_env.yaml [key=value ...]
"""

import json
import re
import time
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from danalm.config import config_from_cli
from danalm.utils.run import start_run
from danalm.utils.seed import set_seed


def matmul_tflops(n: int, iters: int, dtype: torch.dtype) -> float:
    """Achieved TFLOP/s for an n x n x n matmul on the current GPU."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):  # warm-up
        a @ b
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        a @ b
    end.record()
    torch.cuda.synchronize()
    return 2 * n**3 * iters / (start.elapsed_time(end) / 1000) / 1e12


def attention_backends(c: dict[str, Any]) -> dict[str, str]:
    """Try each SDPA kernel on a causal bf16 call; report its time or why it is unavailable."""
    shape = (c["attn_batch"], c["attn_heads"], c["attn_seq_len"], c["attn_head_dim"])
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    results = {}
    for backend in (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ):
        with warnings.catch_warnings(record=True) as caught, sdpa_kernel(backend):
            warnings.simplefilter("always")
            try:
                F.scaled_dot_product_attention(q, k, v, is_causal=True)  # warm-up
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(c["attn_iters"]):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / c["attn_iters"] * 1e3
                results[backend.name] = f"ok, {ms:.2f} ms/call"
            except RuntimeError:
                results[backend.name] = f"unavailable ({_kernel_reason(caught)})"
    return results


def _kernel_reason(caught: list[warnings.WarningMessage]) -> str:
    """The informative part of PyTorch's 'kernel not used' warnings, without boilerplate."""
    reasons = []
    for w in caught:
        msg = re.sub(r"\s*\(Triggered internally at .*?\)", "", str(w.message), flags=re.S).strip()
        if msg and not msg.endswith("because:") and "runtime disabled" not in msg:
            reasons.append(msg)
    return "; ".join(dict.fromkeys(reasons)) or "no reason given"


def try_compile() -> str:
    """Compile a tiny function with the default Inductor backend and run it once."""
    try:
        fn = torch.compile(lambda x: F.silu(x) * x)
        fn(torch.randn(1024, device="cuda"))
        return "ok"
    except Exception as err:  # diagnostic: report any failure instead of crashing
        first_sentence = str(err).strip().split(". ")[0][:160]
        return f"unavailable ({type(err).__name__}: {first_sentence})"


def main() -> None:
    cfg = config_from_cli(__doc__)
    set_seed(cfg["seed"], cfg["deterministic"])
    run = start_run(cfg, job_type="check-env")
    c = cfg["check_env"]

    results: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        results["vram_total_gb"] = round(total / 2**30, 2)
        results["vram_free_gb"] = round(free / 2**30, 2)
        results["bf16_supported"] = torch.cuda.is_bf16_supported()
        for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            tflops = matmul_tflops(c["matmul_size"], c["matmul_iters"], dtype)
            results[f"matmul_tflops_{name}"] = round(tflops, 1)
        results["attention"] = attention_backends(c)
        results["torch_compile"] = try_compile() if c["try_compile"] else "skipped"
        results["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    for key, value in {**run.meta["env"], **results}.items():
        if isinstance(value, dict):
            for sub, sub_value in value.items():
                print(f"  {key}.{sub:<22} {sub_value}")
        else:
            print(f"  {key:<32} {value}")
    (run.dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    run.wandb.summary.update(results)
    run.wandb.finish()
    print(f"\nSaved to {run.dir}")


if __name__ == "__main__":
    main()
