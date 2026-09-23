"""Run bookkeeping: each run gets a folder holding its resolved config, git commit and environment,
plus a W&B run carrying the same information. With the seed in the config, that is everything
needed to reproduce the run.
"""

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from danalm.config import save_config


@dataclass
class Run:
    """A started run. `wandb` is the W&B run; with tracking.mode=disabled its methods are no-ops."""

    name: str
    dir: Path
    meta: dict[str, Any]
    wandb: Any


def start_run(cfg: dict[str, Any], job_type: str) -> Run:
    """Create `<runs_dir>/<run_name>-<timestamp>/`, write provenance files and start W&B."""
    import wandb  # slow import; keep it out of module import time

    if not cfg.get("run_name"):
        raise ValueError("config must set run_name")
    name = f"{cfg['run_name']}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(cfg["paths"]["runs_dir"]) / name
    meta = write_provenance(run_dir, cfg, {"run_name": name, "job_type": job_type})
    t = cfg["tracking"]
    wb = wandb.init(
        project=t["project"],
        entity=t["entity"],
        mode=t["mode"],
        name=name,
        job_type=job_type,
        tags=t["tags"],
        dir=run_dir,
        config={**cfg, "meta": meta},
    )
    return Run(name=name, dir=run_dir, meta=meta, wandb=wb)


def write_provenance(
    out_dir: Path, cfg: dict[str, Any], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Write config.yaml, meta.json and (if the git tree is dirty) git_diff.patch into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    git = git_info()
    meta = {
        **(extra or {}),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(["python", *sys.argv]),
        "seed": cfg.get("seed"),
        "git": git,
        "env": env_info(),
    }
    save_config(cfg, out_dir / "config.yaml")
    if git["dirty"]:
        (out_dir / "git_diff.patch").write_text(_git("diff", "HEAD"), encoding="utf-8")
        print(f"[warn] uncommitted changes saved to {out_dir / 'git_diff.patch'}; commit first")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    return meta


def git_info(cwd: str | Path = ".") -> dict[str, Any]:
    """Commit hash, dirty flag and untracked files; all None outside a git repository."""
    try:
        commit = _git("rev-parse", "HEAD", cwd=cwd)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None, "untracked": None}
    status = _git("status", "--porcelain", cwd=cwd).splitlines()
    return {
        "commit": commit,
        "dirty": bool(status),
        "untracked": [line[3:] for line in status if line.startswith("??")],
    }


def env_info() -> dict[str, Any]:
    """Python, OS, PyTorch, CUDA and GPU versions for the run record."""
    import torch

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_capability"] = f"{props.major}.{props.minor}"
        info["gpu_memory_gb"] = round(props.total_memory / 2**30, 2)
    return info


def _git(*args: str, cwd: str | Path = ".") -> str:
    # utf-8 explicitly: the Windows default code page cannot decode Arabic text in diffs.
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return result.stdout.strip()
