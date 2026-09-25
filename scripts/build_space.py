"""Assemble the Hugging Face Space folder (Phase 7, D-034).

Usage: uv run python scripts/build_space.py --config configs/deploy/phase7.yaml
Writes phase7.space_dir: space/app.py, README.md and requirements.txt, the
torch-free danalm modules (the same files as the Docker image), and the deployed model directory
from phase7.selected as model/. Publishing it is the owner's step, e.g.
  hf upload <user>/danalm artifacts/space . --repo-type space
Nothing is uploaded here.
"""

import json
import shutil
from pathlib import Path

from danalm.config import config_from_cli

MODULES = [  # the torch-free modules of the Docker image, without the HTTP service
    "src/danalm/__init__.py", "src/danalm/text.py", "src/danalm/sft/__init__.py",
    "src/danalm/sft/format.py", "src/danalm/infer",
]  # fmt: skip


def main() -> None:
    cfg = config_from_cli(__doc__)
    p = cfg["phase7"]
    out = Path(p["space_dir"])
    if out.exists() and any(
        out.iterdir()
    ):  # only an empty folder or a previous build of this script
        if not (out / "app.py").exists() or not (out / "model" / "danalm.json").exists():
            raise SystemExit(f"{out} exists and is not a Space build; choose another space_dir")
        for child in out.iterdir():  # empty it; the folder itself may be in use
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("app.py", "README.md", "requirements.txt"):
        shutil.copyfile(Path("space") / name, out / name)
    ignore = shutil.ignore_patterns("__pycache__")
    for m in MODULES:
        src, dst = Path(m), out / Path(m).relative_to("src")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=ignore)
        else:
            shutil.copyfile(src, dst)
    selected = json.loads(Path(p["selected"]).read_text(encoding="utf-8"))
    shutil.copytree(selected["model_dir"], out / "model", ignore=ignore)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: variant {selected['variant']}, {size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
