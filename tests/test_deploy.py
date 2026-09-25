"""The deployment files stay consistent (Phase 7, D-034): pinned versions and the Docker image."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def locked_versions() -> dict[str, str]:
    text = (ROOT / "uv.lock").read_text(encoding="utf-8")
    return dict(re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', text))


def test_serving_requirements_pin_the_locked_versions():
    locked = locked_versions()
    for line in (ROOT / "deploy" / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            assert (
                locked[name] == version
            ), f"{name}: requirements {version}, uv.lock {locked[name]}"


def test_the_docker_image_files_are_enough_to_import_the_service(tmp_path):
    """Copy exactly what the Dockerfile copies and import the service with torch blocked."""
    copies = [line.split()[1:] for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
              if line.startswith("COPY src/")]  # fmt: skip
    for parts in copies:
        *sources, dest = parts
        target = tmp_path / dest.removeprefix("/app/")
        for src in sources:
            s = ROOT / src
            if s.is_dir():
                shutil.copytree(s, target, dirs_exist_ok=True)
            else:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy(s, target / s.name)
        for src in sources:  # .dockerignore must let every copied path through
            assert f"!{src}" in (ROOT / ".dockerignore").read_text(encoding="utf-8")
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'datasketch'):\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import danalm.serve.app, danalm\n"
        f"assert danalm.__file__.startswith({str(tmp_path)!r}), danalm.__file__\n"
    )
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}  # the copied package comes first
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path, env=env)
