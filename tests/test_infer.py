"""Inference without PyTorch (Phase 7, D-034): the serving path must not import torch."""

import subprocess
import sys

BLOCK_HEAVY = """
import sys


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "datasketch"):
            raise ImportError(f"blocked: {name}")


sys.meta_path.insert(0, Block())
"""


def test_serving_modules_import_without_torch_or_datasketch():
    code = BLOCK_HEAVY + "import danalm.text, danalm.sft.format\n"
    subprocess.run([sys.executable, "-c", code], check=True)
