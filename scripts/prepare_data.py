"""Clean, PII-mask, deduplicate, language-tag and split raw text; optionally tokenize to .bin.

Usage: uv run python scripts/prepare_data.py --config configs/data/<name>.yaml [key=value ...]
Writes train/val.jsonl, stats.json and provenance (config.yaml, meta.json) to `data.out_dir`.
"""

import json
from pathlib import Path

from danalm.config import config_from_cli
from danalm.data.pipeline import PipelineConfig, run_pipeline
from danalm.utils.run import write_provenance


def main() -> None:
    cfg = config_from_cli(__doc__)
    data_cfg = PipelineConfig(**cfg["data"])
    write_provenance(Path(data_cfg.out_dir), cfg)
    stats = run_pipeline(data_cfg, seed=cfg["seed"])
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
