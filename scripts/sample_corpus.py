"""Download a seeded sample of every Hugging Face source in a corpus config (text columns only)
and record each one in the data ledger.

Usage: uv run python scripts/sample_corpus.py --config configs/data/<name>.yaml [key=value ...]
Writes <sample.out_dir>/<source>.jsonl, manifest.json and provenance files.
"""

import hashlib
import json
import time
from pathlib import Path

from danalm.config import config_from_cli
from danalm.data import ledger
from danalm.data.hub import HubSource, sample_source
from danalm.utils.run import write_provenance


def source_seed(seed: int, name: str) -> int:
    """A per-source seed that does not change when sources are added or reordered."""
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16)


def main() -> None:
    cfg = config_from_cli(__doc__)
    sample = cfg["sample"]
    out = Path(sample["out_dir"])
    write_provenance(out, cfg)
    manifests, entries = [], []
    for raw in sample["sources"]:
        src = HubSource(**raw)
        start = time.time()
        texts, manifest = sample_source(src, seed=source_seed(cfg["seed"], src.name))
        with open(out / f"{src.name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for text in texts:
                fh.write(json.dumps({"text": text, "source": src.name}, ensure_ascii=False) + "\n")
        manifest["seconds"] = round(time.time() - start, 1)
        manifests.append(manifest)
        entries.append(
            {
                "id": f"{cfg['run_name']}/{src.name}",
                "source": f"{src.repo_id} `{src.files}`",
                "revision": src.revision,
                "licence": src.licence,
                "kind": src.kind,
                "collected_on": manifest["collected_on"],
                "used_for": sample["used_for"],
                "docs_raw": manifest["docs"],
                "utf8_bytes_raw": manifest["utf8_bytes"],
                "words_raw": manifest["words"],
                "notes": f"{len(manifest['row_groups_read'])} row groups; {src.description}",
            }
        )
        print(
            f"{src.name:18s} {manifest['docs']:>9,} docs {manifest['chars']:>13,} chars "
            f"{len(manifest['row_groups_read']):>4} row groups  {manifest['seconds']:>6}s"
        )
    (out / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    ledger.write_markdown(cfg["ledger"]["markdown"], ledger.upsert(cfg["ledger"]["jsonl"], entries))
    print(f"\nSaved to {out}; ledger updated ({cfg['ledger']['markdown']})")


if __name__ == "__main__":
    main()
