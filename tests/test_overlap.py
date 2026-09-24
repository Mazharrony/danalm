"""Tests for the test-set overlap check (scripts/check_overlap.py)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "check_overlap", REPO / "scripts" / "check_overlap.py"
)
check_overlap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_overlap)


def write_jsonl(path: Path, field: str, texts: list[str]) -> None:
    path.write_text("".join(json.dumps({field: t}) + "\n" for t in texts), encoding="utf-8")


def run(tmp_path: Path, test_texts: list[str], monkeypatch) -> int:
    write_jsonl(tmp_path / "train.jsonl", "message", ["my card is blocked since morning pls help"])
    write_jsonl(tmp_path / "docs.jsonl", "text", ["A long pretraining document about cards."])
    write_jsonl(tmp_path / "test.jsonl", "text", test_texts)
    cfg = {
        "base": str(REPO / "configs" / "base.yaml"),
        "run_name": "overlap-test",
        "overlap": {
            "test_files": [str(tmp_path / "test.jsonl")],
            "test_field": "text",
            "normalize": {"strip_diacritics": True, "unify_alef": False},
            "char_ngram": 3,
            "num_perm": 128,
            "near_dup_threshold": 0.6,
            "compare_up_to": 4,
            "train_sets": [
                {"files": [str(tmp_path / "train.jsonl")], "field": "message", "near_dup": True},
                {"files": [str(tmp_path / "docs.jsonl")], "field": "text", "near_dup": False},
            ],
        },
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_overlap.py", "--config", str(tmp_path / "cfg.yaml")])
    with pytest.raises(SystemExit) as exit_info:
        check_overlap.main()
    return exit_info.value.code


def test_clean_test_set_passes(tmp_path, monkeypatch):
    assert run(tmp_path, ["where is my parcel, it is 3 days late"], monkeypatch) == 0


def test_exact_and_near_copies_fail(tmp_path, monkeypatch):
    texts = [
        "My card is blocked since morning pls help",
        "a long pretraining document about cards.",
    ]
    assert run(tmp_path, texts, monkeypatch) == 1
    near = ["my card is blocked since morning plz help!!"]
    assert run(tmp_path, near, monkeypatch) == 1
    report = json.loads((tmp_path / "overlap_report.json").read_text(encoding="utf-8"))
    assert report["overlaps"] == 1 and report["hits"][0]["kind"] == "near"


def test_arabic_spelling_variants_count_as_exact_copies(tmp_path, monkeypatch):
    write_jsonl(tmp_path / "sft.jsonl", "message", ["البطاقة مب راضية تشتغل من الصبح"])
    assert check_overlap.canonical(
        "البطاقه مب راضيه تشتغل من الصبح!", {"strip_diacritics": True, "unify_alef": False}
    ) == check_overlap.canonical(
        "البطاقة مب راضية تشتغل من الصبح", {"strip_diacritics": True, "unify_alef": False}
    )
