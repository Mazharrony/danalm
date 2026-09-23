"""Tests for the YAML config loader."""

from pathlib import Path

import pytest
import yaml

from danalm.config import apply_override, deep_merge, load_config, resolve_refs, save_config

REPO = Path(__file__).resolve().parents[1]


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_base_chain_merges_nested_keys(tmp_path):
    write_yaml(
        tmp_path / "base.yaml", {"seed": 1, "paths": {"data_dir": "data", "runs_dir": "runs"}}
    )
    (tmp_path / "sub").mkdir()
    child = write_yaml(
        tmp_path / "sub" / "child.yaml", {"base": "../base.yaml", "paths": {"runs_dir": "out"}}
    )
    assert load_config(child) == {"seed": 1, "paths": {"data_dir": "data", "runs_dir": "out"}}


def test_list_of_bases_merges_left_to_right(tmp_path):
    write_yaml(tmp_path / "a.yaml", {"x": 1, "t": {"model": "9b", "port": 1}})
    write_yaml(tmp_path / "b.yaml", {"y": 2, "t": {"model": "35b"}})
    child = write_yaml(tmp_path / "c.yaml", {"base": ["a.yaml", "b.yaml"], "y": 3})
    assert load_config(child) == {"x": 1, "y": 3, "t": {"model": "35b", "port": 1}}


def test_circular_base_is_rejected(tmp_path):
    write_yaml(tmp_path / "a.yaml", {"base": "b.yaml"})
    write_yaml(tmp_path / "b.yaml", {"base": "a.yaml"})
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("3", 3), ("0.5", 0.5), ("true", True), ("null", None), ("[1, 2]", [1, 2]), ("abc", "abc")],
)
def test_override_values_are_parsed_as_yaml(raw, expected):
    cfg = {"a": {"b": 0}}
    apply_override(cfg, f"a.b={raw}")
    assert cfg["a"]["b"] == expected


@pytest.mark.parametrize("item", ["a.typo=1", "missing=1", "a.b.c=1"])
def test_override_of_unknown_key_fails(item):
    with pytest.raises(KeyError):
        apply_override({"a": {"b": 0}}, item)


def test_override_needs_equals_sign():
    with pytest.raises(ValueError):
        apply_override({"a": 1}, "a")


def test_refs_keep_type_or_substitute_into_strings():
    cfg = {"paths": {"data_dir": "data"}, "n": 4, "out": "${paths.data_dir}/smoke", "copy": "${n}"}
    resolved = resolve_refs(cfg)
    assert resolved["out"] == "data/smoke"
    assert resolved["copy"] == 4


def test_refs_are_resolved_after_overrides(tmp_path):
    path = write_yaml(
        tmp_path / "c.yaml", {"paths": {"data_dir": "data"}, "out": "${paths.data_dir}/x"}
    )
    assert load_config(path, ["paths.data_dir=E:/d"])["out"] == "E:/d/x"


def test_unknown_or_circular_refs_fail():
    with pytest.raises(KeyError):
        resolve_refs({"a": "${nope}"})
    with pytest.raises(ValueError):
        resolve_refs({"a": "${b}", "b": "${a}"})


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_save_and_reload_round_trip_keeps_arabic(tmp_path):
    cfg = {"seed": 1, "name": "دانة", "items": [1, 2]}
    save_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg
    assert "دانة" in (tmp_path / "c.yaml").read_text(encoding="utf-8")


RUN_CONFIGS = [  # files with a base: (or base.yaml itself); others, like intents.yaml, are data
    p
    for p in sorted((REPO / "configs").rglob("*.yaml"))
    if p.name == "base.yaml" or "base:" in p.read_text(encoding="utf-8")
]


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: str(p.relative_to(REPO / "configs")))
def test_repo_configs_load(path):
    cfg = load_config(path)
    assert {"seed", "deterministic", "paths", "tracking"} <= cfg.keys()
