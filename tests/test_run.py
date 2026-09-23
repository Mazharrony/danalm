"""Tests for run bookkeeping (provenance files + tracking)."""

import json
import subprocess

import pytest
import yaml

from danalm.utils.run import git_info, start_run


def make_cfg(tmp_path) -> dict:
    return {
        "seed": 3,
        "deterministic": False,
        "run_name": "unit-test",
        "paths": {"data_dir": str(tmp_path / "data"), "runs_dir": str(tmp_path / "runs")},
        "tracking": {"project": "danalm-tests", "entity": None, "mode": "disabled", "tags": []},
    }


def git(repo, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_git_info_outside_a_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_info(tmp_path) == {"commit": None, "dirty": None, "untracked": None}


def test_git_info_reports_commit_and_dirty_state(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "init")
    clean = git_info(tmp_path)
    assert len(clean["commit"]) == 40
    assert clean["dirty"] is False
    (tmp_path / "new.txt").write_text("x", encoding="utf-8")
    dirty = git_info(tmp_path)
    assert dirty["dirty"] is True
    assert dirty["untracked"] == ["new.txt"]


def test_start_run_writes_config_and_meta(tmp_path):
    cfg = make_cfg(tmp_path)
    run = start_run(cfg, job_type="test")
    try:
        assert run.dir.parent == tmp_path / "runs"
        assert run.name.startswith("unit-test-")
        saved = yaml.safe_load((run.dir / "config.yaml").read_text(encoding="utf-8"))
        assert saved == cfg
        meta = json.loads((run.dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["seed"] == 3
        assert meta["job_type"] == "test"
        assert {"commit", "dirty", "untracked"} <= meta["git"].keys()
        assert meta["env"]["torch"]
        run.wandb.log({"loss": 1.0})  # a no-op when tracking is disabled; must not raise
    finally:
        run.wandb.finish()


def test_start_run_requires_run_name(tmp_path):
    cfg = make_cfg(tmp_path) | {"run_name": None}
    with pytest.raises(ValueError, match="run_name"):
        start_run(cfg, job_type="test")
