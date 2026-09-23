"""YAML configs: `base:` inheritance, `${dotted.key}` references and `key=value` CLI overrides.

Every script takes `--config file.yaml` plus optional overrides, for example:
    uv run python scripts/check_env.py --config configs/check_env.yaml seed=1 tracking.mode=offline
The fully resolved config is saved with each run (danalm.utils.run), so a run can be reproduced
from that file, the seed inside it and the recorded git commit.
"""

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import yaml

_REF = re.compile(r"\$\{([\w.]+)\}")
_MAX_REF_DEPTH = 10


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a config, merge it over its `base:` chain, apply overrides, then resolve `${...}`."""
    cfg = _load_with_bases(Path(path))
    for item in overrides or []:
        apply_override(cfg, item)
    return resolve_refs(cfg)


def config_from_cli(description: str | None = None) -> dict[str, Any]:
    """Parse the standard script CLI (`--config file.yaml [key=value ...]`) and load the config."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="overrides such as tracking.mode=offline")
    args = parser.parse_args()
    return load_config(args.config, args.overrides)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write a (resolved) config back to YAML, keeping key order and Arabic text readable."""
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    Path(path).write_text(text, encoding="utf-8")


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `base` with `update` merged in (dicts merge recursively, the rest replaces)."""
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(cfg: dict[str, Any], item: str) -> None:
    """Apply one `dotted.key=value` override in place; the value is parsed as YAML.

    The key must already exist in the config, so a typo fails loudly instead of being ignored.
    """
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"override must look like key=value, got {item!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for part in parents:
        if not isinstance(node.get(part), dict):
            raise KeyError(f"unknown config key {key!r}")
        node = node[part]
    if leaf not in node:
        raise KeyError(f"unknown config key {key!r}")
    node[leaf] = yaml.safe_load(raw)


def resolve_refs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Replace `${dotted.key}` inside string values with the referenced value.

    A value that is exactly one reference keeps the referenced type (int, list, ...);
    a reference embedded in a longer string is substituted as text.
    """

    def lookup(key: str) -> Any:
        node: Any = cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"config reference to unknown key {key!r}")
            node = node[part]
        return node

    def resolve(value: Any, depth: int) -> Any:
        if depth > _MAX_REF_DEPTH:
            raise ValueError("config references nested too deeply (circular reference?)")
        if isinstance(value, dict):
            return {k: resolve(v, depth) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v, depth) for v in value]
        if not isinstance(value, str) or "${" not in value:
            return value
        whole = _REF.fullmatch(value)
        if whole:
            return resolve(lookup(whole.group(1)), depth + 1)
        return _REF.sub(lambda m: str(resolve(lookup(m.group(1)), depth + 1)), value)

    return resolve(cfg, 0)


def _load_with_bases(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load one YAML file and merge it over its `base:` file(s) (relative paths). A list of
    bases is merged left to right, so later ones override earlier ones."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular `base:` chain at {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    bases = cfg.pop("base", None)
    if bases is None:
        return cfg
    merged: dict[str, Any] = {}
    for base in [bases] if isinstance(bases, str) else bases:
        merged = deep_merge(merged, _load_with_bases(path.parent / base, (*seen, path)))
    return deep_merge(merged, cfg)
