"""The intent taxonomy (configs/sft/intents.yaml, D-021)."""

import yaml


def load_intents(path: str) -> list[dict[str, str]]:
    """[{name, domain, description}, ...] in taxonomy order."""
    with open(path, encoding="utf-8") as fh:
        intents = yaml.safe_load(fh)["intents"]
    names = [i["name"] for i in intents]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate intent names in {path}")
    return intents
