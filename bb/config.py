from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def parse_override(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_dotlist_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got: {item}")
        key_path, raw_value = item.split("=", 1)
        cursor = out
        parts = key_path.split(".")
        for key in parts[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot set nested override through non-mapping: {key_path}")
        cursor[parts[-1]] = parse_override(raw_value)
    return out


def require(config: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for key in dotted_key.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            raise KeyError(f"Missing config key: {dotted_key}")
        cursor = cursor[key]
    return cursor
