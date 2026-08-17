from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != "1.0":
        raise ValueError("Unsupported configuration schema")
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root())
    config["_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

