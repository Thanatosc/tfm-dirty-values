from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_path
from .full_study import (
    full_study_preflight,
    full_task_manifest,
    runtime_code_manifest,
    sha256_file,
)


def full_authorization_path(config: dict[str, Any]) -> Path:
    return resolve_path(config, config["paths"]["authorization"])


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_full_authorization(config: dict[str, Any]) -> dict[str, Any] | None:
    return _load_json(full_authorization_path(config))


def _bound_inputs(config: dict[str, Any]) -> dict[str, Any]:
    task_path = resolve_path(config, config["paths"]["task_manifest"])
    split_path = resolve_path(config, config["paths"]["data_processed"]) / "manifest.json"
    panel_path = resolve_path(config, config["full_panel"]["path"])
    phase1_path = resolve_path(config, config["paths"]["phase1_decision"])
    validation_path = resolve_path(config, config["paths"]["phase1_validation"])
    for path in (task_path, split_path, panel_path, phase1_path, validation_path):
        if not path.is_file():
            raise ValueError(f"Missing authorization input: {path}")
    manifest = full_task_manifest(config)
    phase1 = _load_json(phase1_path)
    if phase1 is None or phase1.get("decision") != "GO":
        raise ValueError("Phase 1 decision must be GO before full-study authorization")
    return {
        "config_sha256": config["_config_sha256"],
        "task_count": manifest["task_count"],
        "task_manifest_identity_sha256": manifest["task_manifest_sha256"],
        "task_manifest_file_sha256": sha256_file(task_path),
        "split_manifest_file_sha256": sha256_file(split_path),
        "full_panel_file_sha256": sha256_file(panel_path),
        "phase1_decision_file_sha256": sha256_file(phase1_path),
        "phase1_validation_file_sha256": sha256_file(validation_path),
        "runtime_code_identity_sha256": runtime_code_manifest(config)["identity_sha256"],
    }


def validate_full_authorization(
    config: dict[str, Any],
) -> tuple[bool, str, dict[str, Any] | None]:
    payload = read_full_authorization(config)
    if payload is None:
        return False, f"Missing full-study authorization: {full_authorization_path(config)}", None
    if payload.get("status") != "authorized":
        return False, "Full-study authorization status is not 'authorized'", payload
    if payload.get("experiment_id") != config["experiment_id"]:
        return False, "Authorization experiment_id does not match configuration", payload
    try:
        expected = _bound_inputs(config)
    except ValueError as error:
        return False, str(error), payload
    for key, value in expected.items():
        if payload.get(key) != value:
            return False, f"Authorization bound input does not match: {key}", payload
    return True, "authorized", payload


def require_full_authorized(config: dict[str, Any]) -> dict[str, Any]:
    valid, reason, payload = validate_full_authorization(config)
    if not valid:
        raise SystemExit(f"Full-study authorization refused: {reason}")
    return payload or {}


def write_full_authorization(config: dict[str, Any], *, command: str) -> dict[str, Any]:
    preflight = full_study_preflight(config)
    if preflight["authorization_present"]:
        raise SystemExit("A full-study authorization artifact already exists")
    if not preflight["ready_for_authorization"]:
        raise SystemExit(
            "Cannot authorize full study: " + "; ".join(preflight.get("errors", []))
        )
    try:
        bound = _bound_inputs(config)
    except ValueError as error:
        raise SystemExit(f"Cannot authorize full study: {error}") from error
    payload = {
        "schema_version": "1.0",
        "status": "authorized",
        "experiment_id": config["experiment_id"],
        "scope": "Dirty Value Phase 2 full study only; 4,575 frozen tasks",
        "authorized_at": datetime.now(timezone.utc).isoformat(),
        "authorized_command": command,
        **bound,
    }
    target = full_authorization_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target)
    return payload
