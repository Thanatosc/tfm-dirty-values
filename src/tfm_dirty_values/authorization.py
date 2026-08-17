from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import canonical_json
from .design import kill_test_tasks


AUTHORIZATION_RELATIVE_PATH = Path("research") / "kill-test-authorization.json"
PHASE0_RESULT_RELATIVE_PATH = Path("artifacts") / "phase0" / "migration-reproduction.json"


def authorization_path(config: dict[str, Any]) -> Path:
    return Path(config["_project_root"]) / AUTHORIZATION_RELATIVE_PATH


def phase0_result_path(config: dict[str, Any]) -> Path:
    return Path(config["_project_root"]) / PHASE0_RESULT_RELATIVE_PATH


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def task_manifest_identity(config: dict[str, Any]) -> tuple[int, str]:
    tasks = [task.to_dict() for task in kill_test_tasks(config)]
    return len(tasks), hashlib.sha256(canonical_json(tasks).encode("ascii")).hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_authorization(config: dict[str, Any]) -> dict[str, Any] | None:
    return _load_json(authorization_path(config))


def validate_authorization(config: dict[str, Any]) -> tuple[bool, str, dict[str, Any] | None]:
    payload = read_authorization(config)
    if payload is None:
        return False, f"Missing authorization artifact: {authorization_path(config)}", None
    if payload.get("status") != "authorized":
        return False, "Authorization artifact status is not 'authorized'", payload
    if payload.get("experiment_id") != config["experiment_id"]:
        return False, "Authorization experiment_id does not match configuration", payload
    if payload.get("config_sha256") != config["_config_sha256"]:
        return False, "Authorization config hash does not match frozen configuration", payload
    expected_count, expected_manifest = task_manifest_identity(config)
    if payload.get("stage1_task_count") != expected_count:
        return False, "Authorization task count does not match frozen task manifest", payload
    if payload.get("stage1_task_manifest_sha256") != expected_manifest:
        return False, "Authorization task manifest hash does not match frozen tasks", payload
    phase0_path = phase0_result_path(config)
    if not phase0_path.is_file():
        return False, f"Missing verified Phase 0 result: {phase0_path}", payload
    try:
        phase0 = json.loads(phase0_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "Phase 0 result is not valid JSON", payload
    if phase0.get("verification_status") != "VERIFIED":
        return False, "Phase 0 migration result is not VERIFIED", payload
    actual_phase0_hash = _sha256_file(phase0_path)
    if payload.get("phase0_result_sha256") != actual_phase0_hash:
        return False, "Authorization Phase 0 result hash does not match", payload
    return True, "authorized", payload


def require_authorized(config: dict[str, Any]) -> dict[str, Any]:
    valid, reason, payload = validate_authorization(config)
    if not valid:
        raise SystemExit(f"Kill Test authorization refused: {reason}")
    return payload or {}


def write_authorization(config: dict[str, Any], *, command: str) -> dict[str, Any]:
    phase0_path = phase0_result_path(config)
    if not phase0_path.is_file():
        raise SystemExit(f"Cannot authorize Kill Test without verified Phase 0 result: {phase0_path}")
    try:
        phase0 = json.loads(phase0_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read Phase 0 result: {error}") from error
    if phase0.get("verification_status") != "VERIFIED":
        raise SystemExit("Cannot authorize Kill Test: Phase 0 migration result is not VERIFIED")
    task_count, manifest_sha256 = task_manifest_identity(config)
    payload = {
        "schema_version": "1.0",
        "status": "authorized",
        "experiment_id": config["experiment_id"],
        "scope": "Dirty Value Kill Test Stage 1 only",
        "authorized_at": datetime.now(timezone.utc).isoformat(),
        "authorized_command": command,
        "config_sha256": config["_config_sha256"],
        "stage1_task_count": task_count,
        "stage1_task_manifest_sha256": manifest_sha256,
        "phase0_result_sha256": _sha256_file(phase0_path),
    }
    target = authorization_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)
    return payload
