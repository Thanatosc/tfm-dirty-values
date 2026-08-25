"""Phase 3 authorization gate.

Mirrors the Phase 2 pattern: the frozen config is never rewritten to grant
permission. Instead a separate artifact records the authorization and binds it by
SHA-256 to every input that must not drift afterwards, including the Phase 2
evidence Phase 3 depends on and the paired-identity probe verdict.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tfm_dirty_values.config import canonical_json, resolve_path
from tfm_dirty_values.full_study import sha256_file

from .design import ablation_task_manifest, verify_expected_counts

REQUIRED_PROBE_VERDICT = "VERIFIED"


def authorization_path(config: dict[str, Any]) -> Path:
    return resolve_path(config, config["paths"]["authorization"])


def probe_report_path(config: dict[str, Any]) -> Path:
    return resolve_path(config, "artifacts/phase3/paired-identity-probe.json")


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


def bound_inputs(config: dict[str, Any], phase2_config: dict[str, Any]) -> dict[str, Any]:
    """Collect every hash the authorization pins. Raises if an input is missing."""
    manifest_path = resolve_path(config, config["paths"]["task_manifest"])
    prereg_path = resolve_path(config, config["paths"]["preregistration"])
    phase2_auth = resolve_path(config, config["paths"]["phase2_authorization"])
    phase2_analysis = resolve_path(config, config["paths"]["phase2_analysis"])
    phase2_db = resolve_path(config, config["paths"]["phase2_database"])
    probe_path = probe_report_path(config)

    for path in (manifest_path, prereg_path, phase2_auth, phase2_analysis, phase2_db, probe_path):
        if not path.is_file():
            raise ValueError(f"Missing Phase 3 authorization input: {path}")

    probe = _load_json(probe_path)
    if probe is None:
        raise ValueError(f"Unreadable paired-identity probe report: {probe_path}")
    if probe.get("verdict") != REQUIRED_PROBE_VERDICT:
        raise ValueError(
            "Paired-identity probe verdict must be "
            f"{REQUIRED_PROBE_VERDICT}, found {probe.get('verdict')!r}"
        )
    if probe.get("matched_cell_count") != probe.get("probe_cell_count"):
        raise ValueError("Paired-identity probe did not match every probe cell")

    manifest = ablation_task_manifest(config, phase2_config)
    count_errors = verify_expected_counts(manifest, config)
    if count_errors:
        raise ValueError("; ".join(count_errors))

    expected_phase2_runtime = str(config["inherits"]["phase2_runtime_code_identity"])
    from tfm_dirty_values.full_study import runtime_code_manifest

    actual_phase2_runtime = runtime_code_manifest(phase2_config)["identity_sha256"]
    if actual_phase2_runtime != expected_phase2_runtime:
        raise ValueError(
            "Phase 2 runtime code identity drifted: "
            f"{actual_phase2_runtime} != {expected_phase2_runtime}"
        )

    return {
        "config_sha256": config["_config_sha256"],
        "phase2_config_sha256": phase2_config["_config_sha256"],
        "task_count": manifest["task_count"],
        "task_manifest_identity_sha256": manifest["task_manifest_sha256"],
        "task_manifest_file_sha256": sha256_file(manifest_path),
        "preregistration_file_sha256": sha256_file(prereg_path),
        "paired_identity_report_sha256": sha256_file(probe_path),
        "paired_identity_verdict": probe["verdict"],
        "phase2_authorization_file_sha256": sha256_file(phase2_auth),
        "phase2_analysis_file_sha256": sha256_file(phase2_analysis),
        "phase2_runtime_code_identity_sha256": actual_phase2_runtime,
        "cleaner_identity": canonical_json(config["cleaner"]),
    }


def validate_authorization(
    config: dict[str, Any],
    phase2_config: dict[str, Any],
) -> tuple[bool, str, dict[str, Any] | None]:
    payload = read_authorization(config)
    if payload is None:
        return False, f"Missing Phase 3 authorization: {authorization_path(config)}", None
    if payload.get("status") != "authorized":
        return False, "Phase 3 authorization status is not 'authorized'", payload
    if payload.get("experiment_id") != config["experiment_id"]:
        return False, "Authorization experiment_id does not match configuration", payload
    try:
        expected = bound_inputs(config, phase2_config)
    except ValueError as error:
        return False, str(error), payload
    for key, value in expected.items():
        if payload.get(key) != value:
            return False, f"Authorization bound input does not match: {key}", payload
    return True, "authorized", payload


def require_authorized(config: dict[str, Any], phase2_config: dict[str, Any]) -> dict[str, Any]:
    valid, reason, payload = validate_authorization(config, phase2_config)
    if not valid:
        raise SystemExit(f"Phase 3 authorization refused: {reason}")
    return payload or {}


def write_authorization(
    config: dict[str, Any],
    phase2_config: dict[str, Any],
    *,
    command: str,
) -> dict[str, Any]:
    target = authorization_path(config)
    if target.is_file():
        raise SystemExit("A Phase 3 authorization artifact already exists")

    database = resolve_path(config, config["paths"]["database"])
    if database.is_file():
        raise SystemExit("A Phase 3 result database exists without an authorization artifact")

    try:
        bound = bound_inputs(config, phase2_config)
    except ValueError as error:
        raise SystemExit(f"Cannot authorize Phase 3: {error}") from error

    payload = {
        "schema_version": "1.0",
        "status": "authorized",
        "experiment_id": config["experiment_id"],
        "scope": (
            "Dirty Value Phase 3 cleaning ablation only; 2,425 frozen tasks in arms C and D. "
            "Does not authorize any change to Phase 2 results, new corruption mechanisms, "
            "cleaner tuning, or additional affected-cell rates."
        ),
        "charter_basis": config["charter_basis"],
        "authorized_at": datetime.now(timezone.utc).isoformat(),
        "authorized_command": command,
        **bound,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target)
    return payload
