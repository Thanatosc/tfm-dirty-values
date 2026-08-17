from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import canonical_json, resolve_path
from .design import full_study_tasks
from .models import verify_model_artifacts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def runtime_code_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Hash the execution surface that must remain fixed after authorization."""
    root = Path(__file__).resolve().parents[2]
    package_root = root / "src" / "tfm_dirty_values"
    script_paths = [
        root / "pyproject.toml",
        root / "scripts" / "runtime.ps1",
        root / "scripts" / "workspace_env.ps1",
        root / "scripts" / "start_full_study.ps1",
        root / "scripts" / "resume_full_study.ps1",
        root / "scripts" / "stop_full_study.ps1",
        root / "scripts" / "full_study_status.ps1",
    ]
    paths = sorted(
        [*package_root.rglob("*.py"), *script_paths],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Missing runtime code artifact: {path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    identity = hashlib.sha256(canonical_json(records).encode("ascii")).hexdigest()
    return {
        "identity_sha256": identity,
        "file_count": len(records),
        "files": records,
    }


def full_task_manifest(config: dict[str, Any]) -> dict[str, Any]:
    tasks = [task.to_dict() for task in full_study_tasks(config)]
    return {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "protocol_version": config["protocol_version"],
        "stage": 3,
        "config_sha256": config["_config_sha256"],
        "task_count": len(tasks),
        "task_manifest_sha256": hashlib.sha256(
            canonical_json(tasks).encode("ascii")
        ).hexdigest(),
        "tasks": tasks,
    }


def freeze_full_task_manifest(config: dict[str, Any]) -> dict[str, Any]:
    payload = full_task_manifest(config)
    expected = int(config["expected_counts"]["total_tasks"])
    if payload["task_count"] != expected:
        raise ValueError(
            f"Full-study task count {payload['task_count']} does not match preregistered {expected}"
        )
    _atomic_json(resolve_path(config, config["paths"]["task_manifest"]), payload)
    return payload


def _validate_panel(config: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    path = resolve_path(config, config["full_panel"]["path"])
    if not path.is_file():
        errors.append(f"Missing full-panel metadata: {path}")
        return {"path": str(path), "present": False}
    actual_hash = sha256_file(path)
    expected_hash = str(config["full_panel"]["sha256"]).upper()
    if actual_hash != expected_hash:
        errors.append(f"Full-panel hash mismatch: {actual_hash} != {expected_hash}")
    panel = _read_json(path)
    records = panel.get("datasets", [])
    if len(records) != int(config["full_panel"]["expected_records"]):
        errors.append("Full-panel record count does not match the preregistration")
    config_ids = {int(item["data_id"]) for item in config["datasets"]}
    panel_ids = {int(item["data_id"]) for item in records}
    if config_ids != panel_ids:
        errors.append("Full-panel dataset IDs do not match the configuration")
    c5_ids = {int(value) for value in panel.get("c5_data_ids", [])}
    configured_c5 = {
        int(item["data_id"]) for item in config["datasets"] if bool(item.get("c5"))
    }
    if c5_ids != configured_c5:
        errors.append("C5 dataset IDs do not match the configuration")
    if len(c5_ids) != int(config["full_panel"]["expected_c5_records"]):
        errors.append("C5 record count does not match the preregistration")
    missing_citations = [
        int(item["data_id"])
        for item in records
        if not str(item.get("citation") or "").strip()
        or str(item.get("citation_status") or "").startswith("missing")
    ]
    if missing_citations:
        errors.append(f"Dataset citations remain missing for OpenML IDs {missing_citations}")
    return {
        "path": str(path),
        "present": True,
        "sha256": actual_hash,
        "records": len(records),
        "c5_records": len(c5_ids),
        "missing_citations": missing_citations,
    }


def _validate_task_manifest(config: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    path = resolve_path(config, config["paths"]["task_manifest"])
    expected = full_task_manifest(config)
    if not path.is_file():
        errors.append(f"Missing frozen task manifest: {path}")
        return {
            "path": str(path),
            "present": False,
            "expected_task_count": expected["task_count"],
            "expected_identity_sha256": expected["task_manifest_sha256"],
        }
    actual = _read_json(path)
    for key in ("experiment_id", "config_sha256", "task_count", "task_manifest_sha256"):
        if actual.get(key) != expected.get(key):
            errors.append(f"Frozen task manifest field does not match: {key}")
    if actual.get("tasks") != expected["tasks"]:
        errors.append("Frozen task manifest task content does not match the generator")
    return {
        "path": str(path),
        "present": True,
        "sha256": sha256_file(path),
        "task_count": actual.get("task_count"),
        "identity_sha256": actual.get("task_manifest_sha256"),
    }


def _validate_split_manifest(config: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    processed_dir = resolve_path(config, config["paths"]["data_processed"])
    path = processed_dir / "manifest.json"
    expected_keys = {
        (int(dataset["data_id"]), int(seed))
        for dataset in config["datasets"]
        for seed in config["seeds"]
    }
    if not path.is_file():
        errors.append(f"Missing full-study split manifest: {path}")
        return {"path": str(path), "present": False, "expected_splits": len(expected_keys)}
    manifest = _read_json(path)
    if manifest.get("config_sha256") != config["_config_sha256"]:
        errors.append("Split manifest uses a different configuration hash")
    splits = manifest.get("splits", [])
    actual_keys = {(int(row["data_id"]), int(row["seed"])) for row in splits}
    if actual_keys != expected_keys:
        errors.append("Split manifest does not contain exactly the preregistered dataset-seed panel")
    if len(splits) != len(expected_keys):
        errors.append("Split manifest contains duplicate or extra dataset-seed records")
    if int(manifest.get("split_count", -1)) != len(expected_keys):
        errors.append("Split manifest count does not match the preregistration")
    file_records = manifest.get("split_files", [])
    if len(file_records) != len(expected_keys):
        errors.append("Split file-hash records are incomplete")
    file_keys = {
        (int(row["data_id"]), int(row["seed"]))
        for row in file_records
        if "data_id" in row and "seed" in row
    }
    if file_keys != expected_keys or len(file_records) != len(file_keys):
        errors.append("Split file-hash records do not match the dataset-seed panel")
    file_errors: list[str] = []
    root = Path(config["_project_root"])
    for record in file_records:
        for path_key, hash_key in (
            ("array_path", "array_sha256"),
            ("metadata_path", "metadata_sha256"),
        ):
            candidate = root / str(record.get(path_key, ""))
            if not candidate.is_file():
                file_errors.append(f"missing:{candidate}")
                continue
            if sha256_file(candidate) != str(record.get(hash_key, "")).upper():
                file_errors.append(f"hash:{candidate}")
    if file_errors:
        errors.append(f"Split file verification failed for {len(file_errors)} files")
    return {
        "path": str(path),
        "present": True,
        "sha256": sha256_file(path),
        "split_count": len(splits),
        "content_manifest_sha256": manifest.get("manifest_sha256"),
        "file_manifest_sha256": manifest.get("split_file_manifest_sha256"),
        "file_errors": file_errors,
    }


def full_study_preflight(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    panel = _validate_panel(config, errors)
    tasks = _validate_task_manifest(config, errors)
    splits = _validate_split_manifest(config, errors)
    try:
        runtime_code = runtime_code_manifest(config)
    except ValueError as error:
        errors.append(str(error))
        runtime_code = {"error": str(error)}

    phase1_path = resolve_path(config, config["paths"]["phase1_decision"])
    phase1_decision: dict[str, Any] | None = None
    if not phase1_path.is_file():
        errors.append(f"Missing Phase 1 decision: {phase1_path}")
    else:
        phase1_decision = _read_json(phase1_path)
        if phase1_decision.get("decision") != "GO":
            errors.append("Phase 1 decision is not GO")

    validation_path = resolve_path(config, config["paths"]["phase1_validation"])
    if not validation_path.is_file():
        errors.append(f"Missing Phase 1 validation report: {validation_path}")

    try:
        model_artifacts = verify_model_artifacts(config)
    except (FileNotFoundError, ValueError) as error:
        errors.append(str(error))
        model_artifacts = {"error": str(error)}

    authorization_path = resolve_path(config, config["paths"]["authorization"])
    database_path = resolve_path(config, config["paths"]["database"])
    authorization_present = authorization_path.is_file()
    database_present = database_path.is_file()
    if database_present and not authorization_present:
        errors.append("Full-study database exists without an authorization artifact")

    ready = not errors
    gate = (
        "PREFLIGHT_FAILED"
        if errors
        else "AUTHORIZATION_PRESENT"
        if authorization_present
        else "READY_FOR_HUMAN_AUTHORIZATION"
    )
    return {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "protocol_version": config["protocol_version"],
        "config_sha256": config["_config_sha256"],
        "gate": gate,
        "ready_for_authorization": ready and not authorization_present,
        "authorization_present": authorization_present,
        "authorization_path": str(authorization_path),
        "result_database_present": database_present,
        "outcomes_accessed": database_present,
        "errors": errors,
        "full_panel": panel,
        "task_manifest": tasks,
        "split_manifest": splits,
        "runtime_code": runtime_code,
        "phase1_decision": {
            "path": str(phase1_path),
            "sha256": sha256_file(phase1_path) if phase1_path.is_file() else None,
            "decision": phase1_decision.get("decision") if phase1_decision else None,
        },
        "phase1_validation": {
            "path": str(validation_path),
            "sha256": sha256_file(validation_path) if validation_path.is_file() else None,
        },
        "model_artifacts": model_artifacts,
    }


def write_full_study_preflight(config: dict[str, Any]) -> dict[str, Any]:
    report = full_study_preflight(config)
    _atomic_json(resolve_path(config, config["paths"]["preflight_report"]), report)
    return report
