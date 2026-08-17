from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from tfm_dirty_values.config import load_config, resolve_path
from tfm_dirty_values.design import full_study_tasks
from tfm_dirty_values.full_authorization import (
    validate_full_authorization,
    write_full_authorization,
)
from tfm_dirty_values.full_study import (
    full_task_manifest,
    runtime_code_manifest,
    sha256_file,
)


def test_full_study_task_counts_and_unique_ids() -> None:
    config = load_config("configs/full_study.yaml")
    tasks = full_study_tasks(config)
    clean = [task for task in tasks if task.corruption == "clean"]
    numeric = [
        task
        for task in tasks
        if task.corruption in {"measurement_noise", "outlier", "quantization", "unit_error"}
    ]
    categorical = [task for task in tasks if task.corruption == "categorical_error"]

    assert len(tasks) == 4575
    assert len(clean) == 275
    assert len(numeric) == 3600
    assert len(categorical) == 700
    assert {task.stage for task in tasks} == {3}
    assert len({task.task_id for task in tasks}) == len(tasks)


def test_full_task_manifest_identity_is_stable() -> None:
    payload = full_task_manifest(load_config("configs/full_study.yaml"))
    assert payload["task_count"] == 4575
    assert payload["task_manifest_sha256"] == (
        "b2bb293fedbb497534311b7bb353e10de2cc4d6a4525e30a292717fcf581cc8c"
    )


def test_full_panel_citations_are_complete_and_frozen() -> None:
    config = load_config("configs/full_study.yaml")
    path = resolve_path(config, config["full_panel"]["path"])
    panel = json.loads(path.read_text(encoding="utf-8"))
    records = panel["datasets"]

    assert len(records) == 11
    assert sha256_file(path) == config["full_panel"]["sha256"]
    assert all(str(record.get("citation") or "").strip() for record in records)
    assert all(
        not str(record.get("citation_status") or "").startswith("missing")
        for record in records
    )


def test_runtime_code_manifest_covers_resume_and_monitoring() -> None:
    config = load_config("configs/full_study.yaml")
    manifest = runtime_code_manifest(config)
    paths = {record["path"] for record in manifest["files"]}

    assert len(manifest["identity_sha256"]) == 64
    assert manifest["file_count"] == len(paths)
    assert {
        "src/tfm_dirty_values/state.py",
        "src/tfm_dirty_values/worker.py",
        "src/tfm_dirty_values/cli.py",
        "scripts/full_study_status.ps1",
        "scripts/resume_full_study.ps1",
        "scripts/stop_full_study.ps1",
    } <= paths


def _write_authorization_inputs(config: dict) -> None:
    task_path = resolve_path(config, config["paths"]["task_manifest"])
    split_path = resolve_path(config, config["paths"]["data_processed"]) / "manifest.json"
    panel_path = resolve_path(config, config["full_panel"]["path"])
    phase1_path = resolve_path(config, config["paths"]["phase1_decision"])
    validation_path = resolve_path(config, config["paths"]["phase1_validation"])
    for path in (task_path, split_path, panel_path, phase1_path, validation_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text('{"frozen":true}\n', encoding="utf-8")
    split_path.write_text('{"splits":[]}\n', encoding="utf-8")
    panel_path.write_text('{"datasets":[]}\n', encoding="utf-8")
    phase1_path.write_text('{"decision":"GO"}\n', encoding="utf-8")
    validation_path.write_text("# Phase 1 validation\n", encoding="utf-8")


def test_full_authorization_requires_external_artifact(tmp_path: Path) -> None:
    config = deepcopy(load_config("configs/full_study.yaml"))
    config["_project_root"] = str(tmp_path)

    valid, reason, payload = validate_full_authorization(config)

    assert not valid
    assert payload is None
    assert "missing full-study authorization" in reason.lower()


def test_full_authorization_detects_bound_input_tampering(
    tmp_path: Path, monkeypatch,
) -> None:
    config = deepcopy(load_config("configs/full_study.yaml"))
    config["_project_root"] = str(tmp_path)
    _write_authorization_inputs(config)
    monkeypatch.setattr(
        "tfm_dirty_values.full_authorization.full_study_preflight",
        lambda ignored: {
            "authorization_present": False,
            "ready_for_authorization": True,
            "errors": [],
        },
    )

    written = write_full_authorization(config, command="test-only")
    valid, reason, loaded = validate_full_authorization(config)
    assert valid, reason
    assert loaded == written
    assert written["task_count"] == 4575
    assert len(written["runtime_code_identity_sha256"]) == 64

    split_path = resolve_path(config, config["paths"]["data_processed"]) / "manifest.json"
    split_path.write_text('{"splits":[],"tampered":true}\n', encoding="utf-8")
    valid, reason, _ = validate_full_authorization(config)
    assert not valid
    assert "split_manifest_file_sha256" in reason
