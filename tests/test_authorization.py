from copy import deepcopy
import json
from pathlib import Path

from tfm_dirty_values.authorization import (
    phase0_result_path,
    task_manifest_identity,
    validate_authorization,
    write_authorization,
)
from tfm_dirty_values.config import load_config


def test_frozen_task_manifest_identity_is_stable() -> None:
    count, manifest = task_manifest_identity(load_config("configs/kill_test.yaml"))
    assert count == 126
    assert manifest == "7226d32ba2eea35bb33cf6e3be5e64a08b53b33b237252fa5433992ff55c5034"


def test_authorization_requires_external_artifact(tmp_path: Path) -> None:
    config = deepcopy(load_config("configs/kill_test.yaml"))
    config["_project_root"] = str(tmp_path)
    valid, reason, payload = validate_authorization(config)
    assert not valid
    assert payload is None
    assert "authorization artifact" in reason.lower()


def test_external_authorization_binds_frozen_inputs(tmp_path: Path) -> None:
    config = deepcopy(load_config("configs/kill_test.yaml"))
    config["_project_root"] = str(tmp_path)
    result = phase0_result_path(config)
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"verification_status": "VERIFIED"}), encoding="utf-8")
    written = write_authorization(config, command="test-only")
    valid, reason, loaded = validate_authorization(config)
    assert valid, reason
    assert loaded == written
    assert written["config_sha256"] == config["_config_sha256"]
    assert written["stage1_task_count"] == 126


def test_phase0_tampering_invalidates_authorization(tmp_path: Path) -> None:
    config = deepcopy(load_config("configs/kill_test.yaml"))
    config["_project_root"] = str(tmp_path)
    result = phase0_result_path(config)
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"verification_status": "VERIFIED"}), encoding="utf-8")
    write_authorization(config, command="test-only")
    result.write_text(
        json.dumps({"verification_status": "VERIFIED", "tampered": True}), encoding="utf-8"
    )
    valid, reason, _ = validate_authorization(config)
    assert not valid
    assert "phase 0 result hash" in reason.lower()
