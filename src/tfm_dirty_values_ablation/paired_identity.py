"""Paired-identity precondition for the Phase 3 ablation.

The ablation compares executed arm C against frozen Phase 2 arm B. That
comparison is only valid if Phase 3 reproduces Phase 2 bit-for-bit when the
cleaner is switched off. This module proves it before any Phase 3 outcome is
produced.

For each preregistered probe cell it runs the Phase 3 runner with
``cleaned=False`` and requires exact equality against the stored Phase 2 record
on four metrics and on the corrupted-array hashes fed to the model. Tolerance is
zero. Any mismatch blocks execution.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from tfm_dirty_values.config import resolve_path

from .design import normalize_parameters
from .runner import AblationExperimentRunner

METRIC_KEYS = ("roc_auc", "balanced_accuracy", "log_loss", "ece_10_equal_width")


def _phase2_record(database: Path, cell: dict[str, Any]) -> dict[str, Any] | None:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        row = connection.execute(
            """
            SELECT task_id, status, metrics_json
              FROM runs
             WHERE dataset_id = ? AND seed = ? AND corruption = ?
               AND rate = ? AND model = ?
            """,
            (
                int(cell["dataset_id"]),
                int(cell["seed"]),
                str(cell["corruption"]),
                float(cell["rate"]),
                str(cell["model"]),
            ),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return {
        "task_id": row[0],
        "status": row[1],
        "metrics": json.loads(row[2]) if row[2] else None,
    }


def run_paired_identity_probe(
    config: dict[str, Any],
    phase2_config: dict[str, Any],
) -> dict[str, Any]:
    """Execute the probe and return a machine-readable verdict."""
    database = resolve_path(config, config["paths"]["phase2_database"])
    if not database.is_file():
        return {
            "verdict": "BLOCKED",
            "errors": [f"Missing frozen Phase 2 database: {database}"],
            "cells": [],
        }

    parameters = phase2_config["corruptions"]["parameters"]
    datasets = {int(item["data_id"]): str(item["name"]) for item in phase2_config["datasets"]}
    runner = AblationExperimentRunner(phase2_config)

    errors: list[str] = []
    cells: list[dict[str, Any]] = []

    for cell in config["paired_identity_probe"]["probe_cells"]:
        data_id = int(cell["dataset_id"])
        mechanism = str(cell["corruption"])
        record = _phase2_record(database, cell)
        entry: dict[str, Any] = {
            "cell": {
                "dataset_id": data_id,
                "seed": int(cell["seed"]),
                "corruption": mechanism,
                "rate": float(cell["rate"]),
                "model": str(cell["model"]),
            }
        }

        if record is None or record["metrics"] is None:
            errors.append(f"No frozen Phase 2 record for probe cell {entry['cell']}")
            entry["match"] = False
            entry["reason"] = "missing_phase2_record"
            cells.append(entry)
            continue
        if record["status"] != "completed":
            errors.append(f"Frozen Phase 2 probe cell is not completed: {entry['cell']}")
            entry["match"] = False
            entry["reason"] = f"phase2_status_{record['status']}"
            cells.append(entry)
            continue

        task = {
            "dataset_id": data_id,
            "dataset_name": datasets[data_id],
            "seed": int(cell["seed"]),
            "corruption": mechanism,
            "rate": float(cell["rate"]),
            "model": str(cell["model"]),
            "cleaned": False,
            "arm": "b_probe",
            "parameters": normalize_parameters(parameters[mechanism]),
        }
        replay = runner.run(task)

        frozen = record["metrics"]
        metric_diffs: dict[str, Any] = {}
        for key in METRIC_KEYS:
            expected = frozen.get(key)
            actual = replay.get(key)
            metric_diffs[key] = {
                "phase2": expected,
                "phase3_replay": actual,
                "absolute_difference": (
                    abs(float(expected) - float(actual))
                    if expected is not None and actual is not None
                    else None
                ),
            }

        hash_diffs: dict[str, Any] = {}
        frozen_provenance = frozen.get("input_provenance") or {}
        replay_provenance = replay.get("input_provenance") or {}
        for key in ("model_X_train_sha256", "model_X_test_sha256"):
            hash_diffs[key] = {
                "phase2": frozen_provenance.get(key),
                "phase3_replay": replay_provenance.get(key),
                "equal": frozen_provenance.get(key) == replay_provenance.get(key),
            }

        metrics_equal = all(
            diff["absolute_difference"] == 0.0 for diff in metric_diffs.values()
        )
        hashes_equal = all(diff["equal"] for diff in hash_diffs.values())
        entry["metrics"] = metric_diffs
        entry["array_hashes"] = hash_diffs
        entry["match"] = bool(metrics_equal and hashes_equal)
        if not entry["match"]:
            entry["reason"] = "metric_or_hash_mismatch"
            errors.append(f"Paired-identity mismatch for probe cell {entry['cell']}")
        cells.append(entry)

    matched = sum(1 for entry in cells if entry.get("match"))
    return {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "tolerance": float(config["paired_identity_probe"]["tolerance"]),
        "metric_keys": list(METRIC_KEYS),
        "probe_cell_count": len(cells),
        "matched_cell_count": matched,
        "verdict": "VERIFIED" if cells and not errors else "BLOCKED",
        "errors": errors,
        "cells": cells,
    }
