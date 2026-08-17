from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from tfm_dirty_values.config import canonical_json, load_config
from tfm_dirty_values.full_analysis import evaluate_full_study


EXPECTED_ANALYSIS_SHA256 = (
    "7d9389384762bc04bd352f0540f225d05d92b8f60b99d309232adc0d9ebbfe30"
)
EXPECTED_RUNTIME_IDENTITY = (
    "9325544b8adb82eaa4ddbc99c12b01364315edeaddf17e7fe68d7e41efe8700d"
)
EXPECTED_TASK_COUNT = 4575


class ReadOnlyStore:
    def __init__(self, database: Path) -> None:
        self.database = database.resolve()

    def rows(self, *, stage: int | None = None, status: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if stage is not None:
            clauses.append("stage=?")
            values.append(stage)
        if status is not None:
            clauses.append("status=?")
            values.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        uri = self.database.as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            records = connection.execute(f"SELECT * FROM runs{where}", values).fetchall()
        return [dict(row) for row in records]


class MetricAliasStore:
    def __init__(self, store: ReadOnlyStore) -> None:
        self.store = store
        self.alias_count = 0

    def rows(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        adapted: list[dict[str, Any]] = []
        for source_row in self.store.rows(*args, **kwargs):
            row = dict(source_row)
            raw_metrics = row.get("metrics_json")
            if raw_metrics is None:
                adapted.append(row)
                continue
            metrics = json.loads(str(raw_metrics))
            stored = metrics.get("ece_10_equal_width")
            generic = metrics.get("ece")
            if generic is not None and stored is not None and float(generic) != float(stored):
                raise ValueError(f"Conflicting ECE values for task {row.get('task_id')}")
            if generic is None:
                if stored is None:
                    raise ValueError(f"Missing ECE value for task {row.get('task_id')}")
                metrics["ece"] = float(stored)
                row["metrics_json"] = canonical_json(metrics)
                self.alias_count += 1
            adapted.append(row)
        return adapted


def validate_tasks(store: ReadOnlyStore, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = {str(task["task_id"]) for task in manifest["tasks"]}
    rows = store.rows(stage=3)
    actual_ids = {str(row["task_id"]) for row in rows}
    if len(rows) != EXPECTED_TASK_COUNT or len(expected_ids) != EXPECTED_TASK_COUNT:
        raise RuntimeError("The archived task panel does not contain exactly 4,575 tasks.")
    if actual_ids != expected_ids:
        raise RuntimeError("The archived database task identities do not match the frozen manifest.")
    if any(row["status"] != "completed" or row["metrics_json"] is None for row in rows):
        raise RuntimeError("The archived database contains an incomplete or metric-free task.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_study.yaml")
    parser.add_argument("--database", default="results/full_study/state.sqlite3")
    parser.add_argument(
        "--task-manifest", default="artifacts/phase2/full-study-task-manifest.json"
    )
    parser.add_argument("--output", default="results/full_study/analysis-public-rerun.json")
    args = parser.parse_args()

    config = load_config(args.config)
    source = ReadOnlyStore(Path(args.database))
    validate_tasks(source, Path(args.task_manifest))
    store = MetricAliasStore(source)
    report = evaluate_full_study(store, config)
    report["compatibility_amendment"] = {
        "type": "metric_key_alias",
        "source_key": "ece_10_equal_width",
        "analysis_key": "ece",
        "aliased_task_count": store.alias_count,
        "source_database_mutated": False,
        "authorized_runtime_code_identity_sha256": EXPECTED_RUNTIME_IDENTITY,
    }
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    observed = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    if observed != EXPECTED_ANALYSIS_SHA256:
        raise RuntimeError(
            "Recomputed analysis identity changed: "
            f"expected {EXPECTED_ANALYSIS_SHA256}, observed {observed}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"OK: 4,575 tasks reproduced analysis SHA-256 {observed}")


if __name__ == "__main__":
    main()
