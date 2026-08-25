"""Verify every Phase 3 atomic output against the SQLite queue.

Checks, per task: the output file exists, its task_id matches, its status is
completed, its arm agrees with the sidecar table, and its metrics are byte-equal
to the canonical JSON stored in SQLite. Also confirms there are no extra output
files and emits an aggregate output-set identity hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfm_dirty_values.config import canonical_json, load_config, resolve_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 3 results")
    parser.add_argument("--config", default="configs/cleaning_ablation.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    database = resolve_path(config, config["paths"]["database"])
    output_dir = resolve_path(config, config["paths"]["outputs"])

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in connection.execute(
                """SELECT r.task_id, r.status, r.metrics_json, a.arm AS arm,
                          a.cleaned AS cleaned
                     FROM runs r JOIN ablation_arms a ON a.task_id = r.task_id"""
            )
        ]
    finally:
        connection.close()

    errors: list[str] = []
    hashes: list[tuple[str, str]] = []
    for row in rows:
        task_id = str(row["task_id"])
        path = output_dir / f"{task_id}.json"
        if not path.is_file():
            errors.append(f"missing:{task_id}")
            continue
        raw = path.read_bytes()
        hashes.append((task_id, hashlib.sha256(raw).hexdigest()))
        output = json.loads(raw)
        # RunStore.complete writes the Phase 2 nested shape, so arm and cleaned
        # live inside the "task" object rather than at the top level.
        task = output.get("task") or {}
        if task.get("task_id") != task_id:
            errors.append(f"task_id:{task_id}")
        if row["status"] != "completed":
            errors.append(f"status:{task_id}:{row['status']}")
        if task.get("arm") != str(row["arm"]):
            errors.append(f"arm:{task_id}")
        if bool(task.get("cleaned")) != bool(row["cleaned"]):
            errors.append(f"cleaned:{task_id}")
        if canonical_json(output.get("metrics")) != row["metrics_json"]:
            errors.append(f"metrics:{task_id}")

    on_disk = sorted(path.name for path in output_dir.glob("*.json"))
    expected_names = sorted(f"{str(row['task_id'])}.json" for row in rows)
    extra = sorted(set(on_disk) - set(expected_names))
    if extra:
        errors.append(f"extra_output_files:{len(extra)}")

    identity = hashlib.sha256(canonical_json(sorted(hashes)).encode("ascii")).hexdigest()
    report = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "rows": len(rows),
        "outputs_on_disk": len(on_disk),
        "verified_outputs": len(hashes),
        "expected_tasks": int(config["expected_counts"]["total_tasks"]),
        "error_count": len(errors),
        "errors": errors[:50],
        "output_set_identity_sha256": identity.upper(),
        "status": "VERIFIED" if not errors and len(rows) == int(
            config["expected_counts"]["total_tasks"]
        ) else "FAILED",
    }

    target = resolve_path(config, "results/cleaning_ablation/result-verification.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target)

    print(f"status   : {report['status']}")
    print(f"rows     : {report['rows']} / {report['expected_tasks']}")
    print(f"outputs  : {report['verified_outputs']} verified, {report['outputs_on_disk']} on disk")
    print(f"errors   : {report['error_count']}")
    for error in report["errors"]:
        print(f"  - {error}")
    print(f"identity : {report['output_set_identity_sha256'][:24]}...")
    print(f"written  : {target}")
    return 0 if report["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
