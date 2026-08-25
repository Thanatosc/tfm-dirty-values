"""Read-only Phase 3 progress snapshot. Creates no state."""

from __future__ import annotations

import argparse
import json
import statistics
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfm_dirty_values.config import load_config, resolve_path  # noqa: E402


def _age_seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - moment).total_seconds()


def snapshot(config: dict) -> dict:
    database = resolve_path(config, config["paths"]["database"])
    expected = int(config["expected_counts"]["total_tasks"])
    if not database.is_file():
        return {"state": "not_initialized", "expected_tasks": expected}

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        status_counts = {
            str(row["status"]): int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM runs GROUP BY status"
            )
        }
        durations = [
            float(row["duration_sec"])
            for row in connection.execute(
                "SELECT duration_sec FROM runs WHERE duration_sec IS NOT NULL"
            )
        ]
        arms = {}
        for row in connection.execute(
            """SELECT a.arm AS arm, r.status AS status, COUNT(*) AS n
                 FROM ablation_arms a JOIN runs r ON r.task_id = a.task_id
             GROUP BY a.arm, r.status"""
        ):
            arms.setdefault(str(row["arm"]), {})[str(row["status"])] = int(row["n"])
        current = [
            dict(row)
            for row in connection.execute(
                "SELECT task_id, corruption, rate, model, dataset_name, started_at"
                " FROM runs WHERE status='running'"
            )
        ]
    finally:
        connection.close()

    completed = status_counts.get("completed", 0)
    remaining = expected - completed
    median = statistics.median(durations) if durations else None
    heartbeat_path = resolve_path(config, config["paths"]["heartbeat"])
    heartbeat = (
        json.loads(heartbeat_path.read_text(encoding="utf-8")) if heartbeat_path.is_file() else None
    )

    return {
        "state": "running" if current else ("complete" if remaining == 0 else "idle"),
        "expected_tasks": expected,
        "status_counts": status_counts,
        "arm_counts": arms,
        "completed": completed,
        "remaining": remaining,
        "percent": round(100.0 * completed / expected, 2) if expected else None,
        "median_task_seconds": round(median, 3) if median else None,
        "eta_minutes": round(remaining * median / 60.0, 1) if median and remaining > 0 else None,
        "current_tasks": current,
        "heartbeat_age_seconds": _age_seconds((heartbeat or {}).get("updated_at")),
        "heartbeat_status": (heartbeat or {}).get("status"),
        "heartbeat_pid": (heartbeat or {}).get("pid"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 status")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "cleaning_ablation.yaml")
    report = snapshot(config)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    print(f"state      : {report['state']}")
    if report["state"] == "not_initialized":
        print(f"expected   : {report['expected_tasks']} tasks (nothing queued yet)")
        return 0
    print(f"progress   : {report['completed']}/{report['expected_tasks']}  ({report['percent']}%)")
    print(f"statuses   : {report['status_counts']}")
    print(f"arms       : {report['arm_counts']}")
    print(f"median task: {report['median_task_seconds']} s")
    if report["eta_minutes"] is not None:
        print(f"eta        : ~{report['eta_minutes']} min")
    if report["heartbeat_age_seconds"] is not None:
        print(
            f"heartbeat  : {report['heartbeat_status']} "
            f"pid={report['heartbeat_pid']} age={report['heartbeat_age_seconds']:.0f}s"
        )
    for task in report["current_tasks"]:
        print(
            f"running    : {task['dataset_name']} {task['corruption']}"
            f"@{task['rate']} {task['model']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
