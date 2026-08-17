from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import canonical_json
from .design import Task


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, database: Path, events_path: Path, outputs_dir: Path):
        self.database = database
        self.events_path = events_path
        self.outputs_dir = outputs_dir
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._event_lock = threading.Lock()
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    task_id TEXT PRIMARY KEY,
                    stage INTEGER NOT NULL,
                    dataset_id INTEGER NOT NULL,
                    dataset_name TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    corruption TEXT NOT NULL,
                    rate REAL NOT NULL,
                    model TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    interruptions INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    queued_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    duration_sec REAL,
                    metrics_json TEXT,
                    error_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_status_stage
                    ON runs(status, stage, dataset_id, seed, corruption, rate, model);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "interruptions" not in columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN interruptions INTEGER NOT NULL DEFAULT 0"
                )

    def set_metadata(self, key: str, value: Any) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO metadata(key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at""",
                (key, canonical_json(value), utc_now()),
            )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM metadata WHERE key=?",
                (key,),
            ).fetchone()
        return default if row is None else json.loads(row["value_json"])

    def add_tasks(self, tasks: Sequence[Task]) -> int:
        inserted = 0
        with self.connect() as connection:
            for task in tasks:
                before = connection.total_changes
                connection.execute(
                    """INSERT OR IGNORE INTO runs(
                        task_id, stage, dataset_id, dataset_name, seed,
                        corruption, rate, model, parameters_json, queued_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        task.task_id,
                        task.stage,
                        task.dataset_id,
                        task.dataset_name,
                        task.seed,
                        task.corruption,
                        task.rate,
                        task.model,
                        canonical_json(task.parameters),
                        utc_now(),
                    ),
                )
                inserted += connection.total_changes - before
        if inserted:
            self.event("tasks_added", {"count": inserted})
        return inserted

    def recover_interrupted(self, worker_id: str) -> int:
        with self.connect() as connection:
            rows = connection.execute("SELECT task_id FROM runs WHERE status='running'").fetchall()
            connection.execute(
                """UPDATE runs SET status='pending', worker_id=NULL, started_at=NULL,
                attempts=CASE WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                interruptions=interruptions + 1, error_json=? WHERE status='running'""",
                (
                    canonical_json(
                        {"type": "InterruptedRun", "recovered_by": worker_id, "at": utc_now()}
                    ),
                ),
            )
        if rows:
            self.event("interrupted_tasks_requeued", {"count": len(rows), "worker_id": worker_id})
        return len(rows)

    def claim_next(self, worker_id: str, max_attempts: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM runs
                WHERE status IN ('pending', 'retry') AND attempts < ?
                ORDER BY stage, dataset_id, seed, corruption, rate, model
                LIMIT 1""",
                (max_attempts,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE runs SET status='running', attempts=attempts+1,
                worker_id=?, started_at=?, finished_at=NULL WHERE task_id=?""",
                (worker_id, utc_now(), row["task_id"]),
            )
            claimed = dict(row)
            claimed["attempts"] += 1
            claimed["status"] = "running"
            claimed["worker_id"] = worker_id
            claimed["parameters"] = json.loads(claimed.pop("parameters_json"))
            return claimed

    def complete(self, task: dict[str, Any], metrics: dict[str, Any], duration: float) -> None:
        finished = utc_now()
        output = {"task": task, "metrics": metrics, "finished_at": finished, "duration_sec": duration}
        target = self.outputs_dir / f"{task['task_id']}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        with self.connect() as connection:
            connection.execute(
                """UPDATE runs SET status='completed', finished_at=?, duration_sec=?,
                metrics_json=?, error_json=NULL WHERE task_id=?""",
                (finished, duration, canonical_json(metrics), task["task_id"]),
            )
        self.event("task_completed", {"task_id": task["task_id"], "duration_sec": duration})

    def fail(
        self,
        task: dict[str, Any],
        error: dict[str, Any],
        duration: float,
        max_attempts: int,
    ) -> str:
        status = "retry" if int(task["attempts"]) < max_attempts else "failed"
        with self.connect() as connection:
            connection.execute(
                """UPDATE runs SET status=?, finished_at=?, duration_sec=?, error_json=?
                WHERE task_id=?""",
                (status, utc_now(), duration, canonical_json(error), task["task_id"]),
            )
        self.event("task_failed", {"task_id": task["task_id"], "status": status, "error": error})
        return status

    def counts(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT stage, status, COUNT(*) AS count,
                AVG(CASE WHEN status='completed' THEN duration_sec END) AS avg_duration
                FROM runs GROUP BY stage, status ORDER BY stage, status"""
            ).fetchall()
        return [dict(row) for row in rows]

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
        with self.connect() as connection:
            records = connection.execute(f"SELECT * FROM runs{where}", values).fetchall()
        return [dict(row) for row in records]

    def has_open_tasks(self, stage: int | None = None) -> bool:
        where = " AND stage=?" if stage is not None else ""
        params = (stage,) if stage is not None else ()
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM runs "
                f"WHERE status IN ('pending','retry','running'){where}",
                params,
            ).fetchone()
        return bool(row["count"])

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {"timestamp": utc_now(), "event_type": event_type, "payload": payload}
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._event_lock:
            with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            with self.connect() as connection:
                connection.execute(
                    "INSERT INTO events(timestamp, event_type, payload_json) VALUES (?, ?, ?)",
                    (record["timestamp"], event_type, canonical_json(payload)),
                )
