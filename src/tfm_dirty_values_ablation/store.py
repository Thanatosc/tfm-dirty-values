"""Phase 3 arm sidecar on top of the frozen Phase 2 ``RunStore``.

``RunStore`` is reused unchanged for queueing, claiming, atomic result commit,
recovery, and event logging. Its ``runs`` schema has no arm or cleaned column,
and the arm cannot be folded into ``parameters_json`` because the runner passes
that dictionary directly into ``inject_corruption``, where any extra key would
change the derived corruption seed and break pairing with frozen arm B.

So the arm assignment lives in a separate ``ablation_arms`` table keyed by
``task_id`` in the same Phase 3 database. ``parameters_json`` stays exactly the
corruption parameters Phase 2 used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from tfm_dirty_values.state import RunStore, utc_now

from .design import AblationTask


class AblationStore(RunStore):
    """RunStore plus a task_id -> (arm, cleaned) sidecar."""

    def __init__(self, database: Path, events_path: Path, outputs_dir: Path):
        super().__init__(database, events_path, outputs_dir)
        self._initialize_arms()

    def _initialize_arms(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ablation_arms (
                    task_id TEXT PRIMARY KEY,
                    arm TEXT NOT NULL,
                    cleaned INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ablation_arms_arm
                    ON ablation_arms(arm);
                """
            )

    def add_ablation_tasks(self, tasks: Sequence[AblationTask]) -> int:
        inserted = super().add_tasks(tasks)
        with self.connect() as connection:
            for task in tasks:
                connection.execute(
                    """INSERT OR IGNORE INTO ablation_arms(task_id, arm, cleaned)
                       VALUES (?, ?, ?)""",
                    (task.task_id, task.arm, int(task.cleaned)),
                )
        return inserted

    def arm_for(self, task_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT arm, cleaned FROM ablation_arms WHERE task_id=?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"No arm assignment recorded for task {task_id}")
        return {"arm": str(row["arm"]), "cleaned": bool(row["cleaned"])}

    def claim_next_with_arm(self, worker_id: str, max_attempts: int) -> dict[str, Any] | None:
        task = self.claim_next(worker_id, max_attempts)
        if task is None:
            return None
        task.update(self.arm_for(str(task["task_id"])))
        return task

    def arm_counts(self) -> dict[str, dict[str, int]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT a.arm AS arm, r.status AS status, COUNT(*) AS n
                     FROM ablation_arms a
                     JOIN runs r ON r.task_id = a.task_id
                 GROUP BY a.arm, r.status""",
            ).fetchall()
        counts: dict[str, dict[str, int]] = {}
        for row in rows:
            counts.setdefault(str(row["arm"]), {})[str(row["status"])] = int(row["n"])
        return counts

    def verify_arm_coverage(self) -> list[str]:
        """Every queued run must have exactly one recorded arm."""
        errors: list[str] = []
        with self.connect() as connection:
            missing = connection.execute(
                """SELECT COUNT(*) AS n FROM runs r
                    LEFT JOIN ablation_arms a ON a.task_id = r.task_id
                    WHERE a.task_id IS NULL"""
            ).fetchone()
            orphan = connection.execute(
                """SELECT COUNT(*) AS n FROM ablation_arms a
                    LEFT JOIN runs r ON r.task_id = a.task_id
                    WHERE r.task_id IS NULL"""
            ).fetchone()
        if int(missing["n"]):
            errors.append(f"{int(missing['n'])} queued runs have no arm assignment")
        if int(orphan["n"]):
            errors.append(f"{int(orphan['n'])} arm assignments have no queued run")
        return errors

    def mark_initialized(self, manifest: dict[str, Any], probe: dict[str, Any]) -> None:
        self.set_metadata("phase3_task_manifest_sha256", manifest["task_manifest_sha256"])
        self.set_metadata("phase3_task_count", manifest["task_count"])
        self.set_metadata("phase3_arm_counts", manifest["arm_counts"])
        self.set_metadata("phase3_paired_identity_verdict", probe["verdict"])
        self.set_metadata("phase3_initialized_at", utc_now())
