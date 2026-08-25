"""Phase 3 ablation worker.

Single serial worker with the same discipline as Phase 2: process lock, atomic
per-task JSON written before the SQLite completion mark, heartbeat, graceful stop
after the current task, and resume that requeues an interrupted task without
consuming a model-failure retry.

It refuses to initialize or run without a valid Phase 3 authorization artifact.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from tfm_dirty_values.config import resolve_path
from tfm_dirty_values.models import verify_model_artifacts
from tfm_dirty_values.state import utc_now
from tfm_dirty_values.worker import Heartbeat, SingleWorkerLock

from .authorization import require_authorized
from .design import ablation_task_manifest, ablation_tasks, verify_expected_counts
from .paired_identity import run_paired_identity_probe
from .runner import AblationExperimentRunner
from .store import AblationStore


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def freeze_task_manifest(config: dict[str, Any], phase2_config: dict[str, Any]) -> dict[str, Any]:
    """Write the frozen Phase 3 task manifest. Produces no model outcome."""
    manifest = ablation_task_manifest(config, phase2_config)
    errors = verify_expected_counts(manifest, config)
    if errors:
        raise SystemExit("Cannot freeze Phase 3 manifest: " + "; ".join(errors))
    _atomic_json(resolve_path(config, config["paths"]["task_manifest"]), manifest)
    return manifest


def _store(config: dict[str, Any]) -> AblationStore:
    return AblationStore(
        resolve_path(config, config["paths"]["database"]),
        resolve_path(config, config["paths"]["events"]),
        resolve_path(config, config["paths"]["outputs"]),
    )


def initialize(config: dict[str, Any], phase2_config: dict[str, Any]) -> dict[str, Any]:
    """Queue the 2,425 Phase 3 tasks. Requires authorization; writes no outcome."""
    require_authorized(config, phase2_config)

    manifest_path = resolve_path(config, config["paths"]["task_manifest"])
    if not manifest_path.is_file():
        raise SystemExit(f"Missing frozen Phase 3 task manifest: {manifest_path}")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    live = ablation_task_manifest(config, phase2_config)
    if frozen.get("task_manifest_sha256") != live["task_manifest_sha256"]:
        raise SystemExit("Frozen Phase 3 task manifest does not match the generator")

    probe_path = resolve_path(config, "artifacts/phase3/paired-identity-probe.json")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe.get("verdict") != "VERIFIED":
        raise SystemExit("Paired-identity probe is not VERIFIED")

    store = _store(config)
    inserted = store.add_ablation_tasks(ablation_tasks(config, phase2_config))
    coverage = store.verify_arm_coverage()
    if coverage:
        raise SystemExit("Arm coverage check failed: " + "; ".join(coverage))
    store.mark_initialized(live, probe)
    store.event(
        "phase3_initialized",
        {"inserted": inserted, "task_count": live["task_count"], "arm_counts": live["arm_counts"]},
    )
    return {
        "inserted": inserted,
        "task_count": live["task_count"],
        "arm_counts": live["arm_counts"],
        "arm_coverage_errors": coverage,
    }


def _stop_requested(config: dict[str, Any]) -> Path:
    return resolve_path(config, config["paths"]["control_dir"]) / "stop.request"


def run_worker(config: dict[str, Any], phase2_config: dict[str, Any]) -> dict[str, Any]:
    """Execute queued Phase 3 tasks until the queue drains or a stop is requested."""
    require_authorized(config, phase2_config)
    verify_model_artifacts(phase2_config)

    database = resolve_path(config, config["paths"]["database"])
    if not database.is_file():
        raise SystemExit("Phase 3 database does not exist; run initialize first")

    # RunStore.__init__ creates the outputs directory and owns writing the
    # atomic per-task JSON, so the worker does not manage it separately.
    store = _store(config)

    worker_id = f"phase3-{uuid.uuid4().hex[:12]}"
    lock_path = resolve_path(config, config["paths"]["control_dir"]) / "worker.lock"
    heartbeat_path = resolve_path(config, config["paths"]["heartbeat"])
    stop_path = _stop_requested(config)
    if stop_path.is_file():
        stop_path.unlink()

    max_attempts = int(config["worker"]["max_attempts"])
    runner = AblationExperimentRunner(phase2_config)

    completed = 0
    failed = 0
    started = time.perf_counter()

    with SingleWorkerLock(lock_path):
        requeued = store.recover_interrupted(worker_id)
        if requeued:
            store.event("phase3_recovered_interrupted", {"count": requeued})

        heartbeat = Heartbeat(heartbeat_path, int(config["worker"]["heartbeat_seconds"]), worker_id)
        heartbeat.status = "running"
        heartbeat.start()
        try:
            while True:
                if stop_path.is_file():
                    store.event("phase3_stop_requested", {"worker_id": worker_id})
                    break
                task = store.claim_next_with_arm(worker_id, max_attempts)
                if task is None:
                    break

                heartbeat.start_task(str(task["task_id"]))
                task_started = time.perf_counter()
                try:
                    metrics = runner.run(task)
                    duration = time.perf_counter() - task_started
                    # RunStore.complete writes the atomic per-task JSON itself,
                    # in the Phase 2 nested {"task": ..., "metrics": ...} shape,
                    # before marking the row completed in SQLite. The claimed
                    # task dict already carries arm and cleaned, so they are
                    # persisted inside the "task" object. Do not write a second
                    # output file here; it would be overwritten.
                    store.complete(task, metrics, duration)
                    completed += 1
                except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                    duration = time.perf_counter() - task_started
                    store.fail(
                        task,
                        {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                        duration,
                    )
                    failed += 1
                finally:
                    heartbeat.finish_task()
        finally:
            heartbeat.close("stopped")

    wall = time.perf_counter() - started
    summary = {
        "worker_id": worker_id,
        "completed": completed,
        "failed": failed,
        "wall_seconds": wall,
        "open_tasks_remain": store.has_open_tasks(),
        "arm_counts": store.arm_counts(),
    }
    store.event("phase3_worker_finished", summary)
    return summary


def completion_report(config: dict[str, Any]) -> dict[str, Any]:
    store = _store(config)
    counts: dict[str, int] = {}
    for row in store.counts():
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + int(row["count"])
    expected = int(config["expected_counts"]["total_tasks"])
    total_completed = counts.get("completed", 0)
    gate = (
        "COMPLETE_PENDING_VALIDATION"
        if total_completed == expected and not counts.get("failed") and not store.has_open_tasks()
        else "INCOMPLETE"
    )
    report = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "expected_tasks": expected,
        "status_counts": counts,
        "arm_counts": store.arm_counts(),
        "arm_coverage_errors": store.verify_arm_coverage(),
        "gate": gate,
        "generated_at": utc_now(),
    }
    _atomic_json(resolve_path(config, config["paths"]["completion_report"]), report)
    return report
