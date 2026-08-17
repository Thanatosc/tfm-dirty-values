from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import psutil

from .analysis import evaluate_kill_test
from .authorization import require_authorized
from .config import resolve_path
from .experiment import ExperimentRunner
from .full_authorization import require_full_authorized
from .models import verify_model_artifacts
from .state import RunStore, utc_now


class SingleWorkerLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "SingleWorkerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if self.handle.read(1) == b"":
            self.handle.seek(0)
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            raise RuntimeError("Another experiment worker holds the process lock") from error
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


class Heartbeat:
    def __init__(self, path: Path, interval: int, worker_id: str):
        self.path = path
        self.interval = interval
        self.worker_id = worker_id
        self.stop_event = threading.Event()
        self.current_task: str | None = None
        self.current_task_started_at: str | None = None
        self.status = "starting"
        self.started_at = utc_now()
        self.initial_rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        self._write_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="dirty-audit-heartbeat", daemon=True)

    def _write(self) -> None:
        process = psutil.Process(os.getpid())
        payload = {
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "status": self.status,
            "current_task": self.current_task,
            "current_task_started_at": self.current_task_started_at,
            "initial_rss_bytes": self.initial_rss_bytes,
            "rss_bytes": process.memory_info().rss,
        }
        with self._write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            self._write()

    def start(self) -> None:
        self._write()
        self.thread.start()

    def start_task(self, task_id: str) -> None:
        self.current_task = task_id
        self.current_task_started_at = utc_now()
        self._write()

    def finish_task(self) -> None:
        self.current_task = None
        self.current_task_started_at = None
        self._write()

    def close(self, status: str) -> None:
        self.status = status
        self.current_task = None
        self.current_task_started_at = None
        self.stop_event.set()
        self.thread.join(timeout=5)
        self._write()


def create_store(config: dict[str, Any]) -> RunStore:
    return RunStore(
        resolve_path(config, config["paths"]["database"]),
        resolve_path(config, config["paths"]["events"]),
        resolve_path(config, config["paths"]["outputs"]),
    )


def _atomic_decision(config: dict[str, Any], report: dict[str, Any]) -> Path:
    if config.get("protocol_version") == "phase2_prereg_v1":
        path = resolve_path(config, config["paths"]["completion_report"])
    else:
        result_dir = resolve_path(config, config["paths"]["result_dir"])
        path = result_dir / "decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _failure_fraction(store: RunStore, stage: int) -> float:
    rows = store.rows(stage=stage)
    if not rows:
        return 1.0
    return sum(row["status"] == "failed" for row in rows) / len(rows)


def run_worker(config: dict[str, Any]) -> int:
    full_study = config.get("protocol_version") == "phase2_prereg_v1"
    if full_study:
        require_full_authorized(config)
    else:
        require_authorized(config)
    result_dir = resolve_path(config, config["paths"]["result_dir"])
    control_dir = resolve_path(config, config["paths"]["control_dir"])
    heartbeat_path = resolve_path(config, config["paths"]["heartbeat"])
    stop_path = control_dir / "stop.request"
    control_dir.mkdir(parents=True, exist_ok=True)
    worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:10]}"
    heartbeat = Heartbeat(heartbeat_path, int(config["worker"]["heartbeat_seconds"]), worker_id)
    stop_event = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        with SingleWorkerLock(result_dir / "worker.lock"):
            store = create_store(config)
            rows = store.rows()
            if not rows:
                raise RuntimeError("No tasks are initialized; run the explicit init command first")
            existing_hash = store.get_metadata("config_sha256")
            if existing_hash != config["_config_sha256"]:
                raise RuntimeError("Configuration hash differs from the initialized run")
            store.set_metadata("model_artifacts", verify_model_artifacts(config))
            recovered = store.recover_interrupted(worker_id)
            store.event("worker_started", {"worker_id": worker_id, "pid": os.getpid(), "recovered": recovered})
            heartbeat.status = "running"
            heartbeat.start()
            runner = ExperimentRunner(config)
            max_attempts = int(config["worker"]["max_attempts"])

            while True:
                if stop_event.is_set() or stop_path.exists():
                    store.event("worker_stop_requested", {"worker_id": worker_id})
                    heartbeat.close("stopped")
                    store.set_metadata("worker_state", "stopped")
                    return 0
                task = store.claim_next(worker_id, max_attempts)
                if task is None:
                    break
                heartbeat.start_task(task["task_id"])
                started = time.perf_counter()
                try:
                    metrics = runner.run(task)
                    store.complete(task, metrics, time.perf_counter() - started)
                except Exception as error:  # noqa: BLE001
                    failure = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    store.fail(task, failure, time.perf_counter() - started, max_attempts)
                finally:
                    heartbeat.finish_task()

            latest_stage = max(int(row["stage"]) for row in store.rows())
            incomplete_rows = [
                row for row in store.rows(stage=latest_stage) if row["status"] != "completed"
            ]
            if full_study and incomplete_rows:
                incomplete_statuses: dict[str, int] = {}
                for row in incomplete_rows:
                    status = str(row["status"])
                    incomplete_statuses[status] = incomplete_statuses.get(status, 0) + 1
                report = {
                    "stage": latest_stage,
                    "status": "BLOCKED",
                    "reason": "Confirmatory analysis requires every frozen task to complete",
                    "incomplete_statuses": incomplete_statuses,
                }
            elif full_study:
                report = {
                    "stage": latest_stage,
                    "status": "COMPLETE_PENDING_VALIDATION",
                    "task_count": len(store.rows(stage=latest_stage)),
                    "reason": "Model execution is complete; preregistered analysis has not yet run",
                }
            elif _failure_fraction(store, latest_stage) > 0.05:
                report = {
                    "stage": latest_stage,
                    "decision": "BLOCKED",
                    "reason": "More than 5% of tasks failed after the permitted deterministic retry",
                }
            else:
                try:
                    report = evaluate_kill_test(store, config, stage=latest_stage)
                except (KeyError, TypeError, ValueError) as error:
                    report = {
                        "stage": latest_stage,
                        "decision": "BLOCKED",
                        "reason": "Preregistered analysis validation failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
            metadata_key = "full_study_completion" if full_study else f"stage{latest_stage}_decision"
            store.set_metadata(metadata_key, report)
            store.event(
                "full_study_execution_complete" if full_study else "kill_test_stage_complete",
                {
                    "stage": latest_stage,
                    "status": report.get("status", report.get("decision")),
                },
            )
            _atomic_decision(config, report)
            heartbeat.close("complete")
            store.set_metadata("worker_state", "complete")
            return 0
    except Exception:  # noqa: BLE001
        try:
            if not heartbeat.stop_event.is_set():
                heartbeat.close("crashed")
        except Exception:  # noqa: BLE001
            pass
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
