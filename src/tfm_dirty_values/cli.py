from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from .analysis import evaluate_kill_test
from .authorization import read_authorization, require_authorized, write_authorization
from .config import canonical_json, load_config, resolve_path
from .datasets import DatasetManager
from .design import full_study_tasks, kill_test_tasks
from .full_analysis import evaluate_full_study
from .full_authorization import (
    read_full_authorization,
    require_full_authorized,
    write_full_authorization,
)
from .full_study import (
    freeze_full_task_manifest,
    full_study_preflight,
    full_task_manifest,
    write_full_study_preflight,
)
from .models import verify_model_artifacts
from .worker import create_store, run_worker


def _default_config() -> str:
    return str(Path(__file__).resolve().parents[2] / "configs" / "kill_test.yaml")


def _task_manifest(config: dict[str, Any], *, stress: bool = False) -> dict[str, Any]:
    tasks = kill_test_tasks(config, stress=stress)
    payload = [task.to_dict() for task in tasks]
    return {
        "experiment_id": config["experiment_id"],
        "stage": 2 if stress else 1,
        "config_sha256": config["_config_sha256"],
        "task_count": len(tasks),
        "task_manifest_sha256": hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest(),
        "tasks": payload,
    }


def _read_heartbeat(config: dict[str, Any]) -> dict[str, Any] | None:
    path = resolve_path(config, config["paths"]["heartbeat"])
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(payload["updated_at"])
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        payload["age_seconds"] = max(
            0.0, (datetime.now(timezone.utc) - updated).total_seconds()
        )
        payload["process_alive"] = psutil.pid_exists(int(payload["pid"]))
        payload["stale"] = payload["age_seconds"] > float(
            config["worker"]["stale_heartbeat_seconds"]
        )
        initial_rss = int(payload.get("initial_rss_bytes") or 0)
        current_rss = int(payload.get("rss_bytes") or 0)
        payload["memory_multiplier"] = (
            float(current_rss / initial_rss) if initial_rss > 0 else None
        )
        return payload
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return {"status": "invalid", "process_alive": False}


def _worker_status(heartbeat: dict[str, Any] | None) -> str:
    if heartbeat is None:
        return "not_started"
    status = str(heartbeat.get("status", "invalid"))
    if status in {"starting", "running"}:
        if not heartbeat.get("process_alive"):
            return "interrupted"
        if heartbeat.get("stale"):
            return "stale"
    return status


def _status_breakdown(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = str(row[field])
        counts[key][str(row["status"])] += 1
        counts[key]["total"] += 1
    records: list[dict[str, Any]] = []
    for key in sorted(counts):
        record = {field: key, **dict(counts[key])}
        completed = int(record.get("completed", 0))
        total = int(record["total"])
        record["percent"] = 100.0 * completed / total if total else 0.0
        records.append(record)
    return records


def _duration_estimate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        row
        for row in rows
        if row["status"] == "completed"
        and row.get("duration_sec") is not None
        and math.isfinite(float(row["duration_sec"]))
        and float(row["duration_sec"]) >= 0.0
    ]
    remaining = [
        row for row in rows if row["status"] in {"pending", "retry", "running"}
    ]
    if not remaining:
        return {
            "basis": "all_tasks_terminal",
            "observed_completed_tasks": len(completed),
            "median_task_seconds": (
                float(statistics.median(float(row["duration_sec"]) for row in completed))
                if completed
                else None
            ),
            "estimated_tasks_per_hour": None,
            "eta_seconds": 0.0,
            "eta_hours": 0.0,
            "estimated_completion_at": datetime.now(timezone.utc).isoformat(),
        }
    if not completed:
        return {
            "basis": "unavailable_until_first_task_completes",
            "observed_completed_tasks": 0,
            "median_task_seconds": None,
            "estimated_tasks_per_hour": None,
            "eta_seconds": None,
            "eta_hours": None,
            "estimated_completion_at": None,
        }

    exact_values: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    model_corruption_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    model_values: dict[str, list[float]] = defaultdict(list)
    durations: list[float] = []
    for row in completed:
        duration = float(row["duration_sec"])
        exact_values[(int(row["dataset_id"]), str(row["model"]), str(row["corruption"]))].append(
            duration
        )
        model_corruption_values[(str(row["model"]), str(row["corruption"]))].append(duration)
        model_values[str(row["model"])].append(duration)
        durations.append(duration)
    exact = {key: float(statistics.median(values)) for key, values in exact_values.items()}
    model_corruption = {
        key: float(statistics.median(values)) for key, values in model_corruption_values.items()
    }
    by_model = {key: float(statistics.median(values)) for key, values in model_values.items()}
    global_median = float(statistics.median(durations))

    estimated_seconds = 0.0
    for row in remaining:
        exact_key = (int(row["dataset_id"]), str(row["model"]), str(row["corruption"]))
        model_corruption_key = (str(row["model"]), str(row["corruption"]))
        estimated_seconds += exact.get(
            exact_key,
            model_corruption.get(
                model_corruption_key,
                by_model.get(str(row["model"]), global_median),
            ),
        )
    return {
        "basis": "hierarchical_median_by_dataset_model_corruption",
        "observed_completed_tasks": len(completed),
        "median_task_seconds": global_median,
        "estimated_tasks_per_hour": 3600.0 / global_median if global_median > 0 else None,
        "eta_seconds": estimated_seconds,
        "eta_hours": estimated_seconds / 3600.0,
        "estimated_completion_at": datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + estimated_seconds,
            tz=timezone.utc,
        ).isoformat(),
    }


def _current_task_payload(
    rows: list[dict[str, Any]], heartbeat: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not heartbeat or not heartbeat.get("current_task"):
        return None
    task_id = str(heartbeat["current_task"])
    row = next((candidate for candidate in rows if str(candidate["task_id"]) == task_id), None)
    started_at = heartbeat.get("current_task_started_at")
    elapsed: float | None = None
    if started_at:
        try:
            started = datetime.fromisoformat(str(started_at))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            elapsed = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
        except ValueError:
            elapsed = None
    if row is None:
        return {"task_id": task_id, "elapsed_seconds": elapsed}
    return {
        "task_id": task_id,
        "dataset_id": int(row["dataset_id"]),
        "dataset_name": str(row["dataset_name"]),
        "seed": int(row["seed"]),
        "model": str(row["model"]),
        "corruption": str(row["corruption"]),
        "rate": float(row["rate"]),
        "attempts": int(row["attempts"]),
        "interruptions": int(row.get("interruptions") or 0),
        "elapsed_seconds": elapsed,
    }


def status_payload(config: dict[str, Any]) -> dict[str, Any]:
    store = create_store(config)
    rows = store.rows()
    counts = store.counts()
    total = sum(row["count"] for row in counts)
    completed = sum(row["count"] for row in counts if row["status"] == "completed")
    failed = sum(row["count"] for row in counts if row["status"] == "failed")
    running = sum(row["count"] for row in counts if row["status"] == "running")
    retry = sum(row["count"] for row in counts if row["status"] == "retry")
    heartbeat = _read_heartbeat(config)
    worker_status = _worker_status(heartbeat)
    eta = _duration_estimate(rows)
    current_task = _current_task_payload(rows, heartbeat)
    stop_requested = (
        resolve_path(config, config["paths"]["control_dir"]) / "stop.request"
    ).is_file()
    alerts: list[dict[str, str]] = []
    if worker_status == "interrupted":
        alerts.append(
            {
                "type": "WORKER_INTERRUPTED",
                "message": "Heartbeat says running but the worker process is absent; resume is available.",
            }
        )
    elif worker_status == "stale":
        alerts.append(
            {
                "type": "HEARTBEAT_STALE",
                "message": "Worker heartbeat exceeded the configured freshness threshold.",
            }
        )
    if heartbeat and heartbeat.get("memory_multiplier") is not None:
        if float(heartbeat["memory_multiplier"]) > 3.0:
            alerts.append(
                {
                    "type": "RESOURCE_ANOMALY",
                    "message": "Worker RSS exceeds three times its launch value.",
                }
            )
    if current_task and current_task.get("elapsed_seconds") is not None:
        median_seconds = eta.get("median_task_seconds")
        stall_threshold = max(1800.0, 10.0 * float(median_seconds or 0.0))
        if float(current_task["elapsed_seconds"]) > stall_threshold:
            alerts.append(
                {
                    "type": "TASK_STALL_SUSPECTED",
                    "message": (
                        f"Current task has run for {current_task['elapsed_seconds']:.0f}s; "
                        f"advisory threshold is {stall_threshold:.0f}s."
                    ),
                }
            )
    if failed:
        alerts.append(
            {
                "type": "FAILED_TASKS",
                "message": f"{failed} task(s) exhausted deterministic retries.",
            }
        )
    if stop_requested:
        alerts.append(
            {
                "type": "STOP_REQUESTED",
                "message": "The worker will stop after its current task is persisted.",
            }
        )
    return {
        "experiment_id": config["experiment_id"],
        "config_sha256": config["_config_sha256"],
        "total_tasks": total,
        "completed": completed,
        "failed": failed,
        "running": running,
        "retry": retry,
        "remaining": max(0, total - completed - failed),
        "percent": 100.0 * completed / total if total else 0.0,
        "interruptions": sum(int(row.get("interruptions") or 0) for row in rows),
        "counts": counts,
        "worker_status": worker_status,
        "stop_requested": stop_requested,
        "heartbeat": heartbeat,
        "current_task": current_task,
        "eta": eta,
        "by_model": _status_breakdown(rows, "model"),
        "by_corruption": _status_breakdown(rows, "corruption"),
        "alerts": alerts,
        "worker_state": store.get_metadata("worker_state"),
        "full_study_completion": store.get_metadata("full_study_completion"),
        "stage1_decision": store.get_metadata("stage1_decision"),
        "stage2_decision": store.get_metadata("stage2_decision"),
    }


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unavailable"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="TFM dirty-feature audit controller")
    parser.add_argument("--config", default=_default_config())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="verify checkpoints and resolved configuration")
    subparsers.add_parser("manifest", help="print the frozen Stage 1 task manifest")
    authorize = subparsers.add_parser("authorize", help="write the external Kill Test authorization artifact")
    authorize.add_argument("--confirmed-by-project-owner", action="store_true")
    subparsers.add_parser("prepare-data", help="validate metadata and freeze row splits without running models")
    subparsers.add_parser("init", help="initialize Stage 1 tasks after authorization")
    subparsers.add_parser("init-stress", help="initialize the one permitted stress retest")
    worker = subparsers.add_parser("worker", help="run or resume initialized tasks")
    worker.add_argument("--stdout-log")
    worker.add_argument("--stderr-log")
    status = subparsers.add_parser("status", help="show persisted progress")
    status.add_argument("--json", action="store_true")
    analyze = subparsers.add_parser("analyze", help="recompute the preregistered decision")
    analyze.add_argument("--stage", type=int, choices=[1, 2], default=1)
    subparsers.add_parser("full-manifest", help="print the frozen Phase 2 task manifest")
    subparsers.add_parser(
        "freeze-full-manifest", help="write the Phase 2 task manifest without running models"
    )
    subparsers.add_parser(
        "full-preflight", help="validate all Phase 2 inputs without initializing result state"
    )
    authorize_full = subparsers.add_parser(
        "authorize-full", help="write the external full-study authorization artifact"
    )
    authorize_full.add_argument("--confirmed-by-project-owner", action="store_true")
    subparsers.add_parser("init-full", help="initialize Phase 2 tasks after external authorization")
    full_worker = subparsers.add_parser("full-worker", help="run or resume initialized Phase 2 tasks")
    full_worker.add_argument("--stdout-log")
    full_worker.add_argument("--stderr-log")
    subparsers.add_parser("analyze-full", help="run the frozen dataset-level Phase 2 analysis")
    args = parser.parse_args()
    config = load_config(args.config)
    full_protocol = config.get("protocol_version") == "phase2_prereg_v1"

    kill_only = {"authorize", "manifest", "init", "init-stress", "analyze"}
    full_only = {
        "full-manifest",
        "freeze-full-manifest",
        "full-preflight",
        "authorize-full",
        "init-full",
        "full-worker",
        "analyze-full",
    }
    if full_protocol and args.command in kill_only:
        raise SystemExit(f"Command {args.command} is not legal for the full-study configuration")
    if not full_protocol and args.command in full_only:
        raise SystemExit(f"Command {args.command} requires the full-study configuration")

    if args.command == "authorize":
        if not args.confirmed_by_project_owner:
            raise SystemExit("Refusing authorization without --confirmed-by-project-owner")
        payload = write_authorization(
            config,
            command="powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_kill_test.ps1 -ConfirmPreregisteredRun",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "verify":
        if full_protocol:
            print(json.dumps(full_study_preflight(config), ensure_ascii=False, indent=2))
            return
        print(
            json.dumps(
                {
                    "config_sha256": config["_config_sha256"],
                    "embedded_status": config["status"],
                    "authorization": read_authorization(config),
                    "stage1_manifest": _task_manifest(config),
                    "artifacts": verify_model_artifacts(config),
                },
                indent=2,
            )
        )
    elif args.command == "manifest":
        print(json.dumps(_task_manifest(config), ensure_ascii=False, indent=2))
    elif args.command == "full-manifest":
        print(json.dumps(full_task_manifest(config), ensure_ascii=False, indent=2))
    elif args.command == "freeze-full-manifest":
        print(json.dumps(freeze_full_task_manifest(config), ensure_ascii=False, indent=2))
    elif args.command == "full-preflight":
        print(json.dumps(write_full_study_preflight(config), ensure_ascii=False, indent=2))
    elif args.command == "authorize-full":
        if not args.confirmed_by_project_owner:
            raise SystemExit("Refusing full-study authorization without --confirmed-by-project-owner")
        payload = write_full_authorization(
            config,
            command=(
                "powershell -NoProfile -ExecutionPolicy Bypass "
                "-File scripts/start_full_study.ps1 -ConfirmPreregisteredRun"
            ),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "prepare-data":
        metadata = DatasetManager(config).prepare_all()
        print(json.dumps({"prepared_splits": len(metadata)}, indent=2))
    elif args.command == "init":
        authorization = require_authorized(config)
        store = create_store(config)
        existing_hash = store.get_metadata("config_sha256")
        if existing_hash is not None and existing_hash != config["_config_sha256"]:
            raise SystemExit("Existing database uses a different configuration hash")
        store.set_metadata("config_sha256", config["_config_sha256"])
        store.set_metadata("experiment_id", config["experiment_id"])
        store.set_metadata("authorization", authorization)
        tasks = kill_test_tasks(config)
        print(json.dumps({"inserted": store.add_tasks(tasks), "stage1_total": len(tasks)}, indent=2))
    elif args.command == "init-full":
        authorization = require_full_authorized(config)
        store = create_store(config)
        existing_hash = store.get_metadata("config_sha256")
        if existing_hash is not None and existing_hash != config["_config_sha256"]:
            raise SystemExit("Existing full-study database uses a different configuration hash")
        store.set_metadata("config_sha256", config["_config_sha256"])
        store.set_metadata("experiment_id", config["experiment_id"])
        store.set_metadata("authorization", authorization)
        tasks = full_study_tasks(config)
        print(json.dumps({"inserted": store.add_tasks(tasks), "full_total": len(tasks)}, indent=2))
    elif args.command == "init-stress":
        require_authorized(config)
        store = create_store(config)
        decision = store.get_metadata("stage1_decision")
        if not decision or decision.get("decision") != "STRESS_RETEST":
            raise SystemExit("Stress retest is legal only after a recorded STRESS_RETEST decision")
        tasks = kill_test_tasks(config, stress=True)
        print(json.dumps({"inserted": store.add_tasks(tasks), "stage2_total": len(tasks)}, indent=2))
    elif args.command == "worker":
        require_authorized(config)
        stdout_context = (
            open(args.stdout_log, "a", encoding="utf-8", buffering=1)
            if args.stdout_log
            else contextlib.nullcontext()
        )
        stderr_context = (
            open(args.stderr_log, "a", encoding="utf-8", buffering=1)
            if args.stderr_log
            else contextlib.nullcontext()
        )
        with stdout_context as stdout_handle, stderr_context as stderr_handle:
            with contextlib.redirect_stdout(stdout_handle or os.sys.stdout):
                with contextlib.redirect_stderr(stderr_handle or os.sys.stderr):
                    raise SystemExit(run_worker(config))
    elif args.command == "full-worker":
        require_full_authorized(config)
        stdout_context = (
            open(args.stdout_log, "a", encoding="utf-8", buffering=1)
            if args.stdout_log
            else contextlib.nullcontext()
        )
        stderr_context = (
            open(args.stderr_log, "a", encoding="utf-8", buffering=1)
            if args.stderr_log
            else contextlib.nullcontext()
        )
        with stdout_context as stdout_handle, stderr_context as stderr_handle:
            with contextlib.redirect_stdout(stdout_handle or os.sys.stdout):
                with contextlib.redirect_stderr(stderr_handle or os.sys.stderr):
                    raise SystemExit(run_worker(config))
    elif args.command == "status":
        payload = status_payload(config)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(
                f"[{payload['worker_status']}] "
                f"{payload['completed']}/{payload['total_tasks']} completed "
                f"({payload['percent']:.2f}%), running={payload['running']}, "
                f"retry={payload['retry']}, failed={payload['failed']}, "
                f"interruptions={payload['interruptions']}"
            )
            eta = payload["eta"]
            print(
                f"ETA={_format_duration(eta['eta_seconds'])}, "
                f"median_task={_format_duration(eta['median_task_seconds'])}, "
                f"basis={eta['basis']}"
            )
            if payload["current_task"]:
                current = payload["current_task"]
                print(
                    f"Current={current['task_id']} dataset={current.get('dataset_id')} "
                    f"model={current.get('model')} corruption={current.get('corruption')} "
                    f"rate={current.get('rate')} elapsed={_format_duration(current.get('elapsed_seconds'))}"
                )
            if payload["heartbeat"]:
                heartbeat = payload["heartbeat"]
                print(
                    f"Heartbeat age={heartbeat.get('age_seconds', 0.0):.1f}s, "
                    f"pid={heartbeat.get('pid')}, alive={heartbeat.get('process_alive')}"
                )
            for alert in payload["alerts"]:
                print(f"ALERT {alert['type']}: {alert['message']}")
    elif args.command == "analyze":
        report = evaluate_kill_test(create_store(config), config, stage=args.stage)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "analyze-full":
        require_full_authorized(config)
        report = evaluate_full_study(create_store(config), config)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
