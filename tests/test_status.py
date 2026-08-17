from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import pytest

from tfm_dirty_values.cli import status_payload
from tfm_dirty_values.config import load_config, resolve_path
from tfm_dirty_values.design import Task
from tfm_dirty_values.worker import Heartbeat, create_store


def _task(task_id: str, seed: int) -> Task:
    return Task(
        task_id=task_id,
        stage=3,
        dataset_id=31,
        dataset_name="credit-g",
        seed=seed,
        corruption="measurement_noise",
        rate=0.5,
        model="xgboost",
        parameters={"sigma_ratio": 0.5},
    )


def _config(tmp_path: Path) -> dict:
    config = deepcopy(load_config("configs/full_study.yaml"))
    config["_project_root"] = str(tmp_path)
    return config


def _write_heartbeat(
    config: dict,
    *,
    task_id: str,
    updated_at: datetime,
    initial_rss: int = 100,
    rss: int = 200,
) -> None:
    path = resolve_path(config, config["paths"]["heartbeat"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "worker_id": "test-worker",
                "pid": os.getpid(),
                "started_at": updated_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "status": "running",
                "current_task": task_id,
                "current_task_started_at": updated_at.isoformat(),
                "initial_rss_bytes": initial_rss,
                "rss_bytes": rss,
            }
        ),
        encoding="utf-8",
    )


def test_status_reports_progress_eta_and_current_task(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = create_store(config)
    tasks = [_task(f"task-{index}", 1907 + index) for index in range(4)]
    store.add_tasks(tasks)
    now = datetime.now(timezone.utc)
    with store.connect() as connection:
        connection.execute(
            "UPDATE runs SET status='completed', duration_sec=10.0, metrics_json='{}' "
            "WHERE task_id='task-0'"
        )
        connection.execute(
            "UPDATE runs SET status='completed', duration_sec=20.0, metrics_json='{}' "
            "WHERE task_id='task-1'"
        )
        connection.execute(
            "UPDATE runs SET status='running', attempts=1, worker_id='test-worker', started_at=? "
            "WHERE task_id='task-2'",
            (now.isoformat(),),
        )
    _write_heartbeat(config, task_id="task-2", updated_at=now)

    payload = status_payload(config)

    assert payload["worker_status"] == "running"
    assert payload["total_tasks"] == 4
    assert payload["completed"] == 2
    assert payload["running"] == 1
    assert payload["remaining"] == 2
    assert payload["percent"] == pytest.approx(50.0)
    assert payload["eta"]["median_task_seconds"] == pytest.approx(15.0)
    assert payload["eta"]["estimated_tasks_per_hour"] == pytest.approx(240.0)
    assert payload["eta"]["eta_seconds"] == pytest.approx(30.0)
    assert payload["current_task"]["task_id"] == "task-2"
    assert payload["current_task"]["model"] == "xgboost"
    assert payload["alerts"] == []


def test_status_flags_stale_heartbeat_and_memory_growth(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = create_store(config)
    store.add_tasks([_task("task-stale", 1907)])
    with store.connect() as connection:
        connection.execute(
            "UPDATE runs SET status='running', attempts=1 WHERE task_id='task-stale'"
        )
    stale = datetime.now(timezone.utc) - timedelta(
        seconds=int(config["worker"]["stale_heartbeat_seconds"]) + 30
    )
    _write_heartbeat(
        config,
        task_id="task-stale",
        updated_at=stale,
        initial_rss=100,
        rss=400,
    )

    payload = status_payload(config)

    assert payload["worker_status"] == "stale"
    assert {alert["type"] for alert in payload["alerts"]} == {
        "HEARTBEAT_STALE",
        "RESOURCE_ANOMALY",
    }


def test_heartbeat_persists_task_timing_and_memory(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    heartbeat = Heartbeat(path, interval=60, worker_id="test-worker")
    heartbeat.start()
    try:
        heartbeat.start_task("task-live")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "starting"
        assert payload["current_task"] == "task-live"
        assert payload["current_task_started_at"]
        assert payload["initial_rss_bytes"] > 0
        assert payload["rss_bytes"] > 0
    finally:
        heartbeat.close("stopped")
