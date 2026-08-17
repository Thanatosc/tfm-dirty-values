import json
from copy import deepcopy
from pathlib import Path

import pytest

from tfm_dirty_values.analysis import evaluate_kill_test
from tfm_dirty_values.config import load_config
from tfm_dirty_values.design import kill_test_tasks
from tfm_dirty_values.state import RunStore


def _store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "state.sqlite3", tmp_path / "events.jsonl", tmp_path / "outputs")


def _complete(store: RunStore, config: dict, degradation: float) -> None:
    tasks = kill_test_tasks(config)
    store.add_tasks(tasks)
    with store.connect() as connection:
        for task in tasks:
            clean_auc = 0.80
            auc = clean_auc if task.corruption == "clean" else clean_auc - degradation
            connection.execute(
                "UPDATE runs SET status='completed', metrics_json=?, duration_sec=1.0 WHERE task_id=?",
                (json.dumps({"roc_auc": auc}), task.task_id),
            )


def test_kill_test_goes_on_large_tfm_degradation(tmp_path: Path) -> None:
    config = load_config("configs/kill_test.yaml")
    store = _store(tmp_path)
    tasks = kill_test_tasks(config)
    store.add_tasks(tasks)
    with store.connect() as connection:
        for task in tasks:
            degradation = 0.0
            if task.corruption != "clean":
                degradation = 0.06 if task.model == "tabpfn" and task.rate == 0.5 else 0.01
            connection.execute(
                "UPDATE runs SET status='completed', metrics_json=?, duration_sec=1.0 WHERE task_id=?",
                (json.dumps({"roc_auc": 0.80 - degradation}), task.task_id),
            )
    report = evaluate_kill_test(store, config, stage=1)
    assert report["decision"] == "GO"
    assert any(value["qualifies"] for value in report["criterion_a"].values())


def test_kill_test_requests_stress_retest_when_all_effects_are_tiny(tmp_path: Path) -> None:
    config = deepcopy(load_config("configs/kill_test.yaml"))
    store = _store(tmp_path)
    _complete(store, config, degradation=0.005)
    report = evaluate_kill_test(store, config, stage=1)
    assert report["decision"] == "STRESS_RETEST"


def test_kill_test_rejects_missing_primary_dataset_group(tmp_path: Path) -> None:
    config = load_config("configs/kill_test.yaml")
    store = _store(tmp_path)
    _complete(store, config, degradation=0.02)
    with store.connect() as connection:
        connection.execute(
            """UPDATE runs SET status='failed', metrics_json=NULL
            WHERE dataset_id=31 AND corruption='outlier' AND rate=0.5 AND model='tabpfn'"""
        )
    with pytest.raises(ValueError, match="Primary panel is missing"):
        evaluate_kill_test(store, config, stage=1)


def test_kill_test_discloses_incomplete_seed_groups(tmp_path: Path) -> None:
    config = load_config("configs/kill_test.yaml")
    store = _store(tmp_path)
    _complete(store, config, degradation=0.02)
    with store.connect() as connection:
        connection.execute(
            """UPDATE runs SET status='failed', metrics_json=NULL
            WHERE dataset_id=31 AND seed=1907 AND corruption='outlier'
            AND rate=0.5 AND model='tabpfn'"""
        )
    report = evaluate_kill_test(store, config, stage=1)
    assert report["expected_seeds_per_group"] == 3
    assert report["incomplete_seed_groups"] == [
        {
            "dataset_id": 31,
            "corruption": "outlier",
            "model": "tabpfn",
            "completed_seeds": 2,
        }
    ]
