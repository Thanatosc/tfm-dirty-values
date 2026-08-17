from pathlib import Path

from tfm_dirty_values.design import Task
from tfm_dirty_values.state import RunStore


def make_store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "state.sqlite3", tmp_path / "events.jsonl", tmp_path / "outputs")


def make_task(task_id: str = "task-1") -> Task:
    return Task(
        task_id=task_id,
        stage=1,
        dataset_id=31,
        dataset_name="credit-g",
        seed=1907,
        corruption="measurement_noise",
        rate=0.5,
        model="xgboost",
        parameters={"sigma_ratio": 0.5},
    )


def test_claim_recover_and_resume(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.add_tasks([make_task()]) == 1
    assert store.add_tasks([make_task()]) == 0
    claimed = store.claim_next("worker-a", max_attempts=2)
    assert claimed is not None
    assert claimed["parameters"] == {"sigma_ratio": 0.5}
    assert store.recover_interrupted("worker-b") == 1
    resumed = store.claim_next("worker-b", max_attempts=2)
    assert resumed is not None
    assert resumed["attempts"] == 1
    assert resumed["interruptions"] == 1
    store.complete(resumed, {"roc_auc": 0.75}, 1.5)
    assert store.rows(status="completed")[0]["task_id"] == "task-1"


def test_repeated_interruptions_do_not_consume_failure_retries(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_tasks([make_task("task-interrupted")])

    for index in range(3):
        claimed = store.claim_next(f"worker-{index}", max_attempts=2)
        assert claimed is not None
        assert claimed["attempts"] == 1
        assert store.recover_interrupted(f"worker-{index + 1}") == 1

    resumed = store.claim_next("worker-final", max_attempts=2)
    assert resumed is not None
    assert resumed["attempts"] == 1
    assert resumed["interruptions"] == 3


def test_failed_task_stops_after_deterministic_retry(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_tasks([make_task("task-fail")])
    first = store.claim_next("worker", max_attempts=2)
    assert first is not None
    assert store.fail(first, {"message": "first"}, 0.1, 2) == "retry"
    second = store.claim_next("worker", max_attempts=2)
    assert second is not None
    assert store.fail(second, {"message": "second"}, 0.1, 2) == "failed"
    assert store.claim_next("worker", max_attempts=2) is None
