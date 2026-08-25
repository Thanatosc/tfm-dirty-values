from __future__ import annotations

import json
from pathlib import Path

import pytest

from tfm_dirty_values_ablation.design import AblationTask
from tfm_dirty_values_ablation.store import AblationStore


def _task(task_id: str, arm: str, *, corruption: str = "outlier", rate: float = 0.5) -> AblationTask:
    return AblationTask(
        task_id=task_id,
        stage=4,
        arm=arm,
        dataset_id=31,
        dataset_name="credit-g",
        seed=1907,
        corruption=corruption,
        rate=rate,
        model="xgboost",
        cleaned=True,
        parameters={"k": 10.0},
    )


@pytest.fixture
def store(tmp_path: Path) -> AblationStore:
    return AblationStore(
        tmp_path / "state.sqlite3",
        tmp_path / "events.jsonl",
        tmp_path / "outputs",
    )


def test_arm_sidecar_records_arm_and_cleaned(store: AblationStore) -> None:
    inserted = store.add_ablation_tasks([_task("t1", "c"), _task("t2", "d", corruption="clean", rate=0.0)])
    assert inserted == 2

    assert store.arm_for("t1") == {"arm": "c", "cleaned": True}
    assert store.arm_for("t2") == {"arm": "d", "cleaned": True}


def test_unknown_task_arm_lookup_raises(store: AblationStore) -> None:
    with pytest.raises(ValueError, match="No arm assignment"):
        store.arm_for("nope")


def test_parameters_json_holds_only_corruption_parameters(store: AblationStore) -> None:
    """Arm metadata must never leak into parameters_json.

    The runner feeds parameters straight into inject_corruption, where any extra
    key would change the derived seed and break pairing with frozen arm B.
    """
    store.add_ablation_tasks([_task("t1", "c")])
    with store.connect() as connection:
        row = connection.execute(
            "SELECT parameters_json FROM runs WHERE task_id='t1'"
        ).fetchone()
    stored = json.loads(row["parameters_json"])
    assert stored == {"k": 10.0}
    assert "arm" not in stored
    assert "cleaned" not in stored


def test_claim_next_with_arm_attaches_the_arm(store: AblationStore) -> None:
    store.add_ablation_tasks([_task("t1", "c")])
    claimed = store.claim_next_with_arm("w1", 2)

    assert claimed is not None
    assert claimed["task_id"] == "t1"
    assert claimed["arm"] == "c"
    assert claimed["cleaned"] is True


def test_claim_next_with_arm_returns_none_on_empty_queue(store: AblationStore) -> None:
    assert store.claim_next_with_arm("w1", 2) is None


def test_arm_counts_groups_by_arm_and_status(store: AblationStore) -> None:
    store.add_ablation_tasks(
        [_task("t1", "c"), _task("t2", "c"), _task("t3", "d", corruption="clean", rate=0.0)]
    )
    claimed = store.claim_next_with_arm("w1", 2)
    assert claimed is not None
    store.complete(claimed, {"roc_auc": 0.8}, 1.0)

    # RunStore.claim_next imposes its own ordering (clean conditions first), so
    # the claimed task is not necessarily the first one inserted. Assert against
    # whichever arm was actually claimed rather than assuming insertion order.
    claimed_arm = claimed["arm"]
    other_arm = "d" if claimed_arm == "c" else "c"
    expected_remaining = {"c": 2, "d": 1}
    expected_remaining[claimed_arm] -= 1

    counts = store.arm_counts()
    assert counts[claimed_arm]["completed"] == 1
    assert counts[claimed_arm].get("pending", 0) == expected_remaining[claimed_arm]
    assert counts[other_arm]["pending"] == expected_remaining[other_arm]
    assert sum(sum(statuses.values()) for statuses in counts.values()) == 3


def test_verify_arm_coverage_detects_a_run_without_an_arm(store: AblationStore) -> None:
    store.add_ablation_tasks([_task("t1", "c")])
    assert store.verify_arm_coverage() == []

    with store.connect() as connection:
        connection.execute("DELETE FROM ablation_arms WHERE task_id='t1'")

    errors = store.verify_arm_coverage()
    assert errors and "no arm assignment" in errors[0]


def test_verify_arm_coverage_detects_an_orphan_arm(store: AblationStore) -> None:
    store.add_ablation_tasks([_task("t1", "c")])
    with store.connect() as connection:
        connection.execute("DELETE FROM runs WHERE task_id='t1'")

    errors = store.verify_arm_coverage()
    assert errors and "no queued run" in errors[0]


def test_add_ablation_tasks_is_idempotent(store: AblationStore) -> None:
    tasks = [_task("t1", "c"), _task("t2", "d", corruption="clean", rate=0.0)]
    assert store.add_ablation_tasks(tasks) == 2
    assert store.add_ablation_tasks(tasks) == 0

    counts = store.arm_counts()
    assert counts["c"]["pending"] == 1
    assert counts["d"]["pending"] == 1


def test_mark_initialized_records_metadata(store: AblationStore) -> None:
    store.add_ablation_tasks([_task("t1", "c")])
    store.mark_initialized(
        {"task_manifest_sha256": "abc", "task_count": 1, "arm_counts": {"c": 1}},
        {"verdict": "VERIFIED"},
    )

    assert store.get_metadata("phase3_task_manifest_sha256") == "abc"
    assert store.get_metadata("phase3_task_count") == 1
    assert store.get_metadata("phase3_paired_identity_verdict") == "VERIFIED"
    assert store.get_metadata("phase3_initialized_at") is not None
