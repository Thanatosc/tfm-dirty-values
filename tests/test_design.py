from tfm_dirty_values.config import load_config
from tfm_dirty_values.design import kill_test_tasks


def test_preregistered_task_counts_and_unique_ids() -> None:
    config = load_config("configs/kill_test.yaml")
    tasks = kill_test_tasks(config)
    clean = [task for task in tasks if task.corruption == "clean"]
    polluted = [task for task in tasks if task.corruption != "clean"]
    assert len(tasks) == 126
    assert len(clean) == 18
    assert len(polluted) == 108
    assert len({task.task_id for task in tasks}) == len(tasks)


def test_stress_retest_has_no_duplicate_clean_fits() -> None:
    config = load_config("configs/kill_test.yaml")
    tasks = kill_test_tasks(config, stress=True)
    assert len(tasks) == 54
    assert all(task.stage == 2 for task in tasks)
    assert all(task.corruption != "clean" for task in tasks)
    assert {task.rate for task in tasks} == {0.5}
    assert len({task.task_id for task in tasks}) == len(tasks)

