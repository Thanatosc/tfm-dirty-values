from __future__ import annotations

from pathlib import Path

import pytest

from tfm_dirty_values.config import load_config
from tfm_dirty_values.design import full_study_tasks
from tfm_dirty_values.full_analysis import (
    evaluate_full_study,
    exact_sign_flip_pvalue,
    holm_adjust,
)
from tfm_dirty_values.state import RunStore


def _store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "state.sqlite3", tmp_path / "events.jsonl", tmp_path / "outputs")


def test_exact_sign_flip_pvalue() -> None:
    assert exact_sign_flip_pvalue([1.0, 1.0, 1.0]) == pytest.approx(0.25)
    assert exact_sign_flip_pvalue([1.0, -1.0]) == pytest.approx(1.0)


def test_holm_adjustment_preserves_input_order_and_monotonicity() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_full_analysis_rejects_nonmanifest_task_panel(tmp_path: Path) -> None:
    config = load_config("configs/full_study.yaml")
    tasks = full_study_tasks(config)
    store = _store(tmp_path)
    store.add_tasks(tasks[:-1])

    with pytest.raises(ValueError, match="do not match the frozen manifest"):
        evaluate_full_study(store, config)


def test_full_analysis_requires_every_task_to_complete(tmp_path: Path) -> None:
    config = load_config("configs/full_study.yaml")
    store = _store(tmp_path)
    store.add_tasks(full_study_tasks(config))

    with pytest.raises(ValueError, match="4575 are incomplete"):
        evaluate_full_study(store, config)
